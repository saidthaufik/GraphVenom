from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Hindari bentrok antara nama file `xgboost.py` dan package `xgboost`.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
from xgboost import XGBClassifier


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from path import model_path, output_roots
from reporting import (
    METRIC_COLUMNS,
    build_execution_report,
    dataframe_to_string,
    render_candidate_detail,
    render_selected_detail,
)


MODEL_NAME = "model2_sing_reg"
SCENARIO = "unseen_peptide"
SCENARIO_LABEL = "Unseen Peptide"
INPUT_CSV = model_path("model2", "0-neg", "gan_decoy_sequence_clean.csv")
DEFAULT_SPLIT_ROOT = PROJECT_ROOT.parent / "1-data" / "splits" / "unseen" / "model2"
DATASET_NAME = "rki"
MODEL_DIR = model_path("model2")
REG_ROOT = PROJECT_ROOT / "cache" / "runs" / "unseen" / "peptide"
_OUTPUT_ROOTS = {
    "log_root": REG_ROOT / "b-log_result" / MODEL_NAME,
    "fold_root": REG_ROOT / "a-fold" / MODEL_NAME,
    "save_model_root": REG_ROOT / "c-model_save" / MODEL_NAME,
}
LOG_ROOT = _OUTPUT_ROOTS["log_root"]
FOLD_ROOT = _OUTPUT_ROOTS["fold_root"]
SAVE_MODEL_ROOT = _OUTPUT_ROOTS["save_model_root"]
SCRIPT_NAME = Path(__file__).name
RANDOM_SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
N_ITER_SEARCH = 1
THRESHOLD_GRID = np.array(sorted(set(np.round(np.concatenate([np.arange(0.1, 1.01, 0.1), [0.512]]), 3))))
SINGLE_BEST_PARAMS = {
    # Selected from target_auc_pack_v2 candidate 2 for unseen sequence baseline.
    "classifier__colsample_bytree": 0.0195,
    "classifier__gamma": 37.0,
    "classifier__learning_rate": 0.0018,
    "classifier__max_depth": 1,
    "classifier__min_child_weight": 180,
    "classifier__n_estimators": 223,
    "classifier__scale_pos_weight": 1.0,
    "classifier__subsample": 0.272,
    "classifier__reg_lambda": 190.0,
    "classifier__reg_alpha": 46.0,
}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(to_serializable(payload), sort_keys=True, separators=(",", ":"))


def compute_config_signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def load_run_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "resume_meta": {},
            "folds": {},
            "final": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("resume_meta", {})
    state.setdefault("folds", {})
    state.setdefault("final", {})
    return state


def save_run_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(state), handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def make_output_dirs(run_name: str) -> dict[str, Path]:
    log_root = ensure_dir(REG_ROOT / "b-log_result" / run_name)
    fold_root = ensure_dir(REG_ROOT / "a-fold" / run_name)
    model_root = ensure_dir(REG_ROOT / "c-model_save" / run_name)
    resume_root = ensure_dir(model_root / "resume")
    return {
        "run_name": Path(run_name),
        "log_root": log_root,
        "fold_root": fold_root,
        "model_root": model_root,
        "text": log_root / f"{run_name}_results_{DATASET_NAME}.txt",
        "reports": ensure_dir(log_root / "reports"),
        "plots": ensure_dir(log_root / "plots"),
        "fold_reports": ensure_dir(fold_root / "reports"),
        "predictions": ensure_dir(fold_root / "predictions"),
        "checkpoints": ensure_dir(model_root / "checkpoints"),
        "resume": resume_root,
        "state": log_root / f"{run_name}_resume_{DATASET_NAME}.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Training model2 XGBoost RKI - {SCENARIO_LABEL}.")
    parser.add_argument("--run-name", type=str, default="", help=f"Nama run untuk metadata. Default: {MODEL_NAME}.")
    parser.add_argument("--resume-run", type=str, default="", help="Nama run lama untuk dilanjutkan.")
    parser.add_argument("--threshold", type=float, default=0.512, help="Threshold klasifikasi binary. Default: 0.512.")
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT, help="Root folder sequence-level unseen split.")
    parser.add_argument("--max-folds", type=int, default=0, help="Batasi jumlah fold untuk smoke test. Default: semua fold.")
    parser.add_argument("--tune-grid", choices=("none", "target80_small", "target80_wide", "mid_stable_small", "target_auc_pack", "target_auc_pack_v2"), default="none", help="Preset grid search hyperparameter. Default: none.")
    parser.add_argument("--selection-metric", choices=("roc_auc", "pr_auc", "f1", "accuracy"), default="roc_auc", help="Metric untuk memilih kandidat pada inner CV. Default: roc_auc.")
    parser.add_argument("--xgb-colsample-bytree", type=float, default=None, help="Override classifier__colsample_bytree.")
    parser.add_argument("--xgb-gamma", type=float, default=None, help="Override classifier__gamma.")
    parser.add_argument("--xgb-learning-rate", type=float, default=None, help="Override classifier__learning_rate.")
    parser.add_argument("--xgb-max-depth", type=int, default=None, help="Override classifier__max_depth.")
    parser.add_argument("--xgb-min-child-weight", type=float, default=None, help="Override classifier__min_child_weight.")
    parser.add_argument("--xgb-n-estimators", type=int, default=None, help="Override classifier__n_estimators.")
    parser.add_argument("--xgb-scale-pos-weight", type=float, default=None, help="Override classifier__scale_pos_weight.")
    parser.add_argument("--xgb-subsample", type=float, default=None, help="Override classifier__subsample.")
    parser.add_argument("--xgb-reg-lambda", type=float, default=None, help="Override classifier__reg_lambda.")
    parser.add_argument("--xgb-reg-alpha", type=float, default=None, help="Override classifier__reg_alpha.")
    return parser.parse_args()


def cli_param_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "classifier__colsample_bytree": args.xgb_colsample_bytree,
        "classifier__gamma": args.xgb_gamma,
        "classifier__learning_rate": args.xgb_learning_rate,
        "classifier__max_depth": args.xgb_max_depth,
        "classifier__min_child_weight": args.xgb_min_child_weight,
        "classifier__n_estimators": args.xgb_n_estimators,
        "classifier__scale_pos_weight": args.xgb_scale_pos_weight,
        "classifier__subsample": args.xgb_subsample,
        "classifier__reg_lambda": args.xgb_reg_lambda,
        "classifier__reg_alpha": args.xgb_reg_alpha,
    }


def apply_param_overrides(params: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = dict(params)
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


def build_candidate_params(args: argparse.Namespace) -> dict[str, Any]:
    return apply_param_overrides(dict(SINGLE_BEST_PARAMS), cli_param_overrides(args))


def build_candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    overrides = cli_param_overrides(args)
    if args.tune_grid == "none":
        return [build_candidate_params(args)]

    target80_small = [
        {"classifier__colsample_bytree": 0.018, "classifier__gamma": 38.0, "classifier__learning_rate": 0.0016, "classifier__max_depth": 1, "classifier__min_child_weight": 180, "classifier__n_estimators": 220, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.26, "classifier__reg_lambda": 185.0, "classifier__reg_alpha": 45.0},
        {"classifier__colsample_bytree": 0.019, "classifier__gamma": 36.0, "classifier__learning_rate": 0.0018, "classifier__max_depth": 1, "classifier__min_child_weight": 172, "classifier__n_estimators": 222, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.27, "classifier__reg_lambda": 178.0, "classifier__reg_alpha": 42.0},
        {"classifier__colsample_bytree": 0.020, "classifier__gamma": 35.0, "classifier__learning_rate": 0.0019, "classifier__max_depth": 1, "classifier__min_child_weight": 170, "classifier__n_estimators": 222, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.275, "classifier__reg_lambda": 175.0, "classifier__reg_alpha": 41.0},
        {"classifier__colsample_bytree": 0.019, "classifier__gamma": 38.0, "classifier__learning_rate": 0.0020, "classifier__max_depth": 1, "classifier__min_child_weight": 185, "classifier__n_estimators": 222, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.27, "classifier__reg_lambda": 195.0, "classifier__reg_alpha": 40.0},
        {"classifier__colsample_bytree": 0.023, "classifier__gamma": 32.0, "classifier__learning_rate": 0.0022, "classifier__max_depth": 1, "classifier__min_child_weight": 155, "classifier__n_estimators": 230, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.29, "classifier__reg_lambda": 160.0, "classifier__reg_alpha": 37.0},
        {"classifier__colsample_bytree": 0.024, "classifier__gamma": 30.0, "classifier__learning_rate": 0.0023, "classifier__max_depth": 1, "classifier__min_child_weight": 150, "classifier__n_estimators": 235, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.30, "classifier__reg_lambda": 155.0, "classifier__reg_alpha": 35.0},
    ]
    target80_wide = target80_small + [
        {"classifier__colsample_bytree": 0.016, "classifier__gamma": 40.0, "classifier__learning_rate": 0.0014, "classifier__max_depth": 1, "classifier__min_child_weight": 190, "classifier__n_estimators": 215, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.25, "classifier__reg_lambda": 195.0, "classifier__reg_alpha": 48.0},
        {"classifier__colsample_bytree": 0.017, "classifier__gamma": 39.0, "classifier__learning_rate": 0.0015, "classifier__max_depth": 1, "classifier__min_child_weight": 185, "classifier__n_estimators": 220, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.255, "classifier__reg_lambda": 190.0, "classifier__reg_alpha": 46.0},
        {"classifier__colsample_bytree": 0.026, "classifier__gamma": 28.0, "classifier__learning_rate": 0.0025, "classifier__max_depth": 1, "classifier__min_child_weight": 145, "classifier__n_estimators": 240, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.31, "classifier__reg_lambda": 145.0, "classifier__reg_alpha": 32.0},
        {"classifier__colsample_bytree": 0.028, "classifier__gamma": 26.0, "classifier__learning_rate": 0.0027, "classifier__max_depth": 1, "classifier__min_child_weight": 140, "classifier__n_estimators": 245, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.32, "classifier__reg_lambda": 140.0, "classifier__reg_alpha": 30.0},
    ]
    mid_stable_small = [
        {"classifier__colsample_bytree": 0.022, "classifier__gamma": 32.0, "classifier__learning_rate": 0.0021, "classifier__max_depth": 1, "classifier__min_child_weight": 160, "classifier__n_estimators": 222, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.27, "classifier__reg_lambda": 165.0, "classifier__reg_alpha": 38.0},
        {"classifier__colsample_bytree": 0.024, "classifier__gamma": 30.0, "classifier__learning_rate": 0.0023, "classifier__max_depth": 1, "classifier__min_child_weight": 155, "classifier__n_estimators": 230, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.29, "classifier__reg_lambda": 155.0, "classifier__reg_alpha": 35.0},
        {"classifier__colsample_bytree": 0.026, "classifier__gamma": 28.0, "classifier__learning_rate": 0.0025, "classifier__max_depth": 1, "classifier__min_child_weight": 150, "classifier__n_estimators": 235, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.30, "classifier__reg_lambda": 145.0, "classifier__reg_alpha": 32.0},
        {"classifier__colsample_bytree": 0.028, "classifier__gamma": 26.0, "classifier__learning_rate": 0.0027, "classifier__max_depth": 1, "classifier__min_child_weight": 145, "classifier__n_estimators": 240, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.31, "classifier__reg_lambda": 140.0, "classifier__reg_alpha": 30.0},
        {"classifier__colsample_bytree": 0.030, "classifier__gamma": 24.0, "classifier__learning_rate": 0.0030, "classifier__max_depth": 1, "classifier__min_child_weight": 135, "classifier__n_estimators": 245, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.32, "classifier__reg_lambda": 130.0, "classifier__reg_alpha": 28.0},
        {"classifier__colsample_bytree": 0.032, "classifier__gamma": 22.0, "classifier__learning_rate": 0.0030, "classifier__max_depth": 1, "classifier__min_child_weight": 130, "classifier__n_estimators": 250, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.33, "classifier__reg_lambda": 125.0, "classifier__reg_alpha": 26.0},
    ]
    target_auc_pack = [
        {"classifier__colsample_bytree": 0.016, "classifier__gamma": 42.0, "classifier__learning_rate": 0.0014, "classifier__max_depth": 1, "classifier__min_child_weight": 205, "classifier__n_estimators": 215, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.25, "classifier__reg_lambda": 220.0, "classifier__reg_alpha": 55.0},
        {"classifier__colsample_bytree": 0.018, "classifier__gamma": 40.0, "classifier__learning_rate": 0.0015, "classifier__max_depth": 1, "classifier__min_child_weight": 195, "classifier__n_estimators": 220, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.26, "classifier__reg_lambda": 210.0, "classifier__reg_alpha": 52.0},
        {"classifier__colsample_bytree": 0.019, "classifier__gamma": 38.0, "classifier__learning_rate": 0.0017, "classifier__max_depth": 1, "classifier__min_child_weight": 185, "classifier__n_estimators": 222, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.27, "classifier__reg_lambda": 195.0, "classifier__reg_alpha": 48.0},
        {"classifier__colsample_bytree": 0.020, "classifier__gamma": 36.0, "classifier__learning_rate": 0.0019, "classifier__max_depth": 1, "classifier__min_child_weight": 175, "classifier__n_estimators": 224, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.275, "classifier__reg_lambda": 185.0, "classifier__reg_alpha": 45.0},
        {"classifier__colsample_bytree": 0.021, "classifier__gamma": 34.0, "classifier__learning_rate": 0.0020, "classifier__max_depth": 1, "classifier__min_child_weight": 168, "classifier__n_estimators": 225, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.28, "classifier__reg_lambda": 175.0, "classifier__reg_alpha": 42.0},
        {"classifier__colsample_bytree": 0.022, "classifier__gamma": 32.0, "classifier__learning_rate": 0.0021, "classifier__max_depth": 1, "classifier__min_child_weight": 160, "classifier__n_estimators": 225, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.28, "classifier__reg_lambda": 165.0, "classifier__reg_alpha": 38.0},
    ]
    target_auc_pack_v2 = [
        {"classifier__colsample_bytree": 0.0192, "classifier__gamma": 37.5, "classifier__learning_rate": 0.00175, "classifier__max_depth": 1, "classifier__min_child_weight": 183, "classifier__n_estimators": 222, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.270, "classifier__reg_lambda": 193.0, "classifier__reg_alpha": 47.0},
        {"classifier__colsample_bytree": 0.0195, "classifier__gamma": 37.0, "classifier__learning_rate": 0.00180, "classifier__max_depth": 1, "classifier__min_child_weight": 180, "classifier__n_estimators": 223, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.272, "classifier__reg_lambda": 190.0, "classifier__reg_alpha": 46.0},
        {"classifier__colsample_bytree": 0.0198, "classifier__gamma": 36.5, "classifier__learning_rate": 0.00185, "classifier__max_depth": 1, "classifier__min_child_weight": 178, "classifier__n_estimators": 224, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.274, "classifier__reg_lambda": 187.0, "classifier__reg_alpha": 45.0},
        {"classifier__colsample_bytree": 0.0200, "classifier__gamma": 36.0, "classifier__learning_rate": 0.00190, "classifier__max_depth": 1, "classifier__min_child_weight": 175, "classifier__n_estimators": 224, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.275, "classifier__reg_lambda": 185.0, "classifier__reg_alpha": 45.0},
        {"classifier__colsample_bytree": 0.0203, "classifier__gamma": 35.5, "classifier__learning_rate": 0.00195, "classifier__max_depth": 1, "classifier__min_child_weight": 173, "classifier__n_estimators": 225, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.276, "classifier__reg_lambda": 182.0, "classifier__reg_alpha": 44.0},
        {"classifier__colsample_bytree": 0.0206, "classifier__gamma": 35.0, "classifier__learning_rate": 0.00200, "classifier__max_depth": 1, "classifier__min_child_weight": 171, "classifier__n_estimators": 225, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.278, "classifier__reg_lambda": 180.0, "classifier__reg_alpha": 43.0},
        {"classifier__colsample_bytree": 0.0210, "classifier__gamma": 34.0, "classifier__learning_rate": 0.00205, "classifier__max_depth": 1, "classifier__min_child_weight": 168, "classifier__n_estimators": 226, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.280, "classifier__reg_lambda": 175.0, "classifier__reg_alpha": 42.0},
        {"classifier__colsample_bytree": 0.0215, "classifier__gamma": 33.0, "classifier__learning_rate": 0.00210, "classifier__max_depth": 1, "classifier__min_child_weight": 165, "classifier__n_estimators": 226, "classifier__scale_pos_weight": 1.0, "classifier__subsample": 0.282, "classifier__reg_lambda": 170.0, "classifier__reg_alpha": 40.0},
    ]
    presets = {
        "target80_small": target80_small,
        "target80_wide": target80_wide,
        "mid_stable_small": mid_stable_small,
        "target_auc_pack": target_auc_pack,
        "target_auc_pack_v2": target_auc_pack_v2,
    }
    preset = presets[args.tune_grid]
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for params in preset:
        candidate = apply_param_overrides(params, overrides)
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def read_indices(path: Path, n_samples: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Split index tidak ditemukan: {path}")
    df = pd.read_csv(path)
    if "row_index" not in df.columns:
        raise ValueError(f"Kolom row_index tidak ditemukan pada {path}")
    values = df["row_index"].to_numpy(dtype=np.int64)
    if values.size == 0:
        raise ValueError(f"Split index kosong: {path}")
    if int(values.min()) < 0 or int(values.max()) >= n_samples:
        raise ValueError(f"Index split di luar rentang dataset pada {path}")
    return values


def load_unseen_folds(split_root: Path, scenario: str, n_samples: int, max_folds: int) -> list[dict[str, Any]]:
    scenario_dir = split_root / scenario
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Folder split scenario tidak ditemukan: {scenario_dir}")

    folds: list[dict[str, Any]] = []
    fold_idx = 1
    while True:
        train_path = scenario_dir / f"fold_{fold_idx}_train_idx.csv"
        test_path = scenario_dir / f"fold_{fold_idx}_test_idx.csv"
        if not train_path.exists() and not test_path.exists():
            break
        train_idx = read_indices(train_path, n_samples)
        valid_idx = read_indices(test_path, n_samples)
        overlap = np.intersect1d(train_idx, valid_idx)
        if overlap.size:
            raise ValueError(f"Train/test overlap pada {scenario} fold {fold_idx}: {overlap[:5].tolist()}")
        folds.append({"fold": fold_idx, "train_idx": train_idx, "valid_idx": valid_idx})
        if max_folds and len(folds) >= max_folds:
            break
        fold_idx += 1

    if not folds:
        raise ValueError(f"Tidak ada fold ditemukan untuk {scenario_dir}")
    return folds


def infer_identifier_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    candidate_keywords = ("id", "seq", "identifier", "index")
    identifier_cols = []
    for column in df.columns:
        if column == target_col:
            continue
        lower_name = column.lower()
        if any(keyword in lower_name for keyword in candidate_keywords):
            identifier_cols.append(column)
            continue
        if df[column].dtype == "object":
            identifier_cols.append(column)
    return sorted(set(identifier_cols))


def find_possible_targets(df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    target_keywords = ("label", "target", "class", "y", "outcome")
    row_count = len(df)
    for column in df.columns:
        series = df[column]
        unique_count = int(series.nunique(dropna=False))
        lower_name = column.lower()
        score = 0
        reasons = []

        if any(keyword == lower_name or keyword in lower_name for keyword in target_keywords):
            score += 5
            reasons.append("nama kolom mirip target")
        if 2 <= unique_count <= min(20, max(2, row_count // 10)):
            score += 2
            reasons.append("jumlah kelas kecil")
        if unique_count == 2:
            score += 2
            reasons.append("biner")
        if series.dtype == "object" and unique_count <= 20:
            score += 1
            reasons.append("kategori berhingga")
        if unique_count == row_count:
            score -= 3
            reasons.append("unik per baris, cenderung identifier")

        if score > 0:
            candidates.append(
                {
                    "column": column,
                    "dtype": str(series.dtype),
                    "unique_count": unique_count,
                    "score": score,
                    "reasons": reasons,
                    "sample_values": series.astype(str).value_counts(dropna=False).head(5).to_dict(),
                }
            )

    candidates.sort(key=lambda item: (-item["score"], item["unique_count"], item["column"]))
    return candidates


def detect_duplicate_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    signature_map: dict[tuple[int, ...], list[str]] = defaultdict(list)
    duplicates: list[tuple[str, str]] = []

    for column in df.columns:
        hashed = tuple(pd.util.hash_pandas_object(df[column], index=False).to_numpy(dtype=np.uint64))
        signature_map[hashed].append(column)

    for columns in signature_map.values():
        if len(columns) < 2:
            continue
        base = columns[0]
        for other in columns[1:]:
            if df[base].equals(df[other]):
                duplicates.append((base, other))
    return duplicates


def dataframe_preview_text(df: pd.DataFrame, rows: int = 5) -> str:
    preview = df.head(rows)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        return preview.to_string(index=False)


def build_preprocessor(
    X_train: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_scale_pos_weight(y_train: np.ndarray) -> float | None:
    counter = Counter(y_train.tolist())
    if len(counter) != 2:
        return None
    if 0 not in counter or 1 not in counter:
        return None
    if counter[1] == 0:
        return None
    return float(counter[0]) / float(counter[1])


def make_xgb_classifier(num_classes: int, scale_pos_weight: float | None) -> XGBClassifier:
    params: dict[str, Any] = {
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 4,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if num_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = num_classes
    return XGBClassifier(**params)


def hyperparameter_space(num_classes: int, scale_pos_weight: float | None) -> dict[str, list[Any]]:
    # Single-grid model2: best param dikunci dari artefak 22-05-2026.
    return {key: [value] for key, value in SINGLE_BEST_PARAMS.items()}


def evaluate_binary_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    return {
        "ROC-AUC": float(roc_auc_score(y_true, y_score)),
        "PR-AUC": float(average_precision_score(y_true, y_score)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_binary_threshold_recap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray = THRESHOLD_GRID,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= float(threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "F1": float(f1_score(y_true, y_pred, zero_division=0)),
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            }
        )
    return pd.DataFrame(rows)


def evaluate_multiclass_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[dict[str, float], str, str]:
    metrics = {
        "ROC-AUC": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
        "PR-AUC": float(average_precision_score(pd.get_dummies(y_true), y_proba, average="macro")),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    roc_note = "multiclass ROC-AUC menggunakan one-vs-rest macro average"
    pr_note = "multiclass PR-AUC menggunakan average precision macro average one-vs-rest"
    return metrics, roc_note, pr_note


def confusion_counts_for_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, int]:
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    tp = int(np.trace(cm))
    total = int(cm.sum())
    return {"TP": tp, "TN": 0, "FP": int(total - tp), "FN": int(total - tp)}


def save_confusion_matrix_plot(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_metric_bar_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metric_columns = list(METRIC_COLUMNS)
    mean_values = metrics_df[metric_columns].mean()
    std_values = metrics_df[metric_columns].std(ddof=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(mean_values.index, mean_values.values, yerr=std_values.values, capsize=4, color="#4C78A8")
    ax.set_ylim(0, min(1.05, max(mean_values.max() + std_values.max() + 0.05, 0.2)))
    ax.set_title("Average CV Metrics")
    ax.set_ylabel("Score")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_threshold_recap_plot(threshold_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        ax.plot(threshold_df["threshold"], threshold_df[metric], marker="o", label=metric)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("OOF Threshold Recap")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(threshold_df["threshold"])
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_roc_plot(roc_entries: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for entry in roc_entries:
        ax.plot(entry["fpr"], entry["tpr"], label=f'Fold {entry["fold"]} (AUC={entry["auc"]:.4f})', alpha=0.85)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Across Folds")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_pr_plot(pr_entries: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for entry in pr_entries:
        ax.plot(entry["recall"], entry["precision"], label=f'Fold {entry["fold"]} (AP={entry["ap"]:.4f})', alpha=0.85)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Across Folds")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def prettify_params(params: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in params.items():
        simple_key = key.replace("classifier__", "")
        if isinstance(value, np.generic):
            cleaned[simple_key] = value.item()
        else:
            cleaned[simple_key] = value
    return cleaned


def format_candidate_params_for_cli(params: dict[str, Any]) -> str:
    cleaned = prettify_params(params)
    ordered_keys = [
        "learning_rate",
        "n_estimators",
        "colsample_bytree",
        "max_depth",
        "min_child_weight",
        "subsample",
        "gamma",
        "scale_pos_weight",
    ]
    parts = []
    for key in ordered_keys:
        if key in cleaned:
            parts.append(f"{key}={cleaned[key]}")
    for key in sorted(cleaned):
        if key not in ordered_keys:
            parts.append(f"{key}={cleaned[key]}")
    return " | ".join(parts)


def metric_display_name(selection_metric: str) -> str:
    return {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "f1": "F1",
        "accuracy": "Accuracy",
    }[selection_metric]


def score_inner_fold(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    num_classes: int,
    threshold: float,
    selection_metric: str,
) -> float:
    X_inner_train = X_train.iloc[train_index]
    X_inner_valid = X_train.iloc[valid_index]
    y_inner_train = y_train[train_index]
    y_inner_valid = y_train[valid_index]

    model.fit(X_inner_train, y_inner_train)
    y_inner_proba = model.predict_proba(X_inner_valid)

    if num_classes == 2:
        y_inner_pred = (y_inner_proba[:, 1] >= threshold).astype(int)
        metrics = evaluate_binary_fold(y_inner_valid, y_inner_pred, y_inner_proba[:, 1])
    else:
        y_inner_pred = model.predict(X_inner_valid)
        metrics, _, _ = evaluate_multiclass_fold(y_inner_valid, y_inner_pred, y_inner_proba)
    return float(metrics[metric_display_name(selection_metric)])


def params_to_hashable(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    normalized = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, float):
            value = round(value, 10)
        normalized.append((key, value))
    return tuple(normalized)


def choose_target(df: pd.DataFrame, candidates: list[dict[str, Any]]) -> tuple[str, str]:
    if "label" in df.columns:
        return "label", (
            "Kolom `label` dipilih sebagai target karena ada secara eksplisit, "
            "jumlah kelasnya kecil, dan script pembentuk dataset random menambahkan "
            "`label=1` untuk positif serta `label=0` untuk negatif acak."
        )
    if candidates:
        top = candidates[0]
        return top["column"], (
            f"Kolom `{top['column']}` dipilih sebagai target terbaik berdasarkan skor heuristik "
            f"{top['score']} dan alasan: {', '.join(top['reasons'])}."
        )
    raise ValueError("Tidak ditemukan kandidat target yang memadai.")


def format_target_candidates_for_cli(candidates: list[dict[str, Any]], limit: int = 10) -> list[str]:
    lines: list[str] = []
    total = min(len(candidates), limit)
    for idx, candidate in enumerate(candidates[:limit], start=1):
        reasons = ", ".join(candidate["reasons"]) if candidate["reasons"] else "-"
        lines.append(
            f"Kandidat {idx}/{total}: {candidate['column']} | "
            f"score={candidate['score']} | dtype={candidate['dtype']} | "
            f"unique={candidate['unique_count']} | alasan={reasons}"
        )
    return lines


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    resume_requested = bool(args.resume_run.strip())
    threshold = float(args.threshold)
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1.")
    candidate_grid = build_candidate_grid(args)
    selection_metric = args.selection_metric
    selection_metric_name = metric_display_name(selection_metric)
    run_name = (
        args.resume_run.strip()
        or args.run_name.strip()
        or MODEL_NAME
    )
    out_dirs = make_output_dirs(run_name)
    fold_model_dir = out_dirs["checkpoints"]
    log_path = out_dirs["text"]
    run_state = load_run_state(out_dirs["state"])
    resume_config_payload = {
        "constants": {
            "INPUT_CSV": str(INPUT_CSV),
            "RANDOM_SEED": RANDOM_SEED,
            "OUTER_SPLITS": OUTER_SPLITS,
            "INNER_SPLITS": INNER_SPLITS,
            "N_ITER_SEARCH": N_ITER_SEARCH,
            "THRESHOLD": threshold,
            "SCENARIO": SCENARIO,
            "SPLIT_ROOT": str(args.split_root),
            "MAX_FOLDS": int(args.max_folds),
            "CANDIDATE_GRID": candidate_grid,
            "SELECTION_METRIC": selection_metric,
        },
        "script_name": SCRIPT_NAME,
    }
    config_signature = compute_config_signature(resume_config_payload)
    saved_signature = str(run_state.get("resume_meta", {}).get("config_signature", "")).strip()
    if resume_requested and not out_dirs["state"].exists():
        raise FileNotFoundError(f"Resume state tidak ditemukan: {out_dirs['state']}")
    if saved_signature and saved_signature != config_signature:
        raise RuntimeError(
            "Resume config mismatch untuk run {}. saved_signature={}, current_signature={}.".format(
                out_dirs["run_name"].name, saved_signature, config_signature
            )
        )
    run_state["resume_meta"] = {
        "config_signature": config_signature,
        "config_payload": resume_config_payload,
    }
    save_run_state(out_dirs["state"], run_state)

    print(f"[{now_text()}] Scenario: {SCENARIO_LABEL}")
    print(f"[{now_text()}] Split root: {args.split_root}")
    print(f"[{now_text()}] Candidate grid: {len(candidate_grid)} candidate(s)")
    for idx, params in enumerate(candidate_grid, start=1):
        print(f"[{now_text()}] Candidate {idx}: {json.dumps(prettify_params(params), ensure_ascii=False)}")
    print(f"[{now_text()}] Selection metric: {selection_metric_name}")
    print(f"[{now_text()}] Loading dataset from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"[{now_text()}] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    if "label" in df.columns:
        possible_targets = [
            {
                "column": "label",
                "dtype": str(df["label"].dtype),
                "unique_count": int(df["label"].nunique(dropna=False)),
                "score": 999,
                "reasons": ["kolom label eksplisit"],
                "sample_values": df["label"].value_counts(dropna=False).head(5).to_dict(),
            }
        ]
        target_col, target_reason = choose_target(df, possible_targets)
        print(f"[{now_text()}] Target column detected directly: label")
        print(f"Target terpilih: {target_col}")
        print(f"Alasan: {target_reason}")
    else:
        print(f"[{now_text()}] Analysing target candidates")
        possible_targets = find_possible_targets(df)
        target_col, target_reason = choose_target(df, possible_targets)
        print("Analisis target kandidat:")
        for line in format_target_candidates_for_cli(possible_targets, limit=10):
            print(f"  {line}")
        print(f"Target terpilih: {target_col}")
        print(f"Alasan: {target_reason}")

    print(f"[{now_text()}] Detecting identifier, feature types, missing values, constants, and duplicate columns")
    identifier_cols = infer_identifier_columns(df, target_col)
    numeric_cols_all = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols_all = [
        column
        for column in df.drop(columns=[target_col]).select_dtypes(exclude=[np.number]).columns.tolist()
        if column not in identifier_cols
    ]
    feature_cols = [column for column in df.columns if column not in set(identifier_cols + [target_col])]
    dropped_shortcut_cols: list[str] = []

    column_analysis = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "all_columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "unique_values": {column: int(value) for column, value in df.nunique(dropna=False).items()},
        "preview": dataframe_preview_text(df, rows=5),
        "identifier_columns": identifier_cols,
        "dropped_shortcut_columns": dropped_shortcut_cols,
        "dropped_shortcut_prefixes": [],
        "possible_targets": possible_targets[:10],
        "numeric_feature_columns": numeric_cols_all,
        "categorical_feature_columns": categorical_cols_all,
        "constant_columns": [column for column in df.columns if df[column].nunique(dropna=False) <= 1],
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_columns": detect_duplicate_columns(df),
    }
    print(
        f"[{now_text()}] Analysis summary | identifiers={len(identifier_cols)} | "
        f"dropped_shortcut_features={len(dropped_shortcut_cols)} | "
        f"numeric_features={len(numeric_cols_all)} | categorical_features={len(categorical_cols_all)} | "
        f"constant_cols={len(column_analysis['constant_columns'])} | duplicate_cols={len(column_analysis['duplicate_columns'])}"
    )

    X = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_labels = [str(label) for label in label_encoder.classes_]
    num_classes = len(class_labels)
    print(f"[{now_text()}] Encoded target classes: {class_labels}")

    if num_classes < 2:
        raise ValueError(f"Target `{target_col}` hanya memiliki {num_classes} kelas; klasifikasi tidak dapat dijalankan.")

    unseen_folds = load_unseen_folds(args.split_root, SCENARIO, len(df), args.max_folds)

    per_fold_metrics: list[dict[str, Any]] = []
    per_fold_params: list[dict[str, Any]] = []
    detail_sections: list[str] = []
    roc_entries: list[dict[str, Any]] = []
    pr_entries: list[dict[str, Any]] = []
    fold_prediction_payloads: list[dict[str, Any]] = []
    fold_split_payloads: list[dict[str, Any]] = []

    oof_pred = np.full_like(y, fill_value=-1)
    oof_proba = np.full((len(y), num_classes), np.nan, dtype=float)

    print(f"[{now_text()}] Dataset ready: {X.shape[0]} rows, {X.shape[1]} final features, target={target_col}")

    for fold_payload in unseen_folds:
        fold_idx = int(fold_payload["fold"])
        train_idx = np.asarray(fold_payload["train_idx"], dtype=np.int64)
        valid_idx = np.asarray(fold_payload["valid_idx"], dtype=np.int64)
        fold_start = time.perf_counter()
        print(f"[{now_text()}] Fold {fold_idx}/{len(unseen_folds)}: unseen split + single-grid best-param")
        print(f"[{now_text()}] Fold {fold_idx}: train={len(train_idx)} test={len(valid_idx)}")
        fold_key = str(fold_idx)
        run_state.setdefault("folds", {})
        fold_state = run_state["folds"].setdefault(
            fold_key,
            {
                "completed": False,
                "candidate_results": {},
            },
        )
        fold_state.setdefault("candidate_results", {})
        save_run_state(out_dirs["state"], run_state)
        saved_model_path = str(fold_state.get("model_path", "")).strip()
        saved_best_candidate_idx = fold_state.get("best_candidate_idx")
        saved_best_candidate_params = fold_state.get("best_candidate_params")
        saved_best_candidate_score = fold_state.get("best_candidate_score")
        saved_fold_metric = fold_state.get("fold_metric")
        saved_fold_prediction = fold_state.get("fold_prediction")
        saved_fold_split = fold_state.get("fold_split")
        saved_detail_section = fold_state.get("detail_section")
        payload_complete = all(
            item is not None
            for item in (
                saved_best_candidate_idx,
                saved_best_candidate_params,
                saved_best_candidate_score,
                saved_fold_metric,
                saved_fold_prediction,
                saved_fold_split,
            )
        )
        if fold_state.get("completed"):
            if saved_model_path and os.path.exists(saved_model_path) and payload_complete:
                print(f"[{now_text()}] [RESUME] Skip fold {fold_idx}")
                per_fold_metrics.append(saved_fold_metric)
                per_fold_params.append({"fold": fold_idx, **prettify_params(saved_best_candidate_params)})
                fold_prediction_payloads.append(saved_fold_prediction)
                fold_split_payloads.append(saved_fold_split)
                if saved_detail_section is not None:
                    detail_sections.append(str(saved_detail_section))
                if num_classes == 2:
                    y_true_saved = np.asarray(saved_fold_prediction["y_true"], dtype=np.int64)
                    y_prob_saved = np.asarray(saved_fold_prediction["y_prob"], dtype=np.float32)
                    if y_true_saved.size and y_prob_saved.size and len(np.unique(y_true_saved)) > 1:
                        fpr_saved, tpr_saved, _ = roc_curve(y_true_saved, y_prob_saved)
                        precision_saved, recall_saved, _ = precision_recall_curve(y_true_saved, y_prob_saved)
                        metrics_saved = evaluate_binary_fold(
                            y_true_saved,
                            np.asarray(saved_fold_prediction["y_pred"], dtype=np.int64),
                            y_prob_saved,
                        )
                        roc_entries.append(
                            {"fold": fold_idx, "fpr": fpr_saved, "tpr": tpr_saved, "auc": metrics_saved["ROC-AUC"]}
                        )
                        pr_entries.append(
                            {"fold": fold_idx, "precision": precision_saved, "recall": recall_saved, "ap": metrics_saved["PR-AUC"]}
                        )
                continue
            print(
                f"[{now_text()}] [RESUME] Fold {fold_idx} ditandai completed tetapi model/payload tidak valid. "
                "Fold akan dihitung ulang."
            )

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

        scale_pos_weight = get_scale_pos_weight(y_train) if num_classes == 2 else None
        class_distribution = Counter(y_train.tolist())

        preprocessor = build_preprocessor(X_train, numeric_cols, categorical_cols)
        classifier = make_xgb_classifier(num_classes=num_classes, scale_pos_weight=scale_pos_weight)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        sampled_candidates = [dict(params) for params in candidate_grid]

        print(f"Fold {fold_idx}/{len(unseen_folds)}")
        inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED + fold_idx)
        candidate_logs: list[str] = []
        best_candidate_score = float("-inf")
        best_candidate_params: dict[str, Any] | None = None
        best_candidate_idx: int | None = None

        for candidate_idx, candidate_params in enumerate(sampled_candidates, start=1):
            candidate_key = str(candidate_idx)
            saved_candidate = fold_state["candidate_results"].get(candidate_key)
            if saved_candidate and saved_candidate.get("completed"):
                saved_params = saved_candidate.get("params")
                saved_inner_scores = saved_candidate.get("inner_scores")
                saved_mean_score = saved_candidate.get("mean_score")
                saved_std_score = saved_candidate.get("std_score")
                saved_metrics = saved_candidate.get("metrics")
                saved_cm = saved_candidate.get("confusion")
                saved_elapsed = saved_candidate.get("elapsed_seconds")
                payload_complete = all(
                    item is not None
                    for item in (
                        saved_params,
                        saved_inner_scores,
                        saved_mean_score,
                        saved_std_score,
                        saved_metrics,
                        saved_cm,
                        saved_elapsed,
                    )
                )
                if payload_complete:
                    print(f"  [RESUME] Skip fold {fold_idx} candidate {candidate_idx}")
                    cached_detail = render_candidate_detail(
                        index=candidate_idx,
                        param_string=format_candidate_params_for_cli(saved_params),
                        metrics=saved_metrics,
                        confusion=saved_cm,
                        elapsed_seconds=float(saved_elapsed),
                        selection_metric_name=selection_metric_name,
                        selection_metric_value=float(saved_mean_score),
                    )
                    candidate_logs.extend(cached_detail)
                    candidate_logs.append("")
                    if float(saved_mean_score) > best_candidate_score:
                        best_candidate_score = float(saved_mean_score)
                        best_candidate_params = dict(saved_params)
                        best_candidate_idx = int(candidate_idx)
                    continue

            candidate_start = time.perf_counter()
            print(f"  Kandidat {candidate_idx}/{len(sampled_candidates)}")
            print(f"    {format_candidate_params_for_cli(candidate_params)}")

            candidate_model = clone(pipeline)
            candidate_model.set_params(**candidate_params)

            inner_scores = []
            for inner_fold_idx, (inner_train_idx, inner_valid_idx) in enumerate(inner_cv.split(X_train, y_train), start=1):
                fold_score = score_inner_fold(
                    model=clone(candidate_model),
                    X_train=X_train,
                    y_train=y_train,
                    train_index=inner_train_idx,
                    valid_index=inner_valid_idx,
                    num_classes=num_classes,
                    threshold=threshold,
                    selection_metric=selection_metric,
                )
                inner_scores.append(fold_score)
                print(f"    Inner fold {inner_fold_idx}/{INNER_SPLITS} score={fold_score:.6f}")

            mean_score = float(np.mean(inner_scores))
            std_score = float(np.std(inner_scores, ddof=1)) if len(inner_scores) > 1 else 0.0
            print(f"    Mean inner score={mean_score:.6f} | std={std_score:.6f}")

            candidate_eval_model = clone(candidate_model)
            candidate_eval_model.fit(X_train, y_train)
            candidate_valid_proba = candidate_eval_model.predict_proba(X_valid)
            if num_classes == 2:
                candidate_valid_pred = (candidate_valid_proba[:, 1] >= threshold).astype(int)
            else:
                candidate_valid_pred = candidate_eval_model.predict(X_valid)
            candidate_elapsed = time.perf_counter() - candidate_start
            if num_classes == 2:
                candidate_metrics = evaluate_binary_fold(y_valid, candidate_valid_pred, candidate_valid_proba[:, 1])
            else:
                candidate_metrics, _, _ = evaluate_multiclass_fold(y_valid, candidate_valid_pred, candidate_valid_proba)
            candidate_cm = confusion_counts_for_report(y_valid, candidate_valid_pred, num_classes)
            candidate_detail = render_candidate_detail(
                index=candidate_idx,
                param_string=format_candidate_params_for_cli(candidate_params),
                metrics=candidate_metrics,
                confusion=candidate_cm,
                elapsed_seconds=candidate_elapsed,
                selection_metric_name=selection_metric_name,
                selection_metric_value=mean_score,
            )
            for line in candidate_detail:
                print(line)
            candidate_logs.extend(candidate_detail)
            candidate_logs.append("")

            fold_state["candidate_results"][candidate_key] = {
                "completed": True,
                "params": to_serializable(candidate_params),
                "inner_scores": to_serializable(inner_scores),
                "mean_score": float(mean_score),
                "std_score": float(std_score),
                "metrics": to_serializable(candidate_metrics),
                "confusion": to_serializable(candidate_cm),
                "elapsed_seconds": float(candidate_elapsed),
            }
            save_run_state(out_dirs["state"], run_state)

            if mean_score > best_candidate_score:
                best_candidate_score = mean_score
                best_candidate_params = dict(candidate_params)
                best_candidate_idx = int(candidate_idx)

        if best_candidate_params is None:
            raise RuntimeError(f"Gagal menentukan best hyperparameter pada fold {fold_idx}.")

        best_model = clone(pipeline)
        best_model.set_params(**best_candidate_params)
        best_model.fit(X_train, y_train)

        predict_start = time.perf_counter()
        y_valid_proba = best_model.predict_proba(X_valid)
        if num_classes == 2:
            y_valid_pred = (y_valid_proba[:, 1] >= threshold).astype(int)
        else:
            y_valid_pred = best_model.predict(X_valid)
        prediction_time = time.perf_counter() - predict_start
        total_time = time.perf_counter() - fold_start
        training_time = max(total_time - prediction_time, 0.0)

        oof_pred[valid_idx] = y_valid_pred
        oof_proba[valid_idx, :] = y_valid_proba

        if num_classes == 2:
            y_valid_score = y_valid_proba[:, 1]
            metrics = evaluate_binary_fold(y_valid, y_valid_pred, y_valid_score)
            fpr, tpr, _ = roc_curve(y_valid, y_valid_score)
            precision_curve, recall_curve, _ = precision_recall_curve(y_valid, y_valid_score)
            roc_entries.append({"fold": fold_idx, "fpr": fpr, "tpr": tpr, "auc": metrics["ROC-AUC"]})
            pr_entries.append(
                {"fold": fold_idx, "precision": precision_curve, "recall": recall_curve, "ap": metrics["PR-AUC"]}
            )
            auc_note = "ROC-AUC dan PR-AUC dihitung pada probabilitas kelas positif."
        else:
            metrics, roc_note, pr_note = evaluate_multiclass_fold(y_valid, y_valid_pred, y_valid_proba)
            auc_note = f"{roc_note}; {pr_note}."

        cm = confusion_matrix(y_valid, y_valid_pred, labels=np.arange(num_classes))
        save_confusion_matrix_plot(
            cm=cm,
            labels=class_labels,
            title=f"Confusion Matrix Fold {fold_idx}",
            output_path=out_dirs["plots"] / f"confusion_matrix_fold_{fold_idx}.png",
        )

        best_params_clean = prettify_params(best_candidate_params)
        per_fold_metrics.append(
            {
                "fold": fold_idx,
                **metrics,
                "train_time_s": training_time,
                "eval_time_s": prediction_time,
            }
        )
        per_fold_params.append({"fold": fold_idx, **best_params_clean})

        fold_split_payloads.append(
            {
                "fold": int(fold_idx),
                "train_indices": np.asarray(train_idx, dtype=np.int64),
                "val_indices": np.asarray(valid_idx, dtype=np.int64),
            }
        )

        fold_prob_output: Any
        if num_classes == 2:
            fold_prob_output = y_valid_proba[:, 1]
        else:
            fold_prob_output = y_valid_proba
        fold_prediction_payloads.append(
            {
                "fold": int(fold_idx),
                "sample_indices": np.asarray(valid_idx, dtype=np.int64),
                "y_true": np.asarray(y_valid, dtype=np.int64),
                "y_pred": np.asarray(y_valid_pred, dtype=np.int64),
                "y_prob": np.asarray(fold_prob_output),
            }
        )

        model_path = fold_model_dir / f"fold_{fold_idx}.pkl"
        joblib.dump(best_model, model_path)

        cm_report = confusion_counts_for_report(y_valid, y_valid_pred, num_classes)
        selected_detail = render_selected_detail(
            param_string=format_candidate_params_for_cli(best_candidate_params),
            selection_metric_name=selection_metric_name,
            selection_metric_value=best_candidate_score,
            metrics=metrics,
            confusion=cm_report,
        )
        for line in selected_detail:
            print(line)

        detail_section = "\n".join(
            [
                f"Fold {fold_idx}/{len(unseen_folds)}",
                *candidate_logs,
                *selected_detail,
            ]
        )
        detail_sections.append(detail_section)
        fold_state["completed"] = True
        fold_state["model_path"] = str(model_path)
        fold_state["best_candidate_idx"] = int(best_candidate_idx) if best_candidate_idx is not None else None
        fold_state["best_candidate_params"] = to_serializable(best_candidate_params)
        fold_state["best_candidate_score"] = float(best_candidate_score)
        fold_state["fold_metric"] = to_serializable(per_fold_metrics[-1])
        fold_state["fold_prediction"] = to_serializable(fold_prediction_payloads[-1])
        fold_state["fold_split"] = to_serializable(fold_split_payloads[-1])
        fold_state["detail_section"] = detail_section
        save_run_state(out_dirs["state"], run_state)

        print(
            f"[{now_text()}] Fold {fold_idx}/{len(unseen_folds)} done | "
            f"ROC-AUC={metrics['ROC-AUC']:.4f} | PR-AUC={metrics['PR-AUC']:.4f} | "
            f"Acc={metrics['Accuracy']:.4f} | F1={metrics['F1']:.4f}"
        )

    metrics_df = pd.DataFrame(per_fold_metrics)
    params_df = pd.DataFrame(per_fold_params)

    metric_columns = ["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1"]
    aggregated_metrics = pd.DataFrame(
        [metrics_df[metric_columns].mean(), metrics_df[metric_columns].std(ddof=1)],
        index=["mean", "std"],
    )

    fold_param_counter = Counter(params_to_hashable({k: v for k, v in record.items() if k != "fold"}) for record in per_fold_params)
    voted_hashable, voted_count = fold_param_counter.most_common(1)[0]
    voted_params = dict(voted_hashable)

    param_value_votes: dict[str, Counter] = defaultdict(Counter)
    for record in per_fold_params:
        for key, value in record.items():
            if key == "fold":
                continue
            normalized = round(value, 10) if isinstance(value, float) else value
            param_value_votes[key][normalized] += 1

    voting_lines = []
    for key in sorted(param_value_votes):
        voted_value, count = param_value_votes[key].most_common(1)[0]
        voting_lines.append(f"{key}: {voted_value} (dipilih {count}/{len(unseen_folds)} fold)")

    final_numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    final_categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    final_preprocessor = build_preprocessor(X, final_numeric_cols, final_categorical_cols)
    final_classifier = make_xgb_classifier(
        num_classes=num_classes,
        scale_pos_weight=get_scale_pos_weight(y) if num_classes == 2 else None,
    )
    final_classifier.set_params(**voted_params)

    final_model = Pipeline(
        steps=[
            ("preprocessor", final_preprocessor),
            ("classifier", final_classifier),
        ]
    )

    print(f"[{now_text()}] Retraining final model on full clean dataset with voted params")
    final_model.fit(X, y)

    joblib.dump(final_model, out_dirs["checkpoints"] / "final_model.pkl")
    joblib.dump(final_preprocessor, out_dirs["checkpoints"] / "preprocessor.pkl")
    joblib.dump(label_encoder, out_dirs["checkpoints"] / "label_encoder.pkl")
    final_model.named_steps["classifier"].save_model(str(out_dirs["checkpoints"] / "final_model.json"))

    safe_json_dump(
        {
            "input_csv": str(INPUT_CSV),
            "split_root": str(args.split_root),
            "scenario": SCENARIO,
            "scenario_label": SCENARIO_LABEL,
            "output_dir": str(out_dirs["log_root"]),
            "selected_target": target_col,
            "selected_target_reason": target_reason,
            "threshold": threshold,
            "voted_param_set_count": voted_count,
            "voted_params": voted_params,
            "parameter_value_votes": {
                key: dict(counter)
                for key, counter in sorted(param_value_votes.items())
            },
            "class_labels": class_labels,
            "random_seed": RANDOM_SEED,
            "split_root": str(args.split_root),
            "scenario": SCENARIO,
        },
        out_dirs["reports"] / "best_params_final.json",
    )

    if num_classes == 2:
        save_roc_plot(roc_entries, out_dirs["plots"] / "roc_auc_kfold_all.png")
        save_pr_plot(pr_entries, out_dirs["plots"] / "pr_curve_kfold_all.png")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axis("off")
        ax.text(
            0.01,
            0.95,
            "Plot ROC aggregate tidak dibuat untuk multiclass di script ini.\nLihat log untuk pendekatan OVR macro average.",
            va="top",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dirs["plots"] / "roc_auc_kfold_all.png", dpi=200)
        fig.savefig(out_dirs["plots"] / "pr_curve_kfold_all.png", dpi=200)
        plt.close(fig)

    save_metric_bar_plot(metrics_df, out_dirs["plots"] / "avg_metric_kfold.png")

    evaluated_mask = oof_pred >= 0
    if not np.any(evaluated_mask):
        raise RuntimeError("Tidak ada sample test yang dievaluasi untuk OOF summary.")
    y_oof = y[evaluated_mask]
    oof_pred_eval = oof_pred[evaluated_mask]
    oof_proba_eval = oof_proba[evaluated_mask]

    oof_cm = confusion_matrix(y_oof, oof_pred_eval, labels=np.arange(num_classes))
    save_confusion_matrix_plot(
        cm=oof_cm,
        labels=class_labels,
        title="OOF Confusion Matrix",
        output_path=out_dirs["plots"] / "oof_confusion_matrix.png",
    )

    threshold_recap_df = pd.DataFrame()
    best_threshold_recap: dict[str, Any] = {}
    if num_classes == 2:
        threshold_recap_df = evaluate_binary_threshold_recap(y_oof, oof_proba_eval[:, 1])
        best_threshold_recap = threshold_recap_df.loc[threshold_recap_df["F1"].idxmax()].to_dict()
        threshold_recap_df.to_csv(out_dirs["reports"] / "threshold_recap_0_1_to_1_0.csv", index=False)
        threshold_recap_df.to_csv(out_dirs["fold_reports"] / "threshold_recap_0_1_to_1_0.csv", index=False)
        save_threshold_recap_plot(threshold_recap_df, out_dirs["plots"] / "threshold_recap_0_1_to_1_0.png")
        safe_json_dump(
            {
                "thresholds": threshold_recap_df["threshold"].tolist(),
                "selection_metric": "F1",
                "best_threshold": best_threshold_recap,
            },
            out_dirs["reports"] / "threshold_recap_0_1_to_1_0.json",
        )

    metrics_df.to_csv(out_dirs["reports"] / "kfold_metrics.csv", index=False)
    metrics_df.to_csv(out_dirs["fold_reports"] / "kfold_metrics.csv", index=False)
    params_df.to_csv(out_dirs["reports"] / "kfold_params.csv", index=False)
    aggregated_metrics.to_csv(out_dirs["reports"] / "cv_summary.csv")
    safe_json_dump(
        {
            "num_samples": int(len(df)),
            "num_features": int(X.shape[1]),
            "target_col": target_col,
            "class_labels": class_labels,
            "num_classes": int(num_classes),
            "num_evaluated_oof_samples": int(evaluated_mask.sum()),
            "num_buffer_or_unevaluated_samples": int((~evaluated_mask).sum()),
        },
        out_dirs["reports"] / "dataset_summary.json",
    )
    np.savez_compressed(
        out_dirs["fold_root"] / "split_indices.npz",
        folds=np.asarray(fold_split_payloads, dtype=object),
    )

    for item in fold_prediction_payloads:
        fold = int(item["fold"])
        sample_indices = np.asarray(item["sample_indices"], dtype=np.int64)
        y_true_fold = np.asarray(item["y_true"], dtype=np.int64)
        y_pred_fold = np.asarray(item["y_pred"], dtype=np.int64)
        y_prob_fold = np.asarray(item["y_prob"])
        np.savez_compressed(
            out_dirs["predictions"] / f"fold_{fold}_predictions.npz",
            sample_indices=sample_indices,
            y_true=y_true_fold,
            y_pred=y_pred_fold,
            y_prob=y_prob_fold,
        )
        fold_df = pd.DataFrame(
            {
                "sample_indices": sample_indices,
                "y_true": y_true_fold,
                "y_pred": y_pred_fold,
            }
        )
        if y_prob_fold.ndim == 1:
            fold_df["y_prob"] = y_prob_fold
        else:
            for cls_idx in range(y_prob_fold.shape[1]):
                fold_df[f"y_prob_class_{cls_idx}"] = y_prob_fold[:, cls_idx]
        fold_df.to_csv(out_dirs["predictions"] / f"fold_{fold}_predictions.csv", index=False)

    np.savez_compressed(
        out_dirs["predictions"] / "kfold_predictions.npz",
        folds=np.asarray(fold_prediction_payloads, dtype=object),
    )
    oof_prob_output = oof_proba[:, 1] if num_classes == 2 else oof_proba
    np.savez_compressed(
        out_dirs["predictions"] / "oof_predictions.npz",
        sample_indices=np.arange(len(y), dtype=np.int64),
        y_true=np.asarray(y, dtype=np.int64),
        y_pred=np.asarray(oof_pred, dtype=np.int64),
        y_prob=np.asarray(oof_prob_output),
    )

    timing_lines = build_execution_report(start_time)
    per_fold_text = dataframe_to_string(metrics_df[[*METRIC_COLUMNS, "fold"]], index=False)
    aggregated_text = dataframe_to_string(aggregated_metrics, index=True)
    params_text = params_df.to_string(index=False)

    summary_lines = [
        "Detail single-grid best-param & evaluasi",
        *detail_sections,
        "",
        "Per-fold metrics",
        per_fold_text,
        "",
        "Aggregated metrics",
        aggregated_text,
        "",
        "Threshold recap OOF (0.1-1.0)",
        dataframe_to_string(threshold_recap_df, index=False) if not threshold_recap_df.empty else "Tidak dibuat untuk multiclass.",
        f"Best threshold by F1: {json.dumps(best_threshold_recap, ensure_ascii=False)}" if best_threshold_recap else "",
        "",
        "Model params per fold",
        params_text,
        "",
        "Voting Best Model Param",
        *voting_lines,
        "",
        "Informasi run",
        f"Input CSV path: {INPUT_CSV}",
        f"Run name: {out_dirs['run_name'].name}",
        f"Log directory: {out_dirs['log_root']}",
        f"Fold directory: {out_dirs['fold_root']}",
        f"Model directory: {out_dirs['model_root']}",
        f"Timestamp: {now_text()}",
        f"Script: {SCRIPT_NAME}",
        f"Dataset shape: {df.shape}",
        "",
        "Hasil analisis kolom",
        f"- Jumlah baris: {df.shape[0]}",
        f"- Jumlah kolom: {df.shape[1]}",
        f"- Kolom identifier: {identifier_cols}",
        f"- Kandidat target teratas: {json.dumps(possible_targets[:5], ensure_ascii=False, indent=2)}",
        f"- Target yang dipilih: {target_col}",
        f"- Alasan target: {target_reason}",
        f"- Kolom numerik final: {len(final_numeric_cols)}",
        f"- Kolom kategorikal final: {len(final_categorical_cols)}",
        f"- Kolom konstanta: {column_analysis['constant_columns']}",
        f"- Duplicate rows: {column_analysis['duplicate_rows']}",
        f"- Duplicate columns (sample 20): {column_analysis['duplicate_columns'][:20]}",
        f"- Missing value total: {sum(column_analysis['missing_values'].values())}",
        "",
        "Nama semua kolom",
        ", ".join(df.columns.tolist()),
        "",
        "Dtype tiap kolom",
        pd.Series(column_analysis["dtypes"]).to_string(),
        "",
        "Jumlah missing values per kolom",
        pd.Series(column_analysis["missing_values"]).to_string(),
        "",
        "Jumlah nilai unik per kolom",
        pd.Series(column_analysis["unique_values"]).to_string(),
        "",
        "Preview 5 baris awal",
        column_analysis["preview"],
        "",
        "Distribusi kelas target",
        json.dumps(pd.Series(y_raw).value_counts(dropna=False).to_dict(), ensure_ascii=False),
        "",
        "Daftar fitur final yang digunakan",
        ", ".join(feature_cols),
        "",
    ]

    performance_line = (
        f"Performa rata-rata: ROC-AUC={aggregated_metrics.loc['mean', 'ROC-AUC']:.6f}, "
        f"PR-AUC={aggregated_metrics.loc['mean', 'PR-AUC']:.6f}, "
        f"Accuracy={aggregated_metrics.loc['mean', 'Accuracy']:.6f}, "
        f"F1={aggregated_metrics.loc['mean', 'F1']:.6f}."
    )
    stability_line = (
        f"Stabilitas antar fold (std): ROC-AUC={aggregated_metrics.loc['std', 'ROC-AUC']:.6f}, "
        f"PR-AUC={aggregated_metrics.loc['std', 'PR-AUC']:.6f}, "
        f"Accuracy={aggregated_metrics.loc['std', 'Accuracy']:.6f}, "
        f"F1={aggregated_metrics.loc['std', 'F1']:.6f}."
    )
    consistency_line = f"Hyperparameter yang paling konsisten: {json.dumps(voted_params, ensure_ascii=False)}."
    notes = (
        f"Catatan data: target seimbang {pd.Series(y_raw).value_counts().to_dict()}, "
        f"missing values = 0, identifier dibuang = {identifier_cols}, "
        f"duplikasi sequence-level ada tetapi full-row duplicate = {column_analysis['duplicate_rows']}."
    )

    summary_lines.extend(
        [
            "Kesimpulan akhir",
            performance_line,
            stability_line,
            consistency_line,
            notes,
            "",
            *timing_lines,
        ]
    )

    log_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\nPer-fold metrics")
    print(per_fold_text)
    print("\nAggregated metrics")
    print(aggregated_text)
    if not threshold_recap_df.empty:
        print("\nThreshold recap OOF (0.1-1.0)")
        print(dataframe_to_string(threshold_recap_df, index=False))
        print(f"\nBest threshold by F1: {json.dumps(best_threshold_recap, ensure_ascii=False)}")
    print("\nModel params per fold")
    print(params_text)
    print("\nVoting Best Model Param")
    print("\n".join(voting_lines))
    for line in timing_lines:
        print(line)
    safe_json_dump(
        {
            "run_name": out_dirs["run_name"].name,
            "selected_target": target_col,
            "selected_target_reason": target_reason,
            "kfold_mean": aggregated_metrics.loc["mean"].to_dict(),
            "kfold_std": aggregated_metrics.loc["std"].to_dict(),
            "voted_params": voted_params,
            "num_folds": int(len(unseen_folds)),
        },
        out_dirs["reports"] / "run_summary.json",
    )
    safe_json_dump(
        {
            "input_csv": str(INPUT_CSV),
            "random_seed": RANDOM_SEED,
            "split_root": str(args.split_root),
            "scenario": SCENARIO,
            "outer_splits": len(unseen_folds),
            "inner_splits": INNER_SPLITS,
            "n_iter_search": N_ITER_SEARCH,
            "script_name": SCRIPT_NAME,
        },
        out_dirs["reports"] / "config.json",
    )

    print(f"[{now_text()}] Saved log to {log_path}")
    print(f"[{now_text()}] All artifacts saved to {out_dirs['log_root']}")


if __name__ == "__main__":
    main()
