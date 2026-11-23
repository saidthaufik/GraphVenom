"""Final CAMP training script for wide numeric peptide-protein interaction data.

This script reproduces the agreed CAMP pipeline using prefix-based feature
extraction, leakage-safe target handling, cross-validation, early stopping,
artifact saving, early stopping, and test-set evaluation.

This regularized single-grid variant intentionally uses a smaller CAMP
backbone plus stronger regularization so model1 is a more conservative
baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:
        Literal = str  # type: ignore

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_cuda_library_path() -> None:
    """Re-exec with bundled NVIDIA library paths so cuDNN sub-libraries resolve."""
    if os.environ.get("CAMP_CUDA_LIBRARY_PATH_READY") == "1":
        return

    site_packages = Path(sys.executable).resolve().parents[1] / "lib" / "python3.10" / "site-packages"
    candidate_dirs = [
        site_packages / "nvidia" / "cudnn" / "lib",
        site_packages / "nvidia" / "cublas" / "lib",
        site_packages / "nvidia" / "cuda_runtime" / "lib",
    ]
    library_dirs = [str(path) for path in candidate_dirs if path.exists()]
    if not library_dirs:
        os.environ["CAMP_CUDA_LIBRARY_PATH_READY"] = "1"
        return

    current_paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    merged_paths = library_dirs + [path for path in current_paths if path not in library_dirs]
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged_paths)
    os.environ["CAMP_CUDA_LIBRARY_PATH_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_library_path()

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from path import model_path, output_roots
from reporting import (
    build_execution_report,
    dataframe_to_string,
    format_confusion_matrix,
    format_metrics,
    render_candidate_detail,
    render_sections,
    render_selected_detail,
    voting_lines,
)


ModeName = Literal["strict_notebook_mode", "clean_mode"]
DATASET_NAME = "rki"

EXPECTED_BLOCK_SIZES: Dict[str, int] = {
    "pep_pssm": 400,
    "pep_id": 150,
    "pep_aas": 50,
    "pep_pp": 50,
    "prot_pssm": 400,
    "prot_id": 4500,
    "prot_aas": 1500,
    "prot_pp": 1500,
}

CONCAT_ORDER: List[str] = [
    "pep_aas",
    "prot_aas",
    "pep_pssm",
    "prot_pssm",
    "pep_pp",
    "prot_pp",
    "pep_id",
    "prot_id",
]

REQUIRED_METADATA_COLUMNS: List[str] = [
    "pdb_id",
    "pep_seq",
    "Uniprot_id",
    "prot_seq",
]


def get_default_device() -> str:
    return "cuda"


@dataclass
class Config:
    # Hyperparameter utama model1 didefinisikan di sini sebagai fixed config.
    # Ini bukan grid search; nilainya berubah hanya jika diubah manual / via CLI.
    csv_path: Path = model_path("model1", "0-neg", "gan_decoy.csv")
    output_dir: Path = PROJECT_ROOT / "cache" / "runs" / "benchmark" / "b-log_result" / "model1_sing_reg"
    target_col: str = "label"
    mode: ModeName = "clean_mode"
    run_name: str = ""
    random_state: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    batch_size: int = 256
    learning_rate: float = 3e-4
    epochs: int = 200
    dropout_rate: float = 0.7
    weight_decay: float = 5e-3
    conv1_channels: int = 16
    conv2_channels: int = 32
    dense_hidden: int = 16
    num_workers: int = 0
    threshold: float = 0.5
    tune_grid: bool = False
    early_stopping: bool = True
    early_stopping_monitor: str = "val_pr_auc"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-3
    strict_fail_on_degenerate_fold: bool = False
    save_fold_predictions: bool = True
    save_plots: bool = True
    device: str = get_default_device()

    def validate(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError("CSV not found: {}".format(self.csv_path))
        if self.mode not in ("strict_notebook_mode", "clean_mode"):
            raise ValueError("Unsupported mode: {}".format(self.mode))
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.early_stopping_monitor not in {"val_loss", "val_roc_auc", "val_pr_auc", "val_f1"}:
            raise ValueError("Unsupported early_stopping_monitor: {}".format(self.early_stopping_monitor))
        if self.early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping_mode must be min or max.")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("threshold must be between 0 and 1.")
        if self.conv1_channels <= 0 or self.conv2_channels <= 0 or self.dense_hidden <= 0:
            raise ValueError("Model channel and hidden sizes must be positive.")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train and evaluate CAMP on a wide numeric CSV dataset.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=model_path("model1", "0-neg", "gan_decoy.csv"),
        help="Path to wide numeric CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "cache" / "runs" / "benchmark" / "b-log_result" / "model1_sing_reg",
        help="Root directory untuk log artifacts.",
    )
    parser.add_argument("--run-name", type=str, default="model1_sing_reg", help="Nama run untuk metadata. Default: model1_sing_reg.")
    parser.add_argument(
        "--mode",
        choices=["strict_notebook_mode", "clean_mode"],
        default="clean_mode",
        help="Execution mode.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4, dest="learning_rate")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dropout-rate", type=float, default=0.7)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--conv1-channels", type=int, default=16)
    parser.add_argument("--conv2-channels", type=int, default=32)
    parser.add_argument("--dense-hidden", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument(
        "--tune-grid",
        action="store_true",
        help="Jalankan grid search 12 kombinasi batch_size, learning_rate, dan dropout_rate sebelum evaluasi final.",
    )
    parser.add_argument("--no-early-stopping", action="store_true", help="Matikan early stopping dan jalankan semua epoch.")
    parser.add_argument("--early-stopping-monitor", choices=["val_loss", "val_roc_auc", "val_pr_auc", "val_f1"], default="val_pr_auc")
    parser.add_argument("--early-stopping-mode", choices=["min", "max"], default="max")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    args = parser.parse_args()

    config = Config(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        mode=args.mode,
        run_name=args.run_name.strip(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay,
        conv1_channels=args.conv1_channels,
        conv2_channels=args.conv2_channels,
        dense_hidden=args.dense_hidden,
        num_workers=args.num_workers,
        threshold=args.threshold,
        tune_grid=args.tune_grid,
        early_stopping=not args.no_early_stopping,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_mode=args.early_stopping_mode,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        device=args.device,
        strict_fail_on_degenerate_fold=(args.mode == "clean_mode"),
    )
    config.validate()
    return config


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the wide numeric CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty: {}".format(csv_path))
    return df


def _sort_prefixed_columns(columns: Iterable[str], prefix: str) -> List[str]:
    prefix_with_sep = prefix + "_"
    matched = [col for col in columns if col.startswith(prefix_with_sep)]
    if not matched:
        return []

    def suffix_index(name: str) -> int:
        suffix = name[len(prefix_with_sep) :]
        if not suffix.isdigit():
            raise ValueError("Column does not end with numeric suffix: {}".format(name))
        return int(suffix)

    matched.sort(key=suffix_index)
    return matched


def validate_schema(df: pd.DataFrame, target_col: str = "label") -> Dict[str, Any]:
    """Validate mandatory columns, block sizes, target integrity, and missing values."""
    missing_required = [col for col in REQUIRED_METADATA_COLUMNS + [target_col] if col not in df.columns]
    if missing_required:
        raise ValueError("Missing required columns: {}".format(missing_required))

    feature_block_columns = get_feature_block_columns(df)
    for block_name, expected_size in EXPECTED_BLOCK_SIZES.items():
        actual_size = len(feature_block_columns.get(block_name, []))
        if actual_size != expected_size:
            raise ValueError(
                "Block '{}' expected {} columns but found {}.".format(
                    block_name, expected_size, actual_size
                )
            )

    if df[target_col].isna().any():
        raise ValueError("Target column contains missing values: {}".format(target_col))

    unique_targets = set(pd.Series(df[target_col]).dropna().astype(str).unique().tolist())
    normalized = {str(v) for v in unique_targets}
    if normalized != {"0", "1"}:
        raise ValueError("Target column must be binary 0/1. Found: {}".format(sorted(unique_targets)))

    feature_columns: List[str] = []
    for block_name in CONCAT_ORDER:
        feature_columns.extend(feature_block_columns[block_name])

    if target_col in feature_columns:
        raise ValueError("Target column leaked into feature columns.")

    numeric_frame = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        raise ValueError("Feature matrix contains non-numeric or missing values after coercion.")

    schema_info: Dict[str, Any] = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "target_col": target_col,
        "target_distribution": df[target_col].value_counts(dropna=False).to_dict(),
        "feature_blocks": {name: len(cols) for name, cols in feature_block_columns.items()},
        "concat_order": CONCAT_ORDER,
        "input_dim": sum(EXPECTED_BLOCK_SIZES[name] for name in CONCAT_ORDER),
    }
    return schema_info


def get_feature_block_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return ordered block columns using prefix-based matching."""
    feature_blocks: Dict[str, List[str]] = {}
    for block_name in EXPECTED_BLOCK_SIZES:
        block_cols = _sort_prefixed_columns(df.columns, block_name)
        if block_cols:
            expected_suffixes = list(range(len(block_cols)))
            actual_suffixes = [int(col.rsplit("_", 1)[1]) for col in block_cols]
            if actual_suffixes != expected_suffixes:
                raise ValueError("Non-contiguous suffix order detected in block '{}'.".format(block_name))
        feature_blocks[block_name] = block_cols
    return feature_blocks


def build_feature_matrix(
    df: pd.DataFrame,
    block_columns: Dict[str, List[str]],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build leakage-safe feature matrix and binary target vector."""
    feature_columns: List[str] = []
    for block_name in CONCAT_ORDER:
        feature_columns.extend(block_columns[block_name])

    if target_col in feature_columns:
        raise ValueError("Target column '{}' must not be included in features.".format(target_col))

    metadata = df[REQUIRED_METADATA_COLUMNS].copy()
    X = df[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    y = pd.to_numeric(df[target_col], errors="raise").to_numpy(dtype=np.float32)

    expected_input_dim = sum(EXPECTED_BLOCK_SIZES[name] for name in CONCAT_ORDER)
    if X.ndim != 2 or X.shape[1] != expected_input_dim:
        raise ValueError(
            "Feature matrix shape mismatch. Expected second dim {}, got {}.".format(
                expected_input_dim, X.shape[1] if X.ndim == 2 else "invalid"
            )
        )
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("Target vector shape mismatch.")

    return X, y, metadata


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    config: Config,
) -> Dict[str, Any]:
    """Split dataset into train/validation pool and hold-out test set."""
    indices = np.arange(len(y))
    split = train_test_split(
        X,
        y,
        metadata,
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    (
        X_train_val,
        X_test,
        y_train_val,
        y_test,
        meta_train_val,
        meta_test,
        idx_train_val,
        idx_test,
    ) = split

    return {
        "X_train_val": X_train_val,
        "X_test": X_test,
        "y_train_val": y_train_val,
        "y_test": y_test,
        "meta_train_val": meta_train_val,
        "meta_test": meta_test,
        "idx_train_val": idx_train_val,
        "idx_test": idx_test,
    }


def tensorize_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert numpy arrays to float32 tensors."""
    if X.ndim != 2:
        raise ValueError("Expected X to be 2D, got shape {}".format(X.shape))
    if y.ndim != 1:
        raise ValueError("Expected y to be 1D, got shape {}".format(y.shape))
    if len(X) != len(y):
        raise ValueError("X and y row counts do not match.")
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class CAMPModel(nn.Module):
    """1D CNN model faithful to the notebook architecture."""

    def __init__(
        self,
        input_dim: int,
        dropout_rate: float = 0.7,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        dense_hidden: int = 16,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv1_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(conv2_channels, dense_hidden),
            nn.ReLU(),
            nn.Linear(dense_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Expected input shape (batch, input_dim), got {}".format(tuple(x.shape)))
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return torch.sigmoid(out).squeeze(-1)


def _compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute binary classification metrics with guards for degenerate cases."""
    labels = np.asarray(labels).astype(np.float32)
    probs = np.asarray(probs).astype(np.float32)
    preds = (probs >= float(threshold)).astype(np.float32) if probs.size else np.array([], dtype=np.float32)
    if probs.size:
        cm = confusion_matrix(labels.astype(int), preds.astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, preds)) if probs.size else 0.0,
        "precision": float(precision_score(labels, preds, zero_division=0)) if probs.size else 0.0,
        "recall": float(recall_score(labels, preds, zero_division=0)) if probs.size else 0.0,
        "f1": float(f1_score(labels, preds, zero_division=0)) if probs.size else 0.0,
        "roc_auc": None,
        "pr_auc": None,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "preds": preds,
        "fpr": None,
        "tpr": None,
    }
    if probs.size and len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
        fpr, tpr, _ = roc_curve(labels, probs)
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
    return metrics


def _log(message: str) -> None:
    print(message, flush=True)
    if _ACTIVE_RESULTS_LOG is not None:
        _ACTIVE_RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _ACTIVE_RESULTS_LOG.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    serializable = json.loads(json.dumps(payload, default=str))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
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


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(_to_serializable(payload), sort_keys=True, separators=(",", ":"))


def compute_config_signature(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def load_run_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "resume_meta": {},
            "folds": {},
            "final": {},
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    state.setdefault("resume_meta", {})
    state.setdefault("folds", {})
    state.setdefault("final", {})
    return state


def save_run_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_to_serializable(state), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def has_resume_progress(state: Dict[str, Any]) -> bool:
    for fold in state.get("folds", {}).values():
        if fold.get("completed"):
            return True
        resume_path = fold.get("resume_path")
        if resume_path and Path(str(resume_path)).exists():
            return True
    return bool(state.get("final"))


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def safe_torch_save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def set_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_cpu_state = state.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(torch_cpu_state)
    torch_cuda_state = state.get("torch_cuda")
    if torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_state)


def assert_checkpoint_signature(payload: Dict[str, Any], expected_signature: str, label: str) -> None:
    saved_signature = str(payload.get("config_signature", "")).strip()
    if saved_signature and saved_signature != expected_signature:
        raise RuntimeError(
            "Resume checkpoint mismatch pada {}: saved_signature={}, expected_signature={}.".format(
                label, saved_signature, expected_signature
            )
        )


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_output_dirs(config: Config) -> Dict[str, Path]:
    run_name = config.run_name.strip() if config.run_name.strip() else "model1_sing_reg"
    es_root = PROJECT_ROOT / "cache" / "runs" / "benchmark"
    log_root = _ensure_dir(config.output_dir)
    fold_root = _ensure_dir(es_root / "a-fold" / run_name)
    model_root = _ensure_dir(es_root / "c-model_save" / run_name)
    resume_root = _ensure_dir(model_root / "resume" / run_name)
    return {
        "run_name": Path(run_name),
        "log_root": log_root,
        "fold_root": fold_root,
        "model_root": model_root,
        "text": log_root / f"{run_name}_results_{DATASET_NAME}.txt",
        "reports": _ensure_dir(log_root / "reports"),
        "plots": _ensure_dir(log_root / "plots"),
        "fold_reports": _ensure_dir(fold_root / "reports"),
        "predictions": _ensure_dir(fold_root / "predictions"),
        "checkpoints": _ensure_dir(model_root / "checkpoints"),
        "resume": resume_root,
        "state": log_root / f"{run_name}_resume_{DATASET_NAME}.json",
    }


def _summarize_dataset(df: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_samples": int(len(df)),
        "num_positive": int(np.sum(y == 1)),
        "num_negative": int(np.sum(y == 0)),
    }
    if "pep_seq" in df.columns:
        summary["avg_peptide_len"] = float(df["pep_seq"].astype(str).str.len().mean())
    if "prot_seq" in df.columns:
        summary["avg_protein_len"] = float(df["prot_seq"].astype(str).str.len().mean())
    return summary


_ACTIVE_RESULTS_LOG: Optional[Path] = None


def _build_cv_mean_std_summary(fold_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build mean ± std summary rows from fold-level validation metrics."""
    if not fold_metrics:
        return []

    metrics_df = pd.DataFrame(fold_metrics)
    metric_columns = [
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_roc_auc",
        "val_pr_auc",
        "val_loss",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for column in metric_columns:
        if column not in metrics_df.columns:
            continue
        values = pd.to_numeric(metrics_df[column], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric": column,
                "mean": float(values.mean()) if not values.empty else None,
                "std": float(values.std(ddof=0)) if not values.empty else None,
                "mean_pm_std": (
                    "{:.4f} ± {:.4f}".format(values.mean(), values.std(ddof=0))
                    if not values.empty
                    else "NA"
                ),
            }
        )
    return summary_rows


def _metric_report_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ROC-AUC": metrics.get("roc_auc"),
        "PR-AUC": metrics.get("pr_auc"),
        "Accuracy": metrics.get("accuracy"),
        "Precision": metrics.get("precision"),
        "Recall": metrics.get("recall"),
        "F1": metrics.get("f1"),
    }


def _fold_metric_report_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ROC-AUC": row.get("val_roc_auc"),
        "PR-AUC": row.get("val_pr_auc"),
        "Accuracy": row.get("val_accuracy"),
        "Precision": row.get("val_precision"),
        "Recall": row.get("val_recall"),
        "F1": row.get("val_f1"),
        "fold": row.get("fold"),
    }


def _confusion_report_row(metrics: Dict[str, Any]) -> Dict[str, int]:
    return {
        "TP": int(metrics.get("tp", 0)),
        "TN": int(metrics.get("tn", 0)),
        "FP": int(metrics.get("fp", 0)),
        "FN": int(metrics.get("fn", 0)),
    }


def _candidate_param_string(config: Config, runtime: Dict[str, Any], best_epoch: Optional[int] = None) -> str:
    parts = [
        "model=CAMP",
        "mode={}".format(config.mode),
        "epochs={}".format(runtime["epochs"]),
        "batch_size={}".format(runtime["batch_size"]),
        "learning_rate={}".format(runtime["learning_rate"]),
        "dropout_rate={}".format(config.dropout_rate),
        "weight_decay={}".format(config.weight_decay),
        "conv_channels={}-{}".format(config.conv1_channels, config.conv2_channels),
        "dense_hidden={}".format(config.dense_hidden),
    ]
    if best_epoch is not None:
        parts.append("best_epoch={}".format(best_epoch))
    return ", ".join(parts)


def get_hyperparameter_grid() -> List[Dict[str, Any]]:
    """Return the agreed 12-combination tuning grid."""
    grid: List[Dict[str, Any]] = []
    for learning_rate in (1e-4, 3e-4, 5e-4):
        for dropout_rate in (0.1, 0.2):
            for batch_size in (128, 256):
                grid.append(
                    {
                        "learning_rate": float(learning_rate),
                        "dropout_rate": float(dropout_rate),
                        "batch_size": int(batch_size),
                    }
                )
    return grid


def _grid_trial_name(index: int, params: Dict[str, Any]) -> str:
    lr_label = "{:.0e}".format(float(params["learning_rate"])).replace("+", "")
    dropout_label = str(params["dropout_rate"]).replace(".", "p")
    return "grid{:02d}_bs{}_lr{}_drop{}".format(
        index,
        int(params["batch_size"]),
        lr_label,
        dropout_label,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        probs = model(X_batch)
        loss = criterion(probs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y_batch.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(y_batch.detach().cpu().numpy())

    epoch_probs = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    epoch_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    metrics = _compute_binary_metrics(epoch_labels, epoch_probs, threshold)
    metrics["loss"] = total_loss / max(len(dataloader.dataset), 1)
    metrics["labels"] = epoch_labels
    metrics["probs"] = epoch_probs
    return metrics


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run one evaluation epoch without weight updates."""
    model.eval()
    total_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            probs = model(X_batch)
            loss = criterion(probs, y_batch)
            total_loss += float(loss.item()) * y_batch.size(0)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y_batch.detach().cpu().numpy())

    epoch_probs = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    epoch_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    metrics = _compute_binary_metrics(epoch_labels, epoch_probs, threshold)
    metrics["loss"] = total_loss / max(len(dataloader.dataset), 1)
    metrics["labels"] = epoch_labels
    metrics["probs"] = epoch_probs
    return metrics


def _early_stopping_score(stats: Dict[str, Any], monitor: str) -> Optional[float]:
    key_map = {
        "val_loss": "loss",
        "val_roc_auc": "roc_auc",
        "val_pr_auc": "pr_auc",
        "val_f1": "f1",
    }
    value = stats.get(key_map[monitor])
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value_float):
        return None
    return value_float


def _is_early_stopping_improvement(
    current: Optional[float],
    best: Optional[float],
    *,
    mode: str,
    min_delta: float,
) -> bool:
    if current is None:
        return False
    if best is None:
        return True
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


def _resolve_mode_config(config: Config) -> Dict[str, Any]:
    """Derive effective runtime settings from the selected mode."""
    if config.mode == "strict_notebook_mode":
        return {
            "epochs": 100,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "strict_fail_on_degenerate_fold": False,
        }
    return {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "strict_fail_on_degenerate_fold": config.strict_fail_on_degenerate_fold,
    }


def run_cross_validation(
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    idx_train_val: np.ndarray,
    config: Config,
    out_dirs: Dict[str, Path],
    run_state: Dict[str, Any],
    config_signature: str,
) -> Dict[str, Any]:
    """Run full cross-validation for the configured epoch count."""
    device = torch.device(config.device)
    runtime = _resolve_mode_config(config)
    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    criterion = nn.BCELoss()

    fold_metrics: List[Dict[str, Any]] = []
    fold_histories: List[Dict[str, Any]] = []
    fold_predictions: List[Dict[str, Any]] = []
    roc_data: List[Dict[str, Any]] = []

    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_auc: float = -np.inf
    best_fold: Optional[int] = None

    for fold_index, (train_idx, val_idx) in enumerate(kf.split(X_train_val), start=1):
        _log("\n==== Fold {}/{} ====".format(fold_index, config.n_splits))
        fold_key = str(fold_index)
        fold_resume_path = out_dirs["resume"] / "fold{}.pth".format(fold_index)
        saved_fold = run_state.get("folds", {}).get(fold_key)
        resume_payload: Optional[Dict[str, Any]] = None
        if fold_resume_path.exists():
            try:
                resume_payload = safe_torch_load(fold_resume_path)
                assert_checkpoint_signature(resume_payload, config_signature, "fold {}".format(fold_index))
            except Exception as exc:
                _log(
                    "[RESUME] Checkpoint fold {} rusak/tidak valid: {}. Fold akan dihitung ulang.".format(
                        fold_index, exc
                    )
                )
                resume_payload = None
        elif saved_fold and saved_fold.get("completed"):
            _log(
                "[RESUME] Fold {} ditandai completed di run_state tetapi checkpoint fold hilang. "
                "Fold akan dihitung ulang.".format(fold_index)
            )

        if resume_payload is not None and resume_payload.get("completed"):
            saved_best_state = resume_payload.get("best_state_dict")
            saved_fold_metric = resume_payload.get("fold_metric")
            saved_history = resume_payload.get("history")
            saved_prediction = resume_payload.get("fold_prediction")
            saved_best_val_metrics = resume_payload.get("best_val_metrics")
            if (
                saved_best_state is None
                or saved_fold_metric is None
                or saved_history is None
                or saved_prediction is None
                or saved_best_val_metrics is None
            ):
                _log(
                    "[RESUME] Checkpoint fold {} completed tetapi tidak lengkap. Fold akan dihitung ulang.".format(
                        fold_index
                    )
                )
            else:
                _log("[RESUME] Skip fold {}".format(fold_index))
                run_state.setdefault("folds", {})
                run_state["folds"][fold_key] = {
                    "completed": True,
                    "resume_path": str(fold_resume_path),
                    "best_epoch": int(resume_payload.get("best_epoch", saved_fold_metric.get("best_epoch", 1))),
                    "best_val_metrics": _to_serializable(saved_best_val_metrics),
                    "fold_metric": _to_serializable(saved_fold_metric),
                    "history": _to_serializable(saved_history),
                    "fold_prediction": _to_serializable(saved_prediction),
                }
                save_run_state(out_dirs["state"], run_state)
                fold_metrics.append(saved_fold_metric)
                fold_histories.append({"fold": fold_index, "history": saved_history})
                fold_predictions.append(saved_prediction)
                y_true_saved = np.asarray(saved_prediction.get("y_true", []), dtype=np.float32)
                y_prob_saved = np.asarray(saved_prediction.get("y_prob", []), dtype=np.float32)
                if y_true_saved.size and y_prob_saved.size:
                    recomputed_metrics = _compute_binary_metrics(y_true_saved, y_prob_saved, config.threshold)
                    for key in ("pr_auc", "tp", "tn", "fp", "fn"):
                        saved_best_val_metrics.setdefault(key, recomputed_metrics.get(key))
                    saved_fold_metric.setdefault("val_pr_auc", recomputed_metrics.get("pr_auc"))
                    saved_fold_metric.setdefault("tp", recomputed_metrics.get("tp", 0))
                    saved_fold_metric.setdefault("tn", recomputed_metrics.get("tn", 0))
                    saved_fold_metric.setdefault("fp", recomputed_metrics.get("fp", 0))
                    saved_fold_metric.setdefault("fn", recomputed_metrics.get("fn", 0))
                if y_true_saved.size and y_prob_saved.size and len(np.unique(y_true_saved)) > 1:
                    fpr_saved, tpr_saved, _ = roc_curve(y_true_saved, y_prob_saved)
                    roc_data.append(
                        {
                            "fold": fold_index,
                            "fpr": fpr_saved,
                            "tpr": tpr_saved,
                            "roc_auc": float(roc_auc_score(y_true_saved, y_prob_saved)),
                        }
                    )
                saved_auc = saved_best_val_metrics.get("roc_auc")
                if saved_auc is not None and float(saved_auc) > best_auc:
                    best_auc = float(saved_auc)
                    best_fold = fold_index
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in saved_best_state.items()
                    }
                continue
        run_state.setdefault("folds", {})
        run_state["folds"].setdefault(
            fold_key,
            {
                "completed": False,
                "resume_path": str(fold_resume_path),
            },
        )
        save_run_state(out_dirs["state"], run_state)
        fold_start_time = time.perf_counter()

        X_train = X_train_val[train_idx]
        X_val = X_train_val[val_idx]
        y_train = y_train_val[train_idx]
        y_val = y_train_val[val_idx]
        val_sample_indices = np.asarray(idx_train_val[val_idx], dtype=np.int64)

        if runtime["strict_fail_on_degenerate_fold"] and len(np.unique(y_val)) < 2:
            raise ValueError("Degenerate validation fold detected in clean_mode: fold {}".format(fold_index))

        X_train_tensor, y_train_tensor = tensorize_dataset(X_train, y_train)
        X_val_tensor, y_val_tensor = tensorize_dataset(X_val, y_val)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=runtime["batch_size"],
            shuffle=True,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=runtime["batch_size"],
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = CAMPModel(
            input_dim=X_train_val.shape[1],
            dropout_rate=config.dropout_rate,
            conv1_channels=config.conv1_channels,
            conv2_channels=config.conv2_channels,
            dense_hidden=config.dense_hidden,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=runtime["learning_rate"], weight_decay=config.weight_decay)

        best_fold_state = deepcopy(model.state_dict())
        best_fold_val_loss = np.inf
        best_fold_monitor_score: Optional[float] = None
        best_fold_metrics: Optional[Dict[str, Any]] = None
        history_rows: List[Dict[str, Any]] = []
        best_fold_epoch = 1
        epochs_without_improvement = 0
        early_stop_triggered = False
        early_stop_reason = ""
        start_epoch = 1

        if resume_payload is not None:
            try:
                if not resume_payload.get("completed"):
                    model.load_state_dict(resume_payload["model_state_dict"])
                    optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                    history_rows = resume_payload.get("history", history_rows)
                    best_fold_state = resume_payload.get("best_state_dict", best_fold_state)
                    best_fold_metrics = resume_payload.get("best_val_metrics")
                    best_fold_epoch = int(resume_payload.get("best_epoch", 1))
                    best_fold_val_loss = float(resume_payload.get("best_val_loss", best_fold_val_loss))
                    saved_monitor_score = resume_payload.get("best_monitor_score")
                    best_fold_monitor_score = float(saved_monitor_score) if saved_monitor_score is not None else None
                    epochs_without_improvement = int(resume_payload.get("epochs_without_improvement", 0))
                    set_rng_state(resume_payload.get("rng_state"))
                    start_epoch = int(resume_payload.get("epoch", 0)) + 1
                    if start_epoch <= runtime["epochs"]:
                        _log("[RESUME] Resume fold {} dari epoch {}.".format(fold_index, start_epoch))
                    else:
                        _log(
                            "[RESUME] Checkpoint incomplete fold {} melewati total epoch. "
                            "Fold akan dihitung ulang dari awal.".format(fold_index)
                        )
                        history_rows = []
                        best_fold_state = deepcopy(model.state_dict())
                        best_fold_val_loss = np.inf
                        best_fold_monitor_score = None
                        best_fold_metrics = None
                        best_fold_epoch = 1
                        epochs_without_improvement = 0
                        early_stop_triggered = False
                        early_stop_reason = ""
                        start_epoch = 1
            except Exception as exc:
                _log(
                    "[RESUME] Checkpoint incomplete fold {} rusak/tidak valid: {}. "
                    "Fold akan dihitung ulang dari awal.".format(fold_index, exc)
                )
                history_rows = []
                best_fold_state = deepcopy(model.state_dict())
                best_fold_val_loss = np.inf
                best_fold_monitor_score = None
                best_fold_metrics = None
                best_fold_epoch = 1
                epochs_without_improvement = 0
                early_stop_triggered = False
                early_stop_reason = ""
                start_epoch = 1

        for epoch in range(start_epoch, runtime["epochs"] + 1):
            train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, config.threshold)
            val_stats = evaluate_one_epoch(model, val_loader, criterion, device, config.threshold)

            history_row = {
                "fold": fold_index,
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "train_precision": train_stats["precision"],
                "train_recall": train_stats["recall"],
                "train_f1": train_stats["f1"],
                "train_roc_auc": train_stats["roc_auc"],
                "train_pr_auc": train_stats["pr_auc"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "val_precision": val_stats["precision"],
                "val_recall": val_stats["recall"],
                "val_f1": val_stats["f1"],
                "val_roc_auc": val_stats["roc_auc"],
                "val_pr_auc": val_stats["pr_auc"],
            }
            history_rows.append(history_row)

            _log(
                "Fold {fold} | Epoch {epoch}/{epochs} | "
                "Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
                "Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Val AUC {val_auc}".format(
                    fold=fold_index,
                    epoch=epoch,
                    epochs=runtime["epochs"],
                    train_loss=train_stats["loss"],
                    train_acc=train_stats["accuracy"],
                    val_loss=val_stats["loss"],
                    val_acc=val_stats["accuracy"],
                    val_auc="{:.4f}".format(val_stats["roc_auc"]) if val_stats["roc_auc"] is not None else "NA",
                )
            )

            monitor_score = _early_stopping_score(val_stats, config.early_stopping_monitor)
            if _is_early_stopping_improvement(
                monitor_score,
                best_fold_monitor_score,
                mode=config.early_stopping_mode,
                min_delta=config.early_stopping_min_delta,
            ):
                best_fold_monitor_score = monitor_score
                best_fold_val_loss = val_stats["loss"]
                best_fold_state = deepcopy(model.state_dict())
                best_fold_metrics = deepcopy(val_stats)
                best_fold_epoch = int(epoch)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            safe_torch_save(
                fold_resume_path,
                {
                    "completed": False,
                    "fold": int(fold_index),
                    "epoch": int(epoch),
                    "model_state_dict": deepcopy(model.state_dict()),
                    "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                    "history": deepcopy(history_rows),
                    "best_state_dict": deepcopy(best_fold_state),
                    "best_val_metrics": deepcopy(best_fold_metrics),
                    "best_epoch": int(best_fold_epoch),
                    "best_val_loss": float(best_fold_val_loss),
                    "best_monitor": config.early_stopping_monitor,
                    "best_monitor_score": best_fold_monitor_score,
                    "epochs_without_improvement": int(epochs_without_improvement),
                    "config_signature": config_signature,
                    "rng_state": get_rng_state(),
                },
            )

            if (
                config.early_stopping
                and config.early_stopping_patience > 0
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                early_stop_triggered = True
                early_stop_reason = (
                    "monitor={} mode={} patience={} min_delta={} best_epoch={} best_score={}"
                ).format(
                    config.early_stopping_monitor,
                    config.early_stopping_mode,
                    config.early_stopping_patience,
                    config.early_stopping_min_delta,
                    best_fold_epoch,
                    "{:.6f}".format(best_fold_monitor_score) if best_fold_monitor_score is not None else "NA",
                )
                _log("[EARLY STOP] Fold {} berhenti di epoch {} | {}".format(fold_index, epoch, early_stop_reason))
                break

        if best_fold_metrics is None:
            raise RuntimeError("Fold {} did not produce validation metrics.".format(fold_index))

        fold_summary = {
            "fold": fold_index,
            "best_epoch": int(best_fold_epoch),
            "elapsed_seconds": float(time.perf_counter() - fold_start_time),
            "stopped_epoch": int(history_rows[-1]["epoch"]) if history_rows else int(best_fold_epoch),
            "early_stop_triggered": bool(early_stop_triggered),
            "early_stop_reason": early_stop_reason,
            "monitor": config.early_stopping_monitor,
            "best_monitor_score": best_fold_monitor_score,
            "val_loss": best_fold_val_loss,
            "val_accuracy": best_fold_metrics["accuracy"],
            "val_precision": best_fold_metrics["precision"],
            "val_recall": best_fold_metrics["recall"],
            "val_f1": best_fold_metrics["f1"],
            "val_roc_auc": best_fold_metrics["roc_auc"],
            "val_pr_auc": best_fold_metrics["pr_auc"],
            "tp": best_fold_metrics["tp"],
            "tn": best_fold_metrics["tn"],
            "fp": best_fold_metrics["fp"],
            "fn": best_fold_metrics["fn"],
        }
        fold_metrics.append(fold_summary)
        fold_histories.append({"fold": fold_index, "history": history_rows})
        fold_predictions.append(
            {
                "fold": fold_index,
                "sample_indices": val_sample_indices,
                "y_true": best_fold_metrics["labels"],
                "y_prob": best_fold_metrics["probs"],
            }
        )
        safe_torch_save(
            fold_resume_path,
            {
                "completed": True,
                "fold": int(fold_index),
                "epoch": int(history_rows[-1]["epoch"]) if history_rows else int(best_fold_epoch),
                "model_state_dict": deepcopy(model.state_dict()),
                "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                "history": deepcopy(history_rows),
                "best_state_dict": deepcopy(best_fold_state),
                "best_val_metrics": deepcopy(best_fold_metrics),
                "best_epoch": int(best_fold_epoch),
                "best_val_loss": float(best_fold_val_loss),
                "best_monitor": config.early_stopping_monitor,
                "best_monitor_score": best_fold_monitor_score,
                "early_stop_triggered": bool(early_stop_triggered),
                "early_stop_reason": early_stop_reason,
                "fold_metric": deepcopy(fold_summary),
                "fold_prediction": deepcopy(fold_predictions[-1]),
                "config_signature": config_signature,
                "rng_state": get_rng_state(),
            },
        )
        run_state["folds"][fold_key]["completed"] = True
        run_state["folds"][fold_key]["resume_path"] = str(fold_resume_path)
        run_state["folds"][fold_key]["best_epoch"] = int(fold_summary["best_epoch"])
        run_state["folds"][fold_key]["best_val_metrics"] = _to_serializable(best_fold_metrics)
        run_state["folds"][fold_key]["fold_metric"] = _to_serializable(fold_summary)
        run_state["folds"][fold_key]["history"] = _to_serializable(history_rows)
        run_state["folds"][fold_key]["fold_prediction"] = _to_serializable(fold_predictions[-1])
        save_run_state(out_dirs["state"], run_state)

        if best_fold_metrics["fpr"] is not None and best_fold_metrics["tpr"] is not None:
            roc_data.append(
                {
                    "fold": fold_index,
                    "fpr": best_fold_metrics["fpr"],
                    "tpr": best_fold_metrics["tpr"],
                    "roc_auc": best_fold_metrics["roc_auc"],
                }
            )

        auc_value = best_fold_metrics["roc_auc"]
        if auc_value is not None and auc_value > best_auc:
            best_auc = auc_value
            best_fold = fold_index
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in best_fold_state.items()
            }

    if best_state_dict is None:
        raise RuntimeError("No valid best model was selected during cross-validation.")

    return {
        "best_state_dict": best_state_dict,
        "best_auc": best_auc if best_auc != -np.inf else None,
        "best_fold": best_fold,
        "fold_metrics": fold_metrics,
        "fold_histories": fold_histories,
        "fold_predictions": fold_predictions,
        "roc_data": roc_data,
    }


def evaluate_test_set(
    best_state_dict: Optional[Dict[str, torch.Tensor]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    idx_test: np.ndarray,
    config: Config,
) -> Dict[str, Any]:
    """Evaluate the best checkpoint on the hold-out test set."""
    if best_state_dict is None:
        raise ValueError("best_state_dict is None. Cross-validation did not produce a checkpoint.")

    device = torch.device(config.device)
    X_test_tensor, y_test_tensor = tensorize_dataset(X_test, y_test)
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = CAMPModel(
        input_dim=X_test.shape[1],
        dropout_rate=config.dropout_rate,
        conv1_channels=config.conv1_channels,
        conv2_channels=config.conv2_channels,
        dense_hidden=config.dense_hidden,
    ).to(device)
    model.load_state_dict(best_state_dict)
    criterion = nn.BCELoss()
    result = evaluate_one_epoch(model, test_loader, criterion, device, config.threshold)
    result["sample_indices"] = np.asarray(idx_test, dtype=np.int64)
    return result


def _plot_cv_roc(roc_data: List[Dict[str, Any]], output_path: Path) -> None:
    if not roc_data:
        return
    plt.figure(figsize=(8, 6))
    for item in roc_data:
        plt.plot(item["fpr"], item["tpr"], label="Fold {} (AUC = {:.4f})".format(item["fold"], item["roc_auc"]))
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve per Fold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_test_roc(test_results: Dict[str, Any], output_path: Path) -> None:
    if test_results.get("fpr") is None or test_results.get("tpr") is None or test_results.get("roc_auc") is None:
        return
    plt.figure(figsize=(8, 6))
    plt.plot(test_results["fpr"], test_results["tpr"], label="Test AUC = {:.4f}".format(test_results["roc_auc"]))
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_learning_curves(fold_histories: List[Dict[str, Any]], output_path: Path) -> None:
    if not fold_histories:
        return
    plt.figure(figsize=(10, 6))
    for item in fold_histories:
        history_df = pd.DataFrame(item["history"])
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Fold {} Val Loss".format(item["fold"]))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss per Fold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_artifacts(
    out_dirs: Dict[str, Path],
    cv_results: Dict[str, Any],
    test_results: Dict[str, Any],
    config: Config,
) -> None:
    """Persist model artifacts and summary tables."""
    reports_dir = out_dirs["reports"]
    plots_dir = out_dirs["plots"]
    predictions_dir = out_dirs["predictions"]
    checkpoints_dir = out_dirs["checkpoints"]
    fold_reports_dir = out_dirs["fold_reports"]

    fold_metrics = cv_results.get("fold_metrics", [])
    if fold_metrics:
        pd.DataFrame(fold_metrics).to_csv(reports_dir / "kfold_metrics.csv", index=False)
        pd.DataFrame(fold_metrics).to_csv(fold_reports_dir / "kfold_metrics.csv", index=False)
        cv_summary_rows = _build_cv_mean_std_summary(fold_metrics)
        pd.DataFrame(cv_summary_rows).to_csv(reports_dir / "cv_summary_mean_std.csv", index=False)
    else:
        cv_summary_rows = []

    history_rows: List[Dict[str, Any]] = []
    for item in cv_results.get("fold_histories", []):
        history_rows.extend(item["history"])
    if history_rows:
        pd.DataFrame(history_rows).to_csv(reports_dir / "fold_history.csv", index=False)

    summary_row = {
        "mode": config.mode,
        "best_fold": cv_results.get("best_fold"),
        "best_auc": cv_results.get("best_auc"),
        "test_loss": test_results.get("loss"),
        "test_accuracy": test_results.get("accuracy"),
        "test_precision": test_results.get("precision"),
        "test_recall": test_results.get("recall"),
        "test_f1": test_results.get("f1"),
        "test_roc_auc": test_results.get("roc_auc"),
        "test_pr_auc": test_results.get("pr_auc"),
        "test_tp": test_results.get("tp"),
        "test_tn": test_results.get("tn"),
        "test_fp": test_results.get("fp"),
        "test_fn": test_results.get("fn"),
    }
    pd.DataFrame([summary_row]).to_csv(reports_dir / "test_metrics.csv", index=False)
    _save_json(
        reports_dir / "final_test_metrics.json",
        {
            "loss": test_results.get("loss"),
            "accuracy": test_results.get("accuracy"),
            "precision": test_results.get("precision"),
            "recall": test_results.get("recall"),
            "f1": test_results.get("f1"),
            "roc_auc": test_results.get("roc_auc"),
            "pr_auc": test_results.get("pr_auc"),
            "tp": test_results.get("tp"),
            "tn": test_results.get("tn"),
            "fp": test_results.get("fp"),
            "fn": test_results.get("fn"),
        },
    )

    best_state_dict = cv_results.get("best_state_dict")
    if best_state_dict is not None:
        torch.save(best_state_dict, checkpoints_dir / "final_model.pth")

    if config.save_fold_predictions:
        fold_prediction_payloads: List[Dict[str, Any]] = []
        for item in cv_results.get("fold_predictions", []):
            fold = item["fold"]
            sample_indices = np.asarray(item["sample_indices"], dtype=np.int64)
            y_true = np.asarray(item["y_true"])
            y_prob = np.asarray(item["y_prob"])
            y_pred = (y_prob >= float(config.threshold)).astype(np.int64)
            np.save(predictions_dir / "y_true_fold{}.npy".format(fold), y_true)
            np.save(predictions_dir / "y_prob_fold{}.npy".format(fold), y_prob)
            np.savez_compressed(
                predictions_dir / "fold_{}_predictions.npz".format(fold),
                sample_indices=sample_indices,
                y_true=y_true,
                y_prob=y_prob,
                y_pred=y_pred,
            )
            pd.DataFrame({"sample_indices": sample_indices, "y_true": y_true.astype(np.int64), "y_prob": y_prob, "y_pred": y_pred}).to_csv(
                predictions_dir / "fold_{}_predictions.csv".format(fold),
                index=False,
            )
            fold_prediction_payloads.append(
                {
                    "fold": int(fold),
                    "sample_indices": sample_indices,
                "y_true": y_true,
                "y_prob": y_prob,
                "y_pred": y_pred,
            }
            )
        if fold_prediction_payloads:
            np.savez_compressed(predictions_dir / "kfold_predictions.npz", folds=np.asarray(fold_prediction_payloads, dtype=object))

    np.save(predictions_dir / "test_y_true.npy", test_results["labels"])
    np.save(predictions_dir / "test_y_prob.npy", test_results["probs"])
    np.savez_compressed(
        predictions_dir / "final_test_predictions.npz",
        sample_indices=np.asarray(test_results["sample_indices"], dtype=np.int64),
        y_true=np.asarray(test_results["labels"]),
        y_prob=np.asarray(test_results["probs"]),
        y_pred=(np.asarray(test_results["probs"]) >= float(config.threshold)).astype(np.int64),
    )

    if config.save_plots:
        _plot_cv_roc(cv_results.get("roc_data", []), plots_dir / "roc_auc_kfold_all.png")
        _plot_test_roc(test_results, plots_dir / "final_test_roc_curve.png")
        _plot_learning_curves(cv_results.get("fold_histories", []), plots_dir / "avg_loss_acc_kfold.png")

    _save_json(reports_dir / "config.json", asdict(config))
    _save_json(
        reports_dir / "run_summary.json",
        {
            "run_name": out_dirs["run_name"].name,
            "best_fold": cv_results.get("best_fold"),
            "best_auc": cv_results.get("best_auc"),
            "num_folds": len(cv_results.get("fold_metrics", [])),
            "cv_mean_std": cv_summary_rows,
            "test_metrics": {
                "loss": test_results.get("loss"),
                "accuracy": test_results.get("accuracy"),
                "precision": test_results.get("precision"),
                "recall": test_results.get("recall"),
                "f1": test_results.get("f1"),
                "roc_auc": test_results.get("roc_auc"),
                "pr_auc": test_results.get("pr_auc"),
                "tp": test_results.get("tp"),
                "tn": test_results.get("tn"),
                "fp": test_results.get("fp"),
                "fn": test_results.get("fn"),
            },
        },
    )


def build_formatted_report(
    *,
    out_dirs: Dict[str, Path],
    cv_results: Dict[str, Any],
    test_results: Dict[str, Any],
    config: Config,
    start_time: float,
) -> str:
    runtime = _resolve_mode_config(config)
    fold_metrics = cv_results.get("fold_metrics", [])
    fold_predictions_by_fold = {
        int(item["fold"]): item
        for item in cv_results.get("fold_predictions", [])
        if "fold" in item
    }

    detail_lines: List[str] = []
    model_param_rows: List[Dict[str, Any]] = []
    for row in fold_metrics:
        fold_no = int(row["fold"])
        prediction = fold_predictions_by_fold.get(fold_no, {})
        y_true = np.asarray(prediction.get("y_true", []), dtype=np.float32)
        y_prob = np.asarray(prediction.get("y_prob", []), dtype=np.float32)
        metrics = _compute_binary_metrics(y_true, y_prob, config.threshold) if y_true.size and y_prob.size else {
            "accuracy": row.get("val_accuracy"),
            "precision": row.get("val_precision"),
            "recall": row.get("val_recall"),
            "f1": row.get("val_f1"),
            "roc_auc": row.get("val_roc_auc"),
            "pr_auc": row.get("val_pr_auc"),
            "tp": row.get("tp", 0),
            "tn": row.get("tn", 0),
            "fp": row.get("fp", 0),
            "fn": row.get("fn", 0),
        }
        param_string = _candidate_param_string(config, runtime, best_epoch=int(row.get("best_epoch", 0) or 0))
        detail_lines.append("Fold {}/{}".format(fold_no, config.n_splits))
        detail_lines.extend(
            render_candidate_detail(
                index=1,
                param_string=param_string,
                metrics=_metric_report_row(metrics),
                confusion=_confusion_report_row(metrics),
                elapsed_seconds=float(row["elapsed_seconds"]) if row.get("elapsed_seconds") is not None else None,
            )
        )
        detail_lines.append("")
        detail_lines.extend(
            render_selected_detail(
                param_string=param_string,
                selection_metric_name="ROC-AUC",
                selection_metric_value=float(metrics["roc_auc"]) if metrics.get("roc_auc") is not None else float("nan"),
                metrics=_metric_report_row(metrics),
                confusion=_confusion_report_row(metrics),
            )
        )
        detail_lines.append("")
        model_param_rows.append(
            {
                "fold": fold_no,
                "model": "CAMP",
                "mode": config.mode,
                "epochs": runtime["epochs"],
                "batch_size": runtime["batch_size"],
                "learning_rate": runtime["learning_rate"],
                "dropout_rate": config.dropout_rate,
                "weight_decay": config.weight_decay,
                "conv1_channels": config.conv1_channels,
                "conv2_channels": config.conv2_channels,
                "dense_hidden": config.dense_hidden,
                "best_epoch": row.get("best_epoch"),
            }
        )

    report_fold_metrics_df = pd.DataFrame([_fold_metric_report_row(row) for row in fold_metrics])
    metric_columns = ["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1"]
    aggregated_df = report_fold_metrics_df[metric_columns].agg(["mean", "std"]) if not report_fold_metrics_df.empty else pd.DataFrame()
    params_df = pd.DataFrame(model_param_rows)

    final_test_lines = [
        "Model final: fold {}".format(cv_results.get("best_fold")),
        "-> {}".format(format_metrics(_metric_report_row(test_results))),
        format_confusion_matrix(_confusion_report_row(test_results)),
    ]

    sections = [
        ("Detail single-grid best-param & evaluasi", "\n".join(detail_lines).rstrip()),
        ("Per-fold metrics", dataframe_to_string(report_fold_metrics_df, index=False)),
        ("Aggregated metrics", dataframe_to_string(aggregated_df, index=True)),
        ("Model params per fold", dataframe_to_string(params_df, index=False)),
        ("Voting Best Model Param", "\n".join(voting_lines(model_param_rows, exclude=("fold", "best_epoch")))),
        ("Evaluasi final test", "\n".join(final_test_lines)),
    ]
    report_text = render_sections(sections, build_execution_report(start_time))
    (out_dirs["reports"] / "formatted_report.txt").write_text(report_text, encoding="utf-8")
    return report_text


def _prepare_run_state(config: Config, out_dirs: Dict[str, Path]) -> Tuple[Dict[str, Any], str]:
    runtime = _resolve_mode_config(config)
    resume_config_payload = {
        "config": _to_serializable(asdict(config)),
        "runtime": _to_serializable(runtime),
    }
    config_signature = compute_config_signature(resume_config_payload)
    run_state_path = out_dirs["state"]
    run_state = load_run_state(run_state_path)
    saved_signature = str(run_state.get("resume_meta", {}).get("config_signature", "")).strip()
    if saved_signature and saved_signature != config_signature:
        if has_resume_progress(run_state):
            raise RuntimeError(
                "Resume config mismatch untuk run {}. saved_signature={}, current_signature={}.".format(
                    config.run_name, saved_signature, config_signature
                )
            )
        _log(
            "[RESUME] Mengabaikan run_state lama karena config berubah dan belum ada checkpoint/progress final."
        )
        run_state = {"resume_meta": {}, "folds": {}, "final": {}}
    run_state["resume_meta"] = {
        "config_signature": config_signature,
        "config_payload": resume_config_payload,
    }
    save_run_state(run_state_path, run_state)
    return run_state, config_signature


def run_grid_search(
    *,
    config: Config,
    base_out_dirs: Dict[str, Path],
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    idx_train_val: np.ndarray,
) -> Tuple[Config, List[Dict[str, Any]]]:
    grid = get_hyperparameter_grid()
    base_run_name = config.run_name.strip() if config.run_name.strip() else "model1_sing_reg"
    grid_rows: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params: Optional[Dict[str, Any]] = None

    _log("Running grid search: {} kombinasi.".format(len(grid)))
    for index, params in enumerate(grid, start=1):
        trial_name = _grid_trial_name(index, params)
        trial_config = deepcopy(config)
        trial_config.run_name = "{}_{}".format(base_run_name, trial_name)
        trial_config.output_dir = base_out_dirs["log_root"] / "grid_search" / trial_name
        trial_config.batch_size = int(params["batch_size"])
        trial_config.learning_rate = float(params["learning_rate"])
        trial_config.dropout_rate = float(params["dropout_rate"])

        trial_out_dirs = _make_output_dirs(trial_config)
        trial_config.run_name = trial_out_dirs["run_name"].name
        trial_run_state, trial_signature = _prepare_run_state(trial_config, trial_out_dirs)

        _log(
            "\n[GRID {}/{}] batch_size={} | learning_rate={} | dropout_rate={}".format(
                index,
                len(grid),
                trial_config.batch_size,
                trial_config.learning_rate,
                trial_config.dropout_rate,
            )
        )
        cv_results = run_cross_validation(
            X_train_val=X_train_val,
            y_train_val=y_train_val,
            idx_train_val=idx_train_val,
            config=trial_config,
            out_dirs=trial_out_dirs,
            run_state=trial_run_state,
            config_signature=trial_signature,
        )

        metrics_df = pd.DataFrame(cv_results.get("fold_metrics", []))
        roc_values = (
            pd.to_numeric(metrics_df["val_roc_auc"], errors="coerce").dropna()
            if "val_roc_auc" in metrics_df
            else pd.Series(dtype=float)
        )
        pr_values = (
            pd.to_numeric(metrics_df["val_pr_auc"], errors="coerce").dropna()
            if "val_pr_auc" in metrics_df
            else pd.Series(dtype=float)
        )
        f1_values = (
            pd.to_numeric(metrics_df["val_f1"], errors="coerce").dropna()
            if "val_f1" in metrics_df
            else pd.Series(dtype=float)
        )
        selection_score = float(roc_values.mean()) if not roc_values.empty else -np.inf

        row = {
            "rank": None,
            "grid_index": index,
            "trial_name": trial_name,
            "run_name": trial_config.run_name,
            "batch_size": trial_config.batch_size,
            "learning_rate": trial_config.learning_rate,
            "dropout_rate": trial_config.dropout_rate,
            "mean_val_roc_auc": float(roc_values.mean()) if not roc_values.empty else None,
            "std_val_roc_auc": float(roc_values.std(ddof=0)) if not roc_values.empty else None,
            "mean_val_pr_auc": float(pr_values.mean()) if not pr_values.empty else None,
            "std_val_pr_auc": float(pr_values.std(ddof=0)) if not pr_values.empty else None,
            "mean_val_f1": float(f1_values.mean()) if not f1_values.empty else None,
            "std_val_f1": float(f1_values.std(ddof=0)) if not f1_values.empty else None,
            "best_fold": cv_results.get("best_fold"),
            "best_fold_auc": cv_results.get("best_auc"),
        }
        grid_rows.append(row)
        _log(
            "[GRID {}] mean ROC-AUC={} | mean PR-AUC={} | mean F1={}".format(
                index,
                "{:.4f}".format(row["mean_val_roc_auc"]) if row["mean_val_roc_auc"] is not None else "NA",
                "{:.4f}".format(row["mean_val_pr_auc"]) if row["mean_val_pr_auc"] is not None else "NA",
                "{:.4f}".format(row["mean_val_f1"]) if row["mean_val_f1"] is not None else "NA",
            )
        )

        if selection_score > best_score:
            best_score = selection_score
            best_params = params

    if best_params is None:
        raise RuntimeError("Grid search tidak menghasilkan kandidat valid.")

    ranked_rows = sorted(
        grid_rows,
        key=lambda row: float(row["mean_val_roc_auc"]) if row["mean_val_roc_auc"] is not None else -np.inf,
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    pd.DataFrame(ranked_rows).to_csv(base_out_dirs["reports"] / "grid_search_results.csv", index=False)
    _save_json(
        base_out_dirs["reports"] / "grid_search_best_params.json",
        {
            "selection_metric": "mean_val_roc_auc",
            "best_score": best_score if best_score != -np.inf else None,
            "best_params": best_params,
            "num_combinations": len(grid),
        },
    )

    tuned_config = deepcopy(config)
    tuned_config.batch_size = int(best_params["batch_size"])
    tuned_config.learning_rate = float(best_params["learning_rate"])
    tuned_config.dropout_rate = float(best_params["dropout_rate"])
    _log(
        "Best grid params: batch_size={} | learning_rate={} | dropout_rate={} | mean ROC-AUC={}".format(
            tuned_config.batch_size,
            tuned_config.learning_rate,
            tuned_config.dropout_rate,
            "{:.4f}".format(best_score) if best_score != -np.inf else "NA",
        )
    )
    return tuned_config, ranked_rows


def main() -> None:
    global _ACTIVE_RESULTS_LOG
    start_time = time.perf_counter()
    config = parse_args()
    out_dirs = _make_output_dirs(config)
    config.run_name = out_dirs["run_name"].name
    _ACTIVE_RESULTS_LOG = out_dirs["text"]

    _log("Run name: {}".format(config.run_name))
    _log("Log root: {}".format(out_dirs["log_root"]))
    _log("Fold root: {}".format(out_dirs["fold_root"]))
    _log("Model root: {}".format(out_dirs["model_root"]))
    _log("Loading dataset from: {}".format(config.csv_path))
    df = load_dataset(config.csv_path)

    _log("Validating schema...")
    schema_info = validate_schema(df, target_col=config.target_col)
    _log("Schema info: {}".format(schema_info))

    _log("Collecting feature block columns...")
    block_columns = get_feature_block_columns(df)
    for block_name in CONCAT_ORDER:
        _log("Block {:>10s}: {}".format(block_name, len(block_columns[block_name])))

    _log("Building feature matrix...")
    X, y, metadata = build_feature_matrix(df, block_columns, target_col=config.target_col)
    _log("Feature matrix shape: {}".format(X.shape))
    _log("Target shape: {}".format(y.shape))
    _log("Metadata shape: {}".format(metadata.shape))
    _save_json(out_dirs["reports"] / "dataset_summary.json", _summarize_dataset(df, y))

    _log("Splitting dataset...")
    split_data = split_dataset(X, y, metadata, config)
    _log("Train/Val shape: {} | Test shape: {}".format(split_data["X_train_val"].shape, split_data["X_test"].shape))
    np.savez_compressed(
        out_dirs["fold_root"] / "split_indices.npz",
        idx_train_val=np.asarray(split_data["idx_train_val"]),
        idx_test=np.asarray(split_data["idx_test"]),
    )
    runtime = _resolve_mode_config(config)
    if config.tune_grid:
        _log("Single-grid mode: --tune-grid diabaikan; memakai best param artefak 22-05-2026.")
        config.tune_grid = False
    if config.tune_grid:
        final_grid_run_name = "{}_grid_best".format(config.run_name)
        config, _ = run_grid_search(
            config=config,
            base_out_dirs=out_dirs,
            X_train_val=split_data["X_train_val"],
            y_train_val=split_data["y_train_val"],
            idx_train_val=np.asarray(split_data["idx_train_val"], dtype=np.int64),
        )
        config.run_name = final_grid_run_name
        out_dirs = _make_output_dirs(config)
        config.run_name = out_dirs["run_name"].name
        runtime = _resolve_mode_config(config)

    run_state, config_signature = _prepare_run_state(config, out_dirs)

    _log("Running cross-validation in mode: {}".format(config.mode))
    cv_results = run_cross_validation(
        X_train_val=split_data["X_train_val"],
        y_train_val=split_data["y_train_val"],
        idx_train_val=np.asarray(split_data["idx_train_val"], dtype=np.int64),
        config=config,
        out_dirs=out_dirs,
        run_state=run_state,
        config_signature=config_signature,
    )
    _log("Best fold: {} | Best CV AUC: {}".format(cv_results["best_fold"], cv_results["best_auc"]))
    cv_summary_rows = _build_cv_mean_std_summary(cv_results["fold_metrics"])
    if cv_summary_rows:
        _log("CV mean ± std summary:")
        for row in cv_summary_rows:
            _log("  {}: {}".format(row["metric"], row["mean_pm_std"]))

    _log("Evaluating test set...")
    test_results = evaluate_test_set(
        best_state_dict=cv_results["best_state_dict"],
        X_test=split_data["X_test"],
        y_test=split_data["y_test"],
        idx_test=np.asarray(split_data["idx_test"], dtype=np.int64),
        config=config,
    )
    _log(
        "Test metrics | Loss {loss:.4f} | Acc {acc:.4f} | Prec {prec:.4f} | "
        "Recall {rec:.4f} | F1 {f1:.4f} | AUC {auc} | PR-AUC {pr_auc}".format(
            loss=test_results["loss"],
            acc=test_results["accuracy"],
            prec=test_results["precision"],
            rec=test_results["recall"],
            f1=test_results["f1"],
            auc="{:.4f}".format(test_results["roc_auc"]) if test_results["roc_auc"] is not None else "NA",
            pr_auc="{:.4f}".format(test_results["pr_auc"]) if test_results["pr_auc"] is not None else "NA",
        )
    )

    _log("Saving artifacts to: {}".format(out_dirs["log_root"]))
    save_artifacts(out_dirs, cv_results, test_results, config)
    _log(
        build_formatted_report(
            out_dirs=out_dirs,
            cv_results=cv_results,
            test_results=test_results,
            config=config,
            start_time=start_time,
        )
    )

    _log("Done.")


if __name__ == "__main__":
    main()
