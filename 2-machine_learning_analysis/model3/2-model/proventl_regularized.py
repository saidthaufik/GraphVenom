import json
import logging
import os
import random
import re
import shutil
import sys
import time
import traceback
import hashlib
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from path import model_path, output_roots, repo_root
from reporting import (
    build_execution_report,
    dataframe_to_string,
    render_candidate_detail,
    render_sections,
    render_selected_detail,
    voting_lines,
)


NOTEBOOK_REFERENCE_PATH = str(PROJECT_ROOT / "0-arsip" / "ProVenTL.ipynb")
DATA_PATH = str(model_path("model3", "1-feat_prott5", "prott5pn.csv"))
DATASET_NAME = "rki"
REG_ROOT = PROJECT_ROOT / "cache" / "runs" / "benchmark"
_OUTPUT_ROOTS = {
    "log_root": REG_ROOT / "b-log_result" / "model3_sing_reg",
    "fold_root": REG_ROOT / "a-fold" / "model3_sing_reg",
    "save_model_root": REG_ROOT / "c-model_save" / "model3_sing_reg",
}
LOG_ROOT = str(_OUTPUT_ROOTS["log_root"])
FOLD_ROOT = str(_OUTPUT_ROOTS["fold_root"])
SAVE_MODEL_ROOT = str(_OUTPUT_ROOTS["save_model_root"])

# Hyperparameter utama model3 didefinisikan sebagai fixed config di bawah ini.
# Ini bukan grid search; nilainya dipakai langsung saat training kecuali diubah manual.
RANDOM_SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
SAE_EPOCHS = 1
DNN_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
HIDDEN_NODES = 16
ACTIVATION = "relu"
NUM_LAYERS = 1
DROPOUT_RATE = 0.88
FEATURE_REDUCTION = 0.3
EARLY_STOPPING_PATIENCE = 2

TARGET_CANONICAL_NAME = "label"
PEP_PREFIX = "pep_embed_"
PROT_PREFIX = "prot_embed_"
MODEL_SELECTION_METRIC = "roc_auc"

METADATA_COLUMNS = ["pdb_id", "Uniprot_id", "prot_seq"]
DISALLOWED_FEATURE_COLUMNS = {"pdb_id", "Uniprot_id", "prot_seq", "inter_res", "label"}


@dataclass
class SchemaInfo:
    target_column_original: str
    target_column_canonical: str
    metadata_columns: List[str]
    pep_feature_columns: List[str]
    prot_feature_columns: List[str]
    feature_columns: List[str]
    row_count: int
    column_count: int
    class_distribution: Dict[str, int]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
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


def load_run_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "resume_meta": {},
            "folds": {},
            "final": {},
        }
    with open(path, "r", encoding="utf-8") as fh:
        state = json.load(fh)
    state.setdefault("resume_meta", {})
    state.setdefault("folds", {})
    state.setdefault("final", {})
    return state


def save_run_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_to_serializable(state), fh, indent=2)
    os.replace(tmp, path)


def build_resume_config_payload() -> Dict[str, Any]:
    return {
        "constants": {
            "DATA_PATH": DATA_PATH,
            "RANDOM_SEED": RANDOM_SEED,
            "TEST_SIZE": TEST_SIZE,
            "N_SPLITS": N_SPLITS,
            "SAE_EPOCHS": SAE_EPOCHS,
            "DNN_EPOCHS": DNN_EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "HIDDEN_NODES": HIDDEN_NODES,
            "ACTIVATION": ACTIVATION,
            "NUM_LAYERS": NUM_LAYERS,
            "DROPOUT_RATE": DROPOUT_RATE,
            "FEATURE_REDUCTION": FEATURE_REDUCTION,
            "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
            "MODEL_SELECTION_METRIC": MODEL_SELECTION_METRIC,
            "TARGET_CANONICAL_NAME": TARGET_CANONICAL_NAME,
        }
    }


def fixed_param_string() -> str:
    return _stable_json_dumps(
        {
            "sae_epochs": SAE_EPOCHS,
            "dnn_epochs": DNN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "hidden_nodes": HIDDEN_NODES,
            "activation": ACTIVATION,
            "num_layers": NUM_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            "feature_reduction": FEATURE_REDUCTION,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        }
    )


def get_hyperparameter_grid() -> List[Dict[str, Any]]:
    """Return the agreed 8-combination ProVenTL tuning grid."""
    grid: List[Dict[str, Any]] = []
    for learning_rate in (1e-3, 5e-4):
        for dropout_rate in (0.3, 0.5):
            for batch_size in (32, 64):
                grid.append(
                    {
                        "learning_rate": float(learning_rate),
                        "dropout_rate": float(dropout_rate),
                        "batch_size": int(batch_size),
                    }
                )
    return grid


def apply_hyperparams(params: Dict[str, Any]) -> None:
    global BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE
    BATCH_SIZE = int(params["batch_size"])
    LEARNING_RATE = float(params["learning_rate"])
    DROPOUT_RATE = float(params["dropout_rate"])


def _grid_trial_name(index: int, params: Dict[str, Any]) -> str:
    lr_label = "{:.0e}".format(float(params["learning_rate"])).replace("+", "")
    dropout_label = str(params["dropout_rate"]).replace(".", "p")
    return "grid{:02d}_bs{}_lr{}_drop{}".format(
        index,
        int(params["batch_size"]),
        lr_label,
        dropout_label,
    )


def snakebite_metrics_from_fold(fold_metric: Dict[str, Any]) -> Dict[str, float]:
    return {
        "ROC-AUC": float(fold_metric["roc_auc"]),
        "PR-AUC": float(fold_metric["average_precision"]),
        "Accuracy": float(fold_metric["accuracy"]),
        "Precision": float(fold_metric["precision"]),
        "Recall": float(fold_metric["recall"]),
        "F1": float(fold_metric["f1_score"]),
    }


def confusion_counts_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    return {
        "TP": int(cm[1, 1]),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
    }


def make_output_dirs(run_name: str) -> Dict[str, str]:
    log_root = ensure_dir(os.path.join(LOG_ROOT, run_name))
    fold_root = ensure_dir(os.path.join(FOLD_ROOT, run_name))
    model_root = ensure_dir(os.path.join(SAVE_MODEL_ROOT, run_name))
    resume_root = ensure_dir(os.path.join(model_root, "resume"))
    return {
        "run_name": run_name,
        "log_root": log_root,
        "fold_root": fold_root,
        "model_root": model_root,
        "text": os.path.join(log_root, "model3_sing_reg_results_rki.txt"),
        "reports": ensure_dir(os.path.join(log_root, "reports")),
        "plots": ensure_dir(os.path.join(log_root, "plots")),
        "logs": ensure_dir(os.path.join(log_root, "logs")),
        "fold_reports": ensure_dir(os.path.join(fold_root, "reports")),
        "predictions": ensure_dir(os.path.join(fold_root, "predictions")),
        "checkpoints": ensure_dir(os.path.join(model_root, "checkpoints")),
        "resume": resume_root,
        "state": os.path.join(log_root, "model3_sing_reg_resume_rki.json"),
    }


def setup_environment(out_dirs: Dict[str, str]) -> Dict[str, Any]:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    for directory in [
        out_dirs["log_root"],
        out_dirs["fold_root"],
        out_dirs["model_root"],
        out_dirs["reports"],
        out_dirs["plots"],
        out_dirs["fold_reports"],
        out_dirs["predictions"],
        out_dirs["checkpoints"],
    ]:
        os.makedirs(directory, exist_ok=True)

    device_info: Dict[str, Any] = {
        "seed": RANDOM_SEED,
        "gpu_available": False,
        "gpu_devices": [],
    }

    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        device_info["gpu_available"] = True
        device_info["gpu_devices"] = [device.name for device in physical_devices]
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except Exception:
                # TODO: On some TensorFlow/CUDA stacks memory growth can fail even when GPU exists.
                pass

    return device_info


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("proventl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def prepare_run_state(out_dirs: Dict[str, str], logger: logging.Logger) -> Tuple[Dict[str, Any], str]:
    run_state = load_run_state(out_dirs["state"])
    resume_config_payload = build_resume_config_payload()
    config_signature = compute_config_signature(resume_config_payload)
    saved_signature = str(run_state.get("resume_meta", {}).get("config_signature", "")).strip()
    if saved_signature and saved_signature != config_signature:
        raise RuntimeError(
            "Resume config mismatch untuk run {}. saved_signature={}, current_signature={}.".format(
                out_dirs["run_name"], saved_signature, config_signature
            )
        )
    run_state["resume_meta"] = {
        "config_signature": config_signature,
        "config_payload": resume_config_payload,
    }
    save_run_state(out_dirs["state"], run_state)
    logger.info("Config signature: %s", config_signature)
    return run_state, config_signature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model3 ProVenTL regularized fixed-config trainer.")
    parser.add_argument("--run-name", type=str, default="", help="Nama run baru. Kosongkan agar memakai default.")
    parser.add_argument("--resume-run", type=str, default="", help="Nama run untuk melanjutkan state yang sudah ada.")
    parser.add_argument(
        "--tune-grid",
        action="store_true",
        help="Jalankan grid search 8 kombinasi batch_size, learning_rate, dan dropout_rate sebelum evaluasi final.",
    )
    return parser.parse_args()


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH tidak ditemukan: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def _extract_numeric_suffix(column_name: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", column_name)
    if not match:
        raise ValueError(f"Kolom tidak mengikuti format indeks numerik: {column_name}")
    return int(match.group(1))


def _sorted_embedding_columns(columns: List[str], prefix: str) -> List[str]:
    selected = [column for column in columns if column.startswith(prefix)]
    return sorted(selected, key=lambda column: _extract_numeric_suffix(column, prefix))


def validate_schema(df: pd.DataFrame) -> SchemaInfo:
    if df.empty:
        raise ValueError("Dataset kosong.")

    if df.columns.duplicated().any():
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Nama kolom duplikat ditemukan: {duplicate_columns}")

    missing_required_metadata = [column for column in METADATA_COLUMNS if column not in df.columns]
    if missing_required_metadata:
        raise ValueError(f"Kolom metadata wajib hilang: {missing_required_metadata}")

    has_label = "label" in df.columns
    has_inter_res = "inter_res" in df.columns
    if not has_label and not has_inter_res:
        raise ValueError("Target tidak ditemukan. Dataset harus memiliki kolom 'label' atau 'inter_res'.")

    if has_label and has_inter_res and not df["label"].equals(df["inter_res"]):
        raise ValueError("Kolom 'label' dan 'inter_res' sama-sama ada tetapi nilainya berbeda.")

    target_column_original = "label" if has_label else "inter_res"

    pep_feature_columns = _sorted_embedding_columns(df.columns.tolist(), PEP_PREFIX)
    prot_feature_columns = _sorted_embedding_columns(df.columns.tolist(), PROT_PREFIX)
    if not pep_feature_columns:
        raise ValueError("Tidak ditemukan kolom fitur dengan prefix 'pep_embed_'.")
    if not prot_feature_columns:
        raise ValueError("Tidak ditemukan kolom fitur dengan prefix 'prot_embed_'.")

    feature_columns = pep_feature_columns + prot_feature_columns

    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Kolom embedding harus numerik: {column}")

    if df[feature_columns].isna().any().any():
        nan_columns = df[feature_columns].columns[df[feature_columns].isna().any()].tolist()
        raise ValueError(f"NaN ditemukan pada fitur embedding: {nan_columns}")

    if df[target_column_original].isna().any():
        raise ValueError(f"NaN ditemukan pada target '{target_column_original}'.")

    invalid_target_values = set(pd.Series(df[target_column_original]).dropna().unique().tolist()) - {0, 1}
    if invalid_target_values:
        raise ValueError(f"Target harus biner 0/1. Nilai invalid: {sorted(invalid_target_values)}")

    metadata_columns = [column for column in METADATA_COLUMNS if column in df.columns]
    class_distribution = df[target_column_original].value_counts().sort_index().to_dict()

    return SchemaInfo(
        target_column_original=target_column_original,
        target_column_canonical=TARGET_CANONICAL_NAME,
        metadata_columns=metadata_columns,
        pep_feature_columns=pep_feature_columns,
        prot_feature_columns=prot_feature_columns,
        feature_columns=feature_columns,
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        class_distribution={str(key): int(value) for key, value in class_distribution.items()},
    )


def extract_features(
    df: pd.DataFrame, schema_info: SchemaInfo
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, Any]]:
    working_df = df.copy()
    if schema_info.target_column_original != TARGET_CANONICAL_NAME:
        working_df[TARGET_CANONICAL_NAME] = working_df[schema_info.target_column_original]

    X = working_df.loc[:, schema_info.feature_columns].copy()
    y = working_df[TARGET_CANONICAL_NAME].astype(np.int32).copy()
    metadata_df = working_df.loc[:, schema_info.metadata_columns].copy()

    unexpected_feature_columns = [
        column
        for column in schema_info.feature_columns
        if column in DISALLOWED_FEATURE_COLUMNS
    ]
    if unexpected_feature_columns:
        raise ValueError(f"Metadata/target masuk ke fitur secara tidak valid: {unexpected_feature_columns}")

    feature_info = {
        "feature_count": len(schema_info.feature_columns),
        "pep_feature_count": len(schema_info.pep_feature_columns),
        "prot_feature_count": len(schema_info.prot_feature_columns),
        "feature_columns": schema_info.feature_columns,
    }
    return X, y, metadata_df, feature_info


def split_data(
    X: pd.DataFrame, y: pd.Series, metadata_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        metadata_df,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    return X_train, X_test, y_train, y_test, meta_train, meta_test


def build_sae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    logger: logging.Logger = None,
    epochs: int = SAE_EPOCHS,
    batch_size: int = BATCH_SIZE,
    hl_node: int = HIDDEN_NODES,
    lr: float = LEARNING_RATE,
    af: str = ACTIVATION,
    num_layers: int = NUM_LAYERS,
    dropout_rate: float = DROPOUT_RATE,
    feature_reduction: float = FEATURE_REDUCTION,
) -> List[List[np.ndarray]]:
    X_train_current = np.asarray(X_train, dtype=np.float32)
    X_val_current = np.asarray(X_val, dtype=np.float32)
    sae_weights: List[List[np.ndarray]] = []

    for layer_idx in range(num_layers):
        if logger is not None:
            logger.info(
                "Pretraining SAE layer %s/%s | input_dim=%s | hidden_dim=%s",
                layer_idx + 1,
                num_layers,
                X_train_current.shape[1],
                int(hl_node * (feature_reduction ** layer_idx)),
            )

        input_layer = Input(shape=(X_train_current.shape[1],))
        hidden_layer = Dropout(dropout_rate)(input_layer)
        encoded = Dense(int(hl_node * (feature_reduction ** layer_idx)), activation=af)(hidden_layer)
        decoded = Dense(X_train_current.shape[1], activation="linear")(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
        autoencoder.fit(
            X_train_current,
            X_train_current,
            validation_data=(X_val_current, X_val_current),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0,
        )

        encoder = Model(autoencoder.input, encoded)
        X_train_current = encoder.predict(X_train_current, verbose=0)
        X_val_current = encoder.predict(X_val_current, verbose=0)

        dense_layer = [layer for layer in autoencoder.layers if isinstance(layer, Dense)][0]
        sae_weights.append(dense_layer.get_weights())

        if logger is not None:
            logger.info("SAE layer %s/%s selesai", layer_idx + 1, num_layers)

    return sae_weights


def build_dnn(
    input_dim: int,
    sae_weights: List[List[np.ndarray]],
    n_outputs: int = 1,
    lr: float = LEARNING_RATE,
    hl_node: int = HIDDEN_NODES,
    af: str = ACTIVATION,
    num_layers: int = NUM_LAYERS,
    dropout_rate: float = DROPOUT_RATE,
    feature_reduction: float = FEATURE_REDUCTION,
) -> Model:
    input_layer = Input(shape=(input_dim,))
    hidden_layer = BatchNormalization()(input_layer)
    hidden_layer = Dropout(dropout_rate)(hidden_layer)

    for layer_idx in range(num_layers):
        hidden_layer = Dense(int(hl_node * (feature_reduction ** layer_idx)), activation=af)(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Dropout(dropout_rate)(hidden_layer)

    output_layer = Dense(n_outputs, activation="sigmoid")(hidden_layer)
    model = Model(input_layer, output_layer)

    if sae_weights is not None:
        dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)][:-1]
        if len(sae_weights) != len(dense_layers):
            raise ValueError(
                f"Jumlah bobot SAE ({len(sae_weights)}) tidak cocok dengan jumlah Dense layer ({len(dense_layers)})."
            )
        for pretrained_weights, dense_layer in zip(sae_weights, dense_layers):
            dense_layer.set_weights(pretrained_weights)

    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }
    return metrics


def train_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    logger: logging.Logger,
    out_dirs: Dict[str, str],
    run_state: Dict[str, Any],
) -> Dict[str, Any]:
    X_train_np = X_train.to_numpy(dtype=np.float32)
    y_train_np = y_train.to_numpy(dtype=np.int32)
    sample_indices = X_train.index.to_numpy(dtype=np.int64)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_metrics: List[Dict[str, Any]] = []
    history_frames: List[pd.DataFrame] = []
    roc_curves: List[Dict[str, Any]] = []
    pr_curves: List[Dict[str, Any]] = []
    fold_model_paths: List[str] = []
    fold_predictions: List[Dict[str, Any]] = []
    fold_splits: List[Dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np), start=1):
        fold_key = str(fold)
        logger.info("Mulai fold %s/%s", fold, N_SPLITS)
        saved_fold = run_state.get("folds", {}).get(fold_key)
        if saved_fold and saved_fold.get("completed"):
            saved_model_path = str(saved_fold.get("model_path", "")).strip()
            saved_fold_metric = saved_fold.get("fold_metric")
            saved_history_rows = saved_fold.get("history_rows")
            saved_roc_curve = saved_fold.get("roc_curve")
            saved_pr_curve = saved_fold.get("pr_curve")
            saved_fold_prediction = saved_fold.get("fold_prediction")
            saved_fold_split = saved_fold.get("fold_split")
            payload_complete = all(
                item is not None
                for item in (
                    saved_fold_metric,
                    saved_history_rows,
                    saved_roc_curve,
                    saved_pr_curve,
                    saved_fold_prediction,
                    saved_fold_split,
                )
            )
            if saved_model_path and os.path.exists(saved_model_path) and payload_complete:
                logger.info("[RESUME] Skip fold %s", fold)
                fold_metrics.append(saved_fold_metric)
                fold_model_paths.append(saved_model_path)
                history_frames.append(pd.DataFrame(saved_history_rows))
                roc_curves.append(saved_roc_curve)
                pr_curves.append(saved_pr_curve)
                fold_predictions.append(saved_fold_prediction)
                fold_splits.append(saved_fold_split)
                continue
            logger.warning(
                "[RESUME] Fold %s ditandai completed tetapi model/payload tidak valid. Fold akan dihitung ulang.",
                fold,
            )

        run_state.setdefault("folds", {})
        run_state["folds"].setdefault(fold_key, {"completed": False})
        save_run_state(out_dirs["state"], run_state)

        X_fold_train = X_train_np[train_idx]
        X_fold_val = X_train_np[val_idx]
        y_fold_train = y_train_np[train_idx].astype(np.float32)
        y_fold_val = y_train_np[val_idx].astype(np.float32)
        val_sample_indices = sample_indices[val_idx]
        train_sample_indices = sample_indices[train_idx]

        sae_weights = build_sae(
            X_train=X_fold_train,
            X_val=X_fold_val,
            logger=logger,
            epochs=SAE_EPOCHS,
            batch_size=BATCH_SIZE,
            hl_node=HIDDEN_NODES,
            lr=LEARNING_RATE,
            af=ACTIVATION,
            num_layers=NUM_LAYERS,
            dropout_rate=DROPOUT_RATE,
            feature_reduction=FEATURE_REDUCTION,
        )

        model = build_dnn(
            input_dim=X_fold_train.shape[1],
            sae_weights=sae_weights,
            n_outputs=1,
            lr=LEARNING_RATE,
            hl_node=HIDDEN_NODES,
            af=ACTIVATION,
            num_layers=NUM_LAYERS,
            dropout_rate=DROPOUT_RATE,
            feature_reduction=FEATURE_REDUCTION,
        )

        fold_model_path = os.path.join(out_dirs["checkpoints"], f"model_fold{fold}.keras")
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=0),
            ModelCheckpoint(filepath=fold_model_path, monitor="val_loss", save_best_only=True, verbose=0),
        ]

        logger.info("Mulai training DNN fold %s/%s", fold, N_SPLITS)
        history = model.fit(
            X_fold_train,
            y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=DNN_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        saved_model = load_model(fold_model_path)
        y_val_score = saved_model.predict(X_fold_val, verbose=0).flatten()
        fold_metric = _compute_binary_metrics(y_fold_val.astype(int), y_val_score)
        fold_metric["fold"] = fold
        fold_metrics.append(fold_metric)
        fold_model_paths.append(fold_model_path)

        history_df = pd.DataFrame(history.history)
        history_df["epoch"] = np.arange(1, len(history_df) + 1)
        history_df["fold"] = fold
        history_frames.append(history_df)

        fpr, tpr, _ = roc_curve(y_fold_val, y_val_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_fold_val, y_val_score)

        roc_curves.append({"fold": fold, "fpr": fpr, "tpr": tpr, "roc_auc": fold_metric["roc_auc"]})
        pr_curves.append(
            {
                "fold": fold,
                "precision": precision_curve,
                "recall": recall_curve,
                "average_precision": fold_metric["average_precision"],
            }
        )
        fold_predictions.append(
            {
                "fold": int(fold),
                "sample_indices": np.asarray(val_sample_indices, dtype=np.int64),
                "y_true": y_fold_val.astype(np.int64),
                "y_prob": y_val_score.astype(np.float32),
                "y_pred": (y_val_score >= 0.5).astype(np.int64),
            }
        )
        fold_splits.append(
            {
                "fold": int(fold),
                "train_indices": np.asarray(train_sample_indices, dtype=np.int64),
                "val_indices": np.asarray(val_sample_indices, dtype=np.int64),
            }
        )
        run_state["folds"][fold_key] = {
            "completed": True,
            "model_path": fold_model_path,
            "fold_metric": _to_serializable(fold_metric),
            "history_rows": _to_serializable(history_df.to_dict(orient="records")),
            "fold_prediction": _to_serializable(fold_predictions[-1]),
            "roc_curve": _to_serializable(roc_curves[-1]),
            "pr_curve": _to_serializable(pr_curves[-1]),
            "fold_split": _to_serializable(fold_splits[-1]),
        }
        save_run_state(out_dirs["state"], run_state)

        logger.info(
            "Fold %s selesai | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f ap=%.4f",
            fold,
            fold_metric["accuracy"],
            fold_metric["precision"],
            fold_metric["recall"],
            fold_metric["f1_score"],
            fold_metric["roc_auc"],
            fold_metric["average_precision"],
        )

    fold_metrics_df = pd.DataFrame(fold_metrics).sort_values("fold").reset_index(drop=True)
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()

    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"]
    summary_records = []
    for metric_name in metric_columns:
        summary_records.append(
            {
                "metric": metric_name,
                "mean": float(fold_metrics_df[metric_name].mean()),
                "std": float(fold_metrics_df[metric_name].std(ddof=1)),
            }
        )
    cv_summary_df = pd.DataFrame(summary_records)

    return {
        "fold_metrics_df": fold_metrics_df,
        "cv_summary_df": cv_summary_df,
        "history_df": history_df,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "fold_model_paths": fold_model_paths,
        "fold_predictions": fold_predictions,
        "fold_splits": fold_splits,
    }


def select_best_model(cv_result: Dict[str, Any], logger: logging.Logger, out_dirs: Dict[str, str]) -> Dict[str, Any]:
    fold_metrics_df = cv_result["fold_metrics_df"]
    best_row = fold_metrics_df.sort_values(
        by=[MODEL_SELECTION_METRIC, "average_precision", "accuracy"],
        ascending=[False, False, False],
    ).iloc[0]
    best_fold = int(best_row["fold"])
    source_model_path = os.path.join(out_dirs["checkpoints"], f"model_fold{best_fold}.keras")
    if not os.path.exists(source_model_path):
        raise FileNotFoundError(f"Model terbaik tidak ditemukan: {source_model_path}")

    best_model_path = os.path.join(out_dirs["checkpoints"], "best_model.keras")
    shutil.copy2(source_model_path, best_model_path)
    best_info = {
        "best_fold": best_fold,
        "best_model_path": best_model_path,
        "selection_metric": MODEL_SELECTION_METRIC,
        "roc_auc": float(best_row["roc_auc"]),
        "average_precision": float(best_row["average_precision"]),
        "accuracy": float(best_row["accuracy"]),
    }
    logger.info(
        "Best model dipilih dari fold %s | roc_auc=%.4f | disalin ke %s",
        best_fold,
        best_info["roc_auc"],
        best_model_path,
    )
    return best_info


def evaluate_test(
    best_model_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    logger: logging.Logger,
) -> Dict[str, Any]:
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model tidak ditemukan: {best_model_path}")

    model = load_model(best_model_path)
    X_test_np = X_test.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.int32)
    y_score = model.predict(X_test_np, verbose=0).flatten()
    y_pred = (y_score >= 0.5).astype(int)

    metrics = _compute_binary_metrics(y_test_np, y_score)

    predictions_df = meta_test.copy().reset_index(drop=True)
    predictions_df["sample_indices"] = X_test.index.to_numpy(dtype=np.int64)
    predictions_df[TARGET_CANONICAL_NAME] = y_test.reset_index(drop=True)
    predictions_df["y_pred"] = y_pred
    predictions_df["y_score"] = y_score

    fpr, tpr, _ = roc_curve(y_test_np, y_score)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_np, y_score)
    cm = confusion_matrix(y_test_np, y_pred)

    logger.info(
        "Evaluasi test selesai | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f ap=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
        metrics["average_precision"],
    )

    return {
        "metrics": metrics,
        "predictions_df": predictions_df,
        "roc_curve": {"fpr": fpr, "tpr": tpr, "roc_auc": metrics["roc_auc"]},
        "pr_curve": {
            "precision": precision_curve,
            "recall": recall_curve,
            "average_precision": metrics["average_precision"],
        },
        "confusion_matrix": cm,
        "sample_indices": X_test.index.to_numpy(dtype=np.int64),
        "y_true": y_test_np.astype(np.int64),
        "y_prob": y_score.astype(np.float32),
        "y_pred": y_pred.astype(np.int64),
    }


def _write_text_report(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def save_outputs(
    schema_info: SchemaInfo,
    feature_info: Dict[str, Any],
    cv_result: Dict[str, Any],
    best_model_info: Dict[str, Any],
    test_result: Dict[str, Any],
    environment_info: Dict[str, Any],
    out_dirs: Dict[str, str],
    logger: logging.Logger,
) -> None:
    cv_metrics_path = os.path.join(out_dirs["reports"], "kfold_metrics.csv")
    cv_summary_path = os.path.join(out_dirs["reports"], "cv_summary.csv")
    cv_history_path = os.path.join(out_dirs["reports"], "all_history.csv")
    best_fold_info_path = os.path.join(out_dirs["reports"], "best_fold_info.csv")
    test_metrics_path = os.path.join(out_dirs["reports"], "final_test_metrics.csv")
    test_predictions_path = os.path.join(out_dirs["predictions"], "final_test_predictions.csv")
    cv_result["fold_metrics_df"].to_csv(cv_metrics_path, index=False)
    cv_result["fold_metrics_df"].to_csv(os.path.join(out_dirs["fold_reports"], "kfold_metrics.csv"), index=False)
    cv_result["cv_summary_df"].to_csv(cv_summary_path, index=False)
    cv_result["history_df"].to_csv(cv_history_path, index=False)
    pd.DataFrame([best_model_info]).to_csv(best_fold_info_path, index=False)
    pd.DataFrame([test_result["metrics"]]).to_csv(test_metrics_path, index=False)
    test_result["predictions_df"].to_csv(test_predictions_path, index=False)

    run_config = {
        "notebook_reference_path": NOTEBOOK_REFERENCE_PATH,
        "data_path": DATA_PATH,
        "output_dir": out_dirs["log_root"],
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "n_splits": N_SPLITS,
        "sae_epochs": SAE_EPOCHS,
        "dnn_epochs": DNN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "hidden_nodes": HIDDEN_NODES,
        "activation": ACTIVATION,
        "num_layers": NUM_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "feature_reduction": FEATURE_REDUCTION,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "selection_metric": MODEL_SELECTION_METRIC,
    }

    schema_report = {
        "row_count": schema_info.row_count,
        "column_count": schema_info.column_count,
        "target_column_original": schema_info.target_column_original,
        "target_column_canonical": schema_info.target_column_canonical,
        "metadata_columns": schema_info.metadata_columns,
        "pep_feature_count": len(schema_info.pep_feature_columns),
        "prot_feature_count": len(schema_info.prot_feature_columns),
        "feature_count": len(schema_info.feature_columns),
        "class_distribution": schema_info.class_distribution,
        "feature_info": feature_info,
        "environment_info": environment_info,
    }

    _write_text_report(os.path.join(out_dirs["reports"], "run_config.txt"), json.dumps(run_config, indent=2))
    _write_text_report(os.path.join(out_dirs["reports"], "schema_report.txt"), json.dumps(schema_report, indent=2))
    with open(os.path.join(out_dirs["reports"], "config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)
    with open(os.path.join(out_dirs["reports"], "dataset_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "num_samples": schema_info.row_count,
                "num_positive": int(schema_info.class_distribution.get("1", 0)),
                "num_negative": int(schema_info.class_distribution.get("0", 0)),
                "feature_count": len(schema_info.feature_columns),
            },
            fh,
            indent=2,
        )

    plt.figure(figsize=(8, 6))
    for curve in cv_result["roc_curves"]:
        plt.plot(curve["fpr"], curve["tpr"], label=f"Fold {curve['fold']} (AUC={curve['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.title("ROC Curve - Stratified 5-Fold CV")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dirs["plots"], "roc_auc_kfold_all.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    for curve in cv_result["pr_curves"]:
        plt.plot(
            curve["recall"],
            curve["precision"],
            label=f"Fold {curve['fold']} (AP={curve['average_precision']:.4f})",
        )
    plt.title("Precision-Recall Curve - Stratified 5-Fold CV")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dirs["plots"], "pr_curve_kfold_all.png"), dpi=150)
    plt.close()

    history_df = cv_result["history_df"]
    if not history_df.empty:
        mean_history = (
            history_df.groupby("epoch")[["loss", "val_loss", "accuracy", "val_accuracy"]].mean().reset_index()
        )
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(mean_history["epoch"], mean_history["accuracy"], label="Train Accuracy")
        plt.plot(mean_history["epoch"], mean_history["val_accuracy"], label="Val Accuracy")
        plt.title("Average Accuracy Across Folds")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(mean_history["epoch"], mean_history["loss"], label="Train Loss")
        plt.plot(mean_history["epoch"], mean_history["val_loss"], label="Val Loss")
        plt.title("Average Loss Across Folds")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dirs["plots"], "avg_loss_acc_kfold.png"), dpi=150)
        plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(
        test_result["roc_curve"]["fpr"],
        test_result["roc_curve"]["tpr"],
        label=f"AUC={test_result['roc_curve']['roc_auc']:.4f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        test_result["pr_curve"]["recall"],
        test_result["pr_curve"]["precision"],
        label=f"AP={test_result['pr_curve']['average_precision']:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dirs["plots"], "final_test_roc_pr_curve.png"), dpi=150)
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=test_result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dirs["plots"], "final_test_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    np.savez_compressed(
        os.path.join(out_dirs["fold_root"], "split_indices.npz"),
        folds=np.asarray(cv_result["fold_splits"], dtype=object),
        test_indices=np.asarray(test_result["sample_indices"], dtype=np.int64),
    )
    for item in cv_result["fold_predictions"]:
        fold = int(item["fold"])
        np.savez_compressed(
            os.path.join(out_dirs["predictions"], f"fold_{fold}_predictions.npz"),
            sample_indices=np.asarray(item["sample_indices"], dtype=np.int64),
            y_true=np.asarray(item["y_true"], dtype=np.int64),
            y_prob=np.asarray(item["y_prob"], dtype=np.float32),
            y_pred=np.asarray(item["y_pred"], dtype=np.int64),
        )
        pd.DataFrame(
            {
                "sample_indices": np.asarray(item["sample_indices"], dtype=np.int64),
                "y_true": np.asarray(item["y_true"], dtype=np.int64),
                "y_prob": np.asarray(item["y_prob"], dtype=np.float32),
                "y_pred": np.asarray(item["y_pred"], dtype=np.int64),
            }
        ).to_csv(os.path.join(out_dirs["predictions"], f"fold_{fold}_predictions.csv"), index=False)
    np.savez_compressed(
        os.path.join(out_dirs["predictions"], "kfold_predictions.npz"),
        folds=np.asarray(cv_result["fold_predictions"], dtype=object),
    )
    np.savez_compressed(
        os.path.join(out_dirs["predictions"], "final_test_predictions.npz"),
        sample_indices=np.asarray(test_result["sample_indices"], dtype=np.int64),
        y_true=np.asarray(test_result["y_true"], dtype=np.int64),
        y_prob=np.asarray(test_result["y_prob"], dtype=np.float32),
        y_pred=np.asarray(test_result["y_pred"], dtype=np.int64),
    )
    with open(os.path.join(out_dirs["reports"], "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_name": out_dirs["run_name"],
                "selection_metric": MODEL_SELECTION_METRIC,
                "best_fold": best_model_info["best_fold"],
                "kfold_mean": cv_result["cv_summary_df"].set_index("metric")["mean"].to_dict(),
                "kfold_std": cv_result["cv_summary_df"].set_index("metric")["std"].to_dict(),
                "final_test_metrics": test_result["metrics"],
            },
            fh,
            indent=2,
        )

    logger.info("Semua artefak berhasil disimpan ke %s", out_dirs["log_root"])


def build_snakebite_report(
    cv_result: Dict[str, Any],
    test_result: Dict[str, Any],
    best_model_info: Dict[str, Any],
    start_time: float,
) -> str:
    param_string = fixed_param_string()
    detail_lines: List[str] = []
    best_params_per_fold: List[Dict[str, Any]] = []
    for _, row in cv_result["fold_metrics_df"].sort_values("fold").iterrows():
        fold_no = int(row["fold"])
        prediction = next(item for item in cv_result["fold_predictions"] if int(item["fold"]) == fold_no)
        metrics = snakebite_metrics_from_fold(row.to_dict())
        confusion = confusion_counts_from_arrays(np.asarray(prediction["y_true"]), np.asarray(prediction["y_pred"]))
        detail_lines.append(f"Fold {fold_no}/{N_SPLITS}")
        detail_lines.extend(
            render_candidate_detail(
                index=1,
                param_string=param_string,
                metrics=metrics,
                confusion=confusion,
            )
        )
        detail_lines.extend(
            render_selected_detail(
                param_string=param_string,
                selection_metric_name="ROC-AUC",
                selection_metric_value=float(row["roc_auc"]),
                metrics=metrics,
                confusion=confusion,
            )
        )
        best_params_per_fold.append(
            {
                "fold": fold_no,
                "sae_epochs": SAE_EPOCHS,
                "dnn_epochs": DNN_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "hidden_nodes": HIDDEN_NODES,
                "activation": ACTIVATION,
                "num_layers": NUM_LAYERS,
                "dropout_rate": DROPOUT_RATE,
                "feature_reduction": FEATURE_REDUCTION,
            }
        )

    report_fold_metrics_df = pd.DataFrame(
        [
            {
                "ROC-AUC": row["roc_auc"],
                "PR-AUC": row["average_precision"],
                "Accuracy": row["accuracy"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1_score"],
                "fold": int(row["fold"]),
            }
            for row in cv_result["fold_metrics_df"].sort_values("fold").to_dict(orient="records")
        ]
    )
    aggregated_metrics_df = pd.DataFrame(
        {
            "ROC-AUC": [float(report_fold_metrics_df["ROC-AUC"].mean()), float(report_fold_metrics_df["ROC-AUC"].std(ddof=0))],
            "PR-AUC": [float(report_fold_metrics_df["PR-AUC"].mean()), float(report_fold_metrics_df["PR-AUC"].std(ddof=0))],
            "Accuracy": [float(report_fold_metrics_df["Accuracy"].mean()), float(report_fold_metrics_df["Accuracy"].std(ddof=0))],
            "Precision": [float(report_fold_metrics_df["Precision"].mean()), float(report_fold_metrics_df["Precision"].std(ddof=0))],
            "Recall": [float(report_fold_metrics_df["Recall"].mean()), float(report_fold_metrics_df["Recall"].std(ddof=0))],
            "F1": [float(report_fold_metrics_df["F1"].mean()), float(report_fold_metrics_df["F1"].std(ddof=0))],
        },
        index=["mean", "std"],
    )
    params_per_fold_df = pd.DataFrame(best_params_per_fold)
    final_confusion = confusion_counts_from_arrays(np.asarray(test_result["y_true"]), np.asarray(test_result["y_pred"]))
    final_metrics = {
        "ROC-AUC": float(test_result["metrics"]["roc_auc"]),
        "PR-AUC": float(test_result["metrics"]["average_precision"]),
        "Accuracy": float(test_result["metrics"]["accuracy"]),
        "Precision": float(test_result["metrics"]["precision"]),
        "Recall": float(test_result["metrics"]["recall"]),
        "F1": float(test_result["metrics"]["f1_score"]),
    }
    return render_sections(
        [
            ("Detail single-grid best-param & evaluasi", "\n".join(detail_lines)),
            ("Per-fold metrics", dataframe_to_string(report_fold_metrics_df, index=False)),
            ("Aggregated metrics", dataframe_to_string(aggregated_metrics_df, index=True)),
            ("Model params per fold", dataframe_to_string(params_per_fold_df, index=False)),
            ("Voting Best Model Param", "\n".join(voting_lines(best_params_per_fold))),
            (
                "Evaluasi final test",
                "\n".join(
                    [
                        f"Model final: fold_{best_model_info['best_fold']}",
                        f"ROC-AUC={final_metrics['ROC-AUC']:.4f} | PR-AUC={final_metrics['PR-AUC']:.4f} | Accuracy={final_metrics['Accuracy']:.4f} | Precision={final_metrics['Precision']:.4f} | Recall={final_metrics['Recall']:.4f} | F1={final_metrics['F1']:.4f}",
                        f"True Positive (TP) = {final_confusion['TP']} | True Negative (TN) = {final_confusion['TN']} | False Positive (FP) = {final_confusion['FP']} | False Negative (FN) = {final_confusion['FN']}",
                    ]
                ),
            ),
        ],
        build_execution_report(start_time),
    )


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_run_name: str,
    base_out_dirs: Dict[str, str],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    grid = get_hyperparameter_grid()
    grid_rows: List[Dict[str, Any]] = []
    logger.info("Mulai grid search ProVenTL: %s kombinasi", len(grid))

    for index, params in enumerate(grid, start=1):
        apply_hyperparams(params)
        trial_name = _grid_trial_name(index, params)
        trial_run_name = f"{base_run_name}_{trial_name}"
        trial_out_dirs = make_output_dirs(os.path.join(base_run_name, "grid_search", trial_name))
        setup_environment(trial_out_dirs)
        trial_run_state, _ = prepare_run_state(trial_out_dirs, logger)

        logger.info(
            "[GRID %s/%s] run=%s | batch_size=%s | learning_rate=%s | dropout_rate=%s",
            index,
            len(grid),
            trial_run_name,
            BATCH_SIZE,
            LEARNING_RATE,
            DROPOUT_RATE,
        )
        cv_result = train_with_cv(X_train, y_train, logger, trial_out_dirs, trial_run_state)
        metrics_df = cv_result["fold_metrics_df"]
        row = {
            "rank": None,
            "grid_index": int(index),
            "trial_name": trial_name,
            "run_name": trial_run_name,
            "batch_size": int(BATCH_SIZE),
            "learning_rate": float(LEARNING_RATE),
            "dropout_rate": float(DROPOUT_RATE),
            "mean_val_roc_auc": float(metrics_df["roc_auc"].mean()),
            "std_val_roc_auc": float(metrics_df["roc_auc"].std(ddof=0)),
            "mean_val_pr_auc": float(metrics_df["average_precision"].mean()),
            "std_val_pr_auc": float(metrics_df["average_precision"].std(ddof=0)),
            "mean_val_f1": float(metrics_df["f1_score"].mean()),
            "std_val_f1": float(metrics_df["f1_score"].std(ddof=0)),
        }
        grid_rows.append(row)
        logger.info(
            "[GRID %s] mean ROC-AUC=%.4f | mean PR-AUC=%.4f | mean F1=%.4f",
            index,
            row["mean_val_roc_auc"],
            row["mean_val_pr_auc"],
            row["mean_val_f1"],
        )

    ranked_rows = sorted(grid_rows, key=lambda row: row["mean_val_roc_auc"], reverse=True)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = int(rank)

    pd.DataFrame(ranked_rows).to_csv(os.path.join(base_out_dirs["reports"], "grid_search_results.csv"), index=False)
    _write_text_report(
        os.path.join(base_out_dirs["reports"], "grid_search_best_params.json"),
        json.dumps(
            {
                "selection_metric": "mean_val_roc_auc",
                "best_score": ranked_rows[0]["mean_val_roc_auc"] if ranked_rows else None,
                "best_params": {
                    "batch_size": ranked_rows[0]["batch_size"],
                    "learning_rate": ranked_rows[0]["learning_rate"],
                    "dropout_rate": ranked_rows[0]["dropout_rate"],
                }
                if ranked_rows
                else None,
                "num_combinations": len(grid),
            },
            indent=2,
        ),
    )
    return ranked_rows


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    if args.tune_grid:
        print("Single-grid mode: --tune-grid diabaikan; memakai best param artefak 22-05-2026.")
        args.tune_grid = False
    if args.tune_grid and args.resume_run.strip():
        raise ValueError("--tune-grid tidak bisa dipakai bersamaan dengan --resume-run.")
    if args.resume_run.strip():
        run_name = args.resume_run.strip()
    elif args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        run_name = "model3_sing_reg"
    out_dirs = make_output_dirs(run_name)
    environment_info = setup_environment(out_dirs)
    logger = setup_logging(os.path.join(out_dirs["logs"], "run.log"))
    if args.resume_run.strip() and not os.path.exists(out_dirs["state"]):
        raise FileNotFoundError(f"Resume state untuk run '{run_name}' tidak ditemukan di {out_dirs['state']}")

    try:
        logger.info("Mulai pipeline ProVenTL standalone.")
        logger.info("Run name: %s", out_dirs["run_name"])
        logger.info("Log root: %s", out_dirs["log_root"])
        logger.info("Fold root: %s", out_dirs["fold_root"])
        logger.info("Model root: %s", out_dirs["model_root"])
        logger.info("Notebook referensi dicatat untuk dokumentasi saja: %s", NOTEBOOK_REFERENCE_PATH)
        logger.info("GPU tersedia: %s | devices=%s", environment_info["gpu_available"], environment_info["gpu_devices"])

        df = load_data()
        logger.info("Dataset dimuat dari %s | shape=%s", DATA_PATH, df.shape)

        schema_info = validate_schema(df)
        logger.info(
            "Schema valid | target=%s -> %s | pep_features=%s | prot_features=%s",
            schema_info.target_column_original,
            schema_info.target_column_canonical,
            len(schema_info.pep_feature_columns),
            len(schema_info.prot_feature_columns),
        )

        X, y, metadata_df, feature_info = extract_features(df, schema_info)
        logger.info("Fitur diekstrak | X=%s | y=%s | metadata=%s", X.shape, y.shape, metadata_df.shape)

        X_train, X_test, y_train, y_test, meta_train, meta_test = split_data(X, y, metadata_df)
        logger.info("Split selesai | train=%s | test=%s", X_train.shape, X_test.shape)

        if args.tune_grid:
            ranked_grid = run_grid_search(X_train, y_train, run_name, out_dirs, logger)
            if not ranked_grid:
                raise RuntimeError("Grid search tidak menghasilkan kandidat valid.")
            best_grid = ranked_grid[0]
            apply_hyperparams(best_grid)
            final_run_name = f"{run_name}_grid_best"
            out_dirs = make_output_dirs(final_run_name)
            environment_info = setup_environment(out_dirs)
            logger = setup_logging(os.path.join(out_dirs["logs"], "run.log"))
            logger.info(
                "Final grid-best run | batch_size=%s | learning_rate=%s | dropout_rate=%s | mean_val_roc_auc=%.4f",
                BATCH_SIZE,
                LEARNING_RATE,
                DROPOUT_RATE,
                float(best_grid["mean_val_roc_auc"]),
            )

        run_state, _ = prepare_run_state(out_dirs, logger)
        cv_result = train_with_cv(X_train, y_train, logger, out_dirs, run_state)
        best_model_info = select_best_model(cv_result, logger, out_dirs)
        test_result = evaluate_test(best_model_info["best_model_path"], X_test, y_test, meta_test, logger)

        save_outputs(
            schema_info=schema_info,
            feature_info=feature_info,
            cv_result=cv_result,
            best_model_info=best_model_info,
            test_result=test_result,
            environment_info=environment_info,
            out_dirs=out_dirs,
            logger=logger,
        )

        final_report = build_snakebite_report(cv_result, test_result, best_model_info, start_time)
        _write_text_report(out_dirs["text"], final_report)
        print(final_report, end="")

        logger.info("Pipeline selesai tanpa error.")
    except Exception as exc:
        if "logger" in locals():
            logger.exception("Pipeline gagal: %s", exc)
        else:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
