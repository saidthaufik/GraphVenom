from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


def _load_base_module():
    module_path = Path(__file__).resolve().with_name("proventl_regularized.py")
    spec = importlib.util.spec_from_file_location("model3_proventl_regularized_base", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Gagal memuat base trainer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "model3_sing_reg"
DATA_PATH = PROJECT_ROOT.parent / "1-data" / "features" / "model3" / "1-feat_prott5" / "prott5pn_sequence_clean.csv"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT.parent / "1-data" / "splits" / "unseen" / "model3"
DATASET_NAME = "rki"
N_SPLITS = 5
THRESHOLD_GRID = np.array(sorted(set(np.round(np.concatenate([np.arange(0.1, 1.01, 0.1), [0.5]]), 3))))
SCENARIO_ROOTS = {
    "unseen_protein": PROJECT_ROOT / "cache" / "runs" / "unseen" / "protein",
    "unseen_peptide": PROJECT_ROOT / "cache" / "runs" / "unseen" / "peptide",
    "unseen_both": PROJECT_ROOT / "cache" / "runs" / "unseen" / "both",
}
SCENARIO_LABELS = {
    "unseen_protein": "Unseen Protein",
    "unseen_peptide": "Unseen Peptide",
    "unseen_both": "Unseen Both",
}


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return obj


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def parse_args(scenario: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Model 3 ProVenTL regularized - {SCENARIO_LABELS[scenario]}.")
    parser.add_argument("--data-csv", type=Path, default=DATA_PATH)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=SCENARIO_ROOTS[scenario])
    parser.add_argument("--run-name", type=str, default=MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--inner-val-size", type=float, default=0.2)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_output_dirs(output_root: Path, run_name: str) -> dict[str, Path]:
    log_root = ensure_dir(output_root / "b-log_result" / run_name)
    fold_root = ensure_dir(output_root / "a-fold" / run_name)
    model_root = ensure_dir(output_root / "c-model_save" / run_name)
    return {
        "run_name": run_name,
        "log_root": log_root,
        "fold_root": fold_root,
        "model_root": model_root,
        "reports": ensure_dir(log_root / "reports"),
        "plots": ensure_dir(log_root / "plots"),
        "logs": ensure_dir(log_root / "logs"),
        "predictions": ensure_dir(fold_root / "predictions"),
        "fold_reports": ensure_dir(fold_root / "reports"),
        "checkpoints": ensure_dir(model_root / "checkpoints"),
        "text": log_root / f"{run_name}_results_{DATASET_NAME}.txt",
        "state": log_root / f"{run_name}_resume_{DATASET_NAME}.json",
    }


def log_line(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


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
    fold_no = 1
    while True:
        train_path = scenario_dir / f"fold_{fold_no}_train_idx.csv"
        test_path = scenario_dir / f"fold_{fold_no}_test_idx.csv"
        if not train_path.exists() and not test_path.exists():
            break
        train_idx = read_indices(train_path, n_samples)
        test_idx = read_indices(test_path, n_samples)
        overlap = np.intersect1d(train_idx, test_idx)
        if overlap.size:
            raise ValueError(f"Train/test overlap pada {scenario} fold {fold_no}: {overlap[:5].tolist()}")
        folds.append({"fold": fold_no, "train_idx": train_idx, "test_idx": test_idx})
        if max_folds > 0 and len(folds) >= max_folds:
            break
        fold_no += 1
    if not folds:
        raise ValueError(f"Tidak ada fold ditemukan untuk {scenario_dir}")
    return folds


def setup_environment() -> dict[str, Any]:
    random.seed(base.RANDOM_SEED)
    np.random.seed(base.RANDOM_SEED)
    tf.keras.utils.set_random_seed(base.RANDOM_SEED)
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            pass
    return {
        "seed": base.RANDOM_SEED,
        "gpu_available": bool(physical_devices),
        "gpu_devices": [device.name for device in physical_devices],
    }


def split_inner_train_val(train_idx: np.ndarray, y: pd.Series, val_size: float, fold_no: int) -> tuple[np.ndarray, np.ndarray]:
    if val_size <= 0:
        return train_idx, train_idx
    y_train = y.iloc[train_idx].to_numpy(dtype=np.int32)
    stratify = y_train if len(np.unique(y_train)) == 2 else None
    inner_train, inner_val = train_test_split(
        train_idx,
        test_size=val_size,
        shuffle=True,
        stratify=stratify,
        random_state=base.RANDOM_SEED + int(fold_no),
    )
    return np.asarray(inner_train, dtype=np.int64), np.asarray(inner_val, dtype=np.int64)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def confusion_counts(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, int]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}


def threshold_recap(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLD_GRID:
        metrics = binary_metrics(y_true, y_score, float(threshold))
        rows.append({"threshold": float(threshold), **metrics, **confusion_counts(y_true, y_score, float(threshold))})
    return pd.DataFrame(rows)


def save_curve_plots(roc_entries: list[dict[str, Any]], pr_entries: list[dict[str, Any]], out_dirs: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for item in roc_entries:
        ax.plot(item["fpr"], item["tpr"], label=f"Fold {item['fold']} (AUC={item['auc']:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Across Unseen Folds")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dirs["plots"] / "roc_auc_kfold_all.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for item in pr_entries:
        ax.plot(item["recall"], item["precision"], label=f"Fold {item['fold']} (AP={item['ap']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Across Unseen Folds")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dirs["plots"] / "pr_curve_kfold_all.png", dpi=200)
    plt.close(fig)


def save_threshold_plot(df: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        ax.plot(df["threshold"], df[metric], marker="o", label=metric)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("OOF Threshold Recap")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dirs["plots"] / "threshold_recap_0_1_to_1_0.png", dpi=200)
    plt.close(fig)


def run_scenario(scenario: str) -> None:
    start = time.perf_counter()
    args = parse_args(scenario)
    if not 0 <= float(args.threshold) <= 1:
        raise ValueError("threshold harus berada pada rentang 0..1")

    out_dirs = make_output_dirs(args.output_root, args.run_name)
    if out_dirs["text"].exists() and not args.resume:
        out_dirs["text"].unlink()
    env_info = setup_environment()

    df = pd.read_csv(args.data_csv)
    schema_info = base.validate_schema(df)
    X, y, metadata_df, feature_info = base.extract_features(df, schema_info)
    folds = load_unseen_folds(args.split_root, scenario, len(df), int(args.max_folds))

    config = {
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "model_name": MODEL_NAME,
        "data_csv": str(args.data_csv),
        "split_root": str(args.split_root),
        "output_root": str(args.output_root),
        "run_name": args.run_name,
        "threshold": float(args.threshold),
        "inner_val_size": float(args.inner_val_size),
        "random_seed": base.RANDOM_SEED,
        "n_splits": len(folds),
        "sae_epochs": base.SAE_EPOCHS,
        "dnn_epochs": base.DNN_EPOCHS,
        "batch_size": base.BATCH_SIZE,
        "learning_rate": base.LEARNING_RATE,
        "hidden_nodes": base.HIDDEN_NODES,
        "activation": base.ACTIVATION,
        "num_layers": base.NUM_LAYERS,
        "dropout_rate": base.DROPOUT_RATE,
        "feature_reduction": base.FEATURE_REDUCTION,
        "early_stopping_patience": base.EARLY_STOPPING_PATIENCE,
        "environment_info": env_info,
    }
    save_json(out_dirs["reports"] / "config.json", config)

    log_line(out_dirs["text"], f"Model 3 {SCENARIO_LABELS[scenario]} | run={args.run_name}")
    log_line(out_dirs["text"], f"Data: {args.data_csv} | rows={len(df)} | class={schema_info.class_distribution}")
    log_line(out_dirs["text"], f"Split root: {args.split_root}")

    fold_metrics: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    roc_entries: list[dict[str, Any]] = []
    pr_entries: list[dict[str, Any]] = []
    fold_predictions: list[dict[str, Any]] = []
    fold_splits: list[dict[str, Any]] = []

    for fold_spec in folds:
        fold_no = int(fold_spec["fold"])
        train_idx = np.asarray(fold_spec["train_idx"], dtype=np.int64)
        test_idx = np.asarray(fold_spec["test_idx"], dtype=np.int64)
        inner_train_idx, inner_val_idx = split_inner_train_val(train_idx, y, float(args.inner_val_size), fold_no)

        X_inner_train = X.iloc[inner_train_idx].to_numpy(dtype=np.float32)
        X_inner_val = X.iloc[inner_val_idx].to_numpy(dtype=np.float32)
        X_test = X.iloc[test_idx].to_numpy(dtype=np.float32)
        y_inner_train = y.iloc[inner_train_idx].to_numpy(dtype=np.float32)
        y_inner_val = y.iloc[inner_val_idx].to_numpy(dtype=np.float32)
        y_test = y.iloc[test_idx].to_numpy(dtype=np.int32)

        log_line(
            out_dirs["text"],
            f"Fold {fold_no}/{len(folds)} | train={len(train_idx)} inner_train={len(inner_train_idx)} "
            f"inner_val={len(inner_val_idx)} test={len(test_idx)}",
        )

        fold_model_path = out_dirs["checkpoints"] / f"model_fold{fold_no}.keras"
        if args.resume and fold_model_path.exists():
            model = load_model(fold_model_path)
            history_df = pd.DataFrame()
        else:
            sae_weights = base.build_sae(
                X_train=X_inner_train,
                X_val=X_inner_val,
                logger=None,
                epochs=base.SAE_EPOCHS,
                batch_size=base.BATCH_SIZE,
                hl_node=base.HIDDEN_NODES,
                lr=base.LEARNING_RATE,
                af=base.ACTIVATION,
                num_layers=base.NUM_LAYERS,
                dropout_rate=base.DROPOUT_RATE,
                feature_reduction=base.FEATURE_REDUCTION,
            )
            model = base.build_dnn(
                input_dim=X_inner_train.shape[1],
                sae_weights=sae_weights,
                n_outputs=1,
                lr=base.LEARNING_RATE,
                hl_node=base.HIDDEN_NODES,
                af=base.ACTIVATION,
                num_layers=base.NUM_LAYERS,
                dropout_rate=base.DROPOUT_RATE,
                feature_reduction=base.FEATURE_REDUCTION,
            )
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=base.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=0),
                ModelCheckpoint(filepath=str(fold_model_path), monitor="val_loss", save_best_only=True, verbose=0),
            ]
            history = model.fit(
                X_inner_train,
                y_inner_train,
                validation_data=(X_inner_val, y_inner_val),
                epochs=base.DNN_EPOCHS,
                batch_size=base.BATCH_SIZE,
                callbacks=callbacks,
                verbose=1,
            )
            model = load_model(fold_model_path)
            history_df = pd.DataFrame(history.history)
            history_df["epoch"] = np.arange(1, len(history_df) + 1)
            history_df["fold"] = fold_no

        y_score = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_score >= float(args.threshold)).astype(np.int64)
        metrics = binary_metrics(y_test, y_score, float(args.threshold))
        counts = confusion_counts(y_test, y_score, float(args.threshold))
        fold_row = {"fold": fold_no, **metrics, **counts}
        fold_metrics.append(fold_row)
        if not history_df.empty:
            history_frames.append(history_df)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
        roc_entries.append({"fold": fold_no, "fpr": fpr, "tpr": tpr, "auc": metrics["roc_auc"]})
        pr_entries.append({"fold": fold_no, "precision": precision_curve, "recall": recall_curve, "ap": metrics["average_precision"]})

        pred_df = metadata_df.iloc[test_idx].copy().reset_index(drop=True)
        pred_df["sample_indices"] = test_idx
        pred_df["y_true"] = y_test
        pred_df["y_prob"] = y_score
        pred_df["y_pred"] = y_pred
        pred_df.to_csv(out_dirs["predictions"] / f"fold_{fold_no}_predictions.csv", index=False)
        np.savez_compressed(
            out_dirs["predictions"] / f"fold_{fold_no}_predictions.npz",
            sample_indices=test_idx,
            y_true=y_test,
            y_prob=y_score.astype(np.float32),
            y_pred=y_pred,
        )
        fold_predictions.append({"fold": fold_no, "sample_indices": test_idx, "y_true": y_test, "y_prob": y_score, "y_pred": y_pred})
        fold_splits.append({"fold": fold_no, "train_indices": train_idx, "inner_train_indices": inner_train_idx, "inner_val_indices": inner_val_idx, "test_indices": test_idx})
        log_line(
            out_dirs["text"],
            "Fold {fold} selesai | ROC-AUC={roc_auc:.4f} PR-AUC={average_precision:.4f} "
            "Accuracy={accuracy:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1_score:.4f} "
            "TP={TP} TN={TN} FP={FP} FN={FN}".format(**fold_row),
        )

    metrics_df = pd.DataFrame(fold_metrics).sort_values("fold").reset_index(drop=True)
    summary_rows = []
    for metric in ["roc_auc", "average_precision", "accuracy", "precision", "recall", "f1_score"]:
        summary_rows.append({"metric": metric, "mean": float(metrics_df[metric].mean()), "std": float(metrics_df[metric].std(ddof=1))})
    summary_df = pd.DataFrame(summary_rows)
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()

    metrics_df.to_csv(out_dirs["reports"] / "kfold_metrics.csv", index=False)
    metrics_df.to_csv(out_dirs["fold_reports"] / "kfold_metrics.csv", index=False)
    summary_df.to_csv(out_dirs["reports"] / "cv_summary.csv", index=False)
    if not history_df.empty:
        history_df.to_csv(out_dirs["reports"] / "all_history.csv", index=False)

    all_y_true = np.concatenate([np.asarray(item["y_true"], dtype=np.int64) for item in fold_predictions])
    all_y_score = np.concatenate([np.asarray(item["y_prob"], dtype=np.float32) for item in fold_predictions])
    recap_df = threshold_recap(all_y_true, all_y_score)
    recap_df.to_csv(out_dirs["reports"] / "threshold_recap_0_1_to_1_0.csv", index=False)
    recap_df.to_csv(out_dirs["fold_reports"] / "threshold_recap_0_1_to_1_0.csv", index=False)
    save_json(
        out_dirs["reports"] / "threshold_recap_0_1_to_1_0.json",
        {"thresholds": recap_df["threshold"].tolist(), "best_threshold": recap_df.loc[recap_df["f1_score"].idxmax()].to_dict()},
    )
    save_curve_plots(roc_entries, pr_entries, out_dirs)
    save_threshold_plot(recap_df, out_dirs)

    np.savez_compressed(out_dirs["predictions"] / "kfold_predictions.npz", folds=np.asarray(fold_predictions, dtype=object))
    np.savez_compressed(out_dirs["fold_root"] / "split_indices.npz", folds=np.asarray(fold_splits, dtype=object))

    dataset_summary = {
        "num_samples": schema_info.row_count,
        "num_positive": int(schema_info.class_distribution.get("1", 0)),
        "num_negative": int(schema_info.class_distribution.get("0", 0)),
        "feature_count": len(schema_info.feature_columns),
        "note": "Model 3 clean dataset is imbalanced after sequence-pair conflict removal.",
    }
    save_json(out_dirs["reports"] / "dataset_summary.json", dataset_summary)
    save_json(
        out_dirs["reports"] / "run_summary.json",
        {
            "run_name": args.run_name,
            "scenario": scenario,
            "threshold": float(args.threshold),
            "kfold_mean": summary_df.set_index("metric")["mean"].to_dict(),
            "kfold_std": summary_df.set_index("metric")["std"].to_dict(),
            "elapsed_seconds": float(time.perf_counter() - start),
        },
    )

    log_line(out_dirs["text"], "\nPer-fold metrics:")
    log_line(out_dirs["text"], metrics_df.to_string(index=False))
    log_line(out_dirs["text"], "\nCV summary:")
    log_line(out_dirs["text"], summary_df.to_string(index=False))
    log_line(out_dirs["text"], f"\nSelesai dalam {time.perf_counter() - start:.2f} detik. Output: {out_dirs['log_root']}")


if __name__ == "__main__":
    run_scenario("unseen_protein")
