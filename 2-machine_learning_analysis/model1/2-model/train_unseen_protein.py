#!/usr/bin/env python3
"""Model 1 CAMP training for sequence-level unseen splits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def _load_base_module():
    module_path = Path(__file__).resolve().with_name("camp_sing_regularized.py")
    spec = importlib.util.spec_from_file_location("model1_camp_sing_reg_base", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Gagal memuat base trainer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "model1_sing_reg"
DEFAULT_CSV = PROJECT_ROOT.parent / "1-data" / "features" / "model1" / "0-neg" / "gan_decoy_sequence_clean.csv"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT.parent / "1-data" / "splits" / "unseen" / "model1"
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base._to_serializable(payload), indent=2), encoding="utf-8")


def log_line(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def parse_args(scenario: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Model 1 CAMP {SCENARIO_LABELS[scenario]} training.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=SCENARIO_ROOTS[scenario])
    parser.add_argument("--inner-val-size", type=float, default=0.20)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", type=str, default=MODEL_NAME)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Any:
    cfg = base.Config(
        csv_path=args.csv_path,
        output_dir=args.output_root / "b-log_result" / args.run_name,
        run_name=args.run_name,
    )
    if args.epochs > 0:
        cfg.epochs = int(args.epochs)
    if args.batch_size > 0:
        cfg.batch_size = int(args.batch_size)
    if args.threshold is not None:
        cfg.threshold = float(args.threshold)
    if args.device.strip():
        cfg.device = args.device.strip()
    if args.random_state is not None:
        cfg.random_state = int(args.random_state)
    cfg.validate()
    return cfg


def build_output_dirs(output_root: Path, run_name: str) -> Dict[str, Path]:
    log_root = ensure_dir(output_root / "b-log_result" / run_name)
    fold_root = ensure_dir(output_root / "a-fold" / run_name)
    model_root = ensure_dir(output_root / "c-model_save" / run_name)
    return {
        "run_name": run_name,
        "log_root": log_root,
        "fold_root": fold_root,
        "model_root": model_root,
        "reports": ensure_dir(log_root / "reports"),
        "logs": ensure_dir(log_root / "logs"),
        "plots": ensure_dir(log_root / "plots"),
        "predictions": ensure_dir(fold_root / "predictions"),
        "fold_reports": ensure_dir(fold_root / "reports"),
        "checkpoints": ensure_dir(model_root / "checkpoints"),
        "resume": ensure_dir(model_root / "resume"),
        "text": log_root / f"{run_name}_results_rki.txt",
        "state": log_root / f"{run_name}_resume_rki.json",
    }


def read_indices(path: Path, n_samples: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Split index tidak ditemukan: {path}")
    df = pd.read_csv(path)
    if "row_index" not in df.columns:
        raise ValueError(f"Kolom row_index tidak ditemukan pada {path}")
    values = df["row_index"].to_numpy(dtype=np.int64)
    if values.size == 0:
        raise ValueError(f"Split index kosong: {path}")
    if values.min() < 0 or values.max() >= n_samples:
        raise ValueError(f"Index di luar rentang dataset pada {path}")
    return values


def load_unseen_folds(split_root: Path, scenario: str, n_samples: int, max_folds: int) -> List[Dict[str, np.ndarray]]:
    scenario_dir = split_root / scenario
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Folder split scenario tidak ditemukan: {scenario_dir}")
    folds: List[Dict[str, np.ndarray]] = []
    fold_id = 1
    while True:
        train_path = scenario_dir / f"fold_{fold_id}_train_idx.csv"
        test_path = scenario_dir / f"fold_{fold_id}_test_idx.csv"
        if not train_path.exists() and not test_path.exists():
            break
        train_idx = read_indices(train_path, n_samples)
        test_idx = read_indices(test_path, n_samples)
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"Train/test overlap pada fold {fold_id}")
        folds.append({"fold": fold_id, "train_indices": train_idx, "test_indices": test_idx})
        if max_folds and len(folds) >= max_folds:
            break
        fold_id += 1
    if not folds:
        raise ValueError(f"Tidak ada fold ditemukan untuk {scenario_dir}")
    return folds


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    X_tensor, y_tensor = base.tensorize_dataset(X, y.astype(np.float32))
    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_fold(
    fold: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    config: Any,
    out_dirs: Dict[str, Path],
    log_path: Path,
    inner_val_size: float,
    config_signature: str,
) -> Dict[str, Any]:
    fold_id = int(fold["fold"])
    checkpoint_path = out_dirs["checkpoints"] / f"camp_model_fold{fold_id}.pth"
    prediction_path = out_dirs["predictions"] / f"fold_{fold_id}_predictions.csv"
    runtime = base._resolve_mode_config(config)

    if checkpoint_path.exists() and prediction_path.exists():
        pred_df = pd.read_csv(prediction_path)
        y_true = pred_df["y_true"].to_numpy(dtype=np.float32)
        y_prob = pred_df["y_prob"].to_numpy(dtype=np.float32)
        y_pred = pred_df["y_pred"].to_numpy(dtype=np.int64)
        metric = base._compute_binary_metrics(y_true, y_prob, config.threshold)
        metric["fold"] = fold_id
        log_line(log_path, f"[RESUME] Fold {fold_id} dilewati, artefak sudah ada.")
        return {
            "metric": metric,
            "history": pd.DataFrame(),
            "prediction": pred_df,
            "roc_curve": _roc_curve_dict(fold_id, y_true, y_prob, metric),
            "pr_curve": _pr_curve_dict(fold_id, y_true, y_prob, metric),
        }

    train_indices = fold["train_indices"]
    test_indices = fold["test_indices"]
    inner_train_idx, inner_val_idx = train_test_split(
        train_indices,
        test_size=inner_val_size,
        stratify=y[train_indices].astype(int),
        random_state=config.random_state + fold_id,
    )

    X_train = X[inner_train_idx]
    y_train = y[inner_train_idx]
    X_val = X[inner_val_idx]
    y_val = y[inner_val_idx]
    X_test = X[test_indices]
    y_test = y[test_indices]

    log_line(
        log_path,
        f"[INFO] Fold {fold_id}: train={len(inner_train_idx)} val={len(inner_val_idx)} test={len(test_indices)}",
    )

    device = torch.device(config.device)
    criterion = nn.BCELoss()
    train_loader = make_loader(X_train, y_train, runtime["batch_size"], True, config.num_workers)
    val_loader = make_loader(X_val, y_val, runtime["batch_size"], False, config.num_workers)
    test_loader = make_loader(X_test, y_test, runtime["batch_size"], False, config.num_workers)

    model = base.CAMPModel(
        input_dim=X.shape[1],
        dropout_rate=config.dropout_rate,
        conv1_channels=config.conv1_channels,
        conv2_channels=config.conv2_channels,
        dense_hidden=config.dense_hidden,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=runtime["learning_rate"], weight_decay=config.weight_decay)

    best_state = deepcopy(model.state_dict())
    best_metric: Dict[str, Any] | None = None
    best_score: float | None = None
    best_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history_rows: List[Dict[str, Any]] = []

    for epoch in range(1, runtime["epochs"] + 1):
        train_stats = base.train_one_epoch(model, train_loader, criterion, optimizer, device, config.threshold)
        val_stats = base.evaluate_one_epoch(model, val_loader, criterion, device, config.threshold)
        history_rows.append(
            {
                "fold": fold_id,
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
        )
        monitor_score = base._early_stopping_score(val_stats, config.early_stopping_monitor)
        if base._is_early_stopping_improvement(
            monitor_score,
            best_score,
            mode=config.early_stopping_mode,
            min_delta=config.early_stopping_min_delta,
        ):
            best_score = monitor_score
            best_state = deepcopy(model.state_dict())
            best_metric = deepcopy(val_stats)
            best_epoch = epoch
            best_val_loss = float(val_stats["loss"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        log_line(
            log_path,
            "Fold {fold} | Epoch {epoch}/{epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val AUC {val_auc}".format(
                fold=fold_id,
                epoch=epoch,
                epochs=runtime["epochs"],
                train_loss=train_stats["loss"],
                val_loss=val_stats["loss"],
                val_auc="{:.4f}".format(val_stats["roc_auc"]) if val_stats["roc_auc"] is not None else "NA",
            ),
        )

        if (
            config.early_stopping
            and config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            log_line(log_path, f"[EARLY STOP] Fold {fold_id} berhenti di epoch {epoch} | best_epoch={best_epoch}")
            break

    if best_metric is None:
        raise RuntimeError(f"Fold {fold_id} tidak menghasilkan validasi terbaik.")

    model.load_state_dict(best_state)
    test_stats = base.evaluate_one_epoch(model, test_loader, criterion, device, config.threshold)
    y_true = np.asarray(test_stats["labels"], dtype=np.float32)
    y_prob = np.asarray(test_stats["probs"], dtype=np.float32)
    y_pred = (y_prob >= float(config.threshold)).astype(np.int64)
    metric = base._compute_binary_metrics(y_true, y_prob, config.threshold)
    metric.update(
        {
            "fold": fold_id,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_monitor_score": best_score,
            "elapsed_seconds": float(sum(0 for _ in history_rows)),
        }
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "fold": fold_id,
            "metric": base._to_serializable(metric),
            "config_signature": config_signature,
        },
        checkpoint_path,
    )

    pred_df = metadata.iloc[test_indices].reset_index(drop=True)
    pred_df["sample_indices"] = test_indices.astype(np.int64)
    pred_df["y_true"] = y_true.astype(np.int64)
    pred_df["y_prob"] = y_prob.astype(np.float32)
    pred_df["y_pred"] = y_pred.astype(np.int64)
    pred_df.to_csv(prediction_path, index=False)
    np.savez_compressed(
        out_dirs["predictions"] / f"fold_{fold_id}_predictions.npz",
        sample_indices=test_indices.astype(np.int64),
        y_true=y_true.astype(np.int64),
        y_prob=y_prob.astype(np.float32),
        y_pred=y_pred.astype(np.int64),
    )

    log_line(
        log_path,
        "[INFO] Fold {} selesai | ROC-AUC={} PR-AUC={} Acc={:.4f} Prec={:.4f} Recall={:.4f} F1={:.4f}".format(
            fold_id,
            "{:.4f}".format(metric["roc_auc"]) if metric["roc_auc"] is not None else "NA",
            "{:.4f}".format(metric["pr_auc"]) if metric["pr_auc"] is not None else "NA",
            metric["accuracy"],
            metric["precision"],
            metric["recall"],
            metric["f1"],
        ),
    )
    return {
        "metric": metric,
        "history": pd.DataFrame(history_rows),
        "prediction": pred_df,
        "roc_curve": _roc_curve_dict(fold_id, y_true, y_prob, metric),
        "pr_curve": _pr_curve_dict(fold_id, y_true, y_prob, metric),
    }


def _roc_curve_dict(fold: int, y_true: np.ndarray, y_prob: np.ndarray, metric: Dict[str, Any]) -> Dict[str, Any]:
    if metric.get("roc_auc") is None:
        return {"fold": fold, "fpr": np.array([]), "tpr": np.array([]), "roc_auc": None}
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {"fold": fold, "fpr": fpr, "tpr": tpr, "roc_auc": float(metric["roc_auc"])}


def _pr_curve_dict(fold: int, y_true: np.ndarray, y_prob: np.ndarray, metric: Dict[str, Any]) -> Dict[str, Any]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {"fold": fold, "precision": precision, "recall": recall, "pr_auc": metric.get("pr_auc")}


def save_outputs(
    results: List[Dict[str, Any]],
    folds: List[Dict[str, np.ndarray]],
    out_dirs: Dict[str, Path],
    scenario: str,
    config: Any,
    start_time: float,
) -> None:
    metrics_df = pd.DataFrame([item["metric"] for item in results]).sort_values("fold").reset_index(drop=True)
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    summary_df = pd.DataFrame(
        [{"metric": col, "mean": float(metrics_df[col].mean()), "std": float(metrics_df[col].std(ddof=1))} for col in metric_columns]
    )
    history_df = pd.concat([item["history"] for item in results if not item["history"].empty], ignore_index=True) if results else pd.DataFrame()
    predictions_df = pd.concat([item["prediction"] for item in results], ignore_index=True)

    metrics_df.to_csv(out_dirs["reports"] / "kfold_metrics.csv", index=False)
    metrics_df.to_csv(out_dirs["fold_reports"] / "kfold_metrics.csv", index=False)
    summary_df.to_csv(out_dirs["reports"] / "cv_summary.csv", index=False)
    history_df.to_csv(out_dirs["reports"] / "fold_history.csv", index=False)
    predictions_df.to_csv(out_dirs["predictions"] / "kfold_predictions.csv", index=False)

    np.savez_compressed(
        out_dirs["fold_root"] / "split_indices.npz",
        folds=np.asarray(
            [
                {
                    "fold": int(item["fold"]),
                    "train_indices": item["train_indices"].astype(np.int64),
                    "test_indices": item["test_indices"].astype(np.int64),
                }
                for item in folds
            ],
            dtype=object,
        ),
    )
    np.savez_compressed(
        out_dirs["predictions"] / "kfold_predictions.npz",
        sample_indices=predictions_df["sample_indices"].to_numpy(dtype=np.int64),
        y_true=predictions_df["y_true"].to_numpy(dtype=np.int64),
        y_prob=predictions_df["y_prob"].to_numpy(dtype=np.float32),
        y_pred=predictions_df["y_pred"].to_numpy(dtype=np.int64),
    )

    plt.figure(figsize=(8, 6))
    for curve in [item["roc_curve"] for item in results]:
        if curve["roc_auc"] is None:
            continue
        plt.plot(curve["fpr"], curve["tpr"], label=f"Fold {curve['fold']} (AUC={curve['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title(f"ROC Curve - {SCENARIO_LABELS[scenario]}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "roc_auc_kfold_all.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    for curve in [item["pr_curve"] for item in results]:
        label = f"Fold {curve['fold']}"
        if curve["pr_auc"] is not None:
            label += f" (AP={curve['pr_auc']:.4f})"
        plt.plot(curve["recall"], curve["precision"], label=label)
    plt.title(f"Precision-Recall Curve - {SCENARIO_LABELS[scenario]}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "pr_curve_kfold_all.png", dpi=150)
    plt.close()

    if not history_df.empty:
        plt.figure(figsize=(10, 6))
        for fold, sub in history_df.groupby("fold"):
            plt.plot(sub["epoch"], sub["val_loss"], label=f"Fold {fold}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dirs["plots"] / "avg_loss_acc_kfold.png", dpi=150)
        plt.close()

    cm = confusion_matrix(predictions_df["y_true"].astype(int), predictions_df["y_pred"].astype(int), labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(f"OOF Confusion Matrix - {SCENARIO_LABELS[scenario]}")
    fig.tight_layout()
    fig.savefig(out_dirs["plots"] / "oof_confusion_matrix.png", dpi=150)
    plt.close(fig)

    elapsed = time.perf_counter() - start_time
    save_json(
        out_dirs["reports"] / "run_summary.json",
        {
            "run_name": out_dirs["run_name"],
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "threshold": float(config.threshold),
            "elapsed_seconds": float(elapsed),
            "kfold_mean": summary_df.set_index("metric")["mean"].to_dict(),
            "kfold_std": summary_df.set_index("metric")["std"].to_dict(),
        },
    )
    save_json(
        out_dirs["reports"] / "config.json",
        {
            "csv_path": str(config.csv_path),
            "split_root": str(DEFAULT_SPLIT_ROOT),
            "scenario": scenario,
            "random_state": int(config.random_state),
            "n_splits": len(folds),
            "batch_size": int(config.batch_size),
            "learning_rate": float(config.learning_rate),
            "epochs": int(config.epochs),
            "dropout_rate": float(config.dropout_rate),
            "weight_decay": float(config.weight_decay),
            "conv1_channels": int(config.conv1_channels),
            "conv2_channels": int(config.conv2_channels),
            "dense_hidden": int(config.dense_hidden),
            "threshold": float(config.threshold),
            "early_stopping": bool(config.early_stopping),
            "early_stopping_monitor": config.early_stopping_monitor,
            "early_stopping_patience": int(config.early_stopping_patience),
        },
    )

    agg = summary_df.set_index("metric")["mean"].to_dict()
    std = summary_df.set_index("metric")["std"].to_dict()
    text = "\n".join(
        [
            f"Model 1 CAMP - {SCENARIO_LABELS[scenario]}",
            f"Output: {out_dirs['log_root']}",
            "",
            metrics_df[["fold", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]].to_string(index=False),
            "",
            "Aggregated:",
            "ROC-AUC={:.3f}±{:.3f} | PR-AUC={:.3f}±{:.3f} | Accuracy={:.3f}±{:.3f} | Precision={:.3f}±{:.3f} | Recall={:.3f}±{:.3f} | F1={:.3f}±{:.3f}".format(
                agg["roc_auc"],
                std["roc_auc"],
                agg["pr_auc"],
                std["pr_auc"],
                agg["accuracy"],
                std["accuracy"],
                agg["precision"],
                std["precision"],
                agg["recall"],
                std["recall"],
                agg["f1"],
                std["f1"],
            ),
            f"Time={elapsed:.2f}s",
            "",
        ]
    )
    out_dirs["text"].write_text(text, encoding="utf-8")


def main(scenario: str) -> None:
    if scenario not in SCENARIO_ROOTS:
        raise ValueError(f"Scenario tidak dikenal: {scenario}")
    start_time = time.perf_counter()
    args = parse_args(scenario)
    config = build_config(args)
    out_dirs = build_output_dirs(args.output_root, args.run_name)
    log_path = out_dirs["logs"] / "run.log"

    base.random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    log_line(log_path, f"[INFO] Scenario   : {SCENARIO_LABELS[scenario]}")
    log_line(log_path, f"[INFO] Data CSV   : {config.csv_path}")
    log_line(log_path, f"[INFO] Split root : {args.split_root}")
    log_line(log_path, f"[INFO] Log root   : {out_dirs['log_root']}")
    log_line(log_path, f"[INFO] Fold root  : {out_dirs['fold_root']}")
    log_line(log_path, f"[INFO] Model root : {out_dirs['model_root']}")
    log_line(log_path, f"[INFO] Threshold  : {config.threshold}")
    log_line(log_path, f"[INFO] Device     : {config.device}")

    df = base.load_dataset(config.csv_path)
    schema_info = base.validate_schema(df, config.target_col)
    block_columns = base.get_feature_block_columns(df)
    X, y, metadata = base.build_feature_matrix(df, block_columns, config.target_col)
    folds = load_unseen_folds(args.split_root, scenario, len(df), args.max_folds)
    config_signature = base.compute_config_signature({"config": base._to_serializable(config.__dict__), "scenario": scenario})

    save_json(
        out_dirs["reports"] / "dataset_summary.json",
        {
            "num_samples": int(len(df)),
            "num_positive": int(np.sum(y == 1)),
            "num_negative": int(np.sum(y == 0)),
            "schema_info": schema_info,
        },
    )
    log_line(log_path, f"[INFO] Folds      : {len(folds)}")

    results: List[Dict[str, Any]] = []
    for fold in folds:
        log_line(log_path, f"\n[INFO] Mulai fold {fold['fold']}/{len(folds)}")
        results.append(
            train_fold(
                fold=fold,
                X=X,
                y=y,
                metadata=metadata,
                config=config,
                out_dirs=out_dirs,
                log_path=log_path,
                inner_val_size=float(args.inner_val_size),
                config_signature=config_signature,
            )
        )

    save_outputs(results, folds, out_dirs, scenario, config, start_time)
    print(out_dirs["text"].read_text(encoding="utf-8"), end="")
    log_line(log_path, "[INFO] Pipeline selesai.")


if __name__ == "__main__":
    main("unseen_protein")
