import argparse
import copy
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def _load_base_module():
    module_path = Path(__file__).resolve().with_name("train_sing_es.py")
    spec = importlib.util.spec_from_file_location("model6_train_sing_es_base", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Gagal memuat base trainer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "model6_film_gat_af_sing_es"
ACTIVE_LOG_ROOT = PROJECT_ROOT / "results" / "benchmark" / "runs" / MODEL_NAME
DEFAULT_SPLIT_ROOT = PROJECT_ROOT.parent / "1-data" / "splits" / "unseen" / "graphs"
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
    path.write_text(json.dumps(base.to_serializable(payload), indent=2), encoding="utf-8")


def log_line(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def load_active_config() -> Dict[str, Any]:
    summary_path = ACTIVE_LOG_ROOT / "reports" / "run_summary.json"
    config_path = ACTIVE_LOG_ROOT / "reports" / "config.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = payload.get("final_candidate_config") or payload.get("config")
        if cfg:
            return dict(cfg)
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "Config aktif Model 6 tidak ditemukan. Dicari di "
        f"{summary_path} dan {config_path}"
    )


def parse_args(scenario: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Model 6 FiLM-GAT {SCENARIO_LABELS[scenario]} training."
    )
    parser.add_argument("--data-dir", type=Path, default=base.PREPROCESSED_DIR)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=SCENARIO_ROOTS[scenario])
    parser.add_argument("--inner-val-size", type=float, default=None)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", type=str, default=MODEL_NAME)
    return parser.parse_args()


def build_output_dirs(output_root: Path, run_name: str) -> Dict[str, Path]:
    log_root = ensure_dir(output_root / "b-log_result" / run_name)
    fold_root = ensure_dir(output_root / "a-fold" / run_name)
    model_root = ensure_dir(output_root / "c-model_save" / run_name)
    return {
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
        overlap = np.intersect1d(train_idx, test_idx)
        if overlap.size:
            raise ValueError(f"Train/test overlap pada {scenario} fold {fold_id}: {overlap[:5].tolist()}")
        folds.append({"fold": fold_id, "train_indices": train_idx, "test_indices": test_idx})
        fold_id += 1
        if max_folds > 0 and len(folds) >= max_folds:
            break
    if not folds:
        raise ValueError(f"Tidak ada fold yang bisa dimuat dari {scenario_dir}")
    return folds


def split_inner_train_val(train_indices: np.ndarray, labels: np.ndarray, cfg: Dict[str, Any], fold_id: int) -> Tuple[np.ndarray, np.ndarray]:
    val_size = float(cfg["inner_val_size"])
    if val_size <= 0.0:
        return train_indices, train_indices
    y = labels[train_indices]
    stratify = y if len(np.unique(y)) == 2 else None
    inner_train, inner_val = train_test_split(
        train_indices,
        test_size=val_size,
        random_state=int(cfg["random_seed"]) + int(fold_id),
        shuffle=True,
        stratify=stratify,
    )
    return np.asarray(inner_train, dtype=np.int64), np.asarray(inner_val, dtype=np.int64)


def make_loader(peptide_graphs, protein_graphs, labels, indices, metadata, cfg, global_features, shuffle: bool):
    return base.create_dataloader(
        peptide_graphs,
        protein_graphs,
        labels,
        indices,
        metadata,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["num_workers"]),
        global_features=global_features,
    )


def train_fold(
    fold_spec: Dict[str, np.ndarray],
    peptide_graphs,
    protein_graphs,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    global_raw: np.ndarray,
    cfg: Dict[str, Any],
    out_dirs: Dict[str, Path],
    log_path: Path,
    resume: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    fold_id = int(fold_spec["fold"])
    train_indices = np.asarray(fold_spec["train_indices"], dtype=np.int64)
    test_indices = np.asarray(fold_spec["test_indices"], dtype=np.int64)
    inner_train, inner_val = split_inner_train_val(train_indices, labels, cfg, fold_id)

    checkpoint_path = out_dirs["checkpoints"] / f"best_model_fold{fold_id}.pth"
    resume_path = out_dirs["resume"] / f"fold{fold_id}.pth"
    device = torch.device(str(cfg["device"]))
    global_features = base.normalize_global_features(global_raw, inner_train, str(cfg.get("global_mode", "none")))
    fold_cfg = dict(cfg)
    fold_cfg["global_dim"] = int(global_features.shape[1])
    fold_cfg["candidate_id"] = int(fold_cfg.get("candidate_id", 1))
    fold_cfg["candidate_name"] = str(fold_cfg.get("candidate_name", "active_config"))

    train_loader = make_loader(peptide_graphs, protein_graphs, labels, inner_train, metadata, fold_cfg, global_features, True)
    val_loader = make_loader(peptide_graphs, protein_graphs, labels, inner_val, metadata, fold_cfg, global_features, False)
    test_loader = make_loader(peptide_graphs, protein_graphs, labels, test_indices, metadata, fold_cfg, global_features, False)

    model = base.build_model(fold_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fold_cfg["lr"]),
        weight_decay=float(fold_cfg["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(fold_cfg["pos_weight"])], device=device),
        reduction="mean",
    )

    history = {
        "fold": fold_id,
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_roc_auc": [],
        "val_roc_auc": [],
        "train_pr_auc": [],
        "val_pr_auc": [],
    }
    best_state = None
    best_val_metrics = None
    best_epoch = 1
    start_epoch = 1
    epochs_without_improvement = 0
    best_monitor_score = float("-inf")

    if resume and resume_path.exists():
        payload = base.safe_torch_load(resume_path)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = payload.get("history", history)
        best_state = payload.get("best_state_dict")
        best_val_metrics = payload.get("best_val_metrics")
        best_epoch = int(payload.get("best_epoch", 1))
        best_monitor_score = float(payload.get("best_monitor_score", best_monitor_score))
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        start_epoch = int(payload.get("epoch", 0)) + 1
        log_line(log_path, f"[RESUME] Fold {fold_id} lanjut dari epoch {start_epoch}.")

    patience = int(fold_cfg.get("early_stopping_patience", 0))
    min_delta = float(fold_cfg.get("early_stopping_min_delta", 0.0))
    started = time.perf_counter()

    for epoch in range(start_epoch, int(fold_cfg["epochs"]) + 1):
        train_result = base.train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            grad_clip=float(fold_cfg["grad_clip"]),
            threshold=float(fold_cfg["threshold"]),
        )
        val_result = base.evaluate_one_epoch(
            model,
            val_loader,
            device,
            criterion,
            threshold=float(fold_cfg["threshold"]),
        )
        train_metrics = train_result["metrics"]
        val_metrics = val_result["metrics"]

        history["train_loss"].append(float(train_metrics["loss"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["train_accuracy"].append(float(train_metrics["accuracy"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["train_roc_auc"].append(float(train_metrics["roc_auc"]))
        history["val_roc_auc"].append(float(val_metrics["roc_auc"]))
        history["train_pr_auc"].append(float(train_metrics["pr_auc"]))
        history["val_pr_auc"].append(float(val_metrics["pr_auc"]))

        if base.should_update_best(val_metrics, best_val_metrics):
            best_val_metrics = dict(val_metrics)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch)

        monitor = base.safe_float(val_metrics.get("pr_auc", float("nan")))
        if not math.isnan(monitor) and monitor > best_monitor_score + min_delta:
            best_monitor_score = monitor
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % int(fold_cfg["log_every"]) == 0 or epoch == int(fold_cfg["epochs"]):
            log_line(
                log_path,
                f"[INFO] Fold {fold_id} | Epoch {epoch}/{fold_cfg['epochs']} | "
                f"train pr={train_metrics['pr_auc']:.4f} roc={train_metrics['roc_auc']:.4f} "
                f"f1={train_metrics['f1']:.4f} | "
                f"val pr={val_metrics['pr_auc']:.4f} roc={val_metrics['roc_auc']:.4f} "
                f"f1={val_metrics['f1']:.4f}",
            )

        base.safe_torch_save(
            resume_path,
            {
                "epoch": int(epoch),
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": copy.deepcopy(history),
                "best_state_dict": copy.deepcopy(best_state),
                "best_val_metrics": copy.deepcopy(best_val_metrics),
                "best_epoch": int(best_epoch),
                "best_monitor_score": float(best_monitor_score),
                "epochs_without_improvement": int(epochs_without_improvement),
                "config": copy.deepcopy(fold_cfg),
            },
        )

        if patience > 0 and epochs_without_improvement >= patience:
            log_line(
                log_path,
                f"[EARLY STOP] Fold {fold_id} berhenti epoch {epoch}; best_epoch={best_epoch}; best_val_pr={best_monitor_score:.4f}",
            )
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold_id} gagal mendapatkan best_state.")
    model.load_state_dict(best_state)
    test_result = base.evaluate_one_epoch(
        model,
        test_loader,
        device,
        criterion,
        threshold=float(fold_cfg["threshold"]),
    )
    test_metrics = dict(test_result["metrics"])
    confusion = base.binary_confusion_counts(test_result["y_true"], test_result["y_pred"])
    elapsed = time.perf_counter() - started

    base.safe_torch_save(
        checkpoint_path,
        {
            "model_state_dict": best_state,
            "config": fold_cfg,
            "fold": fold_id,
            "best_epoch": int(best_epoch),
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        },
    )
    pred_df = pd.DataFrame(
        {
            "row_index": test_result["sample_indices"].astype(int),
            "y_true": test_result["y_true"].astype(float),
            "y_prob": test_result["y_prob"].astype(float),
            "y_pred": test_result["y_pred"].astype(int),
        }
    )
    pred_df.to_csv(out_dirs["predictions"] / f"fold_{fold_id}_predictions.csv", index=False)

    metric_row = {
        "fold": fold_id,
        "train_size": int(len(inner_train)),
        "inner_val_size": int(len(inner_val)),
        "test_size": int(len(test_indices)),
        "best_epoch": int(best_epoch),
        "threshold": float(fold_cfg["threshold"]),
        "val_loss": float(test_metrics["loss"]),
        "val_accuracy": float(test_metrics["accuracy"]),
        "val_precision": float(test_metrics["precision"]),
        "val_recall": float(test_metrics["recall"]),
        "val_f1": float(test_metrics["f1"]),
        "val_roc_auc": float(test_metrics["roc_auc"]),
        "val_pr_auc": float(test_metrics["pr_auc"]),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "tp": int(confusion["TP"]),
        "tn": int(confusion["TN"]),
        "fp": int(confusion["FP"]),
        "fn": int(confusion["FN"]),
        "elapsed_seconds": float(elapsed),
    }
    fold_prediction = {
        "fold": fold_id,
        "sample_indices": test_result["sample_indices"],
        "y_true": test_result["y_true"],
        "y_prob": test_result["y_prob"],
        "y_pred": test_result["y_pred"],
    }
    log_line(
        log_path,
        f"[RESULT] Fold {fold_id} | ROC={metric_row['test_roc_auc']:.4f} "
        f"PR={metric_row['test_pr_auc']:.4f} Acc={metric_row['test_accuracy']:.4f} "
        f"Prec={metric_row['test_precision']:.4f} Rec={metric_row['test_recall']:.4f} "
        f"F1={metric_row['test_f1']:.4f}",
    )
    return metric_row, history, fold_prediction


def summarize_metrics(metrics_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    columns = {
        "ROC-AUC": "test_roc_auc",
        "PR-AUC": "test_pr_auc",
        "Accuracy": "test_accuracy",
        "Precision": "test_precision",
        "Recall": "test_recall",
        "F1": "test_f1",
    }
    return {
        label: {
            "mean": float(metrics_df[col].mean()),
            "std": float(metrics_df[col].std(ddof=0)),
        }
        for label, col in columns.items()
    }


def save_fold_predictions_npz(fold_predictions: List[Dict[str, Any]], out_path: Path) -> None:
    np.savez_compressed(
        out_path,
        folds=np.asarray([item["fold"] for item in fold_predictions], dtype=np.int64),
        sample_indices=np.asarray([item["sample_indices"] for item in fold_predictions], dtype=object),
        y_true=np.asarray([item["y_true"] for item in fold_predictions], dtype=object),
        y_prob=np.asarray([item["y_prob"] for item in fold_predictions], dtype=object),
        y_pred=np.asarray([item["y_pred"] for item in fold_predictions], dtype=object),
    )


def main(scenario: str) -> None:
    if scenario not in SCENARIO_ROOTS:
        raise ValueError(f"Scenario tidak dikenal: {scenario}")
    args = parse_args(scenario)
    start_time = time.perf_counter()

    cfg = load_active_config()
    cfg["inner_val_size"] = float(args.inner_val_size if args.inner_val_size is not None else cfg.get("test_size", 0.2))
    if int(args.epochs) > 0:
        cfg["epochs"] = int(args.epochs)
    if args.threshold is not None:
        cfg["threshold"] = float(args.threshold)
    if args.device.strip():
        cfg["device"] = args.device.strip()
    if args.random_seed is not None:
        cfg["random_seed"] = int(args.random_seed)
    cfg["num_workers"] = int(cfg.get("num_workers", 0))
    cfg["log_every"] = int(cfg.get("log_every", 1))
    cfg["scenario"] = scenario
    cfg["source_active_log_root"] = str(ACTIVE_LOG_ROOT)

    random.seed(int(cfg["random_seed"]))
    np.random.seed(int(cfg["random_seed"]))
    torch.manual_seed(int(cfg["random_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["random_seed"]))

    out_dirs = build_output_dirs(args.output_root, args.run_name)
    log_path = out_dirs["text"]
    if not args.resume and log_path.exists():
        log_path.unlink()

    log_line(log_path, f"[INFO] Scenario   : {SCENARIO_LABELS[scenario]}")
    log_line(log_path, f"[INFO] Data dir   : {args.data_dir}")
    log_line(log_path, f"[INFO] Split root : {args.split_root / scenario}")
    log_line(log_path, f"[INFO] Log root   : {out_dirs['log_root']}")
    log_line(log_path, f"[INFO] Fold root  : {out_dirs['fold_root']}")
    log_line(log_path, f"[INFO] Model root : {out_dirs['model_root']}")
    log_line(log_path, f"[INFO] Threshold  : {float(cfg['threshold'])}")

    metadata, peptide_graphs, protein_graphs, labels = base.load_preprocessed_data(args.data_dir)
    global_raw = base.build_global_feature_matrix(metadata)
    folds = load_unseen_folds(args.split_root, scenario, len(labels), int(args.max_folds))

    save_json(out_dirs["reports"] / "config.json", cfg)
    save_json(
        out_dirs["reports"] / "dataset_summary.json",
        {
            "num_samples": int(len(labels)),
            "num_positive": int(np.sum(labels == 1)),
            "num_negative": int(np.sum(labels == 0)),
            "scenario": scenario,
            "num_folds": int(len(folds)),
        },
    )
    np.savez_compressed(
        out_dirs["fold_root"] / "split_indices.npz",
        folds=np.asarray(folds, dtype=object),
    )

    all_metrics: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    fold_predictions: List[Dict[str, Any]] = []
    for fold_spec in folds:
        log_line(log_path, f"\n[INFO] Mulai fold {fold_spec['fold']}/{len(folds)}")
        metric_row, history, prediction = train_fold(
            fold_spec,
            peptide_graphs,
            protein_graphs,
            labels,
            metadata,
            global_raw,
            cfg,
            out_dirs,
            log_path,
            bool(args.resume),
        )
        all_metrics.append(metric_row)
        histories.append(history)
        fold_predictions.append(prediction)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dirs["reports"] / "kfold_metrics.csv", index=False)
    metrics_df.to_csv(out_dirs["fold_reports"] / "kfold_metrics.csv", index=False)
    save_json(out_dirs["logs"] / "kfold_histories.json", {"histories": histories})
    save_fold_predictions_npz(fold_predictions, out_dirs["predictions"] / "kfold_predictions.npz")
    base.plot_kfold_roc(fold_predictions, out_dirs["plots"] / "roc_auc_kfold_all.png")
    base.plot_kfold_pr(fold_predictions, out_dirs["plots"] / "pr_curve_kfold_all.png")
    base.plot_avg_loss_acc_kfold(histories, out_dirs["plots"] / "avg_loss_acc_kfold.png")

    summary = {
        "scenario": scenario,
        "config": cfg,
        "kfold_mean": {
            "val_accuracy": float(metrics_df["val_accuracy"].mean()),
            "val_precision": float(metrics_df["val_precision"].mean()),
            "val_recall": float(metrics_df["val_recall"].mean()),
            "val_f1": float(metrics_df["val_f1"].mean()),
            "val_roc_auc": float(metrics_df["val_roc_auc"].mean()),
            "val_pr_auc": float(metrics_df["val_pr_auc"].mean()),
        },
        "metrics": summarize_metrics(metrics_df),
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    save_json(out_dirs["reports"] / "run_summary.json", summary)

    agg = summarize_metrics(metrics_df)
    log_line(log_path, "\n[SUMMARY] Aggregated unseen metrics")
    for key in ["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1"]:
        log_line(log_path, f"{key}: {agg[key]['mean']:.3f}+/-{agg[key]['std']:.3f}")
    log_line(log_path, f"Time: {summary['elapsed_seconds']:.2f}s")
    log_line(log_path, "[INFO] Selesai.")



if __name__ == "__main__":
    main("unseen_protein")
