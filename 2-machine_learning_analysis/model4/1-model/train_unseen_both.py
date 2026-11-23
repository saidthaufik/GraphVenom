import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


def _load_base_module():
    module_path = Path(__file__).resolve().with_name("saednn_sing_reg.py")
    spec = importlib.util.spec_from_file_location("model4_saednn_sing_reg_base", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Gagal memuat base trainer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "model4_sing_reg"
DEFAULT_SPLIT_ROOT = PROJECT_ROOT.parent / "1-data" / "splits" / "unseen" / "graphs"
CANONICAL_METADATA = PROJECT_ROOT / "model5" / "0-feat" / "processed_gat" / "metadata.csv"
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
    parser = argparse.ArgumentParser(description=f"Model 4 SAE-DNN {SCENARIO_LABELS[scenario]} training.")
    parser.add_argument("--input-csv", type=Path, default=Path(base.INPUT_CSV))
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=SCENARIO_ROOTS[scenario])
    parser.add_argument("--inner-val-size", type=float, default=0.20)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--sae-epochs", type=int, default=0)
    parser.add_argument("--dnn-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", type=str, default=MODEL_NAME)
    return parser.parse_args()


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
            raise ValueError(f"Train/test overlap pada fold {fold_id}: {overlap[:10].tolist()}")
        folds.append({"fold": fold_id, "train_indices": train_idx, "test_indices": test_idx})
        if max_folds and len(folds) >= max_folds:
            break
        fold_id += 1
    if not folds:
        raise ValueError(f"Tidak ada fold ditemukan untuk {scenario_dir}")
    return folds


def assert_model4_matches_canonical(df: pd.DataFrame) -> None:
    if not CANONICAL_METADATA.exists():
        raise FileNotFoundError(f"Metadata canonical tidak ditemukan: {CANONICAL_METADATA}")
    canonical = pd.read_csv(CANONICAL_METADATA, usecols=["pep_id", "uniprot_id", "label"])
    if len(df) != len(canonical):
        raise ValueError(f"Row Model 4 ({len(df)}) tidak sama dengan canonical ({len(canonical)}).")
    same_pep = df["pep_id"].astype(str).to_numpy() == canonical["pep_id"].astype(str).to_numpy()
    same_prot = df["Uniprot_id"].astype(str).to_numpy() == canonical["uniprot_id"].astype(str).to_numpy()
    same_label = df[base.TARGET_COLUMN].astype(int).to_numpy() == canonical["label"].astype(int).to_numpy()
    if not (same_pep.all() and same_prot.all() and same_label.all()):
        mismatch = int((~(same_pep & same_prot & same_label)).sum())
        raise ValueError(f"Model 4 tidak row-wise identik dengan canonical split source. mismatch={mismatch}")


def train_fold(
    fold: Dict[str, np.ndarray],
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    out_dirs: Dict[str, Path],
    log_path: Path,
    inner_val_size: float,
    sae_epochs: int,
    dnn_epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    fold_id = int(fold["fold"])
    checkpoint_path = out_dirs["checkpoints"] / f"esm2_model_fold{fold_id}.keras"
    prediction_path = out_dirs["predictions"] / f"fold_{fold_id}_predictions.csv"
    state_path = out_dirs["state"]

    if checkpoint_path.exists() and prediction_path.exists():
        pred_df = pd.read_csv(prediction_path)
        y_true = pred_df["y_true"].to_numpy(dtype=np.int64)
        y_prob = pred_df["y_prob"].to_numpy(dtype=np.float32)
        y_pred = pred_df["y_pred"].to_numpy(dtype=np.int64)
        metric = base._compute_binary_metrics(y_true, y_prob)
        metric["fold"] = fold_id
        log_line(log_path, f"[RESUME] Fold {fold_id} dilewati, artefak sudah ada.")
        return {
            "metric": metric,
            "history": pd.DataFrame(),
            "prediction": pred_df,
            "roc_curve": _roc_curve_dict(fold_id, y_true, y_prob, metric),
            "pr_curve": _pr_curve_dict(fold_id, y_true, y_prob, metric),
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }

    train_indices = fold["train_indices"]
    test_indices = fold["test_indices"]
    inner_train_idx, inner_val_idx = train_test_split(
        train_indices,
        test_size=inner_val_size,
        stratify=y.iloc[train_indices],
        random_state=base.RANDOM_STATE + fold_id,
    )

    X_train = X.iloc[inner_train_idx].to_numpy(dtype=np.float32)
    y_train = y.iloc[inner_train_idx].to_numpy(dtype=np.float32)
    X_val = X.iloc[inner_val_idx].to_numpy(dtype=np.float32)
    y_val = y.iloc[inner_val_idx].to_numpy(dtype=np.float32)
    X_test = X.iloc[test_indices].to_numpy(dtype=np.float32)
    y_test = y.iloc[test_indices].to_numpy(dtype=np.int32)

    log_line(
        log_path,
        f"[INFO] Fold {fold_id}: train={len(inner_train_idx)} val={len(inner_val_idx)} test={len(test_indices)}",
    )
    sae_weights = base.build_sae(
        X_train=X_train,
        X_val=X_val,
        logger=base.logging.getLogger("model4_unseen"),
        epochs=sae_epochs,
        batch_size=batch_size,
        hl_node=base.HIDDEN_NODES,
        lr=base.LEARNING_RATE,
        af=base.ACTIVATION,
        num_layers=base.NUM_LAYERS,
        dropout_rate=base.DROPOUT_RATE,
        feature_reduction=base.FEATURE_REDUCTION,
    )
    model = base.build_model(
        input_dim=X_train.shape[1],
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
        ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=0),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=dnn_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    saved_model = load_model(checkpoint_path)
    y_prob = saved_model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(np.int64)
    metric = base._compute_binary_metrics(y_test, y_prob)
    metric["fold"] = fold_id

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = np.arange(1, len(history_df) + 1)
    history_df["fold"] = fold_id

    pred_df = metadata.iloc[test_indices].reset_index(drop=True)
    pred_df["sample_indices"] = test_indices.astype(np.int64)
    pred_df["y_true"] = y_test.astype(np.int64)
    pred_df["y_prob"] = y_prob.astype(np.float32)
    pred_df["y_pred"] = y_pred.astype(np.int64)
    pred_df.to_csv(prediction_path, index=False)
    np.savez_compressed(
        out_dirs["predictions"] / f"fold_{fold_id}_predictions.npz",
        sample_indices=test_indices.astype(np.int64),
        y_true=y_test.astype(np.int64),
        y_prob=y_prob.astype(np.float32),
        y_pred=y_pred.astype(np.int64),
    )
    save_json(
        state_path,
        {
            "last_completed_fold": fold_id,
            "checkpoint": str(checkpoint_path),
            "metric": metric,
        },
    )
    log_line(
        log_path,
        "[INFO] Fold {} selesai | ROC-AUC={:.4f} PR-AUC={:.4f} Acc={:.4f} Prec={:.4f} Recall={:.4f} F1={:.4f}".format(
            fold_id,
            metric["roc_auc"],
            metric["average_precision"],
            metric["accuracy"],
            metric["precision"],
            metric["recall"],
            metric["f1_score"],
        ),
    )
    return {
        "metric": metric,
        "history": history_df,
        "prediction": pred_df,
        "roc_curve": _roc_curve_dict(fold_id, y_test, y_prob, metric),
        "pr_curve": _pr_curve_dict(fold_id, y_test, y_prob, metric),
        "y_true": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def _roc_curve_dict(fold: int, y_true: np.ndarray, y_prob: np.ndarray, metric: Dict[str, Any]) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {"fold": fold, "fpr": fpr, "tpr": tpr, "roc_auc": float(metric["roc_auc"])}


def _pr_curve_dict(fold: int, y_true: np.ndarray, y_prob: np.ndarray, metric: Dict[str, Any]) -> Dict[str, Any]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {"fold": fold, "precision": precision, "recall": recall, "average_precision": float(metric["average_precision"])}


def save_outputs(
    results: List[Dict[str, Any]],
    folds: List[Dict[str, np.ndarray]],
    out_dirs: Dict[str, Path],
    scenario: str,
    start_time: float,
) -> None:
    metrics_df = pd.DataFrame([item["metric"] for item in results]).sort_values("fold").reset_index(drop=True)
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"]
    summary_df = pd.DataFrame(
        [{"metric": col, "mean": float(metrics_df[col].mean()), "std": float(metrics_df[col].std(ddof=1))} for col in metric_columns]
    )
    history_df = pd.concat([item["history"] for item in results if not item["history"].empty], ignore_index=True) if results else pd.DataFrame()
    predictions_df = pd.concat([item["prediction"] for item in results], ignore_index=True)

    metrics_df.to_csv(out_dirs["reports"] / "kfold_metrics.csv", index=False)
    metrics_df.to_csv(out_dirs["fold_reports"] / "kfold_metrics.csv", index=False)
    summary_df.to_csv(out_dirs["reports"] / "cv_summary.csv", index=False)
    history_df.to_csv(out_dirs["reports"] / "all_history.csv", index=False)
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
        plt.plot(curve["fpr"], curve["tpr"], label=f"Fold {curve['fold']} (AUC={curve['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
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
        plt.plot(curve["recall"], curve["precision"], label=f"Fold {curve['fold']} (AP={curve['average_precision']:.4f})")
    plt.title(f"Precision-Recall Curve - {SCENARIO_LABELS[scenario]}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "pr_curve_kfold_all.png", dpi=150)
    plt.close()

    if not history_df.empty:
        mean_history = history_df.groupby("epoch")[["loss", "val_loss", "accuracy", "val_accuracy"]].mean().reset_index()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(mean_history["epoch"], mean_history["accuracy"], label="Train Accuracy")
        plt.plot(mean_history["epoch"], mean_history["val_accuracy"], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(mean_history["epoch"], mean_history["loss"], label="Train Loss")
        plt.plot(mean_history["epoch"], mean_history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dirs["plots"] / "avg_loss_acc_kfold.png", dpi=150)
        plt.close()

    y_true_all = predictions_df["y_true"].to_numpy(dtype=np.int64)
    y_pred_all = predictions_df["y_pred"].to_numpy(dtype=np.int64)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(f"OOF Confusion Matrix - {SCENARIO_LABELS[scenario]}")
    fig.tight_layout()
    fig.savefig(out_dirs["plots"] / "oof_confusion_matrix.png", dpi=150)
    plt.close(fig)

    elapsed = time.perf_counter() - start_time
    summary_payload = {
        "run_name": out_dirs["run_name"],
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "threshold": 0.5,
        "elapsed_seconds": float(elapsed),
        "kfold_mean": summary_df.set_index("metric")["mean"].to_dict(),
        "kfold_std": summary_df.set_index("metric")["std"].to_dict(),
    }
    save_json(out_dirs["reports"] / "run_summary.json", summary_payload)
    save_json(
        out_dirs["reports"] / "config.json",
        {
            "input_csv": base.INPUT_CSV,
            "split_root": str(DEFAULT_SPLIT_ROOT),
            "scenario": scenario,
            "random_state": base.RANDOM_STATE,
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
        },
    )

    table = metrics_df.rename(
        columns={
            "roc_auc": "ROC-AUC",
            "average_precision": "PR-AUC",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1",
        }
    )[["fold", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1"]]
    agg = summary_payload["kfold_mean"]
    std = summary_payload["kfold_std"]
    text = "\n".join(
        [
            f"Model 4 SAE-DNN - {SCENARIO_LABELS[scenario]}",
            f"Output: {out_dirs['log_root']}",
            "",
            table.to_string(index=False),
            "",
            "Aggregated:",
            "ROC-AUC={:.3f}±{:.3f} | PR-AUC={:.3f}±{:.3f} | Accuracy={:.3f}±{:.3f} | Precision={:.3f}±{:.3f} | Recall={:.3f}±{:.3f} | F1={:.3f}±{:.3f}".format(
                agg["roc_auc"],
                std["roc_auc"],
                agg["average_precision"],
                std["average_precision"],
                agg["accuracy"],
                std["accuracy"],
                agg["precision"],
                std["precision"],
                agg["recall"],
                std["recall"],
                agg["f1_score"],
                std["f1_score"],
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
    if args.sae_epochs > 0:
        base.SAE_EPOCHS = int(args.sae_epochs)
    if args.dnn_epochs > 0:
        base.DNN_EPOCHS = int(args.dnn_epochs)
    if args.batch_size > 0:
        base.BATCH_SIZE = int(args.batch_size)
    base.INPUT_CSV = str(args.input_csv)

    out_dirs = build_output_dirs(args.output_root, args.run_name)
    environment_info = base.setup_environment({key: str(value) for key, value in out_dirs.items() if isinstance(value, Path)})
    logger = base.setup_logging(str(out_dirs["logs"] / "run.log"))
    base.logging.getLogger("model4_unseen").handlers = logger.handlers
    base.logging.getLogger("model4_unseen").setLevel(logger.level)
    log_path = out_dirs["logs"] / "run.log"

    log_line(log_path, f"[INFO] Scenario   : {SCENARIO_LABELS[scenario]}")
    log_line(log_path, f"[INFO] Input CSV  : {base.INPUT_CSV}")
    log_line(log_path, f"[INFO] Split root : {args.split_root}")
    log_line(log_path, f"[INFO] Log root   : {out_dirs['log_root']}")
    log_line(log_path, f"[INFO] Fold root  : {out_dirs['fold_root']}")
    log_line(log_path, f"[INFO] Model root : {out_dirs['model_root']}")
    log_line(log_path, "[INFO] Threshold  : 0.5")

    df = base.load_data()
    assert_model4_matches_canonical(df)
    X, y, metadata, schema_info, feature_info = base.prepare_features(df)
    folds = load_unseen_folds(args.split_root, scenario, len(df), args.max_folds)
    log_line(log_path, f"[INFO] Folds      : {len(folds)}")
    save_json(
        out_dirs["reports"] / "dataset_summary.json",
        {
            "num_samples": int(schema_info.row_count),
            "num_positive": int(schema_info.class_distribution.get("1", 0)),
            "num_negative": int(schema_info.class_distribution.get("0", 0)),
            "feature_count": len(schema_info.feature_columns),
            "environment_info": environment_info,
        },
    )

    results: List[Dict[str, Any]] = []
    for fold in folds:
        log_line(log_path, f"\n[INFO] Mulai fold {fold['fold']}/{len(folds)}")
        result = train_fold(
            fold=fold,
            X=X,
            y=y,
            metadata=metadata,
            out_dirs=out_dirs,
            log_path=log_path,
            inner_val_size=float(args.inner_val_size),
            sae_epochs=base.SAE_EPOCHS,
            dnn_epochs=base.DNN_EPOCHS,
            batch_size=base.BATCH_SIZE,
        )
        results.append(result)

    save_outputs(results, folds, out_dirs, scenario, start_time)
    text = out_dirs["text"].read_text(encoding="utf-8")
    print(text, end="")
    log_line(log_path, "[INFO] Pipeline selesai.")


if __name__ == "__main__":
    main("unseen_both")
