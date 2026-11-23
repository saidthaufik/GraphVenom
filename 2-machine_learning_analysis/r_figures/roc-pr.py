from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


BASE = Path(__file__).resolve().parents[1]
MAIN_BASE = BASE / "results" / "benchmark" / "evaluation"
OUT_DIR = BASE / "cache" / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 7,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "mathtext.fontset": "stix",
    }
)


MODELS = [
    ("Model 1", MAIN_BASE / "model1_sing_reg" / "predictions"),
    ("Model 2", MAIN_BASE / "model2_sing_reg" / "predictions"),
    ("Model 3", MAIN_BASE / "model3_sing_reg" / "model3_sing_reg" / "predictions"),
    ("Model 4", MAIN_BASE / "model4_sing_reg" / "predictions"),
    ("Model 5", MAIN_BASE / "model5_film_gat_sing_es" / "predictions"),
    ("Model 6", MAIN_BASE / "model6_film_gat_af_sing_es" / "predictions"),
    ("Model 7", MAIN_BASE / "model7_film_gine_sing_es" / "predictions"),
    ("Model 8", MAIN_BASE / "model8_film_gine_af_sing_es" / "predictions"),
    ("Model 9", MAIN_BASE / "model9_film_gcn_sing_es" / "predictions"),
    ("Model 10", MAIN_BASE / "model10_film_gcn_af_sing_es" / "predictions"),
]


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
]


def pick_column(df, candidates):
    lowered = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise ValueError(f"None of {candidates} found in columns {list(df.columns)}")


def flatten_values(values):
    out = []
    for item in values:
        arr = np.asarray(item)
        if arr.dtype == object and arr.shape == ():
            arr = np.asarray(arr.item())
        out.append(arr.reshape(-1))
    if not out:
        return np.array([])
    return np.concatenate(out)


def from_fold_objects(folds):
    y_parts = []
    s_parts = []
    for fold in folds:
        item = fold.item() if isinstance(fold, np.ndarray) and fold.shape == () else fold
        if isinstance(item, dict):
            keys = {str(k).lower(): k for k in item.keys()}
            y_key = next((keys[k] for k in ["y_true", "label", "labels", "target", "y"] if k in keys), None)
            s_key = next((keys[k] for k in ["y_prob", "y_score", "score", "prob", "proba", "probability"] if k in keys), None)
            if y_key is None or s_key is None:
                continue
            y_parts.append(np.asarray(item[y_key]).reshape(-1))
            s_parts.append(np.asarray(item[s_key]).reshape(-1))
    if not y_parts:
        raise ValueError("No y_true/y_prob keys found in fold objects")
    return np.concatenate(y_parts), np.concatenate(s_parts)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    if {"y_true", "y_prob"}.issubset(keys):
        return flatten_values(data["y_true"]), flatten_values(data["y_prob"])
    if "folds" in keys:
        return from_fold_objects(data["folds"])
    raise ValueError(f"Unsupported npz keys {data.files}")


def load_csv_folds(pred_dir):
    rows = []
    for path in sorted(pred_dir.glob("fold_*_predictions.csv")):
        df = pd.read_csv(path)
        y_col = pick_column(df, ["y_true", "label", "labels", "target", "y"])
        s_col = pick_column(df, ["y_prob", "y_score", "score", "prob", "proba", "probability"])
        rows.append(df[[y_col, s_col]].rename(columns={y_col: "y_true", s_col: "y_prob"}))
    if not rows:
        raise FileNotFoundError(f"No fold prediction CSV found in {pred_dir}")
    out = pd.concat(rows, ignore_index=True).dropna()
    return out["y_true"].astype(int).to_numpy(), out["y_prob"].astype(float).to_numpy()


def load_predictions(pred_dir):
    npz_path = pred_dir / "kfold_predictions.npz"
    if npz_path.exists():
        try:
            return load_npz(npz_path), str(npz_path)
        except Exception:
            pass
    y, score = load_csv_folds(pred_dir)
    return (y, score), str(pred_dir / "fold_*_predictions.csv")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = []
    for i, (label, pred_dir) in enumerate(MODELS):
        try:
            (y, score), source = load_predictions(pred_dir)
            y = np.asarray(y).astype(int)
            score = np.asarray(score).astype(float)
            mask = ~(np.isnan(y) | np.isnan(score))
            y = y[mask]
            score = score[mask]

            fpr, tpr, _ = roc_curve(y, score)
            precision, recall, _ = precision_recall_curve(y, score)
            roc_auc = roc_auc_score(y, score)
            pr_auc = average_precision_score(y, score)
            style = "-"
            color = PALETTE[i % len(PALETTE)]
            rows.append(
                {
                    "Model": label,
                    "ROC-AUC": roc_auc,
                    "PR-AUC": pr_auc,
                    "n": len(y),
                    "positive": int(y.sum()),
                    "negative": int((1 - y).sum()),
                    "Source": source,
                    "Status": "ok",
                }
            )
            curves.append((label, fpr, tpr, recall, precision, roc_auc, pr_auc, style, color))
        except Exception as exc:
            rows.append({"Model": label, "Source": str(pred_dir), "Status": f"error: {exc}"})

    highlighted_models = {"Model 6", "Model 8", "Model 10"}

    plt.figure(figsize=(9, 7), dpi=220)
    for label, fpr, tpr, _, _, roc_auc, _, style, color in curves:
        is_highlighted = label in highlighted_models
        plt.plot(
            fpr,
            tpr,
            linestyle=style,
            linewidth=3.2 if is_highlighted else 1.45,
            alpha=1.0 if is_highlighted else 0.62,
            color=color,
            zorder=5 if is_highlighted else 2,
            label=f"{label} ({roc_auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1.0, color="0.55", alpha=0.8, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right", frameon=True, framealpha=0.92, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rki_main_roc_models1_10.png")
    plt.close()

    plt.figure(figsize=(9, 7), dpi=220)
    for label, _, _, recall, precision, _, pr_auc, style, color in curves:
        is_highlighted = label in highlighted_models
        plt.plot(
            recall,
            precision,
            linestyle=style,
            linewidth=3.2 if is_highlighted else 1.45,
            alpha=1.0 if is_highlighted else 0.62,
            color=color,
            zorder=5 if is_highlighted else 2,
            label=f"{label} ({pr_auc:.3f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left", frameon=True, framealpha=0.92, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rki_main_pr_models1_10.png")
    plt.close()

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "rki_main_roc_pr_summary_models1_10.csv", index=False)

    ok = summary[summary["Status"] == "ok"].copy()
    if not ok.empty:
        ok["ROC-AUC"] = ok["ROC-AUC"].map(lambda x: f"{x:.6f}")
        ok["PR-AUC"] = ok["PR-AUC"].map(lambda x: f"{x:.6f}")
        print(ok[["Model", "ROC-AUC", "PR-AUC", "n", "positive", "negative"]].to_string(index=False))
    missing = summary[summary["Status"] != "ok"]
    if not missing.empty:
        print("\nMissing/errors:")
        print(missing[["Model", "Source", "Status"]].to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
