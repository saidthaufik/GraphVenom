from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "cache" / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "mathtext.fontset": "stix",
    }
)


MODEL_DIRS = [
    ("Model 1", "model1_sing_reg"),
    ("Model 2", "model2_sing_reg"),
    ("Model 3", "model3_sing_reg"),
    ("Model 4", "model4_sing_reg"),
    ("Model 5", "model5_film_gat_sing_es"),
    ("Model 6", "model6_film_gat_af_sing_es"),
    ("Model 7", "model7_film_gine_sing_es"),
    ("Model 8", "model8_film_gine_af_sing_es"),
    ("Model 9", "model9_film_gcn_sing_es"),
    ("Model 10", "model10_film_gcn_af_sing_es"),
]


SCENARIOS = [
    (
        "unseen_peptide",
        "Unseen Peptide",
        BASE / "results" / "unseen" / "peptide" / "evaluation",
        BASE / "results" / "unseen" / "peptide" / "runs",
    ),
    (
        "unseen_protein",
        "Unseen Protein",
        BASE / "results" / "unseen" / "protein" / "evaluation",
        BASE / "results" / "unseen" / "protein" / "runs",
    ),
    (
        "unseen_both",
        "Unseen Both",
        BASE / "results" / "unseen" / "both" / "evaluation",
        BASE / "results" / "unseen" / "both" / "runs",
    ),
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
]


HIGHLIGHTED_MODELS = {"Model 6", "Model 8", "Model 10"}


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
        if not isinstance(item, dict):
            continue
        keys = {str(k).lower(): k for k in item.keys()}
        y_key = next((keys[k] for k in ["y_true", "label", "labels", "target", "y"] if k in keys), None)
        s_key = next(
            (keys[k] for k in ["y_prob", "y_score", "score", "prob", "proba", "probability"] if k in keys),
            None,
        )
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


def clean_predictions(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if score.ndim == 2:
        score = score[:, 1]
    mask = np.isfinite(y) & np.isfinite(score)
    return y[mask], score[mask]


def collect_results():
    rows = []
    scenario_curves = {}
    for scenario_key, title, fold_base, _ in SCENARIOS:
        curves = []
        for i, (label, model_dir_name) in enumerate(MODEL_DIRS):
            pred_dir = fold_base / model_dir_name / "predictions"
            try:
                (y, score), source = load_predictions(pred_dir)
                y, score = clean_predictions(y, score)

                fpr, tpr, _ = roc_curve(y, score)
                precision, recall, _ = precision_recall_curve(y, score)
                roc_auc = roc_auc_score(y, score)
                pr_auc = average_precision_score(y, score)
                rows.append(
                    {
                        "Scenario": scenario_key,
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
                curves.append(
                    {
                        "label": label,
                        "fpr": fpr,
                        "tpr": tpr,
                        "recall": recall,
                        "precision": precision,
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "color": PALETTE[i % len(PALETTE)],
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "Scenario": scenario_key,
                        "Model": label,
                        "Source": str(pred_dir),
                        "Status": f"error: {exc}",
                    }
                )
        scenario_curves[scenario_key] = {"title": title, "curves": curves}
    return pd.DataFrame(rows), scenario_curves



def parse_metric_pair(text, display_name, table_name):
    patterns = [
        rf"{re.escape(display_name)}\s*[=:]\s*([0-9.]+)\s*(?:±|\+/-)\s*([0-9.]+)",
        rf"{re.escape(table_name)}\s+([0-9.]+)\s+([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))
    raise ValueError(f"Metric {display_name} not found")


def parse_aggregated_metrics_table(text):
    match = re.search(
        r"Aggregated metrics\s*\n\s*(?P<header>ROC-AUC\s+PR-AUC\s+Accuracy\s+Precision\s+Recall\s+F1)\s*\n"
        r"mean\s+(?P<mean>[0-9.\s]+)\n"
        r"std\s+(?P<std>[0-9.\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    headers = match.group("header").split()
    means = [float(x) for x in match.group("mean").split()]
    stds = [float(x) for x in match.group("std").split()]
    values = dict(zip(headers, zip(means, stds)))
    return {
        "ROC-AUC": values["ROC-AUC"],
        "PR-AUC": values["PR-AUC"],
        "F1": values["F1"],
    }


def parse_txt_metrics(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    table_metrics = parse_aggregated_metrics_table(text)
    if table_metrics is not None:
        return table_metrics
    roc_mean, roc_std = parse_metric_pair(text, "ROC-AUC", "roc_auc")
    pr_mean, pr_std = parse_metric_pair(text, "PR-AUC", "average_precision")
    f1_mean, f1_std = parse_metric_pair(text, "F1", "f1_score")
    return {
        "ROC-AUC": (roc_mean, roc_std),
        "PR-AUC": (pr_mean, pr_std),
        "F1": (f1_mean, f1_std),
    }


def load_metric_bar_summary():
    rows = []
    for scenario_key, title, _, log_base in SCENARIOS:
        for label, model_dir_name in MODEL_DIRS:
            model_dir = log_base / model_dir_name
            txt_files = sorted(model_dir.glob("*results*.txt"))
            if not txt_files:
                rows.append(
                    {
                        "Scenario": scenario_key,
                        "Scenario Label": title,
                        "Model": label,
                        "Metric": "",
                        "Mean": np.nan,
                        "Std": np.nan,
                        "Source": str(model_dir),
                        "Status": "missing txt",
                    }
                )
                continue
            source = txt_files[0]
            try:
                metrics = parse_txt_metrics(source)
                for metric, (mean, std) in metrics.items():
                    rows.append(
                        {
                            "Scenario": scenario_key,
                            "Scenario Label": title,
                            "Model": label,
                            "Metric": metric,
                            "Mean": mean,
                            "Std": std,
                            "Source": str(source),
                            "Status": "ok",
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "Scenario": scenario_key,
                        "Scenario Label": title,
                        "Model": label,
                        "Metric": "",
                        "Mean": np.nan,
                        "Std": np.nan,
                        "Source": str(source),
                        "Status": f"error: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def plot_auc_bar_grid(metric_summary):
    metrics = ["ROC-AUC", "PR-AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), dpi=220, sharey=True)
    x = np.arange(len(MODEL_DIRS))
    width = 0.36
    metric_colors = {"ROC-AUC": "#4E79A7", "PR-AUC": "#F28E2B"}

    for ax, (scenario_key, title, _, _) in zip(axes, SCENARIOS):
        part = metric_summary[(metric_summary["Scenario"] == scenario_key) & (metric_summary["Status"] == "ok")]
        for j, metric in enumerate(metrics):
            values = []
            errors = []
            for label, _ in MODEL_DIRS:
                row = part[(part["Model"] == label) & (part["Metric"] == metric)]
                values.append(float(row["Mean"].iloc[0]) if not row.empty else np.nan)
                errors.append(float(row["Std"].iloc[0]) if not row.empty else np.nan)
            edgecolors = ["#111111" if label in HIGHLIGHTED_MODELS else "#263238" for label, _ in MODEL_DIRS]
            line_widths = [1.35 if label in HIGHLIGHTED_MODELS else 0.35 for label, _ in MODEL_DIRS]
            alphas = [0.95 if label in HIGHLIGHTED_MODELS else 0.50 for label, _ in MODEL_DIRS]
            bars = ax.bar(
                x + (j - 0.5) * width,
                values,
                width,
                yerr=errors,
                capsize=3,
                color=metric_colors[metric],
                edgecolor=edgecolors,
                linewidth=line_widths,
                label=metric,
                zorder=3,
            )
            for bar, alpha in zip(bars, alphas):
                bar.set_alpha(alpha)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace("Model ", "M") for label, _ in MODEL_DIRS], rotation=45, ha="right")
        for tick, (model_label, _) in zip(ax.get_xticklabels(), MODEL_DIRS):
            if model_label in HIGHLIGHTED_MODELS:
                tick.set_fontweight("bold")
        ax.set_xlabel("Model")
        ax.set_ylim(0.35, 1.03)
        style_axis(ax)
    axes[0].set_ylabel("Metric Value")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        frameon=True,
        framealpha=0.92,
        ncol=2,
        fontsize=13,
        handlelength=3.2,
        columnspacing=2.0,
    )
    fig.suptitle("ROC-AUC and PR-AUC Unseen Scenario", y=0.98, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.18, 1, 0.93])
    fig.savefig(OUT_DIR / "rki_unseen_auc_bar_models1_10_1x3.png", bbox_inches="tight")
    plt.close(fig)


def plot_f1_bar_grid(metric_summary):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), dpi=220, sharey=True)
    x = np.arange(len(MODEL_DIRS))

    for ax, (scenario_key, title, _, _) in zip(axes, SCENARIOS):
        part = metric_summary[
            (metric_summary["Scenario"] == scenario_key)
            & (metric_summary["Metric"] == "F1")
            & (metric_summary["Status"] == "ok")
        ]
        values = []
        errors = []
        colors = []
        edges = []
        widths = []
        for i, (label, _) in enumerate(MODEL_DIRS):
            row = part[part["Model"] == label]
            values.append(float(row["Mean"].iloc[0]) if not row.empty else np.nan)
            errors.append(float(row["Std"].iloc[0]) if not row.empty else np.nan)
            colors.append(PALETTE[i % len(PALETTE)])
            edges.append("#111111" if label in HIGHLIGHTED_MODELS else "white")
            widths.append(1.15 if label in HIGHLIGHTED_MODELS else 0.45)
        alphas = [0.95 if label in HIGHLIGHTED_MODELS else 0.50 for label, _ in MODEL_DIRS]
        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=3,
            color=colors,
            edgecolor=edges,
            linewidth=widths,
            zorder=3,
        )
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace("Model ", "M") for label, _ in MODEL_DIRS], rotation=45, ha="right")
        for tick, (model_label, _) in zip(ax.get_xticklabels(), MODEL_DIRS):
            if model_label in HIGHLIGHTED_MODELS:
                tick.set_fontweight("bold")
        ax.set_xlabel("Model")
        ax.set_ylim(0.35, 1.03)
        style_axis(ax)
    axes[0].set_ylabel("F1")
    fig.suptitle("F1 Unseen Scenario", y=0.98, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(OUT_DIR / "rki_unseen_f1_bar_models1_10_1x3.png", bbox_inches="tight")
    plt.close(fig)


def style_axis(ax):
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_roc_grid(scenario_curves):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), dpi=220, sharex=True, sharey=True)
    for ax, (scenario_key, title, _, _) in zip(axes, SCENARIOS):
        for curve in scenario_curves[scenario_key]["curves"]:
            label = curve["label"]
            is_highlighted = label in HIGHLIGHTED_MODELS
            ax.plot(
                curve["fpr"],
                curve["tpr"],
                linewidth=3.0 if is_highlighted else 1.35,
                alpha=1.0 if is_highlighted else 0.60,
                color=curve["color"],
                zorder=5 if is_highlighted else 2,
                label=label,
            )
        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.0, color="0.55", alpha=0.8, label="Chance")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        style_axis(ax)
    axes[0].set_ylabel("True Positive Rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        frameon=True,
        framealpha=0.92,
        ncol=11,
        fontsize=11,
        handlelength=3.0,
        columnspacing=1.7,
    )
    fig.suptitle("Unseen Scenario Receiver Operating Characteristic Curve", y=0.98, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.18, 1, 0.93])
    fig.savefig(OUT_DIR / "rki_unseen_roc_models1_10_1x3.png", bbox_inches="tight")
    plt.close(fig)


def plot_pr_grid(scenario_curves):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), dpi=220, sharex=True, sharey=True)
    for ax, (scenario_key, title, _, _) in zip(axes, SCENARIOS):
        for curve in scenario_curves[scenario_key]["curves"]:
            label = curve["label"]
            is_highlighted = label in HIGHLIGHTED_MODELS
            ax.plot(
                curve["recall"],
                curve["precision"],
                linewidth=3.0 if is_highlighted else 1.35,
                alpha=1.0 if is_highlighted else 0.60,
                color=curve["color"],
                zorder=5 if is_highlighted else 2,
                label=label,
            )
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Recall")
        style_axis(ax)
    axes[0].set_ylabel("Precision")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        frameon=True,
        framealpha=0.92,
        ncol=10,
        fontsize=11,
        handlelength=3.0,
        columnspacing=1.7,
    )
    fig.suptitle("Unseen Scenario Precision-Recall Curve", y=0.98, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.18, 1, 0.93])
    fig.savefig(OUT_DIR / "rki_unseen_pr_models1_10_1x3.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(summary):
    ok = summary[summary["Status"] == "ok"].copy()
    if not ok.empty:
        ok["ROC-AUC"] = ok["ROC-AUC"].map(lambda x: f"{x:.6f}")
        ok["PR-AUC"] = ok["PR-AUC"].map(lambda x: f"{x:.6f}")
        for scenario_key, title, _, _ in SCENARIOS:
            part = ok[ok["Scenario"] == scenario_key]
            print(f"\n[{title}]")
            print(part[["Model", "ROC-AUC", "PR-AUC", "n", "positive", "negative"]].to_string(index=False))

    errors = summary[summary["Status"] != "ok"]
    if not errors.empty:
        print("\nMissing/errors:")
        print(errors[["Scenario", "Model", "Source", "Status"]].to_string(index=False))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, scenario_curves = collect_results()
    metric_summary = load_metric_bar_summary()
    summary.to_csv(OUT_DIR / "rki_unseen_roc_pr_summary_models1_10.csv", index=False)
    metric_summary.to_csv(OUT_DIR / "rki_unseen_metric_bar_summary_models1_10.csv", index=False)
    plot_roc_grid(scenario_curves)
    plot_pr_grid(scenario_curves)
    plot_auc_bar_grid(metric_summary)
    plot_f1_bar_grid(metric_summary)
    print_summary(summary)

    metric_errors = metric_summary[metric_summary["Status"] != "ok"]
    if not metric_errors.empty:
        print("\nMetric bar missing/errors:")
        print(metric_errors[["Scenario", "Model", "Source", "Status"]].to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
