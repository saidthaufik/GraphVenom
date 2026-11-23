from datetime import datetime
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
LOG_BASE = BASE / "results" / "benchmark" / "runs"
OUT_DIR = BASE / "cache" / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "mathtext.fontset": "stix",
    }
)


MODELS = [
    ("Model 1", LOG_BASE / "model1_sing_reg"),
    ("Model 2", LOG_BASE / "model2_sing_reg"),
    ("Model 3", LOG_BASE / "model3_sing_reg" / "model3_sing_reg"),
    ("Model 4", LOG_BASE / "model4_sing_reg"),
    ("Model 5", LOG_BASE / "model5_film_gat_sing_es"),
    ("Model 6", LOG_BASE / "model6_film_gat_af_sing_es"),
    ("Model 7", LOG_BASE / "model7_film_gine_sing_es"),
    ("Model 8", LOG_BASE / "model8_film_gine_af_sing_es"),
    ("Model 9", LOG_BASE / "model9_film_gcn_sing_es"),
    ("Model 10", LOG_BASE / "model10_film_gcn_af_sing_es"),
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

HIGHLIGHTED = {"Model 6", "Model 8", "Model 10"}


def model_order(label):
    match = re.search(r"Model\s+(\d+)", label)
    return int(match.group(1)) if match else 999


def parse_csv_elapsed(model_dir):
    path = model_dir / "reports" / "kfold_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ["elapsed_seconds", "elapsed_sec", "time_seconds", "time_sec"]:
        if col in df.columns:
            out = df[["fold", col]].copy()
            out.columns = ["Fold", "Training Time (Seconds)"]
            return out
    if {"train_time_s", "eval_time_s"}.issubset(df.columns):
        out = pd.DataFrame({
            "Fold": df["fold"],
            "Training Time (Seconds)": df["train_time_s"].fillna(0) + df["eval_time_s"].fillna(0),
        })
        return out
    return None


def parse_time_equals(model_dir):
    txt_files = sorted(model_dir.glob("*results_rki.txt"))
    if not txt_files:
        return None
    text = txt_files[0].read_text(encoding="utf-8", errors="ignore")
    values = [float(x) for x in re.findall(r"time=([0-9.]+)s", text)]
    if len(values) < 5:
        return None
    values = values[:5]
    return pd.DataFrame({"Fold": range(1, 6), "Training Time (Seconds)": values})


def parse_resume_elapsed(model_dir):
    json_files = sorted(model_dir.glob("*resume_rki.json"))
    if not json_files:
        return None
    import json

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    rows = []
    for fold_key, fold_state in sorted(payload.get("folds", {}).items(), key=lambda item: int(item[0])):
        candidates = fold_state.get("candidates", {})
        elapsed_values = [
            float(candidate["elapsed_seconds"])
            for candidate in candidates.values()
            if "elapsed_seconds" in candidate
        ]
        if elapsed_values:
            rows.append({"Fold": int(fold_key), "Training Time (Seconds)": sum(elapsed_values)})
    if len(rows) == 5:
        return pd.DataFrame(rows)
    return None


def parse_total_execution_time(model_dir):
    txt_files = sorted(model_dir.glob("*results_rki.txt"))
    if not txt_files:
        return None
    text = txt_files[0].read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Waktu eksekusi:\s*([0-9.]+)\s*detik", text)
    if not matches:
        return None
    total = float(matches[-1])
    return pd.DataFrame({
        "Fold": range(1, 6),
        "Training Time (Seconds)": [total / 5.0] * 5,
    })


def parse_run_log_fold_durations(model_dir):
    path = model_dir / "logs" / "run.log"
    if not path.exists():
        return None

    starts = {}
    ends = {}
    time_pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+"
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stamp_match = re.match(time_pattern, line)
        if not stamp_match:
            continue
        stamp = datetime.strptime(stamp_match.group(1), "%Y-%m-%d %H:%M:%S")
        start_match = re.search(r"Mulai fold\s+(\d+)/5", line)
        end_match = re.search(r"Fold\s+(\d+)\s+selesai", line)
        if start_match:
            starts[int(start_match.group(1))] = stamp
        if end_match:
            ends[int(end_match.group(1))] = stamp

    rows = []
    for fold in range(1, 6):
        if fold in starts and fold in ends:
            rows.append({"Fold": fold, "Training Time (Seconds)": (ends[fold] - starts[fold]).total_seconds()})
    if len(rows) == 5:
        return pd.DataFrame(rows)
    return None


def load_model_times(label, model_dir):
    for loader in [parse_csv_elapsed, parse_time_equals, parse_resume_elapsed, parse_run_log_fold_durations, parse_total_execution_time]:
        df = loader(model_dir)
        if df is not None:
            df["Model"] = label
            df["Source"] = loader.__name__
            return df
    raise ValueError(f"No per-fold training time found for {label}: {model_dir}")


def build_summary():
    frames = []
    for label, model_dir in MODELS:
        frames.append(load_model_times(label, model_dir))
    summary = pd.concat(frames, ignore_index=True)
    return summary[["Model", "Fold", "Training Time (Seconds)", "Source"]]


def plot_training_time(summary):
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=220)
    folds = [1, 2, 3, 4, 5]

    for idx, (label, _) in enumerate(MODELS):
        data = summary[summary["Model"] == label].sort_values("Fold")
        is_highlighted = label in HIGHLIGHTED
        ax.plot(
            data["Fold"],
            data["Training Time (Seconds)"],
            marker="o",
            markersize=6.2 if is_highlighted else 4.6,
            linewidth=3.0 if is_highlighted else 1.55,
            alpha=1.0 if is_highlighted else 0.64,
            color=PALETTE[idx % len(PALETTE)],
            label=label,
            zorder=5 if is_highlighted else 2,
        )

    ax.set_title("Training Time Model")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Training Time (Seconds)")
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.grid(alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_time_model.png")
    plt.close(fig)


def plot_training_time_bar(summary):
    pivot = summary.pivot(index="Fold", columns="Model", values="Training Time (Seconds)").sort_index()
    labels = sorted(list(pivot.columns), key=model_order)
    folds = list(pivot.index)
    width = 0.78 / len(labels)
    offsets = [(i - (len(labels) - 1) / 2) * width for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(11.4, 6.4), dpi=220)
    for idx, label in enumerate(labels):
        is_highlighted = label in HIGHLIGHTED
        color = PALETTE[idx % len(PALETTE)]
        ax.bar(
            [fold + offsets[idx] for fold in folds],
            pivot[label],
            width=width,
            color=color,
            alpha=0.98 if is_highlighted else 0.62,
            edgecolor="#263238" if is_highlighted else "white",
            linewidth=1.0 if is_highlighted else 0.4,
            label=label,
            zorder=5 if is_highlighted else 2,
        )

    ax.set_title("Training Time Model")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Training Time (Seconds)")
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.grid(axis="y", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_time_model_bar.png")
    plt.close(fig)


def plot_training_time_total_bar(summary):
    totals = (
        summary.groupby("Model", as_index=False)["Training Time (Seconds)"]
        .sum()
        .sort_values("Model", key=lambda s: s.map(model_order))
    )

    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=220)
    colors = []
    edges = []
    for label in totals["Model"]:
        if label in HIGHLIGHTED:
            colors.append("#C44536")
            edges.append("#7A1F17")
        else:
            colors.append("#65A89A")
            edges.append("#28695D")

    bars = ax.bar(
        totals["Model"],
        totals["Training Time (Seconds)"],
        color=colors,
        edgecolor=edges,
        linewidth=1.0,
        zorder=3,
    )
    max_time = totals["Training Time (Seconds)"].max()
    for bar, value in zip(bars, totals["Training Time (Seconds)"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max_time * 0.018,
            f"{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#263238",
        )

    ax.set_title("Training Time Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Training Time (Seconds)")
    ax.set_ylim(0, max_time * 1.16)
    ax.grid(axis="y", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_time_model_total_bar.png")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    summary.to_csv(OUT_DIR / "training_time_model.csv", index=False)
    plot_training_time(summary)
    plot_training_time_bar(summary)
    plot_training_time_total_bar(summary)

    print(summary.pivot(index="Model", columns="Fold", values="Training Time (Seconds)").round(2).to_string())
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
