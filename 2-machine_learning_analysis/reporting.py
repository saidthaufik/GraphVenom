from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


METRIC_COLUMNS: Tuple[str, ...] = (
    "ROC-AUC",
    "PR-AUC",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
)


def format_float(value: Any, digits: int = 4) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "nan"
    if value_f != value_f:
        return "nan"
    return f"{value_f:.{digits}f}"


def format_metrics(metrics: Mapping[str, Any], *, digits: int = 4, elapsed_seconds: Optional[float] = None) -> str:
    parts = [f"{name}={format_float(metrics.get(name), digits)}" for name in METRIC_COLUMNS]
    if elapsed_seconds is not None:
        parts.append(f"time={format_float(elapsed_seconds, 2)}s")
    return " | ".join(parts)


def format_confusion_matrix(cm: Mapping[str, Any]) -> str:
    return (
        f"True Positive (TP) = {int(cm.get('TP', 0))} | "
        f"True Negative (TN) = {int(cm.get('TN', 0))} | "
        f"False Positive (FP) = {int(cm.get('FP', 0))} | "
        f"False Negative (FN) = {int(cm.get('FN', 0))}"
    )


def render_candidate_detail(
    *,
    index: int,
    param_string: str,
    metrics: Mapping[str, Any],
    confusion: Mapping[str, Any],
    elapsed_seconds: Optional[float] = None,
    selection_metric_name: Optional[str] = None,
    selection_metric_value: Optional[float] = None,
) -> List[str]:
    header = f"  Kandidat {index}: {param_string}"
    if selection_metric_name is not None and selection_metric_value is not None:
        header += f" (mean {selection_metric_name}={format_float(selection_metric_value, 4)})"
    return [
        header,
        "  -> Evaluasi Metrik:",
        f"  {format_metrics(metrics, elapsed_seconds=elapsed_seconds)}",
        "  -> Evaluasi Confusion Matrix:",
        f"  {format_confusion_matrix(confusion)}",
    ]


def render_selected_detail(
    *,
    param_string: str,
    selection_metric_name: str,
    selection_metric_value: float,
    metrics: Mapping[str, Any],
    confusion: Mapping[str, Any],
) -> List[str]:
    return [
        f"  Dipilih: {param_string} (mean {selection_metric_name}={format_float(selection_metric_value, 4)})",
        f"  Evaluasi fold: {param_string}",
        f"  -> {format_metrics(metrics)}",
        "  -> Evaluasi Confusion Matrix:",
        f"  {format_confusion_matrix(confusion)}",
    ]


def build_execution_report(start_time: float) -> List[str]:
    elapsed = time.perf_counter() - start_time
    total_seconds = int(elapsed)
    minutes_total, seconds = divmod(total_seconds, 60)
    hours_total, minutes = divmod(minutes_total, 60)
    days, hours = divmod(hours_total, 24)
    return [
        f"Waktu eksekusi: {elapsed:.2f} detik",
        f"Waktu eksekusi: {minutes_total} menit {seconds} detik",
        f"Waktu eksekusi: {hours_total} jam {minutes} menit {seconds} detik",
        f"Waktu eksekusi: {days} hari {hours} jam {minutes} menit {seconds} detik",
    ]


def dataframe_to_string(df: Any, *, index: bool = False) -> str:
    return df.to_string(index=index, float_format=lambda value: f"{value:0.4f}")


def render_sections(sections: Sequence[Tuple[str, str]], timing_lines: Sequence[str] = ()) -> str:
    blocks = [f"{title}\n{content}".rstrip() for title, content in sections]
    if timing_lines:
        blocks.append("\n".join(timing_lines))
    return "\n\n".join(blocks).strip() + "\n"


def voting_lines(rows: Iterable[Mapping[str, Any]], *, exclude: Iterable[str] = ("fold",)) -> List[str]:
    exclude_keys = set(exclude)
    materialized = list(rows)
    if not materialized:
        return ["Tidak ada parameter fold untuk voting."]

    keys: List[str] = []
    for row in materialized:
        for key in row:
            if key not in exclude_keys and key not in keys:
                keys.append(key)

    lines: List[str] = []
    for key in keys:
        values = [row.get(key) for row in materialized if key in row]
        if not values:
            continue
        counts = Counter(str(value) for value in values)
        value, count = counts.most_common(1)[0]
        lines.append(f"{key}: {value} (dipilih {count}/{len(materialized)} fold)")
    return lines or ["Tidak ada parameter fold untuk voting."]
