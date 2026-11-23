from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional


_DEFAULT_ROOT = Path(__file__).resolve().parent
_ROOT_ENV_VARS = (
    "PROJEK_RKI_ROOT",
    "PROJECT_ROOT",
)
_DATASET_ENV_VAR = "PROJEK_RKI_DATASET"
_DEFAULT_DATASET = "rki"


def _dataset_slug(dataset_name: Optional[str] = None) -> str:
    raw = (dataset_name or os.environ.get(_DATASET_ENV_VAR, "") or _DEFAULT_DATASET).strip()
    if not raw:
        raw = _DEFAULT_DATASET
    return raw.lower().replace(" ", "_")


def repo_root() -> Path:
    for env_name in _ROOT_ENV_VARS:
        override = os.environ.get(env_name, "").strip()
        if override:
            root = Path(override).expanduser().resolve()
            if (root / "2-machine_learning_analysis").is_dir():
                root = root / "2-machine_learning_analysis"
            return root
    return _DEFAULT_ROOT


def model_root(model_name: str) -> Path:
    return repo_root() / model_name


def data_root() -> Path:
    return repo_root().parent / "1-data" / "processed" / "training"


def data_csv_path(dataset_name: str = _DEFAULT_DATASET) -> Path:
    suffix = _dataset_slug(dataset_name)
    if suffix == "rki":
        return data_root() / "data.csv"
    return data_root() / f"data_{suffix}.csv"


def legacy_data_csv_candidates(dataset_name: str = _DEFAULT_DATASET) -> Iterable[Path]:
    suffix = _dataset_slug(dataset_name)
    yield data_csv_path(suffix)
    yield repo_root() / "1-neg" / "gan_decoy.csv"
    for model_name in ("model1", "model2"):
        yield model_root(model_name) / "0-neg" / "gan_decoy.csv"


def resolve_data_csv(
    dataset_name: str = _DEFAULT_DATASET,
    env_vars: Optional[Iterable[str]] = None,
) -> Path:
    for env_name in env_vars or ():
        env_path = os.environ.get(env_name, "").strip()
        if env_path:
            return Path(env_path).expanduser().resolve()

    for candidate in legacy_data_csv_candidates(dataset_name):
        if candidate.exists():
            return candidate
    return data_csv_path(dataset_name)


def shared_input_csv() -> Path:
    return data_root() / "gan_decoy.csv"


def model_path(model_name: str, *parts: str) -> Path:
    if parts and str(parts[-1]).endswith((".csv", ".npz", ".npy")):
        return repo_root().parent / "1-data" / "features" / model_name / Path(*parts)
    return model_root(model_name).joinpath(*parts)


def fold_root(dataset_name: str = _DEFAULT_DATASET) -> Path:
    return repo_root() / "cache" / "runs" / "benchmark" / "a-fold"


def log_root(dataset_name: str = _DEFAULT_DATASET) -> Path:
    return repo_root() / "cache" / "runs" / "benchmark" / "b-log_result"


def save_model_root(dataset_name: str = _DEFAULT_DATASET) -> Path:
    return repo_root() / "cache" / "runs" / "benchmark" / "c-model_save"


def output_roots(model_name: str, dataset_name: str = _DEFAULT_DATASET) -> Dict[str, Path]:
    return {
        "model": model_root(model_name),
        "fold_root": fold_root(dataset_name) / model_name,
        "log_root": log_root(dataset_name) / model_name,
        "save_model_root": save_model_root(dataset_name) / model_name,
    }


def fold_dir(model_name: str, dataset_name: str = _DEFAULT_DATASET) -> Path:
    return output_roots(model_name, dataset_name)["fold_root"]


def log_dir(model_name: str, dataset_name: str = _DEFAULT_DATASET) -> Path:
    return output_roots(model_name, dataset_name)["log_root"]


def save_model_dir(model_name: str, dataset_name: str = _DEFAULT_DATASET) -> Path:
    return output_roots(model_name, dataset_name)["save_model_root"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
