import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, GraphNorm, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

try:
    from torch.serialization import add_safe_globals
except (ImportError, AttributeError):
    add_safe_globals = None

try:
    from torch_geometric.data.data import DataEdgeAttr
except ImportError:
    DataEdgeAttr = None
else:
    if add_safe_globals is not None:
        add_safe_globals([DataEdgeAttr])


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path import model_root
from reporting import (
    build_execution_report,
    dataframe_to_string,
    render_candidate_detail,
    render_sections,
    render_selected_detail,
    voting_lines,
)


PROJECT_DIR = model_root("model7")
PREPROCESSED_DIR = PROJECT_DIR / "0-feat" / "processed_gine"
PREPROCESS_PATH = PROJECT_DIR / "0-feat" / "pre.py"
LAYERS_PATH = PROJECT_DIR / "0-feat" / "layers_crossmpt_pingpong.py"
DATASET_NAME = "rki"
OUTPUT_MODEL_NAME = "model7_film_gine_sing_es"
SINGLE_GAN_ROOT = PROJECT_ROOT / "cache" / "runs" / "benchmark"
_OUTPUT_ROOTS = {
    "log_root": SINGLE_GAN_ROOT / "b-log_result" / OUTPUT_MODEL_NAME,
    "fold_root": SINGLE_GAN_ROOT / "a-fold" / OUTPUT_MODEL_NAME,
    "save_model_root": SINGLE_GAN_ROOT / "c-model_save" / OUTPUT_MODEL_NAME,
}
LOG_ROOT = _OUTPUT_ROOTS["log_root"]
FOLD_ROOT = _OUTPUT_ROOTS["fold_root"]
MODEL_ROOT = _OUTPUT_ROOTS["save_model_root"]

if str(PREPROCESS_PATH.parent) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_PATH.parent))


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Gagal memuat modul: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pre_module = load_module(PREPROCESS_PATH, "model7_pre_local")
layer_module = load_module(LAYERS_PATH, "model7_layers_local")

BidirectionalCrossMPT = layer_module.BidirectionalCrossMPT
PEPTIDE_NODE_DIM = int(pre_module.PEPTIDE_NODE_DIM)
PROTEIN_NODE_DIM = int(pre_module.PROTEIN_NODE_DIM)
PEPTIDE_EDGE_DIM = int(pre_module.PEPTIDE_EDGE_DIM)
PROTEIN_EDGE_DIM = int(pre_module.PROTEIN_EDGE_DIM)

DEFAULT_CONFIG: Dict[str, Any] = {
    "random_seed": 42,
    "test_size": 0.2,
    "n_splits": 5,
    "grad_clip": 1.0,
    "threshold": 0.5,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_every": 1,
    "early_stopping_monitor": "val_pr_auc",
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.001,
    "restore_best_weights": True,
}

# Single-grid early-stopping training model7 baseline GINE: no FiLM and no AF.
GRID: Dict[str, List[Any]] = {
    "epochs": [200],
    "batch_size": [64],
    "lr": [5e-4],
    "weight_decay": [3e-4],
    "dropout": [0.15],
    "mlp_dropout": [0.2],
    "hidden_dim": [64],
    "out_dim": [256],
    "heads": [4],
    "gnn_layers": [4],
    "mlp_hidden": [[256, 128]],
    "pos_weight": [1.0],
    "global_mode": ["none"],
    "film_target": ["none"],
    "film_scale": [0.0],
}

GLOBAL_FEATURE_NAMES = [
    "pep_len",
    "prot_len",
    "log_pep_len",
    "log_prot_len",
    "pep_prot_len_ratio",
    "prot_num_contacts",
    "prot_contact_density",
    "prot_plddt_mean",
    "prot_plddt_min",
    "prot_plddt_low_ratio",
]
GLOBAL_MODE_COLUMNS = {
    "none": [],
    "length": [0, 1, 2, 3, 4],
    "structure": [5, 6, 7, 8, 9],
    "all": list(range(10)),
}


def first_grid_value(key: str) -> Any:
    values = GRID.get(key)
    if not values:
        raise KeyError(f"GRID key tidak ditemukan atau kosong: {key}")
    return values[0]


_CLI_PROGRESS_ACTIVE = False
_CLI_PROGRESS_LINES = 0


def clear_cli_progress() -> None:
    global _CLI_PROGRESS_ACTIVE, _CLI_PROGRESS_LINES
    if not _CLI_PROGRESS_ACTIVE:
        return
    print("\r\033[2K", end="", flush=False)
    for _ in range(max(_CLI_PROGRESS_LINES - 1, 0)):
        print("\033[F\033[2K", end="", flush=False)
    _CLI_PROGRESS_ACTIVE = False
    _CLI_PROGRESS_LINES = 0


def log_info(message: str) -> None:
    clear_cli_progress()
    print(message, flush=True)
    if _ACTIVE_RESULTS_LOG is not None:
        append_text(_ACTIVE_RESULTS_LOG, message)


def log_epoch_live(
    *,
    fold_idx: int,
    total_folds: int,
    candidate_pos: int,
    total_candidates: int,
    candidate_cfg: Dict[str, Any],
    epoch: int,
    train_result: Dict[str, Any],
    val_result: Dict[str, Any],
    best_epoch: int,
    best_val_metrics: Dict[str, Any],
) -> None:
    global _CLI_PROGRESS_ACTIVE, _CLI_PROGRESS_LINES
    epoch_message = (
        f"[INFO] Fold {fold_idx} kandidat {candidate_cfg['candidate_name']} "
        f"epoch {epoch}/{candidate_cfg['epochs']} "
        f"train_loss={train_result['metrics']['loss']:.4f} "
        f"val_loss={val_result['metrics']['loss']:.4f} "
        f"val_pr={val_result['metrics']['pr_auc']:.4f}"
    )
    if _ACTIVE_RESULTS_LOG is not None:
        append_text(_ACTIVE_RESULTS_LOG, epoch_message)

    current_metrics = snakebite_metrics(val_result["metrics"])
    best_metrics = snakebite_metrics(best_val_metrics)
    current_confusion = binary_confusion_counts(val_result["y_true"], val_result["y_pred"])
    lines = [
        "Detail tuning & evaluasi",
        f"Fold {fold_idx}/{total_folds}",
        f"  Kandidat {candidate_pos}/{total_candidates} (id={candidate_cfg['candidate_id']}): {candidate_param_string(candidate_cfg)}",
        f"  -> Progress: epoch {epoch}/{candidate_cfg['epochs']} | train_loss={train_result['metrics']['loss']:.4f} | val_loss={val_result['metrics']['loss']:.4f}",
        "  -> Evaluasi Metrik (validasi saat ini):",
        f"  ROC-AUC={current_metrics['ROC-AUC']:.4f} | PR-AUC={current_metrics['PR-AUC']:.4f} | Accuracy={current_metrics['Accuracy']:.4f} | Precision={current_metrics['Precision']:.4f} | Recall={current_metrics['Recall']:.4f} | F1={current_metrics['F1']:.4f}",
        f"  -> Best so far: epoch {best_epoch} | ROC-AUC={best_metrics['ROC-AUC']:.4f} | PR-AUC={best_metrics['PR-AUC']:.4f} | F1={best_metrics['F1']:.4f}",
        "  -> Evaluasi Confusion Matrix (validasi saat ini):",
        f"  True Positive (TP) = {current_confusion['TP']} | True Negative (TN) = {current_confusion['TN']} | False Positive (FP) = {current_confusion['FP']} | False Negative (FN) = {current_confusion['FN']}",
    ]
    clear_cli_progress()
    print("\n".join(lines), end="", flush=True)
    _CLI_PROGRESS_ACTIVE = True
    _CLI_PROGRESS_LINES = len(lines)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value: Any) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return float("nan")
        return result
    except Exception:
        return float("nan")


def safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return safe_float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return obj


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def safe_torch_save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_text(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(to_serializable(payload), sort_keys=True, separators=(",", ":"))


def build_resume_config_payload(cfg: Dict[str, Any], candidate_cfgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "config": to_serializable(cfg),
        "candidate_configs": to_serializable(candidate_cfgs),
    }


def compute_config_signature(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_cpu_state = state.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(torch_cpu_state)
    torch_cuda_state = state.get("torch_cuda")
    if torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_state)


def get_scaler_state(scaler: Optional[Any]) -> Optional[Dict[str, Any]]:
    if scaler is None:
        return None
    return scaler.state_dict()


def restore_scaler_state(scaler: Optional[Any], scaler_state: Optional[Dict[str, Any]]) -> None:
    if scaler is None or scaler_state is None:
        return
    scaler.load_state_dict(scaler_state)


def load_run_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"folds": {}, "final": {}, "resume_meta": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    state.setdefault("folds", {})
    state.setdefault("final", {})
    state.setdefault("resume_meta", {})
    return state


def save_run_state(path: Path, state: Dict[str, Any]) -> None:
    save_json(path, state)


def snakebite_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "ROC-AUC": safe_float(metrics.get("roc_auc")),
        "PR-AUC": safe_float(metrics.get("pr_auc")),
        "Accuracy": safe_float(metrics.get("accuracy")),
        "Precision": safe_float(metrics.get("precision")),
        "Recall": safe_float(metrics.get("recall")),
        "F1": safe_float(metrics.get("f1")),
    }


def binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    return {"TP": int(cm[1, 1]), "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0])}


def candidate_param_string(candidate_cfg: Dict[str, Any]) -> str:
    return stable_json_dumps(
        {
            "epochs": int(candidate_cfg["epochs"]),
            "batch_size": int(candidate_cfg["batch_size"]),
            "lr": float(candidate_cfg["lr"]),
            "weight_decay": float(candidate_cfg["weight_decay"]),
            "dropout": float(candidate_cfg["dropout"]),
            "mlp_dropout": float(candidate_cfg["mlp_dropout"]),
            "hidden_dim": int(candidate_cfg["hidden_dim"]),
            "heads": int(candidate_cfg["heads"]),
            "out_dim": int(candidate_cfg["out_dim"]),
            "mlp_hidden": list(candidate_cfg["mlp_hidden"]),
            "gnn_layers": int(candidate_cfg["gnn_layers"]),
            "global_mode": str(candidate_cfg.get("global_mode", "none")),
            "film_target": str(candidate_cfg.get("film_target", "protein")),
            "film_scale": safe_float(candidate_cfg.get("film_scale", 0.1)),
            "global_dim": int(candidate_cfg.get("global_dim", len(select_global_columns(str(candidate_cfg.get("global_mode", "none")))))),
            "threshold": float(candidate_cfg["threshold"]),
        }
    )


def load_saved_resume_config_payload(run_state: Dict[str, Any], out_dirs: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    resume_meta = run_state.get("resume_meta", {})
    payload = resume_meta.get("config_payload")
    if isinstance(payload, dict):
        return payload

    config_path = out_dirs["reports"] / "config.json"
    candidates_path = out_dirs["reports"] / "candidate_configs.json"
    if config_path.exists() and candidates_path.exists():
        try:
            saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            saved_candidates_raw = json.loads(candidates_path.read_text(encoding="utf-8"))
            saved_candidates = saved_candidates_raw.get("candidates", [])
            if isinstance(saved_cfg, dict) and isinstance(saved_candidates, list):
                return build_resume_config_payload(saved_cfg, saved_candidates)
        except Exception:
            return None
    return None


def ensure_resume_compatibility(
    run_state: Dict[str, Any],
    out_dirs: Dict[str, Path],
    current_payload: Dict[str, Any],
    resume_requested: bool,
) -> str:
    current_signature = compute_config_signature(current_payload)
    saved_payload = load_saved_resume_config_payload(run_state, out_dirs)
    if saved_payload is not None:
        saved_signature = compute_config_signature(saved_payload)
        if resume_requested and saved_signature != current_signature:
            raise RuntimeError(
                "Resume config mismatch untuk run "
                f"{out_dirs['run_name']}. "
                f"saved_signature={saved_signature}, current_signature={current_signature}. "
                "Gunakan config yang sama persis dengan run lama atau mulai run baru tanpa --resume-run."
            )
    run_state["resume_meta"] = {
        "config_signature": current_signature,
        "config_payload": to_serializable(current_payload),
    }
    return current_signature


def assert_checkpoint_signature(payload: Dict[str, Any], expected_signature: str, label: str) -> None:
    saved_signature = str(payload.get("config_signature", "")).strip()
    if saved_signature and saved_signature != expected_signature:
        raise RuntimeError(
            f"Resume checkpoint mismatch pada {label}: "
            f"saved_signature={saved_signature}, expected_signature={expected_signature}."
        )


def is_run_unfinished(state: Dict[str, Any]) -> bool:
    final_completed = bool(state.get("final", {}).get("completed"))
    folds = state.get("folds", {})
    if not final_completed:
        return bool(folds) or bool(state.get("final"))
    return any(not bool(fold_state.get("completed")) for fold_state in folds.values())


def find_unfinished_runs(log_root: Path) -> List[str]:
    unfinished: List[str] = []
    if not log_root.exists():
        return unfinished
    for child in sorted(log_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        state_candidates = [child / f"model7_film_gine_sing_es_resume_{DATASET_NAME}.json", child / "run_state.json", MODEL_ROOT / child.name / "resume" / "run_state.json"]
        for state_path in state_candidates:
            if not state_path.exists():
                continue
            try:
                state = load_run_state(state_path)
            except Exception:
                continue
            if is_run_unfinished(state):
                unfinished.append(child.name)
                break
    return unfinished


def as_numpy_prediction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_indices": np.asarray(payload["sample_indices"], dtype=np.int64),
        "y_true": np.asarray(payload["y_true"], dtype=np.float32),
        "y_prob": np.asarray(payload["y_prob"], dtype=np.float32),
        "y_pred": np.asarray(payload["y_pred"], dtype=np.int64),
    }


def load_preprocessed_data(base_dir: Path) -> Tuple[pd.DataFrame, List[Data], List[Data], np.ndarray]:
    metadata = pd.read_csv(base_dir / "metadata.csv")
    peptide_graphs = list(safe_torch_load(base_dir / "peptide_graphs.pt"))
    protein_graphs = list(safe_torch_load(base_dir / "protein_graphs.pt"))
    labels = np.load(base_dir / "y.npy").reshape(-1).astype(np.int64)
    if len({len(metadata), len(peptide_graphs), len(protein_graphs), len(labels)}) != 1:
        raise ValueError("Jumlah sampel preprocess tidak sinkron.")
    return metadata, peptide_graphs, protein_graphs, labels


def build_global_feature_matrix(metadata: pd.DataFrame) -> np.ndarray:
    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in metadata.columns:
            return np.full(len(metadata), float(default), dtype=np.float32)
        return pd.to_numeric(metadata[name], errors="coerce").fillna(default).to_numpy(dtype=np.float32)

    pep_len = np.clip(col("pep_len"), 0.0, None)
    prot_len = np.clip(col("prot_len"), 0.0, None)
    contacts = np.clip(col("prot_num_contacts"), 0.0, None)
    plddt_mean = np.clip(col("prot_plddt_mean"), 0.0, 1.0)
    plddt_min = np.clip(col("prot_plddt_min"), 0.0, 1.0)
    plddt_low = np.clip(col("prot_plddt_low_ratio"), 0.0, 1.0)
    safe_prot_len = np.maximum(prot_len, 1.0)
    features = np.stack(
        [
            pep_len,
            prot_len,
            np.log1p(pep_len),
            np.log1p(prot_len),
            pep_len / safe_prot_len,
            contacts,
            contacts / safe_prot_len,
            plddt_mean,
            plddt_min,
            plddt_low,
        ],
        axis=1,
    ).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def select_global_columns(mode: str) -> List[int]:
    mode_key = str(mode or "none").strip().lower()
    if mode_key not in GLOBAL_MODE_COLUMNS:
        raise ValueError(f"global_mode tidak dikenal: {mode}")
    return list(GLOBAL_MODE_COLUMNS[mode_key])


def normalize_global_features(global_raw: np.ndarray, fit_indices: Sequence[int], mode: str) -> np.ndarray:
    cols = select_global_columns(mode)
    if not cols:
        return np.zeros((global_raw.shape[0], 0), dtype=np.float32)
    selected = global_raw[:, cols].astype(np.float32, copy=False)
    fit = selected[np.asarray(fit_indices, dtype=np.int64)]
    mean = fit.mean(axis=0, keepdims=True)
    std = fit.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((selected - mean) / std).astype(np.float32)


class PairGraphDataset(Dataset):
    def __init__(self, peptide_graphs: Sequence[Data], protein_graphs: Sequence[Data], labels: np.ndarray, indices: Sequence[int], global_features: Optional[np.ndarray] = None):
        self.peptide_graphs = peptide_graphs
        self.protein_graphs = protein_graphs
        self.labels = np.asarray(labels, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        if global_features is None:
            global_features = np.zeros((len(self.labels), 0), dtype=np.float32)
        self.global_features = np.asarray(global_features, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        sample_idx = int(self.indices[idx])
        return (
            self.peptide_graphs[sample_idx].clone(),
            self.protein_graphs[sample_idx].clone(),
            torch.tensor(self.global_features[sample_idx], dtype=torch.float32),
            torch.tensor(self.labels[sample_idx], dtype=torch.float32),
            sample_idx,
        )


def collate_fn(batch):
    peptide_graphs, protein_graphs, global_features, labels, indices = zip(*batch)
    return (
        Batch.from_data_list(list(peptide_graphs)),
        Batch.from_data_list(list(protein_graphs)),
        torch.stack(list(global_features), dim=0),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(indices, dtype=torch.long),
    )


def make_gine_mlp(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))


class FiLMLayer(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.proj = nn.Sequential(
            nn.Linear(int(global_dim), int(hidden_dim) * 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(global_features.to(dtype=x.dtype, device=x.device))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[batch_index]
        beta = beta[batch_index]
        return x * (1.0 + self.scale * gamma) + self.scale * beta


class GINEGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float, global_dim: int = 0, film_scale: float = 0.1):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers harus >= 2")
        self.dropout = nn.Dropout(dropout)
        self.global_dim = int(global_dim)
        self.convs = nn.ModuleList([GINEConv(make_gine_mlp(in_dim, hidden_dim), edge_dim=int(edge_dim), train_eps=True)])
        self.norms = nn.ModuleList([GraphNorm(hidden_dim)])
        dims = [int(hidden_dim)]
        for _ in range(num_layers - 2):
            self.convs.append(GINEConv(make_gine_mlp(hidden_dim, hidden_dim), edge_dim=int(edge_dim), train_eps=True))
            self.norms.append(GraphNorm(hidden_dim))
            dims.append(int(hidden_dim))
        self.convs.append(GINEConv(make_gine_mlp(hidden_dim, out_dim), edge_dim=int(edge_dim), train_eps=True))
        self.norms.append(GraphNorm(out_dim))
        dims.append(int(out_dim))
        self.film_layers = nn.ModuleList(
            [FiLMLayer(self.global_dim, dim, film_scale) for dim in dims]
        ) if self.global_dim > 0 else nn.ModuleList()

    def forward(self, batch: Batch, global_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr, batch_index = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        use_film = self.global_dim > 0 and global_features is not None and global_features.numel() > 0
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_attr)
            x = norm(x, batch_index)
            if use_film:
                x = self.film_layers[layer_idx](x, batch_index, global_features)
            x = F.relu(x)
            x = self.dropout(x)
        graph_emb = torch.cat([global_mean_pool(x, batch_index), global_max_pool(x, batch_index)], dim=-1)
        return x, batch_index, graph_emb


class PeptideProteinGINEModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.global_dim = int(cfg.get("global_dim", 0))
        film_target = str(cfg.get("film_target", "protein")).strip().lower()
        if film_target not in {"protein", "both", "none"}:
            raise ValueError(f"film_target tidak dikenal: {film_target}")
        film_scale = float(cfg.get("film_scale", 0.1))
        pep_global_dim = self.global_dim if film_target == "both" else 0
        prot_global_dim = self.global_dim if film_target in {"protein", "both"} else 0
        self.peptide_encoder = GINEGraphEncoder(PEPTIDE_NODE_DIM, PEPTIDE_EDGE_DIM, int(cfg["hidden_dim"]), int(cfg["out_dim"]), int(cfg["gnn_layers"]), float(cfg["dropout"]), global_dim=pep_global_dim, film_scale=film_scale)
        self.protein_encoder = GINEGraphEncoder(PROTEIN_NODE_DIM, PROTEIN_EDGE_DIM, int(cfg["hidden_dim"]), int(cfg["out_dim"]), int(cfg["gnn_layers"]), float(cfg["dropout"]), global_dim=prot_global_dim, film_scale=film_scale)
        mlp_layers: List[nn.Module] = []
        graph_dim = int(cfg["out_dim"]) * 2
        prev_dim = graph_dim * 4
        for width in cfg["mlp_hidden"]:
            mlp_layers.extend([nn.Linear(prev_dim, int(width)), nn.LayerNorm(int(width)), nn.ReLU(), nn.Dropout(float(cfg["mlp_dropout"]))])
            prev_dim = int(width)
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(self, pep_batch: Batch, prot_batch: Batch, global_features: Optional[torch.Tensor] = None):
        if self.global_dim > 0 and global_features is None:
            raise ValueError("global_features wajib ada saat global_dim > 0")
        _, _, pep_emb = self.peptide_encoder(pep_batch, global_features)
        _, _, prot_emb = self.protein_encoder(prot_batch, global_features)
        fused_feature = torch.cat([pep_emb, prot_emb, torch.abs(pep_emb - prot_emb), pep_emb * prot_emb], dim=-1)
        return self.classifier(fused_feature).squeeze(-1)


def build_model(cfg: Dict[str, Any]) -> PeptideProteinGINEModel:
    cfg = dict(cfg)
    cfg["global_dim"] = len(select_global_columns(str(cfg.get("global_mode", "none"))))
    return PeptideProteinGINEModel(cfg).to(torch.device(cfg["device"]))

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    return {
        "loss": float("nan"),
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "precision": safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": safe_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
        "pr_auc": safe_metric(average_precision_score, y_true, y_prob),
        "threshold": float(threshold),
    }


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, optimizer: Optional[torch.optim.Optimizer], grad_clip: float, threshold: float) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_samples = 0
    y_true_all: List[float] = []
    y_prob_all: List[float] = []
    y_pred_all: List[int] = []
    sample_indices_all: List[int] = []
    for pep_batch, prot_batch, global_features, labels, sample_indices in loader:
        pep_batch = pep_batch.to(device)
        prot_batch = prot_batch.to(device)
        global_features = global_features.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(pep_batch, prot_batch, global_features)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()
        probs = torch.sigmoid(logits)
        preds = (probs >= float(threshold)).long()
        batch_size = int(labels.size(0))
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_samples += batch_size
        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_prob_all.extend(probs.detach().cpu().numpy().tolist())
        y_pred_all.extend(preds.detach().cpu().numpy().tolist())
        sample_indices_all.extend(sample_indices.detach().cpu().numpy().tolist())
    metrics = compute_binary_metrics(np.asarray(y_true_all), np.asarray(y_prob_all), threshold=threshold)
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return {"metrics": metrics, "y_true": np.asarray(y_true_all, dtype=np.float32), "y_prob": np.asarray(y_prob_all, dtype=np.float32), "y_pred": np.asarray(y_pred_all, dtype=np.int64), "sample_indices": np.asarray(sample_indices_all, dtype=np.int64)}


def train_one_epoch(*args, **kwargs) -> Dict[str, Any]:
    return run_epoch(*args, **kwargs)


def evaluate_one_epoch(model, loader, device, criterion, threshold) -> Dict[str, Any]:
    with torch.no_grad():
        return run_epoch(model, loader, device, criterion, None, 0.0, threshold)


def create_dataloader(peptide_graphs: Sequence[Data], protein_graphs: Sequence[Data], labels: np.ndarray, indices: Sequence[int], batch_size: int, shuffle: bool, num_workers: int, global_features: Optional[np.ndarray] = None) -> DataLoader:
    dataset = PairGraphDataset(peptide_graphs, protein_graphs, labels, indices, global_features=global_features)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), collate_fn=collate_fn, num_workers=int(num_workers), persistent_workers=False)


def should_update_best(current: Dict[str, float], best: Optional[Dict[str, float]]) -> bool:
    if best is None:
        return True
    current_pr = safe_float(current.get("pr_auc", float("nan")))
    best_pr = safe_float(best.get("pr_auc", float("nan")))
    if not math.isnan(current_pr) and (math.isnan(best_pr) or current_pr > best_pr):
        return True
    if current_pr == best_pr:
        current_roc = safe_float(current.get("roc_auc", float("nan")))
        best_roc = safe_float(best.get("roc_auc", float("nan")))
        if not math.isnan(current_roc) and (math.isnan(best_roc) or current_roc > best_roc):
            return True
    return False


def make_splits(labels: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, np.ndarray]]]:
    all_indices = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(all_indices, test_size=float(cfg["test_size"]), random_state=int(cfg["random_seed"]), shuffle=True, stratify=labels)
    skf = StratifiedKFold(n_splits=int(cfg["n_splits"]), shuffle=True, random_state=int(cfg["random_seed"]))
    folds = []
    for fold_id, (tr_rel, val_rel) in enumerate(skf.split(trainval_idx, labels[trainval_idx]), start=1):
        folds.append({"fold": np.asarray([fold_id], dtype=np.int64), "train_indices": trainval_idx[tr_rel], "val_indices": trainval_idx[val_rel]})
    return trainval_idx, test_idx, folds


def parse_candidate_ids(raw: str) -> Optional[set[int]]:
    raw = (raw or "").strip()
    if not raw:
        return None
    candidate_ids = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not candidate_ids:
        return None
    return candidate_ids


def build_candidate_configs(base_cfg: Dict[str, Any], candidate_ids: Optional[set[int]] = None) -> List[Dict[str, Any]]:
    candidate_cfgs: List[Dict[str, Any]] = []
    for idx, override in enumerate(ParameterGrid(GRID), start=1):
        if candidate_ids is not None and idx not in candidate_ids:
            continue
        cfg = dict(base_cfg)
        cfg.update({key: override[key] for key in override})
        cfg["epochs"] = int(base_cfg["epochs"])
        cfg["candidate_id"] = idx
        cfg["candidate_name"] = (
            f"cand{idx}_{override['global_mode']}_{override['film_target']}_fs{override['film_scale']}_lr{override['lr']}_wd{override['weight_decay']}"
            f"_out{override['out_dim']}_mlp{'-'.join(str(x) for x in override['mlp_hidden'])}"
        )
        candidate_cfgs.append(cfg)
    if not candidate_cfgs:
        raise ValueError(f"Tidak ada kandidat yang cocok dengan candidate_ids={candidate_ids}")
    return candidate_cfgs


def plot_kfold_roc(fold_predictions: List[Dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for item in fold_predictions:
        try:
            fpr, tpr, _ = roc_curve(item["y_true"], item["y_prob"])
            auc_val = roc_auc_score(item["y_true"], item["y_prob"])
            plt.plot(fpr, tpr, label=f"Fold {item['fold']} AUC={auc_val:.3f}", alpha=0.7)
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_kfold_pr(fold_predictions: List[Dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for item in fold_predictions:
        try:
            precision, recall, _ = precision_recall_curve(item["y_true"], item["y_prob"])
            auc_val = average_precision_score(item["y_true"], item["y_prob"])
            plt.plot(recall, precision, label=f"Fold {item['fold']} AP={auc_val:.3f}", alpha=0.7)
        except Exception:
            continue
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_avg_loss_acc_kfold(histories: List[Dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for history in histories:
        plt.plot(history["train_loss"], alpha=0.4, color="tab:blue")
        plt.plot(history["val_loss"], alpha=0.4, color="tab:orange")
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    for history in histories:
        plt.plot(history["train_accuracy"], alpha=0.4, color="tab:green")
        plt.plot(history["val_accuracy"], alpha=0.4, color="tab:red")
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_final_test_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr)
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[0].set_title("ROC")
    except Exception:
        pass
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[1].plot(recall, precision)
        axes[1].set_title("PR")
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_final_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_kfold_predictions_npz(fold_predictions: List[Dict[str, Any]], out_path: Path) -> None:
    np.savez_compressed(out_path, folds=np.asarray(fold_predictions, dtype=object))


def save_fold_prediction_files(fold_predictions: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in fold_predictions:
        fold = int(item["fold"])
        base_name = out_dir / f"fold_{fold}"
        np.savez_compressed(str(base_name) + "_predictions.npz", sample_indices=np.asarray(item["sample_indices"], dtype=np.int64), y_true=np.asarray(item["y_true"], dtype=np.float32), y_prob=np.asarray(item["y_prob"], dtype=np.float32), y_pred=np.asarray(item["y_pred"], dtype=np.int64))
        pd.DataFrame({"sample_indices": np.asarray(item["sample_indices"], dtype=np.int64), "y_true": np.asarray(item["y_true"], dtype=np.int64), "y_prob": np.asarray(item["y_prob"], dtype=np.float32), "y_pred": np.asarray(item["y_pred"], dtype=np.int64)}).to_csv(str(base_name) + "_predictions.csv", index=False)


def save_final_predictions_npz(final_prediction_payload: Dict[str, Any], out_path: Path) -> None:
    np.savez_compressed(out_path, **final_prediction_payload)


def make_output_dirs(run_name: str) -> Dict[str, Path]:
    return {
        "run_name": run_name,
        "log_root": ensure_dir(LOG_ROOT),
        "fold_root": ensure_dir(FOLD_ROOT),
        "model_root": ensure_dir(MODEL_ROOT),
        "logs": ensure_dir(LOG_ROOT / "json"),
        "reports": ensure_dir(LOG_ROOT / "reports"),
        "plots": ensure_dir(LOG_ROOT / "plots"),
        "text": LOG_ROOT / "model7_film_gine_sing_es_results_rki.txt",
        "state": LOG_ROOT / "model7_film_gine_sing_es_resume_rki.json",
        "predictions": ensure_dir(FOLD_ROOT / "predictions"),
        "fold_reports": ensure_dir(FOLD_ROOT / "reports"),
        "checkpoints": ensure_dir(MODEL_ROOT / "checkpoints"),
        "resume": ensure_dir(MODEL_ROOT / "resume"),
    }


def run_kfold_training(peptide_graphs: Sequence[Data], protein_graphs: Sequence[Data], labels: np.ndarray, fold_specs: List[Dict[str, np.ndarray]], candidate_cfgs: List[Dict[str, Any]], out_dirs: Dict[str, Path], run_state: Dict[str, Any], config_signature: str, global_raw: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int], List[Dict[str, Any]], Dict[str, Any], List[str], List[Dict[str, Any]]]:
    fold_metrics = []
    histories = []
    best_epochs = []
    fold_predictions = []
    best_candidate_names: List[str] = []
    detail_logs: List[str] = []
    best_params_per_fold: List[Dict[str, Any]] = []
    for fold_idx, fold_spec in enumerate(fold_specs, start=1):
        log_info(f"\n[INFO] Fold {fold_idx}/{len(fold_specs)}")
        fold_key = str(fold_idx)
        saved_fold = run_state.get("folds", {}).get(fold_key)
        if saved_fold and saved_fold.get("completed"):
            fold_metrics.append(saved_fold["fold_metric"]); histories.append(saved_fold["history"]); best_epochs.append(int(saved_fold["best_epoch"])); fold_predictions.append(saved_fold["fold_prediction"]); best_candidate_names.append(str(saved_fold["best_cfg"]["candidate_name"]))
            saved_detail = saved_fold.get("detail_logs", [])
            if saved_detail:
                detail_logs.extend(str(line) for line in saved_detail)
            if saved_fold.get("best_cfg"):
                best_cfg_saved = saved_fold["best_cfg"]
                best_params_per_fold.append({"fold": fold_idx, "candidate_id": int(best_cfg_saved["candidate_id"]), "candidate_name": str(best_cfg_saved["candidate_name"]), "lr": float(best_cfg_saved["lr"]), "weight_decay": float(best_cfg_saved["weight_decay"]), "dropout": float(best_cfg_saved["dropout"]), "mlp_dropout": float(best_cfg_saved["mlp_dropout"]), "hidden_dim": int(best_cfg_saved["hidden_dim"]), "heads": int(best_cfg_saved["heads"]), "out_dim": int(best_cfg_saved["out_dim"]), "gnn_layers": int(best_cfg_saved["gnn_layers"]), "global_mode": str(best_cfg_saved.get("global_mode", "none")), "film_target": str(best_cfg_saved.get("film_target", "protein")), "film_scale": safe_float(best_cfg_saved.get("film_scale", 0.1)), "global_dim": int(best_cfg_saved.get("global_dim", 0)), "mlp_hidden": "-".join(str(x) for x in best_cfg_saved["mlp_hidden"]), "best_epoch": int(saved_fold["best_epoch"])})
            continue
        fold_best_cfg = None; fold_best_metrics = None; fold_best_state = None; fold_best_epoch = 1; fold_best_result = None; fold_best_history = None
        fold_detail_lines = [f"Fold {fold_idx}/{len(fold_specs)}"]
        for candidate_pos, candidate_cfg in enumerate(candidate_cfgs, start=1):
            candidate_start = time.perf_counter()
            device = torch.device(candidate_cfg["device"])
            global_features = normalize_global_features(global_raw, fold_spec["train_indices"], str(candidate_cfg.get("global_mode", "none")))
            candidate_cfg["global_dim"] = int(global_features.shape[1])
            train_loader = create_dataloader(peptide_graphs, protein_graphs, labels, fold_spec["train_indices"], candidate_cfg["batch_size"], True, candidate_cfg["num_workers"], global_features=global_features)
            val_loader = create_dataloader(peptide_graphs, protein_graphs, labels, fold_spec["val_indices"], candidate_cfg["batch_size"], False, candidate_cfg["num_workers"], global_features=global_features)
            model = build_model(candidate_cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(candidate_cfg["lr"]), weight_decay=float(candidate_cfg["weight_decay"]))
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(candidate_cfg["pos_weight"])], device=device), reduction="mean")
            scaler = None
            history = {"fold": fold_idx, "candidate_id": int(candidate_cfg["candidate_id"]), "candidate_name": candidate_cfg["candidate_name"], "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
            best_state = None; best_val_metrics = None; best_epoch = 1; best_val_result = None; start_epoch = 1
            candidate_resume_path = out_dirs["resume"] / f"fold{fold_idx}_candidate{candidate_cfg['candidate_id']}.pth"
            fold_state = run_state.setdefault("folds", {}).setdefault(fold_key, {"candidates": {}})
            candidate_state = fold_state.setdefault("candidates", {}).get(str(candidate_cfg["candidate_id"]), {})
            if candidate_state.get("completed") and candidate_resume_path.exists():
                resume_payload = safe_torch_load(candidate_resume_path)
                assert_checkpoint_signature(resume_payload, config_signature, f"fold {fold_idx} candidate {candidate_cfg['candidate_name']}")
                best_state = resume_payload.get("best_state_dict"); best_val_metrics = resume_payload.get("best_val_metrics"); best_epoch = int(resume_payload.get("best_epoch", 1)); best_val_result = resume_payload.get("best_val_result"); history = resume_payload.get("history", history)
                saved_elapsed = candidate_state.get("elapsed_seconds"); saved_confusion = candidate_state.get("confusion")
                if best_val_result is not None and saved_confusion is None:
                    saved_confusion = binary_confusion_counts(np.asarray(best_val_result["y_true"]), np.asarray(best_val_result["y_pred"]))
                candidate_lines = render_candidate_detail(index=int(candidate_cfg["candidate_id"]), param_string=candidate_param_string(candidate_cfg), metrics=snakebite_metrics(best_val_metrics), confusion=saved_confusion or {"TP": 0, "TN": 0, "FP": 0, "FN": 0}, elapsed_seconds=safe_float(saved_elapsed) if saved_elapsed is not None else None)
                fold_detail_lines.extend(candidate_lines)
                for line in candidate_lines:
                    log_info(line)
            else:
                if candidate_resume_path.exists():
                    resume_payload = safe_torch_load(candidate_resume_path)
                    assert_checkpoint_signature(resume_payload, config_signature, f"fold {fold_idx} candidate {candidate_cfg['candidate_name']}")
                    model.load_state_dict(resume_payload["model_state_dict"]); optimizer.load_state_dict(resume_payload["optimizer_state_dict"]); restore_scaler_state(scaler, resume_payload.get("scaler_state_dict")); restore_rng_state(resume_payload.get("rng_state")); history = resume_payload.get("history", history); best_state = resume_payload.get("best_state_dict"); best_val_metrics = resume_payload.get("best_val_metrics"); best_epoch = int(resume_payload.get("best_epoch", 1)); best_val_result = resume_payload.get("best_val_result"); start_epoch = int(resume_payload.get("epoch", 0)) + 1
                    log_info(f"[INFO] Resume fold {fold_idx} kandidat {candidate_cfg['candidate_name']} dari epoch {start_epoch}.")
                early_stop_patience = int(candidate_cfg.get("early_stopping_patience", 0))
                early_stop_min_delta = float(candidate_cfg.get("early_stopping_min_delta", 0.0))
                best_monitor_score = safe_float(best_val_metrics.get("pr_auc", float("nan"))) if best_val_metrics is not None else float("-inf")
                epochs_without_improvement = 0
                for epoch in range(start_epoch, int(candidate_cfg["epochs"]) + 1):
                    train_result = train_one_epoch(model, train_loader, device, criterion, optimizer, float(candidate_cfg["grad_clip"]), float(candidate_cfg["threshold"]))
                    val_result = evaluate_one_epoch(model, val_loader, device, criterion, float(candidate_cfg["threshold"]))
                    history["train_loss"].append(float(train_result["metrics"]["loss"])); history["val_loss"].append(float(val_result["metrics"]["loss"])); history["train_accuracy"].append(float(train_result["metrics"]["accuracy"])); history["val_accuracy"].append(float(val_result["metrics"]["accuracy"]))
                    if should_update_best(val_result["metrics"], best_val_metrics):
                        best_val_metrics = dict(val_result["metrics"]); best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch; best_val_result = {"metrics": dict(val_result["metrics"]), "sample_indices": val_result["sample_indices"], "y_true": val_result["y_true"], "y_prob": val_result["y_prob"], "y_pred": val_result["y_pred"]}
                    current_monitor_score = safe_float(val_result["metrics"].get("pr_auc", float("nan")))
                    if not math.isnan(current_monitor_score) and current_monitor_score > best_monitor_score + early_stop_min_delta:
                        best_monitor_score = current_monitor_score; epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    should_stop_early = early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience
                    if should_stop_early:
                        log_info("[EARLY STOP] Fold {} | Cand {} berhenti di epoch {}/{} | monitor=val_pr_auc | best_epoch={} | best_pr_auc={:.4f}".format(fold_idx, candidate_cfg["candidate_id"], epoch, candidate_cfg["epochs"], best_epoch, best_monitor_score))
                    safe_torch_save(candidate_resume_path, {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "history": history, "best_state_dict": best_state, "best_val_metrics": best_val_metrics, "best_epoch": best_epoch, "best_val_result": best_val_result, "completed": False, "config_signature": config_signature, "rng_state": capture_rng_state(), "scaler_state_dict": get_scaler_state(scaler)})
                    log_epoch_live(
                        fold_idx=fold_idx,
                        total_folds=len(fold_specs),
                        candidate_pos=candidate_pos,
                        total_candidates=len(candidate_cfgs),
                        candidate_cfg=candidate_cfg,
                        epoch=epoch,
                        train_result=train_result,
                        val_result=val_result,
                        best_epoch=best_epoch,
                        best_val_metrics=best_val_metrics,
                    )
                    if should_stop_early:
                        break
                safe_torch_save(candidate_resume_path, {"epoch": int(candidate_cfg["epochs"]), "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "history": history, "best_state_dict": best_state, "best_val_metrics": best_val_metrics, "best_epoch": best_epoch, "best_val_result": best_val_result, "completed": True, "config_signature": config_signature, "rng_state": capture_rng_state(), "scaler_state_dict": get_scaler_state(scaler)})
                fold_state["candidates"][str(candidate_cfg["candidate_id"])] = {"completed": True, "candidate_name": candidate_cfg["candidate_name"], "best_epoch": int(best_epoch), "best_val_metrics": to_serializable(best_val_metrics), "resume_path": str(candidate_resume_path), "confusion": binary_confusion_counts(best_val_result["y_true"], best_val_result["y_pred"]), "elapsed_seconds": float(time.perf_counter() - candidate_start)}
                save_run_state(out_dirs["state"], run_state)
                candidate_lines = render_candidate_detail(index=int(candidate_cfg["candidate_id"]), param_string=candidate_param_string(candidate_cfg), metrics=snakebite_metrics(best_val_metrics), confusion=fold_state["candidates"][str(candidate_cfg["candidate_id"])]["confusion"], elapsed_seconds=fold_state["candidates"][str(candidate_cfg["candidate_id"])]["elapsed_seconds"])
                fold_detail_lines.extend(candidate_lines)
                for line in candidate_lines:
                    log_info(line)
            if should_update_best(best_val_metrics, fold_best_metrics):
                fold_best_cfg = dict(candidate_cfg); fold_best_metrics = dict(best_val_metrics); fold_best_state = copy.deepcopy(best_state); fold_best_epoch = int(best_epoch); fold_best_result = best_val_result; fold_best_history = history
        checkpoint_path = out_dirs["checkpoints"] / f"best_model_fold{fold_idx}.pth"
        safe_torch_save(checkpoint_path, {"model_state_dict": fold_best_state, "config": fold_best_cfg, "fold": fold_idx, "best_epoch": fold_best_epoch})
        best_epochs.append(fold_best_epoch); best_candidate_names.append(str(fold_best_cfg["candidate_name"]))
        fold_confusion = binary_confusion_counts(fold_best_result["y_true"], fold_best_result["y_pred"])
        selected_lines = render_selected_detail(param_string=candidate_param_string(fold_best_cfg), selection_metric_name="PR-AUC", selection_metric_value=safe_float(fold_best_metrics["pr_auc"]), metrics=snakebite_metrics(fold_best_metrics), confusion=fold_confusion)
        fold_detail_lines.extend(selected_lines)
        for line in selected_lines:
            log_info(line)
        fold_metrics.append({"fold": fold_idx, "candidate_id": int(fold_best_cfg["candidate_id"]), "candidate_name": fold_best_cfg["candidate_name"], "best_epoch": fold_best_epoch, "val_loss": float(fold_best_metrics["loss"]), "val_accuracy": float(fold_best_metrics["accuracy"]), "val_precision": float(fold_best_metrics["precision"]), "val_recall": float(fold_best_metrics["recall"]), "val_f1": float(fold_best_metrics["f1"]), "val_roc_auc": float(fold_best_metrics["roc_auc"]), "val_pr_auc": float(fold_best_metrics["pr_auc"]), "tp": int(fold_confusion["TP"]), "tn": int(fold_confusion["TN"]), "fp": int(fold_confusion["FP"]), "fn": int(fold_confusion["FN"])})
        histories.append(fold_best_history); fold_predictions.append({"fold": fold_idx, "candidate_name": fold_best_cfg["candidate_name"], "sample_indices": fold_best_result["sample_indices"], "y_true": fold_best_result["y_true"], "y_prob": fold_best_result["y_prob"], "y_pred": fold_best_result["y_pred"]})
        run_state["folds"][fold_key] = {"completed": True, "best_epoch": int(fold_best_epoch), "best_cfg": to_serializable(fold_best_cfg), "fold_metric": to_serializable(fold_metrics[-1]), "history": to_serializable(fold_best_history), "fold_prediction": to_serializable(fold_predictions[-1]), "detail_logs": list(fold_detail_lines), "candidates": run_state["folds"][fold_key]["candidates"]}
        save_run_state(out_dirs["state"], run_state)
        detail_logs.extend(fold_detail_lines)
        best_params_per_fold.append({"fold": fold_idx, "candidate_id": int(fold_best_cfg["candidate_id"]), "candidate_name": str(fold_best_cfg["candidate_name"]), "lr": float(fold_best_cfg["lr"]), "weight_decay": float(fold_best_cfg["weight_decay"]), "dropout": float(fold_best_cfg["dropout"]), "mlp_dropout": float(fold_best_cfg["mlp_dropout"]), "hidden_dim": int(fold_best_cfg["hidden_dim"]), "heads": int(fold_best_cfg["heads"]), "out_dim": int(fold_best_cfg["out_dim"]), "gnn_layers": int(fold_best_cfg["gnn_layers"]), "global_mode": str(fold_best_cfg.get("global_mode", "none")), "film_target": str(fold_best_cfg.get("film_target", "protein")), "film_scale": safe_float(fold_best_cfg.get("film_scale", 0.1)), "global_dim": int(fold_best_cfg.get("global_dim", 0)), "mlp_hidden": "-".join(str(x) for x in fold_best_cfg["mlp_hidden"]), "best_epoch": int(fold_best_epoch)})
    best_name = max(set(best_candidate_names), key=best_candidate_names.count)
    final_cfg = next((dict(cfg) for cfg in candidate_cfgs if str(cfg["candidate_name"]) == best_name), dict(candidate_cfgs[0]))
    return fold_metrics, histories, best_epochs, fold_predictions, final_cfg, detail_logs, best_params_per_fold


def train_final_model(peptide_graphs: Sequence[Data], protein_graphs: Sequence[Data], labels: np.ndarray, trainval_indices: np.ndarray, test_indices: np.ndarray, cfg: Dict[str, Any], out_dirs: Dict[str, Path], final_epochs: int, run_state: Dict[str, Any], config_signature: str, global_raw: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    device = torch.device(cfg["device"])
    global_features = normalize_global_features(global_raw, trainval_indices, str(cfg.get("global_mode", "none")))
    cfg = dict(cfg)
    cfg["global_dim"] = int(global_features.shape[1])
    train_loader = create_dataloader(peptide_graphs, protein_graphs, labels, trainval_indices, cfg["batch_size"], True, cfg["num_workers"], global_features=global_features)
    test_loader = create_dataloader(peptide_graphs, protein_graphs, labels, test_indices, cfg["batch_size"], False, cfg["num_workers"], global_features=global_features)
    final_state = run_state.setdefault("final", {})
    if final_state.get("completed"):
        return final_state["result"], as_numpy_prediction_payload(final_state["prediction_payload"])
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(cfg["pos_weight"])], device=device), reduction="mean")
    scaler = None
    history = {"epochs": final_epochs, "train_loss": [], "train_accuracy": []}
    final_resume_path = out_dirs["resume"] / "final_training.pth"
    start_epoch = 1
    if final_resume_path.exists():
        resume_payload = safe_torch_load(final_resume_path)
        assert_checkpoint_signature(resume_payload, config_signature, "final training")
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        restore_scaler_state(scaler, resume_payload.get("scaler_state_dict"))
        restore_rng_state(resume_payload.get("rng_state"))
        history = resume_payload.get("history", history)
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
    for epoch in range(start_epoch, final_epochs + 1):
        train_result = train_one_epoch(model, train_loader, device, criterion, optimizer, float(cfg["grad_clip"]), float(cfg["threshold"]))
        history["train_loss"].append(float(train_result["metrics"]["loss"]))
        history["train_accuracy"].append(float(train_result["metrics"]["accuracy"]))
        safe_torch_save(final_resume_path, {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "history": history, "completed": False, "config_signature": config_signature, "rng_state": capture_rng_state(), "scaler_state_dict": get_scaler_state(scaler)})
    checkpoint_path = out_dirs["checkpoints"] / "final_model.pth"
    safe_torch_save(checkpoint_path, {"model_state_dict": model.state_dict(), "config": cfg, "final_epochs": final_epochs})
    test_result = evaluate_one_epoch(model, test_loader, device, criterion, float(cfg["threshold"]))
    test_metrics = dict(test_result["metrics"])
    cm = confusion_matrix(test_result["y_true"].astype(int), test_result["y_pred"].astype(int), labels=[0, 1])
    test_metrics["confusion_matrix"] = cm.tolist()
    test_metrics["tn"] = int(cm[0, 0]); test_metrics["fp"] = int(cm[0, 1]); test_metrics["fn"] = int(cm[1, 0]); test_metrics["tp"] = int(cm[1, 1])
    result = {"metrics": test_metrics, "history": history, "checkpoint_path": str(checkpoint_path)}
    prediction_payload = {"sample_indices": test_result["sample_indices"], "y_true": test_result["y_true"], "y_prob": test_result["y_prob"], "y_pred": test_result["y_pred"]}
    safe_torch_save(final_resume_path, {"epoch": final_epochs, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "history": history, "completed": True, "result": result, "prediction_payload": prediction_payload, "config_signature": config_signature, "rng_state": capture_rng_state(), "scaler_state_dict": get_scaler_state(scaler)})
    run_state["final"] = {"completed": True, "result": to_serializable(result), "prediction_payload": to_serializable(prediction_payload)}
    save_run_state(out_dirs["state"], run_state)
    return result, prediction_payload


def summarize_dataset(labels: np.ndarray, metadata: pd.DataFrame) -> Dict[str, Any]:
    return {"num_samples": int(len(labels)), "num_positive": int(np.sum(labels == 1)), "num_negative": int(np.sum(labels == 0)), "avg_peptide_len": float(pd.to_numeric(metadata["pep_len"], errors="coerce").fillna(0).mean()) if "pep_len" in metadata.columns else None, "avg_protein_len": float(pd.to_numeric(metadata["prot_len"], errors="coerce").fillna(0).mean()) if "prot_len" in metadata.columns else None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training model7 baseline GINE: no FiLM and no AF.")
    parser.add_argument("--data-dir", type=Path, default=PREPROCESSED_DIR, help="Folder hasil preprocessing.")
    parser.add_argument("--epochs", type=int, default=int(first_grid_value("epochs")))
    parser.add_argument("--batch-size", type=int, default=int(first_grid_value("batch_size")))
    parser.add_argument("--hidden-dim", type=int, default=int(first_grid_value("hidden_dim")))
    parser.add_argument("--out-dim", type=int, default=int(first_grid_value("out_dim")))
    parser.add_argument("--heads", type=int, default=int(first_grid_value("heads")))
    parser.add_argument("--gnn-layers", type=int, default=int(first_grid_value("gnn_layers")))
    parser.add_argument("--dropout", type=float, default=float(first_grid_value("dropout")))
    parser.add_argument("--mlp-dropout", type=float, default=float(first_grid_value("mlp_dropout")))
    parser.add_argument("--lr", type=float, default=float(first_grid_value("lr")))
    parser.add_argument("--weight-decay", type=float, default=float(first_grid_value("weight_decay")))
    parser.add_argument("--n-splits", type=int, default=DEFAULT_CONFIG["n_splits"])
    parser.add_argument("--test-size", type=float, default=DEFAULT_CONFIG["test_size"])
    parser.add_argument("--random-seed", type=int, default=DEFAULT_CONFIG["random_seed"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--log-every", type=int, default=DEFAULT_CONFIG["log_every"])
    parser.add_argument("--candidate-ids", type=str, default="", help="Comma-separated candidate ids, contoh: 3.")
    parser.add_argument("--max-folds", type=int, default=0, help="Batasi jumlah fold k-fold. 0 berarti semua fold.")
    parser.add_argument("--skip-final", action="store_true", help="Lewati final training/test setelah k-fold probe.")
    parser.add_argument("--run-name", type=str, default="", help="Nama run untuk metadata. Default: model7_film_gine_sing_es.")
    parser.add_argument("--resume-run", type=str, default="", help="Nama run untuk melanjutkan training yang terputus.")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"fusion": "gine_pairwise", "epochs": int(args.epochs), "batch_size": int(args.batch_size), "hidden_dim": int(args.hidden_dim), "out_dim": int(args.out_dim), "heads": int(args.heads), "gnn_layers": int(args.gnn_layers), "dropout": float(args.dropout), "mlp_dropout": float(args.mlp_dropout), "mlp_hidden": list(first_grid_value("mlp_hidden")), "lr": float(args.lr), "weight_decay": float(args.weight_decay), "pos_weight": float(first_grid_value("pos_weight")), "global_mode": str(first_grid_value("global_mode")), "film_target": str(first_grid_value("film_target")), "film_scale": float(first_grid_value("film_scale")), "n_splits": int(args.n_splits), "test_size": float(args.test_size), "random_seed": int(args.random_seed), "threshold": float(args.threshold), "device": str(args.device), "log_every": int(args.log_every)})
    return cfg


_ACTIVE_RESULTS_LOG: Optional[Path] = None


def main() -> None:
    global _ACTIVE_RESULTS_LOG
    start_time = time.perf_counter()
    args = parse_args()
    cfg = build_config_from_args(args)
    set_all_seeds(int(cfg["random_seed"]))
    resume_requested = bool(args.resume_run.strip())
    if not resume_requested:
        unfinished_runs = find_unfinished_runs(LOG_ROOT)
        if unfinished_runs:
            preview = ", ".join(unfinished_runs[:3])
            suffix = "" if len(unfinished_runs) <= 3 else f", +{len(unfinished_runs) - 3} lainnya"
            log_info("[WARN] Ditemukan unfinished run untuk model7_film_gine_sing_es: " f"{preview}{suffix}. " "Gunakan --resume-run <nama_run> jika ingin melanjutkan run lama; run baru akan dibuat.")
    if resume_requested:
        run_name = args.resume_run.strip()
    elif args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        run_name = "model7_film_gine_sing_es"
    out_dirs = make_output_dirs(run_name)
    _ACTIVE_RESULTS_LOG = out_dirs["text"]
    if resume_requested and not out_dirs["state"].exists():
        raise FileNotFoundError(f"Resume state untuk run '{run_name}' tidak ditemukan di {out_dirs['state']}")
    run_state = load_run_state(out_dirs["state"])
    log_info(f"[INFO] Run mulai: {run_name}")
    metadata, peptide_graphs, protein_graphs, labels = load_preprocessed_data(args.data_dir)
    dataset_summary = summarize_dataset(labels, metadata)
    trainval_indices, test_indices, fold_specs = make_splits(labels, cfg)
    if int(args.max_folds) > 0:
        fold_specs = fold_specs[: int(args.max_folds)]
    candidate_cfgs = build_candidate_configs(cfg, candidate_ids=parse_candidate_ids(args.candidate_ids))
    resume_config_payload = build_resume_config_payload(cfg, candidate_cfgs)
    config_signature = ensure_resume_compatibility(run_state=run_state, out_dirs=out_dirs, current_payload=resume_config_payload, resume_requested=resume_requested)
    save_run_state(out_dirs["state"], run_state)
    save_json(out_dirs["reports"] / "config.json", cfg)
    save_json(out_dirs["reports"] / "candidate_configs.json", {"candidates": candidate_cfgs})
    save_json(out_dirs["reports"] / "dataset_summary.json", dataset_summary)
    global_raw = build_global_feature_matrix(metadata)
    fold_metrics, histories, best_epochs, fold_predictions, final_cfg, detail_logs, best_params_per_fold = run_kfold_training(peptide_graphs, protein_graphs, labels, fold_specs, candidate_cfgs, out_dirs, run_state, config_signature, global_raw)
    final_cfg["device"] = str(cfg["device"])
    final_cfg["num_workers"] = int(cfg["num_workers"])
    final_cfg["grad_clip"] = float(cfg["grad_clip"])
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(out_dirs["reports"] / "kfold_metrics.csv", index=False)
    fold_metrics_df.to_csv(out_dirs["fold_reports"] / "kfold_metrics.csv", index=False)
    save_json(out_dirs["logs"] / "kfold_histories.json", {"histories": histories})
    save_kfold_predictions_npz(fold_predictions, out_dirs["predictions"] / "kfold_predictions.npz")
    save_fold_prediction_files(fold_predictions, out_dirs["predictions"])
    plot_kfold_roc(fold_predictions, out_dirs["plots"] / "roc_auc_kfold_all.png")
    plot_kfold_pr(fold_predictions, out_dirs["plots"] / "pr_curve_kfold_all.png")
    plot_avg_loss_acc_kfold(histories, out_dirs["plots"] / "avg_loss_acc_kfold.png")
    final_epochs = 0
    final_result = {"metrics": {}}
    final_prediction_payload = None
    if not args.skip_final:
        final_epoch_buffer = int(cfg.get("early_stopping_patience", 0))
        final_epochs = int(max(1, min(int(final_cfg["epochs"]), int(max(best_epochs) + final_epoch_buffer))))
        log_info("[INFO] Final epochs dipilih dari max best epoch fold + patience: {} (max_best={}, buffer={})".format(final_epochs, int(max(best_epochs)), final_epoch_buffer))
        final_result, final_prediction_payload = train_final_model(peptide_graphs, protein_graphs, labels, trainval_indices, test_indices, final_cfg, out_dirs, final_epochs, run_state, config_signature, global_raw)
        save_json(out_dirs["reports"] / "final_test_metrics.json", final_result["metrics"])
        save_json(out_dirs["logs"] / "final_train_history.json", final_result["history"])
        save_final_predictions_npz(final_prediction_payload, out_dirs["predictions"] / "final_test_predictions.npz")
        plot_final_test_roc_pr(final_prediction_payload["y_true"], final_prediction_payload["y_prob"], out_dirs["plots"] / "final_test_roc_pr_curve.png")
        plot_final_confusion_matrix(final_prediction_payload["y_true"], final_prediction_payload["y_pred"], out_dirs["plots"] / "final_test_confusion_matrix.png")
    save_json(out_dirs["reports"] / "run_summary.json", {"config": cfg, "final_candidate_config": final_cfg, "dataset_summary": dataset_summary, "kfold_mean": {"val_accuracy": float(fold_metrics_df["val_accuracy"].mean()), "val_precision": float(fold_metrics_df["val_precision"].mean()), "val_recall": float(fold_metrics_df["val_recall"].mean()), "val_f1": float(fold_metrics_df["val_f1"].mean()), "val_roc_auc": float(fold_metrics_df["val_roc_auc"].mean()), "val_pr_auc": float(fold_metrics_df["val_pr_auc"].mean())}, "final_test_metrics": final_result["metrics"], "final_epochs": final_epochs, "best_epochs_per_fold": best_epochs, "skip_final": bool(args.skip_final)})
    report_fold_metrics_df = pd.DataFrame([{"ROC-AUC": row["val_roc_auc"], "PR-AUC": row["val_pr_auc"], "Accuracy": row["val_accuracy"], "Precision": row["val_precision"], "Recall": row["val_recall"], "F1": row["val_f1"], "fold": int(row["fold"])} for row in fold_metrics])
    aggregated_metrics_df = pd.DataFrame({"ROC-AUC": [float(report_fold_metrics_df["ROC-AUC"].mean()), float(report_fold_metrics_df["ROC-AUC"].std(ddof=0))], "PR-AUC": [float(report_fold_metrics_df["PR-AUC"].mean()), float(report_fold_metrics_df["PR-AUC"].std(ddof=0))], "Accuracy": [float(report_fold_metrics_df["Accuracy"].mean()), float(report_fold_metrics_df["Accuracy"].std(ddof=0))], "Precision": [float(report_fold_metrics_df["Precision"].mean()), float(report_fold_metrics_df["Precision"].std(ddof=0))], "Recall": [float(report_fold_metrics_df["Recall"].mean()), float(report_fold_metrics_df["Recall"].std(ddof=0))], "F1": [float(report_fold_metrics_df["F1"].mean()), float(report_fold_metrics_df["F1"].std(ddof=0))]}, index=["mean", "std"])
    params_per_fold_df = pd.DataFrame(best_params_per_fold)
    final_section = "Final training/test dilewati (--skip-final)."
    if not args.skip_final:
        final_section = "\n".join([f"Model final: {final_cfg['candidate_name']}", f"Epoch final: {final_epochs}", f"ROC-AUC={safe_float(final_result['metrics']['roc_auc']):.4f} | PR-AUC={safe_float(final_result['metrics']['pr_auc']):.4f} | Accuracy={safe_float(final_result['metrics']['accuracy']):.4f} | Precision={safe_float(final_result['metrics']['precision']):.4f} | Recall={safe_float(final_result['metrics']['recall']):.4f} | F1={safe_float(final_result['metrics']['f1']):.4f}", "True Positive (TP) = " f"{int(final_result['metrics']['tp'])} | True Negative (TN) = {int(final_result['metrics']['tn'])} | False Positive (FP) = {int(final_result['metrics']['fp'])} | False Negative (FN) = {int(final_result['metrics']['fn'])}"])
    final_report = render_sections(
        [
            ("Detail tuning & evaluasi", "\n".join(detail_logs)),
            ("Per-fold metrics", dataframe_to_string(report_fold_metrics_df, index=False)),
            ("Aggregated metrics", dataframe_to_string(aggregated_metrics_df, index=True)),
            ("Model params per fold", dataframe_to_string(params_per_fold_df, index=False) if not params_per_fold_df.empty else "Tidak ada parameter fold."),
            ("Voting Best Model Param", "\n".join(voting_lines(best_params_per_fold))),
            ("Evaluasi final test", final_section),
        ],
        build_execution_report(start_time),
    )
    for line in final_report.rstrip().splitlines():
        log_info(line)
    print("\n[INFO] Selesai.")
    print(f"[INFO] Run         : {out_dirs['run_name']}")
    print(f"[INFO] Log root    : {out_dirs['log_root']}")
    print(f"[INFO] Fold root   : {out_dirs['fold_root']}")
    print(f"[INFO] Model root  : {out_dirs['model_root']}")


if __name__ == "__main__":
    main()
