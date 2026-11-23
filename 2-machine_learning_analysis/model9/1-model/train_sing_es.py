import argparse
import copy
import hashlib
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
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


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


PROJECT_DIR = model_root("model9")
PREPROCESSED_DIR = PROJECT_DIR / "0-feat" / "processed_gcn"
DATASET_NAME = "rki"
SINGLE_GAN_ROOT = PROJECT_ROOT / "cache" / "runs" / "benchmark"
_OUTPUT_ROOTS = {
    "log_root": SINGLE_GAN_ROOT / "b-log_result" / "model9_film_gcn_sing_es",
    "fold_root": SINGLE_GAN_ROOT / "a-fold" / "model9_film_gcn_sing_es",
    "save_model_root": SINGLE_GAN_ROOT / "c-model_save" / "model9_film_gcn_sing_es",
}
LOG_ROOT = _OUTPUT_ROOTS["log_root"]
FOLD_ROOT = _OUTPUT_ROOTS["fold_root"]
MODEL_ROOT = _OUTPUT_ROOTS["save_model_root"]

PEPTIDE_NODE_DIM = 31
PROTEIN_NODE_DIM = 31
PEPTIDE_EDGE_DIM = 3
PROTEIN_EDGE_DIM = 3

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

# Single-grid early-stopping training model9 baseline GCN: no FiLM and no AF.
# Variabel pada blok GRID inilah yang dibangkitkan menjadi kandidat via ParameterGrid.
# Catatan audit lama: `pairwise_lambda` dan `score_hidden` pernah tercantum di grid
# tetapi perlu dipastikan benar-benar memengaruhi arsitektur / loss saat eksperimen.
GRID: Dict[str, List[Any]] = {
    "epochs": [200],
    "batch": [64],
    "pos_weight": [1.0],
    "dropout": [0.2],
    "mlp_dropout": [0.15],
    "weight_decay": [3e-4],
    "lr": [5e-4],
    "pairwise_lambda": [0.0],
    "gat_hidden": [64],
    "heads": [4],
    "out_dim": [256],
    "mlp_hidden": [[256, 128]],
    "score_hidden": [128],
    "gnn_layers": [4],
    "global_mode": ["none"],
    "film_target": ["none"],
    "film_scale": [0.0],
}

def first_grid_value(key: str) -> Any:
    values = GRID.get(key)
    if not values:
        raise KeyError(f"GRID key tidak ditemukan atau kosong: {key}")
    return values[0]



def log_info(message: str) -> None:
    print(message, flush=True)
    if _ACTIVE_RESULTS_LOG is not None:
        _ACTIVE_RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _ACTIVE_RESULTS_LOG.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict[str, Any]:
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


def set_rng_state(state: Optional[Dict[str, Any]]) -> None:
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(to_serializable(payload), sort_keys=True, separators=(",", ":"))


def compute_config_signature(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def load_run_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "resume_meta": {},
            "folds": {},
            "final": {},
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    state.setdefault("resume_meta", {})
    state.setdefault("folds", {})
    state.setdefault("final", {})
    return state


def save_run_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_serializable(state), indent=2), encoding="utf-8")
    os.replace(tmp, path)


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
    return {
        "TP": int(cm[1, 1]),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
    }


def candidate_param_string(candidate_cfg: Dict[str, Any]) -> str:
    global_mode = str(candidate_cfg.get("global_mode", "none"))
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
            "pairwise_lambda": float(candidate_cfg["pairwise_lambda"]),
            "score_hidden": int(candidate_cfg["score_hidden"]),
            "gnn_layers": int(candidate_cfg["gnn_layers"]),
            "threshold": float(candidate_cfg["threshold"]),
            "global_mode": global_mode,
            "film_target": str(candidate_cfg.get("film_target", "protein")),
            "film_scale": safe_float(candidate_cfg.get("film_scale", 0.1)),
            "global_dim": int(candidate_cfg.get("global_dim", len(select_global_columns(global_mode)))),
        }
    )


def is_run_unfinished(state: Dict[str, Any]) -> bool:
    final_completed = bool(state.get("final", {}).get("completed"))
    if final_completed:
        return False
    folds = state.get("folds", {})
    if folds:
        return True
    return bool(state.get("final"))


def find_unfinished_runs(model_root: Path) -> List[str]:
    unfinished: List[str] = []
    if not model_root.exists():
        return unfinished
    for child in sorted(model_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        state_candidates = [
            child / f"model9_resume_{DATASET_NAME}.json",
            child / "run_state.json",
            MODEL_ROOT / child.name / "resume" / "run_state.json",
        ]
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


def assert_checkpoint_signature(payload: Dict[str, Any], expected_signature: str, label: str) -> None:
    saved_signature = str(payload.get("config_signature", "")).strip()
    if saved_signature and saved_signature != expected_signature:
        raise RuntimeError(
            f"Resume checkpoint mismatch pada {label}: "
            f"saved_signature={saved_signature}, expected_signature={expected_signature}."
        )


def load_preprocessed_data(base_dir: Path) -> Tuple[pd.DataFrame, List[Data], List[Data], np.ndarray]:
    metadata_path = base_dir / "metadata.csv"
    peptide_path = base_dir / "peptide_graphs.pt"
    protein_path = base_dir / "protein_graphs.pt"
    labels_path = base_dir / "y.npy"

    for required in [metadata_path, peptide_path, protein_path, labels_path]:
        if not required.exists():
            raise FileNotFoundError(f"File preprocess tidak ditemukan: {required}")

    metadata = pd.read_csv(metadata_path)
    peptide_graphs = list(safe_torch_load(peptide_path))
    protein_graphs = list(safe_torch_load(protein_path))
    labels = np.load(labels_path).reshape(-1).astype(np.int64)

    n_meta = len(metadata)
    n_pep = len(peptide_graphs)
    n_prot = len(protein_graphs)
    n_labels = len(labels)
    if len({n_meta, n_pep, n_prot, n_labels}) != 1:
        raise ValueError(
            "Jumlah sampel tidak sinkron: "
            f"metadata={n_meta}, peptide={n_pep}, protein={n_prot}, labels={n_labels}"
        )

    if n_pep == 0:
        raise ValueError("Dataset preprocess kosong.")

    pep_node_dim = int(peptide_graphs[0].x.shape[-1])
    pep_edge_dim = int(peptide_graphs[0].edge_attr.shape[-1])
    prot_node_dim = int(protein_graphs[0].x.shape[-1])
    prot_edge_dim = int(protein_graphs[0].edge_attr.shape[-1])

    if pep_node_dim != PEPTIDE_NODE_DIM:
        raise ValueError(f"Peptide node dim mismatch: found {pep_node_dim}, expected {PEPTIDE_NODE_DIM}")
    if pep_edge_dim != PEPTIDE_EDGE_DIM:
        raise ValueError(f"Peptide edge dim mismatch: found {pep_edge_dim}, expected {PEPTIDE_EDGE_DIM}")
    if prot_node_dim != PROTEIN_NODE_DIM:
        raise ValueError(f"Protein node dim mismatch: found {prot_node_dim}, expected {PROTEIN_NODE_DIM}")
    if prot_edge_dim != PROTEIN_EDGE_DIM:
        raise ValueError(f"Protein edge dim mismatch: found {prot_edge_dim}, expected {PROTEIN_EDGE_DIM}")

    return metadata, peptide_graphs, protein_graphs, labels



GLOBAL_FEATURE_NAMES = [
    "pep_len",
    "prot_len",
    "log_pep_len",
    "log_prot_len",
    "pep_prot_len_ratio",
]
GLOBAL_MODE_COLUMNS = {
    "none": [],
    "length": list(range(5)),
}


def build_global_feature_matrix(metadata: pd.DataFrame) -> np.ndarray:
    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in metadata.columns:
            return np.full(len(metadata), float(default), dtype=np.float32)
        return pd.to_numeric(metadata[name], errors="coerce").fillna(default).to_numpy(dtype=np.float32)

    pep_len = np.clip(col("pep_len"), 0.0, None)
    prot_len = np.clip(col("prot_len"), 0.0, None)
    safe_prot_len = np.maximum(prot_len, 1.0)
    features = np.stack(
        [
            pep_len,
            prot_len,
            np.log1p(pep_len),
            np.log1p(prot_len),
            pep_len / safe_prot_len,
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
    """Dataset pasangan peptide-protein graph."""

    def __init__(
        self,
        peptide_graphs: Sequence[Data],
        protein_graphs: Sequence[Data],
        labels: np.ndarray,
        indices: Sequence[int],
        metadata: Optional[pd.DataFrame] = None,
        global_features: Optional[np.ndarray] = None,
    ):
        self.peptide_graphs = peptide_graphs
        self.protein_graphs = protein_graphs
        self.labels = np.asarray(labels, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.metadata = metadata
        if global_features is None:
            global_features = np.zeros((len(self.labels), 0), dtype=np.float32)
        self.global_features = np.asarray(global_features, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        sample_idx = int(self.indices[idx])
        peptide_graph = self.peptide_graphs[sample_idx].clone()
        protein_graph = self.protein_graphs[sample_idx].clone()
        global_feature = torch.tensor(self.global_features[sample_idx], dtype=torch.float32)
        label = torch.tensor(self.labels[sample_idx], dtype=torch.float32)
        return peptide_graph, protein_graph, global_feature, label, sample_idx


def collate_fn(batch):
    peptide_graphs, protein_graphs, global_features, labels, indices = zip(*batch)
    return (
        Batch.from_data_list(list(peptide_graphs)),
        Batch.from_data_list(list(protein_graphs)),
        torch.stack(list(global_features), dim=0),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(indices, dtype=torch.long),
    )


class FiLMLayer(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.proj = nn.Sequential(nn.Linear(int(global_dim), int(hidden_dim) * 2), nn.Tanh())

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(global_features.to(dtype=x.dtype, device=x.device))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[batch_index]
        beta = beta[batch_index]
        return x * (1.0 + self.scale * gamma) + self.scale * beta


class GCNGraphEncoder(nn.Module):
    """Encoder GCN baseline; FiLM disabled when global_dim=0."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        global_dim: int = 0,
        film_scale: float = 0.1,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers harus >= 2")

        self.dropout = nn.Dropout(dropout)
        self.global_dim = int(global_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims: List[int] = []

        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))
        dims.append(hidden_dim)

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            dims.append(hidden_dim)

        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.norms.append(nn.BatchNorm1d(out_dim))
        dims.append(out_dim)
        self.film_layers = nn.ModuleList([FiLMLayer(self.global_dim, dim, film_scale) for dim in dims]) if self.global_dim > 0 else nn.ModuleList()

    def forward(self, batch: Batch, global_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch.x
        edge_index = batch.edge_index
        batch_index = batch.batch
        use_film = self.global_dim > 0 and global_features is not None and global_features.numel() > 0

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if use_film:
                x = self.film_layers[layer_idx](x, batch_index, global_features)
            x = F.relu(x)
            x = self.dropout(x)

        graph_emb = torch.cat([global_mean_pool(x, batch_index), global_max_pool(x, batch_index)], dim=-1)
        return x, batch_index, graph_emb


class PeptideProteinGCNModel(nn.Module):
    """Model9 baseline GCN: dual GCN encoder + pairwise graph head."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = dict(cfg)
        self.global_dim = int(cfg.get("global_dim", 0))
        film_target = str(cfg.get("film_target", "protein")).strip().lower()
        if film_target not in {"protein", "both", "none"}:
            raise ValueError(f"film_target tidak dikenal: {film_target}")
        film_scale = float(cfg.get("film_scale", 0.1))
        pep_global_dim = self.global_dim if film_target == "both" else 0
        prot_global_dim = self.global_dim if film_target in {"protein", "both"} else 0
        self.peptide_encoder = GCNGraphEncoder(PEPTIDE_NODE_DIM, PEPTIDE_EDGE_DIM, int(cfg["hidden_dim"]), int(cfg["out_dim"]), int(cfg["gnn_layers"]), float(cfg["dropout"]), global_dim=pep_global_dim, film_scale=film_scale)
        self.protein_encoder = GCNGraphEncoder(PROTEIN_NODE_DIM, PROTEIN_EDGE_DIM, int(cfg["hidden_dim"]), int(cfg["out_dim"]), int(cfg["gnn_layers"]), float(cfg["dropout"]), global_dim=prot_global_dim, film_scale=film_scale)

        mlp_layers: List[nn.Module] = []
        graph_dim = int(cfg["out_dim"]) * 2
        prev_dim = graph_dim * 4
        for width in cfg["mlp_hidden"]:
            mlp_layers.append(nn.Linear(prev_dim, int(width)))
            mlp_layers.append(nn.LayerNorm(int(width)))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(float(cfg["mlp_dropout"])))
            prev_dim = int(width)
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(self, pep_batch: Batch, prot_batch: Batch, global_features: Optional[torch.Tensor] = None, return_extras: bool = False):
        if self.global_dim > 0 and global_features is None:
            raise ValueError("global_features wajib ada saat global_dim > 0")
        _, _, pep_graph = self.peptide_encoder(pep_batch, global_features)
        _, _, prot_graph = self.protein_encoder(prot_batch, global_features)
        fused_feature = torch.cat([pep_graph, prot_graph, torch.abs(pep_graph - prot_graph), pep_graph * prot_graph], dim=-1)
        logits = self.classifier(fused_feature).squeeze(-1)
        if not return_extras:
            return logits
        return logits, {"probabilities": torch.sigmoid(logits)}

def build_model(cfg: Dict[str, Any]) -> PeptideProteinGCNModel:
    model = PeptideProteinGCNModel(cfg)
    return model.to(torch.device(cfg["device"]))


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: float,
    threshold: float,
) -> Dict[str, Any]:
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

    y_true_np = np.asarray(y_true_all, dtype=np.float32)
    y_prob_np = np.asarray(y_prob_all, dtype=np.float32)
    y_pred_np = np.asarray(y_pred_all, dtype=np.int64)
    metrics = compute_binary_metrics(y_true_np, y_prob_np, threshold=threshold)
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return {
        "metrics": metrics,
        "y_true": y_true_np,
        "y_prob": y_prob_np,
        "y_pred": y_pred_np,
        "sample_indices": np.asarray(sample_indices_all, dtype=np.int64),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    threshold: float,
) -> Dict[str, Any]:
    return run_epoch(model, loader, device, criterion, optimizer, grad_clip, threshold)


def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float,
) -> Dict[str, Any]:
    with torch.no_grad():
        return run_epoch(model, loader, device, criterion, None, 0.0, threshold)


def create_dataloader(
    peptide_graphs: Sequence[Data],
    protein_graphs: Sequence[Data],
    labels: np.ndarray,
    indices: Sequence[int],
    metadata: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    global_features: Optional[np.ndarray] = None,
) -> DataLoader:
    dataset = PairGraphDataset(peptide_graphs, protein_graphs, labels, indices, metadata=metadata, global_features=global_features)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=collate_fn,
        num_workers=int(num_workers),
        persistent_workers=False,
    )


def should_update_best(current: Dict[str, float], best: Optional[Dict[str, float]]) -> bool:
    if best is None:
        return True
    current_pr = safe_float(current.get("pr_auc", float("nan")))
    best_pr = safe_float(best.get("pr_auc", float("nan")))
    if not math.isnan(current_pr) and (math.isnan(best_pr) or current_pr > best_pr):
        return True
    if safe_float(current.get("pr_auc", float("nan"))) == safe_float(best.get("pr_auc", float("nan"))):
        current_roc = safe_float(current.get("roc_auc", float("nan")))
        best_roc = safe_float(best.get("roc_auc", float("nan")))
        if not math.isnan(current_roc) and (math.isnan(best_roc) or current_roc > best_roc):
            return True
    return False


def make_splits(labels: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, np.ndarray]]]:
    all_indices = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["random_seed"]),
        shuffle=True,
        stratify=labels,
    )

    skf = StratifiedKFold(
        n_splits=int(cfg["n_splits"]),
        shuffle=True,
        random_state=int(cfg["random_seed"]),
    )
    trainval_labels = labels[trainval_idx]
    folds: List[Dict[str, np.ndarray]] = []
    for fold_id, (tr_pos, val_pos) in enumerate(skf.split(np.zeros(len(trainval_idx)), trainval_labels), start=1):
        folds.append(
            {
                "fold": np.asarray([fold_id], dtype=np.int64),
                "train_indices": trainval_idx[np.asarray(tr_pos, dtype=np.int64)],
                "val_indices": trainval_idx[np.asarray(val_pos, dtype=np.int64)],
            }
        )
    return np.asarray(trainval_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64), folds


def parse_candidate_ids(raw: str) -> Optional[set]:
    text = str(raw or "").strip()
    if not text:
        return None
    result = set()
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            left, right = chunk.split('-', 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"candidate id range tidak valid: {chunk}")
            result.update(range(start, end + 1))
        else:
            result.add(int(chunk))
    return result


def build_candidate_configs(base_cfg: Dict[str, Any], candidate_ids: Optional[set] = None) -> List[Dict[str, Any]]:
    candidate_cfgs: List[Dict[str, Any]] = []
    for idx, override in enumerate(ParameterGrid(GRID), start=1):
        if candidate_ids is not None and idx not in candidate_ids:
            continue
        cfg = dict(base_cfg)
        cfg.update(
            {
                "epochs": int(base_cfg.get("epochs", override["epochs"])),
                "batch_size": int(base_cfg.get("batch_size", override["batch"])),
                "pos_weight": float(override["pos_weight"]),
                "dropout": float(override["dropout"]),
                "mlp_dropout": float(override["mlp_dropout"]),
                "weight_decay": float(override["weight_decay"]),
                "lr": float(override["lr"]),
                "pairwise_lambda": float(override["pairwise_lambda"]),
                "hidden_dim": int(override["gat_hidden"]),
                "heads": int(override["heads"]),
                "out_dim": int(override["out_dim"]),
                "mlp_hidden": list(override["mlp_hidden"]),
                "score_hidden": int(override["score_hidden"]),
                "gnn_layers": int(override["gnn_layers"]),
                "global_mode": str(override["global_mode"]),
                "film_target": str(override["film_target"]),
                "film_scale": float(override["film_scale"]),
            }
        )
        cfg["candidate_id"] = idx
        cfg["candidate_name"] = (
            f"cand{idx}_{override['global_mode']}_{override['film_target']}_fs{override['film_scale']}_"
            f"lr{override['lr']}_wd{override['weight_decay']}_"
            f"do{override['dropout']}_mdo{override['mlp_dropout']}_"
            f"mlp{'-'.join(str(x) for x in override['mlp_hidden'])}"
        )
        candidate_cfgs.append(cfg)
    return candidate_cfgs


def plot_kfold_roc(fold_predictions: List[Dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    auc_values = []
    mean_fpr = np.linspace(0.0, 1.0, 200)
    interp_tprs = []

    for item in fold_predictions:
        y_true = item["y_true"]
        y_prob = item["y_prob"]
        fold = item["fold"]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            auc_values.append(float(auc_val))
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
            plt.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f"Fold {fold} (AUC={auc_val:.3f})")
        except Exception:
            continue

    if interp_tprs:
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = float(np.mean(auc_values))
        std_auc = float(np.std(auc_values))
        plt.plot(mean_fpr, mean_tpr, color="black", lw=2.5, label=f"Mean ROC (AUC={mean_auc:.3f}±{std_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Across K-Folds")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_kfold_pr(fold_predictions: List[Dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    ap_values = []

    for item in fold_predictions:
        y_true = item["y_true"]
        y_prob = item["y_prob"]
        fold = item["fold"]
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ap_values.append(float(ap))
            plt.plot(recall, precision, lw=1.5, alpha=0.8, label=f"Fold {fold} (AP={ap:.3f})")
        except Exception:
            continue

    if ap_values:
        mean_ap = float(np.mean(ap_values))
        std_ap = float(np.std(ap_values))
        plt.title(f"Precision-Recall Curve Across K-Folds (AP={mean_ap:.3f}±{std_ap:.3f})")
    else:
        plt.title("Precision-Recall Curve Across K-Folds")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_avg_loss_acc_kfold(histories: List[Dict[str, Any]], out_path: Path) -> None:
    max_epochs = max(len(h.get("train_loss", [])) for h in histories)

    def pad_history(key: str) -> np.ndarray:
        rows = []
        for history in histories:
            values = np.asarray(history.get(key, []), dtype=np.float32)
            padded = np.full(max_epochs, np.nan, dtype=np.float32)
            padded[: len(values)] = values
            rows.append(padded)
        return np.vstack(rows)

    train_loss = pad_history("train_loss")
    val_loss = pad_history("val_loss")
    train_acc = pad_history("train_accuracy")
    val_acc = pad_history("val_accuracy")

    epochs = np.arange(1, max_epochs + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, np.nanmean(train_loss, axis=0), label="Train Loss", color="tab:blue")
    axes[0].plot(epochs, np.nanmean(val_loss, axis=0), label="Val Loss", color="tab:orange")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Average Loss Across K-Folds")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, np.nanmean(train_acc, axis=0), label="Train Accuracy", color="tab:green")
    axes[1].plot(epochs, np.nanmean(val_acc, axis=0), label="Val Accuracy", color="tab:red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Average Accuracy Across K-Folds")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_final_test_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, color="tab:blue", lw=2, label=f"AUC={auc_val:.3f}")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
        axes[0].set_title("Final Test ROC Curve")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend()
    except Exception:
        axes[0].text(0.5, 0.5, "ROC unavailable", ha="center", va="center")
        axes[0].set_title("Final Test ROC Curve")

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        axes[1].plot(recall, precision, color="tab:orange", lw=2, label=f"AP={ap:.3f}")
        axes[1].set_title("Final Test PR Curve")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend()
    except Exception:
        axes[1].text(0.5, 0.5, "PR unavailable", ha="center", va="center")
        axes[1].set_title("Final Test PR Curve")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_final_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Final Test Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_kfold_predictions_npz(fold_predictions: List[Dict[str, Any]], out_path: Path) -> None:
    np.savez_compressed(
        out_path,
        folds=np.asarray([item["fold"] for item in fold_predictions], dtype=np.int64),
        sample_indices=np.asarray([item["sample_indices"] for item in fold_predictions], dtype=object),
        y_true=np.asarray([item["y_true"] for item in fold_predictions], dtype=object),
        y_prob=np.asarray([item["y_prob"] for item in fold_predictions], dtype=object),
        y_pred=np.asarray([item["y_pred"] for item in fold_predictions], dtype=object),
    )


def save_final_predictions_npz(payload: Dict[str, Any], out_path: Path) -> None:
    np.savez_compressed(
        out_path,
        sample_indices=np.asarray(payload["sample_indices"], dtype=np.int64),
        y_true=np.asarray(payload["y_true"], dtype=np.float32),
        y_prob=np.asarray(payload["y_prob"], dtype=np.float32),
        y_pred=np.asarray(payload["y_pred"], dtype=np.int64),
    )


def make_output_dirs(run_name: str) -> Dict[str, Path]:
    log_root = ensure_dir(LOG_ROOT)
    fold_root = ensure_dir(FOLD_ROOT)
    model_root = ensure_dir(MODEL_ROOT)
    resume_root = ensure_dir(model_root / "resume")
    return {
        "run_name": run_name,
        "log_root": log_root,
        "fold_root": fold_root,
        "model_root": model_root,
        "text": log_root / "model9_film_gcn_sing_es_results_rki.txt",
        "checkpoints": ensure_dir(model_root / "checkpoints"),
        "resume": resume_root,
        "state": log_root / "model9_film_gcn_sing_es_resume_rki.json",
        "plots": ensure_dir(log_root / "plots"),
        "logs": ensure_dir(log_root / "logs"),
        "predictions": ensure_dir(fold_root / "predictions"),
        "reports": ensure_dir(log_root / "reports"),
        "fold_reports": ensure_dir(fold_root / "reports"),
    }


def run_kfold_training(
    peptide_graphs: Sequence[Data],
    protein_graphs: Sequence[Data],
    labels: np.ndarray,
    metadata: pd.DataFrame,
    global_raw: np.ndarray,
    fold_specs: List[Dict[str, np.ndarray]],
    candidate_cfgs: List[Dict[str, Any]],
    out_dirs: Dict[str, Path],
    run_state: Dict[str, Any],
    config_signature: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[int],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[str],
    List[Dict[str, Any]],
]:
    fold_metrics: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    fold_predictions: List[Dict[str, Any]] = []
    best_epochs: List[int] = []
    best_candidate_names: List[str] = []
    detail_logs: List[str] = []
    best_params_per_fold: List[Dict[str, Any]] = []

    for fold_idx, fold_spec in enumerate(fold_specs, start=1):
        log_info(f"\n[INFO] Fold {fold_idx}/{len(fold_specs)}")
        fold_key = str(fold_idx)
        saved_fold = run_state.get("folds", {}).get(fold_key)
        if saved_fold and saved_fold.get("completed"):
            log_info(f"[RESUME] Skip fold {fold_idx}")
            fold_metrics.append(saved_fold["fold_metric"])
            histories.append(saved_fold["history"])
            best_epochs.append(int(saved_fold["best_epoch"]))
            fold_predictions.append(saved_fold["fold_prediction"])
            best_candidate_names.append(str(saved_fold["best_cfg"]["candidate_name"]))
            saved_detail = saved_fold.get("detail_logs", [])
            if saved_detail:
                detail_logs.extend(str(line) for line in saved_detail)
            if saved_fold.get("best_cfg"):
                best_cfg_saved = saved_fold["best_cfg"]
                best_params_per_fold.append(
                    {
                        "fold": fold_idx,
                        "candidate_id": int(best_cfg_saved["candidate_id"]),
                        "candidate_name": str(best_cfg_saved["candidate_name"]),
                        "lr": float(best_cfg_saved["lr"]),
                        "weight_decay": float(best_cfg_saved["weight_decay"]),
                        "dropout": float(best_cfg_saved["dropout"]),
                        "mlp_dropout": float(best_cfg_saved["mlp_dropout"]),
                        "pairwise_lambda": float(best_cfg_saved["pairwise_lambda"]),
                        "hidden_dim": int(best_cfg_saved["hidden_dim"]),
                        "heads": int(best_cfg_saved["heads"]),
                        "out_dim": int(best_cfg_saved["out_dim"]),
                        "gnn_layers": int(best_cfg_saved["gnn_layers"]),
                        "global_mode": str(best_cfg_saved.get("global_mode", "none")),
                        "film_target": str(best_cfg_saved.get("film_target", "protein")),
                        "film_scale": float(best_cfg_saved.get("film_scale", 0.1)),
                        "global_dim": int(best_cfg_saved.get("global_dim", 0)),
                        "mlp_hidden": "-".join(str(x) for x in best_cfg_saved["mlp_hidden"]),
                        "best_epoch": int(saved_fold["best_epoch"]),
                    }
                )
            continue
        run_state.setdefault("folds", {})
        run_state["folds"].setdefault(
            fold_key,
            {
                "completed": False,
                "candidates": {},
            },
        )
        save_run_state(out_dirs["state"], run_state)
        train_indices = fold_spec["train_indices"]
        val_indices = fold_spec["val_indices"]

        fold_best_cfg = None
        fold_best_metrics = None
        fold_best_state = None
        fold_best_epoch = 1
        fold_best_result = None
        fold_best_history = None
        fold_detail_lines = [f"Fold {fold_idx}/{len(fold_specs)}"]

        for candidate_cfg in candidate_cfgs:
            candidate_start = time.perf_counter()
            candidate_key = str(candidate_cfg["candidate_id"])
            candidate_resume_path = out_dirs["resume"] / f"fold{fold_idx}_candidate{candidate_cfg['candidate_id']}.pth"
            fold_state = run_state.setdefault("folds", {}).setdefault(
                fold_key,
                {
                    "completed": False,
                    "candidates": {},
                },
            )
            candidates_state = fold_state.setdefault("candidates", {})
            had_candidate_state = candidate_key in candidates_state
            candidate_state = candidates_state.get(candidate_key, {})

            if not had_candidate_state:
                candidates_state[candidate_key] = {
                    "completed": False,
                    "resume_path": str(candidate_resume_path),
                }
                candidate_state = candidates_state[candidate_key]
                save_run_state(out_dirs["state"], run_state)
            elif not candidate_state.get("completed") and not candidate_resume_path.exists():
                log_info(
                    f"[RESUME] Checkpoint incomplete hilang untuk fold {fold_idx} "
                    f"candidate {candidate_cfg['candidate_name']}. "
                    "Candidate akan dihitung ulang dari awal."
                )

            if candidate_state.get("completed"):
                if not candidate_resume_path.exists():
                    log_info(
                        f"[RESUME] Candidate checkpoint hilang untuk fold {fold_idx} "
                        f"candidate {candidate_cfg['candidate_name']}. Candidate akan dihitung ulang."
                    )
                else:
                    try:
                        resume_payload = safe_torch_load(candidate_resume_path)
                        assert_checkpoint_signature(
                            resume_payload,
                            config_signature,
                            f"fold {fold_idx} candidate {candidate_cfg['candidate_name']}",
                        )
                        best_state = resume_payload.get("best_state_dict")
                        best_val_metrics = resume_payload.get("best_val_metrics")
                        best_epoch = int(resume_payload.get("best_epoch", 1))
                        best_val_result = resume_payload.get("best_val_result")
                        history = resume_payload.get("history")
                        if (
                            not resume_payload.get("completed")
                            or best_state is None
                            or best_val_metrics is None
                            or best_val_result is None
                            or history is None
                        ):
                            raise RuntimeError("candidate summary checkpoint tidak lengkap")
                        log_info(
                            f"[RESUME] Skip fold {fold_idx} candidate "
                            f"{candidate_cfg['candidate_name']} karena sudah selesai."
                        )
                        if should_update_best(best_val_metrics, fold_best_metrics):
                            fold_best_cfg = dict(candidate_cfg)
                            fold_best_metrics = dict(best_val_metrics)
                            fold_best_state = copy.deepcopy(best_state)
                            fold_best_epoch = int(best_epoch)
                            fold_best_result = best_val_result
                            fold_best_history = history
                        saved_elapsed = candidate_state.get("elapsed_seconds")
                        saved_confusion = candidate_state.get("confusion")
                        if best_val_result is not None and saved_confusion is None:
                            saved_confusion = binary_confusion_counts(
                                np.asarray(best_val_result["y_true"]),
                                np.asarray(best_val_result["y_pred"]),
                            )
                        candidate_lines = render_candidate_detail(
                            index=int(candidate_cfg["candidate_id"]),
                            param_string=candidate_param_string(candidate_cfg),
                            metrics=snakebite_metrics(best_val_metrics),
                            confusion=saved_confusion or {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
                            elapsed_seconds=safe_float(saved_elapsed) if saved_elapsed is not None else None,
                        )
                        fold_detail_lines.extend(candidate_lines)
                        for line in candidate_lines:
                            log_info(line)
                        continue
                    except Exception as exc:
                        log_info(
                            f"[RESUME] Candidate checkpoint rusak/tidak valid untuk fold {fold_idx} "
                            f"candidate {candidate_cfg['candidate_name']}: {exc}. "
                            "Candidate akan dihitung ulang."
                        )

            device = torch.device(candidate_cfg["device"])
            global_features = normalize_global_features(global_raw, train_indices, str(candidate_cfg.get("global_mode", "none")))
            candidate_cfg = dict(candidate_cfg)
            candidate_cfg["global_dim"] = int(global_features.shape[1])
            train_loader = create_dataloader(
                peptide_graphs, protein_graphs, labels, train_indices, metadata,
                batch_size=candidate_cfg["batch_size"], shuffle=True, num_workers=candidate_cfg["num_workers"],
                global_features=global_features,
            )
            val_loader = create_dataloader(
                peptide_graphs, protein_graphs, labels, val_indices, metadata,
                batch_size=candidate_cfg["batch_size"], shuffle=False, num_workers=candidate_cfg["num_workers"],
                global_features=global_features,
            )

            log_info(
                f"[INFO] Fold {fold_idx} | Kandidat {candidate_cfg['candidate_id']}/{len(candidate_cfgs)} "
                f"({candidate_cfg['candidate_name']})"
            )

            model = build_model(candidate_cfg)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(candidate_cfg["lr"]),
                weight_decay=float(candidate_cfg["weight_decay"]),
            )
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([float(candidate_cfg["pos_weight"])], device=device),
                reduction="mean",
            )

            history = {
                "fold": fold_idx,
                "candidate_id": int(candidate_cfg["candidate_id"]),
                "candidate_name": candidate_cfg["candidate_name"],
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
            best_val_result = None
            start_epoch = 1

            if candidate_resume_path.exists():
                try:
                    resume_payload = safe_torch_load(candidate_resume_path)
                    assert_checkpoint_signature(
                        resume_payload,
                        config_signature,
                        f"fold {fold_idx} candidate {candidate_cfg['candidate_name']}",
                    )
                    if not resume_payload.get("completed"):
                        model.load_state_dict(resume_payload["model_state_dict"])
                        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                        history = resume_payload.get("history", history)
                        best_state = resume_payload.get("best_state_dict")
                        best_val_metrics = resume_payload.get("best_val_metrics")
                        best_val_result = resume_payload.get("best_val_result")
                        best_epoch = int(resume_payload.get("best_epoch", 1))
                        set_rng_state(resume_payload.get("rng_state"))
                        start_epoch = int(resume_payload.get("epoch", 0)) + 1
                        if start_epoch <= int(candidate_cfg["epochs"]):
                            log_info(
                                f"[RESUME] Resume fold {fold_idx} candidate "
                                f"{candidate_cfg['candidate_name']} dari epoch {start_epoch}."
                            )
                        else:
                            log_info(
                                f"[RESUME] Checkpoint incomplete untuk fold {fold_idx} "
                                f"candidate {candidate_cfg['candidate_name']} melewati total epoch. "
                                "Candidate akan dihitung ulang dari awal."
                            )
                            history = {
                                "fold": fold_idx,
                                "candidate_id": int(candidate_cfg["candidate_id"]),
                                "candidate_name": candidate_cfg["candidate_name"],
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
                            best_val_result = None
                            start_epoch = 1
                except Exception as exc:
                    log_info(
                        f"[RESUME] Checkpoint incomplete rusak/tidak valid untuk fold {fold_idx} "
                        f"candidate {candidate_cfg['candidate_name']}: {exc}. "
                        "Candidate akan dihitung ulang dari awal."
                    )
                    history = {
                        "fold": fold_idx,
                        "candidate_id": int(candidate_cfg["candidate_id"]),
                        "candidate_name": candidate_cfg["candidate_name"],
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
                    best_val_result = None
                    start_epoch = 1

            early_stop_patience = int(candidate_cfg.get("early_stopping_patience", 0))
            early_stop_min_delta = float(candidate_cfg.get("early_stopping_min_delta", 0.0))
            best_monitor_score = safe_float(best_val_metrics.get("pr_auc", float("nan"))) if best_val_metrics is not None else float("-inf")
            epochs_without_improvement = 0

            for epoch in range(start_epoch, int(candidate_cfg["epochs"]) + 1):
                if epoch == 1 or epoch % int(candidate_cfg["log_every"]) == 0 or epoch == int(candidate_cfg["epochs"]):
                    log_info(
                        f"[INFO] Fold {fold_idx} | Cand {candidate_cfg['candidate_id']} | "
                        f"Epoch {epoch}/{candidate_cfg['epochs']} dimulai..."
                    )
                train_result = train_one_epoch(
                    model, train_loader, device, criterion, optimizer,
                    grad_clip=float(candidate_cfg["grad_clip"]), threshold=float(candidate_cfg["threshold"])
                )
                val_result = evaluate_one_epoch(
                    model, val_loader, device, criterion, threshold=float(candidate_cfg["threshold"])
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

                if epoch == 1 or epoch % int(candidate_cfg["log_every"]) == 0 or epoch == int(candidate_cfg["epochs"]):
                    log_info(
                        f"[INFO] Fold {fold_idx} | Cand {candidate_cfg['candidate_id']} | Epoch {epoch}/{candidate_cfg['epochs']} | "
                        f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
                        f"prec={train_metrics['precision']:.4f} rec={train_metrics['recall']:.4f} "
                        f"f1={train_metrics['f1']:.4f} roc={train_metrics['roc_auc']:.4f} pr={train_metrics['pr_auc']:.4f} | "
                        f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
                        f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} "
                        f"f1={val_metrics['f1']:.4f} roc={val_metrics['roc_auc']:.4f} pr={val_metrics['pr_auc']:.4f}"
                    )

                if should_update_best(val_metrics, best_val_metrics):
                    best_val_metrics = dict(val_metrics)
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = int(epoch)
                    best_val_result = val_result

                current_monitor_score = safe_float(val_metrics.get("pr_auc", float("nan")))
                if not math.isnan(current_monitor_score) and current_monitor_score > best_monitor_score + early_stop_min_delta:
                    best_monitor_score = current_monitor_score
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                should_stop_early = early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience
                if should_stop_early:
                    log_info(
                        "[EARLY STOP] Fold {} | Cand {} berhenti di epoch {}/{} | monitor=val_pr_auc | best_epoch={} | best_pr_auc={:.4f}".format(
                            fold_idx, candidate_cfg["candidate_id"], epoch, candidate_cfg["epochs"], best_epoch, best_monitor_score
                        )
                    )

                safe_torch_save(
                    candidate_resume_path,
                    {
                        "completed": False,
                        "epoch": int(epoch),
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                        "history": copy.deepcopy(history),
                        "best_state_dict": copy.deepcopy(best_state),
                        "best_val_metrics": copy.deepcopy(best_val_metrics),
                        "best_val_result": copy.deepcopy(best_val_result),
                        "best_epoch": int(best_epoch),
                        "candidate_cfg": copy.deepcopy(candidate_cfg),
                        "config_signature": config_signature,
                        "rng_state": get_rng_state(),
                    },
                )

                if should_stop_early:
                    break

            if best_state is None or best_val_result is None or best_val_metrics is None:
                raise RuntimeError(f"Fold {fold_idx} kandidat {candidate_cfg['candidate_name']} gagal mendapatkan checkpoint terbaik.")

            safe_torch_save(
                candidate_resume_path,
                {
                    "completed": True,
                    "candidate_cfg": copy.deepcopy(candidate_cfg),
                    "best_epoch": int(best_epoch),
                    "best_val_metrics": copy.deepcopy(best_val_metrics),
                    "best_val_result": copy.deepcopy(best_val_result),
                    "best_state_dict": copy.deepcopy(best_state),
                    "history": copy.deepcopy(history),
                    "config_signature": config_signature,
                },
            )
            candidate_elapsed = time.perf_counter() - candidate_start
            candidate_confusion = binary_confusion_counts(best_val_result["y_true"], best_val_result["y_pred"])
            fold_state["candidates"][candidate_key] = {
                "completed": True,
                "resume_path": str(candidate_resume_path),
                "best_epoch": int(best_epoch),
                "best_val_metrics": to_serializable(best_val_metrics),
                "confusion": candidate_confusion,
                "elapsed_seconds": float(candidate_elapsed),
            }
            save_run_state(out_dirs["state"], run_state)
            candidate_lines = render_candidate_detail(
                index=int(candidate_cfg["candidate_id"]),
                param_string=candidate_param_string(candidate_cfg),
                metrics=snakebite_metrics(best_val_metrics),
                confusion=candidate_confusion,
                elapsed_seconds=candidate_elapsed,
            )
            fold_detail_lines.extend(candidate_lines)
            for line in candidate_lines:
                log_info(line)

            if should_update_best(best_val_metrics, fold_best_metrics):
                fold_best_cfg = dict(candidate_cfg)
                fold_best_metrics = dict(best_val_metrics)
                fold_best_state = copy.deepcopy(best_state)
                fold_best_epoch = int(best_epoch)
                fold_best_result = best_val_result
                fold_best_history = history

        if fold_best_state is None or fold_best_result is None or fold_best_metrics is None or fold_best_cfg is None or fold_best_history is None:
            raise RuntimeError(f"Fold {fold_idx} gagal mendapatkan kandidat terbaik.")

        checkpoint_path = out_dirs["checkpoints"] / f"best_model_fold{fold_idx}.pth"
        safe_torch_save(
            checkpoint_path,
            {
                "model_state_dict": fold_best_state,
                "config": fold_best_cfg,
                "fold": fold_idx,
                "best_epoch": fold_best_epoch,
                "best_val_metrics": fold_best_metrics,
            },
        )

        best_epochs.append(fold_best_epoch)
        best_candidate_names.append(str(fold_best_cfg["candidate_name"]))
        log_info(
            f"[INFO] Fold {fold_idx} terbaik -> {fold_best_cfg['candidate_name']} | "
            f"epoch={fold_best_epoch} | val_acc={fold_best_metrics['accuracy']:.4f} | "
            f"val_prec={fold_best_metrics['precision']:.4f} | val_rec={fold_best_metrics['recall']:.4f} | "
            f"val_f1={fold_best_metrics['f1']:.4f} | val_roc={fold_best_metrics['roc_auc']:.4f} | "
            f"val_pr={fold_best_metrics['pr_auc']:.4f}"
        )
        fold_confusion = binary_confusion_counts(fold_best_result["y_true"], fold_best_result["y_pred"])
        selected_lines = render_selected_detail(
            param_string=candidate_param_string(fold_best_cfg),
            selection_metric_name="PR-AUC",
            selection_metric_value=safe_float(fold_best_metrics["pr_auc"]),
            metrics=snakebite_metrics(fold_best_metrics),
            confusion=fold_confusion,
        )
        fold_detail_lines.extend(selected_lines)
        for line in selected_lines:
            log_info(line)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "candidate_id": int(fold_best_cfg["candidate_id"]),
                "candidate_name": fold_best_cfg["candidate_name"],
                "best_epoch": fold_best_epoch,
                "val_loss": float(fold_best_metrics["loss"]),
                "val_accuracy": float(fold_best_metrics["accuracy"]),
                "val_precision": float(fold_best_metrics["precision"]),
                "val_recall": float(fold_best_metrics["recall"]),
                "val_f1": float(fold_best_metrics["f1"]),
                "val_roc_auc": float(fold_best_metrics["roc_auc"]),
                "val_pr_auc": float(fold_best_metrics["pr_auc"]),
                "threshold": float(fold_best_cfg["threshold"]),
                "tp": int(fold_confusion["TP"]),
                "tn": int(fold_confusion["TN"]),
                "fp": int(fold_confusion["FP"]),
                "fn": int(fold_confusion["FN"]),
            }
        )
        histories.append(fold_best_history)
        fold_predictions.append(
            {
                "fold": fold_idx,
                "candidate_id": int(fold_best_cfg["candidate_id"]),
                "candidate_name": fold_best_cfg["candidate_name"],
                "sample_indices": fold_best_result["sample_indices"],
                "y_true": fold_best_result["y_true"],
                "y_prob": fold_best_result["y_prob"],
                "y_pred": fold_best_result["y_pred"],
            }
        )
        run_state["folds"][fold_key]["completed"] = True
        run_state["folds"][fold_key]["best_epoch"] = int(fold_best_epoch)
        run_state["folds"][fold_key]["best_cfg"] = to_serializable(fold_best_cfg)
        run_state["folds"][fold_key]["fold_metric"] = to_serializable(fold_metrics[-1])
        run_state["folds"][fold_key]["history"] = to_serializable(fold_best_history)
        run_state["folds"][fold_key]["fold_prediction"] = to_serializable(fold_predictions[-1])
        run_state["folds"][fold_key]["detail_logs"] = list(fold_detail_lines)
        save_run_state(out_dirs["state"], run_state)
        detail_logs.extend(fold_detail_lines)
        best_params_per_fold.append(
            {
                "fold": fold_idx,
                "candidate_id": int(fold_best_cfg["candidate_id"]),
                "candidate_name": str(fold_best_cfg["candidate_name"]),
                "lr": float(fold_best_cfg["lr"]),
                "weight_decay": float(fold_best_cfg["weight_decay"]),
                "dropout": float(fold_best_cfg["dropout"]),
                "mlp_dropout": float(fold_best_cfg["mlp_dropout"]),
                "pairwise_lambda": float(fold_best_cfg["pairwise_lambda"]),
                "hidden_dim": int(fold_best_cfg["hidden_dim"]),
                "heads": int(fold_best_cfg["heads"]),
                "out_dim": int(fold_best_cfg["out_dim"]),
                "gnn_layers": int(fold_best_cfg["gnn_layers"]),
                "global_mode": str(fold_best_cfg.get("global_mode", "none")),
                "film_target": str(fold_best_cfg.get("film_target", "protein")),
                "film_scale": float(fold_best_cfg.get("film_scale", 0.1)),
                "global_dim": int(fold_best_cfg.get("global_dim", 0)),
                "mlp_hidden": "-".join(str(x) for x in fold_best_cfg["mlp_hidden"]),
                "best_epoch": int(fold_best_epoch),
            }
        )

    best_name = max(set(best_candidate_names), key=best_candidate_names.count)
    final_cfg = None
    for cfg in candidate_cfgs:
        if str(cfg["candidate_name"]) == best_name:
            final_cfg = dict(cfg)
            break
    if final_cfg is None:
        final_cfg = dict(candidate_cfgs[0])

    return fold_metrics, histories, best_epochs, fold_predictions, final_cfg, detail_logs, best_params_per_fold


def train_final_model(
    peptide_graphs: Sequence[Data],
    protein_graphs: Sequence[Data],
    labels: np.ndarray,
    metadata: pd.DataFrame,
    global_raw: np.ndarray,
    trainval_indices: np.ndarray,
    test_indices: np.ndarray,
    cfg: Dict[str, Any],
    out_dirs: Dict[str, Path],
    final_epochs: int,
    run_state: Dict[str, Any],
    config_signature: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    device = torch.device(cfg["device"])
    global_features = normalize_global_features(global_raw, trainval_indices, str(cfg.get("global_mode", "none")))
    cfg = dict(cfg)
    cfg["global_dim"] = int(global_features.shape[1])

    train_loader = create_dataloader(
        peptide_graphs, protein_graphs, labels, trainval_indices, metadata,
        batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
        global_features=global_features,
    )
    test_loader = create_dataloader(
        peptide_graphs, protein_graphs, labels, test_indices, metadata,
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
        global_features=global_features,
    )

    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(cfg["pos_weight"])], device=device),
        reduction="mean",
    )

    final_train_history = {
        "epochs": final_epochs,
        "train_loss": [],
        "train_accuracy": [],
        "train_roc_auc": [],
        "train_pr_auc": [],
    }
    final_state = run_state.setdefault("final", {})
    final_resume_path = out_dirs["resume"] / "final_training.pth"

    if final_state.get("completed"):
        if final_resume_path.exists():
            try:
                resume_payload = safe_torch_load(final_resume_path)
                assert_checkpoint_signature(resume_payload, config_signature, "final training")
                if resume_payload.get("completed"):
                    result = resume_payload.get("result")
                    prediction_payload = resume_payload.get("prediction_payload")
                    if result is not None and prediction_payload is not None:
                        log_info("[RESUME] Skip final training karena sudah selesai.")
                        return result, prediction_payload
            except Exception as exc:
                log_info(
                    f"[RESUME] Final checkpoint completed rusak/tidak valid: {exc}. "
                    "Mencoba memakai hasil final dari run_state."
                )
        saved_result = final_state.get("result")
        saved_prediction_payload = final_state.get("prediction_payload")
        if saved_result is not None and saved_prediction_payload is not None:
            log_info("[RESUME] Skip final training karena hasil final sudah ada di run_state.")
            return saved_result, saved_prediction_payload

    start_epoch = 1
    if final_resume_path.exists():
        try:
            resume_payload = safe_torch_load(final_resume_path)
            assert_checkpoint_signature(resume_payload, config_signature, "final training")
            if not resume_payload.get("completed"):
                model.load_state_dict(resume_payload["model_state_dict"])
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                final_train_history = resume_payload.get("history", final_train_history)
                set_rng_state(resume_payload.get("rng_state"))
                start_epoch = int(resume_payload.get("epoch", 0)) + 1
                if start_epoch <= int(final_epochs):
                    log_info(f"[RESUME] Resume final training dari epoch {start_epoch}.")
                else:
                    log_info(
                        "[RESUME] Checkpoint incomplete final training melewati total epoch. "
                        "Final training akan dihitung ulang dari awal."
                    )
                    final_train_history = {
                        "epochs": final_epochs,
                        "train_loss": [],
                        "train_accuracy": [],
                        "train_roc_auc": [],
                        "train_pr_auc": [],
                    }
                    start_epoch = 1
        except Exception as exc:
            log_info(
                f"[RESUME] Checkpoint incomplete final training rusak/tidak valid: {exc}. "
                "Final training akan dihitung ulang dari awal."
            )
            final_train_history = {
                "epochs": final_epochs,
                "train_loss": [],
                "train_accuracy": [],
                "train_roc_auc": [],
                "train_pr_auc": [],
            }
            start_epoch = 1

    for epoch in range(start_epoch, int(final_epochs) + 1):
        if epoch == 1 or epoch % int(cfg["log_every"]) == 0 or epoch == int(final_epochs):
            log_info(f"[INFO] Final Train | Epoch {epoch}/{final_epochs} dimulai...")
        train_result = train_one_epoch(
            model, train_loader, device, criterion, optimizer,
            grad_clip=float(cfg["grad_clip"]), threshold=float(cfg["threshold"])
        )
        train_metrics = train_result["metrics"]
        final_train_history["train_loss"].append(float(train_metrics["loss"]))
        final_train_history["train_accuracy"].append(float(train_metrics["accuracy"]))
        final_train_history["train_roc_auc"].append(float(train_metrics["roc_auc"]))
        final_train_history["train_pr_auc"].append(float(train_metrics["pr_auc"]))

        if epoch == 1 or epoch % int(cfg["log_every"]) == 0 or epoch == int(final_epochs):
            log_info(
                f"[INFO] Final Train | Epoch {epoch}/{final_epochs} | "
                f"loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
                f"prec={train_metrics['precision']:.4f} rec={train_metrics['recall']:.4f} "
                f"f1={train_metrics['f1']:.4f} roc={train_metrics['roc_auc']:.4f} pr={train_metrics['pr_auc']:.4f}"
            )
        safe_torch_save(
            final_resume_path,
            {
                "completed": False,
                "epoch": int(epoch),
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "history": copy.deepcopy(final_train_history),
                "result": None,
                "prediction_payload": None,
                "config_signature": config_signature,
                "rng_state": get_rng_state(),
            },
        )

    final_checkpoint_path = out_dirs["checkpoints"] / "final_model.pth"
    safe_torch_save(
        final_checkpoint_path,
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "final_epochs": int(final_epochs),
        },
    )

    test_result = evaluate_one_epoch(
        model, test_loader, device, criterion, threshold=float(cfg["threshold"])
    )
    test_metrics = dict(test_result["metrics"])
    cm = confusion_matrix(test_result["y_true"].astype(int), test_result["y_pred"].astype(int), labels=[0, 1])
    test_metrics["confusion_matrix"] = cm.tolist()
    test_metrics["tn"] = int(cm[0, 0])
    test_metrics["fp"] = int(cm[0, 1])
    test_metrics["fn"] = int(cm[1, 0])
    test_metrics["tp"] = int(cm[1, 1])
    test_metrics["final_epochs"] = int(final_epochs)
    log_info(
        "[INFO] Final Test | "
        f"loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} "
        f"prec={test_metrics['precision']:.4f} rec={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f} roc={test_metrics['roc_auc']:.4f} pr={test_metrics['pr_auc']:.4f} | "
        f"TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']} TP={test_metrics['tp']}"
    )

    final_prediction_payload = {
        "sample_indices": test_result["sample_indices"],
        "y_true": test_result["y_true"],
        "y_prob": test_result["y_prob"],
        "y_pred": test_result["y_pred"],
    }
    result = {
        "metrics": test_metrics,
        "history": final_train_history,
        "checkpoint_path": str(final_checkpoint_path),
    }
    safe_torch_save(
        final_resume_path,
        {
            "completed": True,
            "epoch": int(final_epochs),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "history": copy.deepcopy(final_train_history),
            "result": copy.deepcopy(result),
            "prediction_payload": copy.deepcopy(final_prediction_payload),
            "config_signature": config_signature,
            "rng_state": get_rng_state(),
        },
    )
    run_state["final"] = {
        "completed": True,
        "resume_path": str(final_resume_path),
        "result": to_serializable(result),
        "prediction_payload": to_serializable(final_prediction_payload),
    }
    save_run_state(out_dirs["state"], run_state)
    return result, final_prediction_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model9 baseline GCN: no FiLM and no AF.")
    parser.add_argument("--data-dir", type=Path, default=PREPROCESSED_DIR, help="Folder hasil preprocessing.")
    parser.add_argument("--epochs", type=int, default=int(first_grid_value("epochs")))
    parser.add_argument("--batch-size", type=int, default=int(first_grid_value("batch")))
    parser.add_argument("--hidden-dim", type=int, default=int(first_grid_value("gat_hidden")))
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
    parser.add_argument("--run-name", type=str, default="", help="Nama run untuk metadata. Default: model9_film_gcn_sing_es.")
    parser.add_argument("--resume-run", type=str, default="", help="Nama run untuk melanjutkan training yang terputus.")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "hidden_dim": int(args.hidden_dim),
            "out_dim": int(args.out_dim),
            "heads": int(args.heads),
            "gnn_layers": int(args.gnn_layers),
            "dropout": float(args.dropout),
            "mlp_dropout": float(args.mlp_dropout),
            "mlp_hidden": list(first_grid_value("mlp_hidden")),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pos_weight": float(first_grid_value("pos_weight")),
            "pairwise_lambda": float(first_grid_value("pairwise_lambda")),
            "score_hidden": int(first_grid_value("score_hidden")),
            "global_mode": str(first_grid_value("global_mode")),
            "film_target": str(first_grid_value("film_target")),
            "film_scale": float(first_grid_value("film_scale")),
            "n_splits": int(args.n_splits),
            "test_size": float(args.test_size),
            "random_seed": int(args.random_seed),
            "threshold": float(args.threshold),
            "device": str(args.device),
            "log_every": int(args.log_every),
        }
    )
    return cfg


def summarize_dataset(labels: np.ndarray, metadata: pd.DataFrame) -> Dict[str, Any]:
    return {
        "num_samples": int(len(labels)),
        "num_positive": int(np.sum(labels == 1)),
        "num_negative": int(np.sum(labels == 0)),
        "avg_peptide_len": float(pd.to_numeric(metadata["pep_len"], errors="coerce").fillna(0).mean()) if "pep_len" in metadata.columns else None,
        "avg_protein_len": float(pd.to_numeric(metadata["prot_len"], errors="coerce").fillna(0).mean()) if "prot_len" in metadata.columns else None,
    }


_ACTIVE_RESULTS_LOG: Optional[Path] = None


def main() -> None:
    global _ACTIVE_RESULTS_LOG
    start_time = time.perf_counter()
    args = parse_args()
    cfg = build_config_from_args(args)
    set_all_seeds(int(cfg["random_seed"]))
    if not args.resume_run.strip():
        unfinished_runs = find_unfinished_runs(MODEL_ROOT)
        if unfinished_runs:
            log_info(
                f"[RESUME] Found unfinished run: {unfinished_runs[0]}. "
                f"Use --resume-run {unfinished_runs[0]} to continue."
            )

    if args.resume_run.strip():
        run_name = args.resume_run.strip()
    elif args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        run_name = "model9_film_gcn_sing_es"
    out_dirs = make_output_dirs(run_name)
    _ACTIVE_RESULTS_LOG = out_dirs["text"]
    run_state_path = out_dirs["state"]
    if args.resume_run.strip() and not run_state_path.exists():
        raise FileNotFoundError(
            f"Resume state untuk run '{run_name}' tidak ditemukan di {run_state_path}"
        )
    run_state = load_run_state(run_state_path)

    log_info(f"[INFO] Run name    : {out_dirs['run_name']}")
    log_info(f"[INFO] Log root    : {out_dirs['log_root']}")
    log_info(f"[INFO] Fold root   : {out_dirs['fold_root']}")
    log_info(f"[INFO] Model root  : {out_dirs['model_root']}")

    log_info("[INFO] Memuat data preprocess...")
    metadata, peptide_graphs, protein_graphs, labels = load_preprocessed_data(args.data_dir)
    dataset_summary = summarize_dataset(labels, metadata)
    global_raw = build_global_feature_matrix(metadata)

    trainval_indices, test_indices, fold_specs = make_splits(labels, cfg)
    if int(args.max_folds) > 0:
        fold_specs = fold_specs[: int(args.max_folds)]
    candidate_ids = parse_candidate_ids(args.candidate_ids)
    candidate_cfgs = build_candidate_configs(cfg, candidate_ids=candidate_ids)
    if not candidate_cfgs:
        raise ValueError("Tidak ada kandidat yang dipilih. Cek --candidate-ids.")
    resume_config_payload = {
        "config": to_serializable(cfg),
        "candidate_configs": to_serializable(candidate_cfgs),
    }
    config_signature = compute_config_signature(resume_config_payload)
    saved_signature = str(run_state.get("resume_meta", {}).get("config_signature", "")).strip()
    if saved_signature and saved_signature != config_signature:
        raise RuntimeError(
            "Resume config mismatch untuk run "
            f"{run_name}. saved_signature={saved_signature}, current_signature={config_signature}."
        )
    run_state["resume_meta"] = {
        "config_signature": config_signature,
        "config_payload": resume_config_payload,
    }
    save_run_state(run_state_path, run_state)

    save_json(out_dirs["reports"] / "config.json", cfg)
    save_json(out_dirs["reports"] / "candidate_configs.json", {"candidates": candidate_cfgs})
    save_json(out_dirs["reports"] / "dataset_summary.json", dataset_summary)
    np.savez_compressed(
        out_dirs["fold_root"] / "split_indices.npz",
        trainval_indices=trainval_indices,
        test_indices=test_indices,
        folds=np.asarray(
            [
                {
                    "fold": int(spec["fold"][0]),
                    "train_indices": spec["train_indices"],
                    "val_indices": spec["val_indices"],
                }
                for spec in fold_specs
            ],
            dtype=object,
        ),
    )

    log_info("[INFO] Menjalankan k-fold training/validation...")
    fold_metrics, histories, best_epochs, fold_predictions, final_cfg, detail_logs, best_params_per_fold = run_kfold_training(
        peptide_graphs=peptide_graphs,
        protein_graphs=protein_graphs,
        labels=labels,
        metadata=metadata,
        global_raw=global_raw,
        fold_specs=fold_specs,
        candidate_cfgs=candidate_cfgs,
        out_dirs=out_dirs,
        run_state=run_state,
        config_signature=config_signature,
    )

    final_cfg["epochs"] = int(cfg["epochs"])
    final_cfg["threshold"] = float(cfg["threshold"])
    final_cfg["random_seed"] = int(cfg["random_seed"])
    final_cfg["test_size"] = float(cfg["test_size"])
    final_cfg["n_splits"] = int(cfg["n_splits"])
    final_cfg["num_workers"] = int(cfg["num_workers"])
    final_cfg["device"] = str(cfg["device"])
    final_cfg["log_every"] = int(cfg["log_every"])
    final_cfg["pos_weight"] = float(cfg["pos_weight"])
    final_cfg["grad_clip"] = float(cfg["grad_clip"])

    final_epoch_buffer = int(cfg.get("early_stopping_patience", 0))
    final_epochs = int(max(1, min(int(final_cfg["epochs"]), int(max(best_epochs) + final_epoch_buffer))))
    log_info("[INFO] Final epochs dipilih dari max best epoch fold + patience: {} (max_best={}, buffer={})".format(final_epochs, int(max(best_epochs)), final_epoch_buffer))
    log_info(f"[INFO] Kandidat final terpilih: {final_cfg['candidate_name']}")

    log_info("[INFO] Melatih model final dan evaluasi final test...")
    final_result, final_prediction_payload = train_final_model(
        peptide_graphs=peptide_graphs,
        protein_graphs=protein_graphs,
        labels=labels,
        metadata=metadata,
        global_raw=global_raw,
        trainval_indices=trainval_indices,
        test_indices=test_indices,
        cfg=final_cfg,
        out_dirs=out_dirs,
        final_epochs=final_epochs,
        run_state=run_state,
        config_signature=config_signature,
    )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(out_dirs["reports"] / "kfold_metrics.csv", index=False)
    fold_metrics_df.to_csv(out_dirs["fold_reports"] / "kfold_metrics.csv", index=False)
    save_json(out_dirs["logs"] / "kfold_histories.json", {"histories": histories})
    save_kfold_predictions_npz(fold_predictions, out_dirs["predictions"] / "kfold_predictions.npz")
    save_json(out_dirs["reports"] / "final_test_metrics.json", final_result["metrics"])
    save_json(out_dirs["logs"] / "final_train_history.json", final_result["history"])
    save_final_predictions_npz(final_prediction_payload, out_dirs["predictions"] / "final_test_predictions.npz")

    plot_kfold_roc(fold_predictions, out_dirs["plots"] / "roc_auc_kfold_all.png")
    plot_kfold_pr(fold_predictions, out_dirs["plots"] / "pr_curve_kfold_all.png")
    plot_avg_loss_acc_kfold(histories, out_dirs["plots"] / "avg_loss_acc_kfold.png")
    plot_final_test_roc_pr(
        final_prediction_payload["y_true"],
        final_prediction_payload["y_prob"],
        out_dirs["plots"] / "final_test_roc_pr_curve.png",
    )
    plot_final_confusion_matrix(
        final_prediction_payload["y_true"],
        final_prediction_payload["y_pred"],
        out_dirs["plots"] / "final_test_confusion_matrix.png",
    )

    summary_report = {
        "config": cfg,
        "final_candidate_config": final_cfg,
        "dataset_summary": dataset_summary,
        "kfold_mean": {
            "val_accuracy": float(fold_metrics_df["val_accuracy"].mean()),
            "val_precision": float(fold_metrics_df["val_precision"].mean()),
            "val_recall": float(fold_metrics_df["val_recall"].mean()),
            "val_f1": float(fold_metrics_df["val_f1"].mean()),
            "val_roc_auc": float(fold_metrics_df["val_roc_auc"].mean()),
            "val_pr_auc": float(fold_metrics_df["val_pr_auc"].mean()),
        },
        "final_test_metrics": final_result["metrics"],
        "final_epochs": final_epochs,
        "best_epochs_per_fold": best_epochs,
    }
    save_json(out_dirs["reports"] / "run_summary.json", summary_report)

    report_fold_metrics_df = pd.DataFrame(
        [
            {
                "ROC-AUC": row["val_roc_auc"],
                "PR-AUC": row["val_pr_auc"],
                "Accuracy": row["val_accuracy"],
                "Precision": row["val_precision"],
                "Recall": row["val_recall"],
                "F1": row["val_f1"],
                "fold": int(row["fold"]),
            }
            for row in fold_metrics
        ]
    )
    aggregated_metrics_df = pd.DataFrame(
        {
            "ROC-AUC": [
                float(report_fold_metrics_df["ROC-AUC"].mean()),
                float(report_fold_metrics_df["ROC-AUC"].std(ddof=0)),
            ],
            "PR-AUC": [
                float(report_fold_metrics_df["PR-AUC"].mean()),
                float(report_fold_metrics_df["PR-AUC"].std(ddof=0)),
            ],
            "Accuracy": [
                float(report_fold_metrics_df["Accuracy"].mean()),
                float(report_fold_metrics_df["Accuracy"].std(ddof=0)),
            ],
            "Precision": [
                float(report_fold_metrics_df["Precision"].mean()),
                float(report_fold_metrics_df["Precision"].std(ddof=0)),
            ],
            "Recall": [
                float(report_fold_metrics_df["Recall"].mean()),
                float(report_fold_metrics_df["Recall"].std(ddof=0)),
            ],
            "F1": [
                float(report_fold_metrics_df["F1"].mean()),
                float(report_fold_metrics_df["F1"].std(ddof=0)),
            ],
        },
        index=["mean", "std"],
    )
    params_per_fold_df = pd.DataFrame(best_params_per_fold)
    timing_lines = build_execution_report(start_time)
    sections = [
        ("Detail tuning & evaluasi", "\n".join(detail_logs)),
        ("Per-fold metrics", dataframe_to_string(report_fold_metrics_df, index=False)),
        ("Aggregated metrics", dataframe_to_string(aggregated_metrics_df, index=True)),
        (
            "Model params per fold",
            dataframe_to_string(params_per_fold_df, index=False) if not params_per_fold_df.empty else "Tidak ada parameter fold.",
        ),
        ("Voting Best Model Param", "\n".join(voting_lines(best_params_per_fold))),
        (
            "Evaluasi final test",
            "\n".join(
                [
                    f"Model final: {final_cfg['candidate_name']}",
                    f"Epoch final: {final_epochs}",
                    f"ROC-AUC={safe_float(final_result['metrics']['roc_auc']):.4f} | "
                    f"PR-AUC={safe_float(final_result['metrics']['pr_auc']):.4f} | "
                    f"Accuracy={safe_float(final_result['metrics']['accuracy']):.4f} | "
                    f"Precision={safe_float(final_result['metrics']['precision']):.4f} | "
                    f"Recall={safe_float(final_result['metrics']['recall']):.4f} | "
                    f"F1={safe_float(final_result['metrics']['f1']):.4f}",
                    "True Positive (TP) = "
                    f"{int(final_result['metrics']['tp'])} | True Negative (TN) = {int(final_result['metrics']['tn'])} | "
                    f"False Positive (FP) = {int(final_result['metrics']['fp'])} | False Negative (FN) = {int(final_result['metrics']['fn'])}",
                ]
            ),
        ),
    ]
    final_report = render_sections(sections, timing_lines)
    for line in final_report.rstrip().splitlines():
        log_info(line)

    print("\n[INFO] Selesai.")
    print(f"[INFO] Run name    : {out_dirs['run_name']}")
    print(f"[INFO] Log root    : {out_dirs['log_root']}")
    print(f"[INFO] Fold root   : {out_dirs['fold_root']}")
    print(f"[INFO] Model root  : {out_dirs['model_root']}")
    print(f"[INFO] Checkpoints : {out_dirs['checkpoints']}")
    print(f"[INFO] Plots       : {out_dirs['plots']}")
    print(f"[INFO] Logs        : {out_dirs['logs']}")
    print(f"[INFO] Predictions : {out_dirs['predictions']}")
    print(f"[INFO] Reports     : {out_dirs['reports']}")


if __name__ == "__main__":
    main()
