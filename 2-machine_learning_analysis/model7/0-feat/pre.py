import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from pybiomed_helper import calculate_edge_distance, residue_properties


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path import model_root, shared_input_csv


INPUT_CSV = shared_input_csv()
OUTPUT_DIR = model_root("model7") / "0-feat" / "processed_gine"

AMINO_ACIDS = list("ARNDCEQGHILKMFPSTWYV")
STANDARD_RESIDUES = set(AMINO_ACIDS)

PEPTIDE_NODE_DIM = 31
PROTEIN_NODE_DIM = 31
PEPTIDE_EDGE_DIM = 3
PROTEIN_EDGE_DIM = 3
LOG_EVERY = 250

REQUIRED_COLUMNS = ["pep_id", "pep_seq", "uniprot_id", "prot_seq", "label"]


def one_hot(value: str, allowable: Sequence[str]) -> List[float]:
    return [1.0 if value == item else 0.0 for item in allowable]


def clean_sequence(seq: str) -> str:
    seq = "" if seq is None else str(seq)
    return "".join(ch for ch in seq.upper() if ch.isalpha())


def is_standard_residue(residue: str) -> bool:
    return residue in STANDARD_RESIDUES


def residue_feature_vector(residue: str) -> np.ndarray:
    if not is_standard_residue(residue):
        return np.zeros((PEPTIDE_NODE_DIM,), dtype=np.float32)
    one_hot_feat = np.asarray(one_hot(residue, AMINO_ACIDS), dtype=np.float32)
    phys_feat = np.asarray(residue_properties(residue, AMINO_ACIDS), dtype=np.float32)
    return np.concatenate([one_hot_feat, phys_feat], axis=0)


def build_empty_graph(node_dim: int, edge_dim: int) -> Data:
    return Data(
        x=torch.zeros((1, node_dim), dtype=torch.float32),
        edge_index=torch.zeros((2, 1), dtype=torch.long),
        edge_attr=torch.zeros((1, edge_dim), dtype=torch.float32),
    )


def build_sequence_graph(sequence: str, node_dim: int, edge_dim: int) -> Tuple[Data, Dict[str, Any]]:
    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        return build_empty_graph(node_dim, edge_dim), {
            "clean_length": 0,
            "unknown_count": 0,
            "is_dummy": 1,
            "is_invalid": 1,
        }

    unknown_count = sum(1 for ch in clean_seq if not is_standard_residue(ch))
    x = torch.from_numpy(np.stack([residue_feature_vector(ch) for ch in clean_seq], axis=0))

    edges: List[Tuple[int, int]] = []
    edge_feats: List[List[float]] = []
    for idx in range(len(clean_seq) - 1):
        res_u = clean_seq[idx] if is_standard_residue(clean_seq[idx]) else "A"
        res_v = clean_seq[idx + 1] if is_standard_residue(clean_seq[idx + 1]) else "A"
        qso1, qso2 = calculate_edge_distance(res_u, res_v)
        feat = [float(qso1), float(qso2), 1.0]
        edges.append((idx, idx + 1))
        edge_feats.append(feat)
        edges.append((idx + 1, idx))
        edge_feats.append(feat)

    for idx in range(len(clean_seq)):
        edges.append((idx, idx))
        edge_feats.append([0.0, 0.0, 0.0])

    graph = Data(
        x=x,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_feats, dtype=torch.float32),
    )
    return graph, {
        "clean_length": len(clean_seq),
        "unknown_count": int(unknown_count),
        "is_dummy": 0,
        "is_invalid": 0,
    }


def build_peptide_graph(sequence: str) -> Tuple[Data, Dict[str, Any]]:
    return build_sequence_graph(sequence, PEPTIDE_NODE_DIM, PEPTIDE_EDGE_DIM)


def build_protein_graph(sequence: str) -> Tuple[Data, Dict[str, Any]]:
    return build_sequence_graph(sequence, PROTEIN_NODE_DIM, PROTEIN_EDGE_DIM)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df["pep_seq"] = df["pep_seq"].fillna("")
    df["prot_seq"] = df["prot_seq"].fillna("")
    df["pep_id"] = df["pep_id"].fillna("")
    df["uniprot_id"] = df["uniprot_id"].fillna("")
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def preprocess_and_save(csv_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)

    print("[INFO] Memulai preprocessing GINE peptide-protein sequence-only")
    print(f"[INFO] CSV input : {csv_path}")
    print(f"[INFO] Out dir   : {out_dir}")
    print(f"[INFO] Jumlah sampel: {len(df)}")

    peptide_graphs: List[Data] = []
    protein_graphs: List[Data] = []
    labels: List[int] = []
    metadata_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        pep_graph, pep_meta = build_peptide_graph(row.pep_seq)
        prot_graph, prot_meta = build_protein_graph(row.prot_seq)

        peptide_graphs.append(pep_graph)
        protein_graphs.append(prot_graph)
        labels.append(int(row.label))
        metadata_rows.append(
            {
                "pep_id": row.pep_id,
                "uniprot_id": row.uniprot_id,
                "raw_pep_seq": row.pep_seq,
                "raw_prot_seq": row.prot_seq,
                "pep_len": pep_meta["clean_length"],
                "prot_len": prot_meta["clean_length"],
                "pep_unknown_count": pep_meta["unknown_count"],
                "prot_unknown_count": prot_meta["unknown_count"],
                "pep_is_dummy": pep_meta["is_dummy"],
                "prot_is_dummy": prot_meta["is_dummy"],
                "label": int(row.label),
            }
        )

        if idx == 1 or idx % LOG_EVERY == 0 or idx == len(df):
            print(
                "[INFO] Progress: "
                f"{idx}/{len(df)} | "
                f"pep_len={pep_meta['clean_length']} | "
                f"prot_len={prot_meta['clean_length']}"
            )

    metadata = pd.DataFrame(metadata_rows)
    labels_np = np.asarray(labels, dtype=np.int64)

    torch.save(peptide_graphs, out_dir / "peptide_graphs.pt")
    torch.save(protein_graphs, out_dir / "protein_graphs.pt")
    np.save(out_dir / "y.npy", labels_np)
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    stats = {
        "num_samples": int(len(df)),
        "num_positive": int(np.sum(labels_np == 1)),
        "num_negative": int(np.sum(labels_np == 0)),
        "peptide_node_dim": PEPTIDE_NODE_DIM,
        "protein_node_dim": PROTEIN_NODE_DIM,
        "peptide_edge_dim": PEPTIDE_EDGE_DIM,
        "protein_edge_dim": PROTEIN_EDGE_DIM,
    }
    (out_dir / "preprocess_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("[INFO] Preprocessing selesai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing peptide-protein GINE sequence-only untuk model7.")
    parser.add_argument("--csv", type=Path, default=INPUT_CSV, help="Path ke CSV input.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Folder output preprocessing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_and_save(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
