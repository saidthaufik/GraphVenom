import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


AMINO_ACIDS = list("ARNDCEQGHILKMFPSTWYV")
STANDARD_RESIDUES = set(AMINO_ACIDS)
NODE_DIM = 31
EDGE_DIM = 3
REQUIRED_COLUMNS = ["pep_id", "pep_seq", "uniprot_id", "prot_seq", "label"]

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
MODEL_DIR = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT.parent / "1-data" / "processed" / "training" / "gan_decoy.csv"
OUTPUT_DIR = MODEL_DIR / "processed_gat"
HELPER_PATH = MODEL_DIR / "pybiomed_helper.py"
LOG_EVERY = 500


def load_pybiomed_symbols(helper_path: Path) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    namespace: Dict[str, object] = {}
    code = helper_path.read_text(encoding="utf-8")
    exec(compile(code, str(helper_path), "exec"), namespace)
    return namespace["residue_properties"], namespace["calculate_edge_distance"]


RESIDUE_PROPERTIES, CALCULATE_EDGE_DISTANCE = load_pybiomed_symbols(HELPER_PATH)


def one_hot(value: str, allowable: Sequence[str]) -> List[float]:
    return [1.0 if value == a else 0.0 for a in allowable]


def one_hot_unk(value: str, allowable: Sequence[str]) -> List[float]:
    if value not in allowable:
        return [0.0] * len(allowable)
    return one_hot(value, allowable)


def clean_sequence(seq: str) -> str:
    seq = "" if seq is None else str(seq)
    seq = "".join(ch for ch in seq.upper() if ch.isalpha())
    return seq


def is_standard_residue(residue: str) -> bool:
    return residue in STANDARD_RESIDUES


def residue_feature_vector(residue: str) -> np.ndarray:
    if not is_standard_residue(residue):
        return np.zeros((NODE_DIM,), dtype=np.float32)
    one_hot_feat = np.array(one_hot(residue, AMINO_ACIDS), dtype=np.float32)
    phys_feat = np.asarray(RESIDUE_PROPERTIES(residue, AMINO_ACIDS), dtype=np.float32)
    return np.concatenate([one_hot_feat, phys_feat], axis=0)


def pair_edge_feature(res_u: str, res_v: str) -> List[float]:
    if is_standard_residue(res_u) and is_standard_residue(res_v):
        chem1, chem2 = CALCULATE_EDGE_DISTANCE(res_u, res_v)
        return [float(chem1), float(chem2), 1.0]
    return [0.0, 0.0, 1.0]


def build_sequence_edges(sequence: str) -> Tuple[List[Tuple[int, int]], List[List[float]]]:
    edges: List[Tuple[int, int]] = []
    edge_feats: List[List[float]] = []
    seq_len = len(sequence)

    for idx in range(seq_len - 1):
        feat = pair_edge_feature(sequence[idx], sequence[idx + 1])
        edges.append((idx, idx + 1))
        edge_feats.append(feat)
        edges.append((idx + 1, idx))
        edge_feats.append(feat)

    for idx in range(seq_len):
        edges.append((idx, idx))
        edge_feats.append([0.0, 0.0, 0.0])

    if not edges:
        edges.append((0, 0))
        edge_feats.append([0.0, 0.0, 0.0])

    return edges, edge_feats


def build_dummy_graph() -> Data:
    x = torch.zeros((1, NODE_DIM), dtype=torch.float32)
    edge_index = torch.zeros((2, 1), dtype=torch.long)
    edge_attr = torch.zeros((1, EDGE_DIM), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def sequence_to_graph(sequence: str) -> Tuple[Data, Dict[str, Union[int, str]]]:
    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        return build_dummy_graph(), {
            "clean_sequence": "",
            "clean_length": 0,
            "unknown_count": 0,
            "is_dummy": 1,
            "is_invalid": 1,
        }

    unknown_count = sum(1 for ch in clean_seq if not is_standard_residue(ch))
    feat_rows = [residue_feature_vector(ch) for ch in clean_seq]
    x = torch.from_numpy(np.stack(feat_rows, axis=0))
    edges, edge_feats = build_sequence_edges(clean_seq)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph, {
        "clean_sequence": clean_seq,
        "clean_length": len(clean_seq),
        "unknown_count": int(unknown_count),
        "is_dummy": 0,
        "is_invalid": 0,
    }


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


def build_metadata_row(
    row: pd.Series,
    pep_meta: Dict[str, Union[int, str]],
    prot_meta: Dict[str, Union[int, str]],
) -> Dict[str, object]:
    return {
        "pep_id": row["pep_id"],
        "uniprot_id": row["uniprot_id"],
        "raw_pep_seq": row["pep_seq"],
        "raw_prot_seq": row["prot_seq"],
        "clean_pep_seq": pep_meta["clean_sequence"],
        "clean_prot_seq": prot_meta["clean_sequence"],
        "pep_len": pep_meta["clean_length"],
        "prot_len": prot_meta["clean_length"],
        "pep_unknown_count": pep_meta["unknown_count"],
        "prot_unknown_count": prot_meta["unknown_count"],
        "pep_is_dummy": pep_meta["is_dummy"],
        "prot_is_dummy": prot_meta["is_dummy"],
        "pep_is_invalid": pep_meta["is_invalid"],
        "prot_is_invalid": prot_meta["is_invalid"],
        "label": int(row["label"]),
    }


def preprocess_and_save(csv_path: Path, out_dir: Path) -> None:
    print("[INFO] Memulai preprocessing GAT sequence-only")
    print(f"[INFO] CSV input  : {csv_path}")
    print(f"[INFO] Output dir : {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    print(f"[INFO] Jumlah sampel: {len(df)}")

    peptide_graphs: List[Data] = []
    protein_graphs: List[Data] = []
    labels: List[int] = []
    metadata_rows: List[Dict[str, object]] = []

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        pep_graph, pep_meta = sequence_to_graph(row["pep_seq"])
        prot_graph, prot_meta = sequence_to_graph(row["prot_seq"])

        peptide_graphs.append(pep_graph)
        protein_graphs.append(prot_graph)
        labels.append(int(row["label"]))
        metadata_rows.append(build_metadata_row(row, pep_meta, prot_meta))

        if idx == 1 or idx % LOG_EVERY == 0 or idx == len(df):
            print(
                "[INFO] Progress: "
                f"{idx}/{len(df)} | "
                f"pep_len={pep_meta['clean_length']} dummy={pep_meta['is_dummy']} | "
                f"prot_len={prot_meta['clean_length']} dummy={prot_meta['is_dummy']}"
            )

    metadata = pd.DataFrame(metadata_rows)
    labels_np = np.asarray(labels, dtype=np.int64)

    print("[INFO] Menyimpan output preprocessing...")
    torch.save(peptide_graphs, out_dir / "peptide_graphs.pt")
    torch.save(protein_graphs, out_dir / "protein_graphs.pt")
    np.save(out_dir / "y.npy", labels_np)
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    stats = {
        "num_samples": int(len(df)),
        "num_positive": int(labels_np.sum()),
        "num_negative": int((labels_np == 0).sum()),
        "peptide_dummy_count": int(metadata["pep_is_dummy"].sum()),
        "protein_dummy_count": int(metadata["prot_is_dummy"].sum()),
        "peptide_unknown_total": int(metadata["pep_unknown_count"].sum()),
        "protein_unknown_total": int(metadata["prot_unknown_count"].sum()),
        "node_dim": NODE_DIM,
        "edge_dim": EDGE_DIM,
    }
    (out_dir / "preprocess_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print("[INFO] Preprocessing selesai")
    print(f"[INFO] peptide_graphs.pt   -> {out_dir / 'peptide_graphs.pt'}")
    print(f"[INFO] protein_graphs.pt   -> {out_dir / 'protein_graphs.pt'}")
    print(f"[INFO] y.npy               -> {out_dir / 'y.npy'}")
    print(f"[INFO] metadata.csv        -> {out_dir / 'metadata.csv'}")
    print(f"[INFO] preprocess_stats.json -> {out_dir / 'preprocess_stats.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequence-only GAT preprocessing for peptide-protein interaction data.")
    parser.add_argument("--csv", type=Path, default=INPUT_CSV, help="Path ke file CSV input.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Folder output preprocessing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_and_save(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
