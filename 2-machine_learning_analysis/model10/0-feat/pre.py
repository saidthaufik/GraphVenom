import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from Bio.PDB.Polypeptide import PPBuilder
from torch_geometric.data import Data

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
    from Bio.PDB.SASA import ShrakeRupley
except Exception:  # pragma: no cover
    PDBParser = None
    DSSP = None
    ShrakeRupley = None

from pybiomed_helper import calculate_edge_distance, residue_properties


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path import model_root, shared_input_csv


INPUT_CSV = shared_input_csv()
OUTPUT_DIR = model_root("model10") / "0-feat" / "processed_gcn_af"
DEFAULT_AF_PDB_DIR = PROJECT_ROOT.parent / "1-data" / "structures" / "proteins"

AMINO_ACIDS = list("ARNDCEQGHILKMFPSTWYV")
STANDARD_RESIDUES = set(AMINO_ACIDS)

PEPTIDE_NODE_DIM = 31
PROTEIN_NODE_DIM = 38
PEPTIDE_EDGE_DIM = 3
PROTEIN_EDGE_DIM = 4
CONTACT_THRESHOLD = 8.0
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


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _build_empty_graph(node_dim: int, edge_dim: int, include_struct: bool = False) -> Data:
    data = Data(
        x=torch.zeros((1, node_dim), dtype=torch.float32),
        edge_index=torch.zeros((2, 1), dtype=torch.long),
        edge_attr=torch.zeros((1, edge_dim), dtype=torch.float32),
    )
    if include_struct:
        data.struct_feats = torch.zeros((1, 3), dtype=torch.float32)
    return data


def _sequence_backbone_edges(sequence: str, edge_dim: int) -> Tuple[List[Tuple[int, int]], List[List[float]]]:
    edges: List[Tuple[int, int]] = []
    edge_feats: List[List[float]] = []
    seq_len = len(sequence)

    for idx in range(seq_len - 1):
        res_u = sequence[idx] if is_standard_residue(sequence[idx]) else "A"
        res_v = sequence[idx + 1] if is_standard_residue(sequence[idx + 1]) else "A"
        qso1, qso2 = calculate_edge_distance(res_u, res_v)
        feat = [float(qso1), float(qso2)]
        if edge_dim == PEPTIDE_EDGE_DIM:
            feat.append(1.0)
        else:
            feat.extend([0.0, 1.0])
        edges.append((idx, idx + 1))
        edge_feats.append(feat)
        edges.append((idx + 1, idx))
        edge_feats.append(feat)

    return edges, edge_feats


def build_peptide_graph(sequence: str) -> Tuple[Data, Dict[str, Any]]:
    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        graph = _build_empty_graph(PEPTIDE_NODE_DIM, PEPTIDE_EDGE_DIM)
        return graph, {
            "clean_length": 0,
            "unknown_count": 0,
            "is_dummy": 1,
            "is_invalid": 1,
        }

    unknown_count = sum(1 for ch in clean_seq if not is_standard_residue(ch))
    x = torch.from_numpy(np.stack([residue_feature_vector(ch) for ch in clean_seq], axis=0))
    edges, edge_feats = _sequence_backbone_edges(clean_seq, edge_dim=PEPTIDE_EDGE_DIM)

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


def extract_pdb_features(pdb_path: Path) -> Optional[Dict[str, torch.Tensor]]:
    if PDBParser is None or ShrakeRupley is None:
        return None
    try:
        if not pdb_path.exists() or pdb_path.stat().st_size <= 0:
            return None

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("af", str(pdb_path))
        model = next(structure.get_models())
        chain = next(model.get_chains())

        sr = ShrakeRupley()
        sr.compute(model, level="R")

        ppb = PPBuilder()
        phi_psi_dict: Dict[Any, List[float]] = {}
        for poly in ppb.build_peptides(chain):
            for residue, (phi, psi) in zip(poly, poly.get_phi_psi_list()):
                phi_val = 0.0 if phi is None else float(phi)
                psi_val = 0.0 if psi is None else float(psi)
                phi_psi_dict[residue.get_id()] = [
                    math.sin(phi_val),
                    math.cos(phi_val),
                    math.sin(psi_val),
                    math.cos(psi_val),
                ]

        coords = []
        plddt = []
        sasa = []
        dihedrals = []
        residue_ids = []
        for residue in chain:
            if getattr(residue, "id", ("",))[0] != " ":
                continue
            if "CA" not in residue:
                continue
            ca = residue["CA"]
            coord = ca.get_coord()
            coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
            plddt.append([_clip01(float(ca.get_bfactor()) / 100.0)])
            sasa.append([_clip01(float(getattr(residue, "sasa", 0.0) or 0.0) / 200.0)])
            dihedrals.append(phi_psi_dict.get(residue.get_id(), [0.0, 1.0, 0.0, 1.0]))
            residue_ids.append(residue.get_id())

        if not coords:
            return None

        return {
            "coords": torch.tensor(np.asarray(coords, dtype=np.float32)),
            "plddt": torch.tensor(np.asarray(plddt, dtype=np.float32)),
            "sasa": torch.tensor(np.asarray(sasa, dtype=np.float32)),
            "dihedrals": torch.tensor(np.asarray(dihedrals, dtype=np.float32)),
            "residue_ids": residue_ids,
        }
    except Exception:
        return None


def _parse_pdb_structure(pdb_path: str):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("prot", pdb_path)


def _safe_first_model_chain(structure):
    model = next(structure.get_models())
    chains = list(model.get_chains())
    if not chains:
        return model, None
    return model, chains[0].id


def _compute_dssp_rsa_map(pdb_path: str, dssp_bin: str = "mkdssp") -> Dict[Tuple[str, Tuple[str, int, str]], float]:
    if DSSP is None:
        return {}
    structure = _parse_pdb_structure(pdb_path)
    model = next(structure.get_models())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dssp = DSSP(model, pdb_path, dssp=dssp_bin)

    rsa_map: Dict[Tuple[str, Tuple[str, int, str]], float] = {}
    for key in dssp.keys():
        try:
            rsa_map[key] = float(dssp[key][3])
        except Exception:
            continue
    return rsa_map


def _fill_rsa_by_order(
    rsa_tensor: torch.Tensor,
    rsa_map: Dict[Tuple[str, Tuple[str, int, str]], float],
    chain_id: str,
) -> torch.Tensor:
    keys = [key for key in rsa_map.keys() if key[0] == chain_id]
    keys_sorted = sorted(keys, key=lambda item: (item[1][1], item[1][2]))
    for idx in range(min(rsa_tensor.size(0), len(keys_sorted))):
        rsa_tensor[idx, 0] = float(rsa_map[keys_sorted[idx]])
    return rsa_tensor


def _build_contact_pairs(coords_np: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    n = int(coords_np.shape[0])
    if n <= 1:
        return []

    thr2 = float(threshold) ** 2
    pairs: List[Tuple[int, int, float]] = []
    chunk = 512
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        block = coords_np[i0:i1]
        diff = block[:, None, :] - coords_np[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        ii, jj = np.where(d2 < thr2)
        for di, j in zip(ii.tolist(), jj.tolist()):
            i = i0 + di
            if j <= i or j - i <= 1:
                continue
            pairs.append((i, j, float(np.sqrt(float(d2[di, j])))))
    return pairs


def resolve_pdb_path(uniprot_id: str, pdb_dir: Optional[Path]) -> Optional[Path]:
    if pdb_dir is None:
        return None

    clean_id = str(uniprot_id).strip()
    if not clean_id:
        return None

    direct_dir = Path(pdb_dir)
    candidate_dirs = [direct_dir]
    nested_dir = direct_dir / "alphafold_pdb"
    if nested_dir != direct_dir:
        candidate_dirs.append(nested_dir)

    stem_candidates = [
        clean_id,
        clean_id.upper(),
        clean_id.lower(),
        f"AF-{clean_id}-F1-model_v4",
        f"AF-{clean_id}-F1-model_v3",
        f"AF-{clean_id}-F1-model_v2",
        f"AF-{clean_id}-F1-model_v1",
    ]

    for directory in candidate_dirs:
        for stem in stem_candidates:
            candidate = directory / f"{stem}.pdb"
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    return None


def resolve_default_pdb_dir(csv_path: Path) -> Optional[Path]:
    candidates = [
        DEFAULT_AF_PDB_DIR,
        csv_path.parent / "alphafold_pdb",
        csv_path.parent / "afprot" / "alphafold_pdb",
        csv_path.parent / "afprot",
        PROJECT_ROOT / "0-data" / "afprot",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_protein_graph(sequence: str, uniprot_id: str, pdb_dir: Optional[Path]) -> Tuple[Data, Dict[str, Any]]:
    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        graph = _build_empty_graph(PROTEIN_NODE_DIM, PROTEIN_EDGE_DIM, include_struct=True)
        return graph, {
            "clean_length": 0,
            "unknown_count": 0,
            "is_dummy": 1,
            "is_invalid": 1,
            "used_structure": 0,
            "num_contacts": 0,
        }

    pdb_path = resolve_pdb_path(uniprot_id, pdb_dir)
    pdb_feat = None if pdb_path is None else extract_pdb_features(pdb_path)
    used_structure = int(pdb_feat is not None)

    if pdb_feat is not None:
        coords = pdb_feat["coords"]
        plddt = pdb_feat["plddt"]
        sasa = pdb_feat["sasa"]
        dihedrals = pdb_feat["dihedrals"]
        min_len = min(len(clean_seq), int(coords.shape[0]))
        clean_seq = clean_seq[:min_len]
        coords = coords[:min_len]
        plddt = plddt[:min_len]
        sasa = sasa[:min_len]
        dihedrals = dihedrals[:min_len]
    else:
        min_len = len(clean_seq)
        coords = None
        plddt = torch.zeros((min_len, 1), dtype=torch.float32)
        sasa = torch.zeros((min_len, 1), dtype=torch.float32)
        dihedrals = torch.zeros((min_len, 4), dtype=torch.float32)

    unknown_count = sum(1 for ch in clean_seq if not is_standard_residue(ch))
    base_x = torch.from_numpy(np.stack([residue_feature_vector(ch) for ch in clean_seq], axis=0))

    rsa = torch.ones((len(clean_seq), 1), dtype=torch.float32)
    if pdb_path is not None and pdb_path.exists() and pdb_path.stat().st_size > 0 and DSSP is not None:
        try:
            rsa_map = _compute_dssp_rsa_map(str(pdb_path), dssp_bin="mkdssp")
            _, chain_id = _safe_first_model_chain(_parse_pdb_structure(str(pdb_path)))
            if chain_id is not None:
                rsa = _fill_rsa_by_order(rsa, rsa_map, chain_id)
        except Exception:
            pass

    x = torch.cat([base_x, plddt, sasa, dihedrals, rsa], dim=1)

    edges, edge_feats = _sequence_backbone_edges(clean_seq, edge_dim=PROTEIN_EDGE_DIM)
    num_contacts = 0
    if coords is not None and len(clean_seq) > 1:
        contacts = _build_contact_pairs(coords.detach().cpu().numpy().astype(np.float32, copy=False), CONTACT_THRESHOLD)
        num_contacts = len(contacts)
        for i, j, dist in contacts:
            res_u = clean_seq[i] if is_standard_residue(clean_seq[i]) else "A"
            res_v = clean_seq[j] if is_standard_residue(clean_seq[j]) else "A"
            qso1, qso2 = calculate_edge_distance(res_u, res_v)
            norm_dist = _clip01(float(dist) / 10.0)
            feat = [float(qso1), float(qso2), float(norm_dist), 0.0]
            edges.append((i, j))
            edge_feats.append(feat)
            edges.append((j, i))
            edge_feats.append(feat)

    for idx in range(len(clean_seq)):
        edges.append((idx, idx))
        edge_feats.append([0.0, 0.0, 0.0, 1.0])

    graph = Data(
        x=x,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_feats, dtype=torch.float32),
    )
    graph.struct_feats = torch.cat([plddt, sasa, rsa], dim=1)
    return graph, {
        "clean_length": len(clean_seq),
        "unknown_count": int(unknown_count),
        "is_dummy": 0,
        "is_invalid": 0,
        "used_structure": used_structure,
        "num_contacts": int(num_contacts),
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


def preprocess_and_save(csv_path: Path, out_dir: Path, pdb_dir: Optional[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)

    print("[INFO] Memulai preprocessing GCN peptide-protein + AlphaFold")
    print(f"[INFO] CSV input : {csv_path}")
    print(f"[INFO] PDB dir   : {pdb_dir if pdb_dir is not None else 'None'}")
    print(f"[INFO] Out dir   : {out_dir}")
    print(f"[INFO] Jumlah sampel: {len(df)}")

    peptide_graphs: List[Data] = []
    protein_graphs: List[Data] = []
    labels: List[int] = []
    metadata_rows: List[Dict[str, Any]] = []

    structure_found = 0
    total_contacts = 0

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        pep_graph, pep_meta = build_peptide_graph(row.pep_seq)
        prot_graph, prot_meta = build_protein_graph(row.prot_seq, row.uniprot_id, pdb_dir)

        peptide_graphs.append(pep_graph)
        protein_graphs.append(prot_graph)
        labels.append(int(row.label))
        structure_found += int(prot_meta["used_structure"])
        total_contacts += int(prot_meta["num_contacts"])

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
                "prot_used_structure": prot_meta["used_structure"],
                "prot_num_contacts": prot_meta["num_contacts"],
                "label": int(row.label),
            }
        )

        if idx == 1 or idx % LOG_EVERY == 0 or idx == len(df):
            print(
                "[INFO] Progress: "
                f"{idx}/{len(df)} | "
                f"pep_len={pep_meta['clean_length']} | "
                f"prot_len={prot_meta['clean_length']} | "
                f"prot_struct={prot_meta['used_structure']} | "
                f"contacts={prot_meta['num_contacts']}"
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
        "protein_structure_found": int(structure_found),
        "protein_structure_missing": int(len(df) - structure_found),
        "avg_protein_contacts": float(total_contacts / max(len(df), 1)),
    }
    (out_dir / "preprocess_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("[INFO] Preprocessing selesai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing peptide-protein GCN + AlphaFold untuk model10.")
    parser.add_argument("--csv", type=Path, default=INPUT_CSV, help="Path ke CSV input.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Folder output preprocessing.")
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Folder PDB AlphaFold. Jika kosong, otomatis cari di afprot/alphafold_pdb proyek lalu fallback ke lokasi dekat CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdb_dir = args.pdb_dir
    if pdb_dir is None:
        pdb_dir = resolve_default_pdb_dir(args.csv)
    preprocess_and_save(args.csv, args.out_dir, pdb_dir)


if __name__ == "__main__":
    main()
