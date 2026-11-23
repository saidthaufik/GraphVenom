#!/usr/bin/env python3
"""Audit Model 3 ProtT5 data for sequence-level unseen splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL3_FEAT = ROOT / "model3" / "1-feat_prott5"
POSITIVE_CSV = MODEL3_FEAT / "prott5.csv"
ACTIVE_CSV = MODEL3_FEAT / "prott5pn.csv"
RANDOM_CSV = MODEL3_FEAT / "prott5_random_1to1.csv"
OUT_ROOT = ROOT.parent / "1-data" / "splits" / "unseen" / "model3" / "audit"
REPORT_JSON = OUT_ROOT / "model3_unseen_sequence_audit.json"
RECOVERED_CSV = OUT_ROOT / "prott5pn_with_recovered_pep_seq_audit.csv"
CONFLICT_ROWS_CSV = OUT_ROOT / "conflict_rows_candidate.csv"
CONFLICT_PAIRS_CSV = OUT_ROOT / "conflict_pairs_candidate.csv"

ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: object) -> str:
    return "".join(aa for aa in str(seq or "").upper() if aa in ALLOWED_AA)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    def suffix(col: str) -> int:
        return int(col.rsplit("_", 1)[1])

    return sorted([c for c in df.columns if c.startswith(prefix)], key=suffix)


def vector_fingerprints(df: pd.DataFrame, cols: list[str], decimals: int = 8) -> list[str]:
    arr = df[cols].to_numpy(dtype=np.float64, copy=True)
    arr = np.round(arr, decimals=decimals)
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in arr]


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().to_dict().items()}


def summarize_frame(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "has_pep_seq": bool("pep_seq" in df.columns),
        "has_prot_seq": bool("prot_seq" in df.columns),
        "has_label": bool("label" in df.columns),
    }
    if "label" in df.columns:
        summary["label_counts"] = value_counts_dict(df["label"])
    for cols, name in [
        (["pdb_id"], "unique_pdb_id"),
        (["Uniprot_id"], "unique_uniprot_id"),
        (["pdb_id", "Uniprot_id"], "unique_id_pairs"),
        (["pep_seq"], "unique_pep_seq"),
        (["prot_seq"], "unique_prot_seq"),
        (["pep_seq", "prot_seq"], "unique_sequence_pairs"),
    ]:
        if all(c in df.columns for c in cols):
            summary[name] = int(df[cols].astype(str).drop_duplicates().shape[0])
    if all(c in df.columns for c in ["pdb_id", "Uniprot_id"]):
        summary["duplicate_id_pairs"] = int(df.duplicated(["pdb_id", "Uniprot_id"]).sum())
    return summary


def recover_pep_sequence_by_embedding(pos_df: pd.DataFrame, active_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pep_cols = embedding_columns(pos_df, "pep_embed_")
    pos = pos_df.copy()
    active = active_df.copy()
    pos["pep_embedding_fp"] = vector_fingerprints(pos, pep_cols)
    active["pep_embedding_fp"] = vector_fingerprints(active, pep_cols)

    fp_to_sequences = (
        pos.groupby("pep_embedding_fp")["pep_seq"]
        .agg(lambda values: sorted(set(map(str, values))))
        .reset_index(name="candidate_pep_seq")
    )
    fp_to_sequences["candidate_count"] = fp_to_sequences["candidate_pep_seq"].map(len)

    active = active.merge(fp_to_sequences, on="pep_embedding_fp", how="left")
    active["recovered_pep_seq"] = active["candidate_pep_seq"].map(
        lambda values: values[0] if isinstance(values, list) and len(values) == 1 else None
    )

    unresolved = active["recovered_pep_seq"].isna()
    ambiguous = active["candidate_count"].fillna(0).gt(1)
    missing_fp = active["candidate_count"].isna()
    summary = {
        "pep_embedding_columns": int(len(pep_cols)),
        "source_unique_peptide_embedding_fp": int(pos["pep_embedding_fp"].nunique()),
        "source_embedding_fp_with_multiple_pep_seq": int(fp_to_sequences["candidate_count"].gt(1).sum()),
        "active_rows": int(len(active)),
        "active_rows_recovered_unique": int((~unresolved).sum()),
        "active_rows_ambiguous": int(ambiguous.sum()),
        "active_rows_missing_embedding_match": int(missing_fp.sum()),
        "active_rows_unresolved": int(unresolved.sum()),
    }
    return active, summary


def conflict_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    work["clean_pep_seq"] = work["recovered_pep_seq"].map(clean_sequence)
    work["clean_prot_seq"] = work["prot_seq"].map(clean_sequence)
    work["label"] = pd.to_numeric(work["label"], errors="raise").astype(int)

    empty_mask = work["clean_pep_seq"].eq("") | work["clean_prot_seq"].eq("")
    pair_cols = ["clean_pep_seq", "clean_prot_seq"]
    label_counts = work.groupby(pair_cols)["label"].nunique().reset_index(name="num_labels")
    conflict_pairs = label_counts[label_counts["num_labels"].gt(1)][pair_cols].copy()
    conflict_key = set(map(tuple, conflict_pairs[pair_cols].to_numpy()))
    row_keys = work[pair_cols].apply(tuple, axis=1)
    conflict_rows = work.loc[row_keys.isin(conflict_key)].copy()
    clean_df = work.loc[~row_keys.isin(conflict_key)].copy()

    summary = {
        "raw_rows": int(len(work)),
        "raw_pos": int(work["label"].sum()),
        "raw_neg": int(len(work) - work["label"].sum()),
        "empty_clean_sequence_rows": int(empty_mask.sum()),
        "conflict_sequence_pairs": int(len(conflict_pairs)),
        "conflict_rows": int(len(conflict_rows)),
        "conflict_pos": int(conflict_rows["label"].sum()) if len(conflict_rows) else 0,
        "conflict_neg": int(len(conflict_rows) - conflict_rows["label"].sum()) if len(conflict_rows) else 0,
        "clean_rows_if_conflicts_dropped": int(len(clean_df)),
        "clean_pos_if_conflicts_dropped": int(clean_df["label"].sum()),
        "clean_neg_if_conflicts_dropped": int(len(clean_df) - clean_df["label"].sum()),
        "clean_unique_peptide_sequence": int(clean_df["clean_pep_seq"].nunique()),
        "clean_unique_protein_sequence": int(clean_df["clean_prot_seq"].nunique()),
        "clean_unique_sequence_pair": int(clean_df[pair_cols].drop_duplicates().shape[0]),
        "remaining_sequence_pair_label_conflicts": int(
            (clean_df.groupby(pair_cols)["label"].nunique().gt(1)).sum()
        ),
    }
    return conflict_pairs, conflict_rows, summary


def compare_active_and_random(active_df: pd.DataFrame, random_df: pd.DataFrame) -> dict[str, Any]:
    id_cols = ["pdb_id", "Uniprot_id", "label"]
    active_keys = set(map(tuple, active_df[id_cols].astype(str).to_numpy()))
    random_keys = set(map(tuple, random_df[id_cols].astype(str).to_numpy()))
    return {
        "same_shape": bool(active_df.shape == random_df.shape),
        "same_columns": bool(list(active_df.columns) == list(random_df.columns)),
        "same_id_label_key_set": bool(active_keys == random_keys),
        "active_only_id_label_keys": int(len(active_keys - random_keys)),
        "random_only_id_label_keys": int(len(random_keys - active_keys)),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pos_df = pd.read_csv(POSITIVE_CSV, sep=";")
    active_df = pd.read_csv(ACTIVE_CSV)
    random_df = pd.read_csv(RANDOM_CSV)

    recovered_df, recovery_summary = recover_pep_sequence_by_embedding(pos_df, active_df)
    conflict_pairs, conflict_rows, conflict_summary = conflict_audit(recovered_df)

    recovered_df.to_csv(RECOVERED_CSV, index=False)
    conflict_pairs.to_csv(CONFLICT_PAIRS_CSV, index=False)
    conflict_rows.to_csv(CONFLICT_ROWS_CSV, index=False)

    pos_cast = pos_df.copy()
    pos_cast["pdb_id"] = pos_cast["pdb_id"].astype(str)
    pos_cast["Uniprot_id"] = pos_cast["Uniprot_id"].astype(str)

    active_cast = active_df.copy()
    active_cast["pdb_id"] = active_cast["pdb_id"].astype(str)
    active_cast["Uniprot_id"] = active_cast["Uniprot_id"].astype(str)

    pos_pairs = set(map(tuple, pos_cast[["pdb_id", "Uniprot_id"]].to_numpy()))
    active_positive_pairs = list(
        map(tuple, active_cast.loc[active_cast["label"].eq(1), ["pdb_id", "Uniprot_id"]].to_numpy())
    )
    active_negative_pairs = list(
        map(tuple, active_cast.loc[active_cast["label"].eq(0), ["pdb_id", "Uniprot_id"]].to_numpy())
    )

    report = {
        "paths": {
            "positive_csv": str(POSITIVE_CSV),
            "active_csv": str(ACTIVE_CSV),
            "random_csv": str(RANDOM_CSV),
            "report_json": str(REPORT_JSON),
            "recovered_csv": str(RECOVERED_CSV),
            "conflict_pairs_csv": str(CONFLICT_PAIRS_CSV),
            "conflict_rows_csv": str(CONFLICT_ROWS_CSV),
        },
        "file_md5": {
            "positive_csv": file_md5(POSITIVE_CSV),
            "active_csv": file_md5(ACTIVE_CSV),
            "random_csv": file_md5(RANDOM_CSV),
        },
        "positive_source_summary": summarize_frame(pos_df),
        "active_dataset_summary": summarize_frame(active_df),
        "random_dataset_summary": summarize_frame(random_df),
        "active_vs_random": compare_active_and_random(active_df, random_df),
        "positive_source_identity_issues": {
            "pdb_id_with_multiple_pep_seq": int(
                pos_cast.groupby("pdb_id")["pep_seq"].nunique().gt(1).sum()
            ),
            "pdb_uniprot_with_multiple_pep_seq": int(
                pos_cast.groupby(["pdb_id", "Uniprot_id"])["pep_seq"].nunique().gt(1).sum()
            ),
            "duplicate_pdb_uniprot_rows": int(pos_cast.duplicated(["pdb_id", "Uniprot_id"]).sum()),
            "duplicate_pep_prot_sequence_rows": int(pos_df.duplicated(["pep_seq", "prot_seq"]).sum()),
        },
        "active_pair_consistency": {
            "positive_rows_not_in_source_id_pairs": int(sum(pair not in pos_pairs for pair in active_positive_pairs)),
            "negative_rows_colliding_source_positive_id_pairs": int(sum(pair in pos_pairs for pair in active_negative_pairs)),
        },
        "pep_seq_recovery_by_embedding": recovery_summary,
        "sequence_pair_conflict_audit_after_recovery": conflict_summary,
        "audit_decision": {
            "pep_seq_recovery_ready": bool(recovery_summary["active_rows_unresolved"] == 0),
            "sequence_cleaning_ready": bool(conflict_summary["empty_clean_sequence_rows"] == 0),
            "can_generate_model3_unseen_splits_after_dropping_conflicts": bool(
                recovery_summary["active_rows_unresolved"] == 0
                and conflict_summary["empty_clean_sequence_rows"] == 0
                and conflict_summary["remaining_sequence_pair_label_conflicts"] == 0
            ),
            "recommended_next_step": (
                "Generate model3-specific clean CSV and unseen_sequence folds from recovered_pep_seq, "
                "dropping contradictory clean peptide+protein sequence pairs."
            ),
        },
    }

    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["audit_decision"], indent=2))
    print(json.dumps(report["pep_seq_recovery_by_embedding"], indent=2))
    print(json.dumps(report["sequence_pair_conflict_audit_after_recovery"], indent=2))
    print(f"Saved: {REPORT_JSON}")


if __name__ == "__main__":
    main()
