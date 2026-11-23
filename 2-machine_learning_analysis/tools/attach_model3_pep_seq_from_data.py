#!/usr/bin/env python3
"""Attach peptide sequence metadata to Model 3 active ProtT5 baseline data.

This script does not train any model and does not rebuild GAN negatives. It
only enriches ``model3/1-feat_prott5/prott5pn.csv`` with ``pep_seq`` from
``0-data/data.csv`` while preserving the active baseline rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = ROOT.parent / "1-data" / "processed" / "training" / "data.csv"
DEFAULT_TARGET_CSV = ROOT.parent / "1-data" / "features" / "model3" / "1-feat_prott5" / "prott5pn.csv"
DEFAULT_OUTPUT_CSV = ROOT / "model3" / "1-feat_prott5" / "prott5pn_sequence_aware.csv"
DEFAULT_AUDIT_JSON = ROOT.parent / "1-data" / "splits" / "unseen" / "model3" / "audit" / "attach_pep_seq_from_data.json"
DEFAULT_UNRESOLVED_CSV = ROOT.parent / "1-data" / "splits" / "unseen" / "model3" / "audit" / "attach_pep_seq_unresolved.csv"

ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: object) -> str:
    return "".join(aa for aa in str(seq or "").upper() if aa in ALLOWED_AA)


def read_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source CSV tidak ditemukan: {path}")
    df = pd.read_csv(path)
    required = ["pep_id", "Uniprot_id", "pep_seq", "prot_seq"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang dari source data.csv: {missing}")
    df = df[required].copy()
    for col in required:
        df[col] = df[col].astype(str)
    df["clean_pep_seq"] = df["pep_seq"].map(clean_sequence)
    df["clean_prot_seq"] = df["prot_seq"].map(clean_sequence)
    return df


def read_target(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Target CSV tidak ditemukan: {path}")
    df = pd.read_csv(path)
    required = ["pdb_id", "Uniprot_id", "prot_seq", "label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang dari target prott5pn.csv: {missing}")
    df = df.copy()
    df["pdb_id"] = df["pdb_id"].astype(str)
    df["Uniprot_id"] = df["Uniprot_id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    return df


def unique_lookup(df: pd.DataFrame, keys: list[str], value: str) -> tuple[dict[tuple[str, ...], str], set[tuple[str, ...]]]:
    grouped = df.groupby(keys)[value].agg(lambda values: sorted(set(map(str, values))))
    unique: dict[tuple[str, ...], str] = {}
    ambiguous: set[tuple[str, ...]] = set()
    for key, values in grouped.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        if len(values) == 1:
            unique[tuple(map(str, key_tuple))] = values[0]
        else:
            ambiguous.add(tuple(map(str, key_tuple)))
    return unique, ambiguous


def first_lookup(df: pd.DataFrame, keys: list[str], value: str) -> dict[tuple[str, ...], str]:
    deduped = df.drop_duplicates(subset=keys, keep="first")
    lookup: dict[tuple[str, ...], str] = {}
    for row in deduped[keys + [value]].itertuples(index=False, name=None):
        key = tuple(map(str, row[:-1]))
        lookup[key] = str(row[-1])
    return lookup


def attach_pep_seq(source: pd.DataFrame, target: pd.DataFrame, ambiguous_policy: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    pair_unique, pair_ambiguous = unique_lookup(source, ["pep_id", "Uniprot_id"], "pep_seq")
    pep_unique, pep_ambiguous = unique_lookup(source, ["pep_id"], "pep_seq")
    pair_first = first_lookup(source, ["pep_id", "Uniprot_id"], "pep_seq")
    pep_first = first_lookup(source, ["pep_id"], "pep_seq")

    pep_seq_values: list[str | None] = []
    source_values: list[str] = []
    status_values: list[str] = []

    for row in target[["pdb_id", "Uniprot_id"]].itertuples(index=False):
        pair_key = (str(row.pdb_id), str(row.Uniprot_id))
        pep_key = (str(row.pdb_id),)

        if pair_key in pair_unique:
            pep_seq_values.append(pair_unique[pair_key])
            source_values.append("pair_unique")
            status_values.append("resolved")
        elif pep_key in pep_unique:
            pep_seq_values.append(pep_unique[pep_key])
            source_values.append("pep_id_unique")
            status_values.append("resolved")
        elif ambiguous_policy == "first" and pair_key in pair_first:
            pep_seq_values.append(pair_first[pair_key])
            source_values.append("pair_first")
            status_values.append("resolved_ambiguous_first")
        elif ambiguous_policy == "first" and pep_key in pep_first:
            pep_seq_values.append(pep_first[pep_key])
            source_values.append("pep_id_first")
            status_values.append("resolved_ambiguous_first")
        elif pair_key in pair_ambiguous:
            pep_seq_values.append(None)
            source_values.append("pair_ambiguous")
            status_values.append("unresolved_ambiguous")
        elif pep_key in pep_ambiguous:
            pep_seq_values.append(None)
            source_values.append("pep_id_ambiguous")
            status_values.append("unresolved_ambiguous")
        else:
            pep_seq_values.append(None)
            source_values.append("not_found")
            status_values.append("unresolved_not_found")

    output = target.copy()
    insert_at = 1 if "pep_seq" not in output.columns else output.columns.get_loc("pep_seq")
    if "pep_seq" in output.columns:
        output = output.drop(columns=["pep_seq"])
    output.insert(insert_at, "pep_seq", pep_seq_values)
    output["pep_seq_mapping_source"] = source_values
    output["pep_seq_mapping_status"] = status_values
    output["clean_pep_seq"] = output["pep_seq"].map(clean_sequence)
    output["clean_prot_seq"] = output["prot_seq"].map(clean_sequence)

    unresolved = output["pep_seq"].isna()
    pair_cols = ["clean_pep_seq", "clean_prot_seq"]
    resolvable = output.loc[~unresolved].copy()
    conflict_pairs = int((resolvable.groupby(pair_cols)["label"].nunique().gt(1)).sum()) if len(resolvable) else 0

    audit = {
        "source_rows": int(len(source)),
        "target_rows": int(len(target)),
        "output_rows": int(len(output)),
        "row_count_preserved": bool(len(target) == len(output)),
        "target_label_counts": {str(k): int(v) for k, v in target["label"].value_counts().sort_index().to_dict().items()},
        "source_unique_pep_id": int(source["pep_id"].nunique()),
        "source_pep_id_with_multiple_pep_seq": int(len(pep_ambiguous)),
        "source_pair_with_multiple_pep_seq": int(len(pair_ambiguous)),
        "ambiguous_policy": ambiguous_policy,
        "mapping_status_counts": {str(k): int(v) for k, v in output["pep_seq_mapping_status"].value_counts().to_dict().items()},
        "mapping_source_counts": {str(k): int(v) for k, v in output["pep_seq_mapping_source"].value_counts().to_dict().items()},
        "resolved_rows": int((~unresolved).sum()),
        "unresolved_rows": int(unresolved.sum()),
        "unresolved_by_label": {
            str(k): int(v) for k, v in output.loc[unresolved, "label"].value_counts().sort_index().to_dict().items()
        },
        "empty_clean_pep_seq_rows": int(output["clean_pep_seq"].eq("").sum()),
        "empty_clean_prot_seq_rows": int(output["clean_prot_seq"].eq("").sum()),
        "sequence_pair_label_conflicts_on_resolved_rows": conflict_pairs,
        "ready_for_unseen_sequence_split": bool(
            unresolved.sum() == 0
            and output["clean_pep_seq"].eq("").sum() == 0
            and output["clean_prot_seq"].eq("").sum() == 0
            and conflict_pairs == 0
        ),
    }
    return output, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach pep_seq from 0-data/data.csv to Model 3 prott5pn.csv.")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--unresolved-csv", type=Path, default=DEFAULT_UNRESOLVED_CSV)
    parser.add_argument(
        "--ambiguous-policy",
        choices=["keep-unresolved", "first"],
        default="keep-unresolved",
        help="Cara menangani pep_id/pair yang punya lebih dari satu pep_seq.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output sudah ada: {args.output_csv}. Gunakan --overwrite untuk menimpa.")

    source = read_source(args.source_csv)
    target = read_target(args.target_csv)
    output, audit = attach_pep_seq(source, target, args.ambiguous_policy)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.unresolved_csv.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(args.output_csv, index=False)
    output.loc[output["pep_seq"].isna()].to_csv(args.unresolved_csv, index=False)

    audit["paths"] = {
        "source_csv": str(args.source_csv),
        "target_csv": str(args.target_csv),
        "output_csv": str(args.output_csv),
        "audit_json": str(args.audit_json),
        "unresolved_csv": str(args.unresolved_csv),
    }
    args.audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
