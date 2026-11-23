import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT.parent / "1-data" / "processed" / "graphs" / "model5" / "metadata.csv"
OUT_ROOT = ROOT / "cache" / "splits" / "unseen"
SCENARIOS = {
    "unseen_peptide": "peptide",
    "unseen_protein": "protein",
    "unseen_both": "both",
}


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str:
    lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    raise ValueError(f"None of {list(candidates)} found in columns {list(columns)}")


def _required_column(columns: Sequence[str], name: str) -> str:
    lookup = {c.lower(): c for c in columns}
    if name.lower() not in lookup:
        raise ValueError(f"Required column not found for strict sequence-level split: {name}")
    return lookup[name.lower()]


def clean_sequence_df(csv_path: Path, peptide_level: str, protein_level: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    label_col = _first_existing(df.columns, ["label", "target", "y"])

    if peptide_level == "clean":
        pep_group_col = _required_column(df.columns, "clean_pep_seq")
    elif peptide_level == "raw":
        pep_group_col = _required_column(df.columns, "raw_pep_seq")
    else:
        raise ValueError(f"peptide_level tidak dikenal: {peptide_level}")

    if protein_level == "clean":
        prot_group_col = _required_column(df.columns, "clean_prot_seq")
    elif protein_level == "raw":
        prot_group_col = _required_column(df.columns, "raw_prot_seq")
    else:
        raise ValueError(f"protein_level tidak dikenal: {protein_level}")

    keep = [
        "pep_id",
        "uniprot_id",
        "raw_pep_seq",
        "raw_prot_seq",
        "clean_pep_seq",
        "clean_prot_seq",
        "pep_len",
        "prot_len",
        label_col,
    ]
    keep = [col for col in keep if col in df.columns]
    out = df[keep].dropna(subset=[pep_group_col, prot_group_col, label_col]).reset_index(drop=True)
    out["label"] = pd.to_numeric(out[label_col], errors="raise").astype(int)
    out["peptide_group"] = out[pep_group_col].astype(str)
    out["protein_group"] = out[prot_group_col].astype(str)
    out["peptide_group_source"] = peptide_level
    out["protein_group_source"] = protein_level
    out.insert(0, "row_index", range(len(out)))
    return out


def _group_counts(df: pd.DataFrame, group_col: str) -> List[Dict]:
    grouped = (
        df.groupby(group_col)["label"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"sum": "pos"})
    )
    grouped["neg"] = grouped["count"] - grouped["pos"]
    return grouped.to_dict("records")


def _greedy_group_folds(df: pd.DataFrame, group_col: str, n_splits: int, seed: int) -> List[Dict]:
    records = _group_counts(df, group_col)
    rng = random.Random(seed)
    rng.shuffle(records)
    records.sort(key=lambda r: (int(r["count"]), abs(int(r["pos"]) - int(r["neg"]))), reverse=True)

    folds = [{"groups": [], "n": 0, "pos": 0, "neg": 0} for _ in range(n_splits)]
    total_pos = int(df["label"].sum())
    target_pos = total_pos / n_splits

    if len(records) < n_splits:
        raise RuntimeError(f"Not enough {group_col} groups ({len(records)}) for {n_splits} folds")

    def add_to_fold(i: int, rec: Dict) -> None:
        folds[i]["groups"].append(rec[group_col])
        folds[i]["n"] += int(rec["count"])
        folds[i]["pos"] += int(rec["pos"])
        folds[i]["neg"] += int(rec["neg"])

    for i, rec in enumerate(records[:n_splits]):
        add_to_fold(i, rec)

    for rec in records[n_splits:]:
        n = int(rec["count"])
        pos = int(rec["pos"])
        neg = int(rec["neg"])

        def score(i: int):
            f = folds[i]
            new_n = f["n"] + n
            new_pos = f["pos"] + pos
            new_neg = f["neg"] + neg
            return (new_n, abs(new_pos - new_neg), abs(new_pos - target_pos))

        add_to_fold(min(range(n_splits), key=score), rec)

    return folds


def _write_indices(path: Path, idx: Sequence[int]) -> None:
    pd.DataFrame({"row_index": list(map(int, idx))}).to_csv(path, index=False)


def _write_rows(df: pd.DataFrame, path: Path, idx: Sequence[int]) -> None:
    df.loc[df["row_index"].isin(idx)].copy().to_csv(path, index=False)


def _count_labels(df: pd.DataFrame, idx: Sequence[int]) -> Dict[str, int]:
    sub = df.loc[df["row_index"].isin(idx)]
    pos = int(sub["label"].sum())
    return {"rows": int(len(sub)), "pos": pos, "neg": int(len(sub) - pos)}


def _write_stratified_group_scenario(
    df: pd.DataFrame,
    scenario: str,
    group_col: str,
    n_splits: int,
    seed: int,
    out_root: Path,
) -> pd.DataFrame:
    out_dir = out_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    y = df["label"].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    summary_rows = []
    groups_per_fold = []

    for fold_no, (train_pos, test_pos) in enumerate(splitter.split(df, y, groups), start=1):
        train_idx = df.iloc[train_pos]["row_index"].astype(int).tolist()
        test_idx = df.iloc[test_pos]["row_index"].astype(int).tolist()
        train_groups = set(df.iloc[train_pos][group_col].astype(str))
        test_groups = set(df.iloc[test_pos][group_col].astype(str))
        leakage = train_groups.intersection(test_groups)
        if leakage:
            raise RuntimeError(f"Leakage in {scenario} fold {fold_no}: {sorted(leakage)[:5]}")

        _write_indices(out_dir / f"fold_{fold_no}_train_idx.csv", train_idx)
        _write_indices(out_dir / f"fold_{fold_no}_test_idx.csv", test_idx)
        _write_rows(df, out_dir / f"fold_{fold_no}_train.csv", train_idx)
        _write_rows(df, out_dir / f"fold_{fold_no}_test.csv", test_idx)

        train_counts = _count_labels(df, train_idx)
        test_counts = _count_labels(df, test_idx)
        summary_rows.append(
            {
                "scenario": scenario,
                "fold": fold_no,
                "split_method": "StratifiedGroupKFold",
                "split_group": group_col,
                "test_groups": len(test_groups),
                "train_rows": train_counts["rows"],
                "train_pos": train_counts["pos"],
                "train_neg": train_counts["neg"],
                "test_rows": test_counts["rows"],
                "test_pos": test_counts["pos"],
                "test_neg": test_counts["neg"],
                "buffer_rows": 0,
                "peptide_leakage": 0 if group_col == "peptide_group" else "",
                "protein_leakage": 0 if group_col == "protein_group" else "",
            }
        )
        groups_per_fold.append(sorted(test_groups))

    (out_dir / "groups.json").write_text(json.dumps(groups_per_fold, indent=2), encoding="utf-8")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "fold_summary.csv", index=False)
    return summary


def _write_both_scenario(df: pd.DataFrame, n_splits: int, seed: int, out_root: Path) -> pd.DataFrame:
    out_dir = out_root / "unseen_both"
    out_dir.mkdir(parents=True, exist_ok=True)

    pep_folds = _greedy_group_folds(df, "peptide_group", n_splits, seed)
    prot_folds = _greedy_group_folds(df, "protein_group", n_splits, seed + 7919)
    summary_rows = []
    group_records = []

    for fold_no in range(1, n_splits + 1):
        pep_groups = set(pep_folds[fold_no - 1]["groups"])
        prot_groups = set(prot_folds[fold_no - 1]["groups"])
        pep_in = df["peptide_group"].isin(pep_groups)
        prot_in = df["protein_group"].isin(prot_groups)

        test_mask = pep_in & prot_in
        train_mask = (~pep_in) & (~prot_in)
        buffer_mask = ~(test_mask | train_mask)

        train_idx = df.loc[train_mask, "row_index"].tolist()
        test_idx = df.loc[test_mask, "row_index"].tolist()
        buffer_idx = df.loc[buffer_mask, "row_index"].tolist()

        train_pep = set(df.loc[train_mask, "peptide_group"])
        test_pep = set(df.loc[test_mask, "peptide_group"])
        train_prot = set(df.loc[train_mask, "protein_group"])
        test_prot = set(df.loc[test_mask, "protein_group"])
        pep_leakage = train_pep.intersection(test_pep)
        prot_leakage = train_prot.intersection(test_prot)
        if pep_leakage or prot_leakage:
            raise RuntimeError(
                f"Leakage in unseen_both fold {fold_no}: peptide={sorted(pep_leakage)[:5]}, protein={sorted(prot_leakage)[:5]}"
            )

        _write_indices(out_dir / f"fold_{fold_no}_train_idx.csv", train_idx)
        _write_indices(out_dir / f"fold_{fold_no}_test_idx.csv", test_idx)
        _write_indices(out_dir / f"fold_{fold_no}_buffer_idx.csv", buffer_idx)
        _write_rows(df, out_dir / f"fold_{fold_no}_train.csv", train_idx)
        _write_rows(df, out_dir / f"fold_{fold_no}_test.csv", test_idx)
        _write_rows(df, out_dir / f"fold_{fold_no}_buffer.csv", buffer_idx)

        train_counts = _count_labels(df, train_idx)
        test_counts = _count_labels(df, test_idx)
        buffer_counts = _count_labels(df, buffer_idx)
        summary_rows.append(
            {
                "scenario": "unseen_both",
                "fold": fold_no,
                "split_method": "greedy_sequence_group_with_buffer",
                "split_group": "peptide_group+protein_group",
                "test_peptide_groups": len(pep_groups),
                "test_protein_groups": len(prot_groups),
                "train_rows": train_counts["rows"],
                "train_pos": train_counts["pos"],
                "train_neg": train_counts["neg"],
                "test_rows": test_counts["rows"],
                "test_pos": test_counts["pos"],
                "test_neg": test_counts["neg"],
                "buffer_rows": buffer_counts["rows"],
                "buffer_pos": buffer_counts["pos"],
                "buffer_neg": buffer_counts["neg"],
                "peptide_leakage": 0,
                "protein_leakage": 0,
            }
        )
        group_records.append({"fold": fold_no, "peptide_groups": sorted(pep_groups), "protein_groups": sorted(prot_groups)})

    (out_dir / "groups.json").write_text(json.dumps(group_records, indent=2), encoding="utf-8")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "fold_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sequence-level unseen splits for RKI.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS.keys()], default="all")
    parser.add_argument("--peptide-level", choices=["raw", "clean"], default="clean")
    parser.add_argument("--protein-level", choices=["raw", "clean"], default="clean")
    args = parser.parse_args()

    df = clean_sequence_df(args.csv, args.peptide_level, args.protein_level)
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "split_type": "sequence-level",
        "csv": str(args.csv),
        "out_root": str(args.out_root),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "peptide_level": args.peptide_level,
        "protein_level": args.protein_level,
        "rows": int(len(df)),
        "positive": int(df["label"].sum()),
        "negative": int(len(df) - df["label"].sum()),
        "unique_peptide_sequence_group": int(df["peptide_group"].nunique()),
        "unique_protein_sequence_group": int(df["protein_group"].nunique()),
        "single_domain_split_method": "StratifiedGroupKFold",
        "both_split_method": "greedy_sequence_group_with_buffer",
        "scenarios": list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario],
    }
    (args.out_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        "Dataset: {csv}\nRows={rows} | peptide_seq_groups={pep} | protein_seq_groups={prot} | pos={pos} | neg={neg}".format(
            csv=args.csv,
            rows=len(df),
            pep=df["peptide_group"].nunique(),
            prot=df["protein_group"].nunique(),
            pos=int(df["label"].sum()),
            neg=int(len(df) - df["label"].sum()),
        )
    )

    summaries = []
    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        if scenario == "unseen_peptide":
            summary = _write_stratified_group_scenario(df, scenario, "peptide_group", args.n_splits, args.seed, args.out_root)
        elif scenario == "unseen_protein":
            summary = _write_stratified_group_scenario(df, scenario, "protein_group", args.n_splits, args.seed, args.out_root)
        elif scenario == "unseen_both":
            summary = _write_both_scenario(df, args.n_splits, args.seed, args.out_root)
        else:
            raise ValueError(scenario)
        summaries.append(summary)
        print(f"\n[{scenario}]")
        print(summary.to_string(index=False))

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(args.out_root / "split_summary.csv", index=False)
    print(f"\nSaved: {args.out_root}")


if __name__ == "__main__":
    main()
