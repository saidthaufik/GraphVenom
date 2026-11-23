from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = ROOT.parent / "1-data" / "splits" / "unseen" / "graphs"
SCENARIOS = ["unseen_peptide", "unseen_protein", "unseen_both"]
N_SPLITS = 5


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    index_path = path.with_name(path.stem + "_idx.csv")
    if not index_path.exists():
        raise FileNotFoundError(path)
    indices = pd.read_csv(index_path)["row_index"].to_numpy(dtype=int)
    metadata = pd.read_csv(ROOT.parent / "1-data" / "processed" / "graphs" / "model5" / "metadata.csv")
    frame = metadata.iloc[indices].copy()
    frame["row_index"] = indices
    frame["peptide_group"] = frame["clean_pep_seq"].astype(str)
    frame["protein_group"] = frame["clean_prot_seq"].astype(str)
    return frame


def label_stats(df: pd.DataFrame, prefix: str) -> dict:
    n = int(len(df))
    pos = int(df["label"].sum()) if n else 0
    return {
        f"{prefix}_rows": n,
        f"{prefix}_pos": pos,
        f"{prefix}_neg": int(n - pos),
        f"{prefix}_pos_rate": round(pos / n, 6) if n else None,
    }


def pair_set(df: pd.DataFrame) -> set:
    if df.empty:
        return set()
    return set(zip(df["peptide_group"].astype(str), df["protein_group"].astype(str)))


def audit_fold(scenario: str, fold: int) -> dict:
    scenario_dir = SPLIT_ROOT / scenario
    train = read_csv(scenario_dir / f"fold_{fold}_train.csv")
    test = read_csv(scenario_dir / f"fold_{fold}_test.csv")
    buffer_path = scenario_dir / f"fold_{fold}_buffer.csv"
    buffer = read_csv(buffer_path) if buffer_path.exists() or buffer_path.with_name(buffer_path.stem + "_idx.csv").exists() else pd.DataFrame(columns=train.columns)

    train_idx = set(train["row_index"].astype(int))
    test_idx = set(test["row_index"].astype(int))
    buffer_idx = set(buffer["row_index"].astype(int)) if len(buffer) else set()

    train_pep = set(train["peptide_group"].astype(str))
    test_pep = set(test["peptide_group"].astype(str))
    train_prot = set(train["protein_group"].astype(str))
    test_prot = set(test["protein_group"].astype(str))
    buffer_pep = set(buffer["peptide_group"].astype(str)) if len(buffer) else set()
    buffer_prot = set(buffer["protein_group"].astype(str)) if len(buffer) else set()

    row = {
        "scenario": scenario,
        "fold": fold,
        **label_stats(train, "train"),
        **label_stats(test, "test"),
        **label_stats(buffer, "buffer"),
        "train_unique_peptide_seq": int(train["peptide_group"].nunique()),
        "test_unique_peptide_seq": int(test["peptide_group"].nunique()),
        "train_unique_protein_seq": int(train["protein_group"].nunique()),
        "test_unique_protein_seq": int(test["protein_group"].nunique()),
        "train_test_index_overlap": len(train_idx & test_idx),
        "train_buffer_index_overlap": len(train_idx & buffer_idx),
        "test_buffer_index_overlap": len(test_idx & buffer_idx),
        "peptide_sequence_overlap_train_test": len(train_pep & test_pep),
        "protein_sequence_overlap_train_test": len(train_prot & test_prot),
        "pair_sequence_overlap_train_test": len(pair_set(train) & pair_set(test)),
        "buffer_peptide_overlap_train": len(buffer_pep & train_pep),
        "buffer_peptide_overlap_test": len(buffer_pep & test_pep),
        "buffer_protein_overlap_train": len(buffer_prot & train_prot),
        "buffer_protein_overlap_test": len(buffer_prot & test_prot),
    }

    if scenario == "unseen_peptide":
        row["concept_ok"] = row["peptide_sequence_overlap_train_test"] == 0 and row["train_test_index_overlap"] == 0
    elif scenario == "unseen_protein":
        row["concept_ok"] = row["protein_sequence_overlap_train_test"] == 0 and row["train_test_index_overlap"] == 0
    elif scenario == "unseen_both":
        row["concept_ok"] = (
            row["peptide_sequence_overlap_train_test"] == 0
            and row["protein_sequence_overlap_train_test"] == 0
            and row["pair_sequence_overlap_train_test"] == 0
            and row["train_test_index_overlap"] == 0
            and row["train_buffer_index_overlap"] == 0
            and row["test_buffer_index_overlap"] == 0
        )
    else:
        row["concept_ok"] = False

    row["has_both_classes_train"] = row["train_pos"] > 0 and row["train_neg"] > 0
    row["has_both_classes_test"] = row["test_pos"] > 0 and row["test_neg"] > 0
    row["ready"] = bool(row["concept_ok"] and row["has_both_classes_train"] and row["has_both_classes_test"])
    return row


def main() -> None:
    rows = []
    for scenario in SCENARIOS:
        for fold in range(1, N_SPLITS + 1):
            rows.append(audit_fold(scenario, fold))

    audit = pd.DataFrame(rows)
    out_path = SPLIT_ROOT / "audit_summary.csv"
    audit.to_csv(out_path, index=False)

    cols = [
        "scenario",
        "fold",
        "train_rows",
        "test_rows",
        "buffer_rows",
        "test_pos_rate",
        "peptide_sequence_overlap_train_test",
        "protein_sequence_overlap_train_test",
        "pair_sequence_overlap_train_test",
        "concept_ok",
        "has_both_classes_test",
        "ready",
    ]
    print(audit[cols].to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
