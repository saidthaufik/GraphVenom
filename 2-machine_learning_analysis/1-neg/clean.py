from pathlib import Path
import pandas as pd

# Input raw file (adjust if needed)
INPUT_PATH = Path(__file__).resolve().parents[2] / "1-data" / "original" / "training" / "positive.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "1-data" / "processed" / "training" / "data.csv"

ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    seq = (seq or "").upper()
    return "".join(aa for aa in seq if aa in ALLOWED_AA)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    required = ["pdb_id", "pep_seq", "Uniprot_id", "prot_seq"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang di input: {missing}")

    df = df[required].copy()

    # create pep_id
    df["pep_id"] = df["pdb_id"].astype(str)

    # clean sequences
    df["pep_seq"] = df["pep_seq"].apply(clean_sequence)
    df["prot_seq"] = df["prot_seq"].apply(clean_sequence)

    # drop empty/NA
    before = len(df)
    df = df.dropna(subset=["pep_id", "Uniprot_id", "pep_seq", "prot_seq"])
    df = df[(df["pep_seq"].str.len() > 0) & (df["prot_seq"].str.len() > 0)]
    removed = before - len(df)

    df_out = df[["pep_id", "Uniprot_id", "pep_seq", "prot_seq"]].drop_duplicates().reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"Input rows: {before}")
    print(f"Removed rows: {removed}")
    print(f"Output rows: {len(df_out)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
