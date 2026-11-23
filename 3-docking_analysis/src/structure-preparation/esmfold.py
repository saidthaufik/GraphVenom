#!/usr/bin/env python3
"""Prepare peptide FASTA and optionally run ESMFold to create peptide PDBs."""

from __future__ import annotations

# Publication layout bootstrap; model equations and training parameters are unchanged.
from pathlib import Path as _PublicationPath
import sys as _publication_sys
_PUB_ROOT = next(p for p in _PublicationPath(__file__).resolve().parents if (p / ".rki-publication.json").is_file())
_publication_sys.path.insert(0, str(_PUB_ROOT / "2-machine_learning_analysis"))

import argparse
import csv
import re
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUTS = [_PUB_ROOT / "1-data" / "processed" / "screening" / "snake" / "a51.csv", _PUB_ROOT / "1-data" / "processed" / "screening" / "snake" / "ab66.csv"]
DEFAULT_OUT_DIR = _PUB_ROOT / "3-docking_analysis" / "cache" / "structures" / "peptides"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read snake peptide CSV files, write a FASTA for ESMFold, and "
            "optionally run ESMFold to generate peptide PDB files."
        )
    )
    parser.add_argument("--input", action="append", type=Path, default=None, help="Input CSV file. Can be passed more than once. Defaults to a51.csv and ab66.csv.")
    parser.add_argument("--from-check", type=Path, default=None, help="Use pdb_check/missing_peptides_to_fold.csv from check.py instead of raw snake CSV files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for FASTA, manifest, and PDB files.")
    parser.add_argument("--run-esmfold", action="store_true", help="Run ESMFold after writing the FASTA. Requires the esmfold conda env dependencies.")
    parser.add_argument("--num-recycles", type=int, default=None, help="Optional number of recycles for ESMFold inference.")
    parser.add_argument("--chunk-size", type=int, default=128, help="Axial attention chunk size to reduce memory use. Use 0 to disable.")
    parser.add_argument("--cpu-only", action="store_true", help="Run ESMFold on CPU instead of CUDA.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N peptides. Useful for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Regenerate PDB files even when they already exist.")
    return parser.parse_args()


def dataset_from_path(path: Path) -> str:
    name = path.stem.lower()
    if name in {"a51", "ab51"}:
        return name
    if name == "ab66":
        return name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def normalize_sequence(sequence: str) -> str:
    return re.sub(r"\s+", "", sequence).upper()


def load_check_peptides(check_file: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    with check_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "pep_id", "seq_pep"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{check_file} missing required columns: {', '.join(sorted(missing))}")

        for line_no, row in enumerate(reader, start=2):
            dataset = (row.get("dataset") or "").strip()
            pep_id = (row.get("pep_id") or "").strip()
            sequence = normalize_sequence(row.get("seq_pep") or "")
            if not dataset or not pep_id or not sequence:
                warnings.append(f"{check_file}:{line_no}: skipped empty dataset/pep_id/seq_pep")
                continue
            invalid = sorted(set(sequence).difference(VALID_AA))
            if invalid:
                warnings.append(f"{check_file}:{line_no}: skipped {pep_id}; invalid amino acids: {''.join(invalid)}")
                continue

            key = (dataset, pep_id, sequence)
            if key in seen:
                continue
            seen.add(key)

            fasta_id = (row.get("fasta_id") or f"{dataset}__{pep_id}").strip()
            expected_pdb = (row.get("expected_pdb") or f"{fasta_id}.pdb").strip()
            rows.append({
                "dataset": dataset,
                "pep_id": pep_id,
                "sequence_peptide": sequence,
                "length": str(len(sequence)),
                "uniprot_id": (row.get("first_uniprot_id") or "").strip(),
                "species": "",
                "family": "",
                "subtype": "",
                "fasta_id": fasta_id,
                "expected_pdb": expected_pdb,
            })

    return rows, warnings


def load_peptides(input_files: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for input_file in input_files:
        dataset = dataset_from_path(input_file)
        with input_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"pep_id", "sequence_peptide"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{input_file} missing required columns: {', '.join(sorted(missing))}")

            for line_no, row in enumerate(reader, start=2):
                pep_id = (row.get("pep_id") or "").strip()
                sequence = normalize_sequence(row.get("sequence_peptide") or "")
                uniprot_id = (row.get("uniprot_id") or "").strip()
                species = (row.get("species") or "").strip()
                family = (row.get("family") or "").strip()
                subtype = (row.get("subtype") or "").strip()

                if not pep_id or not sequence:
                    warnings.append(f"{input_file}:{line_no}: skipped empty pep_id/sequence")
                    continue
                invalid = sorted(set(sequence).difference(VALID_AA))
                if invalid:
                    warnings.append(f"{input_file}:{line_no}: skipped {pep_id}; invalid amino acids: {''.join(invalid)}")
                    continue

                key = (dataset, pep_id, sequence)
                if key in seen:
                    continue
                seen.add(key)

                fasta_id = f"{dataset}__{pep_id}"
                rows.append({
                    "dataset": dataset,
                    "pep_id": pep_id,
                    "sequence_peptide": sequence,
                    "length": str(len(sequence)),
                    "uniprot_id": uniprot_id,
                    "species": species,
                    "family": family,
                    "subtype": subtype,
                    "fasta_id": fasta_id,
                    "expected_pdb": f"{fasta_id}.pdb",
                })

    return rows, warnings


def write_fasta(rows: list[dict[str, str]], fasta_file: Path) -> None:
    with fasta_file.open("w") as handle:
        for row in rows:
            handle.write(f">{row['fasta_id']}\n")
            handle.write(f"{row['sequence_peptide']}\n")


def parse_mean_plddt(pdb_file: Path) -> float | None:
    values: list[float] = []
    if not pdb_file.exists():
        return None
    with pdb_file.open(errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
                try:
                    values.append(float(line[60:66]))
                except ValueError:
                    continue
    if not values:
        return None
    return sum(values) / len(values)


def write_manifest(rows: list[dict[str, str]], manifest_file: Path, pdb_dir: Path) -> None:
    fieldnames = ["dataset", "pep_id", "sequence_peptide", "length", "uniprot_id", "species", "family", "subtype", "fasta_id", "pdb_file", "pdb_status", "mean_plddt"]
    with manifest_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pdb_file = pdb_dir / row["expected_pdb"]
            plddt = parse_mean_plddt(pdb_file)
            writer.writerow({
                "dataset": row["dataset"],
                "pep_id": row["pep_id"],
                "sequence_peptide": row["sequence_peptide"],
                "length": row["length"],
                "uniprot_id": row["uniprot_id"],
                "species": row["species"],
                "family": row["family"],
                "subtype": row["subtype"],
                "fasta_id": row["fasta_id"],
                "pdb_file": str(pdb_file),
                "pdb_status": "ok" if pdb_file.exists() else "missing",
                "mean_plddt": "" if plddt is None else f"{plddt:.3f}",
            })


def run_esmfold(args: argparse.Namespace, rows: list[dict[str, str]], pdb_dir: Path) -> None:
    import torch
    import esm

    device = "cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda"
    print(f"ESMFold       : loading model on {device}", flush=True)
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if device == "cuda":
        model = model.cuda()
    if args.chunk_size and args.chunk_size > 0:
        model.set_chunk_size(args.chunk_size)

    selected_rows = rows[: args.limit] if args.limit else rows
    total = len(selected_rows)
    started_all = time.time()
    print(f"ESMFold       : folding {total} peptide(s)", flush=True)

    for index, row in enumerate(selected_rows, start=1):
        pdb_file = pdb_dir / row["expected_pdb"]
        if pdb_file.exists() and not args.force:
            print(f"[{index}/{total}] skip {row['fasta_id']} -> exists", flush=True)
            continue

        started = time.time()
        print(f"[{index}/{total}] fold {row['fasta_id']} len={row['length']} seq={row['sequence_peptide']}", flush=True)
        infer_kwargs = {}
        if args.num_recycles is not None:
            infer_kwargs["num_recycles"] = args.num_recycles
        with torch.no_grad():
            pdb = model.infer_pdb(row["sequence_peptide"], **infer_kwargs)
        pdb_file.write_text(pdb)
        plddt = parse_mean_plddt(pdb_file)
        plddt_text = "NA" if plddt is None else f"{plddt:.2f}"
        print(f"[{index}/{total}] done {row['fasta_id']} sec={time.time() - started:.1f} mean_plddt={plddt_text}", flush=True)

    print(f"ESMFold       : finished in {time.time() - started_all:.1f} sec", flush=True)


def main() -> int:
    args = parse_args()
    if args.from_check and args.input:
        raise SystemExit("Use either --from-check or --input, not both.")
    input_files = args.input or DEFAULT_INPUTS
    input_files = [path.resolve() for path in input_files]
    out_dir = args.out_dir.resolve()
    pdb_dir = out_dir / "pdb"
    fasta_file = out_dir / "snake_peptides.fasta"
    manifest_file = out_dir / "peptide_pdb_manifest.csv"
    warnings_file = out_dir / "warnings.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    if args.from_check:
        rows, warnings = load_check_peptides(args.from_check.resolve())
        input_label = str(args.from_check.resolve())
    else:
        rows, warnings = load_peptides(input_files)
        input_label = ", ".join(str(path) for path in input_files)
    if not rows:
        raise SystemExit("No valid peptide sequences found.")

    write_fasta(rows, fasta_file)
    if warnings:
        warnings_file.write_text("\n".join(warnings) + "\n")
    elif warnings_file.exists():
        warnings_file.unlink()

    print(f"Inputs        : {input_label}")
    print(f"Peptides      : {len(rows)}")
    print(f"FASTA         : {fasta_file}")
    print(f"PDB directory : {pdb_dir}")

    if args.run_esmfold:
        run_esmfold(args, rows, pdb_dir)
    else:
        print("ESMFold       : skipped; add --run-esmfold after the esmfold env is ready")

    write_manifest(rows, manifest_file, pdb_dir)
    print(f"Manifest      : {manifest_file}")
    if warnings:
        print(f"Warnings      : {warnings_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
