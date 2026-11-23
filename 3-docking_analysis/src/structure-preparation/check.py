#!/usr/bin/env python3
"""Check whether HADDOCK input PDB files exist for top100 peptide-protein pairs."""

from __future__ import annotations

# Publication layout bootstrap; model equations and training parameters are unchanged.
from pathlib import Path as _PublicationPath
import sys as _publication_sys
_PUB_ROOT = next(p for p in _PublicationPath(__file__).resolve().parents if (p / ".rki-publication.json").is_file())
_publication_sys.path.insert(0, str(_PUB_ROOT / "2-machine_learning_analysis"))

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = _PUB_ROOT
SHORTLIST_DIR = _PUB_ROOT / "2-machine_learning_analysis" / "cache" / "screening" / "voting" / "shortlist"
PEPTIDE_PDB_DIR = _PUB_ROOT / "1-data" / "structures" / "peptides"
PROTEIN_PDB_DIR = PROJECT_DIR / "1-data" / "structures" / "proteins"
OUT_DIR = _PUB_ROOT / "3-docking_analysis" / "cache" / "readiness"

XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

TOP100_FILES = {
    "ab51": SHORTLIST_DIR / "ab51" / "ab51_top100.xlsx",
    "ab66": SHORTLIST_DIR / "ab66" / "ab66_top100.xlsx",
}

PEPTIDE_LEGACY_DATASETS = {
    "ab51": ["ab66"],
    "ab66": ["a51"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Directory for CSV check reports.")
    parser.add_argument("--peptide-pdb-dir", type=Path, default=PEPTIDE_PDB_DIR, help="Directory containing peptide PDB files.")
    parser.add_argument("--protein-pdb-dir", type=Path, default=PROTEIN_PDB_DIR, help="Directory containing receptor/protein PDB files.")
    return parser.parse_args()


def _col_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _shared_strings(zf: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(t.text or "" for t in si.findall(".//a:t", XLSX_NS)) for si in root.findall("a:si", XLSX_NS)]


def _first_sheet_path(zf: ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("pr:Relationship", REL_NS)}
    sheet = workbook.find("a:sheets/a:sheet", XLSX_NS)
    if sheet is None:
        raise ValueError("Workbook has no sheet")
    rid = sheet.attrib[f"{{{OFFICE_REL}}}id"]
    target = targets[rid]
    return target if target.startswith("xl/") else f"xl/{target}"


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", XLSX_NS))
    value = cell.find("a:v", XLSX_NS)
    if value is None:
        return ""
    text = value.text or ""
    if cell.attrib.get("t") == "s" and text.isdigit():
        index = int(text)
        return shared[index] if index < len(shared) else text
    return text


def read_xlsx(path: Path) -> list[dict[str, str]]:
    rows: list[list[str]] = []
    with ZipFile(path) as zf:
        shared = _shared_strings(zf)
        root = ET.fromstring(zf.read(_first_sheet_path(zf)))

    for row_node in root.findall(".//a:sheetData/a:row", XLSX_NS):
        values: list[str] = []
        for cell in row_node.findall("a:c", XLSX_NS):
            index = _col_index(cell.attrib.get("r", "A1"))
            while len(values) <= index:
                values.append("")
            values[index] = _cell_value(cell, shared)
        rows.append(values)

    if not rows:
        return []
    header = [str(value).strip() for value in rows[0]]
    width = len(header)
    records: list[dict[str, str]] = []
    for raw in rows[1:]:
        row = raw[:width] + [""] * max(0, width - len(raw))
        if any(str(value).strip() for value in row):
            records.append({header[i]: row[i] for i in range(width)})
    return records


def parse_mean_plddt(pdb_file: Path) -> str:
    values: list[float] = []
    if not pdb_file.exists():
        return ""
    with pdb_file.open(errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
                try:
                    values.append(float(line[60:66]))
                except ValueError:
                    continue
    if not values:
        return ""
    return f"{sum(values) / len(values):.3f}"


def peptide_candidates(dataset: str, pep_id: str, peptide_pdb_dir: Path) -> list[Path]:
    dataset_names = [dataset] + PEPTIDE_LEGACY_DATASETS.get(dataset, [])
    seen: set[Path] = set()
    candidates: list[Path] = []
    for dataset_name in dataset_names:
        path = peptide_pdb_dir / f"{dataset_name}__{pep_id}.pdb"
        if path not in seen:
            seen.add(path)
            candidates.append(path)
    return candidates


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def check_dataset(dataset: str, shortlist_file: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    records = read_xlsx(shortlist_file)
    checked: list[dict[str, str]] = []
    for record in records:
        pep_id = record.get("PepID", "").strip()
        seq_pep = record.get("Seq peptida", "").strip()
        uniprot_id = record.get("Protein (Uniprot_ID)", "").strip()

        peptide_paths = peptide_candidates(dataset, pep_id, args.peptide_pdb_dir)
        peptide_pdb = first_existing(peptide_paths)
        protein_pdb = args.protein_pdb_dir / f"{uniprot_id}.pdb"

        peptide_ok = peptide_pdb is not None
        protein_ok = protein_pdb.exists()
        if peptide_ok and protein_ok:
            status = "READY"
        elif peptide_ok:
            status = "MISSING_PROTEIN_PDB"
        elif protein_ok:
            status = "MISSING_PEPTIDE_PDB"
        else:
            status = "MISSING_BOTH"

        checked.append({
            "dataset": dataset,
            "rank": record.get("No", ""),
            "pep_id": pep_id,
            "seq_pep": seq_pep,
            "uniprot_id": uniprot_id,
            "snake_venom_protein": record.get("Snake Venom Protein", ""),
            "cancer_protein": record.get("Cancer Protein", ""),
            "pred_probability": record.get("Predicted Probability", ""),
            "vote": record.get("Vote", ""),
            "consensus": record.get("Consensus", ""),
            "peptide_pdb_exists": str(peptide_ok),
            "protein_pdb_exists": str(protein_ok),
            "peptide_pdb_path": "" if peptide_pdb is None else str(peptide_pdb),
            "protein_pdb_path": str(protein_pdb),
            "canonical_peptide_pdb_path": str(args.peptide_pdb_dir / f"{dataset}__{pep_id}.pdb"),
            "peptide_legacy_candidates": ";".join(str(path) for path in peptide_paths[1:]),
            "mean_plddt": "" if peptide_pdb is None else parse_mean_plddt(peptide_pdb),
            "dock_input_status": status,
        })
    return checked


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(f">{row['fasta_id']}\n")
            handle.write(f"{row['seq_pep']}\n")


def missing_peptides_to_fold(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if row["peptide_pdb_exists"] == "True":
            continue
        key = (row["dataset"], row["pep_id"])
        counts[key] = counts.get(key, 0) + 1
        if key in by_key:
            continue
        fasta_id = f"{row['dataset']}__{row['pep_id']}"
        by_key[key] = {
            "dataset": row["dataset"],
            "pep_id": row["pep_id"],
            "seq_pep": row["seq_pep"],
            "length": str(len(row["seq_pep"])),
            "fasta_id": fasta_id,
            "expected_pdb": f"{fasta_id}.pdb",
            "canonical_peptide_pdb_path": row["canonical_peptide_pdb_path"],
            "first_rank": row["rank"],
            "first_uniprot_id": row["uniprot_id"],
            "first_cancer_protein": row["cancer_protein"],
            "snake_venom_protein": row["snake_venom_protein"],
            "pair_count": "0",
        }
    out = list(by_key.values())
    for row in out:
        row["pair_count"] = str(counts[(row["dataset"], row["pep_id"])])
    out.sort(key=lambda row: (row["dataset"], int(row["first_rank"] or "999999"), row["pep_id"]))
    return out


def missing_proteins(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_uniprot: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        if row["protein_pdb_exists"] == "True":
            continue
        uniprot_id = row["uniprot_id"]
        counts[uniprot_id] = counts.get(uniprot_id, 0) + 1
        if uniprot_id in by_uniprot:
            continue
        by_uniprot[uniprot_id] = {
            "uniprot_id": uniprot_id,
            "cancer_protein": row["cancer_protein"],
            "expected_protein_pdb_path": row["protein_pdb_path"],
            "first_dataset": row["dataset"],
            "first_rank": row["rank"],
            "pair_count": "0",
        }
    out = list(by_uniprot.values())
    for row in out:
        row["pair_count"] = str(counts[row["uniprot_id"]])
    out.sort(key=lambda row: (row["first_dataset"], int(row["first_rank"] or "999999"), row["uniprot_id"]))
    return out


def summarize(rows: list[dict[str, str]]) -> dict[str, int]:
    summary = {
        "pairs": len(rows),
        "ready_pairs": 0,
        "missing_peptide_pairs": 0,
        "missing_protein_pairs": 0,
        "unique_peptides": 0,
        "unique_missing_peptides": 0,
        "unique_proteins": 0,
        "unique_missing_proteins": 0,
    }
    peptides = {(row["dataset"], row["pep_id"]) for row in rows}
    missing_peptides = {(row["dataset"], row["pep_id"]) for row in rows if row["peptide_pdb_exists"] != "True"}
    proteins = {row["uniprot_id"] for row in rows}
    missing_proteins = {row["uniprot_id"] for row in rows if row["protein_pdb_exists"] != "True"}
    summary["ready_pairs"] = sum(row["dock_input_status"] == "READY" for row in rows)
    summary["missing_peptide_pairs"] = sum(row["peptide_pdb_exists"] != "True" for row in rows)
    summary["missing_protein_pairs"] = sum(row["protein_pdb_exists"] != "True" for row in rows)
    summary["unique_peptides"] = len(peptides)
    summary["unique_missing_peptides"] = len(missing_peptides)
    summary["unique_proteins"] = len(proteins)
    summary["unique_missing_proteins"] = len(missing_proteins)
    return summary


def print_dataset_report(dataset: str, summary: dict[str, int]) -> None:
    print(f"[{dataset}]")
    print(f"  pairs                    : {summary['pairs']}")
    print(f"  ready                    : {summary['ready_pairs']} / {summary['pairs']}")
    print(f"  missing peptide PDB      : {summary['missing_peptide_pairs']} pairs")
    print(f"  missing protein PDB      : {summary['missing_protein_pairs']} pairs")
    print(f"  unique peptides          : {summary['unique_peptides']}")
    print(f"  unique peptides missing  : {summary['unique_missing_peptides']}")
    print(f"  unique proteins          : {summary['unique_proteins']}")
    print(f"  unique proteins missing  : {summary['unique_missing_proteins']}")
    print()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[CHECK] HADDOCK input readiness")
    print("Shortlists:")
    for dataset, path in TOP100_FILES.items():
        print(f"  {dataset}: {path}")
    print("Protein PDB dir:")
    print(f"  {args.protein_pdb_dir}")
    print("Peptide PDB dir:")
    print(f"  {args.peptide_pdb_dir}")
    print()

    all_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for dataset, shortlist_file in TOP100_FILES.items():
        if not shortlist_file.exists():
            raise FileNotFoundError(shortlist_file)
        rows = check_dataset(dataset, shortlist_file, args)
        all_rows.extend(rows)
        write_csv(args.out_dir / f"{dataset}_pdb_check.csv", rows)
        summary = summarize(rows)
        summary_rows.append({"dataset": dataset, **{key: str(value) for key, value in summary.items()}})
        print_dataset_report(dataset, summary)

    write_csv(args.out_dir / "all_pdb_check.csv", all_rows)
    missing_peptide_rows = missing_peptides_to_fold(all_rows)
    missing_protein_rows = missing_proteins(all_rows)
    missing_peptide_csv = args.out_dir / "missing_peptides_to_fold.csv"
    missing_peptide_fasta = args.out_dir / "missing_peptides_to_fold.fasta"
    missing_protein_csv = args.out_dir / "missing_proteins.csv"
    write_csv(missing_peptide_csv, missing_peptide_rows)
    write_fasta(missing_peptide_fasta, missing_peptide_rows)
    write_csv(missing_protein_csv, missing_protein_rows)

    total_summary = summarize(all_rows)
    summary_rows.append({"dataset": "ALL", **{key: str(value) for key, value in total_summary.items()}})
    write_csv(args.out_dir / "pdb_check_summary.csv", summary_rows)

    print("[ALL]")
    print(f"  pairs                    : {total_summary['pairs']}")
    print(f"  ready                    : {total_summary['ready_pairs']} / {total_summary['pairs']}")
    print(f"  missing peptide PDB      : {total_summary['missing_peptide_pairs']} pairs")
    print(f"  missing protein PDB      : {total_summary['missing_protein_pairs']} pairs")
    print(f"  unique peptides missing  : {total_summary['unique_missing_peptides']}")
    print(f"  unique proteins missing  : {total_summary['unique_missing_proteins']}")
    print()
    print("[NEXT]")
    if total_summary["unique_missing_peptides"]:
        print(f"  Fold missing peptides    : {total_summary['unique_missing_peptides']} unique peptide PDBs")
        print(f"  Missing peptide CSV      : {missing_peptide_csv}")
        print(f"  Missing peptide FASTA    : {missing_peptide_fasta}")
    else:
        print("  Fold missing peptides    : none")
    if total_summary["unique_missing_proteins"]:
        print(f"  Add missing proteins     : {total_summary['unique_missing_proteins']} protein PDBs")
        print(f"  Missing protein CSV      : {missing_protein_csv}")
    else:
        print("  Add missing proteins     : none")
    print(f"  Reports written to       : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
