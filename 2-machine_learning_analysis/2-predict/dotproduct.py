#!/usr/bin/env python3
import argparse
import csv
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANCER = PROJECT_ROOT.parent / "1-data" / "processed" / "screening" / "cancer" / "cancer.xlsx"
DEFAULT_SNAKE_DIR = PROJECT_ROOT.parent / "1-data" / "processed" / "screening" / "snake"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cache" / "screening"
OUTPUT_COLUMNS = ["pep_id", "pep_name", "seq_pep", "uniprot_id", "prot_name", "seq_prot"]

XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def col_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def read_shared_strings(zf: ZipFile) -> List[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values = []
    for item in root.findall("a:si", XLSX_NS):
        values.append("".join(node.text or "" for node in item.findall(".//a:t", XLSX_NS)))
    return values


def first_sheet_path(zf: ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("pr:Relationship", REL_NS)}
    sheet = workbook.find("a:sheets/a:sheet", XLSX_NS)
    if sheet is None:
        raise ValueError("Workbook XLSX tidak punya sheet")
    rel_id = sheet.attrib[f"{{{OFFICE_REL}}}id"]
    target = targets[rel_id]
    return target if target.startswith("xl/") else f"xl/{target}"


def cell_text(cell, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", XLSX_NS))
    value_node = cell.find("a:v", XLSX_NS)
    if value_node is None:
        return ""
    value = value_node.text or ""
    if cell_type == "s":
        idx = int(value)
        return shared_strings[idx] if idx < len(shared_strings) else value
    return value


def read_xlsx(path: Path) -> List[Dict[str, str]]:
    with ZipFile(path) as zf:
        shared = read_shared_strings(zf)
        sheet = first_sheet_path(zf)
        root = ET.fromstring(zf.read(sheet))

    raw_rows: List[List[str]] = []
    for row_node in root.findall(".//a:sheetData/a:row", XLSX_NS):
        values: List[str] = []
        for cell in row_node.findall("a:c", XLSX_NS):
            idx = col_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = cell_text(cell, shared)
        raw_rows.append(values)

    if not raw_rows:
        return []
    header = [str(x).strip() for x in raw_rows[0]]
    last = max((i for i, h in enumerate(header) if h), default=-1)
    header = header[: last + 1]
    rows = []
    for raw in raw_rows[1:]:
        row = raw[: len(header)] + [""] * max(0, len(header) - len(raw))
        if any(str(v).strip() for v in row):
            rows.append(dict(zip(header, row)))
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def get(row: Dict[str, str], *names: str) -> str:
    lookup = {norm_col(k): v for k, v in row.items()}
    for name in names:
        value = lookup.get(norm_col(name))
        if value is not None:
            return str(value).strip()
    return ""


def load_cancer(path: Path) -> List[Dict[str, str]]:
    rows = read_xlsx(path)
    out = []
    seen = set()
    skipped = 0
    for row in rows:
        uid = get(row, "uniprot_id")
        seq = get(row, "prot_seq")
        name = get(row, "name_protein", "prot_name")
        if not uid or not seq:
            skipped += 1
            continue
        key = (uid, seq)
        if key in seen:
            continue
        seen.add(key)
        out.append({"uniprot_id": uid, "prot_name": name, "seq_prot": seq})
    print(f"[INFO] cancer proteins: valid={len(out)}, skipped_empty={skipped}, source={path}", flush=True)
    return out


def load_snake(path: Path) -> List[Dict[str, str]]:
    rows = read_csv(path)
    out = []
    skipped = 0
    for row in rows:
        pep_id = get(row, "pep_id")
        seq = get(row, "sequence_peptide", "pep_seq", "seq_pep")
        name = get(row, "subtype", "pep_name")
        if not pep_id or not seq:
            skipped += 1
            continue
        out.append({"pep_id": pep_id, "pep_name": name, "seq_pep": seq})
    print(f"[INFO] snake peptides: valid={len(out)}, skipped_empty={skipped}, source={path}", flush=True)
    return out


def write_cross_product(dataset_name: str, snake_path: Path, cancer_rows: List[Dict[str, str]], out_path: Path, flush_every: int) -> None:
    snake_rows = load_snake(snake_path)
    total_pairs = len(snake_rows) * len(cancer_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    written = 0
    print(f"[START] {dataset_name}: {len(snake_rows)} peptide x {len(cancer_rows)} cancer protein = {total_pairs} pasangan", flush=True)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for pep_idx, pep in enumerate(snake_rows, start=1):
            for cancer in cancer_rows:
                writer.writerow(
                    {
                        "pep_id": pep["pep_id"],
                        "pep_name": pep["pep_name"],
                        "seq_pep": pep["seq_pep"],
                        "uniprot_id": cancer["uniprot_id"],
                        "prot_name": cancer["prot_name"],
                        "seq_prot": cancer["seq_prot"],
                    }
                )
                written += 1

            if pep_idx == 1 or pep_idx % flush_every == 0 or pep_idx == len(snake_rows):
                handle.flush()
                elapsed = max(time.time() - start, 1e-6)
                rate = written / elapsed
                pct = (written / total_pairs * 100.0) if total_pairs else 100.0
                eta = (total_pairs - written) / rate if rate > 0 else 0.0
                print(
                    f"[{dataset_name}] peptide {pep_idx}/{len(snake_rows)} | "
                    f"written {written}/{total_pairs} ({pct:.1f}%) | "
                    f"speed {rate:.0f} rows/s | ETA {eta/60:.1f} min",
                    flush=True,
                )

    elapsed = time.time() - start
    print(f"[DONE] {dataset_name}: {out_path} | rows={written} | elapsed={elapsed/60:.1f} min", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross product peptide ular x protein cancer.")
    parser.add_argument("--cancer", type=Path, default=DEFAULT_CANCER)
    parser.add_argument("--snake-dir", type=Path, default=DEFAULT_SNAKE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="a51,ab66", help="Comma-separated: a51,ab66")
    parser.add_argument("--flush-every", type=int, default=10, help="Print progress setiap N peptide.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_map = {
        "a51": (args.snake_dir / "a51.csv", args.out_dir / "c_a51.csv"),
        "ab66": (args.snake_dir / "ab66.csv", args.out_dir / "c_ab66.csv"),
    }

    if not args.cancer.exists():
        raise FileNotFoundError(f"Cancer XLSX tidak ditemukan: {args.cancer}")
    cancer_rows = load_cancer(args.cancer)
    if not cancer_rows:
        raise ValueError("Tidak ada cancer protein valid. Cek kolom uniprot_id/prot_seq pada cancer.xlsx")

    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        if name not in dataset_map:
            raise ValueError(f"Dataset tidak dikenal: {name}. Pilihan: {', '.join(dataset_map)}")
        snake_path, out_path = dataset_map[name]
        if not snake_path.exists():
            raise FileNotFoundError(f"Snake CSV tidak ditemukan: {snake_path}")
        write_cross_product(name, snake_path, cancer_rows, out_path, max(1, int(args.flush_every)))


if __name__ == "__main__":
    main()
