
# Publication layout bootstrap; model equations and training parameters are unchanged.
from pathlib import Path as _PublicationPath
import sys as _publication_sys
_PUB_ROOT = next(p for p in _PublicationPath(__file__).resolve().parents if (p / ".rki-publication.json").is_file())
_publication_sys.path.insert(0, str(_PUB_ROOT / "2-machine_learning_analysis"))
#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = _PUB_ROOT / "1-data" / "processed" / "screening" / "snake"
DEFAULT_JOBS = [
    (_PUB_ROOT / "1-data" / "original" / "screening" / "ab66_naja_sumatrana.xlsx", DATA_DIR / "ab66.csv"),
    (_PUB_ROOT / "1-data" / "original" / "screening" / "ab51_trimeresurus_albolabris.xlsx", DATA_DIR / "a51.csv"),
]

XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb/"


def column_index(cell_ref):
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def read_shared_strings(zip_file):
    try:
        root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    strings = []
    for item in root.findall("a:si", XLSX_NS):
        parts = [node.text or "" for node in item.findall(".//a:t", XLSX_NS)]
        strings.append("".join(parts))
    return strings


def cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        parts = [node.text or "" for node in cell.findall(".//a:t", XLSX_NS)]
        return "".join(parts)

    value_node = cell.find("a:v", XLSX_NS)
    if value_node is None:
        return ""

    value = value_node.text or ""
    if cell_type == "s":
        index = int(value)
        return shared_strings[index] if index < len(shared_strings) else value

    return value


def first_sheet_path(zip_file):
    workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
    rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pr:Relationship", REL_NS)
    }

    sheet = workbook.find("a:sheets/a:sheet", XLSX_NS)
    if sheet is None:
        raise ValueError("Workbook tidak memiliki sheet.")

    rel_id = sheet.attrib[f"{{{OFFICE_REL}}}id"]
    target = rel_targets[rel_id]
    return target if target.startswith("xl/") else f"xl/{target}"


def read_xlsx_first_sheet(path):
    with ZipFile(path) as zip_file:
        shared_strings = read_shared_strings(zip_file)
        sheet_path = first_sheet_path(zip_file)
        root = ET.fromstring(zip_file.read(sheet_path))

    rows = []
    for row_node in root.findall(".//a:sheetData/a:row", XLSX_NS):
        values = []
        for cell in row_node.findall("a:c", XLSX_NS):
            index = column_index(cell.attrib.get("r", "A1"))
            while len(values) <= index:
                values.append("")
            values[index] = cell_value(cell, shared_strings)
        rows.append(values)

    if not rows:
        return [], []

    header = [str(value).strip() for value in rows[0]]
    last_named_col = max((i for i, name in enumerate(header) if name), default=-1)
    if last_named_col < 0:
        return [], []

    header = header[: last_named_col + 1]
    data_rows = []
    for raw_row in rows[1:]:
        row = list(raw_row[: len(header)])
        row.extend([""] * (len(header) - len(row)))
        if any(str(value).strip() for value in row):
            data_rows.append(dict(zip(header, row)))

    return header, data_rows


def request_text(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": "projek-rki-sequence-fetcher/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def fasta_to_sequence(fasta_text):
    sequence_lines = []
    for line in fasta_text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        sequence_lines.append(line)
    return re.sub(r"[^A-Za-z*]", "", "".join(sequence_lines))


def get_protein_sequence_from_ncbi(accession, delay, timeout):
    query = urllib.parse.urlencode(
        {
            "db": "protein",
            "term": f"{accession}[accn]",
            "usehistory": "y",
            "retmode": "json",
            "retmax": "1",
        }
    )
    search_text = request_text(f"{NCBI_BASE_URL}esearch.fcgi?{query}", timeout)
    search_data = json.loads(search_text)
    result = search_data.get("esearchresult", {})

    count = int(result.get("count", "0"))
    if count == 0:
        return ""

    web_env = result["webenv"]
    query_key = result["querykey"]

    time.sleep(delay)

    fetch_query = urllib.parse.urlencode(
        {
            "db": "protein",
            "query_key": query_key,
            "WebEnv": web_env,
            "rettype": "fasta",
            "retmode": "text",
        }
    )
    fasta_text = request_text(f"{NCBI_BASE_URL}efetch.fcgi?{fetch_query}", timeout)
    return fasta_to_sequence(fasta_text)


def get_protein_sequence_from_uniprot(accession, timeout):
    url = f"{UNIPROT_BASE_URL}{urllib.parse.quote(accession)}.json"
    data = json.loads(request_text(url, timeout))
    sequence = data.get("sequence", {}).get("value", "")
    return re.sub(r"[^A-Za-z*]", "", sequence)


def read_existing_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return header, rows


def insert_after(items, new_item, after_item):
    output = []
    inserted = False
    for item in items:
        output.append(item)
        if item == after_item:
            output.append(new_item)
            inserted = True
    if not inserted:
        output.append(new_item)
    return output


def process_file(input_path, output_path, cache, delay, timeout):
    if output_path.exists():
        header, rows = read_existing_csv(output_path)
        source_label = f"resume CSV {output_path.name}"
    else:
        header, rows = read_xlsx_first_sheet(input_path)
        source_label = f"XLSX {input_path.name}"

    if "uniprot_id" not in header:
        raise ValueError(f"Kolom 'uniprot_id' tidak ditemukan di {input_path}")

    if "prot_seq" in header:
        output_header = header
    else:
        output_header = insert_after(header, "prot_seq", "uniprot_id")

    for row in rows:
        row.setdefault("prot_seq", "")

    stats = {
        "rows": len(rows),
        "blank_uniprot_id": 0,
        "skipped_existing": 0,
        "processed_rows": 0,
        "success_rows": 0,
        "failed_rows": 0,
        "unique_requested": 0,
        "ncbi_success": 0,
        "uniprot_success": 0,
    }

    for row in rows:
        accession = str(row.get("uniprot_id", "")).strip()
        existing_seq = str(row.get("prot_seq", "")).strip()
        if not accession:
            stats["blank_uniprot_id"] += 1
            continue
        if existing_seq:
            stats["skipped_existing"] += 1
            cache.setdefault(accession, (existing_seq, "existing"))

    pending_rows = [
        row for row in rows
        if str(row.get("uniprot_id", "")).strip()
        and not str(row.get("prot_seq", "")).strip()
    ]
    total_to_process = len(pending_rows)
    progress = 0

    print(
        f"Sumber data: {source_label}; rows={len(rows)}, "
        f"sudah_ada_seq={stats['skipped_existing']}, perlu_proses={total_to_process}",
        flush=True,
    )

    for row in pending_rows:
        accession = str(row.get("uniprot_id", "")).strip()
        progress += 1
        prefix = f"[{output_path.name}] {progress}/{total_to_process} {accession}"

        if accession in cache:
            sequence, source = cache[accession]
            row["prot_seq"] = sequence
            if sequence:
                print(f"{prefix} pakai cache {source} OK len={len(sequence)}", flush=True)
            else:
                print(f"{prefix} pakai cache gagal/tidak ditemukan", flush=True)
            continue

        stats["unique_requested"] += 1
        sequence = ""
        source = ""

        print(f"{prefix} coba NCBI ...", flush=True)
        try:
            sequence = get_protein_sequence_from_ncbi(accession, delay, timeout)
            if sequence:
                source = "NCBI"
                stats["ncbi_success"] += 1
                print(f"{prefix} OK dari NCBI len={len(sequence)}", flush=True)
            else:
                print(f"{prefix} NCBI kosong/tidak ditemukan, coba UniProt ...", flush=True)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"{prefix} NCBI gagal: {exc}; coba UniProt ...", flush=True)

        if not sequence:
            try:
                sequence = get_protein_sequence_from_uniprot(accession, timeout)
                if sequence:
                    source = "UniProt"
                    stats["uniprot_success"] += 1
                    print(f"{prefix} OK dari UniProt len={len(sequence)}", flush=True)
                else:
                    print(f"{prefix} UniProt kosong/tidak ditemukan", flush=True)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                print(f"{prefix} UniProt gagal: {exc}", flush=True)

        row["prot_seq"] = sequence
        cache[accession] = (sequence, source or "not_found")
        time.sleep(delay)

    stats["processed_rows"] = total_to_process
    stats["success_rows"] = sum(
        1 for row in rows
        if str(row.get("uniprot_id", "")).strip()
        and str(row.get("prot_seq", "")).strip()
    )
    stats["failed_rows"] = sum(
        1 for row in rows
        if str(row.get("uniprot_id", "")).strip()
        and not str(row.get("prot_seq", "")).strip()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return stats

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ambil sequence protein dari NCBI berdasarkan kolom uniprot_id XLSX."
    )
    parser.add_argument("--delay", type=float, default=0.4, help="Jeda antar request NCBI dalam detik.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout request NCBI dalam detik.")
    return parser.parse_args()


def main():
    args = parse_args()
    cache = {}
    total_success = 0
    total_failed = 0
    total_ncbi_success = 0
    total_uniprot_success = 0

    for input_path, output_path in DEFAULT_JOBS:
        print(f"Memproses: {input_path}")
        stats = process_file(input_path, output_path, cache, args.delay, args.timeout)
        total_success += stats["success_rows"]
        total_failed += stats["failed_rows"]
        total_ncbi_success += stats["ncbi_success"]
        total_uniprot_success += stats["uniprot_success"]
        print(f"Output: {output_path}")
        print(
            "Ringkasan: "
            f"rows={stats['rows']}, "
            f"berhasil_total={stats['success_rows']}, "
            f"gagal_sisa={stats['failed_rows']}, "
            f"sudah_ada_seq={stats['skipped_existing']}, "
            f"diproses_run_ini={stats['processed_rows']}, "
            f"dari_ncbi={stats['ncbi_success']}, "
            f"dari_uniprot={stats['uniprot_success']}, "
            f"uniprot_id_kosong={stats['blank_uniprot_id']}, "
            f"unique_diminta={stats['unique_requested']}"
        )

    print(f"Total protein yang sudah memiliki sequence: {total_success}")
    print(f"Total protein berhasil dari NCBI pada run ini: {total_ncbi_success}")
    print(f"Total protein berhasil dari UniProt pada run ini: {total_uniprot_success}")
    print(f"Total protein gagal/belum ditemukan sequence-nya: {total_failed}")


if __name__ == "__main__":
    main()
