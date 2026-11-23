#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT.parent / "1-data" / "original" / "training" / "positive.csv"
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "1-data" / "processed" / "screening" / "cancer" / "cancer.xlsx"
UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb/"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OUTPUT_COLUMNS = ["uniprot_id", "gene_id", "prot_seq", "name_protein"]
XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def request_text(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "projek-rki-cancer-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def clean_sequence(seq: Any) -> str:
    return re.sub(r"[^A-Za-z*]", "", str(seq or ""))


def protein_name_from_uniprot(data: Dict[str, Any]) -> str:
    desc = data.get("proteinDescription", {}) or {}
    recommended = desc.get("recommendedName", {}) or {}
    full = recommended.get("fullName", {}) or {}
    if full.get("value"):
        return str(full["value"])
    submitted = desc.get("submissionNames", []) or []
    for item in submitted:
        full = (item or {}).get("fullName", {}) or {}
        if full.get("value"):
            return str(full["value"])
    alt_names = desc.get("alternativeNames", []) or []
    for item in alt_names:
        full = (item or {}).get("fullName", {}) or {}
        if full.get("value"):
            return str(full["value"])
    return ""


def gene_id_from_uniprot(data: Dict[str, Any]) -> str:
    ids = []
    for xref in data.get("uniProtKBCrossReferences", []) or []:
        if str(xref.get("database", "")).lower() == "geneid":
            value = str(xref.get("id", "")).strip()
            if value:
                ids.append(value)
    return ";".join(dict.fromkeys(ids))


def fetch_uniprot(accession: str, timeout: float) -> Dict[str, str]:
    url = f"{UNIPROT_BASE_URL}{urllib.parse.quote(accession)}.json"
    data = json.loads(request_text(url, timeout))
    return {
        "uniprot_id": accession,
        "gene_id": gene_id_from_uniprot(data),
        "prot_seq": clean_sequence((data.get("sequence", {}) or {}).get("value", "")),
        "name_protein": protein_name_from_uniprot(data),
    }


def fasta_to_sequence(fasta_text: str) -> str:
    lines = []
    for line in fasta_text.splitlines():
        line = line.strip()
        if line and not line.startswith(">"):
            lines.append(line)
    return clean_sequence("".join(lines))


def fetch_ncbi_sequence(accession: str, timeout: float, delay: float) -> str:
    query = urllib.parse.urlencode({"db": "protein", "term": f"{accession}[accn]", "usehistory": "y", "retmode": "json", "retmax": "1"})
    search = json.loads(request_text(f"{NCBI_BASE_URL}esearch.fcgi?{query}", timeout))
    result = search.get("esearchresult", {})
    if int(result.get("count", "0")) == 0:
        return ""
    time.sleep(delay)
    fetch_query = urllib.parse.urlencode({"db": "protein", "query_key": result["querykey"], "WebEnv": result["webenv"], "rettype": "fasta", "retmode": "text"})
    return fasta_to_sequence(request_text(f"{NCBI_BASE_URL}efetch.fcgi?{fetch_query}", timeout))


def read_unique_uniprot_ids(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        field_map = {norm_col(name): name for name in (reader.fieldnames or [])}
        source_col = field_map.get("uniprot_id")
        if source_col is None:
            raise ValueError(f"Kolom uniprot_id tidak ditemukan di {path}; kolom tersedia: {reader.fieldnames}")
        ids = []
        seen = set()
        for row in reader:
            value = str(row.get(source_col, "")).strip()
            if value and value not in seen:
                ids.append(value)
                seen.add(value)
    return ids


def excel_col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def excel_col_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def xlsx_cell(value: Any, row_num: int, col_idx: int) -> str:
    ref = f"{excel_col_name(col_idx)}{row_num}"
    text = "" if value is None else str(value)
    if text.lower() == "nan":
        text = ""
    if len(text) > 32767:
        text = text[:32767]
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'


def write_xlsx_basic(rows: List[Dict[str, Any]], columns: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xml_rows = ['<row r="1">' + ''.join(xlsx_cell(col, 1, i) for i, col in enumerate(columns)) + '</row>']
    for row_num, row in enumerate(rows, start=2):
        xml_rows.append('<row r="%d">%s</row>' % (row_num, ''.join(xlsx_cell(row.get(col, ""), row_num, i) for i, col in enumerate(columns))))
    last_col = excel_col_name(max(len(columns) - 1, 0))
    last_row = max(len(rows) + 1, 1)
    sheet_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><dimension ref="A1:%s%d"/><sheetData>%s</sheetData></worksheet>' % (last_col, last_row, ''.join(xml_rows))
    workbook_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="cancer" sheetId="1" r:id="rId1"/></sheets></workbook>'
    rels_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    workbook_rels_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
    content_types_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>'
    with ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types_xml, compress_type=ZIP_DEFLATED)
        zf.writestr("_rels/.rels", rels_xml, compress_type=ZIP_DEFLATED)
        zf.writestr("xl/workbook.xml", workbook_xml, compress_type=ZIP_DEFLATED)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml, compress_type=ZIP_DEFLATED)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml, compress_type=ZIP_DEFLATED)


def read_existing_xlsx(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with ZipFile(path) as zf:
        try:
            shared = read_shared_strings(zf)
            sheet_path = first_sheet_path(zf)
            root = ET.fromstring(zf.read(sheet_path))
        except Exception:
            return {}
    rows = []
    for row_node in root.findall(".//a:sheetData/a:row", XLSX_NS):
        values = []
        for cell in row_node.findall("a:c", XLSX_NS):
            idx = excel_col_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = cell_text(cell, shared)
        rows.append(values)
    if not rows:
        return {}
    header = [str(x).strip() for x in rows[0]]
    existing = {}
    for raw in rows[1:]:
        row = raw[: len(header)] + [""] * max(0, len(header) - len(raw))
        item = dict(zip(header, row))
        uid = str(item.get("uniprot_id", "")).strip()
        if uid:
            existing[uid] = item
    return existing


def read_shared_strings(zf: ZipFile) -> List[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(node.text or "" for node in item.findall(".//a:t", XLSX_NS)) for item in root.findall("a:si", XLSX_NS)]


def first_sheet_path(zf: ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("pr:Relationship", REL_NS)}
    sheet = workbook.find("a:sheets/a:sheet", XLSX_NS)
    if sheet is None:
        raise ValueError("Workbook tidak punya sheet")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Ambil unique cancer protein dari positive.csv via UniProtKB.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah ID untuk tes. 0 berarti semua.")
    parser.add_argument("--no-ncbi-fallback", action="store_true", help="Matikan fallback sequence ke NCBI jika UniProt gagal/kosong.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = read_unique_uniprot_ids(args.input)
    if args.limit > 0:
        ids = ids[: args.limit]
    existing = read_existing_xlsx(args.output)
    rows: List[Dict[str, str]] = []
    success = 0
    failed = 0

    print(f"[INFO] unique uniprot_id: {len(ids)}", flush=True)
    if existing:
        print(f"[INFO] resume dari existing output: {len(existing)} ID", flush=True)

    for idx, accession in enumerate(ids, start=1):
        if accession in existing and str(existing[accession].get("prot_seq", "")).strip():
            item = {col: existing[accession].get(col, "") for col in OUTPUT_COLUMNS}
            rows.append(item)
            success += 1
            print(f"[{idx}/{len(ids)}] {accession} skip existing len={len(item['prot_seq'])}", flush=True)
            continue

        print(f"[{idx}/{len(ids)}] {accession} ambil UniProt ...", flush=True)
        item = {"uniprot_id": accession, "gene_id": "", "prot_seq": "", "name_protein": ""}
        try:
            item.update(fetch_uniprot(accession, args.timeout))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"  WARNING UniProt gagal: {exc}", flush=True)

        if not item["prot_seq"] and not args.no_ncbi_fallback:
            print(f"  UniProt sequence kosong/gagal, coba NCBI ...", flush=True)
            try:
                item["prot_seq"] = fetch_ncbi_sequence(accession, args.timeout, args.delay)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                print(f"  WARNING NCBI gagal: {exc}", flush=True)

        rows.append(item)
        if item["prot_seq"]:
            success += 1
            print(f"  OK len={len(item['prot_seq'])} gene_id={item['gene_id'] or '-'} name={item['name_protein'] or '-'}", flush=True)
        else:
            failed += 1
            print("  GAGAL sequence kosong", flush=True)
        time.sleep(args.delay)

    write_xlsx_basic(rows, OUTPUT_COLUMNS, args.output)
    print(f"[DONE] output: {args.output}", flush=True)
    print(f"[DONE] berhasil sequence: {success}; gagal/kosong: {failed}; total: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
