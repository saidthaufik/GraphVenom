#!/usr/bin/env python3
import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "cache" / "screening" / "predictions"
VOTING_DIR = PROJECT_ROOT / "cache" / "screening" / "voting"
DATASETS = ["c_a51", "c_ab66"]
MODELS = ["m6", "m8", "m10"]
XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


DISPLAY_COLUMNS = [
    "No",
    "PepID",
    "Seq peptida",
    "Snake Venom Protein",
    "Protein (Uniprot_ID)",
    "Cancer Protein",
    "Predicted Probability",
    "Vote",
    "Consensus",
    "Ensemble Label",
]


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


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
        raise ValueError("Workbook tidak punya sheet")
    rel_id = sheet.attrib[f"{{{OFFICE_REL}}}id"]
    target = targets[rel_id]
    return target if target.startswith("xl/") else f"xl/{target}"


def read_xlsx_basic(path: Path) -> List[Dict[str, str]]:
    with ZipFile(path) as zf:
        shared_strings = read_shared_strings(zf)
        sheet_path = first_sheet_path(zf)
        root = ET.fromstring(zf.read(sheet_path))

    rows: List[List[str]] = []
    for row_node in root.findall(".//a:sheetData/a:row", XLSX_NS):
        values: List[str] = []
        for cell in row_node.findall("a:c", XLSX_NS):
            idx = excel_col_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = cell_text(cell, shared_strings)
        rows.append(values)

    if not rows:
        return []
    header = [str(x).strip() for x in rows[0]]
    last = max((i for i, h in enumerate(header) if h), default=-1)
    header = header[: last + 1]
    out = []
    for raw in rows[1:]:
        row = raw[: len(header)] + [""] * max(0, len(header) - len(raw))
        if any(str(v).strip() for v in row):
            out.append(dict(zip(header, row)))
    return out


def read_table(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return read_xlsx_basic(path)


def xlsx_cell(value: Any, row_num: int, col_idx: int) -> str:
    ref = f"{excel_col_name(col_idx)}{row_num}"
    if value is None:
        text = ""
    else:
        text = str(value)
    if text.lower() == "nan":
        text = ""
    if len(text) > 32767:
        text = text[:32767]
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'


def write_xlsx_basic(rows: List[Dict[str, Any]], columns: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xml_rows = []
    xml_rows.append('<row r="1">' + ''.join(xlsx_cell(col, 1, i) for i, col in enumerate(columns)) + '</row>')
    for row_num, row in enumerate(rows, start=2):
        xml_rows.append('<row r="%d">%s</row>' % (row_num, ''.join(xlsx_cell(row.get(col, ""), row_num, i) for i, col in enumerate(columns))))

    last_col = excel_col_name(max(len(columns) - 1, 0))
    last_row = max(len(rows) + 1, 1)
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:{last_col}{last_row}"/>'
        '<sheetData>' + ''.join(xml_rows) + '</sheetData>'
        '</worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="ensemble" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def get_value(row: Dict[str, Any], *names: str) -> str:
    lookup = {norm_col(k): v for k, v in row.items()}
    for name in names:
        key = norm_col(name)
        if key in lookup:
            return str(lookup[key])
    return ""


def to_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return 0.0


def to_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def row_key(row: Dict[str, Any], fallback_idx: int) -> Tuple[str, str, str, str]:
    pep_id = get_value(row, "pep_id", "PepID")
    pep_seq = get_value(row, "seq_pep", "sequence_peptide", "pep_seq", "Seq peptida")
    uniprot = get_value(row, "uniprot_id", "Protein (Uniprot_ID)")
    prot_seq = get_value(row, "seq_prot", "prot_seq")
    if pep_id or pep_seq or uniprot:
        return pep_id, pep_seq, uniprot, prot_seq
    return str(fallback_idx), "", "", ""


def consensus_name(vote_pos: int) -> str:
    if vote_pos == 3:
        return "Strong"
    if vote_pos == 2:
        return "Majority"
    if vote_pos == 1:
        return "Weak"
    return "Negative"


def build_ensemble(dataset: str, threshold: float) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    model_rows: Dict[str, Dict[Tuple[str, str, str, str], Dict[str, str]]] = {}
    key_order: List[Tuple[str, str, str, str]] = []
    base_rows: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}

    for model in MODELS:
        path = PRED_DIR / f"{dataset}_pred_{model}.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"File prediksi belum ada: {path}")
        rows = read_table(path)
        by_key = {}
        for idx, row in enumerate(rows):
            key = row_key(row, idx)
            by_key[key] = row
            if key not in base_rows:
                base_rows[key] = row
                key_order.append(key)
        model_rows[model] = by_key
        print(f"[INFO] {dataset} {model}: {len(rows)} baris", flush=True)

    output_rows: List[Dict[str, Any]] = []
    stats = {"total": 0, "vote3": 0, "vote2": 0, "vote1": 0, "vote0": 0, "ensemble_positive": 0}

    for key in key_order:
        base = base_rows[key]
        probs = []
        labels = []
        model_values = {}
        for model in MODELS:
            row = model_rows[model].get(key, {})
            prob = to_float(get_value(row, "pred_prob", "Predicted Probability"))
            label = to_int(get_value(row, "pred_label"))
            probs.append(prob)
            labels.append(label)
            model_values[f"pred_prob_{model}"] = f"{prob:.6f}"
            model_values[f"pred_label_{model}"] = label

        vote_pos = int(sum(labels))
        pred_mean = sum(probs) / max(len(probs), 1)
        ensemble_label = 1 if vote_pos >= 2 else 0
        stats["total"] += 1
        stats[f"vote{vote_pos}"] += 1
        stats["ensemble_positive"] += ensemble_label

        out = {
            "PepID": get_value(base, "pep_id", "PepID"),
            "Seq peptida": get_value(base, "seq_pep", "sequence_peptide", "pep_seq", "Seq peptida"),
            "Snake Venom Protein": get_value(base, "pep_name", "family", "Snake Venom Protein"),
            "Protein (Uniprot_ID)": get_value(base, "uniprot_id", "Protein (Uniprot_ID)"),
            "Cancer Protein": get_value(base, "prot_name", "subtype", "Cancer Protein"),
            "Predicted Probability": f"{pred_mean:.6f}",
            "Vote": f"{vote_pos}/3",
            "Consensus": consensus_name(vote_pos),
            "Ensemble Label": ensemble_label,
            "vote_pos": vote_pos,
            "pred_mean_sort": pred_mean,
        }
        out.update(model_values)
        output_rows.append(out)

    output_rows.sort(key=lambda r: (-int(r["vote_pos"]), -float(r["pred_mean_sort"]), str(r.get("PepID", ""))))
    for idx, row in enumerate(output_rows, start=1):
        row["No"] = idx

    return output_rows, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Voting ensemble prediksi m6/m8/m10 untuk c_a51 dan c_ab66.")
    parser.add_argument("--datasets", default="c_a51,c_ab66", help="Comma-separated dataset: c_a51,c_ab66")
    parser.add_argument("--threshold", type=float, default=0.5, help="Disimpan sebagai info; label model sudah berasal dari predict.py.")
    parser.add_argument("--keep-detail", action="store_true", help="Tambahkan kolom pred_prob/pred_label tiap model di output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"Dataset tidak dikenal: {dataset}. Pilihan: {', '.join(DATASETS)}")
        rows, stats = build_ensemble(dataset, float(args.threshold))
        columns = list(DISPLAY_COLUMNS)
        if args.keep_detail:
            for model in MODELS:
                columns.extend([f"pred_prob_{model}", f"pred_label_{model}"])
        out_path = VOTING_DIR / f"{dataset}_ensemble.xlsx"
        write_xlsx_basic(rows, columns, out_path)
        print(
            f"[OK] {out_path} | total={stats['total']} | "
            f"3/3={stats['vote3']} | 2/3={stats['vote2']} | "
            f"1/3={stats['vote1']} | 0/3={stats['vote0']} | "
            f"ensemble_positif={stats['ensemble_positive']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
