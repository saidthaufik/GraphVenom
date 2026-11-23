#!/usr/bin/env python3
"""Build CSV/XLSX summaries from completed HADDOCK3 batch results."""

from __future__ import annotations

# Publication layout bootstrap; model equations and training parameters are unchanged.
from pathlib import Path as _PublicationPath
import sys as _publication_sys
_PUB_ROOT = next(p for p in _PublicationPath(__file__).resolve().parents if (p / ".rki-publication.json").is_file())
_publication_sys.path.insert(0, str(_PUB_ROOT / "2-machine_learning_analysis"))

import argparse
import csv
import posixpath
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = _PUB_ROOT / "3-docking_analysis" / "cache" / "haddock_pairs_manifest.csv"
DEFAULT_STATUS = _PUB_ROOT / "3-docking_analysis" / "cache" / "haddock_batch_status.csv"
DEFAULT_OUT_DIR = _PUB_ROOT / "3-docking_analysis" / "cache"


SUMMARY_COLUMNS = [
    "dataset",
    "rank",
    "pair_id",
    "pep_id",
    "name_pep",
    "seq_pep",
    "uniprot_id",
    "name_prot",
    "haddock_score",
    "vdw",
    "elec",
    "desolv",
    "bsa",
    "runtime_sec",
    "dock_status",
    "best_model",
    "best_model_pdb_gz",
    "run_folder",
    "pred_probability",
    "vote",
    "consensus",
    "mean_plddt",
    "started_at",
    "finished_at",
    "duplicate_count",
    "duplicate_pep_ids",
    "duplicate_pair_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_best_model_gz(pair_dir: Path, best_model: str) -> str:
    if not best_model:
        return ""

    capri_dir = pair_dir / "run_peptide_score" / "analysis" / "6_caprieval_analysis"
    model_path = (capri_dir / best_model).resolve()
    candidates = [model_path]
    if model_path.suffix == ".pdb":
        candidates.append(model_path.with_suffix(".pdb.gz"))
    elif model_path.suffix == ".gz":
        candidates.append(model_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


def build_summary(manifest_path: Path, status_path: Path) -> list[dict[str, str]]:
    manifest = {row["pair_id"]: row for row in read_csv(manifest_path)}
    status_rows = read_csv(status_path)

    rows: list[dict[str, str]] = []
    for status in status_rows:
        pair_id = status["pair_id"]
        meta = manifest.get(pair_id, {})
        pair_dir = Path(status.get("pair_dir") or meta.get("pair_dir", ""))
        best_model = status.get("best_model", "")

        row = {
            "dataset": status.get("dataset", meta.get("dataset", "")),
            "rank": status.get("rank", meta.get("rank", "")),
            "pair_id": pair_id,
            "pep_id": status.get("pep_id", meta.get("pep_id", "")),
            "name_pep": meta.get("snake_venom_protein", ""),
            "seq_pep": meta.get("seq_pep", ""),
            "uniprot_id": status.get("uniprot_id", meta.get("uniprot_id", "")),
            "name_prot": meta.get("cancer_protein", ""),
            "haddock_score": status.get("haddock_score", ""),
            "vdw": status.get("vdw", ""),
            "elec": status.get("elec", ""),
            "desolv": status.get("desolv", ""),
            "bsa": status.get("bsa", ""),
            "runtime_sec": status.get("runtime_sec", ""),
            "dock_status": status.get("status", ""),
            "best_model": best_model,
            "best_model_pdb_gz": resolve_best_model_gz(pair_dir, best_model),
            "run_folder": str(pair_dir),
            "pred_probability": meta.get("pred_probability", ""),
            "vote": meta.get("vote", ""),
            "consensus": meta.get("consensus", ""),
            "mean_plddt": meta.get("mean_plddt", ""),
            "started_at": status.get("started_at", ""),
            "finished_at": status.get("finished_at", ""),
        }
        rows.append(row)

    rows.sort(key=lambda row: (row["dataset"], numeric(row["haddock_score"], 999999.0)))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["dataset"], row["seq_pep"], row["uniprot_id"])
        groups.setdefault(key, []).append(row)

    unique_rows: list[dict[str, str]] = []
    removed_rows: list[dict[str, str]] = []

    for group_rows in groups.values():
        ranked = sorted(
            group_rows,
            key=lambda row: (
                numeric(row["haddock_score"], 999999.0),
                int(float(row["rank"])) if row["rank"] else 999999,
                row["pair_id"],
            ),
        )
        keeper = dict(ranked[0])
        duplicate_count = len(ranked)
        duplicate_pep_ids = sorted({row["pep_id"] for row in ranked if row["pep_id"]})
        duplicate_pair_ids = [row["pair_id"] for row in ranked if row["pair_id"]]
        keeper["duplicate_count"] = str(duplicate_count)
        keeper["duplicate_pep_ids"] = ";".join(duplicate_pep_ids)
        keeper["duplicate_pair_ids"] = ";".join(duplicate_pair_ids)
        unique_rows.append(keeper)

        for removed in ranked[1:]:
            removed_row = dict(removed)
            removed_row["duplicate_count"] = str(duplicate_count)
            removed_row["duplicate_pep_ids"] = ";".join(duplicate_pep_ids)
            removed_row["duplicate_pair_ids"] = ";".join(duplicate_pair_ids)
            removed_rows.append(removed_row)

    unique_rows.sort(key=lambda row: (row["dataset"], numeric(row["haddock_score"], 999999.0)))
    removed_rows.sort(key=lambda row: (row["dataset"], row["seq_pep"], row["uniprot_id"], int(float(row["rank"]))))
    return unique_rows, removed_rows


def sheet_xml(rows: list[list[str]]) -> str:
    col_widths = [
        10, 8, 34, 22, 32, 18, 14, 32, 14, 12, 12, 12, 12, 12, 12, 24, 80, 80,
        16, 10, 12, 12, 20, 20,
    ]
    cols = "".join(
        f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        for idx, width in enumerate(col_widths, start=1)
    )
    xml_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            col = column_name(c_idx)
            text = escape("" if value is None else str(value))
            cells.append(f'<c r="{col}{r_idx}" t="inlineStr"><is><t>{text}</t></is></c>')
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<cols>{cols}</cols>"
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        f'<autoFilter ref="A1:{column_name(len(SUMMARY_COLUMNS))}1"/>'
        "</worksheet>"
    )


def column_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def write_xlsx(path: Path, sheets: dict[str, list[dict[str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets)

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, len(sheet_names) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{workbook_sheets}</sheets></workbook>"
    )
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx in range(1, len(sheet_names) + 1):
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheet_names) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        zf.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            "</styleSheet>",
        )
        for idx, name in enumerate(sheet_names, start=1):
            rows = [SUMMARY_COLUMNS] + [[row.get(column, "") for column in SUMMARY_COLUMNS] for row in sheets[name]]
            zf.writestr(posixpath.join("xl", "worksheets", f"sheet{idx}.xml"), sheet_xml(rows))


def main() -> None:
    args = parse_args()
    rows = build_summary(args.manifest, args.status_csv)
    unique_rows, removed_rows = deduplicate_rows(rows)
    ab51 = [row for row in unique_rows if row["dataset"] == "ab51"]
    ab66 = [row for row in unique_rows if row["dataset"] == "ab66"]
    top20 = sorted(unique_rows, key=lambda row: numeric(row["haddock_score"], 999999.0))[:20]

    write_csv(args.out_dir / "haddock_summary_unique_all.csv", unique_rows)
    write_csv(args.out_dir / "haddock_summary_unique_ab51.csv", ab51)
    write_csv(args.out_dir / "haddock_summary_unique_ab66.csv", ab66)
    write_csv(args.out_dir / "haddock_summary_duplicates_removed.csv", removed_rows)
    write_xlsx(
        args.out_dir / "haddock_summary_unique.xlsx",
        {
            "all_unique_ranked": sorted(unique_rows, key=lambda row: numeric(row["haddock_score"], 999999.0)),
            "ab51_unique_ranked": ab51,
            "ab66_unique_ranked": ab66,
            "top20_unique": top20,
            "duplicates_removed": removed_rows,
        },
    )

    missing_models = sum(1 for row in unique_rows if row["best_model_pdb_gz"] and not Path(row["best_model_pdb_gz"]).exists())
    print(f"Original rows     : {len(rows)}")
    print(f"Unique rows       : {len(unique_rows)}")
    print(f"Duplicates removed: {len(removed_rows)}")
    print(f"AB51 unique       : {len(ab51)}")
    print(f"AB66 unique       : {len(ab66)}")
    print(f"Missing best model: {missing_models}")
    print(f"XLSX              : {args.out_dir / 'haddock_summary_unique.xlsx'}")
    print(f"CSV all unique    : {args.out_dir / 'haddock_summary_unique_all.csv'}")


if __name__ == "__main__":
    main()
