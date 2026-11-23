#!/usr/bin/env python3
import argparse
import importlib.util
import inspect
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_ROOT = PROJECT_ROOT / "2-predict"
INPUT_DIR = PROJECT_ROOT / "cache" / "screening"
OUTPUT_DIR = PROJECT_ROOT / "cache" / "screening" / "predictions"

DATASETS = {
    "c_a51": INPUT_DIR / "c_a51.csv",
    "c_ab66": INPUT_DIR / "c_ab66.csv",
}

MODELS = {
    "m6": {
        "name": "model6_film_gat_af_sing_es",
        "train": PROJECT_ROOT / "model6" / "1-model" / "train_sing_es.py",
        "pre": PROJECT_ROOT / "model6" / "0-feat" / "pre.py",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "model6_final.pth",
    },
    "m8": {
        "name": "model8_film_gine_af_sing_es",
        "train": PROJECT_ROOT / "model8" / "1-model" / "train_sing_es.py",
        "pre": PROJECT_ROOT / "model8" / "0-feat" / "pre.py",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "model8_final.pth",
    },
    "m10": {
        "name": "model10_film_gcn_af_sing_es",
        "train": PROJECT_ROOT / "model10" / "1-model" / "train_sing_es.py",
        "pre": PROJECT_ROOT / "model10" / "0-feat" / "pre.py",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "model10_final.pth",
    },
}


def load_module(path: Path, name: str):
    path = path.resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Gagal memuat modul: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def resolve_col(columns, *names: str) -> str:
    mapping = {norm_col(col): col for col in columns}
    for name in names:
        key = norm_col(name)
        if key in mapping:
            return mapping[key]
    raise ValueError(f"Kolom tidak ditemukan: salah satu dari {names}. Kolom tersedia: {list(columns)}")


def progress_line(label: str, current: int, total: int, start_time: float, width: int = 18) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    pct = (current / total * 100.0) if total else 100.0
    filled = min(width, max(0, int(width * current / total))) if total else width
    bar = "#" * filled + "-" * (width - filled)
    speed = current / elapsed
    eta = (total - current) / speed if speed > 0 else 0.0
    sys.stdout.write(f"\r\033[K[{label}] [{bar}] {pct:5.1f}% {current}/{total} {speed:,.1f}/s ETA {eta/60:.1f}m")
    sys.stdout.flush()


def finish_progress() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


def _excel_col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def _excel_cell(value: Any, row_num: int, col_idx: int) -> str:
    ref = f"{_excel_col_name(col_idx)}{row_num}"
    text = "" if value is None else str(value)
    if text.lower() == "nan":
        text = ""
    if len(text) > 32767:
        text = text[:32767]
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'


def write_xlsx_basic(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [str(col) for col in df.columns]
    rows_xml = []
    rows_xml.append(f'<row r="1">{"".join(_excel_cell(col, 1, idx) for idx, col in enumerate(columns))}</row>')
    total = len(df)
    start = time.time()
    for row_pos, row in enumerate(df.itertuples(index=False, name=None), start=2):
        rows_xml.append(f'<row r="{row_pos}">{"".join(_excel_cell(value, row_pos, idx) for idx, value in enumerate(row))}</row>')
        done = row_pos - 1
        if done == 1 or done == total or done % 2000 == 0:
            progress_line("write xlsx", done, total, start)
    finish_progress()

    last_col = _excel_col_name(max(len(columns) - 1, 0))
    last_row = max(len(df) + 1, 1)
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:{last_col}{last_row}"/>'
        '<sheetData>' + ''.join(rows_xml) + '</sheetData>'
        '</worksheet>'
    )
    workbook_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="predictions" sheetId="1" r:id="rId1"/></sheets></workbook>'
    rels_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    workbook_rels_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
    content_types_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>'
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def safe_torch_load(torch_module, train_module, path: Path):
    if hasattr(train_module, "safe_torch_load"):
        return train_module.safe_torch_load(path)
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def load_input_dataframe(path: Path, limit: int = 0):
    import pandas as pd

    df = pd.read_csv(path)
    if limit > 0:
        df = df.head(limit).copy()
    pep_id_col = resolve_col(df.columns, "pep_id")
    pep_seq_col = resolve_col(df.columns, "seq_pep", "sequence_peptide", "pep_seq")
    prot_id_col = resolve_col(df.columns, "uniprot_id")
    prot_seq_col = resolve_col(df.columns, "seq_prot", "prot_seq")
    pep_name_col = resolve_col(df.columns, "pep_name", "subtype") if any(norm_col(c) in {"pep_name", "subtype"} for c in df.columns) else None
    prot_name_col = resolve_col(df.columns, "prot_name", "name_protein") if any(norm_col(c) in {"prot_name", "name_protein"} for c in df.columns) else None

    df[pep_id_col] = df[pep_id_col].fillna("").astype(str)
    df[pep_seq_col] = df[pep_seq_col].fillna("").astype(str)
    df[prot_id_col] = df[prot_id_col].fillna("").astype(str)
    df[prot_seq_col] = df[prot_seq_col].fillna("").astype(str)
    return df, {
        "pep_id": pep_id_col,
        "pep_seq": pep_seq_col,
        "prot_id": prot_id_col,
        "prot_seq": prot_seq_col,
        "pep_name": pep_name_col,
        "prot_name": prot_name_col,
    }


def resolve_pdb_dir(pre_module, csv_path: Path):
    if hasattr(pre_module, "resolve_default_pdb_dir"):
        return pre_module.resolve_default_pdb_dir(csv_path)
    return None


def build_unique_graphs(df, cols: Dict[str, str], pre_module, pdb_dir, dataset_key: str, model_key: str):
    peptide_cache: Dict[Tuple[str, str], Tuple[Any, Dict[str, Any]]] = {}
    protein_cache: Dict[Tuple[str, str], Tuple[Any, Dict[str, Any]]] = {}

    peptide_records = df[[cols["pep_id"], cols["pep_seq"]]].drop_duplicates().itertuples(index=False, name=None)
    peptide_records = list(peptide_records)
    protein_records = df[[cols["prot_id"], cols["prot_seq"]]].drop_duplicates().itertuples(index=False, name=None)
    protein_records = list(protein_records)

    start = time.time()
    for idx, (pep_id, pep_seq) in enumerate(peptide_records, start=1):
        graph, meta = pre_module.build_peptide_graph(pep_seq)
        peptide_cache[(str(pep_id), str(pep_seq))] = (graph, meta)
        if idx == 1 or idx == len(peptide_records) or time.time() - start > 1.0:
            progress_line(f"pep graph {dataset_key}/{model_key}", idx, len(peptide_records), start)
    finish_progress()

    start = time.time()
    for idx, (uniprot_id, prot_seq) in enumerate(protein_records, start=1):
        graph, meta = pre_module.build_protein_graph(prot_seq, uniprot_id, pdb_dir)
        protein_cache[(str(uniprot_id), str(prot_seq))] = (graph, meta)
        if idx == 1 or idx == len(protein_records) or time.time() - start > 1.0:
            progress_line(f"prot graph {dataset_key}/{model_key}", idx, len(protein_records), start)
    finish_progress()

    print(f"[INFO] cache graph: peptide={len(peptide_cache)}, protein={len(protein_cache)}", flush=True)
    return peptide_cache, protein_cache


def build_pair_payload(df, cols, peptide_cache, protein_cache):
    import numpy as np
    import pandas as pd

    peptide_graphs: List[Any] = []
    protein_graphs: List[Any] = []
    metadata_rows: List[Dict[str, Any]] = []
    start = time.time()
    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        values = row._asdict() if hasattr(row, "_asdict") else None
        # itertuples renames invalid columns; iloc keeps this robust enough for mixed CSV headers.
        src = df.iloc[idx - 1]
        pep_key = (str(src[cols["pep_id"]]), str(src[cols["pep_seq"]]))
        prot_key = (str(src[cols["prot_id"]]), str(src[cols["prot_seq"]]))
        pep_graph, pep_meta = peptide_cache[pep_key]
        prot_graph, prot_meta = protein_cache[prot_key]
        peptide_graphs.append(pep_graph)
        protein_graphs.append(prot_graph)
        metadata_rows.append(
            {
                "pep_id": pep_key[0],
                "uniprot_id": prot_key[0],
                "raw_pep_seq": pep_key[1],
                "raw_prot_seq": prot_key[1],
                "pep_len": pep_meta.get("clean_length", 0),
                "prot_len": prot_meta.get("clean_length", 0),
                "pep_unknown_count": pep_meta.get("unknown_count", 0),
                "prot_unknown_count": prot_meta.get("unknown_count", 0),
                "pep_is_dummy": pep_meta.get("is_dummy", 0),
                "prot_is_dummy": prot_meta.get("is_dummy", 0),
                "prot_used_structure": prot_meta.get("used_structure", 0),
                "prot_num_contacts": prot_meta.get("num_contacts", 0),
                "label": 0,
            }
        )
        if idx == 1 or idx == total or idx % 5000 == 0:
            progress_line("assemble pairs", idx, total, start)
    finish_progress()
    labels = np.zeros(total, dtype=np.int64)
    metadata = pd.DataFrame(metadata_rows)
    return metadata, peptide_graphs, protein_graphs, labels


def normalized_global_features(train_module, metadata, cfg):
    import numpy as np
    import pandas as pd

    mode = str(cfg.get("global_mode", "none"))
    pred_raw = train_module.build_global_feature_matrix(metadata)
    cols = train_module.select_global_columns(mode)
    if not cols:
        return np.zeros((len(metadata), 0), dtype=np.float32)

    model_name = train_module.PROJECT_DIR.name
    train_metadata = pd.read_csv(PROJECT_ROOT.parent / "1-data" / "processed" / "graphs" / model_name / "metadata.csv")
    train_raw = train_module.build_global_feature_matrix(train_metadata)
    with np.load(PROJECT_ROOT.parent / "1-data" / "splits" / "benchmark" / (model_name + "_membership.npz"), allow_pickle=False) as membership:
        trainval_indices = membership["trainval_indices"].copy()
    fit = train_raw[np.asarray(trainval_indices, dtype=np.int64)][:, cols].astype(np.float32, copy=False)
    mean = fit.mean(axis=0, keepdims=True)
    std = fit.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((pred_raw[:, cols].astype(np.float32, copy=False) - mean) / std).astype(np.float32)


class CachedPairDataset:
    def __init__(self, peptide_graphs, protein_graphs, labels, global_features):
        import numpy as np
        self.peptide_graphs = peptide_graphs
        self.protein_graphs = protein_graphs
        self.labels = np.asarray(labels, dtype=np.float32)
        self.global_features = np.asarray(global_features, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch
        return (
            self.peptide_graphs[idx].clone(),
            self.protein_graphs[idx].clone(),
            torch.tensor(self.global_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            idx,
        )


def predict_probs(train_module, checkpoint: Path, metadata, peptide_graphs, protein_graphs, labels, device_name: str, threshold: float, batch_size: int):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    payload = safe_torch_load(torch, train_module, checkpoint)
    cfg = dict(payload.get("config", {}))
    if not cfg:
        raise ValueError(f"Checkpoint tidak punya config: {checkpoint}")
    cfg["device"] = device_name
    cfg.setdefault("threshold", threshold)
    if "global_dim" not in cfg:
        cfg["global_dim"] = len(train_module.select_global_columns(str(cfg.get("global_mode", "none"))))

    device = torch.device(device_name)
    model = train_module.build_model(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    global_features = normalized_global_features(train_module, metadata, cfg)
    dataset = CachedPairDataset(peptide_graphs, protein_graphs, labels, global_features)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=train_module.collate_fn, num_workers=0)

    probs = np.zeros(len(labels), dtype=np.float32)
    start = time.time()
    done = 0
    with torch.no_grad():
        for pep_batch, prot_batch, global_batch, _, sample_indices in loader:
            pep_batch = pep_batch.to(device)
            prot_batch = prot_batch.to(device)
            global_batch = global_batch.to(device)
            logits = model(pep_batch, prot_batch, global_batch)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            probs[sample_indices.detach().cpu().numpy()] = batch_probs
            done += len(batch_probs)
            progress_line("predict", done, len(labels), start)
    finish_progress()
    return probs, (probs >= float(threshold)).astype(int), cfg


def predict_one(dataset_key: str, csv_path: Path, model_key: str, model_info: Dict[str, Any], args) -> Path:
    import torch

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not model_info["checkpoint"].exists():
        raise FileNotFoundError(model_info["checkpoint"])

    print(f"[INFO] Load input {dataset_key}: {csv_path}", flush=True)
    original_df, cols = load_input_dataframe(csv_path, limit=int(args.limit))
    if len(original_df) == 0:
        raise ValueError(f"Input kosong: {csv_path}")

    train_module = load_module(model_info["train"], f"predict_{model_key}_train")
    pre_module = load_module(model_info["pre"], f"predict_{model_key}_pre")
    pdb_dir = resolve_pdb_dir(pre_module, csv_path)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Fast cached mode {dataset_key}/{model_key}: rows={len(original_df)}, device={device}", flush=True)
    peptide_cache, protein_cache = build_unique_graphs(original_df, cols, pre_module, pdb_dir, dataset_key, model_key)
    metadata, peptide_graphs, protein_graphs, labels = build_pair_payload(original_df, cols, peptide_cache, protein_cache)

    print(f"[INFO] Prediksi {dataset_key} dengan {model_key} ...", flush=True)
    probs, pred_labels, _ = predict_probs(
        train_module=train_module,
        checkpoint=model_info["checkpoint"],
        metadata=metadata,
        peptide_graphs=peptide_graphs,
        protein_graphs=protein_graphs,
        labels=labels,
        device_name=device,
        threshold=float(args.threshold),
        batch_size=int(args.batch_size),
    )

    result = original_df.copy()
    result["pred_prob"] = probs
    result["pred_label"] = pred_labels
    result["model_name"] = model_info["name"]
    result["checkpoint"] = str(model_info["checkpoint"])
    result["threshold"] = float(args.threshold)

    out_path = OUTPUT_DIR / f"{dataset_key}_pred_{model_key}.xlsx"
    print(f"[INFO] Tulis output: {out_path}", flush=True)
    write_xlsx_basic(result, out_path)
    print(f"[OK] {out_path} | n={len(result)} | positif={int(pred_labels.sum())}", flush=True)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Prediksi cached cross product c_a51/c_ab66 dengan model6, model8, model10 sing_es.")
    parser.add_argument("--datasets", default="c_a51,c_ab66", help="Comma-separated: c_a51,c_ab66")
    parser.add_argument("--models", default="m6,m8,m10", help="Comma-separated: m6,m8,m10")
    parser.add_argument("--device", default="auto", help="auto, cpu, atau cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Batasi baris input untuk tes. 0 berarti semua.")
    parser.add_argument("--keep-tmp", action="store_true", help="Tidak dipakai lagi; disimpan agar command lama tetap kompatibel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_keys = [x.strip() for x in args.datasets.split(",") if x.strip()]
    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    outputs = []
    for dataset_key in dataset_keys:
        if dataset_key not in DATASETS:
            raise ValueError(f"Dataset tidak dikenal: {dataset_key}. Pilihan: {', '.join(DATASETS)}")
        for model_key in model_keys:
            if model_key not in MODELS:
                raise ValueError(f"Model tidak dikenal: {model_key}. Pilihan: {', '.join(MODELS)}")
            outputs.append(predict_one(dataset_key, DATASETS[dataset_key], model_key, MODELS[model_key], args))
    print("[DONE] Output dibuat:", flush=True)
    for path in outputs:
        print(f"  - {path}", flush=True)


if __name__ == "__main__":
    main()
