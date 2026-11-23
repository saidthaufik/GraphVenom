"""Read an input CSV, clean `pep_seq` and `prot_seq`, extract ESM-2 sequence embeddings,
and save the combined features to `data_esm2.csv`.
"""

import os
import shutil
import urllib.request
import warnings
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# This script reads the input CSV, cleans `pep_seq` and `prot_seq`,
# extracts sequence-level ESM-2 embeddings, and saves the result to `data_esm2.csv`.

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
MODEL_DIR = Path(__file__).resolve().parent
INPUT_CSV = str(PROJECT_ROOT.parent / "1-data" / "processed" / "training" / "gan_decoy.csv")
OUTPUT_CSV = str(MODEL_DIR / "data_esm2.csv")
MODEL_NAME = "esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
MAX_PEP_LENGTH = None
MAX_PROT_LENGTH = 1024
REPR_LAYER = 33
EMBEDDING_DIM = 1280
CACHE_DIR = str(MODEL_DIR)
MODEL_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt"
REGRESSION_URL = "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt"
REQUIRED_COLUMNS = ["pep_id", "pep_seq", "Uniprot_id", "prot_seq", "label"]
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    """Clean a sequence and keep only the 20 standard amino-acid letters."""
    if pd.isna(seq):
        seq = ""
    else:
        seq = str(seq)

    seq = seq.upper()
    seq = seq.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    return "".join(char for char in seq if char in STANDARD_AMINO_ACIDS)


def truncate_sequence(seq: str, max_length: int | None) -> str:
    """Truncate a sequence to `max_length` if a limit is provided."""
    if max_length is None:
        return seq
    return seq[:max_length]


def download_with_progress(url: str, destination_path: str):
    """Download a file with a visible tqdm progress bar."""
    temp_path = f"{destination_path}.part"

    if os.path.exists(temp_path):
        os.remove(temp_path)

    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        with open(temp_path, "wb") as output_file, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {os.path.basename(destination_path)}",
        ) as progress_bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
                progress_bar.update(len(chunk))

    shutil.move(temp_path, destination_path)


def ensure_model_checkpoint() -> tuple[str, str]:
    """Ensure the ESM-2 model and regression checkpoints exist in the cache directory."""
    checkpoint_dir = os.path.join(CACHE_DIR, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"{MODEL_NAME}.pt")
    regression_path = os.path.join(checkpoint_dir, f"{MODEL_NAME}-contact-regression.pt")

    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint model ditemukan: {checkpoint_path}")
    else:
        print(f"Checkpoint model belum ada. Mengunduh ke: {checkpoint_path}")
        download_with_progress(MODEL_URL, checkpoint_path)

    if os.path.exists(regression_path):
        print(f"Checkpoint regression ditemukan: {regression_path}")
    else:
        print(f"Checkpoint regression belum ada. Mengunduh ke: {regression_path}")
        download_with_progress(REGRESSION_URL, regression_path)

    return checkpoint_path, regression_path


def load_esm2_model():
    """Load the fixed ESM-2 pretrained model and its alphabet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.hub.set_dir(CACHE_DIR)
    checkpoint_path, _ = ensure_model_checkpoint()

    if hasattr(esm.pretrained, "load_model_and_alphabet_local"):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(checkpoint_path)
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    model = model.to(DEVICE)
    model.eval()
    return model, alphabet


def embed_sequences(
    sequences,
    model,
    alphabet,
    device,
    batch_size,
    repr_layer,
    embedding_dim,
    seq_type,
):
    """Embed a list of sequences with ESM-2 using residue-only mean pooling."""
    num_sequences = len(sequences)
    embeddings = np.zeros((num_sequences, embedding_dim), dtype=np.float32)
    batch_converter = alphabet.get_batch_converter()

    for start_idx in tqdm(
        range(0, num_sequences, batch_size),
        desc=f"Embedding {seq_type}",
        unit="batch",
    ):
        end_idx = min(start_idx + batch_size, num_sequences)
        batch_sequences = sequences[start_idx:end_idx]
        non_empty_items = []
        non_empty_indices = []

        for offset, sequence in enumerate(batch_sequences):
            if not sequence:
                warnings.warn(
                    f"{seq_type} sequence at row index {start_idx + offset} is empty after cleaning; "
                    f"using a zero vector of length {embedding_dim}.",
                    stacklevel=2,
                )
                continue

            non_empty_indices.append(offset)
            non_empty_items.append((f"{seq_type}_{start_idx + offset}", sequence))

        if not non_empty_items:
            continue

        _, _, batch_tokens = batch_converter(non_empty_items)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = results["representations"][repr_layer]

        tokens_len = (batch_tokens != alphabet.padding_idx).sum(dim=1)

        for item_idx, original_offset in enumerate(non_empty_indices):
            current_len = int(tokens_len[item_idx].item())
            if current_len <= 2:
                warnings.warn(
                    f"{seq_type} sequence at row index {start_idx + original_offset} has no residue tokens "
                    f"after tokenization; using a zero vector of length {embedding_dim}.",
                    stacklevel=2,
                )
                continue

            sequence_embedding = (
                token_representations[item_idx, 1 : current_len - 1].mean(0).detach().cpu().numpy()
            )
            embeddings[start_idx + original_offset] = sequence_embedding.astype(np.float32, copy=False)

        del batch_tokens
        del results
        del token_representations
        del tokens_len

        if device.startswith("cuda") and end_idx % max(batch_size * 10, 1) == 0:
            torch.cuda.empty_cache()

    return embeddings


def build_embedding_df(embeddings, prefix):
    """Convert an embedding matrix into a DataFrame with the required column names."""
    columns = [f"{prefix}_{idx}" for idx in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=columns)


def main():
    """Run CSV loading, cleaning, ESM-2 embedding extraction, and output generation."""
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"ESM cache directory: {CACHE_DIR}")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")

    print(f"Jumlah data: {len(df)}")

    pep_sequences = df["pep_seq"].apply(clean_sequence).apply(lambda seq: truncate_sequence(seq, MAX_PEP_LENGTH)).tolist()
    prot_sequences = df["prot_seq"].apply(clean_sequence).apply(lambda seq: truncate_sequence(seq, MAX_PROT_LENGTH)).tolist()

    model, alphabet = load_esm2_model()

    print("Mulai proses peptide...")
    pep_embeddings = embed_sequences(
        sequences=pep_sequences,
        model=model,
        alphabet=alphabet,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        repr_layer=REPR_LAYER,
        embedding_dim=EMBEDDING_DIM,
        seq_type="peptide",
    )
    print(f"Shape embedding peptide: {pep_embeddings.shape}")

    print("Mulai proses protein...")
    prot_embeddings = embed_sequences(
        sequences=prot_sequences,
        model=model,
        alphabet=alphabet,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        repr_layer=REPR_LAYER,
        embedding_dim=EMBEDDING_DIM,
        seq_type="protein",
    )
    print(f"Shape embedding protein: {prot_embeddings.shape}")

    pep_embedding_df = build_embedding_df(pep_embeddings, "pep_esm")
    prot_embedding_df = build_embedding_df(prot_embeddings, "prot_esm")

    final_df = pd.concat(
        [
            df[["pep_id", "Uniprot_id"]].reset_index(drop=True),
            pep_embedding_df.reset_index(drop=True),
            prot_embedding_df.reset_index(drop=True),
            df[["label"]].reset_index(drop=True),
        ],
        axis=1,
    )

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Output saved to: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()
