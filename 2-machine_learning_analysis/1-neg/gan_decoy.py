from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from protlearn.features import atc  # ATC protlearn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# CONFIG
# =========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

POSITIVE_PATH = Path(__file__).resolve().parents[2] / "1-data" / "processed" / "training" / "data.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "1-data" / "processed" / "training" / "gan_decoy.csv"

NEGATIVE_RATIO = 1.0          # target akhirnya 50% label=1 dan 50% label=0

NOISE_DIM = 64
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALLOWED_AA = set('ACDEFGHIKLMNPQRSTVWY')

# kontrol kesulitan & Monte Carlo NN
K_MIN = 3                     # skip 3 tetangga terdekat (terlalu borderline)
K_MAX = 30                    # ambil kandidat di rentang rank [K_MIN, K_MAX)
SUBSET_SIZE = 500             # ukuran subset Monte Carlo NN per vektor syn

# kontrol decoy space
DECOY_PER_POS = 20            # jumlah decoy peptide per pasangan positif (per protein)


# =========================
# UTIL
# =========================
def clean_sequence(seq: str) -> str:
    """Buang residue non-standar, hanya simpan 20 AA utama."""
    seq = (seq or '').upper()
    return ''.join(aa for aa in seq if aa in ALLOWED_AA)


def seqs_to_atc_features(sequences):
    """Hitung fitur ATC (atoms + bonds) untuk list sekuens."""
    safe_seqs = [str(s or '') for s in sequences]
    if len(safe_seqs) == 0:
        return np.zeros((0, 0), dtype=float)
    atoms, bonds = atc(safe_seqs)
    atoms = np.asarray(atoms, dtype=float)
    bonds = np.asarray(bonds, dtype=float)
    feats = np.concatenate([atoms, bonds], axis=1)
    return feats


# =========================
# STEP A: DATA & FITUR
# =========================
def load_positive_data(path: Path) -> pd.DataFrame:
    """Load semua pasangan positif dan bersihkan sekuens."""
    df = pd.read_csv(path)
    required = [
        'pep_id',
        'pep_seq',
        'Uniprot_id',
        'prot_seq',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Kolom hilang di file input: {missing}')

    df = df[required].dropna().drop_duplicates().reset_index(drop=True)
    df = df.rename(columns={'Uniprot_id': 'uniprot_id'})

    df['pep_id'] = df['pep_id'].astype(str)

    # bersihkan sekuens
    before = len(df)
    df['pep_seq'] = df['pep_seq'].apply(clean_sequence)
    df['prot_seq'] = df['prot_seq'].apply(clean_sequence)
    df = df[(df['pep_seq'].str.len() > 0) & (df['prot_seq'].str.len() > 0)].reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f'Removed {removed} pairs with empty/unnatural sequences after cleaning.')

    df['label'] = 1
    print('Positive pairs (used):', len(df))
    return df


def build_feature_dicts(df_pos: pd.DataFrame):
    """Bangun dict fitur peptide (ATC) dan protein (ATC)."""
    unique_pep = df_pos[['pep_id', 'pep_seq']].drop_duplicates().reset_index(drop=True)
    unique_prot = df_pos[['uniprot_id', 'prot_seq']].drop_duplicates().reset_index(drop=True)

    # peptide → ATC
    pep_feature_matrix = seqs_to_atc_features(unique_pep['pep_seq'].tolist())
    if pep_feature_matrix.size == 0:
        raise ValueError('Tidak ada peptide unik yang valid untuk fitur ATC.')

    pep_vecs = {
        row.pep_id: pep_feature_matrix[idx]
        for idx, row in enumerate(unique_pep.itertuples(index=False))
    }

    # protein → ATC
    prot_feature_matrix = seqs_to_atc_features(unique_prot['prot_seq'].tolist())
    if prot_feature_matrix.size == 0:
        raise ValueError('Tidak ada protein unik yang valid untuk fitur ATC.')

    prot_vecs = {
        row.uniprot_id: prot_feature_matrix[idx]
        for idx, row in enumerate(unique_prot.itertuples(index=False))
    }

    pep_dim = pep_feature_matrix.shape[1]
    prot_dim = prot_feature_matrix.shape[1]
    latent_dim = pep_dim + prot_dim

    print('Unique peptides:', len(pep_vecs), 'proteins:', len(prot_vecs))
    print('ATC dim (pep):', pep_dim, '| ATC dim (prot):', prot_dim, '| Latent dim:', latent_dim)

    return pep_vecs, prot_vecs, unique_pep, unique_prot, pep_dim, prot_dim, latent_dim


def build_interaction_vec(pep_id: str,
                          uniprot_id: str,
                          pep_vecs: dict,
                          prot_vecs: dict) -> np.ndarray:
    """Concatenate ATC(pep) + ATC(protein) → vektor laten interaksi."""
    fp = pep_vecs[pep_id]
    vec = prot_vecs[uniprot_id]
    return np.concatenate([fp, vec], axis=0).astype(float)


def build_positive_latents(df_pos: pd.DataFrame,
                           pep_vecs: dict,
                           prot_vecs: dict) -> np.ndarray:
    """Bangun matriks latent untuk semua pasangan positif."""
    latents = np.stack([
        build_interaction_vec(row.pep_id, row.uniprot_id, pep_vecs, prot_vecs)
        for row in df_pos.itertuples()
    ], axis=0)
    print('Positive latent shape:', latents.shape)
    return latents


# =========================
# STEP B: DECOY UNKNOWN SPACE
# =========================
def build_decoy_unknown_pairs(df_pos: pd.DataFrame,
                              pep_vecs: dict,
                              prot_vecs: dict) -> list[tuple[str, str]]:
    """
    Batasi ruang unknown dengan decoy:
    untuk tiap pasangan (pep_pos, protein_pos), ambil DECOY_PER_POS peptide mirip
    (berdasarkan cosine similarity ATC) yang belum berinteraksi dengan protein tsb.
    """
    existing = set((row.pep_id, row.uniprot_id) for row in df_pos.itertuples())
    peps = list(pep_vecs.keys())
    prots = list(prot_vecs.keys())
    n_peps = len(peps)

    # matriks ATC peptide (n_peps x pep_dim)
    pep_matrix = np.stack([pep_vecs[pep_id] for pep_id in peps], axis=0)

    # cosine similarity antar peptide
    print('Computing peptide-peptide cosine similarity for decoy space...')
    dist_mat = pairwise_distances(pep_matrix, metric='cosine')
    sim_mat = 1.0 - dist_mat

    pep_to_idx = {pep_id: idx for idx, pep_id in enumerate(peps)}
    decoy_pairs = set()

    # bangun decoy per pasangan positif
    for row in df_pos.itertuples():
        pep_pos = row.pep_id
        prot = row.uniprot_id
        idx_pos = pep_to_idx[pep_pos]

        sims = sim_mat[idx_pos]
        order = np.argsort(-sims)  # descending (paling mirip dulu)

        count_decoy = 0
        for j in order:
            pep_dec = peps[j]
            if pep_dec == pep_pos:
                continue
            if (pep_dec, prot) in existing:
                continue
            decoy_pairs.add((pep_dec, prot))
            count_decoy += 1
            if count_decoy >= DECOY_PER_POS:
                break

    print('Decoy-based unknown candidate pairs:', len(decoy_pairs))

    # fallback: jika kosong, pakai full cartesian unknown
    if not decoy_pairs:
        print('Decoy set empty, fallback to full cartesian unknown space.')
        decoy_pairs = {
            (pep_id, prot)
            for pep_id in peps
            for prot in prots
            if (pep_id, prot) not in existing
        }
        print('Fallback unknown candidate pairs:', len(decoy_pairs))

    return list(decoy_pairs)


def build_unknown_latents(unknown_pairs: list[tuple[str, str]],
                          pep_vecs: dict,
                          prot_vecs: dict) -> np.ndarray:
    """Bangun matriks laten untuk semua pasangan unknown (decoy space)."""
    latents = np.stack([
        build_interaction_vec(pep_id, prot, pep_vecs, prot_vecs)
        for pep_id, prot in unknown_pairs
    ], axis=0)
    print('Unknown latent shape:', latents.shape)
    return latents


# =========================
# STEP C: GAN UNTUK LATENT SYNTHETIC NEGATIVES
# =========================
class Generator(nn.Module):
    """Generator untuk latent synthetic negatives (vektor laten)."""
    def __init__(self, noise_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Discriminator di latent space (real positif vs syn)."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def hard_negativity_loss(syn_scores: torch.Tensor) -> torch.Tensor:
    """Membuat skor D(syn) ~ 0.5 → latent synthetic negatives yang ambigu/hard."""
    target = 0.5
    return ((syn_scores - target) ** 2).mean()


def closeness_loss(syn_batch: torch.Tensor,
                   real_batch: torch.Tensor,
                   tau: float = 0.8) -> torch.Tensor:
    """Menjaga syn cukup dekat ke manifold positif, tapi tidak terlalu nempel."""
    diff = syn_batch.unsqueeze(1) - real_batch.unsqueeze(0)
    dist = torch.norm(diff, dim=2)
    min_dist, _ = torch.min(dist, dim=1)
    loss = torch.relu(min_dist - tau).mean()
    return loss


def train_gan(pos_latents: np.ndarray, latent_dim: int):
    """Latih GAN di latent space untuk menghasilkan latent synthetic negatives."""
    G = Generator(NOISE_DIM, latent_dim).to(DEVICE)
    D = Discriminator(latent_dim).to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    pos_tensor = torch.tensor(pos_latents, dtype=torch.float32)
    loader_pos = DataLoader(
        TensorDataset(pos_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    print('Start training GAN (latent synthetic negatives)...')
    for epoch in range(EPOCHS):
        for (real_batch,) in loader_pos:
            real_batch = real_batch.to(DEVICE)
            bs = real_batch.size(0)

            # ---- Train Discriminator (real vs syn) ----
            noise = torch.randn(bs, NOISE_DIM, device=DEVICE)
            syn_batch = G(noise).detach()

            real_labels = torch.ones(bs, 1, device=DEVICE)
            syn_labels = torch.zeros(bs, 1, device=DEVICE)

            out_real = D(real_batch)
            out_syn = D(syn_batch)

            loss_D_real = bce(out_real, real_labels)
            loss_D_syn = bce(out_syn, syn_labels)
            loss_D = loss_D_real + loss_D_syn

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---- Train Generator (latent syn) ----
            noise = torch.randn(bs, NOISE_DIM, device=DEVICE)
            syn_batch = G(noise)

            out_syn_G = D(syn_batch)
            loss_hard = hard_negativity_loss(out_syn_G)

            LAMBDA_CLOSE = 0.2
            loss_close = closeness_loss(syn_batch, real_batch, tau=0.8)

            loss_G = loss_hard + LAMBDA_CLOSE * loss_close

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        if (epoch + 1) % 30 == 0:
            print(
                f'Epoch {epoch+1}/{EPOCHS} | '
                f'D_loss: {loss_D.item():.4f} | '
                f'G_loss: {loss_G.item():.4f} | '
                f'hard: {loss_hard.item():.4f} | close: {loss_close.item():.4f}'
            )

    print('GAN training done.')
    return G


def generate_synthetic_latents(G: nn.Module,
                               num_pos: int,
                               max_unknown: int) -> tuple[np.ndarray, int]:
    """Bangkitkan latent synthetic negatives secukupnya untuk dipetakan ke unknown."""
    target_negatives = min(int(num_pos * NEGATIVE_RATIO), max_unknown)
    if target_negatives == 0:
        raise ValueError('target_negatives = 0, periksa NEGATIVE_RATIO atau unknown_space.')

    with torch.no_grad():
        G.eval()
        noise = torch.randn(target_negatives * 3, NOISE_DIM, device=DEVICE)
        syn_latents = G(noise).cpu().numpy()

    print('Synthetic latent shape:', syn_latents.shape)
    print('Target negatives:', target_negatives)
    return syn_latents, target_negatives


# =========================
# STEP D: MAP LATENT SYN → UNKNOWN (MC NN)
# =========================
def map_synthetic_to_unknown_mc_nn(syn_latents: np.ndarray,
                                   unknown_latents: np.ndarray,
                                   unknown_pairs: list[tuple[str, str]],
                                   target_negatives: int) -> list[tuple[str, str]]:
    """
    Mapping latent synthetic negatives → pasangan unknown.
    Pakai Monte Carlo nearest neighbor + medium-hard band [K_MIN, K_MAX).
    """
    chosen_indices = set()
    neg_pairs = []
    n_unknown = len(unknown_pairs)

    for syn_vec in syn_latents:
        if len(neg_pairs) >= target_negatives:
            break

        # index unknown yang belum dipakai
        available_idx = [i for i in range(n_unknown) if i not in chosen_indices]
        if not available_idx:
            break

        # Monte Carlo subset
        m = min(SUBSET_SIZE, len(available_idx))
        subset_idx = rng.choice(available_idx, size=m, replace=False)

        subset_latents = unknown_latents[subset_idx]  # (m, latent_dim)
        d = pairwise_distances(syn_vec[None, :], subset_latents, metric='cosine')[0]
        order = np.argsort(d)  # ascending

        # band medium-hard
        k_min = min(K_MIN, len(order) - 1) if len(order) > 0 else 0
        k_max = min(K_MAX, len(order))
        if k_min >= k_max:
            k_min = 0

        cand_rel = order[k_min:k_max]
        if cand_rel.size == 0:
            cand_rel = order

        chosen_rel = cand_rel[0]
        idx_global = subset_idx[chosen_rel]

        if idx_global in chosen_indices:
            continue

        chosen_indices.add(idx_global)
        neg_pairs.append(unknown_pairs[idx_global])

    print('Total pseudo-negative pairs (latent synthetic, MC NN):', len(neg_pairs))
    return neg_pairs


# =========================
# STEP E: BUILD FINAL DATASET
# =========================
def build_and_save_dataset(df_pos: pd.DataFrame,
                           neg_pairs: list[tuple[str, str]],
                           unique_pep: pd.DataFrame,
                           unique_prot: pd.DataFrame,
                           output_path: Path):
    """Bangun dataframe final (pos+neg) dan simpan ke CSV."""
    neg_df = pd.DataFrame(neg_pairs, columns=['pep_id', 'uniprot_id'])
    neg_df = neg_df.merge(unique_pep, on='pep_id', how='left')
    neg_df = neg_df.merge(unique_prot, on='uniprot_id', how='left')
    neg_df['label'] = 0

    if neg_df.empty:
        raise ValueError(
            'Tidak ada pseudo-negative yang bisa dipilih. '
            'Coba ubah DECOY_PER_POS atau hyperparameter GAN.'
        )

    print('Negative df size:', len(neg_df))

    # seimbangkan jumlah pos dan neg
    if len(df_pos) > len(neg_df):
        df_pos_balanced = df_pos.sample(len(neg_df), random_state=SEED).reset_index(drop=True)
    else:
        df_pos_balanced = df_pos.copy()
    df_pos_balanced['label'] = 1

    df_all = pd.concat([
        df_pos_balanced[['pep_id', 'pep_seq', 'uniprot_id', 'prot_seq', 'label']],
        neg_df[['pep_id', 'pep_seq', 'uniprot_id', 'prot_seq', 'label']]
    ], axis=0).reset_index(drop=True)
    df_all = df_all.rename(columns={'uniprot_id': 'Uniprot_id'})

    print('Final size:', len(df_all))
    print('Label counts:')
    print(df_all['label'].value_counts())
    print('Label proportion:')
    print(df_all['label'].value_counts(normalize=True))

    props = df_all['label'].value_counts(normalize=True).to_dict()
    if not (abs(props.get(1, 0) - 0.5) < 1e-6 and abs(props.get(0, 0) - 0.5) < 1e-6):
        print('Warning: proporsi tidak tepat 0.5/0.5, periksa NEGATIVE_RATIO / decoy settings.')

    df_all.to_csv(output_path, index=False)
    print('Saved:', output_path)


# =========================
# MAIN PIPELINE
# =========================
def main():
    # 1) load positif
    df_pos = load_positive_data(POSITIVE_PATH)

    # 2) fitur peptide/protein
    pep_vecs, prot_vecs, unique_pep, unique_prot, pep_dim, prot_dim, latent_dim = build_feature_dicts(df_pos)

    # 3) latent positif
    pos_latents = build_positive_latents(df_pos, pep_vecs, prot_vecs)

    # 4) unknown decoy space
    unknown_pairs = build_decoy_unknown_pairs(df_pos, pep_vecs, prot_vecs)
    unknown_latents = build_unknown_latents(unknown_pairs, pep_vecs, prot_vecs)

    # 5) train GAN di latent space (LSN)
    G = train_gan(pos_latents, latent_dim)

    # 6) generate latent synthetic negatives
    syn_latents, target_negatives = generate_synthetic_latents(
        G, num_pos=pos_latents.shape[0], max_unknown=len(unknown_pairs)
    )

    # 7) map LSN ke pasangan unknown (MC NN)
    neg_pairs = map_synthetic_to_unknown_mc_nn(
        syn_latents, unknown_latents, unknown_pairs, target_negatives
    )

    # 8) build dan simpan dataset final
    build_and_save_dataset(df_pos, neg_pairs, unique_pep, unique_prot, OUTPUT_PATH)


if __name__ == '__main__':
    main()
