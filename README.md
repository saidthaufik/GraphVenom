# GraphVenom: a FiLM-conditioned GNN ensemble framework for predicting cancer protein interactions with snake venom-derived peptides

GraphVenom merupakan framework komputasional untuk memprediksi dan memprioritaskan interaksi antara peptida turunan bisa ular dan protein terkait kanker. Graf tingkat residu diproses oleh GAT, GINE, dan GCN dengan pengayaan struktur protein dari AlphaFold serta pengondisian Feature-wise Linear Modulation (FiLM). Prediksi ketiga model digabungkan sebagai ensemble untuk memilih kandidat yang dikarakterisasi melalui docking, dinamika molekuler, dan pengayaan fungsional.

Repositori ini memuat data, kode, model final, dan hasil yang mendukung penelitian GraphVenom.

## Struktur direktori

```text
GraphVenom/
├── README.md
├── 1-data/                        # Data peptida–protein, fitur, split, dan struktur
├── 2-machine_learning_analysis/   # Evaluasi model, GraphVenom, ablasi, dan prediksi
├── 3-docking_analysis/            # Kode, konfigurasi, skor, dan pose docking
├── 4-dynamics_analysis/           # Kurva, energi, dan gambar dinamika molekuler
└── 5-enrichment_analysis/         # Hasil enrichment GO dan KEGG per spesies
```

## Data

Data awal untuk screening meliputi peptida bisa ular dan protein terkait kanker.

| Data | Asal | Jumlah |
|---|---|---|
| Peptida *Trimeresurus albolabris* | Proteomik LC-MS sampel bisa asal Kulon Progo dari Serpentarium Universitas Gadjah Mada | 182 peptida |
| Peptida *Naja sumatrana* | Proteomik LC-MS sampel bisa asal Jambi dari Serpentarium Universitas Gadjah Mada | 829 peptida |
| Protein terkait kanker | Kurasi terintegrasi dari [RCSB PDB](https://www.rcsb.org/), [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/), [UniProt](https://www.uniprot.org/), dan [DrugBank](https://go.drugbank.com/) | 895 protein |

## Model Evaluation

Kode evaluasi berada di `2-machine_learning_analysis/` dengan penomoran model mengikuti artikel. Evaluasi mencakup validasi silang lima lipatan, studi ablasi, serta skenario unseen peptide, unseen protein, dan unseen both.

| Model | Metode | Sumber |
|---|---|---|
| 1 | CAMP | [Lei dkk. (2021)](https://doi.org/10.1038/s41467-021-25772-4) · [Kode](https://github.com/twopin/CAMP) |
| 2 | XGBoost | Pendekatan peptida–protein oleh [Kusuma dkk. (2023)](https://doi.org/10.1016/j.heliyon.2023.e21149) |
| 3 | ProVenTL (ProtT5 + SAE-DNN) | [Adhiva dkk. (2026)](https://doi.org/10.1007/s10822-026-00801-w) · [Kode](https://github.com/TropBRC-BioinfoLab/ProVenTL) |
| 4 | ESM2 + SAE-DNN | Konfigurasi eksploratif GraphVenom dengan embedding [ESM2, Lin dkk. (2023)](https://doi.org/10.1126/science.ade2574) · [Kode ESM](https://github.com/facebookresearch/esm) |
| 5 | GAT berbasis sekuens | Studi GraphVenom |
| 7 | GINE berbasis sekuens | Studi GraphVenom |
| 9 | GCN berbasis sekuens | Studi GraphVenom |

## GraphVenom

Model 6, 8, dan 10 menggunakan graf tingkat residu dengan pengayaan struktur protein AlphaFold dan pengondisian FiLM. Ketiganya dilatih sebagai model tersendiri dan menjadi komponen ensemble GraphVenom.

| Model | Metode | Konfigurasi |
|---|---|---|
| 6 | GraphVenom-GAT | GAT + AlphaFold + FiLM |
| 8 | GraphVenom-GINE | GINE + AlphaFold + FiLM |
| 10 | GraphVenom-GCN | GCN + AlphaFold + FiLM |

Keputusan positif ensemble memerlukan sedikitnya dua suara positif. Seleksi kandidat lanjutan menggunakan konsensus penuh dari ketiga model, kemudian pasangan diperingkat berdasarkan rata-rata probabilitas interaksi untuk memperoleh 100 kandidat teratas per spesies. Checkpoint tersedia di `2-machine_learning_analysis/checkpoints/` dan hasil evaluasi serta screening di `2-machine_learning_analysis/results/`.

## Docking

Folder `3-docking_analysis/` memuat kode dan konfigurasi docking, skor di `results/scores/`, pose terbaik di `results/best_poses/`, serta perbandingan kandidat dengan kontrol di `results/controls/`. Manifest di `results/manifests/` menghubungkan hasil dengan berkas input menggunakan path dari direktori utama `GraphVenom/`.

## Dinamika molekuler

Folder `4-dynamics_analysis/` memuat hasil simulasi 200 ns untuk NOS2–VMEVLK, NOS2–LETHKLASLW (kontrol), PLCG1–KESEFR, dan PLCG1–QPPVPPQRPM (kontrol). Kurva dan estimasi energi MM/PBSA serta MM/GBSA berada di `results/`, sedangkan gambar berada di `figures/`. Informasi sistem dan seri kurva tersedia dalam `results/systems.csv` dan `results/series_manifest.csv`.

## Enrichment

Folder `5-enrichment_analysis/` memuat hasil pengayaan GO Biological Process, Cellular Component, Molecular Function, dan jalur KEGG. Hasil dipisahkan ke dalam `naja_sumatrana/` dan `trimeresurus_albolabris/`.

## Atribusi

Penelitian ini didanai oleh Lembaga Pengelola Dana Pendidikan (LPDP) atas nama Kementerian Pendidikan Tinggi, Sains, dan Teknologi Indonesia dan dikelola melalui Program EQUITY (Nomor Kontrak 4297/B3/DT.03.08/2025 dan 42011/IT3/HK.07.00-4/P/B/2025).

**Penulis:** Said Thaufik Rizaldi, Gilland Fausta Putra Achyar, Jeni Adhiva, Toto Haryanto, Rudi Heryanto, Fajar Sofyantoro, Annisa Maghfira, Donan Satria Yudha, Slamet Raharjo, Yekti Asih Purwestri, Tri Rini Nuringtyas, Ibnu Agus Ariyanto, Melva Louisa, Tamimah Shafwatul Ishlah, dan Wisnu Ananta Kusuma.
