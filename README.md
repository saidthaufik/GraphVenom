# GraphVenom: a FiLM-conditioned GNN ensemble framework for predicting cancer protein interactions with snake venom-derived peptides

GraphVenom merupakan framework komputasional untuk memprediksi dan memprioritaskan interaksi antara peptida turunan bisa ular dan protein terkait kanker. Representasi graf diproses oleh GAT, GINE, dan GCN dengan pengayaan struktur AlphaFold serta pengondisian FiLM. Prediksi ketiga model digabungkan sebagai ensemble untuk memilih kandidat yang dianalisis lebih lanjut melalui docking dan dinamika molekuler.

Paket ini memuat data, kode, model final, dan hasil yang mendukung artikel. Seluruh dokumentasi penggunaan disatukan dalam README ini.

## Struktur direktori

```text
GraphVenom/
├── README.md
├── 1-data/                        # Data peptida–protein, fitur, split, dan struktur
├── 2-machine_learning_analysis/   # Model benchmark, GraphVenom, ablasi, dan prediksi
├── 3-docking_analysis/            # Kode, konfigurasi, skor, dan pose docking
├── 4-dynamics_analysis/           # Kurva, energi, dan gambar dinamika molekuler
└── 5-enrichment_analysis/         # Hasil enrichment GO dan KEGG per spesies
```

## Data

Folder `1-data/` berisi data pelatihan dan screening peptida–protein. Data masukan berada di `original/`, hasil pengolahan di `processed/`, fitur model di `features/`, pembagian evaluasi di `splits/`, serta struktur protein dan peptida di `structures/`.

Data peptida screening mencakup *Naja sumatrana* dan *Trimeresurus albolabris*. Dataset model graf terdiri atas 6.991 pasangan: 3.470 positif dan 3.521 kandidat negatif hasil seleksi komputasional.

## Machine learning

Kode berada di `2-machine_learning_analysis/`. Folder `model1/` hingga `model10/` mengikuti penomoran model pada artikel; analisis ablasi berada di `ablation/`.

| Model | Metode | Sumber |
|---|---|---|
| 1 | CAMP | [Lei dkk., 2021](https://doi.org/10.1038/s41467-021-25772-4) · [Kode](https://github.com/twopin/CAMP) |
| 2 | XGBoost | Pendekatan peptida–protein: [Kusuma dkk., 2023](https://doi.org/10.1016/j.heliyon.2023.e21149) |
| 3 | ProVenTL (ProtT5 + SAE-DNN) | [Adhiva dkk., 2026](https://doi.org/10.1007/s10822-026-00801-w) · [Kode](https://github.com/TropBRC-BioinfoLab/ProVenTL) |
| 4 | ESM2 + SAE-DNN | Konfigurasi eksploratif GraphVenom; embedding ESM2: [Lin dkk., 2023](https://doi.org/10.1126/science.ade2574) · [Kode ESM](https://github.com/facebookresearch/esm) |
| 5 / 6 | GAT berbasis sekuens / GraphVenom-GAT | Studi GraphVenom |
| 7 / 8 | GINE berbasis sekuens / GraphVenom-GINE | Studi GraphVenom |
| 9 / 10 | GCN berbasis sekuens / GraphVenom-GCN | Studi GraphVenom |

Sumber komponen: XGBoost oleh [Chen dan Guestrin (2016)](https://arxiv.org/abs/1603.02754), serta ProtT5/ProtTrans oleh [Elnaggar dkk. (2022)](https://doi.org/10.1109/TPAMI.2021.3095381) ([kode ProtTrans](https://github.com/agemagician/ProtTrans)).

### Hyperparameter

Konfigurasi final seluruh model tersedia di `configs/benchmark/`. Tabel berikut merangkum pencarian awal GraphVenom-GAT, GINE, dan GCN: masing-masing 6, 8, dan 8 kombinasi yang dievaluasi pada lima fold. Nilai terpilih juga digunakan pada benchmark final. **Tetap** berarti parameter memakai satu nilai sepanjang pencarian.

| Model | Hyperparameter | Kandidat | Nilai terpilih |
|---|---|---|---|
| 6 — GAT | Learning rate | Tetap | 0,0005 |
| | Graph dropout | 0,15; 0,20 | 0,15 |
| | MLP dropout | Tetap | 0,15 |
| | Weight decay | Tetap | 0,0001 |
| | FiLM scale | 0,05; 0,10; 0,20 | 0,20 |
| 8 — GINE | Learning rate | Tetap | 0,0005 |
| | Graph dropout | 0,15; 0,20 | 0,15 |
| | MLP dropout | 0,15; 0,20 | 0,20 |
| | Weight decay | 0,0001; 0,0003 | 0,0003 |
| | FiLM scale | Tetap | 0,20 |
| 10 — GCN | Learning rate | 0,0003; 0,0005 | 0,0005 |
| | Graph dropout | Tetap | 0,20 |
| | MLP dropout | 0,15; 0,20 | 0,20 |
| | Weight decay | Tetap | 0,0003 |
| | FiLM scale | 0,10; 0,20 | 0,20 |

Batch size 64, empat graph layer, hidden dimension 64, output dimension 256, serta MLP berukuran 256 dan 128 digunakan tetap. GAT memakai empat attention head. Pemilihan kandidat per fold menggunakan validation PR-AUC, dengan ROC-AUC sebagai pembanding saat PR-AUC sama; konfigurasi final mengikuti kandidat yang paling sering menang antar-fold.

Daftar kandidat, hasil per fold, dan ringkasan pemilihan tersedia di `results/tuning/`. Konfigurasi benchmark model lainnya dapat dilihat pada `configs/benchmark/`.

### Penggunaan

Siapkan dependensi dari `environments/`. Untuk baseline, ekstrak fitur `.csv.gz` menjadi `.csv` pada folder yang sama. Checkpoint ensemble Model 6, 8, dan 10 tersedia di `checkpoints/`, sedangkan hasil evaluasi dan screening berada di `results/`.

Jalankan screening dari direktori `2-machine_learning_analysis/`:

```bash
python 2-predict/dotproduct.py
python 2-predict/predict.py --device cuda
python 2-predict/voting.py
```

Seleksi kandidat dilanjutkan melalui `2-predict/ensemble_filtering.ipynb`. Untuk pelatihan, gunakan skrip praproses dan trainer di folder model terkait, misalnya `model6/0-feat/pre.py` dan `model6/1-model/train_sing_es.py`.

## Docking

Folder `3-docking_analysis/` berisi kode dan konfigurasi docking, ringkasan skor di `results/scores/`, pose terbaik di `results/best_poses/`, serta perbandingan kandidat–kontrol di `results/controls/`. Manifest di `results/manifests/` menghubungkan hasil dengan berkas input. Path dalam manifest dihitung dari direktori utama `GraphVenom/`.

## Dinamika molekuler

Folder `4-dynamics_analysis/` memuat hasil analisis sistem NOS2–VMEVLK, NOS2–LETHKLASLW (kontrol), PLCG1–KESEFR, dan PLCG1–QPPVPPQRPM (kontrol). Kurva dan ringkasan energi berada di `results/`, sedangkan gambar berada di `figures/`. Informasi sistem dan seri kurva tersedia dalam `systems.csv` dan `series_manifest.csv` di dalam `results/`.

## Enrichment

Folder `5-enrichment_analysis/` memuat gambar hasil GO Biological Process, Cellular Component, Molecular Function, dan jalur KEGG. Hasil dipisahkan ke dalam `naja_sumatrana/` dan `trimeresurus_albolabris/`.

## Sitasi dan atribusi

Gunakan judul artikel GraphVenom di atas untuk merujuk penelitian ini.

**Penulis:** Said Thaufik Rizaldi, Gilland Fausta Putra Achyar, Jeni Adhiva, Toto Haryanto, Rudi Heryanto, Fajar Sofyantoro, Annisa Maghfira, Donan Satria Yudha, Slamet Raharjo, Yekti Asih Purwestri, Tri Rini Nuringtyas, Ibnu Agus Ariyanto, Melva Louisa, Tamimah Shafwatul Ishlah, dan Wisnu Ananta Kusuma.

Docking menggunakan HADDOCK. Rujukan: [HADDOCK3, Bonvin's Lab (2022)](https://github.com/haddocking/haddock3) dan Dominguez, C., Boelens, R. & Bonvin, A. M. J. J. (2003), *HADDOCK: a protein-protein docking approach based on biochemical or biophysical information*, J Am Chem Soc 125, 1731–1737. Lisensi dan metadata HADDOCK tersedia di `3-docking_analysis/docs/third_party/haddock3/`. Pengembangan HADDOCK didukung Horizon 2020 [BioExcel CoE](https://www.bioexcel.eu), proyek [823830](https://cordis.europa.eu/project/id/823830).
