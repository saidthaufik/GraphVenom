# GraphVenom: a FiLM-conditioned GNN ensemble framework for predicting cancer protein interactions with snake venom-derived peptides

GraphVenom is a computational framework for predicting and prioritizing interactions between snake venom-derived peptides and cancer-associated proteins. Residue-level graphs are processed by GAT, GINE, and GCN models with AlphaFold-derived protein structural features and Feature-wise Linear Modulation (FiLM). Predictions from the three models are combined into an ensemble to select candidates for characterization through docking, molecular dynamics, and functional enrichment analysis.

This repository contains the data, code, final models, and results supporting the GraphVenom study.

![GraphVenom workflow](docs/figures/GraphVenom_workflow_white.png)

*Figure 1. GraphVenom workflow.*

## Directory Structure

```text
GraphVenom/
├── README.md
├── 1-data/                        # Peptide–protein data, features, splits, and structures
├── 2-machine_learning_analysis/   # Model evaluation, GraphVenom, ablation, and prediction
├── 3-docking_analysis/            # Docking code, configurations, scores, and poses
├── 4-dynamics_analysis/           # Molecular dynamics curves, energies, and figures
└── 5-enrichment_analysis/         # GO and KEGG enrichment results by species
```

## Data

Input data for screening consist of snake venom-derived peptides and cancer-associated proteins.

| Data | Source | Count |
|---|---|---|
| *Trimeresurus albolabris* peptides | LC-MS proteomics of venom samples originating from Kulon Progo, obtained from the Universitas Gadjah Mada Serpentarium | 182 peptides |
| *Naja sumatrana* peptides | LC-MS proteomics of venom samples originating from Jambi, obtained from the Universitas Gadjah Mada Serpentarium | 829 peptides |
| Cancer-associated proteins | Integrated curation from [RCSB PDB](https://www.rcsb.org/), [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/), [UniProt](https://www.uniprot.org/), and [DrugBank](https://go.drugbank.com/) | 895 proteins |

## Model Evaluation

Evaluation code is available in `2-machine_learning_analysis/`, with model numbering following the article. Evaluation includes five-fold cross-validation, ablation studies, and unseen peptide, unseen protein, and unseen both scenarios.

| Model | Method | Source |
|---|---|---|
| 1 | CAMP | [Lei et al. (2021)](https://doi.org/10.1038/s41467-021-25772-4) |
| 2 | XGBoost | [Kusuma et al. (2023)](https://doi.org/10.1016/j.heliyon.2023.e21149) |
| 3 | ProVenTL (ProtT5 + SAE-DNN) | [Adhiva et al. (2026)](https://doi.org/10.1007/s10822-026-00801-w) |
| 4 | ESM2 + SAE-DNN | This study |
| 5 | Sequence-based GAT | This study |
| **6** | **GraphVenom-GAT** | This study |
| 7 | Sequence-based GINE | This study |
| **8** | **GraphVenom-GINE** | This study |
| 9 | Sequence-based GCN | This study |
| **10** | **GraphVenom-GCN** | This study |

**Models 6, 8, and 10 are the components of the GraphVenom ensemble.** Each uses residue-level graphs enriched with AlphaFold-derived protein structural features and FiLM conditioning, and is trained independently.

A positive ensemble decision requires at least two positive votes. Subsequent candidate selection requires positive predictions from all three models. These pairs are ranked by mean interaction probability to obtain the top 100 candidates per species. Checkpoints are available in `2-machine_learning_analysis/checkpoints/`, and evaluation and screening results are in `2-machine_learning_analysis/results/`.

## Docking

Candidate and control peptides were docked against the same protein targets using HADDOCK3 ([Giulini et al., 2025](https://doi.org/10.1021/acs.jcim.5c00969)). Interface interactions, including contacting residues, hydrogen bonds, and salt bridges, were visualized using [Maestro 14.9, Schrödinger](https://www.schrodinger.com/platform/products/maestro/).

Data locations within `3-docking_analysis/`:

| Data | Location |
|---|---|
| Docking code | `src/` |
| Docking configurations | `configs/` |
| Docking scores | `results/scores/` |
| Best poses | `results/best_poses/` |
| Candidate–control comparisons | `results/controls/` |
| Mappings between results and input files | `results/manifests/` |

Input file paths in the manifests are relative to the `GraphVenom/` repository root.

## Molecular Dynamics

Molecular dynamics simulations were performed for 200 ns using GROMACS 2021 ([Abraham et al., 2015](https://doi.org/10.1016/j.softx.2015.06.001)) with the CHARMM36 force field ([Best et al., 2012](https://doi.org/10.1021/ct300400x)). MM/PBSA and MM/GBSA binding energies were estimated using gmx_MMPBSA 1.6.5 ([Valdés-Tresanco et al., 2021](https://doi.org/10.1021/acs.jctc.1c00645)).

The analysis covers NOS2–VMEVLK, NOS2–LETHKLASLW (control), PLCG1–KESEFR, and PLCG1–QPPVPPQRPM (control). Data locations within `4-dynamics_analysis/`:

| Data | Location |
|---|---|
| Simulation curves | `results/` |
| Binding energy summary | `results/energy_summary.csv` |
| Analysis figures | `figures/` |
| System information | `results/systems.csv` |
| Curve series mappings | `results/series_manifest.csv` |

## Enrichment

Functional enrichment of target proteins was analyzed using Enrichr ([Evangelista et al., 2023](https://doi.org/10.1093/nar/gkad393)), covering GO Biological Process, Cellular Component, Molecular Function, and KEGG pathways. Networks connecting target proteins to GO terms or KEGG pathways were visualized using Cytoscape 3.10.2 ([Shannon et al., 2003](https://doi.org/10.1101/gr.1239303)).

Enrichment result locations within `5-enrichment_analysis/`:

| Data | Location |
|---|---|
| *Naja sumatrana* enrichment | `naja_sumatrana/` |
| *Trimeresurus albolabris* enrichment | `trimeresurus_albolabris/` |

## Attribution

This research was funded by the Indonesia Endowment Fund for Education (LPDP) on behalf of the Ministry of Higher Education, Science, and Technology of Indonesia and administered through the EQUITY Program (Contract Numbers 4297/B3/DT.03.08/2025 and 42011/IT3/HK.07.00-4/P/B/2025).

**Authors:** Said Thaufik Rizaldi, Gilland Fausta Putra Achyar, Jeni Adhiva, Toto Haryanto, Rudi Heryanto, Fajar Sofyantoro, Annisa Maghfira, Donan Satria Yudha, Slamet Raharjo, Yekti Asih Purwestri, Tri Rini Nuringtyas, Ibnu Agus Ariyanto, Melva Louisa, Tamimah Shafwatul Ishlah, and Wisnu Ananta Kusuma.

## Contact

For questions, discussion, or further information about the research and use of GraphVenom, please contact:

Prof. Dr.Eng. Wisnu Ananta Kusuma, S.T., M.T.

[ananta@apps.ipb.ac.id](mailto:ananta@apps.ipb.ac.id)
