# The Shape of a Molecule

### Can Graph Structure Alone Predict Quantum Behavior?

## Project Video

https://youtu.be/nyLsod4H1Tc

## Overview

A single density functional theory (DFT) calculation on a small organic molecule takes hours on a university computing cluster, and the QM9 benchmark (130,831 molecules with 19 quantum properties each) represents thousands of CPU-hours. If a cheap graph algorithm running in seconds on a laptop could approximate those numbers, drug screening, organic photovoltaic design, and materials discovery would all benefit. This project asks how much of a molecule's quantum behavior is actually encoded in its graph topology alone, using the **HOMO-LUMO energy gap** as the single regression target. The investigation walks through three escalating rounds of methods: an atom-count baseline, a hand-crafted graph-mining feature stack drawn from the Data Mining course curriculum, and an edge-aware Graph Neural Network (GINEConv). The observed gap between rounds tells us exactly what classical graph mining can and cannot see.

## Main Deliverable

**The main deliverable is [`main_notebook.ipynb`](main_notebook.ipynb).** It contains the full narrative, code, figures, and results end-to-end. The two checkpoint notebooks under [`checkpoints/`](checkpoints/) are earlier development snapshots kept for reference.

## Research Questions

- **Q0: The Composition Floor:** How accurately can the HOMO-LUMO gap be predicted using *only atom counts*, with no structural information at all?
- **Q1: The Graph-Mining Tour:** How much can classical graph-mining features (rings & motifs, Louvain community detection, centrality / PageRank) improve on the atom-count baseline, and do they tell a chemically interpretable story?
- **Interlude:** When Louvain community detection runs on a molecular graph, does it actually rediscover known chemical fragments (functional groups)?
- **Q2: The GINEConv Challenger:** Can a Graph Neural Network trained end-to-end on the molecular graph (atom types + bond types only) substantially outperform the hand-crafted graph-mining baseline?
- **Q3: Case Studies:** Which specific molecules illuminate where each model wins or fails (the aromatic archetype, the conjugation cliff, the GCN's worst miss, and the course-tech upset)?

## Data

- **Dataset:** [QM9 (Quantum Mechanics 9)](http://quantum-machine.org/datasets/): 130,831 stable, neutral, singlet-state organic molecules with up to 9 heavy atoms drawn from {C, N, O, F} (with H filling out valence). Each molecule comes as a graph (atoms = nodes with atomic number and 3D positions, bonds = edges with bond-type labels) plus 19 DFT-computed quantum properties at the B3LYP/6-31G(2df,p) level.
- **Source:** Loaded via PyTorch Geometric's built-in `QM9` class (the loader sits in [`load_qm9.py`](load_qm9.py)). On first run, ~300 MB is automatically downloaded into `qm9_data/`. ~3,000 SDF entries with malformed structures are skipped, leaving 130,831 valid molecules.
- **Target:** HOMO-LUMO gap (target index 4), measured in eV.
- **Preprocessing:** No manual cleaning; PyG handles parsing. A **Bemis-Murcko scaffold split** (80/10/10) is computed once and reused across all models. Molecules are grouped by their Murcko core so that entire scaffold groups land in a single fold, preventing near-duplicate leakage between train and test. A random split is retained as a side-by-side robustness check; all headline numbers use the scaffold split. Hand-crafted graph features (rings, communities, centrality) are extracted once and cached to `qm9_data/graph_features.parquet` (~5-15 minutes the first time).

## Reproducing the Work

This project was developed in **Google Colab** (also runs locally in a fresh Python 3.11+ virtual environment). To reproduce:

1. **Install dependencies** from [`requirements.txt`](requirements.txt):
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *(In Colab, run `!pip install -r requirements.txt` in the first cell — a GPU runtime is recommended for Q2.)*

2. **Open and run [`main_notebook.ipynb`](main_notebook.ipynb) top to bottom.** It is the only thing you need to execute (as long as the `load_qm9.py` file is also in the directory). The notebook proceeds linearly:
   1. Background, 2. Setup, 3. EDA, 4. Q0, 5. Q1 (a–d), 6. Interlude, 7. Q2, 8. Q3, 9. Failed experiments, 10. Takeaways, 11. Honor statement & citations.
   - First run: QM9 downloads (~300 MB) and graph features are extracted and cached.
   - Q2 trains 100 epochs on GPU (~5–20 min) or falls back to a 20k-molecule / 30-epoch CPU run with a printed warning.
   - Trained GCN weights are cached to `qm9_data/gcn_checkpoint.pt`; reruns load the checkpoint instead of retraining.

3. The two `checkpoints/checkpoint*.ipynb` files are earlier snapshots and **do not need to be run** to reproduce the results.

## Key Dependencies

The full pinned list lives in [`requirements.txt`](requirements.txt). The big ones at a glance:

| Package | Version |
| --- | --- |
| python | 3.11+ (developed on 3.13) |
| torch | 2.10.0 |
| torch-geometric | 2.7.0 |
| rdkit | 2025.3.6 |
| networkx | 3.6.1 |
| python-louvain | 0.16 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.1 |
| numpy | 2.4.3 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| fastparquet | 2026.3.0 |
| py3Dmol | 2.5.4 |

## Repository Structure

```
Analysis-of-QM9/
├── main_notebook.ipynb         # Main deliverable — read this
├── load_qm9.py                 # Thin wrapper around PyG's QM9 loader
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # This file
├── checkpoints/                # Earlier development snapshots (reference only)
│   ├── checkpoint1.ipynb
│   └── checkpoint2.ipynb
└── qm9_data/                   # Created on first run (gitignored)
    ├── raw/                    # Downloaded SDF files
    ├── processed/              # Cached PyG processed graphs
    ├── graph_features.parquet  # Cached hand-crafted graph features
    ├── scaffold_split.npz      # Cached Bemis-Murcko split indices
    ├── gcn_checkpoint.pt       # Trained GINEConv weights
    └── *.png                   # Figures saved during the notebook run
```

## Results Summary

On the Bemis-Murcko scaffold test split, the three rounds form a clean hierarchy on HOMO-LUMO gap prediction:

| Model | Features | Test MAE (eV) | Test R² |
| --- | --- | ---: | ---: |
| **Q0** — Ridge | atom counts only | ~0.88 | ~0.36 |
| **Q1** — Ridge | + rings, conjugation, Louvain communities, centrality / PageRank | ~0.56 | ~0.69 |
| **Q2** — GINEConv | learned message passing over atom + bond types | ~0.25 | ~0.93 |

The graph-mining feature stack cuts MAE by ~36% over the atom-count floor, and the edge-aware GNN cuts MAE by another ~55% on top of that (~71% below Q0). Among hand-crafted features, **double-bond count is the single strongest signal** (Spearman ρ = −0.64), with PageRank max/mean (≈ −0.53) competitive with conjugated-bond count. The Interlude finds that Louvain community detection does **not** rediscover functional groups (mean ARI ≈ −0.11 against atom-type labels) because it is element-blind and bond-order-blind. The headline takeaway: **topology matters, but bond order matters more** — the quantum gap is driven by electron delocalization, which is encoded in the *type* of bonds, not just their existence.
