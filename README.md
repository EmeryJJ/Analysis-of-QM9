# QM9 Quantum Property Prediction & Graph Mining Exploration

## Overview

This repository contains exploratory data analysis (EDA), graph mining basics, and initial insights on the **QM9** dataset — a benchmark for quantum machine learning and molecular property prediction.

QM9 consists of ~130,831 small organic molecules (up to 9 heavy atoms: C, H, O, N, F), each represented as a graph (atoms as nodes, bonds as edges) with 3D coordinates and 19 quantum mechanical properties computed via DFT (B3LYP/6-31G(2df,p)).

**Course alignment** (Data Mining course): Graph mining techniques (centrality, community detection, subgraph patterns, PageRank).  
**Beyond-course extension**: Graph Neural Networks (GNNs) for supervised regression of quantum properties (e.g., HOMO/LUMO gap, dipole moment).

This project is part of my coursework exploring the intersection of **AI/ML**, **Quantum Computing**, and **graph-based data mining**.

Key goals:
- Understand molecular graph structure and distributions
- Identify patterns/challenges for graph algorithms
- Motivate advanced GNN approaches for quantum property prediction

---

## Dataset

**Name**: QM9 (Quantum Mechanics 9)  
**Source**: PyTorch Geometric built-in loader (original: http://quantum-machine.org/datasets/)  
**Size**: 130,831 molecules  
**Structure**: Graph objects with:
- Nodes: Atomic numbers (H=1, C=6, N=7, O=8, F=9)
- Edges: Bond types (single, double, triple, aromatic)
- Node features: Atomic number, 3D positions
- Targets: 19 regression properties (focus on 12 standard: mu, alpha, HOMO, LUMO, gap, etc.)


## Installation & Setup

To reproduce this analysis on a fresh clone:

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook QM9_Analysis.ipynb
```

Or if using VS Code:
- Open `QM9_Analysis.ipynb` in VS Code
- Select the Python interpreter from `.venv/bin/python`
- Run all cells

## What the Notebook Does

1. **Cell 1**: Loads the QM9 dataset (~130K molecules) via `load_qm9.py`
2. **Cell 2**: Computes basic statistics (atom counts, property distributions)
3. **Cell 3**: Creates distribution plots for atomic and quantum properties
4. **Cell 4**: Provides molecule visualization functions (2D, 3D, graph)
5. **Cell 5**: Shows data quality checks and bias considerations

## File Structure

```
.
├── QM9_Analysis.ipynb          # Main notebook
├── load_qm9.py                 # Simple dataset loader
├── requirements.txt            # Python dependencies
└── qm9_data/                   # Dataset directory (created on first run)
    ├── raw/                    # SDF files
    └── processed/              # Cached processed graphs
```

## Troubleshooting

If you encounter import errors, ensure:
1. The virtual environment is activated
2. All packages in `requirements.txt` are installed
3. You're running Python 3.10+

The dataset will be automatically downloaded on first run (~300MB).
