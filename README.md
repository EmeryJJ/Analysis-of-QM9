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

Clone the repository:

```bash
git clone https://github.com/EmeryJJ/Analysis-of-QM9.git
cd Analysis-of-QM9
```
Then simply run the code within the ipynb file. 
