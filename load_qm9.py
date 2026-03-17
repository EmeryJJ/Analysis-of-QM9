"""
Simple QM9 dataset loader for the Analysis-of-QM9 project.
Loads the QM9 dataset from PyTorch Geometric.

Requirements:
- torch-geometric==2.7.0
- rdkit==2025.3.6

The dataset contains ~130,831 valid molecules (out of 133,885 total SDF entries).
~3,000 molecules with malformed structures are automatically skipped.
"""

from torch_geometric.datasets import QM9


def load_qm9_dataset(root: str = './qm9_data') -> QM9:
    """
    Load the QM9 dataset from PyTorch Geometric.
    
    Parameters
    ----------
    root : str
        Root directory where the dataset will be downloaded/stored
        
    Returns
    -------
    QM9
        Loaded QM9 dataset containing molecular graphs and properties
        
    Notes
    -----
    First run will download ~300MB of data. Subsequent runs load from cache.
    Returns 130,831 valid molecules (malformed entries automatically skipped).
    """
    return QM9(root=root)


if __name__ == '__main__':
    # Quick test
    print("Loading QM9 dataset...")
    dataset = load_qm9_dataset()
    print(f"✓ Successfully loaded {len(dataset)} molecules")
    print(f"  Example: {dataset[0]}")