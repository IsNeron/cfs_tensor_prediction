from pathlib import Path
import torch
from torch.utils.data import random_split, TensorDataset
from .data import prepare_data2d, prepare_data3d, prepare_labels

def create_dataset(
    data_path: Path, 
    labels_path: Path,
):
    data = prepare_data3d(data_path)
    labels = prepare_labels(labels_path)

    generator = torch.Generator().manual_seed(42)
    
    dataset = TensorDataset(data, labels)

    return random_split(dataset, [0.6, 0.2, 0.2], generator)
