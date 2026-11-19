from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST


def load_data(root, out, seed, device, smoke=False):
    train = FashionMNIST(root=str(root), train=True, download=True)
    test = FashionMNIST(root=str(root), train=False, download=True)
    images = torch.cat([train.data, test.data]).unsqueeze(1)
    labels = torch.cat([train.targets, test.targets])
    indices = np.arange(len(images))
    train_idx, remaining = train_test_split(indices, test_size=0.2, random_state=seed,
                                           stratify=labels.numpy())
    val_idx, test_idx = train_test_split(remaining, test_size=0.5, random_state=seed,
                                        stratify=labels.numpy()[remaining])
    split_file = Path(out) / 'split_indices.npz'
    if split_file.exists():
        saved = np.load(split_file)
        for key, value in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            if not np.array_equal(saved[key], value):
                raise ValueError('Existing split differs: use a new output directory.')
    else:
        np.savez_compressed(split_file, train=train_idx, val=val_idx, test=test_idx)
    result = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        if smoke:
            idx = idx[:512 if name == 'train' else 128]
        result[name] = TensorDataset(images[idx].float().to(device) / 255,
                                     labels[idx].long().to(device))
    return result

def loader(dataset, batch_size, train=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)
