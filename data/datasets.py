from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class DatasetBundle:
    train: Dataset
    test: Dataset
    num_classes: int


def get_torchvision_dataset(name: str, root: str) -> DatasetBundle:
    name = name.lower().strip()
    if name == "mnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
        return DatasetBundle(train=train, test=test, num_classes=10)

    if name == "cifar10":
        tfm_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        tfm_test = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm_train)
        test = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm_test)
        return DatasetBundle(train=train, test=test, num_classes=10)

    raise ValueError(f"Unsupported dataset: {name}")


def make_dataloaders_for_clients(
    train_dataset: Dataset,
    test_dataset: Dataset,
    client_indices: Dict[int, List[int]],
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    """
    Returns:
      - per-client train loader (subset)
      - global test loader (full)
    """
    client_loaders: Dict[int, DataLoader] = {}
    for cid, idxs in client_indices.items():
        subset = Subset(train_dataset, idxs)
        client_loaders[cid] = DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
        )

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=num_workers)
    return client_loaders, test_loader
