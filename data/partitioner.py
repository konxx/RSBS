from __future__ import annotations

from typing import Dict, List

import numpy as np


def dirichlet_partition_indices(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    num_classes: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Dirichlet Non-IID partition (label distribution skew).
    Returns dict: client_id -> indices list

    Typical usage:
      alpha in {0.1, 0.3, 1.0} (smaller => more skew)
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    # For each class, draw proportions for clients from Dir(alpha)
    for c in range(num_classes):
        idxs = class_indices[c]
        if len(idxs) == 0:
            continue

        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # Convert proportions to counts
        counts = (proportions * len(idxs)).astype(int)

        # Fix rounding: ensure sum(counts) == len(idxs)
        diff = len(idxs) - int(np.sum(counts))
        if diff > 0:
            for k in rng.choice(num_clients, size=diff, replace=True):
                counts[k] += 1
        elif diff < 0:
            for k in rng.choice(np.where(counts > 0)[0], size=(-diff), replace=True):
                counts[k] -= 1

        start = 0
        for cid in range(num_clients):
            take = counts[cid]
            if take > 0:
                client_indices[cid].extend(idxs[start : start + take].tolist())
                start += take

    # Shuffle within each client
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

    return client_indices
