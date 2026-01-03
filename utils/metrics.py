from __future__ import annotations

from typing import Iterable
import numpy as np


def compute_fairness_var(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.var(vals))
