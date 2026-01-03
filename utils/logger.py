from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ExperimentLogger:
    out_dir: str
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)

        self.metrics_path = os.path.join(self.out_dir, "metrics_round.csv")
        self.clients_path = os.path.join(self.out_dir, "clients_round.csv")
        self.meta_path = os.path.join(self.out_dir, "meta.json")

        # ensure parents exist
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.clients_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)

        self._metrics_rows: List[Dict[str, Any]] = []
        self._client_rows: List[Dict[str, Any]] = []

        if self.meta is not None:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def _ensure_parent(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def log_round_metrics(
        self,
        rnd: int,
        strategy: str,
        global_acc: float,
        global_loss: float,
        budget_used: float,
        fairness_var: float,
        # ---- fairness (optional)
        client_acc_var: float = float("nan"),
        worst_client_acc: float = float("nan"),
        # ---- privacy/constraint (optional)
        pool_usage: float = float("nan"),
        num_eps_min: float = float("nan"),
        num_eps_max: float = float("nan"),
        projection_triggered: float = float("nan"),
        # ---- ablation table (optional, usually filled at final round)
        client_acc_mean: float = float("nan"),
        client_acc_p10: float = float("nan"),
        client_acc_p50: float = float("nan"),
        client_acc_p90: float = float("nan"),
        client_acc_worst: float = float("nan"),
    ):
        self._metrics_rows.append(
            dict(
                round=int(rnd),
                strategy=str(strategy),
                global_acc=float(global_acc),
                global_loss=float(global_loss),
                budget_used=float(budget_used),
                fairness_var=float(fairness_var),

                client_acc_var=float(client_acc_var),
                worst_client_acc=float(worst_client_acc),

                pool_usage=float(pool_usage),
                num_eps_min=float(num_eps_min),
                num_eps_max=float(num_eps_max),
                projection_triggered=float(projection_triggered),

                client_acc_mean=float(client_acc_mean),
                client_acc_p10=float(client_acc_p10),
                client_acc_p50=float(client_acc_p50),
                client_acc_p90=float(client_acc_p90),
                client_acc_worst=float(client_acc_worst),
            )
        )
        self._ensure_parent(self.metrics_path)
        pd.DataFrame(self._metrics_rows).to_csv(self.metrics_path, index=False)

    def log_client_rows(self, rows: List[Dict[str, Any]]):
        self._client_rows.extend(rows)
        self._ensure_parent(self.clients_path)
        pd.DataFrame(self._client_rows).to_csv(self.clients_path, index=False)

