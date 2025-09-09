from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
from functools import lru_cache

def get_in(d: Dict, path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

@dataclass
class StandardScaler:
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StandardScaler":
        s = cls()
        s.mean_ = np.asarray(d["mean"], dtype=np.float32)
        s.std_  = np.asarray(d["std"], dtype=np.float32)
        return s

@lru_cache(maxsize=None)
def resolve_scale(database_name: str) -> float:
    """
    Returns the 'scale' (meters per unit) for a given database by reading runs.py.
    If not found, returns 1.0.
    Expected runs.py structure:
        runs = [
          {"database": "...", "scale": 32, ...},
          ...
        ]
    """
    try:
        from runs import runs  # list[dict]
    except Exception:
        return 1.0

    try:
        for r in runs:
            if r.get("database") == database_name and "scale" in r:
                return float(r["scale"])
    except Exception:
        pass
    return 1.0
