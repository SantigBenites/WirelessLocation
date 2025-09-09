from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import StandardScaler
from feature_lists import DATASET_TO_FEATURE  # maps database_name -> List[str]


def _features_for_database(database_name: str) -> List[str]:
    """Get the ordered feature list for a database name (strict)."""
    if database_name not in DATASET_TO_FEATURE:
        raise ValueError(
            f"Unknown database '{database_name}'. "
            f"Known databases: {list(DATASET_TO_FEATURE.keys())}"
        )
    feats = DATASET_TO_FEATURE[database_name]
    if not feats or not isinstance(feats, (list, tuple)):
        raise ValueError(f"DATASET_TO_FEATURE['{database_name}'] must be a non-empty list.")
    return list(feats)


class RssiLocationDataset(Dataset):
    """
    Strict dataset:
      - Uses the exact feature list for the given database.
      - Requires all features to be present & numeric.
      - Requires labels 'location_x', 'location_y' to be present & numeric.
      - If use_timestamp=True, requires 'timestamp' to be present & numeric.
      - Skips any record that fails these checks.
      - No default values are used.
    """
    def __init__(
        self,
        records: List[Dict],
        feature_keys: List[str],
        use_timestamp: bool = False,
        feature_scaler: Optional[StandardScaler] = None,
    ):
        if isinstance(records, np.ndarray):
            records = records.tolist()

        self.feature_keys = list(feature_keys)
        self.use_timestamp = use_timestamp

        X_list, y_list = [], []

        for rec in records:
            # Labels
            if "location_x" not in rec or "location_y" not in rec:
                continue
            try:
                yx = float(rec["location_x"])
                yy = float(rec["location_y"])
            except Exception:
                continue

            # Features
            feats = []
            ok = True
            for k in self.feature_keys:
                if k not in rec or rec[k] is None:
                    ok = False
                    break
                try:
                    feats.append(float(rec[k]))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue

            # Timestamp (if required)
            if self.use_timestamp:
                if "timestamp" not in rec or rec["timestamp"] is None:
                    continue
                try:
                    feats.append(float(rec["timestamp"]))
                except Exception:
                    continue

            X_list.append(feats)
            y_list.append([yx, yy])

        if not X_list:
            raise ValueError(
                "No valid records after strict checks. "
                f"Required features: {self.feature_keys}"
                + (" + 'timestamp'" if self.use_timestamp else "")
                + " and labels 'location_x','location_y' must be numeric."
            )

        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)

        # Scale features
        if feature_scaler is None:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        else:
            self.scaler = feature_scaler
            X = self.scaler.transform(X)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_loaders(
    train_recs,
    val_recs,
    database_name: str,
    use_timestamp: bool,
    batch_size: int,
):
    """
    Build train/val DataLoaders using the feature list for `database_name`.
    Feature order is fixed by DATASET_TO_FEATURE[database_name] and shared by both splits.
    """
    feature_keys = _features_for_database(database_name)

    # Fit scaler on train; reuse for val
    tmp_ds   = RssiLocationDataset(train_recs, feature_keys=feature_keys, use_timestamp=use_timestamp)
    train_ds = RssiLocationDataset(train_recs, feature_keys=feature_keys, use_timestamp=use_timestamp, feature_scaler=tmp_ds.scaler)
    val_ds   = RssiLocationDataset(val_recs,   feature_keys=feature_keys, use_timestamp=use_timestamp, feature_scaler=tmp_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_ds
