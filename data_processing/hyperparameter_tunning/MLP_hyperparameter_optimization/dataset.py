from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_in, StandardScaler

class RssiLocationDataset(Dataset):
    def __init__(self, records: List[Dict], use_timestamp: bool = False, feature_scaler: Optional[StandardScaler] = None):
        X_list, y_list = [], []
        for rec in records:
            feats = []
            for key_pair in [("friend1_rssi", "freind1_rssi"), ("friend2_rssi", "freind2_rssi"), ("friend3_rssi", "freind3_rssi")]:
                val = None
                for k in key_pair:
                    if k in rec and rec[k] is not None:
                        try:
                            val = float(rec[k])
                        except Exception:
                            val = None
                        break
                if val is None:
                    feats = []
                    break
                feats.append(val)
            if not feats:
                continue
            if use_timestamp:
                feats.append(float(rec.get("timestamp", 0.0)))
            if "location_x" not in rec or "location_y" not in rec:
                continue
            X_list.append(feats)
            y_list.append([float(rec["location_x"]), float(rec["location_y"])])

        if not X_list:
            raise ValueError("No valid records. Check your keys and input data.")
        X = np.asarray(X_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)
        if feature_scaler is None:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        else:
            self.scaler = feature_scaler
            X = self.scaler.transform(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_loaders(train_recs, val_recs, use_timestamp: bool, batch_size: int):
    tmp_ds = RssiLocationDataset(train_recs, use_timestamp=use_timestamp)
    train_ds = RssiLocationDataset(train_recs, use_timestamp=use_timestamp, feature_scaler=tmp_ds.scaler)
    val_ds   = RssiLocationDataset(val_recs,   use_timestamp=use_timestamp, feature_scaler=tmp_ds.scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_ds
