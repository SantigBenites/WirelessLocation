#!/usr/bin/env python3
"""
MLP to predict (location_x, location_y) from RSSI features — reads data directly from MongoDB.

Features
--------
- No CLI arguments.
- Connects to MongoDB and loads training records.
- Supports TWO data shapes:
  1) Documents already contain `friend1_rssi`/`freind1_rssi`, `friend2_rssi`/`freind2_rssi`, `friend3_rssi`/`freind3_rssi`.
  2) Documents have a scans array (with BSSID/SSID + RSSI). We map three chosen anchors (by BSSID or SSID) to friend1/2/3.
- Accepts multiple possible paths for coordinates (e.g., `location_x` or `metadata.x`).
- Standardizes inputs, trains an MLP with early stopping, then prints a prediction for one sample.

Setup
-----
- `pip install pymongo torch`
- Fill the CONFIG section below for your DB and (if needed) your anchor identifiers.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------ CONFIG ------------------------------ #
USE_TIMESTAMP = False  # set True to include a timestamp feature if available

# Mongo connection
MONGO_URI: str = "mongodb://localhost:28910/" 
MONGO_DB: str = "wifi_fingerprinting_data_meters"            # change to your DB
MONGO_COLL: str = "equilatero_grande_garage"       # change to your collection
MONGO_QUERY: Dict[str, Any] = {}                   # e.g., {"metadata.triangle_shape": "T1"}
MONGO_PROJECTION: Optional[Dict[str, int]] = None  # or a dict to limit fields
MONGO_LIMIT: Optional[int] = None                  # e.g., 5000

# Coordinate field options (first existing path wins)
X_FIELD_OPTIONS = ["metadata.x", "location_x", "x"]
Y_FIELD_OPTIONS = ["metadata.y", "location_y", "y"]
TIMESTAMP_FIELD_OPTIONS = ["timestamp", "metadata.timestamp"]

# OPTION A: If your docs already have explicit RSSI fields per friend
RSSI_FIELD_OPTIONS = [
    ("friend1_rssi", "freind1_rssi"),
    ("friend2_rssi", "freind2_rssi"),
    ("friend3_rssi", "freind3_rssi"),
]

# OPTION B: If your docs store RSSI in a scans array, configure here
SCAN_ARRAY_PATH = "scan_results"   # e.g., "scan_results" or "scans" (set to your schema)
# Provide either BSSIDs or SSIDs for the three anchors you want to use
SCAN_MATCH_BY = "bssid"            # "bssid" or "ssid"
ANCHORS = {
    "friend1": "aa:bb:cc:dd:ee:ff",  # <- put real BSSID or SSID
    "friend2": "11:22:33:44:55:66",
    "friend3": "77:88:99:aa:bb:cc",
}
# keys inside each scan entry that might hold values
SCAN_RSSI_KEYS = ["RSSI", "rssi", "signal", "signal_level"]
SCAN_BSSID_KEYS = ["BSSID", "bssid"]
SCAN_SSID_KEYS  = ["SSID", "ssid"]

# Training hyperparams
SEED = 42
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
PATIENCE = 20
HIDDEN = [128, 128, 64]

# ------------------------------ Utils ------------------------------ #
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

# ------------------------------ Data Ingest (Mongo) ------------------------------ #

def _first_existing(doc: Dict, paths: List[str]) -> Optional[float]:
    for p in paths:
        v = get_in(doc, p, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                continue
    return None

def _extract_rssi_fields(doc: Dict) -> Optional[List[float]]:
    feats: List[float] = []
    for key_pair in RSSI_FIELD_OPTIONS:
        val = None
        for k in key_pair:
            if k in doc and doc[k] is not None:
                try:
                    val = float(doc[k])
                except Exception:
                    val = None
                if val is not None:
                    break
        if val is None:
            return None
        feats.append(val)
    return feats

def _extract_from_scans(doc: Dict) -> Optional[List[float]]:
    scans = get_in(doc, SCAN_ARRAY_PATH, [])
    if not isinstance(scans, list) or not scans:
        return None

    def get_field(entry: Dict, keys: List[str]) -> Optional[Any]:
        for k in keys:
            if k in entry:
                return entry[k]
        return None

    def match_entry(entry: Dict, target: str) -> bool:
        if SCAN_MATCH_BY.lower() == "bssid":
            b = get_field(entry, SCAN_BSSID_KEYS)
            return isinstance(b, str) and b.lower() == target.lower()
        else:
            s = get_field(entry, SCAN_SSID_KEYS)
            return isinstance(s, str) and s == target

    def extract_best(target: str) -> Optional[float]:
        best: Optional[float] = None
        for e in scans:
            if not isinstance(e, dict):
                continue
            if match_entry(e, target):
                r = get_field(e, SCAN_RSSI_KEYS)
                try:
                    r = float(r)
                except Exception:
                    r = None
                if r is None:
                    continue
                # Choose strongest (max, since RSSI is negative dBm)
                if best is None or r > best:
                    best = r
        return best

    f1 = extract_best(ANCHORS["friend1"]) if ANCHORS.get("friend1") else None
    f2 = extract_best(ANCHORS["friend2"]) if ANCHORS.get("friend2") else None
    f3 = extract_best(ANCHORS["friend3"]) if ANCHORS.get("friend3") else None

    if None in (f1, f2, f3):
        return None
    return [f1, f2, f3]


def load_records_from_mongo() -> List[Dict]:
    try:
        from pymongo import MongoClient  # type: ignore
    except Exception as e:
        raise RuntimeError("pymongo is not installed. Run: pip install pymongo") from e

    client = MongoClient(MONGO_URI)
    coll = client[MONGO_DB][MONGO_COLL]

    cursor = coll.find(MONGO_QUERY, MONGO_PROJECTION or {})
    if MONGO_LIMIT:
        cursor = cursor.limit(MONGO_LIMIT)

    records: List[Dict] = []
    for doc in cursor:
        # Coordinates
        x = _first_existing(doc, X_FIELD_OPTIONS)
        y = _first_existing(doc, Y_FIELD_OPTIONS)
        if x is None or y is None:
            continue

        # Features (prefer explicit fields; otherwise try scans)
        feats = _extract_rssi_fields(doc)
        if feats is None:
            feats = _extract_from_scans(doc)
        if feats is None:
            continue

        rec: Dict[str, float] = {
            "location_x": float(x),
            "location_y": float(y),
            # Use the misspelled keys to match your sample; the Dataset also accepts the correct spelling.
            "freind1_rssi": float(feats[0]),
            "freind2_rssi": float(feats[1]),
            "freind3_rssi": float(feats[2]),
        }
        if USE_TIMESTAMP:
            ts = _first_existing(doc, TIMESTAMP_FIELD_OPTIONS)
            if ts is not None:
                rec["timestamp"] = float(ts)
        records.append(rec)

    client.close()
    if not records:
        print("[WARN] No records loaded from Mongo — check CONFIG (DB names, field paths, anchors, scan array path).")
    else:
        print(f"Loaded {len(records)} records from MongoDB.")
    return records

# ------------------------------ Dataset & Model ------------------------------ #
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

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            last = h
        layers.append(nn.Linear(last, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------------ Training & Inference ------------------------------ #

def split_train_val(records: List[Dict], val_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    idxs = list(range(len(records)))
    random.shuffle(idxs)
    cut = int(len(idxs) * (1 - val_ratio))
    train_idx = set(idxs[:cut])
    train, val = [], []
    for i, r in enumerate(records):
        (train if i in train_idx else val).append(r)
    return train, val

@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(nn.functional.mse_loss(pred, target)).item()


def fit_mlp(
    records: List[Dict],
    hidden: List[int] = HIDDEN,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    dropout: float = DROPOUT,
    patience: int = PATIENCE,
    use_timestamp: bool = USE_TIMESTAMP,
):
    if len(records) < 4:
        raise ValueError("Need at least a handful of samples to train. Provide more records.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_recs, val_recs = split_train_val(records, 0.2)

    tmp_ds = RssiLocationDataset(train_recs, use_timestamp=use_timestamp)
    train_ds = RssiLocationDataset(train_recs, use_timestamp=use_timestamp, feature_scaler=tmp_ds.scaler)
    val_ds   = RssiLocationDataset(val_recs,   use_timestamp=use_timestamp, feature_scaler=tmp_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    in_dim = train_ds.X.shape[1]
    model = MLPRegressor(in_dim, hidden, dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=7, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * Xb.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        preds, targs = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                p = model(Xb)
                val_loss += criterion(p, yb).item() * Xb.size(0)
                preds.append(p.cpu())
                targs.append(yb.cpu())
        train_loss = total / len(train_ds)
        val_loss = val_loss / len(val_ds)
        preds = torch.cat(preds, dim=0)
        targs = torch.cat(targs, dim=0)
        val_rmse = rmse(preds, targs)
        sched.step(val_loss)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_RMSE={val_rmse:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_ds.scaler, val_rmse


def _extract_feats(sample: Dict, use_timestamp: bool) -> List[float]:
    feats = []
    for key_pair in [("friend1_rssi", "freind1_rssi"), ("friend2_rssi", "freind2_rssi"), ("friend3_rssi", "freind3_rssi")]:
        val = None
        for k in key_pair:
            if k in sample and sample[k] is not None:
                val = float(sample[k])
                break
        if val is None:
            raise ValueError(f"Missing RSSI for keys {key_pair}.")
        feats.append(val)
    if use_timestamp:
        feats.append(float(sample.get("timestamp", 0.0)))
    return feats

@torch.no_grad()
def predict_xy(model: nn.Module, scaler: StandardScaler, sample: Dict, use_timestamp: bool = USE_TIMESTAMP) -> Tuple[float, float]:
    device = next(model.parameters()).device
    feats = np.asarray([_extract_feats(sample, use_timestamp)], dtype=np.float32)
    X = scaler.transform(feats)
    pred = model(torch.from_numpy(X).to(device)).cpu().numpy()[0]
    return float(pred[0]), float(pred[1])

# ------------------------------ Main ------------------------------ #
if __name__ == "__main__":
    # 1) Load records from MongoDB
    records = load_records_from_mongo()


    print(f"Training on {len(records)} samples…")
    model, scaler, val_rmse = fit_mlp(records, epochs=EPOCHS, use_timestamp=USE_TIMESTAMP)
    print(f"Validation RMSE: {val_rmse:.4f}")

    # 2) Predict for one sample (replace with a real one from your collection)
    sample = {"freind1_rssi": -68, "freind2_rssi": -82, "freind3_rssi": -85}
    x, y = predict_xy(model, scaler, sample, use_timestamp=USE_TIMESTAMP)
    print({"pred_location_x": x, "pred_location_y": y})
