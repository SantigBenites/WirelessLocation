from typing import Iterable, List, Set, Tuple
import math
from pymongo import MongoClient
import torch


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isinf(x) or math.isnan(x)))


def discover_numeric_feature_keys(client: MongoClient, db_name: str, collections: List[str],
                                  exclude_keys: Set[str], sample_per_collection: int = 200) -> List[str]:
    db = client[db_name]
    intersection = None

    for cname in collections:
        col = db[cname]
        keys_here: Set[str] = set()
        cursor = col.find({}, projection=None, batch_size=sample_per_collection).limit(sample_per_collection)
        for doc in cursor:
            for k, v in doc.items():
                if k in exclude_keys:
                    continue
                if is_number(v):
                    keys_here.add(k)
        if intersection is None:
            intersection = keys_here
        else:
            intersection &= keys_here

    feat_keys = sorted([k for k in (intersection or set()) if k not in exclude_keys])
    return feat_keys


def iter_batches_from_collection(col, feature_keys: List[str], batch_size: int = 8192) -> Iterable[torch.Tensor]:
    D = len(feature_keys)
    buf: List[List[float]] = []
    projection = {k: 1 for k in feature_keys}
    cursor = col.find({}, projection=projection, batch_size=batch_size)

    for doc in cursor:
        row = []
        ok = True
        for k in feature_keys:
            v = doc.get(k, None)
            if not is_number(v):
                ok = False
                break
            row.append(float(v))
        if not ok:
            continue
        buf.append(row)
        if len(buf) >= batch_size:
            yield torch.tensor(buf, dtype=torch.float32)
            buf.clear()

    if buf:
        yield torch.tensor(buf, dtype=torch.float32)
        buf.clear()


def accumulate_statistics(client: MongoClient, db_name: str, collections: List[str],
                          feature_keys: List[str], device: torch.device, rank: int, world_size: int,
                          batch_size: int = 8192) -> Tuple[torch.Tensor, torch.Tensor, int]:
    D = len(feature_keys)
    local_sum = torch.zeros(D, dtype=torch.float64, device=device)
    local_XtX = torch.zeros(D, D, dtype=torch.float64, device=device)
    local_count = torch.tensor([0], dtype=torch.int64, device=device)

    db = client[db_name]
    my_collections = [c for i, c in enumerate(collections) if (i % max(world_size, 1)) == rank]

    for cname in my_collections:
        col = db[cname]
        for X_cpu in iter_batches_from_collection(col, feature_keys, batch_size=batch_size):
            X = X_cpu.to(device=device, dtype=torch.float32)
            Xd = X.double()
            local_sum += Xd.sum(dim=0)
            local_XtX += Xd.T @ Xd
            local_count += torch.tensor([X.shape[0]], dtype=torch.int64, device=device)

    return local_sum, local_XtX, int(local_count.item())