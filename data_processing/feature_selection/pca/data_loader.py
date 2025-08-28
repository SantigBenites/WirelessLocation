from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pymongo import MongoClient
import ray
from configs import MongoConfig

def _as_df(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    # Normalize dtypes: keep numerics + safe casts
    for col in df.columns:
        if col == "_id":
            continue
        if df[col].dtype == "O":
            # Try convert object to numeric
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

@ray.remote
def load_collection_df(mongo_cfg: MongoConfig, collection_name: str) -> pd.DataFrame:
    client = MongoClient(mongo_cfg.uri)
    coll = client[mongo_cfg.db_name][collection_name]
    cursor = coll.find(mongo_cfg.query_filter, batch_size=mongo_cfg.batch_size, no_cursor_timeout=True)
    batch, frames = [], []
    try:
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= mongo_cfg.batch_size:
                frames.append(_as_df(batch))
                batch.clear()
        if batch:
            frames.append(_as_df(batch))
    finally:
        cursor.close()
        client.close()
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["__source_collection__"] = collection_name
    return df

def load_all_collections(mongo_cfg: MongoConfig) -> pd.DataFrame:
    ray.init(address="auto", ignore_reinit_error=True) if ray.is_initialized() is False else None
    # Kick off parallel reads
    tasks = [load_collection_df.options(name=f"load:{name}").remote(mongo_cfg, name)
             for name in mongo_cfg.collections]
    dfs: List[pd.DataFrame] = ray.get(tasks)
    # Merge
    df = pd.concat([d for d in dfs if not d.empty], ignore_index=True) if any(not d.empty for d in dfs) else pd.DataFrame()
    return df
