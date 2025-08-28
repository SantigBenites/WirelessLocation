from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient

from configs import Config


def _derive_groups_from_timestamp(ts: pd.Series, window_seconds: int) -> pd.Series:
    # ts is float seconds; bin into windows
    return (ts.values // window_seconds).astype(int)


def load_collections_to_df(cfg: Config) -> pd.DataFrame:
    db_cfg = cfg.database
    client = MongoClient(db_cfg.mongo_uri)
    db = client[db_cfg.db_name]

    frames: List[pd.DataFrame] = []
    for col in db_cfg.collections:
        cursor = db[col].find(db_cfg.query_filter, db_cfg.projection)
        if db_cfg.limit:
            cursor = cursor.limit(db_cfg.limit)
        df = pd.DataFrame(list(cursor))
        if df.empty:
            continue
        df["__collection__"] = col
        frames.append(df)

    if not frames:
        raise RuntimeError("No data returned from the specified collections.")

    df_all = pd.concat(frames, ignore_index=True)

    # Ensure numeric where appropriate; coerce errors
    for c in df_all.columns:
        if c == "_id":
            continue
        if pd.api.types.is_numeric_dtype(df_all[c]):
            continue
        df_all[c] = pd.to_numeric(df_all[c], errors="ignore")

    return df_all


def build_xy_groups(cfg: Config, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str]]:
    fs = cfg.feature_space

    # Determine features
    cols = list(df.columns)

    for c in fs.target_cols + fs.drop_cols + [fs.timestamp_col, "__collection__"]:
        if c in cols:
            cols.remove(c)

    if fs.include_cols is not None:
        feat_cols = [c for c in fs.include_cols if c not in fs.target_cols and c not in fs.drop_cols]
    else:
        feat_cols = [c for c in cols if c not in fs.exclude_cols]

    # Targets
    y = df[fs.target_cols].copy()

    # Features
    X = df[feat_cols].copy()

    # Impute simple
    if fs.impute_strategy == "median":
        X = X.fillna(X.median(numeric_only=True))
    elif fs.impute_strategy == "zero":
        X = X.fillna(0)
    else:
        raise ValueError(f"Unknown impute_strategy: {fs.impute_strategy}")

    # Build groups
    if fs.group_col and fs.group_col in df.columns:
        groups = df[fs.group_col].astype("category").cat.codes.values
        group_name = fs.group_col
    else:
        ts = df[fs.timestamp_col].astype(float)
        groups = _derive_groups_from_timestamp(ts, fs.group_window_seconds)
        group_name = f"timestamp//{fs.group_window_seconds}s"

    return X, y, groups, feat_cols