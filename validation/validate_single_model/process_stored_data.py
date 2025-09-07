import pandas as pd
import re
from typing import Any, Dict, List, Tuple, Union, Dict, Optional
import pandas as pd


_RUN_RE = re.compile(r"(.*)_(run\d+)_depth(\d+)_model(\d+)$")

def _parse_key(key: str) -> Dict[str, Any]:
    """
    Parses keys like:
    'CNN_extra_features_outdoor_and_indoor_run0_depth6_model3'
    Returns:
        {
          'key': key,
          'tag': 'CNN_extra_features_outdoor_and_indoor',
          'run': 'run0',
          'depth': 6,
          'model_id': 3
        }
    """
    m = _RUN_RE.match(key)
    if not m:
        # fallback: keep whole key as tag
        return {"key": key, "tag": key, "run": None, "depth": None, "model_id": None}
    prefix, run, depth, model_id = m.groups()
    return {
        "key": key,
        "tag": prefix,
        "run": run,
        "depth": int(depth),
        "model_id": int(model_id),
    }

def _flatten_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens a metrics dict into prefix_* columns.
    Works for 'overall' and per-group blocks.
    """
    out = {}
    for k, v in metrics.items():
        out[f"{prefix}_{k}"] = v
    return out



def _flatten_entry(
    key: str,
    val: Union[
        Tuple[str, str],  # ("ERROR", "msg")
        Tuple[Dict[str, Any], str],  # (metrics_dict, model_path)
        Dict[str, Any],  # sometimes users store just metrics dict
    ],
) -> Dict[str, Any]:
    """
    Converts a single result entry into a flat row.
    Skips "ERROR" rows by returning {} (caller should filter out empties).
    """
    # Handle explicit error tuples
    if isinstance(val, tuple) and len(val) == 2 and val[0] == "ERROR":
        return {}

    row_meta = _parse_key(key)

    metrics_dict = None
    model_path = None

    if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], dict):
        metrics_dict, model_path = val
    elif isinstance(val, dict):
        metrics_dict = val
    else:
        # Unknown format — skip
        return {}

    row = {**row_meta}
    if model_path:
        row["model_path"] = model_path

    # lightweight known top-level fields
    if metrics_dict is not None:
        row["db_name"] = metrics_dict.get("db_name")

        # overall metrics
        overall = metrics_dict.get("overall", {})
        row.update(_flatten_metrics("overall", overall))

        # per-group metrics (if any)
        per_group = metrics_dict.get("per_group", {})
        for group_name, group_metrics in per_group.items():
            row.update(_flatten_metrics(group_name, group_metrics))

    return row

def build_df(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a tidy pandas DataFrame from your results dict.
    Skips entries with errors or unrecognized shapes.
    """
    rows: List[Dict[str, Any]] = []
    for k, v in results.items():
        row = _flatten_entry(k, v)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Try to derive a compact 'family' and 'datasets' out of 'tag'
    # Heuristic: split tag into tokens, take first two tokens as 'family'
    # and the rest as 'datasets' (works well for your naming style).
    def _split_tag(tag: str):
        toks = tag.split("_")
        if len(toks) <= 2:
            return tag, None
        # Often starts with "CNN", keep first 2-3 as family if they look like modifiers
        if toks[0] == "CNN":
            # Up to 3 tokens for family (CNN + maybe 'extra' + 'features' etc.)
            family_end = min(3, len(toks) - 1)
            family = "_".join(toks[:family_end])
            datasets = "_".join(toks[family_end:])
        else:
            family = toks[0]
            datasets = "_".join(toks[1:])
        return family, datasets

    if "tag" in df.columns:
        fam_ds = df["tag"].astype(str).apply(_split_tag)
        df["family"] = fam_ds.apply(lambda x: x[0])
        df["datasets"] = fam_ds.apply(lambda x: x[1])

    # Consistent column ordering: meta first, then metrics
    meta_cols = [
        "key", "family", "datasets", "tag", "run", "depth", "model_id",
        "db_name", "model_path",
    ]
    metric_cols = [c for c in df.columns if c not in meta_cols]
    df = df[ [c for c in meta_cols if c in df.columns] + sorted(metric_cols) ]

    return df

# Fixed set of groups you asked for (order matters for matching)
GROUP_PREFIXES = [
    "CNN_extra_features_no_leak_xy_free",
    "CNN_extra_features_no_leaking",
    "CNN_extra_features",
    "CNN__meters",              # note the double underscore
    "CNN_raw",
    "CNN_second_experiment",
]

def assign_group(tag_or_key: str) -> Optional[str]:
    """
    Assign one of the six canonical groups based on the start of the tag/key.
    Returns None if no group matches.
    """
    for g in GROUP_PREFIXES:
        if tag_or_key.startswith(g):
            return g
    return None

def add_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'group' column derived from 'tag' (fallback to 'key' if needed).
    """
    if df.empty:
        df["group"] = pd.Series(dtype="object")
        return df

    src = df["tag"] if "tag" in df.columns else df["key"]
    df = df.copy()
    df["group"] = src.astype(str).apply(assign_group)
    return df

def split_by_group(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict: {group_name: dataframe_filtered}
    Only includes groups that appear in df.
    """
    df = add_group_column(df)
    out: Dict[str, pd.DataFrame] = {}
    for g in GROUP_PREFIXES:
        gdf = df[df["group"] == g].copy()
        if not gdf.empty:
            out[g] = gdf.reset_index(drop=True)
    return out

def save_groups_to_csv(group_dfs: Dict[str, pd.DataFrame], prefix: str = "models_") -> None:
    """
    Saves each group's DataFrame to CSV using filenames like: models_<group>.csv
    """
    for g, gdf in group_dfs.items():
        safe = g.replace("/", "_").replace("\\", "_")
        gdf.to_csv(f"{prefix}{safe}.csv", index=False)

def top_n_per_group(
    group_dfs: Dict[str, pd.DataFrame],
    n: int = 10,
    metric: str = "overall_mae_dist",
    ascending: bool = True,
    drop_na: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Returns {group: topN_df} for each group, ranked by the given metric.
    """
    out: Dict[str, pd.DataFrame] = {}
    for g, gdf in group_dfs.items():
        if metric not in gdf.columns:
            # skip quietly if the metric isn't present for this group
            continue
        dfx = gdf if not drop_na else gdf[gdf[metric].notna()]
        out[g] = dfx.sort_values(metric, ascending=ascending).head(n).reset_index(drop=True)
    return out

