import re
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt

def extract_dataset(model: str, group_name: str) -> str:
    """
    Extract the dataset combination (outdoor/indoor/garage combos) from a model string.
    Primary strategy: strip the known group prefix and read up to '_run'.
    Fallback: regex search for the segment immediately before '_run\\d+'.
    """
    # Try removing the exact group prefix
    prefix = f"{group_name}_"
    if model.startswith(prefix):
        rest = model[len(prefix):]
        if "_run" in rest:
            return rest.split("_run", 1)[0]

    # Fallback regex: capture the chunk right before `_run<digits>`
    m = re.search(r"(.+?)_run\d+", model)
    if m:
        # If the group name is embedded, try removing it from the front
        candidate = m.group(1)
        if candidate.startswith(group_name + "_"):
            candidate = candidate[len(group_name) + 1 :]
        return candidate

    # If all else fails, return the whole model (unlikely)
    return model

def count_datasets_per_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Given a dict {table_name: df}, where each df has a 'model' column,
    return a pivot with index=dataset_combo, columns=table_name, values=count.
    """
    records = []
    for table_name, df in tables.items():
        if "model" not in df.columns:
            raise ValueError(f"DataFrame for '{table_name}' is missing a 'model' column.")
        # Extract dataset combo per row
        datasets = df["model"].apply(lambda m: extract_dataset(str(m), table_name))
        counts = datasets.value_counts().rename("count").reset_index().rename(columns={"index": "dataset"})
        counts["table"] = table_name
        records.append(counts)

    all_counts = pd.concat(records, ignore_index=True)
    pivot = all_counts.pivot_table(index="dataset", columns="table", values="count", fill_value=0, aggfunc="sum")
    # Sort datasets alphabetically for stable plots (optional)
    pivot = pivot.sort_index()
    # Sort columns (tables) alphabetically for consistency (optional)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot

def plot_dataset_counts(pivot: pd.DataFrame, title: str = "Runs per dataset combo by table"):
    """
    Plot a grouped bar chart from the pivot DataFrame produced by count_datasets_per_table().
    Each dataset combo (row) appears on the x-axis, with one bar per table (column).
    """
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_xlabel("Dataset combination")
    ax.set_ylabel("Number of runs")
    ax.set_title(title)
    ax.legend(title="Table")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

