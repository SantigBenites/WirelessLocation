from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_scree(explained_ratio: Sequence[float], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    xs = np.arange(1, len(explained_ratio) + 1)
    plt.figure(figsize=(7,5))
    plt.bar(xs, explained_ratio)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_cumulative(explained_ratio: Sequence[float], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    cum = np.cumsum(explained_ratio)
    xs = np.arange(1, len(explained_ratio) + 1)
    plt.figure(figsize=(7,5))
    plt.plot(xs, cum, marker="o")
    plt.axhline(0.95, linestyle="--")
    plt.ylim(0, 1.01)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_loadings_heatmap(loadings: np.ndarray, feature_names: Sequence[str], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(max(8, len(feature_names)*0.25), 6))
    plt.imshow(loadings, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Loading")
    plt.yticks(np.arange(len(feature_names)), feature_names)
    plt.xticks(np.arange(loadings.shape[1]), [f"PC{i+1}" for i in range(loadings.shape[1])], rotation=45)
    plt.title("Feature Loadings (selected PCs)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_pc_scatter(scores: np.ndarray, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    if scores.shape[1] < 2:
        return
    plt.figure(figsize=(6,6))
    plt.scatter(scores[:,0], scores[:,1], s=6, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PC1 vs PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
