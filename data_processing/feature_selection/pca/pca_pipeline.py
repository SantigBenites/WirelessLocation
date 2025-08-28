from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import ray
from typing import Dict, Any, Tuple, Optional, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKPCA
from configs import PCAConfig
from viz import plot_scree, plot_cumulative, plot_loadings_heatmap, plot_pc_scatter

def _select_feature_columns(df: pd.DataFrame, drop_cols: List[str]) -> List[str]:
    cols = [c for c in df.columns if c not in drop_cols]
    # keep only numeric
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols

def _compute_num_components(explained_ratio: np.ndarray, variance_threshold: float, max_components: Optional[int]) -> int:
    cum = np.cumsum(explained_ratio)
    k = int(np.searchsorted(cum, variance_threshold) + 1)
    if max_components is not None:
        k = min(k, max_components)
    return k

class PCAResult:
    def __init__(self,
                 components: np.ndarray,
                 explained_variance: np.ndarray,
                 explained_ratio: np.ndarray,
                 means: Optional[np.ndarray],
                 scales: Optional[np.ndarray],
                 selected_components: int,
                 feature_names: List[str],
                 scores: Optional[np.ndarray] = None):
        self.components = components
        self.explained_variance = explained_variance
        self.explained_ratio = explained_ratio
        self.means = means
        self.scales = scales
        self.selected_components = selected_components
        self.feature_names = feature_names
        self.scores = scores

def _pca_cpu(X: np.ndarray, cfg: PCAConfig) -> PCAResult:
    pca = SKPCA(n_components=min(X.shape), whiten=cfg.whiten, random_state=cfg.random_state)
    pca.fit(X)
    selected = _compute_num_components(pca.explained_variance_ratio_, cfg.variance_threshold, cfg.max_components)
    scores = pca.transform(X)[:, :selected] if cfg.save_projection else None
    return PCAResult(
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_ratio=pca.explained_variance_ratio_,
        means=None,
        scales=None,
        selected_components=selected,
        scores=scores,
        feature_names=[],
    )

@ray.remote(num_gpus=1)
class GPUPCARunner:
    def __init__(self, cfg_dict: Dict[str, Any]):
        self.cfg = PCAConfig(**cfg_dict)
        try:
            from cuml.decomposition import PCA as cuPCA
            import cupy as cp
            self.cuPCA = cuPCA
            self.cp = cp
            self.ok = True
        except Exception as e:
            self.ok = False
            self.err = str(e)

    def run(self, X_host: np.ndarray) -> Dict[str, Any]:
        if not self.ok:
            raise RuntimeError(f"cuML not available in GPUPCARunner: {getattr(self,'err','unknown')}")
        cp = self.cp
        cuPCA = self.cuPCA
        X = cp.asarray(X_host)
        pca = cuPCA(n_components=min(X.shape), whiten=self.cfg.whiten, random_state=self.cfg.random_state)
        pca.fit(X)
        explained_ratio = pca.explained_variance_ratio_.get()
        explained_variance = pca.explained_variance_.get()
        components = pca.components_.get()
        selected = int(np.searchsorted(np.cumsum(explained_ratio), self.cfg.variance_threshold) + 1)
        if self.cfg.max_components is not None:
            selected = min(selected, self.cfg.max_components)
        scores = pca.transform(X).get()[:, :selected] if self.cfg.save_projection else None
        return {
            "components": components,
            "explained_variance": explained_variance,
            "explained_ratio": explained_ratio,
            "selected": selected,
            "scores": scores,
        }

def run_pca(df: pd.DataFrame, cfg: PCAConfig) -> Tuple[PCAResult, pd.DataFrame, pd.DataFrame]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    # Choose features
    feats = _select_feature_columns(df, cfg.drop_columns)
    X = df[feats].copy()

    # Clean -> impute -> scale
    X = X.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = None
    if cfg.scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_proc = scaler.fit_transform(X_imp)
    else:
        X_proc = X_imp

    # PCA (GPU via cuML if available and requested; else CPU sklearn)
    used_gpu = False
    if cfg.use_gpu_for_pca and cfg.use_cuml_if_available:
        try:
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
            runner = GPUPCARunner.options(name="gpu_pca").remote(cfg.__dict__)
            out = ray.get(runner.run.remote(X_proc))
            used_gpu = True
            pca_res = PCAResult(
                components=out["components"],
                explained_variance=out["explained_variance"],
                explained_ratio=out["explained_ratio"],
                means=scaler.mean_ if scaler is not None else None,
                scales=scaler.scale_ if scaler is not None else None,
                selected_components=out["selected"],
                feature_names=feats,
                scores=out["scores"],
            )
        except Exception as e:
            # Fallback to CPU
            pca_res = _pca_cpu(X_proc, cfg)
            pca_res.feature_names = feats
    else:
        pca_res = _pca_cpu(X_proc, cfg)
        pca_res.feature_names = feats

    # Feature selection via loadings across selected components
    # sklearn components_: shape (n_components, n_features). Loadings = components_.T * sqrt(explained_variance)
    # Use selected components to compute importance
    k = pca_res.selected_components
    comps = pca_res.components[:k, :]
    ev = pca_res.explained_variance[:k]
    # Safe: if ev length is 0, bail
    if k == 0:
        raise RuntimeError("No principal components selected (check data variability).")
    loadings = (comps.T * np.sqrt(ev))  # shape (n_features, k)
    importance = np.sum(np.abs(loadings), axis=1)  # aggregate absolute contribution
    feat_importance = pd.DataFrame({
        "feature": pca_res.feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if cfg.feature_selection_top_k is not None:
        selected_features = feat_importance.head(cfg.feature_selection_top_k)["feature"].tolist()
    else:
        selected_features = feat_importance["feature"].tolist()
    selected_df = pd.DataFrame({"selected_feature": selected_features})

    # Save tables
    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca_res.explained_ratio))],
        "explained_variance": pca_res.explained_variance,
        "explained_variance_ratio": pca_res.explained_ratio
    }).to_csv(os.path.join(cfg.output_dir, "explained_variance.csv"), index=False)

    feat_importance.to_csv(os.path.join(cfg.output_dir, "feature_importance_pca.csv"), index=False)
    selected_df.to_csv(os.path.join(cfg.output_dir, "selected_features.csv"), index=False)

    # Save projection (scores)
    if cfg.save_projection and pca_res.scores is not None:
        scores_df = pd.DataFrame(pca_res.scores, columns=[f"PC{i+1}" for i in range(k)])
        # Keep optional metadata columns (e.g., targets) next to scores
        meta_cols = [c for c in ["location_x", "location_y"] if c in df.columns]
        for c in meta_cols:
            scores_df[c] = df[c].values
        scores_df.to_parquet(os.path.join(cfg.output_dir, "pca_projection.parquet"), index=False)

    # Save summary JSON
    summary = {
        "used_gpu": used_gpu,
        "n_samples": int(X.shape[0]),
        "n_features_in": int(X.shape[1]),
        "n_components_selected": int(k),
        "variance_threshold": float(cfg.variance_threshold),
        "top_k_features": int(cfg.feature_selection_top_k) if cfg.feature_selection_top_k is not None else None,
    }
    with open(os.path.join(cfg.output_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plot_scree(pca_res.explained_ratio, os.path.join(cfg.output_dir, "fig_scree.png"))
    plot_cumulative(pca_res.explained_ratio, os.path.join(cfg.output_dir, "fig_cumulative.png"))
    plot_loadings_heatmap(loadings, pca_res.feature_names, os.path.join(cfg.output_dir, "fig_loadings_heatmap.png"))
    if cfg.save_projection and pca_res.scores is not None:
        plot_pc_scatter(pca_res.scores, os.path.join(cfg.output_dir, "fig_pc1_pc2.png"))

    return pca_res, feat_importance, selected_df
