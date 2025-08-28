from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from configs import Config


def compute_mi_scores(X: pd.DataFrame, y: pd.DataFrame, n_neighbors: int) -> pd.Series:
    """Compute MI for each feature vs each target; return mean across targets."""
    feats = X.columns
    mi_all = []
    for t in y.columns:
        mi = mutual_info_regression(X.values, y[t].values, n_neighbors=n_neighbors, random_state=0)
        mi_all.append(pd.Series(mi, index=feats))
    mi_df = pd.concat(mi_all, axis=1)
    return mi_df.mean(axis=1).sort_values(ascending=False)


def lasso_importance(X: pd.DataFrame, y: pd.DataFrame, alpha: float, max_iter: int) -> pd.Series:
    """Fit MultiOutput Lasso (with scaling) and aggregate absolute coefs across outputs."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    base = Lasso(alpha=alpha, max_iter=max_iter)
    model = MultiOutputRegressor(base, n_jobs=None)
    model.fit(Xs, y.values)
    # Collect absolute coefs across outputs and average
    coefs = []
    for est in model.estimators_:
        c = np.abs(est.coef_.ravel())
        coefs.append(c)
    coefs = np.vstack(coefs)  # shape (n_targets, n_features)
    agg = coefs.mean(axis=0)
    return pd.Series(agg, index=X.columns).sort_values(ascending=False)


def rfe_rank(X: pd.DataFrame, y: pd.Series, alpha: float, n_features_to_select: int, step: int) -> pd.Series:
    """Run RFE using a Lasso estimator for a single target; return ranking (1 is selected)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    estimator = Lasso(alpha=alpha, max_iter=5000)
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
    selector.fit(Xs, y.values)
    # sklearn RFE sets ranking_, with 1 as selected (best)
    return pd.Series(selector.ranking_, index=X.columns)


def fold_feature_selection(
    cfg: Config,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Dict:
    sel = cfg.selection

    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    # 1) MI prefilter
    mi_scores = compute_mi_scores(X_tr, y_tr, n_neighbors=sel.mi_neighbors)
    if sel.mi_top_k is not None:
        mi_keep = mi_scores.head(sel.mi_top_k).index.tolist()
    else:
        mi_keep = mi_scores.index.tolist()

    X_tr_mi = X_tr[mi_keep]
    X_va_mi = X_va[mi_keep]

    # 2) Lasso shrinkage
    lasso_scores = lasso_importance(X_tr_mi, y_tr, alpha=sel.lasso_alpha, max_iter=sel.lasso_max_iter)
    if sel.lasso_top_k is not None:
        lasso_keep = lasso_scores.head(sel.lasso_top_k).index.tolist()
    else:
        # non-zero only
        lasso_keep = lasso_scores[lasso_scores > 0].index.tolist()
        if len(lasso_keep) == 0:
            lasso_keep = lasso_scores.index.tolist()

    X_tr_lasso = X_tr_mi[lasso_keep]
    X_va_lasso = X_va_mi[lasso_keep]

    # 3) RFE per target, then average ranks
    ranks: List[pd.Series] = []
    for t in y.columns:
        r = rfe_rank(
            X_tr_lasso,
            y_tr[t],
            alpha=sel.lasso_alpha,
            n_features_to_select=min(sel.rfe_n_features_to_select or X_tr_lasso.shape[1], X_tr_lasso.shape[1]),
            step=sel.rfe_step,
        )
        ranks.append(r)
    rfe_rank_mean = pd.concat(ranks, axis=1).mean(axis=1)

    # Selected after RFE are those with rank == 1 across targets; if none, pick top n by mean rank
    rfe_selected = rfe_rank_mean[rfe_rank_mean <= 1.0].index.tolist()
    if len(rfe_selected) == 0:
        n_final = min(sel.rfe_n_features_to_select or X_tr_lasso.shape[1], X_tr_lasso.shape[1])
        rfe_selected = rfe_rank_mean.sort_values().head(n_final).index.tolist()

    # 4) Simple validation metrics using a final MultiOutput Lasso fit on selected features
    scaler = StandardScaler()
    X_tr_fs = scaler.fit_transform(X_tr_lasso[rfe_selected].values)
    X_va_fs = scaler.transform(X_va_lasso[rfe_selected].values)
    final_model = MultiOutputRegressor(Lasso(alpha=sel.lasso_alpha, max_iter=sel.lasso_max_iter))
    final_model.fit(X_tr_fs, y_tr.values)
    y_hat = final_model.predict(X_va_fs)

    # Metrics per target + macro
    mae_per_t = np.mean(np.abs(y_hat - y_va.values), axis=0)
    r2_per_t = [r2_score(y_va.iloc[:, i], y_hat[:, i]) for i in range(y.shape[1])]

    return {
        "mi_scores": mi_scores.to_dict(),
        "lasso_scores": lasso_scores.to_dict(),
        "rfe_rank_mean": rfe_rank_mean.to_dict(),
        "selected_features": rfe_selected,
        "metrics": {
            "mae_per_target": mae_per_t.tolist(),
            "r2_per_target": r2_per_t,
            "mae_mean": float(np.mean(mae_per_t)),
            "r2_mean": float(np.mean(r2_per_t)),
        },
    }


def aggregate_across_folds(fold_results: List[Dict], feat_all: List[str]) -> Dict:
    # Frequencies
    freq = pd.Series(0, index=feat_all, dtype=float)
    mi_accum = pd.Series(0.0, index=feat_all)
    lasso_accum = pd.Series(0.0, index=feat_all)
    rfe_rank_accum = pd.Series(0.0, index=feat_all)

    for res in fold_results:
        sel = res["selected_features"]
        freq.loc[sel] += 1

        mi = pd.Series(res["mi_scores"])
        lasso = pd.Series(res["lasso_scores"])
        rfe_r = pd.Series(res["rfe_rank_mean"])  # smaller is better

        # Align to full index
        mi_accum = mi_accum.add(mi, fill_value=0)
        lasso_accum = lasso_accum.add(lasso, fill_value=0)
        # For ranks, accumulate; missing ranks treated as worse (max rank + 1)
        rfe_rank_accum = rfe_rank_accum.add(rfe_r.reindex(feat_all).fillna(rfe_r.max() + 1), fill_value=0)

    n_folds = len(fold_results)
    freq /= n_folds

    mi_mean = (mi_accum / n_folds).sort_values(ascending=False)
    lasso_mean = (lasso_accum / n_folds).sort_values(ascending=False)
    rfe_rank_mean = (rfe_rank_accum / n_folds).sort_values(ascending=True)

    return {
        "frequency": freq.sort_values(ascending=False),
        "mi_mean": mi_mean,
        "lasso_mean": lasso_mean,
        "rfe_rank_mean": rfe_rank_mean,
    }