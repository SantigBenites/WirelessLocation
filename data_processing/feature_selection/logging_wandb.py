from typing import Dict, List, Any


def log_to_wandb(enabled: bool, project: str, entity: str | None, run_name: str | None, notes: str | None, mode: str,
                 config: Dict[str, Any], ratios: List[float], cum_ratios: List[float],
                 k: int, feature_importance: Dict[str, float], loadings_table: List[List[float]] | None,
                 loading_headers: List[str] | None, outfile: str | None) -> None:
    if not enabled:
        return
    try:
        import wandb
        if mode:
            import os
            os.environ["WANDB_MODE"] = mode
        run = wandb.init(project=project, entity=entity, name=run_name, notes=notes, config=config, reinit=False)
        wandb.log({
            "selected_components": k,
            "explained_variance_at_k": (cum_ratios[k - 1] if k > 0 else 0.0),
        })
        wandb.log({
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": cum_ratios,
        })
        if feature_importance:
            table = wandb.Table(columns=["feature", "importance"])
            for feat, imp in feature_importance.items():
                table.add_data(feat, imp)
            wandb.log({"feature_importance_topM": table})
        if loadings_table and loading_headers:
            table2 = wandb.Table(columns=loading_headers)
            for row in loadings_table:
                table2.add_data(*row)
            wandb.log({"pca_loadings_topK": table2})
        if outfile:
            art = wandb.Artifact("pca_feature_selection", type="feature-selection")
            art.add_file(outfile)
            wandb.log_artifact(art)
        run.finish()
    except Exception as e:
        print("[W&B] Logging skipped or failed:", e)