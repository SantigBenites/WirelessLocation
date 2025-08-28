# PCA Feature Selection (Ray, Multi‑GPU Optional)

This project loads *all* MongoDB collections in your dataset, merges them into a single DataFrame, and runs a PCA-based feature selection and dimensionality reduction. It uses **Ray** to parallelize data loading and optionally to run PCA on a GPU (via RAPIDS cuML), falling back to CPU (scikit‑learn) if GPU/cuml aren't available.

## What you get
- Parallel Mongo reads across all listed collections.
- Clean + scale features (median impute, standardize).
- PCA on the merged dataset (across all collections).
- PCA-based **feature selection** using aggregated absolute loadings.
- Output **tables** (CSVs) and **graphs** (PNGs): scree, cumulative variance, loadings heatmap, PC1–PC2 scatter.
- All configuration lives in `configs.py` as `@dataclass`es (no CLI args).

## Quick start
1. **Install dependencies** (CPU-only baseline):
   ```bash
   pip install -r requirements.txt
   ```
   For **GPU with cuML** (optional), install RAPIDS matching your CUDA version (see RAPIDS docs). Then set `use_cuml_if_available=True` in `PCAConfig` (default is already True).

2. **Edit configs** in `configs.py` to point to your MongoDB:
   - `MongoConfig.uri`
   - `MongoConfig.db_name`
   - The `collections` list already includes all of yours.
   - (Optional) set `sample` for quicker test runs.

3. **Run**:
   ```bash
   python run_pca.py
   ```

   Ray will auto‑init locally. To connect to an external cluster, set `PCAConfig.ray_address="auto"` and start Ray separately.

4. **Results**:
   See `outputs/pca_feature_selection/` for CSVs and PNGs:
   - `explained_variance.csv`
   - `feature_importance_pca.csv`
   - `selected_features.csv`
   - `pca_projection.parquet` (PC scores)
   - `fig_scree.png`, `fig_cumulative.png`, `fig_loadings_heatmap.png`, `fig_pc1_pc2.png`
   - `dataset_summary.json`

## Notes
- We treat all collections as a **single dataset**.
- We drop `["_id", "timestamp", "location_x", "location_y"]` by default from features (edit in `PCAConfig.drop_columns`).
- Feature selection ranks original features by the sum of absolute loadings across the principal components needed to hit `variance_threshold` (default 95%).
- No command-line arguments are used—everything is in the dataclasses.
