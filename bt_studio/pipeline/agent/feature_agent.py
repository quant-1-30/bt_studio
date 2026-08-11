"""
Feature Agent: orchestrate feature registration, evaluation, and projection.

Usage in tune_train.py::

    agent = FeatureAgent(common_config)
    feature_df = agent.build_features(combined_lf)  # outputs all feature columns
"""

import polars as pl
import numpy as np
from typing import List, Optional, Dict

from bt_studio.pipeline.features import FeatureRegistry, default_registry


class FeatureAgent:
    """Orchestrates multi-feature build, evaluation, and projection.

    Parameters
    ----------
    common_config : dict
        Global config with eps, T1_rets, etc.
    registry : FeatureRegistry, optional
        Custom registry; defaults to built-in ofi + vol.
    projection : str
        "none" (keep all columns), "pca" (project to 1D), "equal" (equal weight sum).
    """

    def __init__(self, common_config: dict, registry: Optional[FeatureRegistry] = None,
                 projection: str = "none"):
        self.common_config = common_config
        self.registry = registry or default_registry()
        self.projection = projection

    def build_features(self, aligned_lf: pl.LazyFrame) -> pl.DataFrame:
        """Run all registered features and merge into a single DataFrame.

        Returns
        -------
        pl.DataFrame with columns: day, sid, bar_idx, open, close, <feature ratios>
        """
        specs = self.registry.list_features()
        if not specs:
            raise ValueError("No features registered in FeatureAgent")

        print(f"  [FeatureAgent] Building {len(specs)} features: {self.registry.names()}")

        merged_df = None
        for spec in specs:
            feat_df = spec.build_fn(aligned_lf, self.common_config).collect()
            output_col = spec.output_col

            if merged_df is None:
                merged_df = feat_df
            else:
                # Join on common keys, add only the new feature column
                merged_df = merged_df.join(
                    feat_df.select(["day", "sid", "bar_idx", output_col]),
                    on=["day", "sid", "bar_idx"],
                    how="inner",
                )

            stats = feat_df.select([
                pl.col(output_col).mean().alias("mean"),
                pl.col(output_col).std().alias("std"),
                pl.col(output_col).is_not_null().sum().alias("non_null"),
            ]).row(0, named=True)
            print(f"    [{spec.name}] mean={stats['mean']:.6f}, std={stats['std']:.6f}, non_null={stats['non_null']}")

        # Apply projection if needed
        if self.projection == "pca" and merged_df is not None:
            merged_df = self._project_pca(merged_df)
        elif self.projection == "equal" and merged_df is not None:
            merged_df = self._project_equal(merged_df)

        return merged_df

    def _project_pca(self, df: pl.DataFrame) -> pl.DataFrame:
        """Project multiple feature columns to a single PCA component -> lag_0."""
        feature_cols = [s.output_col for s in self.registry.list_features() if s.output_col in df.columns]
        if len(feature_cols) <= 1:
            return df

        from sklearn.decomposition import PCA
        mat = df.select(feature_cols).to_numpy()
        mask = np.isfinite(mat).all(axis=1)
        mat_clean = mat[mask]

        if len(mat_clean) < 10:
            return df

        pca = PCA(n_components=1)
        projected = np.full(len(df), np.nan)
        projected[mask] = pca.fit_transform(mat_clean).ravel()

        df = df.with_columns(pl.Series("lag_0_proj", projected))
        print(f"  [FeatureAgent] PCA projection: explained_var={pca.explained_variance_ratio_[0]:.3f}")
        return df

    def _project_equal(self, df: pl.DataFrame) -> pl.DataFrame:
        """Equal-weight sum of all feature columns -> lag_0_proj."""
        feature_cols = [s.output_col for s in self.registry.list_features() if s.output_col in df.columns]
        if len(feature_cols) <= 1:
            return df

        expr = sum(pl.col(c) for c in feature_cols) / len(feature_cols)
        df = df.with_columns(expr.alias("lag_0_proj"))
        print(f"  [FeatureAgent] Equal-weight projection of {len(feature_cols)} features")
        return df

    def get_output_cols(self) -> List[str]:
        """Return the feature column names produced by this agent."""
        return [s.output_col for s in self.registry.list_features()]
