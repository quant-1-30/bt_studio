"""
Standalone test for discover_fsm_pattern using synthetic OFI panel data.
Bypasses gRPC data fetching to directly test the motif discovery pipeline.
"""
import sys
sys.path.insert(0, ".")

import polars as pl
import numpy as np
from bt_studio.pipeline.patterns import discover_fsm_pattern


def make_synthetic_panel(n_stocks: int = 300, bars_per_day: int = 48, seed: int = 42) -> pl.DataFrame:
    """
    Create a synthetic panel where lag_0 contains OFI-like daily curves.
    Inject a known motif pattern into ~30% of stocks to ensure discover_fsm_pattern can find it.
    Also includes fwd_z_* columns and z_gap/intra_* columns needed by evaluate_and_build_fsm.

    IMPORTANT: Multiple stocks per day are needed for cross-sectional ranking (.over("day")).
    """
    rng = np.random.default_rng(seed)

    # The motif: a characteristic OFI pattern (e.g., morning sell-off then recovery)
    motif_len = 16  # 16 bars = ~64 min at downsample=4
    motif_pattern = np.array([
        -0.8, -0.6, -0.4, -0.2, 0.0, 0.1, 0.2, 0.3,
         0.4,  0.5,  0.4,  0.3, 0.2, 0.1, 0.0, -0.1
    ], dtype=np.float64)
    # z-normalize the motif
    motif_z = (motif_pattern - motif_pattern.mean()) / (motif_pattern.std() + 1e-8)

    # Generate multiple stocks per day for cross-sectional ranking
    stocks_per_day = 30
    n_days = n_stocks // stocks_per_day

    sids = []
    days = []
    rows = []
    fwd_z_5m = []
    fwd_z_15m = []
    fwd_z_30m = []
    intra_5m = []
    intra_15m = []
    intra_30m = []
    z_gaps = []

    for day_idx in range(n_days):
        day = 20080101 + day_idx
        for stock_idx in range(stocks_per_day):
            # Base: random walk noise
            curve = rng.standard_normal(bars_per_day) * 0.3
            # Accumulate to make it look like OFI
            curve = np.cumsum(curve)
            curve = (curve - curve.mean()) / (curve.std() + 1e-8)  # z-norm

            # 30% of stocks: inject the motif at a random position
            has_motif = rng.random() < 0.3
            if has_motif:
                start = rng.integers(0, bars_per_day - motif_len)
                curve[start:start + motif_len] = motif_z * 0.8 + curve[start:start + motif_len] * 0.2

            rows.append(curve)
            sids.append(f"stock_{stock_idx}".encode())
            days.append(day)

            # Forward returns (z-scored) - stocks with motif should have STRONG positive returns
            # Use large effect size so Mann-Whitney U test can detect the signal
            base_ret = 1.5 if has_motif else -0.5
            fwd_z_5m.append(rng.normal(base_ret, 0.5))
            fwd_z_15m.append(rng.normal(base_ret * 1.2, 0.5))
            fwd_z_30m.append(rng.normal(base_ret * 1.5, 0.5))

            # Intra-day returns
            intra_5m.append(rng.normal(base_ret * 0.5, 0.3))
            intra_15m.append(rng.normal(base_ret * 0.7, 0.3))
            intra_30m.append(rng.normal(base_ret * 1.0, 0.3))

            # z_gap: difference between triggered and baseline
            z_gaps.append(rng.normal(base_ret * 0.5, 0.2))

    # Build panel DataFrame matching expected schema
    df = pl.DataFrame({
        "sid": sids,
        "day": days,
        "lag_0": rows,
        "fwd_z_open_5m": fwd_z_5m,
        "fwd_z_open_15m": fwd_z_15m,
        "fwd_z_open_30m": fwd_z_30m,
        "intra_open_5m": intra_5m,
        "intra_open_15m": intra_15m,
        "intra_open_30m": intra_30m,
        "z_gap": z_gaps,
    }, schema_overrides={"lag_0": pl.List(pl.Float64)})
    return df


def main():
    print("=" * 70)
    print("Testing discover_fsm_pattern with synthetic data")
    print("=" * 70)

    panel_df = make_synthetic_panel(n_stocks=300, bars_per_day=48, seed=42)
    print(f"Panel shape: {panel_df.shape}")
    print(f"Columns: {panel_df.columns}")
    print(f"Unique days: {panel_df['day'].n_unique()}, stocks/day: {panel_df.height // panel_df['day'].n_unique()}")

    # Inspect lag_0 structure
    first_lag = panel_df["lag_0"][0]
    print(f"lag_0[0] type: {type(first_lag)}, len: {len(first_lag) if hasattr(first_lag, '__len__') else 'N/A'}")

    # Config matching tune_train.py defaults
    tune_config = {
        "cross_days": 1,
        "motif_minutes": 64,   # 64 minutes
        "downsample": 4,       # 4x downsample -> m = 16
        "threshold_r": 0.9,    # tight threshold for balanced triggers
    }

    common_config = {
        "dtw_window_frac": 0.1,
        "stats_windows": [1, 2, 3],
        "decay": 5,
        "ranking_window": 20,
        "ranking_ratio": 0.3,
        "exclude_bars": 12,
        "days_since_ipo": 60,
        "top_k_ratio": 0.2,
        "alternative": "greater",
        "win_rate": 0.55,
        "max_points": 20000,
        "topk": 10,
        "eps": 1e-4,
        "T1_rets": {"open_5m": 5, "open_15m": 15, "open_30m": 30},
        "decay_minutes": 15,
        "min_factor_weight": 0.05,
        "regime_filter": {"ma_window": 20},
        "u_pval": 0.2,
        "min_triggers": 30,
        "trigger": 5,
    }

    # Compute derived params
    m = int(tune_config["motif_minutes"] // tune_config["downsample"])
    threshold_d = float(np.sqrt(2 * m * (1.0 - tune_config["threshold_r"])))
    print(f"\nDerived: m={m}, threshold_d={threshold_d:.4f}")
    print(f"Theoretical random distance (r=0): {np.sqrt(2*m):.4f}")

    # Run discover
    panel_lf = panel_df.lazy()
    result = discover_fsm_pattern(panel_lf, tune_config, common_config)

    print(f"\n{'=' * 70}")
    print(f"RESULT: status={result.get('status')}, score={result.get('metrics_score')}")
    if result.get("status") == "failed":
        print(f"REASON: {result.get('reason')}")
    else:
        print(f"Keys: {list(result.keys())}")
        print(f"Trigger count: {result.get('trigger_count')}")
        print(f"U p-value: {result.get('u_pval')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()