import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ray.tune import ExperimentAnalysis


# ============================================================================
# 1. Window Normalization
# ============================================================================

def load_all_wfo_experiments(base_ray_dir="/tmp/ray_results", target="metrics_score") -> pd.DataFrame:
    """
        MAD Normalize 
    """
    exp_folders = sorted(
        [os.path.join(base_ray_dir, d) for d in os.listdir(base_ray_dir) if d.startswith("fsm_hpo_")]
    )
    if not exp_folders:
        raise FileNotFoundError(f"{base_ray_dir} Not found fsm_hpo_* ")

    all_trials = []

    for folder in exp_folders:
        window_id = os.path.basename(folder).replace("fsm_hpo_", "")
        df_win = None
        
        try:
            analysis = ExperimentAnalysis(folder)
            df_win = analysis.dataframe()
        except Exception:
            pass

        if df_win is not None and len(df_win) > 0 and target in df_win.columns:
            df_win["wfo_window"] = window_id
            
            valid_mask = np.isfinite(df_win[target]) & (df_win[target] > -9999.0)
            if valid_mask.sum() > 3:
                scores = df_win.loc[valid_mask, target]
                med = np.median(scores)
                mad = np.median(np.abs(scores - med))
                scale = 1.4826 * mad + 1e-6
                
                df_win.loc[valid_mask, "window_score_z"] = (scores - med) / scale
                all_trials.append(df_win[valid_mask])

    if not all_trials:
        raise ValueError("Not Effective WFO")

    master_df = pd.concat(all_trials, ignore_index=True)
    return master_df


# ============================================================================
# 2. Parameter Drift Index
# ============================================================================

def compute_wfo_parameter_drift(master_df, search_bounds=None) -> pd.DataFrame:
    """
        Compute Parameter Drift Index
    """
    search_bounds = search_bounds or {}
    param_cols = [c for c in master_df.columns if c.startswith("config/")]
    
    best_per_window = master_df.loc[
        master_df.groupby("wfo_window")["metrics_score"].idxmax()
    ].sort_values("wfo_window")

    drift_report = []
    for col in param_cols:
        pname = col.replace("config/", "")
        vals = best_per_window[col].values.astype(float)
        
        std_val = np.std(vals)
        mean_val = np.mean(vals)
        
        bounds = search_bounds.get(pname, search_bounds.get(col))
        if bounds is not None and len(bounds) >= 2:
            search_range = bounds[-1] - bounds[0]
            drift_index = std_val / search_range if search_range > 0 else 0.0
        else:
            drift_index = np.nan

        drift_report.append({
            "parameter": pname,
            "mean_best_val": mean_val,
            "std_best_val": std_val,
            "drift_index": drift_index,
            "stability_status": "STABLE" if drift_index <= 0.20 else ("MODERATE" if drift_index <= 0.35 else "HIGH_DRIFT")
        })

    return pd.DataFrame(drift_report), best_per_window


# ============================================================================
# 3. WFO Diagnostic Dashboard
# ============================================================================

def plot_wfo_composite_dashboard(master_df, search_bounds=None, main_param="threshold_r"):
    """
    Panel 1: Best Value over Time
    Panel 2: Heatmap
    Panel 3: 2D Contour / Boxplot
    """
    drift_df, best_per_window = compute_wfo_parameter_drift(master_df, search_bounds)
    
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Panel 1: Drift Line Chart----
    ax1 = fig.add_subplot(gs[0, 0])
    windows = best_per_window["wfo_window"].values
    col_main = f"config/{main_param}" if f"config/{main_param}" in master_df.columns else main_param
    
    ax1.plot(windows, best_per_window[col_main].values, '-o', color='darkblue', linewidth=2, label=f"Best {main_param}")
    ax1.set_title(f"Parameter Drift over WFO Windows ({main_param})", fontsize=11, fontweight='bold')
    ax1.set_xlabel("WFO Window (YYYYMM)")
    ax1.set_ylabel(main_param)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Drift Index
    d_row = drift_df[drift_df["parameter"] == main_param]
    if len(d_row) > 0:
        d_idx = d_row["drift_index"].values[0]
        status = d_row["stability_status"].values[0]
        ax1.text(0.03, 0.90, f"Drift Index: {d_idx:.2f} ({status})", transform=ax1.transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    # ---- Panel 2: WFO Stability Heatmap ----
    ax2 = fig.add_subplot(gs[0, 1])
    
    df_heat = master_df.copy()
    df_heat["param_bin"] = pd.cut(df_heat[col_main], bins=8)
    pivot = df_heat.pivot_table(index="wfo_window", columns="param_bin", values="window_score_z", aggfunc="mean")
    
    im = ax2.imshow(pivot.values, aspect='auto', cmap='RdYlGn', origin='lower')
    ax2.set_yticks(np.arange(len(pivot.index)))
    ax2.set_yticklabels(pivot.index, fontsize=8)
    ax2.set_xticks(np.arange(len(pivot.columns)))
    ax2.set_xticklabels([f"{c.left:.2f}-{c.right:.2f}" for c in pivot.columns], rotation=45, fontsize=8)
    ax2.set_title(f"WFO Stability Heatmap ({main_param} vs Window Score Z)", fontsize=11, fontweight='bold')
    ax2.set_xlabel(f"{main_param} Bins")
    ax2.set_ylabel("WFO Window")
    fig.colorbar(im, ax=ax2, shrink=0.8, label="Window Normalized Z-Score")

    # ---- Panel 3: Global Normalized Boxplot ----
    ax3 = fig.add_subplot(gs[1, :])
    df_heat.boxplot(column="window_score_z", by="param_bin", ax=ax3, grid=True)
    ax3.set_title(f"Global Cross-Window Score Distribution by {main_param}", fontsize=11, fontweight='bold')
    ax3.set_xlabel(main_param)
    ax3.set_ylabel("Normalized Score Z")
    fig.suptitle("WFO Multi-Window Parameter Stability & Global Landscape Dashboard", fontsize=14, fontweight='bold', y=0.98)

    return fig, drift_df


if __name__ == "__main__":
    base_dir = "/tmp/ray_results"
    
    try:
        master_df = load_all_wfo_experiments(base_dir)
        
        search_bounds = {
            "downsample": [3, 4, 5],
            "motif_minutes": [30, 45, 60, 90],
            "threshold_r": [0.55, 0.80],
        }

        drift_df, _ = compute_wfo_parameter_drift(master_df, search_bounds)
        print("\n=== WFO Parameter Drift Report ===")
        print(drift_df.to_string(index=False))

        fig, _ = plot_wfo_composite_dashboard(master_df, search_bounds, main_param="threshold_r")
        plt.show()

    except Exception as e:
        print(f"❌ Failure: {e}")
