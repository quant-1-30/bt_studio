#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.interpolate import griddata, RBFInterpolator

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


# Detection engine moved to core diagnostics (no matplotlib dependency);
# re-exported here so existing imports keep working.
from bt_studio.utils.diagnostics.collapse import (  # noqa: F401
    detect_space_collapse,
    print_collapse_report,
    build_collapse_report,
    _to_pandas,
    _get_param_columns,
    _clean_param_name,
    is_discrete_param,
)

# ============================================================================
# CJK Font
# ============================================================================

def _setup_cjk_font():
    _CJK_CANDIDATES = [
        'PingFang SC', 'Heiti SC', 'STHeiti', 'Songti SC',
        'Arial Unicode MS', 'Noto Sans CJK SC', 'Source Han Sans SC',
        'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in _CJK_CANDIDATES if name in available), None)
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return chosen
    plt.rcParams['axes.unicode_minus'] = False
    return None

_CJK_FONT = _setup_cjk_font()


# ============================================================================
# interploate (revise fallback to one dimension and Nan)
# ============================================================================

def _interpolate_surface(x, y, z, grid_size=100, method='cubic'):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # 1. z 1%~99%
    z_min_limit, z_max_limit = np.percentile(z, [1, 99]) if len(z) > 5 else (z.min(), z.max())
    z_clipped = np.clip(z, z_min_limit, z_max_limit)

    # 2. aggregate
    df_temp = pd.DataFrame({'x': x, 'y': y, 'z': z_clipped}).groupby(['x', 'y']).mean().reset_index()
    x_clean, y_clean, z_clean = df_temp['x'].values, df_temp['y'].values, df_temp['z'].values

    # fallback grid
    if len(df_temp) < 4 or len(np.unique(x_clean)) < 2 or len(np.unique(y_clean)) < 2:
        gx = np.linspace(x.min(), x.max(), grid_size)
        gy = np.linspace(y.min(), y.max(), grid_size)
        grid_x, grid_y = np.meshgrid(gx, gy)
        grid_z = np.full((grid_size, grid_size), float(np.nanmean(z)))
        return grid_x, grid_y, grid_z

    gx_1d = np.linspace(x_clean.min(), x_clean.max(), grid_size)
    gy_1d = np.linspace(y_clean.min(), y_clean.max(), grid_size)
    grid_x, grid_y = np.meshgrid(gx_1d, gy_1d)

    if method == 'rbf':
        try:
            rbf = RBFInterpolator(
                np.column_stack([x_clean, y_clean]), z_clean,
                kernel='thin_plate_spline', smoothing=0.5
            )
            pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            grid_z = rbf(pts).reshape(grid_x.shape)
        except Exception:
            grid_z = griddata((x_clean, y_clean), z_clean, (grid_x, grid_y), method='linear')
    else:
        grid_z = griddata((x_clean, y_clean), z_clean, (grid_x, grid_y), method=method)
        if method == 'cubic':
            nan_mask = np.isnan(grid_z)
            if nan_mask.any():
                linear_z = griddata((x_clean, y_clean), z_clean, (grid_x, grid_y), method='linear')
                grid_z[nan_mask] = linear_z[nan_mask]

    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        nearest_z = griddata((x_clean, y_clean), z_clean, (grid_x, grid_y), method='nearest')
        grid_z[nan_mask] = nearest_z[nan_mask]

    return grid_x, grid_y, grid_z


# ============================================================================
# 2D & 3D & Dashboard render
# ============================================================================

def plot_tune_contour(results_df, param_x, param_y, target="metrics_score", method='cubic', **kwargs):
    ax = kwargs.get("ax")
    cmap = kwargs.get("cmap", "viridis")
    levels = int(kwargs.get("levels", 20))
    grid_size = int(kwargs.get("grid_size", 100))
    df = _to_pandas(results_df)
    col_x = param_x if param_x in df.columns else f"config/{param_x}"
    col_y = param_y if param_y in df.columns else f"config/{param_y}"
    valid = df[np.isfinite(df[target])].copy()
    if len(valid) < 3:
        raise ValueError(f"有效样本不足 ({len(valid)} < 3)")

    x, y, z = valid[col_x].values, valid[col_y].values, valid[target].values
    grid_x, grid_y, grid_z = _interpolate_surface(x, y, z, grid_size, method)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    vmin, vmax = np.nanmin(grid_z), np.nanmax(grid_z)
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
        cs = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        cs = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap)

    ax.contour(grid_x, grid_y, grid_z, levels=min(10, levels), colors='white', linewidths=0.4, alpha=0.5)
    ax.scatter(x, y, c=z, cmap=cmap, edgecolors='black', s=30, linewidths=0.5, zorder=3, vmin=vmin, vmax=vmax)

    best_idx = np.nanargmax(z)
    ax.scatter(x[best_idx], y[best_idx], marker='*', s=250, color='red', edgecolors='black', linewidths=1.0, zorder=5,
               label=f"Best ({_clean_param_name(col_x)}={x[best_idx]:.2f}, {_clean_param_name(col_y)}={y[best_idx]:.2f})\n{target}={z[best_idx]:.2f}")
    ax.legend(loc='best', fontsize=8)

    cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
    cbar.set_label(target)
    ax.set_xlabel(_clean_param_name(col_x))
    ax.set_ylabel(_clean_param_name(col_y))
    ax.set_title(f"Contour: {target} vs ({_clean_param_name(col_x)}, {_clean_param_name(col_y)})")

    return fig, ax, cs


def plot_tune_landscape_3d(results_df, param_x, param_y, target="metrics_score", method='cubic', **kwargs):
    ax = kwargs.get("ax")
    cmap = kwargs.get("cmap", "viridis")
    grid_size = int(kwargs.get("grid_size", 80))
    df = _to_pandas(results_df)
    col_x = param_x if param_x in df.columns else f"config/{param_x}"
    col_y = param_y if param_y in df.columns else f"config/{param_y}"
    valid = df[np.isfinite(df[target])].copy()
    if len(valid) < 5:
        raise ValueError(f"有效样本不足 ({len(valid)} < 5)")

    x, y, z = valid[col_x].values, valid[col_y].values, valid[target].values
    grid_x, grid_y, grid_z = _interpolate_surface(x, y, z, grid_size, method)

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    vmin, vmax = np.nanmin(grid_z), np.nanmax(grid_z)
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cmap, edgecolor='none', alpha=0.8, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
    ax.scatter(x, y, z, color='black', s=20, zorder=5, label='Sampled Trials', edgecolors='white', linewidths=0.4)

    best_idx = np.nanargmax(z)
    ax.scatter([x[best_idx]], [y[best_idx]], [z[best_idx]], marker='*', s=300, color='red', edgecolors='black', linewidths=1.0, zorder=6, label='Best')

    ax.set_xlabel(_clean_param_name(col_x))
    ax.set_ylabel(_clean_param_name(col_y))
    ax.set_zlabel(target)
    ax.set_title(f"3D Landscape: {target}")
    ax.legend(loc='best', fontsize=8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=target)

    return fig, ax, surf


def plot_collapse_dashboard(results_df, target="metrics_score", search_bounds=None, method='cubic', param_pair=None):
    cmap = "viridis"
    figsize = (16, 12)
    df = _to_pandas(results_df)
    diagnostics = detect_space_collapse(df, target, search_bounds)

    param_cols = [_clean_param_name(c) for c in diagnostics["_param_cols"]]
    if len(param_cols) < 2:
        raise ValueError(f"至少需要 2 个参数,当前: {param_cols}")

    if param_pair is None:
        scored = [(p, diagnostics[p]["landscape_variance_ratio"]) for p in param_cols if p in diagnostics and isinstance(diagnostics[p], dict)]
        scored = [s for s in scored if np.isfinite(s[1])]
        scored.sort(key=lambda x: x[1], reverse=True)
        # landscape 全 NaN (trial 数少): 回退前两个参数列, 不得 IndexError
        param_x = scored[0][0] if scored else param_cols[0]
        param_y = (scored[1][0] if len(scored) > 1 and scored[1][0] != param_x
                   else (param_cols[1] if len(param_cols) > 1 else param_cols[0]))
    else:
        param_x, param_y = param_pair

    valid = df[np.isfinite(df[target])].copy()

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35, width_ratios=[1.2, 1.2, 1.0])

    # Panel 1: Contour
    ax1 = fig.add_subplot(gs[0, 0])
    plot_tune_contour(df, param_x, param_y, target=target, method=method, ax=ax1, cmap=cmap)

    # Panel 2: 3D Landscape
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_tune_landscape_3d(df, param_x, param_y, target=target, method=method, ax=ax2, cmap=cmap)

    # Panel 3: Collapse text
    ax3 = fig.add_subplot(gs[0, 2])
    _render_collapse_text(ax3, diagnostics, target)

    # Panel 4: Edge hist
    n_params = len(param_cols)
    n_rows_dist = (n_params + 2) // 3
    gs_dist = GridSpecFromSubplotSpec(n_rows_dist, 3, subplot_spec=gs[1, 0:2], hspace=0.4, wspace=0.3)
    for i, pname in enumerate(param_cols):
        r, c = divmod(i, 3)
        ax_d = fig.add_subplot(gs_dist[r, c])
        col_name = pname if pname in valid.columns else f"config/{pname}"

        if col_name not in valid.columns:
            continue

        vals = valid[col_name].dropna().values
        is_disc, _is_cat = is_discrete_param(col_name, valid[col_name], search_bounds)
        if is_disc:
            unique, counts = np.unique(vals, return_counts=True)
            ax_d.bar(unique.astype(str), counts, color='steelblue', edgecolor='black')
            ax_d.set_ylabel("Count")
        else:
            ax_d.hist(vals, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
            ax_d.set_ylabel("Frequency")
        ax_d.set_xlabel(pname)

        d = diagnostics.get(pname, {})
        tag = "[!]" if d.get("is_isolated_spike") or d.get("is_collapsed") else ("[--]" if d.get("is_flat") else "[OK]")
        ax_d.set_title(f"{tag} {pname}", fontsize=9)

    # Panel 5: Trajectory
    ax5 = fig.add_subplot(gs[1, 2])
    trial_idx = np.arange(len(valid))
    colors = plt.cm.tab10(np.linspace(0, 1, len(param_cols)))
    for i, pname in enumerate(param_cols):
        col_name = pname if pname in valid.columns else f"config/{pname}"
        if col_name not in valid.columns:
            continue

        vals = valid[col_name].dropna().values
        vmin, vmax = vals.min(), vals.max()
        normed = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        ax5.plot(trial_idx[:len(normed)], normed, '-o', markersize=2, color=colors[i], label=pname, alpha=0.7)
    ax5.set_xlabel("Trial Index")
    ax5.set_ylabel("Normalized Value [0, 1]")
    ax5.set_title("Sampling Trajectory", fontsize=10)
    ax5.legend(fontsize=7, loc='best')
    ax5.grid(True, alpha=0.3)

    fig.suptitle(
        f"Param Space Dashboard | verdict={diagnostics['overall_verdict'].upper()} | n_trials={diagnostics['_n_trials']}",
        fontsize=13, fontweight='bold', y=0.98
    )

    return fig, diagnostics


def _render_collapse_text(ax, diagnostics, target):
    ax.axis('off')
    verdict = diagnostics["overall_verdict"]
    tag_map = {
        "healthy_plateau": "[OK] HEALTHY PLATEAU",
        "isolated_spike": "[!] ISOLATED SPIKE",
        "collapsed": "[!] SPACE COLLAPSED",
        "flat_landscape": "[--] FLAT LANDSCAPE"
    }
    verdict_text = f"{tag_map.get(verdict, '?')}\n"
    verdict_text += f"Target: {target} | Trials: {diagnostics['_n_trials']}\n"
    verdict_text += "=" * 32 + "\n\n"

    param_cols = [_clean_param_name(c) for c in diagnostics["_param_cols"]]
    for name in param_cols:
        d = diagnostics.get(name)
        if d is None or not isinstance(d, dict):
            continue
        status = "SPIKE" if d["is_isolated_spike"] else ("COLLAPSED" if d["is_collapsed"] else ("FLAT" if d["is_flat"] else "NORMAL"))
        dtype = "discrete" if d["is_discrete"] else "continuous"
        verdict_text += f"[{name}] ({dtype}) → {status}\n"
        spread_str = f"{d['spread_ratio']:.2f}" if not np.isnan(d['spread_ratio']) else "N/A"
        verdict_text += f"  spread: {spread_str}  entropy: {d['kde_entropy']:.2f}\n"
        verdict_text += f"  land_var: {d['landscape_variance_ratio']:.2f}  drop: {d['worst_drop_ratio']:.1f}xIQR\n\n"

    ax.text(0.05, 0.95, verdict_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


# ============================================================================
# Load Ray experiment results
# ============================================================================

def load_real_ray_results(experiment_dir: str) -> pd.DataFrame:
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"⚠️ 找不到 Ray 实验目录: {experiment_dir}")

    try:
        from ray.tune import ExperimentAnalysis
        analysis = ExperimentAnalysis(experiment_dir)
        df = analysis.dataframe()
        if len(df) > 0:
            return df
    except Exception:
        pass

    csv_pattern = os.path.join(experiment_dir, "trainable_fsm_worker_*", "progress.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        json_pattern = os.path.join(experiment_dir, "trainable_fsm_worker_*", "result.json")
        json_files = glob.glob(json_pattern)
        if json_files:
            df_pl = pl.read_ndjson(json_files)
            return df_pl.to_pandas()
        raise FileNotFoundError(f"目录 {experiment_dir} 下未找到 progress.csv 或 result.json")

    df_pl = pl.read_csv(csv_files)
    return df_pl.to_pandas()


# ============================================================================
# Offline Collapse & WFO Multi-Window Parameter Drift Exporters
# ============================================================================

def export_collapse_pngs(
    exp_dir: str,
    out_dir: str = "png",
    target: str = "metrics_score",
    method: str = "cubic",
    search_bounds: dict | None = None,
) -> dict:
    """Generate and save offline contour, 3D landscape, and collapse dashboard PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_real_ray_results(exp_dir)
    bounds = search_bounds or {
        "downsample": [3, 4, 5],
        "motif_minutes": [30, 45, 60, 90],
        "threshold_r": [0.6, 0.8],
    }
    diag = detect_space_collapse(df, target, bounds)
    print_collapse_report(diag)

    fig1, _, _ = plot_tune_contour(df, "threshold_r", "downsample", target, method=method)
    out1 = os.path.join(out_dir, "vis_contour.png")
    fig1.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig1)

    fig2, _, _ = plot_tune_landscape_3d(df, "threshold_r", "downsample", target, method=method)
    out2 = os.path.join(out_dir, "vis_3d.png")
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    fig3, _ = plot_collapse_dashboard(df, target, bounds, method=method)
    out3 = os.path.join(out_dir, "vis_dashboard.png")
    fig3.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig3)

    return {"contour": out1, "landscape_3d": out2, "dashboard": out3, "diagnostics": diag}


def load_all_wfo_experiments(base_ray_dir: str = "/tmp/ray_results", target: str = "metrics_score") -> pd.DataFrame:
    """Load and MAD-normalize all fsm_hpo_* WFO experiment directories under base_ray_dir."""
    from ray.tune import ExperimentAnalysis

    exp_folders = sorted(
        os.path.join(base_ray_dir, d) for d in os.listdir(base_ray_dir) if d.startswith("fsm_hpo_")
    )
    if not exp_folders:
        raise FileNotFoundError(f"No fsm_hpo_* folders found in {base_ray_dir}")

    all_trials = []
    for folder in exp_folders:
        window_id = os.path.basename(folder).replace("fsm_hpo_", "")
        try:
            df_win = ExperimentAnalysis(folder).dataframe()
        except Exception:
            df_win = None
        if df_win is not None and len(df_win) > 0 and target in df_win.columns:
            df_win["wfo_window"] = window_id
            valid_mask = np.isfinite(df_win[target]) & (df_win[target] > -9999.0)
            if valid_mask.sum() > 3:
                scores = df_win.loc[valid_mask, target]
                med = np.median(scores)
                mad = np.median(np.abs(scores - med))
                df_win.loc[valid_mask, "window_score_z"] = (scores - med) / (1.4826 * mad + 1e-6)
                all_trials.append(df_win[valid_mask])

    if not all_trials:
        raise ValueError("No valid WFO experiments loaded")
    return pd.concat(all_trials, ignore_index=True)


def compute_wfo_parameter_drift(master_df: pd.DataFrame, search_bounds: dict | None = None):
    """Compute cross-window hyperparameter drift index across WFO windows."""
    search_bounds = search_bounds or {}
    param_cols = [c for c in master_df.columns if c.startswith("config/")]
    best_per_window = master_df.loc[
        master_df.groupby("wfo_window")["metrics_score"].idxmax()
    ].sort_values("wfo_window")

    drift_report = []
    for col in param_cols:
        pname = col.replace("config/", "")
        vals = best_per_window[col].values.astype(float)
        std_val, mean_val = float(np.std(vals)), float(np.mean(vals))
        bounds = search_bounds.get(pname, search_bounds.get(col))
        if bounds is not None and len(bounds) >= 2:
            span = float(bounds[-1] - bounds[0])
            drift_index = std_val / span if span > 0 else 0.0
        else:
            drift_index = np.nan

        drift_report.append({
            "parameter": pname,
            "mean_best_val": mean_val,
            "std_best_val": std_val,
            "drift_index": drift_index,
            "stability_status": "STABLE" if drift_index <= 0.20 else ("MODERATE" if drift_index <= 0.35 else "HIGH_DRIFT"),
        })
    return pd.DataFrame(drift_report), best_per_window


def plot_wfo_composite_dashboard(master_df: pd.DataFrame, search_bounds: dict | None = None, main_param: str = "threshold_r"):
    """Plot 3-panel WFO multi-window parameter drift, stability heatmap, and global boxplot dashboard."""
    drift_df, best_per_window = compute_wfo_parameter_drift(master_df, search_bounds)
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    windows = best_per_window["wfo_window"].values
    col_main = f"config/{main_param}" if f"config/{main_param}" in master_df.columns else main_param
    ax1.plot(windows, best_per_window[col_main].values, "-o", color="darkblue", linewidth=2, label=f"Best {main_param}")
    ax1.set_title(f"Parameter Drift over WFO Windows ({main_param})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("WFO Window (YYYYMM)")
    ax1.set_ylabel(main_param)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    d_row = drift_df[drift_df["parameter"] == main_param]
    if len(d_row) > 0:
        ax1.text(
            0.03, 0.90,
            f"Drift Index: {d_row['drift_index'].values[0]:.2f} ({d_row['stability_status'].values[0]})",
            transform=ax1.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )

    ax2 = fig.add_subplot(gs[0, 1])
    df_heat = master_df.copy()
    df_heat["param_bin"] = pd.cut(df_heat[col_main], bins=8)
    pivot = df_heat.pivot_table(index="wfo_window", columns="param_bin", values="window_score_z", aggfunc="mean", observed=False)
    im = ax2.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax2.set_yticks(np.arange(len(pivot.index)))
    ax2.set_yticklabels(pivot.index, fontsize=8)
    ax2.set_xticks(np.arange(len(pivot.columns)))
    ax2.set_xticklabels([f"{c.left:.2f}-{c.right:.2f}" for c in pivot.columns], rotation=45, fontsize=8)
    ax2.set_title(f"WFO Stability Heatmap ({main_param} vs Window Score Z)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax2, shrink=0.8, label="Window Normalized Z-Score")

    ax3 = fig.add_subplot(gs[1, :])
    df_heat.boxplot(column="window_score_z", by="param_bin", ax=ax3, grid=True)
    ax3.set_title(f"Global Cross-Window Score Distribution by {main_param}", fontsize=11, fontweight="bold")
    fig.suptitle("WFO Multi-Window Parameter Stability & Global Landscape Dashboard", fontsize=14, fontweight="bold", y=0.98)
    return fig, drift_df


def export_drift_png(
    base_ray_dir: str = "/tmp/ray_results",
    out_dir: str = "png",
    target: str = "metrics_score",
    search_bounds: dict | None = None,
) -> dict:
    """Load all WFO experiments under base_ray_dir and save the composite parameter drift PNG."""
    os.makedirs(out_dir, exist_ok=True)
    master_df = load_all_wfo_experiments(base_ray_dir, target=target)
    fig, drift_df = plot_wfo_composite_dashboard(master_df, search_bounds=search_bounds)
    out_path = os.path.join(out_dir, "wfo_stability_dashboard.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"dashboard": out_path, "drift_df": drift_df}
