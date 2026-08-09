import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.interpolate import griddata, RBFInterpolator
from scipy.stats import gaussian_kde

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


# ============================================================================
# CJK & Tools
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

def _to_pandas(results_df):
    if _HAS_POLARS and isinstance(results_df, pl.DataFrame):
        return results_df.to_pandas()
    return results_df

def _get_param_columns(df):
    param_cols = [c for c in df.columns if c.startswith("config/")]
    if not param_cols:
        param_cols = [c for c in df.columns if c not in ["trial_id", "metrics_score", "u_pval", "trigger_count", "valid_sample_ratio", "autocorr"]]
    return param_cols

def _clean_param_name(name):
    return name.replace("config/", "")

def _is_discrete_param(values):
    arr = np.asarray(values)
    is_int = np.allclose(arr, np.round(arr))
    few_unique = len(np.unique(arr)) <= 8
    return is_int or few_unique


# ============================================================================
# Interpolation Engine for Contour / 3D Landscape
# ============================================================================

def _interpolate_surface(x, y, z, grid_size=100, method='cubic'):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if len(x) < 4:
        gx = np.linspace(x.min(), x.max(), grid_size)
        gy = np.linspace(y.min(), y.max(), grid_size)
        gz = np.full((grid_size, grid_size), float(np.nanmean(z)))
        return gx, gy, gz

    df_temp = pd.DataFrame({'x': x, 'y': y, 'z': z}).groupby(['x', 'y']).mean().reset_index() # x,y to column
    x_clean, y_clean, z_clean = df_temp['x'].values, df_temp['y'].values, df_temp['z'].values

    gx_1d = np.linspace(x_clean.min(), x_clean.max(), grid_size)
    gy_1d = np.linspace(y_clean.min(), y_clean.max(), grid_size)
    grid_x, grid_y = np.meshgrid(gx_1d, gy_1d)

    if method == 'rbf':
        try:
            rbf = RBFInterpolator(
                np.column_stack([x_clean, y_clean]), z_clean,
                kernel='thin_plate_spline', smoothing=0.1
            )
            pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            grid_z = rbf(pts).reshape(grid_x.shape)
        except Exception:
            grid_z = griddata((x_clean, y_clean), z_clean, (grid_x, grid_y), method='linear')
    else:
        grid_z = griddata((x_clean, y_clean), z_clean, (grid_x, grid_y), method=method)
        # cubic -> linear -> nearest
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
# Plateau vs Spike
# ============================================================================

def detect_space_collapse(results_df, target="metrics_score", search_bounds=None):
    """
    1. spread_ratio:              实际探索范围 / 搜索空间范围
    2. kde_entropy:               基于 KDE 的标准化熵 [0, 1]
    3. landscape_variance_ratio:  加权组间方差 / 总方差
    4. peak_drop_std:             孤立刺峰指数 (高原 vs 刺峰)
    """
    df = _to_pandas(results_df)
    param_cols = _get_param_columns(df) # 💥 提前捕获纯净的参数列名
    
    valid = df[np.isfinite(df[target])].copy()
    if len(valid) < 3:
        raise ValueError(f"有效样本不足 ({len(valid)})")

    search_bounds = search_bounds or {}
    diagnostics = {}
    best_idx = np.nanargmax(valid[target].values)
    best_score = valid[target].values[best_idx]

    for col in param_cols:
        name = _clean_param_name(col)
        raw_col_name = col if col in valid.columns else f"config/{col}"
        if raw_col_name not in valid.columns:
            continue
            
        values = valid[raw_col_name].dropna().values.astype(float)
        if len(values) == 0:
            continue

        actual_range = values.max() - values.min()
        is_discrete = _is_discrete_param(values)

        # --- spread_ratio ---
        bounds = search_bounds.get(name, search_bounds.get(col))
        if bounds is not None and len(bounds) >= 2:
            if is_discrete and isinstance(bounds, (list, tuple)) and len(bounds) > 2:
                covered = len(set(values.tolist()) & set(bounds))
                spread_ratio = covered / len(bounds) if len(bounds) > 0 else 0.0
            else:
                search_range = bounds[-1] - bounds[0]
                spread_ratio = actual_range / search_range if search_range > 0 else 0.0
        else:
            spread_ratio = float('nan')

        # --- KDE Entropy ---
        std_val = np.std(values)
        if not is_discrete and std_val > 1e-10 and len(values) > 5:
            try:
                kde = gaussian_kde(values, bw_method='scott')
                eval_pts = np.linspace(values.min(), values.max(), 200)
                p = np.clip(kde(eval_pts), 1e-12, None)
                dx = eval_pts[1] - eval_pts[0]
                diff_entropy = -np.sum(p * np.log(p)) * dx
                H_max = np.log(actual_range) if actual_range > 1e-12 else 0.0
                kde_entropy = float(np.clip(np.exp(diff_entropy - H_max), 0.0, 1.0))
            except Exception:
                kde_entropy = float('nan')
        else:
            unique, counts = np.unique(values, return_counts=True)
            probs = counts / counts.sum()
            shannon_h = -np.sum(probs * np.log(probs))
            max_h = np.log(len(unique)) if len(unique) > 1 else 1.0
            kde_entropy = shannon_h / max_h if max_h > 0 else 0.0

        # --- Landscape Variance Ratio ---
        landscape_var_ratio = _landscape_variance_ratio(values, valid[target].values)

        # --- 孤立刺峰检测 ---
        best_val = valid[raw_col_name].values[best_idx]
        delta = actual_range * 0.15
        if delta == 0:
            delta = 1e-5
            
        neighbors_mask = (values >= best_val - delta) & (values <= best_val + delta)
        neighbors_mask[best_idx] = False 
        
        if neighbors_mask.sum() > 0:
            neighbor_mean_score = valid[target].values[neighbors_mask].mean()
            score_std = np.nanstd(valid[target].values)
            if score_std > 1e-8:
                peak_drop_std = (best_score - neighbor_mean_score) / score_std
            else:
                peak_drop_std = 0.0
        else:
            peak_drop_std = float('nan')

        # --- 判定逻辑 ---
        is_collapsed = False
        is_isolated_spike = False
        is_flat = False
        reasons = []

        if not is_discrete and not np.isnan(spread_ratio) and spread_ratio < 0.4:
            is_collapsed = True
            reasons.append(f"spread_ratio={spread_ratio:.2f} < 0.40 (探索范围坍塌)")

        if not np.isnan(kde_entropy) and kde_entropy < 0.15 and not is_discrete:
            is_collapsed = True
            reasons.append(f"kde_entropy={kde_entropy:.3f} < 0.15 (样本极度集中)")

        if landscape_var_ratio < 0.05:
            is_flat = True
            reasons.append(f"landscape_var_ratio={landscape_var_ratio:.3f} < 0.05 (参数无效果)")

        if not np.isnan(peak_drop_std) and peak_drop_std > 1.2:
            is_isolated_spike = True
            reasons.append(f"peak_drop_std={peak_drop_std:.2f}σ > 1.2σ (孤立刺峰)")

        diagnostics[name] = {
            "n_samples": len(values),
            "is_discrete": is_discrete,
            "spread_ratio": float(spread_ratio),
            "kde_entropy": float(kde_entropy),
            "landscape_variance_ratio": float(landscape_var_ratio),
            "peak_drop_std": float(peak_drop_std),
            "is_collapsed": is_collapsed,
            "is_flat": is_flat,
            "is_isolated_spike": is_isolated_spike,
            "reasons": reasons,
        }

    # --- verdict ---
    any_spike = any(d["is_isolated_spike"] for d in diagnostics.values() if isinstance(d, dict))
    any_collapsed = any(d["is_collapsed"] for d in diagnostics.values() if isinstance(d, dict))
    all_flat = (len(diagnostics) > 0 and all(d["is_flat"] for d in diagnostics.values() if isinstance(d, dict)))
    
    if any_spike:
        verdict = "isolated_spike"
    elif all_flat:
        verdict = "flat_landscape"
    elif any_collapsed:
        verdict = "collapsed"
    else:
        verdict = "healthy_plateau"

    diagnostics["overall_verdict"] = verdict
    diagnostics["_param_cols"] = param_cols
    diagnostics["_target"] = target
    diagnostics["_n_trials"] = len(valid)

    return diagnostics


def _landscape_variance_ratio(param_values, target_values, n_bins=8):
    """ANOVA SS_between / SS_total)"""
    param_values = np.asarray(param_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)

    if len(param_values) < 5:
        return float('nan')

    total_var = np.nanvar(target_values)
    if total_var < 1e-12:
        return 0.0

    grand_mean = np.nanmean(target_values)
    n_unique = len(np.unique(param_values))
    actual_bins = min(n_bins, n_unique)
    if actual_bins < 2:
        return 0.0

    try:
        ss_between = 0.0
        if _is_discrete_param(param_values):
            unique_vals = np.unique(param_values)
            for uv in unique_vals:
                mask = param_values == uv
                n_k = mask.sum()
                if n_k > 0:
                    mean_k = np.nanmean(target_values[mask])
                    ss_between += n_k * ((mean_k - grand_mean) ** 2)
        else:
            bin_edges = np.linspace(param_values.min(), param_values.max(), actual_bins + 1)
            bin_indices = np.digitize(param_values, bin_edges[1:-1])
            for bi in range(actual_bins):
                mask = bin_indices == bi
                n_k = mask.sum()
                if n_k > 0:
                    mean_k = np.nanmean(target_values[mask])
                    ss_between += n_k * ((mean_k - grand_mean) ** 2)

        var_between = ss_between / len(target_values)
        return float(var_between / total_var)
    except Exception:
        return float('nan')


def print_collapse_report(diagnostics):
    verdict = diagnostics["overall_verdict"]
    tag_map = {
        "healthy_plateau": "[OK] HEALTHY PLATEAU", 
        "isolated_spike": "[!] ISOLATED SPIKE", 
        "collapsed": "[!] SPACE COLLAPSED", 
        "flat_landscape": "[--] FLAT LANDSCAPE"
    }
    print(f"\n{'='*65}")
    print(f"  {tag_map.get(verdict, '?')} Diagnosis: {verdict.upper()}")
    print(f"  Trial 数: {diagnostics['_n_trials']} | Target: {diagnostics['_target']}")
    print(f"{'='*65}")

    for name in diagnostics.get("_param_cols", []):
        pname = name.replace("config/", "")
        d = diagnostics.get(pname)
        if d is None or not isinstance(d, dict):
            continue
        status = "[!] SPIKE" if d["is_isolated_spike"] else ("[!] COLLAPSED" if d["is_collapsed"] else ("[--] FLAT" if d["is_flat"] else "[OK] NORMAL"))
        dtype = "discrete" if d["is_discrete"] else "continuous"
        print(f"\n  [{pname}] ({dtype}, n={d['n_samples']}) → {status}")
        spread_str = f"{d['spread_ratio']:.3f}" if not np.isnan(d['spread_ratio']) else "N/A"
        print(f"    spread_ratio        = {spread_str}")
        print(f"    kde_entropy         = {d['kde_entropy']:.3f}")
        print(f"    landscape_var_ratio = {d['landscape_variance_ratio']:.3f}")
        print(f"    peak_drop_std       = {d['peak_drop_std']:.3f}σ")
        if d["reasons"]:
            for r in d["reasons"]:
                print(f"    → {r}")
    print(f"\n{'='*65}\n")


# ============================================================================
# Dashboard
# ============================================================================

def plot_tune_contour(results_df, param_x, param_y, target="metrics_score", ax=None, method='cubic', show_samples=True, show_best=True, grid_size=100, cmap='viridis', levels=20):
    df = _to_pandas(results_df)
    col_x = param_x if param_x in df.columns else f"config/{param_x}"
    col_y = param_y if param_y in df.columns else f"config/{param_y}"
    valid = df[np.isfinite(df[target])].copy()
    if len(valid) < 3:
        raise ValueError(f"Not enough valid samples: {len(valid)} < 3")

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

    if show_samples:
        ax.scatter(x, y, c=z, cmap=cmap, edgecolors='black', s=30, linewidths=0.5, zorder=3, vmin=vmin, vmax=vmax)

    if show_best:
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


def plot_tune_landscape_3d(results_df, param_x, param_y, target="metrics_score", ax=None, method='cubic', grid_size=80, cmap='viridis', show_samples=True, show_best=True):
    df = _to_pandas(results_df)
    col_x = param_x if param_x in df.columns else f"config/{param_x}"
    col_y = param_y if param_y in df.columns else f"config/{param_y}"
    valid = df[np.isfinite(df[target])].copy()
    if len(valid) < 5:
        raise ValueError(f"Not enough valid samples: {len(valid)} < 5")

    x, y, z = valid[col_x].values, valid[col_y].values, valid[target].values
    grid_x, grid_y, grid_z = _interpolate_surface(x, y, z, grid_size, method)

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    vmin, vmax = np.nanmin(grid_z), np.nanmax(grid_z)
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cmap, edgecolor='none', alpha=0.8, vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)

    if show_samples:
        ax.scatter(x, y, z, color='black', s=20, zorder=5, label='Sampled Trials', edgecolors='white', linewidths=0.4)

    if show_best:
        best_idx = np.nanargmax(z)
        ax.scatter([x[best_idx]], [y[best_idx]], [z[best_idx]], marker='*', s=300, color='red', edgecolors='black', linewidths=1.0, zorder=6, label='Best')

    ax.set_xlabel(_clean_param_name(col_x))
    ax.set_ylabel(_clean_param_name(col_y))
    ax.set_zlabel(target)
    ax.set_title(f"3D Landscape: {target}")
    ax.legend(loc='best', fontsize=8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=target)

    return fig, ax, surf


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
        verdict_text += f"  land_var: {d['landscape_variance_ratio']:.2f}  drop: {d['peak_drop_std']:.2f}σ\n\n"

    ax.text(0.05, 0.95, verdict_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


def plot_collapse_dashboard(results_df, target="metrics_score", search_bounds=None, method='cubic', param_pair=None, cmap='viridis', figsize=(16, 12)):
    df = _to_pandas(results_df)
    diagnostics = detect_space_collapse(df, target, search_bounds)

    param_cols = [_clean_param_name(c) for c in diagnostics["_param_cols"]]
    if len(param_cols) < 2:
        raise ValueError(f"Not enough parameters: {param_cols}")

    if param_pair is None:
        scored = [(p, diagnostics[p]["landscape_variance_ratio"]) for p in param_cols if p in diagnostics and isinstance(diagnostics[p], dict)]
        scored = [s for s in scored if not np.isnan(s[1])]
        scored.sort(key=lambda x: x[1], reverse=True)
        param_x, param_y = scored[0][0], scored[1][0]
    else:
        param_x, param_y = param_pair

    valid = df[np.isfinite(df[target])].copy()

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35, width_ratios=[1.2, 1.2, 1.0])

    # Panel 1: Contour
    ax1 = fig.add_subplot(gs[0, 0])
    plot_tune_contour(df, param_x, param_y, target=target, ax=ax1, method=method, show_samples=True, show_best=True, cmap=cmap)

    # Panel 2: 3D Landscape
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_tune_landscape_3d(df, param_x, param_y, target=target, ax=ax2, method=method, show_samples=True, show_best=True, cmap=cmap)

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
        is_disc = _is_discrete_param(vals)
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

# ============================================================================
# Load Ray Tune 
# ============================================================================

def load_ray_results(experiment_path):
    """

    Parameters
    ----------
    experiment_path : str
        Ray Tune experiment /tmp/ray_results/fsm_hpo_201002

    Returns
    -------
    pd.DataFrame
    """
    from ray.tune import ExperimentAnalysis

    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"⚠️ NotFound Ray {experiment_path}")

    try:
        analysis = ExperimentAnalysis(experiment_path)
        df = analysis.dataframe()
        if len(df) > 0:
            print(f"✅ [Ray API] Load from {experiment_path} {len(df)}  Trials")
            return df
    except Exception as e:
        print(f"⚠️ Ray ExperimentAnalysis load failed ({e})...")
