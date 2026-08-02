"""
参数空间可视化与塌陷检测模块。

提供以下核心能力：
1. 2D contour plot —— meshgrid 渲染参数响应曲面（等高线 + 采样点叠加）
2. 3D landscape  —— 改进版 3D 曲面（支持 RBF 外推、标注最优点）
3. 塌陷检测      —— spread_ratio / CV / KDE熵 / landscape方差比，判定参数是否塌陷
4. 综合仪表盘    —— 多面板组合图（contour + 3D + 边缘分布 + 采样轨迹）

兼容 pandas / polars DataFrame（自动转换）。
仅依赖 matplotlib + scipy + numpy（无 plotly）。
"""

import numpy as np
import pandas as pd
import matplotlib
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
# CJK 字体自动配置（修复中文显示 bug）
# ============================================================================

def _setup_cjk_font():
    """
    自动探测并设置支持中文的 matplotlib 字体。
    优先级 macOS 内置 CJK 字体 > Linux 常见字体 > Windows 字体。
    """
    _CJK_CANDIDATES = [
        'PingFang SC', 'Heiti SC', 'STHeiti', 'Songti SC',  # macOS
        'Arial Unicode MS',                                   # macOS / 跨平台
        'Noto Sans CJK SC', 'Source Han Sans SC',             # Linux
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',           # Linux
        'SimHei', 'Microsoft YaHei',                          # Windows
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in _CJK_CANDIDATES if name in available), None)
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return chosen
    # 无 CJK 字体时的 fallback
    plt.rcParams['axes.unicode_minus'] = False
    return None


# 模块加载时自动配置
_CJK_FONT = _setup_cjk_font()

# emoji / CJK 在 matplotlib 中渲染不稳定，统一用 ASCII 标记替代
_STATUS_TAGS = {
    'healthy': '[OK]',
    'collapsed': '[!]',
    'flat_landscape': '[--]',
    'normal': '[OK]',
    'flat': '[--]',
}


# ============================================================================
# 内部工具
# ============================================================================

def _to_pandas(results_df):
    """polars / pandas 统一转 pandas。"""
    if _HAS_POLARS and isinstance(results_df, pl.DataFrame):
        return results_df.to_pandas()
    return results_df


def _get_param_columns(df):
    """提取 config/ 前缀的参数列（去掉前缀）。"""
    return [c.replace("config/", "") for c in df.columns if c.startswith("config/")]


def _clean_param_name(name):
    """config/threshold_r -> threshold_r"""
    return name.replace("config/", "")


def _is_discrete_param(values):
    """判断参数是否为离散型（整数 / 取值种类少）。"""
    arr = np.asarray(values)
    # 整数或唯一值数量 <= 8 视为离散
    is_int = np.allclose(arr, np.round(arr))
    few_unique = len(np.unique(arr)) <= 8
    return is_int or few_unique


def _interpolate_surface(x, y, z, grid_size=100, method='cubic'):
    """
    meshgrid 插值生成曲面。

    Parameters
    ----------
    x, y, z : array-like
        散点数据
    grid_size : int
        网格分辨率
    method : str
        'linear' / 'cubic' (griddata) 或 'rbf' (RBFInterpolator, 支持外推)

    Returns
    -------
    grid_x, grid_y, grid_z : ndarray (grid_size, grid_size)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # 退化几何保护
    if len(x) < 4:
        gx = np.linspace(x.min(), x.max(), grid_size)
        gy = np.linspace(y.min(), y.max(), grid_size)
        gz = np.full((grid_size, grid_size), float(np.nanmean(z)))
        return gx, gy, gz

    gx_1d = np.linspace(x.min(), x.max(), grid_size)
    gy_1d = np.linspace(y.min(), y.max(), grid_size)
    grid_x, grid_y = np.meshgrid(gx_1d, gy_1d)

    if method == 'rbf':
        try:
            rbf = RBFInterpolator(
                np.column_stack([x, y]), z,
                kernel='thin_plate_spline', smoothing=0.1
            )
            pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            grid_z = rbf(pts).reshape(grid_x.shape)
        except Exception:
            # RBF 对退化几何可能失败，退化到 linear
            grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    else:
        grid_z = griddata((x, y), z, (grid_x, grid_y), method=method)
        # cubic 对凸包外返回 nan，用 linear 填补
        if method == 'cubic':
            nan_mask = np.isnan(grid_z)
            if nan_mask.any() and not nan_mask.all():
                linear_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
                grid_z[nan_mask] = linear_z[nan_mask]

    # 极端情况下全 nan，填充均值
    if np.all(np.isnan(grid_z)):
        grid_z = np.full_like(grid_z, float(np.nanmean(z)))

    return grid_x, grid_y, grid_z


# ============================================================================
# 2D Contour Plot
# ============================================================================

def plot_tune_contour(
    results_df,
    param_x,
    param_y,
    target="metrics_score",
    ax=None,
    method='cubic',
    show_samples=True,
    show_best=True,
    grid_size=100,
    cmap='viridis',
    levels=20,
):
    """
    2D 等高线图（contourf）+ 可选采样点叠加。

    Parameters
    ----------
    results_df : pd.DataFrame | pl.DataFrame
        Ray Tune 结果，列含 config/* 与 target
    param_x, param_y : str
        参数名（可带或不带 config/ 前缀）
    target : str
        目标指标列名
    ax : matplotlib.axes.Axes | None
        传入已有 axes；None 则新建
    method : str
        插值方法 'linear' / 'cubic' / 'rbf'
    show_samples : bool
        是否叠加实际 trial 散点
    show_best : bool
        是否用红星标注最优点
    grid_size : int
        网格分辨率
    cmap : str
        colormap
    levels : int
        等高线层数

    Returns
    -------
    (fig, ax, contour_set)
    """
    df = _to_pandas(results_df)
    col_x = param_x if param_x in df.columns else f"config/{param_x}"
    col_y = param_y if param_y in df.columns else f"config/{param_y}"
    if target not in df.columns:
        raise ValueError(f"target '{target}' not in columns: {df.columns.tolist()}")

    valid = df[np.isfinite(df[target])].copy()
    if len(valid) < 3:
        raise ValueError(f"有效样本不足 ({len(valid)} < 3)，无法绘制 contour")

    x = valid[col_x].values
    y = valid[col_y].values
    z = valid[target].values

    grid_x, grid_y, grid_z = _interpolate_surface(x, y, z, grid_size, method)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # 等高线填充
    vmin, vmax = np.nanmin(grid_z), np.nanmax(grid_z)
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
        cs = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap,
                         vmin=vmin, vmax=vmax)
    else:
        cs = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap)

    # 等高线
    ax.contour(grid_x, grid_y, grid_z, levels=min(10, levels),
               colors='white', linewidths=0.4, alpha=0.5)

    # 采样点叠加
    if show_samples:
        sc = ax.scatter(x, y, c=z, cmap=cmap, edgecolors='black',
                        s=30, linewidths=0.5, zorder=3, vmin=vmin, vmax=vmax)

    # 标注最优
    if show_best:
        best_idx = np.nanargmax(z)
        ax.scatter(x[best_idx], y[best_idx], marker='*', s=250,
                   color='red', edgecolors='black', linewidths=1.0, zorder=5,
                   label=f"Best ({_clean_param_name(col_x)}={x[best_idx]:.2f}, "
                         f"{_clean_param_name(col_y)}={y[best_idx]:.2f})\n"
                         f"{target}={z[best_idx]:.2f}")
        ax.legend(loc='best', fontsize=8)

    cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
    cbar.set_label(target)

    ax.set_xlabel(_clean_param_name(col_x))
    ax.set_ylabel(_clean_param_name(col_y))
    ax.set_title(f"Contour: {target} vs ({_clean_param_name(col_x)}, {_clean_param_name(col_y)})")

    return fig, ax, cs


# ============================================================================
# 塌陷检测
# ============================================================================

def detect_space_collapse(results_df, target="metrics_score", search_bounds=None):
    """
    检测参数空间是否塌陷。

    对每个参数维度计算 4 个诊断指标：

    - spread_ratio: 实际探索范围 / 搜索空间范围（连续参数）。
                    离散参数则计算覆盖比例（已采样种类 / 总种类）。
    - cv:           采样分布的变异系数（std/mean），越小说明越集中。
    - kde_entropy:  基于高伟KDE的微分熵（标准化），越小说明分布越尖锐。
    - landscape_variance_ratio:
                    该参数方向上响应曲面方差 / 总方差。
                    低（<0.05）说明该参数对目标几乎无影响（平坦维度）。

    Parameters
    ----------
    results_df : pd.DataFrame | pl.DataFrame
    target : str
    search_bounds : dict | None
        搜索边界字典，用于计算 spread_ratio。
        格式：{"downsample": [3,4,5], "threshold_r": [0.6, 0.8]}
        离散参数传列表，连续参数传 [min, max]。

    Returns
    -------
    dict:
        每个参数的指标 + overall_verdict
    """
    df = _to_pandas(results_df)
    param_cols = [c for c in df.columns if c.startswith("config/")]
    valid = df[np.isfinite(df[target])].copy()
    if len(valid) < 3:
        raise ValueError(f"有效样本不足 ({len(valid)})")

    search_bounds = search_bounds or {}
    diagnostics = {}

    for col in param_cols:
        name = _clean_param_name(col)
        values = valid[col].dropna().values.astype(float)

        if len(values) == 0:
            continue

        actual_range = values.max() - values.min()
        is_discrete = _is_discrete_param(values)

        # --- spread_ratio ---
        bounds = search_bounds.get(name, search_bounds.get(col))
        if bounds is not None and len(bounds) >= 2:
            if is_discrete and isinstance(bounds, (list, tuple)) and len(bounds) > 2:
                # 离散：覆盖比例
                covered = len(set(values.tolist()) & set(bounds))
                spread_ratio = covered / len(bounds) if len(bounds) > 0 else 0.0
            else:
                # 连续：范围比
                search_range = bounds[-1] - bounds[0]
                spread_ratio = actual_range / search_range if search_range > 0 else 0.0
        else:
            # 无边界信息，无法计算，置 nan
            spread_ratio = float('nan')

        # --- CV ---
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else float('inf')

        # --- KDE 熵（微分熵，仅连续参数有意义）---
        if not is_discrete and std_val > 1e-10 and len(values) > 5:
            try:
                kde = gaussian_kde(values, bw_method='scott')
                # 采样点估算微分熵 H = -∫ p(x) ln p(x) dx
                eval_pts = np.linspace(values.min(), values.max(), 200)
                p = kde(eval_pts)
                p = np.clip(p, 1e-12, None)
                dx = eval_pts[1] - eval_pts[0]
                diff_entropy = -np.sum(p * np.log(p)) * dx
                # 标准化：最大微分熵 = ln(range)（均匀分布）。
                # 用 exp(H - H_max) 映射到 [0, 1]，对 range < 1 也鲁棒。
                H_max = np.log(actual_range) if actual_range > 1e-12 else 0.0
                kde_entropy = float(np.clip(np.exp(diff_entropy - H_max), 0.0, 1.0))
            except Exception:
                kde_entropy = float('nan')
        else:
            # 离散参数用 Shannon 熵
            unique, counts = np.unique(values, return_counts=True)
            probs = counts / counts.sum()
            shannon_h = -np.sum(probs * np.log(probs))
            max_h = np.log(len(unique)) if len(unique) > 1 else 1.0
            kde_entropy = shannon_h / max_h if max_h > 0 else 0.0

        # --- landscape variance ratio ---
        landscape_var_ratio = _landscape_variance_ratio(values, valid[target].values)

        # --- 塌陷判定 ---
        is_collapsed = False
        reasons = []

        # 连续参数：spread_ratio < 0.4 视为塌陷
        if not is_discrete and not np.isnan(spread_ratio) and spread_ratio < 0.4:
            is_collapsed = True
            reasons.append(f"spread_ratio={spread_ratio:.2f} < 0.40 (range too narrow)")

        # KDE 熵低（分布尖锐）
        if not np.isnan(kde_entropy) and kde_entropy < 0.15 and not is_discrete:
            is_collapsed = True
            reasons.append(f"kde_entropy={kde_entropy:.3f} < 0.15 (distribution peaked)")

        # 平坦维度（无信息）
        is_flat = False
        if landscape_var_ratio < 0.05:
            is_flat = True
            reasons.append(f"landscape_var_ratio={landscape_var_ratio:.3f} < 0.05 (param has no effect)")

        diagnostics[name] = {
            "n_samples": len(values),
            "is_discrete": is_discrete,
            "spread_ratio": float(spread_ratio),
            "cv": float(cv),
            "kde_entropy": float(kde_entropy),
            "landscape_variance_ratio": float(landscape_var_ratio),
            "is_collapsed": is_collapsed,
            "is_flat": is_flat,
            "reasons": reasons,
        }

    # --- overall verdict ---
    any_collapsed = any(d["is_collapsed"] for d in diagnostics.values())
    all_flat = (len(diagnostics) > 0 and all(d["is_flat"] for d in diagnostics.values()))
    if all_flat:
        verdict = "flat_landscape"
    elif any_collapsed:
        verdict = "collapsed"
    else:
        verdict = "healthy"

    diagnostics["overall_verdict"] = verdict
    diagnostics["_param_cols"] = param_cols
    diagnostics["_target"] = target
    diagnostics["_n_trials"] = len(valid)

    return diagnostics


def _landscape_variance_ratio(param_values, target_values, n_bins=10):
    """
    估算某参数方向上响应曲面的方差占比。

    将参数分箱，计算每个箱内 target 均值，再衡量这些均值的变化幅度。
    返回 box_means 方差 / target 总方差。
    """
    param_values = np.asarray(param_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)

    if len(param_values) < 5:
        return float('nan')

    total_var = np.nanvar(target_values)
    if total_var < 1e-12:
        return 0.0

    # 自适应分箱数
    n_unique = len(np.unique(param_values))
    actual_bins = min(n_bins, n_unique)
    if actual_bins < 2:
        return 0.0

    try:
        # 离散参数直接 groupby
        if _is_discrete_param(param_values):
            unique_vals = np.unique(param_values)
            box_means = []
            for uv in unique_vals:
                mask = param_values == uv
                if mask.sum() > 0:
                    box_means.append(np.nanmean(target_values[mask]))
            box_means = np.array(box_means)
        else:
            bin_edges = np.linspace(param_values.min(), param_values.max(), actual_bins + 1)
            bin_indices = np.digitize(param_values, bin_edges[1:-1])
            box_means = []
            for bi in range(actual_bins):
                mask = bin_indices == bi
                if mask.sum() > 0:
                    box_means.append(np.nanmean(target_values[mask]))
            box_means = np.array(box_means)

        if len(box_means) < 2:
            return 0.0

        between_var = np.nanvar(box_means)
        return float(between_var / total_var)
    except Exception:
        return float('nan')


def print_collapse_report(diagnostics):
    """友好打印塌陷检测报告。"""
    verdict = diagnostics["overall_verdict"]
    tag_map = {"healthy": "[OK]", "collapsed": "[!]", "flat_landscape": "[--]"}
    print(f"\n{'='*60}")
    print(f"  {tag_map.get(verdict, '?')} Param Space Diagnosis: {verdict.upper()}")
    print(f"  Trial 数: {diagnostics['_n_trials']} | Target: {diagnostics['_target']}")
    print(f"{'='*60}")

    for name in diagnostics.get("_param_cols", []):
        if name.startswith("_"):
            continue
        d = diagnostics.get(name.replace("config/", ""))
        if d is None:
            continue
        status = "[!] collapsed" if d["is_collapsed"] else ("[--] flat" if d["is_flat"] else "[OK] normal")
        dtype = "discrete" if d["is_discrete"] else "continuous"
        print(f"\n  [{name}] ({dtype}, n={d['n_samples']}) → {status}")
        print(f"    spread_ratio      = {d['spread_ratio']:.3f}" +
              ("" if not np.isnan(d['spread_ratio']) else "  (无搜索边界)"))
        print(f"    cv                = {d['cv']:.3f}")
        print(f"    kde_entropy       = {d['kde_entropy']:.3f}")
        print(f"    landscape_var_ratio = {d['landscape_variance_ratio']:.3f}")
        if d["reasons"]:
            for r in d["reasons"]:
                print(f"    → {r}")
    print(f"\n{'='*60}\n")


# ============================================================================
# 改进版 3D 曲面
# ============================================================================

def plot_tune_landscape_3d(
    results_df,
    param_x,
    param_y,
    target="metrics_score",
    ax=None,
    method='cubic',
    grid_size=80,
    cmap='viridis',
    show_samples=True,
    show_best=True,
):
    """
    3D Meshgrid 参数响应曲面。

    Parameters
    ----------
    results_df : pd.DataFrame | pl.DataFrame
    param_x, param_y : str
    target : str
    ax : Axes3D | None
    method : str
        'linear' / 'cubic' / 'rbf'
    grid_size : int
    cmap : str
    show_samples : bool
    show_best : bool

    Returns
    -------
    (fig, ax, surf)
    """
    df = _to_pandas(results_df)
    col_x = param_x if param_x in df.columns else f"config/{param_x}"
    col_y = param_y if param_y in df.columns else f"config/{param_y}"
    if target not in df.columns:
        raise ValueError(f"target '{target}' not in columns")

    valid = df[np.isfinite(df[target])].copy()
    if len(valid) < 5:
        raise ValueError(f"有效样本不足 ({len(valid)} < 5)")

    x = valid[col_x].values
    y = valid[col_y].values
    z = valid[target].values

    grid_x, grid_y, grid_z = _interpolate_surface(x, y, z, grid_size, method)

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # 曲面
    vmin, vmax = np.nanmin(grid_z), np.nanmax(grid_z)
    surf = ax.plot_surface(
        grid_x, grid_y, grid_z, cmap=cmap,
        edgecolor='none', alpha=0.8, vmin=vmin, vmax=vmax,
        linewidth=0, antialiased=True,
    )

    # 采样点
    if show_samples:
        ax.scatter(x, y, z, color='black', s=20, zorder=5, label='Sampled Trials',
                   edgecolors='white', linewidths=0.4)

    # 最优点
    if show_best:
        best_idx = np.nanargmax(z)
        ax.scatter([x[best_idx]], [y[best_idx]], [z[best_idx]],
                   marker='*', s=300, color='red', edgecolors='black',
                   linewidths=1.0, zorder=6, label='Best')

    ax.set_xlabel(_clean_param_name(col_x))
    ax.set_ylabel(_clean_param_name(col_y))
    ax.set_zlabel(target)
    title_method = f"[{method}]" if method != 'cubic' else ""
    ax.set_title(f"3D Landscape: {target} {title_method}")
    ax.legend(loc='best', fontsize=8)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=target)
    return fig, ax, surf


# ============================================================================
# 综合仪表盘
# ============================================================================

def plot_collapse_dashboard(
    results_df,
    target="metrics_score",
    search_bounds=None,
    method='cubic',
    param_pair=None,
    cmap='viridis',
    figsize=(16, 12),
):
    """
    参数空间综合诊断仪表盘（2×3 多面板）。

    Layout:
        ┌───────────────┬───────────────┐
        │  Contour (2D) │  3D Landscape │
        ├───────────────┼───────────────┤
        │ 边缘分布直方图 (每个参数一行)    │
        ├───────────────┼───────────────┤
        │ 采样轨迹 (trial index vs value) │
        └───────────────┴───────────────┘

    Parameters
    ----------
    results_df : pd.DataFrame | pl.DataFrame
    target : str
    search_bounds : dict | None
    method : str
        插值方法
    param_pair : tuple(str, str) | None
        指定主 contour/3D 的两个参数；None 则自动选 fANOVA 最重要两维
        （用 landscape_variance_ratio 近似）
    cmap : str
    figsize : tuple

    Returns
    -------
    (fig, diagnostics)
    """
    df = _to_pandas(results_df)
    diagnostics = detect_space_collapse(df, target, search_bounds)

    param_cols = [_clean_param_name(c) for c in diagnostics["_param_cols"]]
    if len(param_cols) < 2:
        raise ValueError(f"至少需要 2 个参数，当前: {param_cols}")

    # 自动选择最重要的两个参数（landscape_variance_ratio 最高）
    if param_pair is None:
        scored = [(p, diagnostics[p]["landscape_variance_ratio"])
                  for p in param_cols if p in diagnostics and not p.startswith("_")]
        scored = [s for s in scored if not np.isnan(s[1])]
        scored.sort(key=lambda x: x[1], reverse=True)
        param_x, param_y = scored[0][0], scored[1][0]
    else:
        param_x, param_y = param_pair

    # 读取列名（补回 config/ 前缀）
    col_x = f"config/{param_x}"
    col_y = f"config/{param_y}"
    valid = df[np.isfinite(df[target])].copy()

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                  width_ratios=[1.2, 1.2, 1.0])

    # ---- Panel 1: Contour ----
    ax1 = fig.add_subplot(gs[0, 0])
    plot_tune_contour(df, param_x, param_y, target=target, ax=ax1,
                      method=method, show_samples=True, show_best=True, cmap=cmap)

    # ---- Panel 2: 3D Landscape ----
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_tune_landscape_3d(df, param_x, param_y, target=target, ax=ax2,
                           method=method, show_samples=True, show_best=True, cmap=cmap)

    # ---- Panel 3: 塌陷诊断文本 ----
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    _render_collapse_text(ax3, diagnostics, target)

    # ---- Panel 4: 边缘分布直方图 ----
    n_params = len(param_cols)
    n_rows_dist = (n_params + 2) // 3
    gs_dist = GridSpecFromSubplotSpec(n_rows_dist, 3, subplot_spec=gs[1, 0:2],
                                      hspace=0.4, wspace=0.3)
    for i, pname in enumerate(param_cols):
        r, c = divmod(i, 3)
        ax_d = fig.add_subplot(gs_dist[r, c])
        vals = valid[f"config/{pname}"].dropna().values
        is_disc = _is_discrete_param(vals)
        if is_disc:
            unique, counts = np.unique(vals, return_counts=True)
            ax_d.bar(unique.astype(str), counts, color='steelblue', edgecolor='black')
            ax_d.set_ylabel("Count")
        else:
            ax_d.hist(vals, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
            ax_d.set_ylabel("Frequency")
        ax_d.set_xlabel(pname)
        # 塌陷标注
        d = diagnostics.get(pname, {})
        tag = "[!]" if d.get("is_collapsed") else ("[--]" if d.get("is_flat") else "[OK]")
        ax_d.set_title(f"{tag} {pname}", fontsize=9)

    # ---- Panel 5: 采样轨迹（TPE 收窄检测） ----
    ax5 = fig.add_subplot(gs[1, 2])
    trial_idx = np.arange(len(valid))
    colors = plt.cm.tab10(np.linspace(0, 1, len(param_cols)))
    for i, pname in enumerate(param_cols):
        vals = valid[f"config/{pname}"].dropna().values
        # 归一化到 [0,1] 便于多参数叠加
        vmin, vmax = vals.min(), vals.max()
        normed = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        ax5.plot(trial_idx[:len(normed)], normed, '-o', markersize=2,
                 color=colors[i], label=pname, alpha=0.7)
    ax5.set_xlabel("Trial Index")
    ax5.set_ylabel("Normalized Value [0, 1]")
    ax5.set_title("Sampling Trajectory (TPE Narrowing)", fontsize=10)
    ax5.legend(fontsize=7, loc='best')
    ax5.grid(True, alpha=0.3)

    fig.suptitle(
        f"Param Space Dashboard | verdict={diagnostics['overall_verdict'].upper()} | "
        f"n_trials={diagnostics['_n_trials']}",
        fontsize=13, fontweight='bold', y=0.98
    )

    return fig, diagnostics


def _render_collapse_text(ax, diagnostics, target):
    """在 axes 上渲染塌陷诊断文本。"""
    ax.axis('off')
    verdict = diagnostics["overall_verdict"]
    tag_map = {"healthy": "[OK]", "collapsed": "[!]", "flat_landscape": "[--]"}
    verdict_text = f"{tag_map.get(verdict, '?')} {verdict.upper()}\n"
    verdict_text += f"Target: {target}\n"
    verdict_text += f"Trials: {diagnostics['_n_trials']}\n"
    verdict_text += "=" * 32 + "\n\n"

    param_cols = [_clean_param_name(c) for c in diagnostics["_param_cols"]]
    for name in param_cols:
        d = diagnostics.get(name)
        if d is None:
            continue
        status = "[!] collapsed" if d["is_collapsed"] else ("[--] flat" if d["is_flat"] else "[OK] normal")
        dtype = "discrete" if d["is_discrete"] else "continuous"
        verdict_text += f"[{name}] {dtype}\n"
        verdict_text += f"  → {status}\n"
        spread_str = f"{d['spread_ratio']:.2f}" if not np.isnan(d['spread_ratio']) else "N/A"
        verdict_text += f"  spread: {spread_str}  "
        verdict_text += f"entropy: {d['kde_entropy']:.2f}  "
        verdict_text += f"land_var: {d['landscape_variance_ratio']:.2f}\n\n"

    ax.text(0.05, 0.95, verdict_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


# ============================================================================
# 便捷入口：加载 Ray Tune 结果
# ============================================================================

def load_ray_results(experiment_path):
    """
    从 Ray Tune experiment 目录加载结果 DataFrame。

    Parameters
    ----------
    experiment_path : str
        Ray Tune experiment 目录路径，如 /tmp/ray_results/fsm_hpo_201002

    Returns
    -------
    pd.DataFrame
    """
    from ray.tune import ExperimentAnalysis
    analysis = ExperimentAnalysis(experiment_path)
    return analysis.dataframe()


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    # 合成 Ray Tune 风格的测试数据
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "trial_id": range(n),
        "config/downsample": np.random.choice([3, 4, 5], size=n),
        "config/motif_minutes": np.random.choice([30, 45, 60, 90], size=n),
        "config/threshold_r": np.random.uniform(0.6, 0.8, size=n),
    })
    # 模拟 metrics_score：threshold_r 有影响，downsample 弱影响，motif_minutes 平坦
    df["metrics_score"] = (
        10 * (df["config/threshold_r"] - 0.6) ** 2
        - 0.5 * (df["config/downsample"] - 4) ** 2
        + np.random.normal(0, 0.5, n)
    )
    df["u_pval"] = np.random.uniform(0, 0.5, n)
    df["trigger_count"] = np.random.randint(10, 100, n)
    df["valid_sample_ratio"] = np.random.uniform(0.3, 1.0, n)
    df["autocorr"] = np.random.uniform(-0.5, 0.5, n)

    print("=== 塌陷检测报告 ===")
    search_bounds = {
        "downsample": [3, 4, 5],
        "motif_minutes": [30, 45, 60, 90],
        "threshold_r": [0.6, 0.8],
    }
    diag = detect_space_collapse(df, "metrics_score", search_bounds)
    print_collapse_report(diag)

    print("=== 绘制综合仪表盘 ===")
    fig, _ = plot_collapse_dashboard(
        df, "metrics_score", search_bounds,
        param_pair=("threshold_r", "downsample"),
        method='cubic',
    )
    plt.savefig("/tmp/param_space_dashboard.png", dpi=120, bbox_inches='tight')
    print("已保存: /tmp/param_space_dashboard.png")
    plt.show()