# bt_studio 可视化模块文档

## 1. 架构总览

`bt_studio/visual/` 提供两大可视化能力:

```
bt_studio/visual/
├── vis.py          # 参数空间可视化与塌陷检测(matplotlib)
├── bkh/            # bt_core 回测结果可视化(Bokeh)
│   ├── plot.py     #   垂直联动多面板图(OHLCV + Indicators + Analyzers)
│   ├── scheme.py   #   配色方案与绘图参数
│   └── utils.py    #   数据加载(bt_core 长表 → 宽表 pivot)
└── app.py          # Streamlit 日志分析应用
```

### 数据流

```
=== 参数空间可视化 (vis.py) ===
Ray Tune / Optuna results_df (config/* + metrics_score)
        │
        ├──> detect_space_collapse()  → 诊断 dict
        ├──> plot_tune_contour()      → 2D 等高线
        ├──> plot_tune_landscape_3d() → 3D 曲面
        └──> plot_collapse_dashboard()→ 综合仪表盘

=== bt_core 回测可视化 (bkh/plot.py) ===
bt_core parquet 日志 (datetime, value, metrics)  [长表]
        │
        ├── load_and_align()  → pivot → 宽表 (datetime + 各指标列)
        │
        └── Plot.plot_from_btcore_log()
                ├── _plot_main()              → OHLCV 主图
                ├── _plot_indicators_stacked() → ind_* 垂直堆叠
                └── _plot_analyzers_stacked()  → Portfolio/drawDown 垂直堆叠
```

---

## 2. vis.py — 参数空间可视化与塌陷检测

### 2.1 核心函数

| 函数 | 功能 | 返回 |
|------|------|------|
| `detect_space_collapse(df, target, search_bounds)` | 塌陷检测引擎 | diagnostics dict |
| `print_collapse_report(diagnostics)` | 打印诊断报告 | None |
| `plot_tune_contour(df, param_x, param_y, target)` | 2D 等高线图 | `(fig, ax, contour_set)` |
| `plot_tune_landscape_3d(df, param_x, param_y, target)` | 3D 曲面 | `(fig, ax, surf)` |
| `plot_collapse_dashboard(df, target, search_bounds)` | 2x3 综合仪表盘 | `(fig, diagnostics)` |
| `load_ray_results(experiment_path)` | 加载 Ray Tune 结果 | pd.DataFrame |

### 2.2 塌陷检测原理

`detect_space_collapse()` 对每个参数维度计算 4 个诊断指标:

| 指标 | 计算方式 | 阈值 | 含义 |
|------|----------|------|------|
| `spread_ratio` | 实际探索范围 / 搜索空间范围 | < 0.40 | 探索范围过窄(TPE 过早收窄) |
| `cv` | std / mean | — | 变异系数,越小说明越集中 |
| `kde_entropy` | `exp(H - H_max)` 标准化微分熵 | < 0.15 | 采样分布尖锐(KDE 峰值) |
| `landscape_variance_ratio` | 分箱后 box_means 方差 / target 总方差 | < 0.05 | 参数对目标几乎无影响(平坦维度) |

**判定逻辑:**
- 任一连续参数 `spread_ratio < 0.4` 或 `kde_entropy < 0.15` → `collapsed`
- 所有参数 `landscape_variance_ratio < 0.05` → `flat_landscape`
- 否则 → `healthy`

### 2.3 使用示例

```python
from bt_studio.visual.vis import (
    detect_space_collapse, print_collapse_report, plot_collapse_dashboard
)

# Ray Tune 结果 DataFrame
df = results.get_dataframe()  # 含 config/downsample, config/threshold_r, metrics_score 等

search_bounds = {
    "downsample": [3, 4, 5],
    "motif_minutes": [30, 45, 60, 90],
    "threshold_r": [0.6, 0.8],
}

# 塌陷检测
diag = detect_space_collapse(df, "metrics_score", search_bounds)
print_collapse_report(diag)

# 综合仪表盘
fig, diag = plot_collapse_dashboard(
    df, "metrics_score", search_bounds,
    param_pair=("threshold_r", "downsample"),
    method='cubic',  # 'linear' / 'cubic' / 'rbf'
)
fig.savefig("dashboard.png", dpi=120, bbox_inches='tight')
```

### 2.4 CJK 字体配置

模块加载时自动探测并设置支持中文的 matplotlib 字体:

```python
# 优先级列表(自动选择第一个可用的)
['PingFang SC', 'Heiti SC', 'STHeiti', 'Songti SC',  # macOS
 'Arial Unicode MS',                                   # 跨平台
 'Noto Sans CJK SC', 'Source Han Sans SC',             # Linux
 'SimHei', 'Microsoft YaHei']                          # Windows
```

图表内的状态标记统一使用 ASCII(`[OK]` / `[!]` / `[--]`)替代 emoji,避免 matplotlib 渲染问题。

---

## 3. bkh/plot.py — bt_core 回测结果可视化

### 3.1 布局设计

垂直堆叠 + X 轴联动:

```
+------------------------------------------+
|  1. OHLCV Main (line/candle + volume)    |  <- 主图
+------------------------------------------+
|  2. Indicators (每个指标独立子图)          |  <- ind_* 列
|     +- ind_RSI                            |
|     +- ind_MACD                           |
+------------------------------------------+
|  3. Analyzers (每个分析器独立子图)         |  <- bt_core metrics
|     +- Portfolio                          |
|     +- drawDown                           |
+------------------------------------------+
```

### 3.2 联动机制

| 联动类型 | 实现方式 |
|----------|----------|
| **X 轴范围联动** | 所有子图 `x_range=self.fig_main.x_range`,缩放/平移任一图全部同步 |
| **十字线联动** | `CustomJS` hover callback → 所有子图的红色 `Span` 垂直线同步移动 |
| **Hover tooltip** | 各子图独立 `HoverTool`,datetime 格式化一致 |

### 3.3 Plot 类 API

```python
from bt_studio.visual.bkh import Plot

plot = Plot()

# 方法 1: bt_core parquet 日志(长表 → 自动 pivot → 可视化)
plot.plot_from_btcore_log(
    file_path="log_cerebro_0.parquet",
    candle=False,     # bt_core 日志通常无 OHLCV,自动降级为折线
    tick_unit='s',    # datetime 列时间单位
)

# 方法 2: 宽表 DataFrame(已 pivot)
plot.plot_from_wide_df(df, candle=True)

# 方法 3: 自适应(有 OHLCV 画 K 线,无则只画指标行)
plot.plot_from_integrated_df(df, candle=True)
```

### 3.4 bt_core 日志格式

bt_core 输出的 parquet 日志为长格式:

| 列 | 类型 | 说明 |
|----|------|------|
| `datetime` | int64 | Unix 时间戳(秒) |
| `value` | float64 | 指标值 |
| `metrics` | bytes/str | 指标名(如 `b'Portfolio'`, `b'drawDown'`) |

常见 metrics(26 种):
`AnnualReturn`, `BenchmarkDret`, `Calmar`, `Cash`, `CumReturn`, `DailyReturn`,
`DailyWinRate`, `MaxDrawdown`, `NetPnL`, `OrdersCnt`, `Pnl`, `Portfolio`,
`SharpeRatio`, `drawDown`, `drawDownLength`, ...

`load_and_align()` 将长表 pivot 为宽表(index=datetime, columns=metrics),并 ffill 对齐。

### 3.5 列分类约定

`plot_from_wide_df()` 自动将列分为 3 类:

| 类别 | 匹配规则 | 渲染位置 |
|------|----------|----------|
| OHLCV | `open, high, low, close, volume` | 主图(K 线/折线 + 成交量) |
| 指标 | `ind_*` 前缀 | 垂直堆叠子图区 |
| 分析器 | 其余非 OHLCV / 非 ind_ / 非 datetime | 垂直堆叠子图区 |

---

## 4. app.py — Streamlit 日志分析应用

```bash
streamlit run bt_studio/visual/app.py
```

已修复导入路径:`from bt_studio.visual.bkh import Plot`

功能:
- 侧边栏输入 parquet 日志路径
- DuckDB 查询合并 OHLCV + 订单信号
- 调用 `Plot().plot_from_integrated_df()` 渲染
- 展示原始数据 expander

---

## 5. 已知问题与注意事项

1. **CJK 字体**:`vis.py` 自动探测系统字体,但 Linux 服务器需安装 `fonts-noto-cjk`
2. **emoji**:matplotlib 无法渲染彩色 emoji,已统一用 ASCII 标记 `[OK]/[!]/[--]`
3. **bt_core 日志无 OHLCV**:`plot_from_btcore_log()` 自动降级为折线模式
4. **Bokeh 3.x 兼容**:`figure()` 的 `width` 参数在 Bokeh 3.x 中可直接使用
5. **RBF 插值**:`method='rbf'` 支持凸包外外推,但对退化几何可能失败(自动降级为 linear)