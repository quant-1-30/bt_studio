# 架构专题 — LLM 特征挖掘环路 / 可视化栈 / DAG 引擎决策

> 整合自原 `llm_agent.md`、`visual.md` 与 v5 DAG 引擎重构决策记录(2026-10-28)。
> 操作指南见 [AGENTS.md](AGENTS.md);生产运行手册见 [runbook.md](runbook.md);项目概览见仓库根 `README.md`。

---

# Part I. LLM 特征挖掘闭环 (bt_studio/agent/)

> 模块位置:`bt_studio/agent/`;触发入口:`scripts/run_agent.py`。

## 1. 四层闭环架构

```
┌─────────────────────────────────────────────────────────────────┐
│ 决策层  RLFeatureFlywheel (agent/flywheel.py)                     │
│   prompt(system+RL context) → LLM → JSON parse → reward → buffer │
│   LLM 供应商经 agent/vendors/ 适配 (openai_compatible/google/dummy)│
└───────────────┬─────────────────────────────────────────────────┘
                │ candidate_asts (AST dict / Recipe list)
┌───────────────▼─────────────────────────────────────────────────┐
│ 评估层  TwoStageAgentHarness (agent/harness.py)                   │
│   编译一次(compiler/ast.py compile_ast/compile_recipe,            │
│   causal+depth+跨族嵌套守卫) → Stage1 物理预过滤 (std/null-ratio)  │
│   → Stage1.5 幸存特征落盘 (FEATURE_DIR 月度 PIT parquet)          │
│   → Stage2 月度滚动窗口 × 固定离散网格 (短路:任一组合过=窗口过)     │
│   → 聚合 (pass-ratio ≥ min_window_pass_ratio, 默认0.6)           │
└───────────────┬─────────────────────────────────────────────────┘
                │ stage3_callback(feature, recipe, cfg)
┌───────────────▼─────────────────────────────────────────────────┐
│ 触发层  AsyncTaskManager (engine/task_manager.py)                 │
│   FIFO 串行队列 + 单 worker 线程 = Ray 集群独占                    │
│   预计算 parquet 命中时跳过特征提取 (agent 命名空间约定)          │
└───────────────┬─────────────────────────────────────────────────┘
                │ run_dag("wfo_hpo") → node_tune_monthly (Ray, exclusive)
┌───────────────▼─────────────────────────────────────────────────┐
│ 执行层  engine/executor.py + tune.py (7 节点)                     │
│   Ray Tune (Optuna TPE) + fANOVA plateau + Pareto 选模 + 落盘     │
└─────────────────────────────────────────────────────────────────┘
```

**数据流**:LLM JSON → AST 编译(一次) → 宽表 → 预过滤 → 落盘复用 → 滚动评估(进程池并行) → 胜者入队 → HPO → model_*.pkl。

## 2. 关键架构决策(ADR 摘要)

| # | 决策 | 理由 |
|---|------|------|
| 1 | 时间窗口由系统控制,不暴露给 LLM | 杜绝 LLM 选"好年份"刷分的周期过拟合 |
| 2 | Stage2 判定 = 统计检验(`is_success`),非 score 绝对值 | 分钟级收益尺度下 `log(ret)` 使绝对 score 不可比 |
| 3 | 滚动窗口(月度) + `min_window_pass_ratio`(0.6) | 因子须跨周期稳健,防单段过拟合 |
| 4 | 固定离散网格 `DEFAULT_GRID` + 短路 | 去除"人为选黄金参数"偏差;短路省算力 |
| 5 | Stage3 全部经 AsyncTaskManager 串行队列 | 单集群资源下 tune 任务必须独占编排 |
| 6 | 特征落盘复用 `hf_{feat}_{ym}.parquet` | tick 数据只拉一次,与生产 PIT 缓存同约定 |
| 7 | 滚动评估经进程池并行(`EVAL_PROCS`),窗口内串行 | stumpy/numba workqueue 非线程安全;进程隔离绕开 GIL |
| 8 | arity 采用 `(min, max)` 区间元组 | 消除魔法数字;talib 变参 `(2, None)` 统一表达 |
| 9 | 通用 `talib` 桥接算子覆盖 160+ C 函数 | `pl.struct().map_batches()` 保持惰性链 |
| 10 | Prompt 双模式范例(简单 AST / Recipe let-binding) | 兼顾动能类与 OFI 类复杂多步逻辑 |

## 3. 组件职责与文件索引

| 组件 | 文件 | 职责 |
|------|------|------|
| Prompt | `agent/prompt.py` | system prompt(算子表由 SAFE_OPS 注册表渲染+双模式范例)+ RL 上下文注入 |
| AST 编译 | `compiler/ast.py` | JSON→Polars Expr;causal/depth/跨族嵌套守卫;arity 区间校验;Recipe let-binding 与中间变量引用改写 |
| 安全算子 | `compiler/ops.py` | `SAFE_OPS` 注册表(arity+缺省窗口)+ `compiler/plugins/talib_ops.py` 桥接 |
| 统一契约 | `schema.py` | EvalResult/PrefilterResult/FeatureMiningResult/WindowReplayState/PipelineDag/EngineResult/TaskResult |
| 挖掘 Harness | `agent/harness.py` | TwoStageAgentHarness 扁平流水线:编译→Stage1 物理初筛→落盘→Stage2 滚动网格评估→聚合→Stage3 回调 |
| RL 飞轮 | `agent/flywheel.py` | ReplayBuffer + FlywheelRunStore + RLFeatureFlywheel 步进闭环 |
| LLM 供应商 | `agent/vendors/` | base.py (含 LlmUsageMeter/MeteredProvider) + openai_compatible/google/dummy 适配 |
| 串行队列 | `engine/task_manager.py` | FIFO + 单 worker;submit_dag_task 契约预检;LLM_PENDING_DIR intake |
| 月份/路径工具 | `utils/months.py` | `calc_warmup_start_date`/`retrieve_available_months`/`resolve_oos_paths_with_warmup`/`find_model_path`/`ym_range` 等 |
| WFO 执行 | `engine/executor.py` + `tune.py` | task_id 分桶 + 签名注入 + node_prepare_macro/extract/check_decay/prepare_train/tune/update_fsm/oos 七节点 |
| 路径与缓存 I/O | `constant.py` + `utils/io.py` | 全部路径常量、原子文件读写、RawPitCacheStore 与 Stage 2 窗口评估缓存读写 |

## 4. 后续迭代方向(prompt 为环路入口)

1. **反馈注入升级**:`build_rl_context_prompt` 汇总失败四分类统计(compile/causal/stage1/stage2),LLM 规避系统性错误。
2. ~~算子表动态生成~~ **已实现**:`agent/prompt.py::_render_layer_one` 从 `SAFE_OPS`
   (arity+缺省窗口)渲染,并由 `tests/agent/test_prompt_examples.py::test_layer_one_matches_registry` 锁定。
3. **JSON Schema 锁定**:structured-output 模型直接锁 schema,解析失败归零。
4. **真实 LLM 接入**:`agent/vendors/{openai_compatible,google,dummy}.py` 适配层已就绪;
   `scripts/run_agent.py` 由 `LLM_PROVIDER` 驱动真实 vendor,`AGENT_STAGE3=off` 可只做
   特征挖掘 + RL 筛选而不触发 tune/HPO 契约。

## 5. Pending 任务契约 (agent → task_manager)

LLM agent 与 orchestrator 是**两条独立环路**,仅通过文件契约耦合:
agent 的 Stage3 winner 落盘为 `result/artifacts/pending/*.json`(fire-and-forget,
`LLM_PENDING_DIR` 常量),随后 agent 环路自行继续 RL 迭代。

```json
{
  "feature_col": "f_{fp12}",
  "feature_id": "f_{fp12}",
  "display_name": "optional human/recipe name",
  "ast_recipe": {},
  "common_params": {},
  "search_bounds": {},
  "dag": "wfo_hpo",
  "created_at": "ISO8601",
  "source": "llm_agent"
}
```

- `feature_col` 采用稳定指纹身份 `f_{fp12}`;展示名在 `viable_features.json` 的 `display_name`。
- 调度/预算控制参数(`num_trials`/`num_workers`/`n_startup_trials`/`storage_path`)
  必须放 `common_params`;放入 `search_bounds/search_spec` 会被 `build_exp_config`
  直接拒绝(防止被当成搜索维度且不改 `TuneConfig.num_samples`)。
- 若只验证特征挖掘 + RL、暂不纳入 tune:用 `AGENT_STAGE3=off` 启动
  `scripts/run_agent.py`(Stage2 后不写 pending、不触发 HPO 契约);默认 `spool`
  仍保持 fire-and-forget 文件契约。
- `dag`:仅允许仓库根 `dags/` 内 basename(缺省 `wfo_hpo`,可带 `.xml`)。
  绝对路径/路径分隔符会被 intake/submit 直接拒绝并隔离为 `.invalid`。
- 消费端 `submit_dag_task` 入队前执行**契约预检**:ast_recipe 静态编译
  (`check_causal=True`)+ `f_{fp12}` 服务端 `ast_fingerprint` 重算核验(拦截损坏/篡改负载)。
- intake 开关:`AsyncTaskManager(llm_intake_dir=...)` 或环境变量
  `BT_STUDIO_LLM_INTAKE=1`(**默认关闭**——硬件资源 ready 前两环互不干扰)。
  开启后 intake 线程扫描 pending 目录 → `submit_dag_task(dag, ...)` 入队 →
  文件改名 `.consumed`(at-most-once);坏 JSON 改名 `.invalid` 并记录日志。
- 扫描间隔:`BT_STUDIO_LLM_INTAKE_INTERVAL`(默认 5s)。

## 6. 生产化契约 (2026-10-11)

### Stage2 语义: 廉价初筛, 非 OOS
- Stage2 = 入 tune 候选池的**筛选层** (u_pval≤0.1 + 5bp/10bp 经济门 + 60% 窗口),
  与 tune HPO 门控**口径独立、用途不同** (产生候选 vs 产生模型); 不称 OOS、不产模型。
- 唯一统计/经济裁决者是集中 tune 的 WFO。

### 断点续跑 (resume)
- run 身份 = sha1(标的池+区间+门控+网格+vendor); `RESUME=auto` 扫描未完成 run 续跑,
  指纹不匹配即新开 (改任何配置 = 断点失效)。
- 三层: state.json (步边界, buffer 全量) / pending_step.json (评价前暂存候选,
  崩溃不重烧 LLM) / EvalResult 窗口缓存 (仅重算缺失月)。

### 缓存体系
- bars (`result/cache/bars/v2/ym={YYYYMM}.parquet` v2 月包): **写时验证** (骨架自洽 ∧
  日线日历覆盖) / **读时存在性**; 非法数据进不了缓存, mid-batch 崩溃下轮精确补拉。
- evals (`result/cache/evals/v{N}/`): (data_fp, ast_fp, ym, grid_hash) → EvalResult;
  **键不含门控值 → 调阈值重放免费**; 评分/轨迹语义变更 bump 版本。

### Token 治理
- MeteredProvider (wrap, 零侵入): result/artifacts/usage/usage_YYYYMMDD.jsonl 全局记账 (跨 run 共享预算);
  LLM_DAILY_TOKEN_BUDGET 调用前熔断; 401/403 fail-fast 不重试。

### 交付物清单 (2026-10-11 A/B/C/D 批)

| 批 | 文件 | 内容 |
|---|---|---|
| B1 | `agent/vendors/metering.py` | MeteredProvider/LlmUsageMeter + `agent/vendors/__init__.py` 工厂包裹 |
| B2 | `pipeline/preprocess/prepare.py` | raw_pit cell 缓存 (v1 写时验证/读时存在性/单月×K 批) |
| C1 | `agent/flywheel.py` | run 目录/state.json/pending_step.json/指纹校验; `scripts/run_agent.py` RESUME=auto + data_fp/run_fp 双指纹 |
| D1 | `agent/eval_cache.py` | EVAL_LOGIC_VERSION + eval cache 键值契约 + 并行编排批前查/批后写 |
| 共用 | `utils/months.py` | `day_month_id_expr` dtype 分派 |
| 测试 | `tests/agent/test_{llm_metering,flywheel_resume,eval_cache}.py` + `tests/monitor/test_raw_pit_cache.py` | 17 个行为锁定 |

运维旋钮: `LLM_DAILY_TOKEN_BUDGET` / `TICK_BATCH_SIDS` / `RESUME=auto|off` /
`AGENT_SCREEN_MAX_PVAL` / `AGENT_SCREEN_MIN_WINDOW_PASS_RATIO` /
`RAW_PIT_MIN_COVERAGE` / `AGENT_STAGE1_TRIM_WARMUP_FRAC` / `decay_eval_months` /
`EVAL_LOGIC_VERSION`(代码常量) / `RAW_PIT_VERSION`(代码常量) /
生产务必显式设置 `BT_STUDIO_ENV=prod`、`BT_STUDIO_API_TOKEN` 与 `BT_STUDIO_CORS_ORIGINS`。

---

# Part II. 可视化栈 (bt_studio/visual/)

## 7. 架构总览

```
bt_studio/visual/
├── collapse.py     # 参数空间绘图(contour/3D/仪表盘, matplotlib)
├── bkh.py          # bt_core 回测结果可视化(Bokeh 垂直联动多面板图 + load_and_align)
└── _streamlit.py   # A/B/C 多通道在线可视化工作台(支持单屏切换与多屏 URL 路由联动)

bt_studio/plugins/api_server/
├── main.py         # FastAPI 网关入口 + /ws/{client_id}
├── monitor_api.py  # /api/v2/* (含 /screener 与 /focus 跨屏焦点同步) + /ws/v2 事件环广播
├── indexer.py      # MonitorIndex (SQLite WAL 索引 + screener 截面查询 + focus 状态持久化)
└── watcher.py      # ResultWatcher 增量目录监听

bt_studio/utils/diagnostics/
├── collapse.py     # detect_space_collapse 塌陷检测算法 + 报告构建/保存/渲染
└── hpo_check.py    # HPO 门控与报告自动落盘入口
```

> 完整的 A / B / C 多通道与多显示器实时联动设计详见 [visual_online_architecture.md](visual_online_architecture.md)。

### 数据流

```
=== 参数空间可视化 (visual/collapse.py + utils/diagnostics) ===
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
                ├── _plot_main()               → OHLCV 主图
                ├── _plot_indicators_stacked() → ind_* 垂直堆叠
                └── _plot_analyzers_stacked()  → Portfolio/drawDown 垂直堆叠
```

## 8. visual/collapse.py — 参数空间可视化与塌陷检测

### 8.1 核心函数

均经 `bt_studio/visual/__init__.py` 再导出:

| 函数 | 功能 | 返回 |
|------|------|------|
| `detect_space_collapse(df, target, search_bounds)` | 塌陷检测引擎 | diagnostics dict |
| `print_collapse_report(diagnostics)` | 打印诊断报告 | None |
| `plot_tune_contour(df, param_x, param_y, target)` | 2D 等高线图 | `(fig, ax, contour_set)` |
| `plot_tune_landscape_3d(df, param_x, param_y, target)` | 3D 曲面 | `(fig, ax, surf)` |
| `plot_collapse_dashboard(df, target, search_bounds)` | 2x3 综合仪表盘 | `(fig, diagnostics)` |
| `load_real_ray_results(experiment_path)` | 加载 Ray Tune 结果(兼容别名 `load_ray_results`,供 `scripts/run_collapse.py`) | pd.DataFrame |

### 8.2 塌陷检测原理

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

### 8.3 使用示例

```python
from bt_studio.visual import (
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

### 8.4 CJK 字体配置

模块加载时自动探测并设置支持中文的 matplotlib 字体:

```python
# 优先级列表(自动选择第一个可用的)
['PingFang SC', 'Heiti SC', 'STHeiti', 'Songti SC',  # macOS
 'Arial Unicode MS',                                   # 跨平台
 'Noto Sans CJK SC', 'Source Han Sans SC',             # Linux
 'SimHei', 'Microsoft YaHei']                          # Windows
```

图表内的状态标记统一使用 ASCII(`[OK]` / `[!]` / `[--]`)替代 emoji,避免 matplotlib 渲染问题。

## 9. bkh/plot.py — bt_core 回测结果可视化

### 9.1 布局设计

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

### 9.2 联动机制

| 联动类型 | 实现方式 |
|----------|----------|
| **X 轴范围联动** | 所有子图 `x_range=self.fig_main.x_range`,缩放/平移任一图全部同步 |
| **十字线联动** | `CustomJS` hover callback → 所有子图的红色 `Span` 垂直线同步移动 |
| **Hover tooltip** | 各子图独立 `HoverTool`,datetime 格式化一致 |

### 9.3 Plot 类 API

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

### 9.4 bt_core 日志格式

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

### 9.5 列分类约定

`plot_from_wide_df()` 自动将列分为 3 类:

| 类别 | 匹配规则 | 渲染位置 |
|------|----------|----------|
| OHLCV | `open, high, low, close, volume` | 主图(K 线/折线 + 成交量) |
| 指标 | `ind_*` 前缀 | 垂直堆叠子图区 |
| 分析器 | 其余非 OHLCV / 非 ind_ / 非 datetime | 垂直堆叠子图区 |

## 10. _streamlit.py — A / B / C 多通道在线量化工作台

```bash
poetry run python scripts/run_streamlit.py   # 启动 bt_studio/visual/_streamlit.py
```

支持 **单屏工作区切换** 与 **多显示器独立 URL 联动（`?view_mode=...`）**：
- **Screen A (`Research & HPO`)**：实时监控 RL 飞轮状态、Stage 2 幸存特征与 Ray Tune 3D 参数空间塌陷诊断；
- **Screen B (`K-line & motif`)**：跟随跨屏焦点 `focus.sid`，通过 `bkh.py::Plot` 渲染联动 OHLCV K 线、指标与分析器子图；
- **Screen C (`Market screener`)**：实时扫描最新 `scores_*.parquet` 展示全市场触发排行榜，支持**点击表格行一键向 Screen A/B 广播焦点标的**。
- 详细架构见 [visual_online_architecture.md](visual_online_architecture.md)。

## 12. 已知问题与注意事项

1. **CJK 字体**:`visual/collapse.py::_setup_cjk_font` 自动探测系统字体,但 Linux 服务器需安装 `fonts-noto-cjk`
2. **emoji**:matplotlib 无法渲染彩色 emoji,已统一用 ASCII 标记 `[OK]/[!]/[--]`
3. **bt_core 日志无 OHLCV**:`plot_from_btcore_log()` 自动降级为折线模式
4. **Bokeh 3.x 兼容**:`figure()` 的 `width` 参数在 Bokeh 3.x 中可直接使用
5. **RBF 插值**:`method='rbf'` 支持凸包外外推,但对退化几何可能失败(自动降级为 linear)

---

# Part III. DAG 引擎设计决策 (v5, 2026-08)

> 源自从硬编码 `wfo_pipeline` 演进到 v2 依赖驱动 DAG 引擎的重构决策记录,此处保留现行有效结论。

## 13. 终态架构

```
agent 环路: LLM → AST 编译/校验 → pending/*.json (fire-and-forget)
                                                      │ (BT_STUDIO_LLM_INTAKE=1 时)
engine 环路: AsyncTaskManager (唯一入口 submit_dag_task)
              ├─ ensure_ray (集群生命周期, 长驻)
              ├─ build_exp_config (严格归一化, 未知键 ValueError)
              ├─ parse_dag (deps 边 + 环/依赖校验)
              └─ 串行队列 → run_dag (拓扑序 + 条件依赖 + 签名注入)
```

## 14. 核心决策记录

| 决策 | 背景 | 结论 |
|------|------|------|
| `build_exp_config` 严格化 | flat 顶层键只并入 `common_params`,`search_bounds` 被静默遗漏;拼写错误静默失效 | 严格白名单 = `common_params`/`search_bounds` + 控制键(`run_name`/`precomputed_features`/`ast_recipe`);未知顶层键直接 ValueError(fail-fast) |
| `feature_col` 必填化 | `.get("feature_col", ...)` 让测试遗留值成为生产默认,特征与模型产物错配 | `ast_recipe` 非 None(agent 路径)时必填(KeyError);仅 legacy `build_ofi` 分支保留显式 `ofi_ratio` |
| Ray 生命周期上移 | `run_dag` 内 init/shutdown 每任务冷启动 Plasma,与串行队列语义冲突 | `engine/ctx.py::ensure_ray` 幂等 init + atexit shutdown;task_manager 构造时长驻持有集群,executor 去 ray 化 |
| 签名注入取代硬分发 | executor 按 node id `if/elif` 硬编码分发,新增节点必须改 executor | `inspect.signature` 按参数名从共享 ctx 自动注入(实现位于 `engine/tasking.py`);缺参 TypeError fail-fast;节点保持纯函数 |
| gate/skip/branch → 显式依赖边 | v1 控制流属性与依赖图语义割裂(skip 需 dead_roles 补救、gate 隐式规则) | 条件依赖 `deps="extract_train>train_paths?gate"`(gate False 级联跳过)、或依赖 `deps="tune|update"`、`output="gate"` 命名 ctx 键、`role` 纯窗口切片声明;**未声明的节点天然不运行**(`wfo_hpo.xml` 精简图) |
| 删除 `submit_hpo_task`/`submit_wfo_task` | 两个封装只是 DAG 名别名,冗余入口 | 唯一提交入口 `submit_dag_task(dag_ref, config)`;语义由 DAG 引用表达 |
| `last_model_id` 网格对齐续跑 | 简单 idx+1 会使续跑窗口边界与冷启动错位 → OOS 切分错误 | 从 last_model_id 对应月份起;续跑窗口必须与冷启动边界一致 |

## 15. 验证体系 (tests/engine/)

`tests/engine/` 按职责拆分:`test_parser.py`(依赖边/循环校验与负例 →
`DagValidationError`)、`test_executor.py`(拓扑执行/窗口切分干跑)、
`test_tasking.py`(task_id 分桶/签名分析)、`test_task_manager.py`(
intake at-most-once `.consumed`/`.invalid`、契约预检、load shedding)、
`test_resume_watermark.py`(断点恢复水位)。
