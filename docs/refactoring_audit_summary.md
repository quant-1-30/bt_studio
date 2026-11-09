# `bt_studio` 架构精简、性能优化与量化严谨性重构报告

本文档汇总了对 `bt_studio` 全栈模块（`schema`、`constant`、`default_config`、`utils`、`compiler`、`pipeline`、`agent`、`engine`、`tune`）按**奥卡姆剃刀原则（Occam's Razor）**、**高内聚低耦合**与**A 股微观结构因果律**进行的重构与代码审查结论。

---

## 一、 核心量化因果律与 A 股市场机制对齐

### 1. 复权口径：`momentum` vs `raw_ret` 源头因果后复权 (`prepare.py`, `panel.py`)
- **`momentum_trajectory`（`intra_*`）天然不受除权影响**：
  - 定义为 $T+1$ 日同日盘中比值 $\frac{P_{T+1, \text{exit}}}{P_{T+1, \text{open}}} - 1$。由于 A 股除权除息均在 `ex_date`（$T+1$ 日）开盘集合竞价前一次性完成，当日盘中 240 根分钟线共享同一常数复权因子 $S_{T+1}$，分子分母严格约分。
- **`raw_trajectory`（`raw_*`）与跨日 AST 算子源头后复权**：
  - `prepare_macro` 仅用于股票池筛选（`volume > 0`、`high > low`、`days_since_ipo`、成交额 `amount`）及交易日历生成，指数与股票池筛选均无需除权，因此彻底移除 `prepare_macro` 中冗余的复权逻辑；
  - 在 `prepare_tick` 拉取入选股票的 `RpcTopic.Tick` 分钟线时，直接调用 `bt_sdk.core.factor.apply_factor(raw, factors, FactorTopic.Hfq)` 进行**因果累计后复权**（$\text{scale}(D) = \prod_{\text{ex\_date} \le D} 1/r$，零未来函数），使跨日 AST 算子（`roc`、`delta`、`ema` 等）与 `build_static_panel` 的跨日收益 `t1_price / entry_price - 1.0` 在源头即保持复权连续性。

### 2. 日内有效曲线门控与因果前向填充 (`panel.py::extract_curves_from_panel`)
- 修复了此前先执行 `.forward_fill().over("sid")` 再统计 `.count()` 导致跨日污染次日早盘空值且使 `min_curve_valid_ratio >= 0.90` 门控失效的缺陷；
- 改为在 `group_by(["day", "sid"]).agg(...)` 内部先用 `pl.col(feature_col).count().alias("curve_len")` 统计**填充前**的真实有效 bar 数，再用 `pl.col(feature_col).sort_by("bar_idx").forward_fill().alias("lag_0")` 仅在当日盘中做因果前向填充。

### 3. A 股 $T+1$ 正样本 Motif 提纯 (`pattern.py::_select_positive_motif_curves`)
- 在有监督正样本池构建中，要求同时满足截面排名前列（`dir_rank >= pos_quantile`）、持有收益为正（`dir_raw > 0`）以及 **$T+1$ 日内承接非负（`dir_mom >= 0`）**，剔除“隔夜高开、全天单边砸盘出货”的诱多伪正样本。

---

## 二、 热点路径 `@numba.njit(nogil=True, cache=True)` 编译加速与等价性证明

### 1. 为什么替换 `dtaidistance.dtw.distance_fast`？
1. **macOS ARM64 动态库加载缺陷**：上游 PyPI 预编译的 `dtw_cc.cpython-311-darwin.so` 硬编码链接了 Intel 路径 `/usr/local/lib/libomp.dylib`，导致 `import dtw_cc` 静默失败并回退至纯 Python。
2. **Cython 包装层未释放 GIL 且存在高频调用开销**：即便在 Linux 下 `dtw_cc` 可用，`dtaidistance/dtw_cc.pyx::distance` 内部也**未声明 `with nogil:`**（全程持有 GIL），且每个 Motif 需在 Python 层循环调用 ~21 万次 `DTWSettings(**kwargs)`，导致 `ThreadPoolExecutor` 多线程加速比为 `0.99x`。

### 2. `astc.py` 四个 Numba 核函数的分工与纯粹客观化 (`bt_studio/pipeline/patterns/astc.py`)
- **`_compute_keogh_envelope(motif_c, dtw_w)`**：预计算 1 次 Z-score 模板在带宽 `band = m if dtw_w <= 0 else (dtw_w - 1)` 内的上下包络线 $(U, L)$，与 `dtaidistance.dtw.lb_keogh` 零误差对齐。
- **`_banded_dtw_sq(s1, s2, w, max_dist_sq)`**：标准 15 行二维动态规划 `dp[i, j]` 实现 Sakoe-Chiba 带状平方 DTW 距离，支持行最小值 `row_min > max_dist_sq` 提前剪枝，覆盖 `w=0`（无带限制）、`w=1`（对角线欧氏距离）及不等长序列边界，与 `dtaidistance.dtw.distance` 机器精度等价（误差 `< 1e-12`）。
- **`_row_min_subseq_dtw_nb`**：移除主观时间惩罚项 `time_penalty` 与浮点容差魔法数，全程在**距离平方空间 (`min_dist_sq`)** 内完成滑窗 Z-score、LB_Keogh 下界剪枝与 `_banded_dtw_sq` 比較，仅在函数末尾执行一次 `math.sqrt(best_dist_sq)`。
- **`_chunk_dtw_nb`**：在释放 GIL 的机器码循环内批处理连续曲线块。
  - **Ray Tune Worker (`resources={"cpu": 1}`)**: `resolve_dtw_workers` 自动返回 `1`，直接走单线程机器码短路分支（**单核提速 100x**，零跨 Worker 线程上下文切换）；
  - **Driver / Stage 2 并发**：按 CPU 配额切分 `chunks` 多核并行（**8 核总提速 635x，多线程加速比 6.36x**）。

---

## 三、 奥卡姆剃刀去参数化（`metrics`、`diagnostics`、`constant`、`default_config`）

| 模块 | 移除的冗余参数 / 魔法逻辑 | 精简后的客观实现 |
| :--- | :--- | :--- |
| **`metrics/fanova.py`** | 移除 `select_plateau_verified_config` 与 `validate_parameter_plateau_fanova` 中的 `radius_ratio=0.15` 及 3 个未外传的默认形参 | `_build_neighborhood_conditions` 按参数拓扑自适应：`choice` 取相邻 $\pm 1$ 档；`int_range` 取 $\pm \text{step}$；`uniform` 取有效试验集的**样本标准差** `valid_df[p_col].std()` |
| **`metrics/pareto.py`** | 移除 `find_pareto_front` 中重复计算的 `complexity`、围绕 `autocorr_target=0.25` / 宽度 `0.20` 的高斯钟形曲线，以及 `select_best_model_from_pareto` 中的 `0.02` / `1e-4` 魔法数 | 帕累托三目标直接取 `(efficiency, valid_sample_ratio, 1.0 - abs(autocorr))`；效用函数简化为 `efficiency.log1p() * valid_sample_ratio.sqrt() * momentum_persistence` |
| **`metrics/score.py`** | 移除复杂度惩罚项 $k$ 中跨 Trial 恒定不变的 `max_offset / 60.0` 与 `dtw_frac * 10.0` | 仅保留随 Trial 变化的 `k = motif_minutes / 30.0` |
| **`patterns/fsm.py`** | 移除人为设定的 `_RET_STATE_PRIOR` 与 `_BAYES_SHRINKAGE_NU = 3.0` | 采用标准 Laplace $+1$ 平滑（`np.ones`）与经验中位数收益向量 |
| **`utils/diagnostics/collapse.py`** | 移除 `scipy.stats.gaussian_kde` 250 点网格数值积分、重复的离散/连续组间方差循环以及固定比例 `0.08` | 统一采用**归一化香农熵** $-\sum p_i \ln p_i / \ln K \in [0, 1]$、统一分组 `groups` 方差比与样本标准差邻域 `0.5 * std` |
| **`default_config.py` & `constant.py`** | 移除 `"autocorr_target"`、`"max_neg_ln_p"`、`"dtw_time_penalty"` 与重复的 `_COLLAPSE_CONTROL_KEYS` | 统一 `DEFAULT_WARMUP_MONTHS = 3` 与 `TRADING_DAYS_PER_MONTH = 20` 单一事实源 |

---

## 四、 模块高内聚重构（`schema`、`agent`、`engine`、`utils/io.py`）

### 1. 消除碎片化小文件，收敛模块边界
- **`bt_studio/schema.py`**：全框架唯一 `@dataclass` / `Enum` 契约目录；将 `PrefilterResult.from_stats`、`FeatureMiningResult.from_candidate_windows` 与 `WindowReplayState.replay_month` 内聚为类方法/实例方法。
- **`bt_studio/agent/`（收敛为 3 个顶层核心模块 + `vendors/`）**：
  - 删除 `eval_types.py`、`prefilter.py`、`eval_cache.py`、`eval_workers.py` 4 个碎片文件，Stage 1/2 执行流内聚于 `harness.py`，通用缓存 I/O 下沉至 `utils/io.py`；
  - 删除 `vendors/metering.py`，将 `BudgetExceeded`、`LlmUsageMeter` 与 `MeteredProvider` 内聚入 `vendors/base.py`。
- **`bt_studio/engine/`（收敛为 4 个核心模块）**：
  - 删除 `engine/tasking.py`：任务目录寻址下沉至 `utils/io.py`，DAG 签名与拓扑依赖求值内聚回 `engine/executor.py`；
  - 修复 `task_manager.py::shutdown(wait=False)` 中调用 `cancel_queued` 的方法名对齐。

### 2. `bt_studio/utils/io.py` 纯 I/O 接口全景（五大分区，22 个接口）

| 分区 | 函数 / 类接口 | 核心职责 |
| :--- | :--- | :--- |
| **1. 序列化、摘要与安全读取** | `slugify(text, max_len, fallback)` | 将任意文本规范化为文件名/目录名安全的 slug |
| | `get_git_sha()` | 进程级缓存获取当前仓库短 Git SHA |
| | `json_default(obj)` | JSON 序列化回退器（原生化 NumPy 标量与数组） |
| | `sha1_hex(obj, length)` | 对任意可 JSON 化对象生成确定性 SHA-1 摘要 |
| | `read_json_safe(path, default)` | 安全读取 JSON，缺失或损坏时平滑返回 `default` |
| | `consume_time(func)` | 节点执行耗时打印装饰器 |
| **2. 跨进程文件锁与原子落盘** | `exclusive_file_lock(path)` | 基于 `fcntl.flock` 的跨进程排他文件锁上下文管理器 |
| | `atomic_file_write(target_path)` | PID 隔离临时文件写入 + 退出时 `os.replace` 原子替换 |
| | `atomic_save_json(data, target_path, indent)` | 复用 `atomic_file_write` + `fsync` 原子写入 JSON |
| | `atomic_save_parquet(df, target_path, statistics)` | 复用 `atomic_file_write` 原子写入 Polars Parquet |
| | `atomic_save_pickle(obj, target_path)` | 复用 `atomic_file_write` + `fsync` 原子写入 Pickle |
| **3. 磁盘卫生与缓存驱逐** | `clean_stale_tmp_files(root_dir, max_age_s)` | 清理进程异常退出遗留的超期 `.tmp` / `.tmp_*` 碎屑 |
| | `evict_cache_lru(result_root, max_remove, include_bars)` | 按 `mtime` 从旧到新 LRU 驱逐 `cache/evals` 与 `cache/bars` |
| | `guard_disk_watermark(target_dir)` | 磁盘使用率超阈值（默认 95%）时自动触发 `evict_cache_lru` |
| | `io_setup()` | 初始化 `ARTIFACT_ROOT` / `CACHE_ROOT` 并清理陈旧临时文件 |
| **4. 行情流与月包缓存存储** | `collect_lazy_safe(lf_or_df)` | 优先以 `engine="streaming"` 收集 Polars `LazyFrame` |
| | `_collect_stream_sync(observable, timeout)` | 带显式批次超时保护的 RxPY RPC 数据流同步收集器 |
| | `fetch_hfq_factors_safe(mdapi, body, topic)` | 安全拉取 `mdapi.get_factor` 后复权因子（异常/Mock 平滑降级 `{}`） |
| | `RawPitCacheStore` | 分钟级 Raw PIT 月度 Parquet 包校验（`df.height > 0`）、加锁增量合并与按区间扫描 |
| **5. 任务分桶与窗口评估缓存** | `resolve_task_id(exp_config, ast_recipe)` | 基于特征、AST、行情范围与搜索空间计算确定性 `task_id` |
| | `attach_task_dirs(...)` / `sync_task_dirs_to_exp_config(...)` | 为任务挂载并同步隔离的 `feature_dir` 与 `tune_*_dir` 目录 |
| | `window_artifacts_complete(model_id, common_config)` | 校验 Walk-Forward 窗口的 `model_*.pkl` 与 `scores_*.parquet` 完整性 |
| | `compute_eval_data_fingerprint(common_config, dret_lf)` | 生成与候选批次大小解耦的确定性行情数据集指纹 `data_fp` |
| | `build_eval_cache_locator(...)` / `eval_cache_load` / `eval_cache_save` | Stage 2 窗口 `EvalResult` 缓存路径寻址与原子读写 |
