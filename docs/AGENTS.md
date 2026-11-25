# AGENTS.md — bt_studio 项目指南

> 面向编码智能体/新成员的**操作型**指南:怎么装、怎么测、怎么跑、改哪里。
> 项目概览与架构叙事见仓库根 `README.md`;架构专题(挖掘环路/可视化/DAG 引擎决策)见
> `docs/architecture.md`;文档索引见 `docs/README.md`。

---

## 1. 项目概述

`bt_studio` 是量化研究工作室(v5 双环路架构),聚焦 A 股日内/个股预测模式挖掘:

- **特征智能体环路**:LLM 产出因子 AST/Recipe → `compiler` 编译为 Polars 表达式 →
  `wfo_hpo` DAG 滚动检验 → 胜出特征经 `LLM_PENDING_DIR` 落盘契约 spool 给执行环路;
- **执行环路**:`dags/*.xml` 声明式 DAG → `engine` 滚动窗口执行 →
  Ray Tune + Optuna HPO(含塌陷检测/衰减门控)→ FSM 模型与 OOS 打分。

核心方法:有监督正样本矩阵轮廓(`stumpy`)发现 motif → Numba `nogil=True` 编译核(`astc.py`)执行带 LB_Keogh 剪枝的子序列 DTW 匹配 →
贝叶斯 FSM 转移网络打分;行情经 `bt_sdk`/`bt_protocol` RPC 获取。

---

## 2. 环境与常用命令

```bash
poetry install                      # 安装依赖 (Python >=3.11,<3.14)

# 前置初始化 (单线程确定性 + 硬件自适应并发 + gRPC 防代理劫持)
source scripts/start_env.sh

# 全量测试套件 (298 个用例: 293 单元 + 5 Ray Tune 全链路)
poetry run python -m pytest tests/ -q -p no:cacheprovider

# LLM 特征挖掘飞轮 (Walk-Forward 按需加载 + 两段式早停 + 断点续跑)
poetry run python scripts/run_agent.py                      # 常用参数: --steps N --vendor zhipu|kimi|google|dummy
                                                            #   --min-amount-rank 0.90 (每月因果成交额 Top 10%)
                                                            #   --train-window 12 --oos-step 6 --resume auto --profile
                                                            #   --stage3 off|spool (默认 off 仅写 viable_features.json)

# 生产 WFO / HPO 跑批 (wfo_production / wfo_hpo DAG, mlflow 默认落本地 sqlite)
poetry run python scripts/run_tune.py                       # 默认取 viable_features.json 首个特征
poetry run python scripts/run_tune.py --feature f_a1b2c3d4e5f6 --dag wfo_hpo \
    --start-date 20200101 --end-date 20221231               # 特征指纹/名 + DAG + 区间覆盖

# 回测 (bt_core; scores 目录三级回退: --score-dir > --task-id 桶 > 最新桶, 自动发现 sid/日期范围)
poetry run python scripts/run_backtest.py --task-id <task_id>

# 可视化统一入口 (四模式, 懒加载)
poetry run python scripts/run_visual.py --mode online       # Streamlit A/B/C 多屏工作台
poetry run python scripts/run_visual.py --mode collapse --exp <ray_results目录>   # 塌陷诊断 PNG
poetry run python scripts/run_visual.py --mode drift --exp <ray_results目录>      # WFO 漂移诊断 PNG
poetry run python scripts/run_visual.py --mode pdf --log <bt_core回测parquet>     # 回测 PDF 报告
```

基础设施(Postgres/MLflow/Prometheus/Grafana)可选,`docker-compose up -d`;
数据库密码经环境变量注入,compose 内默认值仅用于本地开发。

**测试确定性**:两个逐进程随机源都会沿「行序 → 曲线堆叠 → stumpy → 分数 →
门控」链路放大为离散翻转,新代码必须遵守:
- `tests/conftest.py` 在任何 polars/numba import 之前把
  `NUMBA_NUM_THREADS`/`POLARS_MAX_THREADS`/`OMP_NUM_THREADS` 钉为 1
  (浮点归约并行求和顺序);
- **不要用 `set` 决定顺序**:去重用 `dict.fromkeys`(保序),LazyFrame
  collect / group_by 之后如顺序影响下游,显式 `sort` 规范化
  (`PYTHONHASHSEED` 令 set 迭代序逐进程随机)。

---

## 3. 仓库结构

```
dags/                    # XML DAG 声明 (注意: 仓库根目录, 非 bt_studio/ 下)
├── wfo_production.xml   #   生产全量 Walk-Forward 编排
├── wfo_hpo.xml          #   单窗口强制 HPO 筛选 (Agent 评估专用)
└── README.md            #   DAG 语法 (v2: 显式依赖边) 说明

bt_studio/
├── agent/               # LLM 特征智能体 (高内聚 3 核心模块 + vendors)
│   ├── harness.py       #   TwoStageAgentHarness 编排 (Stage1 物理初筛 + Stage2 滚动窗口网格评估)
│   ├── flywheel.py      #   ReplayBuffer + FlywheelRunStore + RLFeatureFlywheel 经验回放循环
│   ├── prompt.py        #   系统提示词 (算子清单由 SAFE_OPS 注册表动态渲染)
│   └── vendors/         #   LLM 供应商适配 + base.py (含 LlmUsageMeter/MeteredProvider)
├── compiler/            # 因子 AST → Polars 表达式编译器
│   ├── ast.py           #   compile_ast/compile_recipe, 因果守卫, 跨族嵌套禁令
│   ├── ops.py           #   SAFE_OPS 注册表 (含 Numba 加速 _rank 等算子)
│   └── plugins/         #   TA-Lib C 底层白名单桥接
├── engine/              # 声明式 DAG 执行引擎 (高内聚 4 核心模块)
│   ├── parser.py        #   XML → networkx 图, 依赖边/循环校验
│   ├── executor.py      #   NODE_REGISTRY/run_dag 滚动窗口编排 + task_id 分桶 + 签名/条件依赖求值
│   ├── ctx.py           #   Ray 生命周期与平台自适应并发 (ensure_ray)
│   └── task_manager.py  #   异步 FIFO 队列 + 契约预检 + LLM_PENDING_DIR intake
├── trade/               # 实盘交易引擎 (scorer/data/sizers/strategy/execution/engine/drift)
├── monitor/             # 产物监控域 (indexer.MonitorIndex, server 与 visual 共享)
├── pipeline/            # 核心算法
│   ├── preprocess/      #   prepare/sample/panel/macro (Tick 因果后复权、曲线提取、静态面板)
│   ├── patterns/        #   astc/fsm/pattern (有监督正样本 Motif 发现 → Numba DTW → FSM)
│   ├── metrics/         #   score/gates/pareto/fanova (奥卡姆精简 HPO 评分与自适应 plateau 校验)
│   └── inference/       #   predict.py FSMPredictor (OOS 打分)
├── plugins/             # 零核心依赖可插拔: zmq_client (通用 ZMQ 传输) + xtp (XTP 行情)
├── server/              # FastAPI/WS 网关 (main/monitor_api/watcher)
├── utils/               # months/io/math_func/manifest + diagnostics/{collapse,hpo_check}
├── visual/              # _streamlit.py (A/B/C 多屏工作台), collapse.py 绘图, bkh.py Bokeh 组件
├── constant.py          # 全部路径常量与固定业务阈值 (RESULT_ROOT 可由 BT_STUDIO_RESULT_DIR 覆盖)
├── schema.py            # 全框架统一 dataclass/Enum 数据契约 (Dep/PipelineDag/EngineResult/TaskResult/EvalResult 等)
├── default_config.py    # 生产参数基线 (Single Source of Truth)
└── tune.py              # 7 大节点函数 (node_prepare_macro ... node_oos_inference_monthly)

scripts/                 # 运行入口 (4 个): run_agent.py / run_tune.py / run_backtest.py / run_visual.py + start_env.sh
tests/                   # pytest 套件: agent/compiler/engine/misc/monitor/pipeline/tune 七个子目录 + conftest.py 确定性环境
result/                  # 输出根目录, 按 task_id 分桶 (见 4.2)
```

---

## 4. 关键机制

### 4.1 配置与 DAG

- 配置单一来源 `bt_studio/default_config.py::build_exp_config(overrides)`;
  `scripts/run_tune.py` 只做覆盖与 `run_dag(parse_dag("wfo_production"), exp_config)`。
- 节点签名由 executor 内联的 `analyze_signature` 依 DAG 声明自动注入
  (`ctx_key` / 条件边 `?`/`?!`),节点函数本身是纯函数,注册于
  `engine/executor.py::NODE_REGISTRY`。
- Agent 落盘任务契约:Stage3 胜出特征写 `result/artifacts/pending/*.json`(含
  `"dag": "wfo_hpo"`),`BT_STUDIO_LLM_INTAKE=1` 启动时 task_manager 每 5s
  消费一批,处理后移 `.consumed`/`.invalid`。

### 4.2 结果目录按 task_id 分桶 (布局 v2: artifacts/ + cache/)

`engine/executor.py::attach_task_dirs` 按
`feature_col + ast_recipe + 日期/基准 + 产物语义 common 键 + search_spec`
的 sha1 派生 task_id,向 `common_config` setdefault 注入;task_manager 的队列
uuid 只用于状态跟踪,不写入 `common_params.task_id`,同实验重跑仍命中同一分桶:

```
result/artifacts/features/<task_id>/                 # 特征 parquet (feature_dir)
result/artifacts/tune/<task_id>/models|scores|collapse/   # _tune_*_dir
result/artifacts/runs/<run_id>/                      # agent 挖掘 run 状态与 step
result/cache/bars/v2/ym={YYYYMM}.parquet             # 分钟 PIT 月包缓存 (含 empty_sids 负缓存)
result/cache/evals/v3/                               # EvalResult 窗口缓存 + static/ 跨步静态面板缓存
```

无 task_id(缺 feature_col 等)时回落旧的扁平布局,行为与旧版一致;
agent 预计算特征经 `FEATURE_DIR/{fp12}/` 纯 12 位指纹目录约定复用(由 `ast_fingerprint(ast_recipe)[:12]` 推导)。

### 4.3 有监督 Motif 提纯、贝叶斯 FSM 与评分门控链路 (2026-11 重构)

`pattern.py::discover_fsm_pattern` → `fsm.py::evaluate_and_build_fsm` → `metrics/score.py`:

- **有监督正样本 Motif 提纯 (Stage A)**: 训练期内不再对全市场盲抽(`get_balanced_samples` 与 `max_points` 已退役)，而是通过 `_select_positive_motif_curves` 直接筛选 `rank_trajectory >= 0.70` 且 `raw_trajectory > 0` 的真实爆发正样本子集 $S^+$ 送入 STUMPY 提取候选 Motif $P(\text{Motif} \mid \text{Win})$，随后回到包含全部正负样本的全集 `panel_df` 上执行无偏 DTW 匹配与门控验证;
- **贝叶斯 Dirichlet 基率平滑与经验贝叶斯收缩 (`fsm.py::extract_fsm_matrix`)**:
  - 状态转移矩阵采用匹配 5 档分位宽度的 Dirichlet 先验 $\alpha_0 = [0.5, 1.0, 2.0, 1.0, 0.5]$，小样本下自动向真实横截面基率 $[10\%, 20\%, 40\%, 20\%, 10\%]$ 收缩;
  - 状态期望收益向量 `state_return_vectors` 采用经验贝叶斯收缩 $\hat{\mu}_s = \frac{n_s}{n_s + \nu_0} \text{median}(R_s)$（$\nu_0 = 3.0$），抑制稀疏分箱单样本极值噪音;
- **统计分与显著性检验**: `daily_paired_pval` 按日配对 Wilcoxon; p 值项软饱和 `cap·(1−e^(−x/cap))`(`max_neg_ln_p`, 默认 10);
- **经济门控与结局截断守卫**: `min_excess_ret` 硬门控原始收益中位数超额; `t1_missing_ratio <= 0.25` 拦截次日停牌/缺失造成的结局截断偏差;
- **衰减遥测与塌陷诊断**: `node_check_decay_monthly` 自动落盘 `decay_{model_id}_{feature_col}.json` 跟踪 OOS $u$-pval、触发率与得分漂移; `collapse.py` 单因子扰动邻域检测空间塌陷。
- **实盘日度漂移哨兵 (`trade/drift.py`)**: 月度衰减门控决定是否重训, 实盘另设日级哨兵 —— `TradeEngine.daily_plan` 打分一次, 把当日 DTW 触发数与 top-k conviction (`fsm_score` / `expected_return`, 即 `FSMSizer` 的加权输入) 对比本模型 OOS 打分 (`scores_{model_id}.parquet`) 的**中位数**: 相对偏移 `|v − median| / |median|` 超过 `drift_rel` 即 `model_drift_*` 告警 (`warn`; 不做窗口过滤、不设经验分位带)。日度记录落盘 `drift_{model_id}_{feature_col}.json` (重启续读, 同日重触发幂等), 由 `MonitorIndex.alerts()` 抬升并经 webhook 推送。只告警不改变交易行为; 基线缺失/损坏时自动禁用, 不阻塞交易进程。

### 4.4 编译器约束 (改动 prompt/agent 前必读)

- 算子唯一注册于 `compiler/ops.py::SAFE_OPS`(TA-Lib 桥接算子由插件注入),
  prompt 的算子清单**从注册表动态渲染**,不允许手抄名单;
- 单棵 AST 内**禁止跨族嵌套**:时序族(按 sid 分区)与截面族(按 day×bar
  分区)算子互相包裹会被 `_family_nesting_violation` 拒绝(polars 跨分区
  窗口嵌套会静默输出全 null);跨族组合唯一合法方式是 recipe 分步
  (时序在前、截面最后);
- `tests/agent/test_prompt_examples.py` 锁定 prompt 范例与编译器的一致性:
  改 prompt 范例或编译器约束,先跑该文件。

---

## 5. 开发约定

- Poetry 管理依赖;不手改 `poetry.lock`,用 `poetry add/lock`。
- 优先 Polars LazyFrame;`group_by().agg()` 内列表列必须 `sort_by` 保序。
- 配置走 `common_config`/`tune_config` 字典,路径一律读 `constant.py`,
  禁止在节点里硬编码输出目录(应 `common_config.get("_xxx", 模块常量)`)。
- 改 `pipeline/patterns/` 的 FSM 逻辑时同步检查 `pipeline/inference/predict.py`。
- 注释清晰一致，标识符英文。
- 涉及随机性的新增代码:显式 `np.random.RandomState(seed)`/`default_rng(seed)`,
  不要依赖全局 numpy 状态(测试套件已按可复现设计)。

---

## 6. 安全与敏感信息

- `.env` 是敏感文件:**不要提交、不要打印其内容**;
- `docker-compose.yml` 通过环境变量注入数据库密码，生产请勿使用弱口令;
- 生产 API 必须显式配置 `BT_STUDIO_API_TOKEN` 与 `BT_STUDIO_CORS_ORIGINS`(默认已收紧为 localhost);
  pending `dag` 仅允许 `DAGS_DIR` basename 白名单,绝对路径会被隔离为 `.invalid`;
- RPC 输入、Parquet 路径、反序列化对象仍需按零信任对待;result 卷生产建议防写/签名 manifest;
- 项目采用 GPL v3 许可证(见 `LICENSE`)。

---

## 7. 已知问题与历史修复记录

1. ~~`tests/tune/test_hpo_full_chain.py` 全量套件低概率抖动~~ **已修复 (2026-08)**。
   根因: `list(set(sids))` 的 set 迭代序受 `PYTHONHASHSEED` 逐进程随机化 →
   tick 帧 concat 顺序 → 特征 parquet 行序 → `prepare_curves` 按行序 vstack
   曲线 → stumpy 输入排列变化 → 同一 trial 分数逐进程漂移 (20.1~21.9),
   离散翻转被塌陷门控 (1.5×IQR) 边界放大为 `models_produced==[]` 级联失败。
   修复: 三处规范化——`tune.py` 与测试 mock 改用 `dict.fromkeys` 保序去重,
   `pipeline/patterns/pattern.py` 在 collect 后 `sort(["day","sid"])` 规范行序。
   修复后 collapse-first ×6 / 单文件 ×3 / 全量 ×2 全绿, 且 trial 分数跨进程
   bit-identical。

2. **`bt_core` `ParquetPatch` 的 `day_id` Int8 溢出 → 回测静默空仓** **已修复 (2026-11)**。
   根因: `pl.col("day").dt.month()` 返回 `Int8`, `* 100` 在 i8 内静默回绕
   (`2024-11-20 → 20240096`, `2025-12-31 → 20249951`, 仅 1 月正确),
   `global_scores` 的键随之错位, `get_topk(current_day)` 除 1 月外一律查空。
   旧用例只用了 1 月的日期, 故长期未暴露; 任何 `dt.month()/dt.day()` 参与乘法
   的表达式都必须先 `.cast(pl.Int32)` (同 `bt_sdk/core/factor.py` 的写法)。
   修复: 补 `.cast(pl.Int32)` + 跨 12 个月回归用例
   (`tests/behavior/test_feed_factor.py::test_parquet_score_feed_day_id_covers_all_months`)。
   **影响**: 在当前依赖 (polars ^1.39) 下, 修复前的回测只有 1 月的交易日拿得到信号,
   其余月份 `get_topk` 返回空 → 不开新仓; 此前跑出的回测结果需重跑核对。

## 8. 性能与资源 (2026-11)

- **RPC 与月包缓存 (`prepare.py` & `utils/io.py`)**: `start_env.sh` 与 `ctx.py` 默认导出 `GRPC_ENABLE_HTTP_PROXY=0` 与 `no_proxy=localhost,127.0.0.1` 防本地代理劫持 gRPC; `_collect_stream_sync` 由 `@_retry_rpc_stream(max_retries=3)` 装饰器托管 (`timeout=60.0s`); `prepare_tick` 按 `chunk` 批量向量化执行 `align_skeleton`, 并在 `ym.meta.json` 中记录 `empty_sids` 负缓存(纯停牌增量仅更新 20KB `meta.json`, 零拷贝跳过 120MB Parquet 重排)。
- **Walk-Forward 按需评估与两段式早停 (`harness.py` & `schema.py`)**: 按 `(train_window=12, oos_step=6)` 切分滚动窗口, `min_amount_rank=0.90` 因果选取每月成交额前 10% 标的(落盘保留 $\mathcal{U}_m \cup \mathcal{U}_{m-1}$ 防止月末 $T \rightarrow T+1$ 跨月出场价丢失); `WindowReplayState` 执行两段式早停(`consec_failures >= max_consec_failures(3)` 或 `failure_counts > max_allowed_failures`); `static_{w_tag}.parquet` 跨步持久化于 `EVAL_CACHE_ROOT/static/`, 100 步 RL 飞轮仅计算一次。
- **回测 (`run_backtest.py` & `sizers.py`)**: 从 `PRODUCTION_COMMON_PARAMS` 自动推导尾盘建仓时点 `entry_bar_idx - 240`（如 `-10m` 即 `14:50`）与次日早盘分批出场时点 `T1_rets`（如 `+5m/+15m/+30m`）及衰减分批比例 `exit_weights`；`node_oos_inference_monthly` 基于昨日收盘价 `preclose` 与 `entry_close` 硬过滤尾盘涨停不可买标的；`FSMSizer` 纯粹由 `fsm_score` 与 `expected_return` 驱动，与特征挖掘层彻底解耦。
- **实时行情缓存 (`plugins/xtp/quote_cache.py::TopkQuoteCache`)**: 零第三方依赖 C-double 连续数组 `(N_sids * 240)`，先握手订阅事件流再订当月 Top-K 标的，`O(1)` 增量聚合 240-bar 分钟骨架并实时应用 `hfq_scales`，实现 10,000+ 帧突发零丢帧。
- **CPU 与内存确定性**: `POLARS/NUMBA/OMP=1` 单线程确定性 + 防 Mach 风暴; 窗口内多特征或跨窗口进程池并行 (`eval_procs`), DTW 线程预算单一权威 `resolve_dtw_workers`。

