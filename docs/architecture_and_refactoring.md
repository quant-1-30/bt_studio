# bt_studio 工业级架构重构与演进总结

> **文档索引** — [LLM 特征挖掘闭环](llm_agent.md) ·
> [FSM 策略分析](fsm_strategy_analysis.md) ·
> [可视化](visual.md) · [Agent 规范](AGENTS.md) ·
> [测试入口](../tests/run_all.py)
>
> 本篇记录全局架构与重构史;LLM 闭环的四层架构/ADR/修复明细见 `llm_agent.md`。
经过多轮彻底排查与深度重构,整个 `bt_studio`(量化研究 / HPO / 特征工程)与 `bt_core`(C++ 极速回测 / 撮合 / 风控)的底层逻辑完成了从“带病运行”到“工业级闭环”的蜕变。

本文档将全盘梳理当前的系统全貌,并详细记录我们在 Agent 算子沙箱、双阶梯评估机制以及性能内存解耦上的核心重构过程。

---

## 🏛️ 一、 整体流水线架构流图 (Pipeline Architecture)

重构后的流水线将数据预处理、因果守护、双阶梯特征筛选和 HPO 搜索解耦,形成了一条高度自动化、无未来函数的生产管线:

```text
 [原始高频 Tick / 1min Data] ──> [align_skeleton] (240 骨架爆破对齐 + 补全)
                                       │
                                       v
 [build_ofi] (混合方向 + 平方根法则 + 动态 MDP 多维合一 ──> Tanh 压缩 [-1, 1])
                                       │
                                       v
 [apply_regime_indicators] (MA20 顺势指标 ──> 注入 is_bull_regime 趋势掩码)
                                       │
                                       v
 [preconstruct_base_panel] (主进程 1 秒预计算:T+1 高频 Offset 目标价 + Median/MAD 截面中性化 + z_gap)
                                       │
                                       v
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 💥 AI 特征挖掘与 Agent 沙箱管线 (Two-Stage Harness)                                          │
│                                                                                             │
│  1. [Agent AST 生成]                                                                        │
│       │                                                                                     │
│       v                                                                                     │
│  2. [ast_compiler.py + causal_guard.py]                                                     │
│       (递归安全编译,动态阻断 shift(-k) 等未来函数,支持嵌套表达式,并计算 AST 唯一指纹)          │
│       │                                                                                     │
│       v                                                                                     │
│  3. [Stage 1: 极速物理粗筛 (mining_agent.py)]                                                │
│       (耗时 <0.001s:直接阻断死水 Std < 1e-6、大面积空值 > 20%、或白噪声 Autocorr < 0.35 的特征)  │
│       │ PASS                                                                                │
│       v                                                                                     │
│  4. [Stage 2: 单点评估与边际消融 (evaluator.py + mining_agent.py)]                           │
│       (耗时 ~0.5s:固定黄金超参进行 FSM 评估,通过 Ablation Test 计算边际贡献,剔除“搭便车”特征)    │
│       │ PASS                                                                                │
│       v                                                                                     │
│  5. [Stage 3: 触发全量生产级 HPO (tune_train.py)]                                            │
│       (仅针对黄金特征触发 400 次 Ray Tune 搜索,极大节约算力)                                     │
└──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                       │ 提取出最佳模型 best_config.pkl
                                       v
 [FSMPredictor] (纯正因果推理:动态 Markov 链前向传播 + ThreadPool C-DTW 多线程极速预测)
                                       │
                                       v
 [SCORE_DIR/scores_YYYYMM.parquet] ──> 传递给 [bt_core] 驱动策略进行实盘/回测撮合
```

---

## ✅ 二、 核心重构与痛点解决

本次迭代聚焦于解决 HPO 内存爆炸、Agent 嵌套解析崩溃以及特征评估效率低下的问题。具体完成了以下 5 个关键模块的重构:

### 1. HPO 性能与内存的终极解耦 (Panel 重构)
* **痛点**:在 400 次 Ray Trial 中,每个 Worker 重复计算 MA20、交易日桥接以及截面 Median/MAD,导致 CPU 被锁死,FAULTS 缺页中断极高。
* **重构**:将计算拆分为 `build_static_panel`(与超参无关的预计算)和 `extract_curves_from_panel`(与超参相关的降采样)。在启动 HPO 前,主进程只需 **1 次**预计算静态面板,所有 Worker 直接在内存/Plasma 中复用该面板,彻底解除了内存锁。

### 2. 挂载 Agent 算子沙箱 (AST Compiler & SafeOps)
* **痛点**:早期的 `safeops.py` 解析器由于只接受字符串列名(如 `_mean("close", 5)`),一旦遇到嵌套 AST(如 `mean(ref(close, 5), 10)`)就会触发 `TypeError` 崩溃。同时缺乏防死循环的保护。
* **重构**:
  * **类型宽容**:所有安全算子签名重构为接受 `Union[str, pl.Expr]`,通过 `_to_expr()` 在内部统一转换为 `pl.Expr` 实现**安全链式调用**。
  * **AST 编译器**:新增 `ast_compiler.py` 模块,支持最大递归深度限制(防恶意/缺陷 Agent 生成无限嵌套),并自动生成清晰的特征列名(如 `cs_zscore(mean(ofi,10))`)。
  * **因果阻断**:在编译期间无缝钩入 `causal_guard.check_causal`,任何带有 `shift(-k)` 或 `center=True` 的特征直接报错抛弃,把未来函数扼杀在编译期。

### 3. 双阶梯特征筛选与消融测试 (Two-Stage Harness)
* **痛点**:直接将 AI Agent 生成的任意特征送入长达 2 小时的 Ray Tune HPO 进行测试,效率极低且无法鉴别冗余特征(搭便车现象)。
* **重构**:新建 `mining_agent.py` 与 `evaluator.py` 实现了三阶段漏斗:
  * **Stage 1 (物理粗筛)**:执行低开销数值检查,秒杀常量序列、空值序列和白噪声。
  * **Stage 2 (单点评估与绝对质量门槛)**:利用预先设定的“黄金超参”(Golden Config: ds=4, motif=60, r=0.60),对通过初筛的特征执行独立且极速的 FSM 单点发现(仅耗时 ~0.5s)。
    * **逻辑修正与升级**:移除了早期容易导致特征内耗的伪消融(Ablation)测试(即以当前批次 `max_score` 作为 Baseline 导致大量优秀特征被误杀),改用 **绝对质量基准线 (Absolute Quality Baseline)**。
    * **晋级条件**:特征自身必须通过所有底层统计学检验 (`is_success == True`) 且其信息增益(BIC Score)必须能够 cover 掉模型复杂度的惩罚,即获得绝对正向 Alpha (`score > 0.0`)。只要达标,无论批次内其他特征多强,该特征都准许通行,彻底杜绝了优秀特征之间的相互踩踏。
  * **Stage 3 (生产 HPO)**:确认“真金不怕火炼”后,通过回调函数正式触发 `node_tune_monthly` 进行全量调优。

### 4. 跨窗口 WFO 全局诊断与防崩溃指标
* **痛点**:之前的评估打分中,容易因为某个窗口中特征得分的极值(Outlier)而产生漂移。
* **重构**:
  * 使用 **IQR 标尺** 替代方差来判定最坏邻居跌幅 (`worst_drop_ratio > 1.5xIQR`)。
  * 在 `wfo_diagnostics.py` 中引入 `Drift_Index` 和瀑布热力图对跨越多年的 WFO 窗口参数稳定性进行直观追踪。

### 5. 评价层与业务层的彻底解耦
* **痛点**:早期代码在底层特征计算中,如果遇到熔断或无流动性,就强行把收益改写为 0.0,污染了纯净的特征面板。
* **重构**:
  * 特征提取层(`panel.py`)只负责诚实记录标的截面涨跌幅与价格行为(`z_gap`)。
  * 打分惩罚层(`score.py`)负责根据 `z_gap` 判断模拟实盘中的执行率(`execution_rate`),并在对数似然度 `Log-Likelihood (ln_L)` 中进行基于执行率与胜率的双重降分惩罚。

---

## 🔁 六、 v2 迭代:闭环工程化(2025-08)

本轮聚焦 LLM 闭环的可运行性与工程治理(明细见 [llm_agent.md](llm_agent.md)):

1. **服务端解耦**:api_server 摆脱 PyQt 依赖(纯 Python `AsyncTaskManager`);Qt/iOS 客户端剥离至 `bt_studio_clients/` 独立仓库,经 REST/WS 对接。
2. **HPO 串行编排**:FIFO 队列 + 单 worker 线程独占 Ray 集群;脚本与服务端触发统一入队(`submit_hpo_task`)。
3. **评估扁平化**:Harness 编译一次 → 预过滤 → 幸存特征落盘(`FEATURE_DIR`) → 月度滚动×固定网格短路 → 聚合;窗口内特征级 ThreadPool 并行。
4. **数据零重复拉取**:Stage3 经 `precomputed_features` 复用 harness 落盘的 PIT parquet,tick 全链路仅拉一次。
5. **治理**:`paths.py` 统一目录/io_setup;`tool.py::select_train_months` 单一事实来源;tune.py 纯净为 WFO 流水线;arity `(min,max)` 区间注册表;`tests/run_all.py` 统一验证入口(3/3 PASS)。

## 🎯 结论

通过本次重构,`bt_studio` 的基础设施成功摆脱了脆弱性和计算冗余:

1. **极速化**:计算剥离了 pandas 慢速 apply,全面切换到 Polars 向量化及 Cython/Numba 并发;
2. **智能化**:Agent 生成的特征不仅能被安全编译(防崩溃、防未来函数),还能通过双阶梯漏斗极速验真,将特征调研周期从几小时缩短到几秒钟;
3. **健壮性**:WFO 全局诊断和多维 Pareto 打分机制确保了输出到 C++ 撮合引擎的策略配置兼具抗震性和物理逻辑基础。



## 七、 v3 迭代:核心/插件解耦(2026-08)

1. **依赖倒置修复**:`AsyncTaskManager` 与 `select_train_months` 上移至核心包 `bt_studio/orchestrator/`;`plugins/api_server` 中旧路径保留为 deprecation shim。脚本(`run_agent.py`)与 Stage3 回调改为依赖核心,不再反向依赖插件。
2. **collapse 自动校验**:检测逻辑自 `visual/collapse.py` 抽取为纯 numpy/scipy 的 `pipeline/diagnostics/collapse.py`;`node_tune_monthly` 在 trials df 生成后自动运行 `run_collapse_check`,报告落盘 `result/tune/collapse/collapse_{model_id}_{feature_col}.json`,verdict 并入 TaskResult;`visual` 仅保留绘图并 re-export。
3. **talib 白名单归位**:`factor_mining/ops/talib_ops.py` 迁至 `pipeline/agent/talib_ops.py`;`talib_bridge.py` 将白名单算子注册进 `SAFE_OPS`(44 ops);`ast_compiler` 编译期挂钩 `validate_node` 静态拦截冻结参数/范围/字段;`prompt._format_ops_help` 改为从注册表动态渲染,杜绝漂移。知识库 md 移至 `docs/priors/`。
4. **api_server 瘦身为常驻结果网关**:移除 HPO/inference 执行端点(410);新增 `watcher.py` 定时轮询 `tune/{models,scores,collapse}`、`llm/runs`、`features/` 五目录,新产物经 WS 推送 iOS/Qt;核心进程与网关以文件系统为唯一契约。
5. **依赖治理**:pyproject 显式声明 scipy;PyQt6 系移入 optional `clients` 组;`tests/test_architecture.py` 用 AST 扫描强制 `核心不得 import plugins` 的分层规则。


## v5 — XML DAG + Engine (2026-08-17)

- `wfo_pipeline` 及其 `mode` 参数删除:pipeline 结构从 Python 剥离为
  **XML 声明**(`bt_studio/dags/{wfo_production,wfo_hpo}.xml`),latest/full
  语义由属性表达(`window from="last-trainable"` + gate skip ⇒ 强制 tune)。
- 新增 `bt_studio/engine/`:`dag_parser`(XML→PipelineDag,stdlib)+
  `executor`(NODE_REGISTRY + `run_dag`:滚动窗口/门控分支/paths 复用/
  断点/ray 生命周期,编排自 wfo_pipeline 平移)。
- `tune.py` 回归纯节点库(7 个 node 函数)。
- `task_manager` 执行输入升级为 DAG 引用:`submit_dag_task(dag_ref, config)`
  主入口；`submit_hpo_task`/`submit_wfo_task` 为指向标准 DAG 的薄封装。
- llm-agent pending intake 落地(默认关闭):文件契约见 docs/llm_agent.md。
- 测试:`tests/test_dag.py`(parser/校验负例/executor 干跑/intake)。
