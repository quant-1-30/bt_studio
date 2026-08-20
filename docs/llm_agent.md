# LLM 特征挖掘闭环架构(llm_agent)

> 本文档记录 LLM 特征挖掘闭环的完整架构设计、关键决策与修复方案。
> 模块位置:`bt_studio/pipeline/agent/`;触发入口:`scripts/run_agent.py`。

## 1. 四层闭环架构

```
┌─────────────────────────────────────────────────────────────────┐
│ 决策层  RLFeatureFlywheel (llm_loop.py)                          │
│   prompt(system+RL context) → LLM → JSON parse → reward → buffer │
└───────────────┬─────────────────────────────────────────────────┘
                │ candidate_asts (AST dict / Recipe list)
┌───────────────▼─────────────────────────────────────────────────┐
│ 评估层  TwoStageAgentHarness (mining_agent.py)                    │
│   编译一次(compile_ast/compile_recipe, causal+depth 守卫)          │
│   → Stage1 物理预过滤 (std/null-ratio)                            │
│   → Stage1.5 幸存特征落盘 (FEATURE_DIR 月度 PIT parquet)          │
│   → Stage2 月度滚动窗口 × 固定离散网格 (短路:任一组合过=窗口过)     │
│   → 聚合 (pass-ratio ≥ min_window_pass_ratio, 默认0.6)           │
└───────────────┬─────────────────────────────────────────────────┘
                │ stage3_callback(feature, recipe, cfg, persisted_paths)
┌───────────────▼─────────────────────────────────────────────────┐
│ 触发层  AsyncTaskManager (plugins/api_server/task_manager.py)     │
│   FIFO 串行队列 + 单 worker 线程 = Ray 集群独占                    │
│   预计算 parquet 存在时跳过特征提取 (precomputed_features)         │
└───────────────┬─────────────────────────────────────────────────┘
                │ node_tune_monthly (Ray remote, exclusive)
┌───────────────▼─────────────────────────────────────────────────┐
│ 执行层  tune.py::wfo_pipeline / node_tune_monthly                 │
│   Ray Tune (Optuna TPE) + fANOVA plateau + Pareto 选模 + 落盘     │
└─────────────────────────────────────────────────────────────────┘
```

**数据流**:LLM JSON → AST 编译(一次) → 宽表 → 预过滤 → 落盘复用 → 滚动评估(特征级并行) → 胜者入队 → HPO → model_*.pkl。

## 2. 关键架构决策(ADR 摘要)

| # | 决策 | 理由 |
|---|------|------|
| 1 | 时间窗口由系统控制,不暴露给 LLM | 杜绝 LLM 选"好年份"刷分的周期过拟合 |
| 2 | Stage2 判定 = 统计检验(`is_success`),非 score 绝对值 | 分钟级收益尺度下 `log(ret)` 使绝对 score 不可比 |
| 3 | 滚动窗口(月度) + `min_window_pass_ratio`(0.6) | 因子须跨周期稳健,防单段过拟合 |
| 4 | 固定离散网格 `DEFAULT_GRID` + 短路 | 去除"人为选黄金参数"偏差;短路省算力 |
| 5 | Stage3 全部经 AsyncTaskManager 串行队列 | 单集群资源下 tune 任务必须独占编排 |
| 6 | 特征落盘复用 `hf_{feat}_{ym}.parquet` | tick 数据只拉一次,与生产 PIT 缓存同约定 |
| 7 | 窗口内特征级 ThreadPool 并行 | Polars collect 释放 GIL;`eval_workers` 可配 |
| 8 | arity 采用 `(min, max)` 区间元组 | 消除魔法数字;talib 变参 `(2, None)` 统一表达 |
| 9 | 通用 `talib` 桥接算子覆盖 160+ C 函数 | `pl.struct().map_batches()` 保持惰性链 |
| 10 | Prompt 双模式范例(简单 AST / Recipe let-binding) | 兼顾动能类与 OFI 类复杂多步逻辑 |

## 3. 组件职责与文件索引

| 组件 | 文件 | 职责 |
|------|------|------|
| Prompt | `agent/prompt.py` | system prompt(算子表+双模式范例)+ RL 上下文注入 |
| AST 编译 | `agent/ast_compiler.py` | JSON→Polars Expr;causal/depth 守卫;arity 区间校验;Recipe let-binding |
| 安全算子 | `agent/safeops.py` | 34 个 Polars 原生算子 + talib 通用桥接(`(min,max)` 注册表) |
| 因果守卫 | `agent/causal_guard.py` | 拒绝 lookahead(负窗口 shift 等) |
| 评估器 | `agent/evaluator.py` | `evaluate_feature`(静态面板复用,cfg 注入) |
| 挖掘 Harness | `agent/mining_agent.py` | 扁平流水线:编译→预过滤→落盘→滚动→聚合→回调 |
| RL 飞轮 | `agent/llm_loop.py` | ReplayBuffer + step 闭环;奖励=stage2_passed |
| 串行队列 | `plugins/api_server/task_manager.py` | FIFO + 单 worker;Phase1 线程预处理 / Phase2 Ray HPO |
| 工具 | `plugins/api_server/tool.py` | `select_train_months`(训练窗口单一事实来源) |
| WFO 执行 | `tune.py` | node_prepare/extract/train_data/tune/update/oos 六节点 DAG |
| 路径治理 | `paths.py` | BASE/MODEL/FEATURE/SCORE_DIR + io_setup(env var 优先) |

## 4. 修复方案记录(本轮迭代)

### 4.1 正确性修复
- **score 判定谬误**:`score > FAILED_SCORE_PENALTY` → `is_success`(决策#2)。
- **static_lazy 未定义崩溃**:`build_static_panel` 失败补 `return None`。
- **month_id 计算**:`(day//100)`(Date 列必炸)→ `date.dt.year()*100+dt.month()`。
- **max_pval 硬编码** → `common_config.get("u_pval", 0.05)`。
- **dummy_llm return None**:悬空 return + 反引号损坏 JSON → 重写。
- **WFO 短窗口静默截断**:补 `assert len(train_yms)==TRAIN_WINDOW`。
- **tune.report 弃用** → `train.report`。
- **task_manager 空数据 IndexError**:`model_id is None` 保护。

### 4.2 架构/冗余治理
- api_server 与 Qt 解耦(纯 Python task_manager);客户端剥离至 `bt_studio_clients/`。
- `run_on_universe` 删除(数据编排在脚本层,harness 纯评估)。
- `run`/`run_on_universe` 嵌套消除 → 扁平单遍(编译一次)。
- 编译失败候选进结果集(rejected),修 zip 错位 + ReplayBuffer 盲区。
- `build_static_panel` 无关 `feature_col` 的歧义赋值删除。
- Stage3 双路径统一入串行队列;数据经 `precomputed_features` 复用(决策#5/#6)。
- `select_train_months` 抽至 `tool.py`;Dir/io_setup 抽至 `paths.py`;tune.py 仅剩流水线逻辑。
- 空文件清理(exceptions.py);.DS_Store/.env 入 .gitignore。

### 4.3 验证体系
- `tests/run_all.py` 统一入口(arity → ast → strategy;`fast` 跳过数据项)。
- `tests/test_arity.py`:注册表 (min,max) 合法性 + 编译 + 错误拒绝。
- 端到端实测:15 月度窗口全评估、聚合判定、ReplayBuffer 闭环(run_agent.py)。

## 5. 后续迭代方向(prompt 为环路入口)

1. **反馈注入升级**:`build_rl_context_prompt` 汇总失败四分类统计(compile/causal/stage1/stage2),LLM 规避系统性错误。
2. **算子表动态生成**:从 `SAFE_OPS`(含 arity 区间)自动渲染 prompt,杜绝漂移。
3. **JSON Schema 锁定**:structured-output 模型直接锁 schema,解析失败归零。
4. **真实 LLM 接入**:`llm_call_fn` 替换 dummy(当前 `scripts/run_agent.py` 为固定 OFI recipe)。
5. **文档联动**:本文件与 `architecture_and_refactoring.md` 保持同步更新。

## Pending 任务契约 (agent → task_manager, v5)

LLM agent 与 orchestrator 是**两条独立环路**,仅通过文件契约耦合:
agent 的 Stage3 winner 落盘为 `result/llm/pending/*.json`(fire-and-forget,
`LLM_PENDING_DIR` 常量),随后 agent 环路自行继续 RL 迭代。

```json
{
  "feature_col": "...",
  "ast_recipe": {},
  "common_params": {},
  "search_bounds": {},
  "precomputed_features": [],
  "dag": "wfo_hpo",
  "created_at": "ISO8601",
  "source": "llm_agent"
}
```

- `dag`:`bt_studio/dags/` 内名称(缺省 `wfo_hpo`)或 XML 绝对路径。
- 消费端:`AsyncTaskManager(llm_intake_dir=...)` 或环境变量
  `BT_STUDIO_LLM_INTAKE=1`(**默认关闭**——硬件资源 ready 前两环互不干扰)。
  开启后 intake 线程扫描 pending 目录 → `submit_dag_task(dag, ...)` 入队 →
  文件改名 `.consumed`(at-most-once)；坏 JSON 改名 `.invalid` 并记录日志。
- 扫描间隔:`BT_STUDIO_LLM_INTAKE_INTERVAL`(默认 5s)。
