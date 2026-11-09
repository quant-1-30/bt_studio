# v5 架构重构结果与决策分析 (Refactor Notes)

本文档完整记录 bt_studio 从硬编码 `wfo_pipeline` 演进到 **v2 依赖驱动 DAG 引擎**的全过程：每轮反馈的问题定位、设计决策、实现结果与验证方式。作为后续维护与审计的依据。

---

## 1. 重构总览

| 阶段 | 核心变更 | 移除的债务 |
|------|----------|------------|
| v1 初版 | `wfo_pipeline(exp_config, ast_recipe, mode)` 硬编码编排 + gate/skip/branch XML 属性 | tune.py 中的 Python 控制流 |
| v2 终版 | **显式依赖边 (`deps`) + 拓扑执行 + 签名注入 + Ray 生命周期上移** | mode 参数歧义、gate/skip/branch 属性、executor 硬编码分发、per-task ray init/shutdown |

**终态架构**：
```
agent 环路: LLM → AST 编译/校验 → pending/*.json (fire-and-forget)
                                                      │ (BT_STUDIO_LLM_INTAKE=1 时)
engine 环路: AsyncTaskManager (唯一入口 submit_dag_task)
              ├─ ensure_ray (集群生命周期, 长驻)
              ├─ build_exp_config (严格归一化, 未知键 ValueError)
              ├─ parse_dag (deps 边 + 环/依赖校验)
              └─ 串行队列 → run_dag (Kahn 拓扑 + 条件依赖 + 签名注入)
```

---

## 2. 逐项反馈与决策记录

### 2.1 `build_exp_config` 严格化（设计缺陷修复）
**问题**：flat 顶层键的"便利形状"只并入 `common_params`（setdefault），`search_bounds` 被静默遗漏；且隐藏拼写错误（如 `feature_col` 顶层传入会静默失效）。
**决策**：删除便利形状。严格白名单 = `common_params` / `search_bounds` + 控制键（`run_name`/`precomputed_features`/`ast_recipe`）；**未知顶层键直接 ValueError** 并列出非法键名（fail-fast）。
**迁移**：`run_agent.py` pending JSON 的 `feature_col` 从顶层移入 `common_params`。

### 2.2 `feature_col` 必填化（测试默认值泄漏）
**问题**：`node_extract_feature_monthly` 的 `.get("feature_col", "ofi_ratio")` 让测试遗留值成为生产默认——特征落盘名与模型产物可能错配。
**决策**：`ast_recipe` 非 None（agent 特征路径）时 `common_config["feature_col"]` **必填**（KeyError）；仅 legacy `build_ofi` 分支保留显式 `ofi_ratio`（其输出硬编码）。`task_manager.submit_dag_task` 入队前 `["feature_col"]` 强校验。`PRODUCTION_COMMON_PARAMS` 不含该键（per-task 属性）。

### 2.3 Ray 生命周期上移 + 硬分发消除
**问题 A**：`run_dag` 内 `ray.init/shutdown` —— 每任务冷启动 Plasma、与串行队列语义冲突（长任务结束后集群销毁）。
**决策 A**：新增 `engine/ray_ctx.py::ensure_ray(num_cpus)`（幂等 init + 单线程 runtime_env + atexit shutdown）；**task_manager 构造时持有集群**（长驻进程），executor 彻底去 ray 化。

**问题 B**：executor 的 `if node.id == "extract" ... elif "tune"` 硬编码分发——新增节点必须改 executor，违反"图即声明"。
**决策 B**：**pytest-fixture 式签名注入**——`inspect.signature` 按参数名从共享 ctx 自动注入实参。XML `deps` 的 `upstream>ctx_key` 绑定 + 内置窗口变量（`train_yms/oos_yms/model_id/...`）共同构成 ctx。节点签名微调对齐：decay 内化 `paths_for_months` 过滤（`train_paths` 参数自派生 `prev_oos_paths`）、`oos_months→oos_yms`。

### 2.4 gate/skip/branch → 显式依赖边（DAG v2 核心）
**问题**：v1 的 gate/skip/branch 是"控制流属性"——与依赖图语义割裂：skip 需要专门的消费者剪枝逻辑（dead_roles）补救、gate 的"skip⇒True"是隐式规则、分支语义不体现在图结构上。
**决策**：**控制流即边的条件**：
- `deps="extract_train>train_paths?gate"`——条件依赖：gate 为 False 时该节点级联跳过（分支）
- `deps="tune|update"`——或依赖（oos 只需其一运行）
- `output="gate"`——decay 返回值写入 ctx（输出命名，非控制流）
- `role="train|oos|warmup|prev_oos"`——纯数据声明（窗口切片映射）
- **`wfo_hpo.xml` 成为真正精简图**：不声明 decay/update/oos/extract_oos——未声明的节点天然不运行、不拉取数据（取代 skip + dead_roles，两条特殊逻辑一并删除）

**gate 语义澄清**（曾造成疑虑）：decay 节点自身在 prev_model 缺失时返回 True（真实节点已如此实现）——冷启动走 tune 分支是**节点语义**而非 executor 特判。executor 保持纯依赖驱动。

### 2.5 删除 `submit_hpo_task` / `submit_wfo_task` 薄封装
**问题**：pipeline 架构下两个封装只是 DAG 名的别名——冗余入口。
**决策**：删除。唯一提交入口 `submit_dag_task(dag_ref, config)`；语义由 DAG 引用表达。全库 grep 确认封装无外部调用（api_server `/api/hpo` 已是 410；agent 走 pending JSON 的 `dag` 字段）——零迁移。

---

## 3. 关键设计权衡记录

| 决策 | 备选 | 取舍理由 |
|------|------|----------|
| 节点级依赖边（非全声明式 DSL） | 窗口迭代/分支全部 DSL 化 | 线性图+单二元分支用拓扑+条件边足够；窗口滚动/断点对齐/ray 属编排语义，留在 executor 避免解释器复杂度 |
| 签名注入（非统一节点协议） | 强制所有节点 `fn(ctx)` | 零侵入复用 7 个既有节点；参数名即契约，缺参 TypeError fail-fast |
| `ensure_ray` 幂等（非每任务 init） | 保持 per-task 生命周期 | 长驻集群 + Plasma 热数据；串行队列天然独占，无需重复建拆 |
| 严格 config 白名单 | 宽松透传 | 隐蔽失效 > 显式报错；拼写错误在入口即暴露 |
| `last_model_id` 网格对齐续跑 | 简单 idx+1 | 续跑窗口必须与冷启动边界一致，否则 OOS 切分错位（曾实际踩坑：断言 3 窗 vs 期望 2 窗暴露） |

## 4. 验证体系（tests/test_dag.py v2）
1. **parser**：两标准图属性断言（deps/role/output/精简图节点集）
2. **负例×5**：未知依赖 / 自环 / 双节点环 / 非法 role / 非法依赖项 → DagValidationError
3. **拓扑**：production 图全序校验（所有上游 idx < 下游）
4. **干跑**（fake 节点 + 15 合成月）：production 4 窗（冷启动 tune + 3 继承 + OR-dep oos×4）；hpo 1 窗（仅 train 切片提取）；resume 网格对齐（202107→恰 2 窗）；decay 内化 paths 复用断言
5. **config 严格性**：flat 键 ValueError / feature_col 缺失 KeyError
6. **intake**：dag 字段解析 / `.consumed`/`.invalid` at-most-once

回归基线：`python tests/run_all.py fast` → ALL PASSED（arity/architecture/collapse/ast/dag）。

## 5. 工程执行备忘
- **sed -i 多行编辑在本机反复卡死**——全部多行修改改用 write_to_file 全量重写或带 anchor 断言的 python 补丁脚本（幂等检查），单行短 sed 保留。
- 长命令输出重定向到文件再 tail（管道偶发 truncated）。
- venv：`/Users/hengxinliu/Library/Caches/pypoetry/virtualenvs/bt-studio-GsPwZx4h-py3.11/bin/python`