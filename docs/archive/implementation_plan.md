# Implementation Plan

## [Overview]

将 `wfo_pipeline` 的硬编码编排重构为「XML DAG 声明 + engine 解析编排」架构:pipeline 结构从 Python 代码中剥离为 XML 文件,新增 `bt_studio/engine/` 负责解析与执行,`wfo_pipeline` 的 `mode` 参数取消(由 XML 节点属性表达),task_manager 的执行输入升级为 DAG 文件路径。

当前 `wfo_pipeline(exp_config, ast_recipe, mode)` 把图结构、窗口迭代、分支语义全部硬编码在 Python 里:`mode` 参数造成同一函数两种人格的歧义(用户明确指出)；task_manager 用 mode 字符串区分执行路径是隐式契约；上一轮的 llm-agent pending 目录契约只完成了 agent 落盘半边(task_manager 的 intake sed 未生效,grep intake=0),消费端缺失。

重构后:tune.py 回归**纯节点库**(7 个 node 函数)；`bt_studio/engine/dag_parser.py` 将 XML 解析为图对象(节点 + 边 + 窗口/门控属性)；`bt_studio/engine/executor.py` 按声明驱动窗口迭代与分支执行(滚动循环、paths 复用、断点恢复等编排语义集中在 executor)；两份示例 DAG(`wfo_full.xml` / `wfo_latest.xml`)覆盖原 full/latest 两种行为——latest 语义由 XML 属性表达(`<window from="last-trainable"/>` + `<node id="decay" skip="true"/>`),engine 无 mode 概念。task_manager 执行层改为 `submit_dag_task(dag_path, config)`,`submit_hpo_task` 保留为 wfo_latest.xml 的便捷封装(run_agent pending JSON 兼容),并补齐 llm intake 消费端(pending JSON 可带 `dag` 字段)。

约束沿用既定纪律:config 入口归一化一次(`build_exp_config`)后纯透传；所有既有编辑用 write_to_file 全量重写(sed -i 长内容已知会卡)。

## [Types]

以数据类与 XML schema 为主,类型系统改动集中在 engine 包。

### `DagNode`(新,`bt_studio/engine/dag_parser.py`)

```python
@dataclass
class DagNode:
    id: str                      # 节点标识, XML id 属性
    fn: str                      # tune.py 节点函数名, 如 "node_prepare_macro"
    per: Optional[str] = None    # 窗口切片角色: "train"|"oos"|"warmup"|"prev_oos" (None=窗口级单次)
    gate: bool = False           # True = 返回值作为分支门控 (decay 节点)
    branch: Optional[str] = None # 分支条件: "if-gate"|"if-not-gate" (None=无条件执行)
    skip: bool = False           # True = 该节点跳过 (latest 图中 decay/oos 用)
    retry: int = 0               # 预留: 失败重试次数 (本轮恒 0)
```

### `DagWindow`(新,同文件)

```python
@dataclass
class DagWindow:
    from_: str = "first"         # 窗口起点: "first"|"last-trainable"|"config:last_model_id"
    step_key: str = "oss_step"   # common_config 中步长键名
    train_key: str = "train_window"
```

### `PipelineDag`(新,同文件)

```python
@dataclass
class PipelineDag:
    name: str                    # XML <pipeline name="...">
    nodes: List[DagNode]         # 声明顺序即执行顺序 (线性图 + 门控分支, 无需通用拓扑排序)
    window: DagWindow
    source_path: str             # 解析来源 XML 路径 (诊断用)
```

### `EngineResult`(新,`bt_studio/engine/executor.py`)

```python
@dataclass
class EngineResult:
    dag_name: str
    windows_executed: int        # 实际执行的窗口数
    models_produced: List[int]   # 各窗口落盘的 model_id
    last_model_id: Optional[int] # 供 TaskResult 载荷与断点续跑
```

### XML Schema(契约,两份示例放 `bt_studio/dags/`)

`bt_studio/dags/wfo_full.xml`(原 mode="full" 语义):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<pipeline name="wfo_full">
  <window from="first" train-window-key="train_window" step-key="oss_step"/>
  <node id="macro"   fn="node_prepare_macro"/>
  <node id="extract" fn="node_extract_feature_monthly" per="train|oos|warmup"/>
  <node id="decay"   fn="node_check_decay_monthly" per="prev_oos" gate="true"/>
  <node id="train_data" fn="node_prepare_train_data" per="train" branch="if-gate"/>
  <node id="tune"    fn="node_tune_monthly" branch="if-gate"/>
  <node id="update"  fn="node_update_fsm_matrix" branch="if-not-gate"/>
  <node id="oos"     fn="node_oos_inference_monthly"/>
</pipeline>
```

`bt_studio/dags/wfo_latest.xml`(原 mode="latest" 语义):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<pipeline name="wfo_latest">
  <window from="last-trainable" train-window-key="train_window" step-key="oss_step"/>
  <node id="macro"   fn="node_prepare_macro"/>
  <node id="extract" fn="node_extract_feature_monthly" per="train|oos|warmup"/>
  <node id="decay"   fn="node_check_decay_monthly" per="prev_oos" skip="true"/>
  <node id="train_data" fn="node_prepare_train_data" per="train"/>
  <node id="tune"    fn="node_tune_monthly"/>
  <node id="update"  fn="node_update_fsm_matrix" skip="true"/>
  <node id="oos"     fn="node_oos_inference_monthly" skip="true"/>
</pipeline>
```

语义规则(parser 校验 + docs 记录):
- `per` 多值用 `|` 分隔；`per="prev_oos"` 的节点其输入路径由 executor 从 train 切片过滤(`paths_for_months`),不重复 extract。
- `gate=true` 节点若 `skip=true`,则门控恒为 True(强制走 tune 分支)——这正是 latest 语义的编码方式。
- `branch` 节点在门控不满足时整体跳过；`skip=true` 无条件跳过。

### Pending task JSON schema(补充 `dag` 字段,上轮已定其余键)

```json
{"feature_col": "...", "ast_recipe": {...}, "common_params": {...},
 "search_bounds": {...}, "precomputed_features": [...],
 "dag": "wfo_latest", "created_at": "...", "source": "llm_agent"}
```
`dag` 值为 `bt_studio/dags/` 下的文件名(不含扩展名)或绝对路径；缺省 `wfo_latest`。

## [Files]

**新建:**

- `bt_studio/engine/__init__.py` — 导出 `parse_dag, run_dag, PipelineDag, DagNode, DagWindow, EngineResult`。
- `bt_studio/engine/dag_parser.py` — XML → `PipelineDag`(`xml.etree.ElementTree`,stdlib 零新依赖)；`parse_dag(path_or_name) -> PipelineDag`(支持 `bt_studio/dags/` 内名称解析与绝对路径)；`validate_dag(dag)` 校验 fn 存在于 tune 节点注册表、gate/branch 配对合法、window 键存在于 common_config 基线。
- `bt_studio/engine/executor.py` — `NODE_REGISTRY`(7 节点名→函数映射)+ `run_dag(dag, exp_config, ast_recipe=None) -> EngineResult`:ray init/runtime_env、窗口迭代(按 `window.from_`)、节点分发(per 切片 / gate / branch / skip)、paths 复用、`last_model_id` 断点、ray shutdown。编排语义从现 `wfo_pipeline` 平移。
- `bt_studio/dags/wfo_full.xml` / `bt_studio/dags/wfo_latest.xml` — 上述 schema 的两份标准图。
- `tests/test_dag.py` — parser/校验/两图快照 + executor 干跑(mock 节点)测试。
- `bt_studio/dags/README.md` — schema 契约文档(节点属性语义 + 两示例)。

**修改:**

- `bt_studio/tune.py` — `wfo_pipeline` 函数体迁移至 executor 后**整体删除**(连同 mode 参数)；7 个 node 函数与 `trainable_fsm_worker` 保留不动；`run_collapse_check` 调用留在 `node_tune_monthly`(不变)。文件头 docstring 更新为「节点库」定位。
- `bt_studio/orchestrator/task_manager.py` — **全量重写**(write_to_file,规避 sed 卡死):执行层改 `run_dag(parse_dag(dag_ref), exp_config, ast_recipe)`；`submit_dag_task(dag_ref, config) -> str` 新主入口；`submit_hpo_task(config)` = `submit_dag_task("wfo_latest", config)` 便捷封装；`submit_wfo_task(config)` = `submit_dag_task("wfo_full", config)`；**补齐上轮 intake**:`__init__(llm_intake_dir=None)` + `BT_STUDIO_LLM_INTAKE=1` env 开关(默认 off),`_intake_loop` 线程消费 pending JSON(读取→解析(含 `dag` 字段)→入队→改名 `.consumed`,at-most-once)；`_execute_pipeline` 的 TaskResult 载荷带 `dag` 名与 `EngineResult.last_model_id`。
- `scripts/run_tune.py` — `wfo_pipeline(exp_config)` → `run_dag(parse_dag("wfo_full"), exp_config)`。
- `bt_studio/plugins/api_server/task_manager.py`(shim)— 确认 re-export 仍成立(无 wfo_pipeline 引用则零改动)。
- `docs/llm_agent.md` — 补「pending JSON 契约 + dag 字段 + intake 开关」小节。
- `docs/architecture_and_refactoring.md` — 追加 v5 小节(XML DAG + engine 架构)。
- `tests/test_architecture.py` — 无需改(engine 不 import plugins,规则自动覆盖)；确认即可。
- `tests/run_all.py` — TESTS 列表加 `("tests/test_dag.py", False)`。

**删除:**

- 无文件删除；`wfo_pipeline` 函数(见 tune.py 条目)为符号级移除,迁移策略即 executor 平移。

**不依赖 sed -i 做任何多行编辑**；所有修改用 write_to_file 全量重写或单行短 sed。

## [Functions]

**新函数:**

- `parse_dag(ref: str) -> PipelineDag` — `engine/dag_parser.py`；ref 为 dags 目录内名称(`"wfo_full"`)或绝对/相对路径；解析+校验+返回。
- `validate_dag(dag: PipelineDag) -> None` — 同上；抛 `DagValidationError`(新异常类,同文件):fn 不在注册表 / gate-branch 配对缺失 / window 键名非法。
- `list_dags() -> list[str]` — 同上；枚举 `bt_studio/dags/*.xml` 供 API/文档用。
- `run_dag(dag: PipelineDag, exp_config: dict, ast_recipe=None) -> EngineResult` — `engine/executor.py`；核心编排(详见 [Classes] 的执行算法)。
- `AsyncTaskManager.submit_dag_task(self, dag_ref: str, config: dict) -> str` — `orchestrator/task_manager.py`；归一化(`build_exp_config`)→ 记录 TaskResult(feature_col/dag 名)→ 入队 `(task_id, dag_path, exp_config, ast_recipe)`。
- `AsyncTaskManager._intake_loop(self) -> None` — 同上；循环:扫 `self._intake_dir` 中 `*.json`(忽略 `*.consumed`)→ 逐个解析入队(经 `submit_dag_task`,dag 取 JSON `dag` 字段缺省 wfo_latest)→ `os.replace` 改名 `{name}.consumed` → sleep `BT_STUDIO_LLM_INTAKE_INTERVAL`(默认 5s)。解析失败文件改名 `.invalid` 并 log,不中断线程。
- `AsyncTaskManager._execute_pipeline(self, task_id, dag_path, exp_config, ast_recipe)` — 重写:`dag = parse_dag(dag_path)` → `result = run_dag(dag, exp_config, ast_recipe)` → TaskResult(`result={"feature_col":..., "dag": dag.name, "model_id": result.last_model_id, "model_dir":..., "collapse":...}`,collapse 读取逻辑保留现内联实现)。

**修改函数:**

- `AsyncTaskManager._submit(self, kind, config)` — 改为携带 dag_ref 而非 mode 字符串:`dag_ref = "wfo_latest" if kind=="hpo" else "wfo_full"`；队列元组第 3 位从 mode 换成 dag_ref(解析延迟到执行时,submit 快速返回)。
- `AsyncTaskManager.__init__(self)` — 加 `llm_intake_dir: Optional[str]=None` 参数与 intake 线程启动(见 [Files])。

**移除函数:**

- `wfo_pipeline(exp_config, ast_recipe=None, mode="full")` — `bt_studio/tune.py`；其编排逻辑(窗口迭代/分支/paths 复用/断点/ray 生命周期)被 `engine/executor.run_dag` 等价平移取代。迁移:调用点仅剩 `scripts/run_tune.py`(同步改)与 task_manager(重写时移除)；`docs` 中引用随文档更新。`@consume_time` 装饰迁移到 `run_dag`。

## [Classes]

**新类:**

- `DagValidationError(Exception)` — `engine/dag_parser.py`；携带 XML 路径与具体原因。
- (数据类 `DagNode` / `DagWindow` / `PipelineDag` / `EngineResult` 见 [Types],均 `dataclass(frozen=True)` 除 EngineResult。)

**修改类:**

- `AsyncTaskManager`(`orchestrator/task_manager.py`)— 全量重写:队列元素 `(task_id, dag_ref, exp_config, ast_recipe)`；worker 执行 `_execute_pipeline`(run_dag 一行调用 + 记账)；新增 intake 线程与 `submit_dag_task`；`submit_hpo_task`/`submit_wfo_task` 变薄封装(各自指向标准 DAG 名)。类 docstring 更新为 DAG 驱动描述。

**无类删除。**

**executor 执行算法(run_dag,编排语义从 wfo_pipeline 平移):**
1. `io_setup()`；ray init(runtime_env 单线程变量组,原样平移)。
2. `global_data = node_prepare_macro(common)`；`yms = list_available_months(...)`。
3. 窗口序列:`from="first"` → `range(TRAIN_WINDOW, len(yms), STEP)`；`"last-trainable"` → `[len(yms)-TRAIN_WINDOW]`(不足则 warning+空结果返回)；`"config:last_model_id"` → 从该 model 对应月份起。
4. 每窗口:train/oos/warmup 月份推导(原逻辑)；遍历 dag.nodes 依声明顺序:
   - `skip` → 跳过(gate 节点 skip ⇒ 门控=True)。
   - `per` 按角色切片调用(extract 三角色、decay 用 `paths_for_months(train_paths, prev_oos_yms)`)。
   - `gate` 节点产出 bool；`branch` 节点按门控执行/跳过。
   - tune 成功后 `last_model_id = model_id`(分支 else 的 update 继承路径平移)。
5. `EngineResult` 汇总；ray shutdown。
节点 fn 分发经 `NODE_REGISTRY = {f.__name__: f for f in (node_prepare_macro, node_extract_feature_monthly, node_check_decay_monthly, node_prepare_train_data, node_tune_monthly, node_update_fsm_matrix, node_oos_inference_monthly)}`；每个节点签名差异由 executor 按节点 id 特化适配(保持既有调用形态,不要求节点签名统一——避免本轮改动波及 7 个节点)。

## [Dependencies]

- **零新增第三方包**:XML 用 stdlib `xml.etree.ElementTree`；engine 其余仅用存量依赖。
- 模块依赖方向:`engine → {tune(节点), utils.months, default_config?否, constant, utils.paths}`；`orchestrator → {engine, default_config, constant}`；`tune` 不 import engine(节点库保持底层)；`pipeline/agent` 与 `orchestrator` 依旧零互import(上轮已隔离,`test_architecture` 持续守护)。
- `pyproject.toml` 无改动。

## [Testing]

- `tests/test_dag.py`(新,standalone 风格与现有测试一致):
  1. **parser**:`parse_dag("wfo_full")` / `parse_dag("wfo_latest")` 成功；节点数=7；关键属性断言(latest 图 decay.skip=True、full 图 decay.gate=True、tune.branch="if-gate")。
  2. **校验负例**:未知 fn / gate 无配对 branch / 非法 window 键 → `DagValidationError`(临时 XML 写 tmp 目录)。
  3. **executor 干跑**:monkeypatch `NODE_REGISTRY` 为 fake 节点(记录调用序列),喂 15 个月合成 dret parquet + 最小 exp_config(num_workers=1、ray 不真启——fake run_dag 环境下以 `RAY_ADDRESS=local` 干跑或把 ray init/shutdown 也 mock),断言:full 图 4 窗口按序执行、decay→tune 分支路径、latest 图单窗口强制 tune、`EngineResult.last_model_id` 正确。
  4. **intake**:tmp pending 目录写两个 JSON(一个带 dag 字段一个缺省)→ 构造 `AsyncTaskManager(llm_intake_dir=tmp)`(不启动真实 worker——直接调 `_drain_intake_once()` 提取出的单次扫描方法,便于测试)→ 断言队列中 dag_ref 正确、文件改 `.consumed`；坏 JSON → `.invalid`。
- 存量回归:`python tests/run_all.py fast` 全绿(arity/architecture/collapse/ast + 新 test_dag)。
- 冒烟:`parse_dag` 两标准图导入即验；`scripts/run_tune.py` dry 语法检查；task_manager 双入口 submit 后队列元素形态断言。

## [Implementation Order]

1. **engine/parser**:`bt_studio/engine/{__init__,dag_parser}.py` + `dags/{wfo_full,wfo_latest}.xml` + `dags/README.md`；`tests/test_dag.py` 的 parser/校验段先行通过。
2. **engine/executor**:`executor.py`(NODE_REGISTRY + run_dag,编排自 wfo_pipeline 平移,`@consume_time` 迁移)；此时 tune.py 暂未删,双实现并存便于比对。
3. **切换调用方**:task_manager 全量重写(submit_dag_task 主入口 + 薄封装 + `_execute_pipeline`→run_dag + intake 线程与 `_drain_intake_once`)；`scripts/run_tune.py` 改 run_dag；run_agent pending JSON 增加 `dag` 字段(默认 wfo_latest,向后兼容)。
4. **tune.py 瘦身**:删除 `wfo_pipeline`(确认全库无引用后),文件头 docstring 更新为节点库定位。
5. **测试收尾**:`tests/test_dag.py` executor 干跑与 intake 段补齐；`tests/run_all.py` 注册新测试。
6. **文档**:`docs/llm_agent.md` 契约小节(pending JSON + dag 字段 + BT_STUDIO_LLM_INTAKE 开关)；`docs/architecture_and_refactoring.md` v5 小节。
7. **全量验证**:`python tests/run_all.py fast` 全绿；导入冒烟(engine/orchestrator/tune 双向无环)；手动核对 wfo_full.xml 节点序列与原 mode="full" 行为逐行等价(diff 比对 executor 与旧 wfo_pipeline 逻辑块)。