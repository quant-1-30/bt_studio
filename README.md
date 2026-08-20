# bt_studio 项目指南

## 1. 项目概述

`bt_studio` 是一个基于 Python 的量化研究与回测工作室，专注于从中国 A 股市场数据中发现日内与个股的预测性模式。项目目前演进至 v5 架构，采用创新的**双环路解耦设计**，核心包含两大引擎：

1. **自动化特征智能体 (LLM Agent)**：基于 LLM 与 AST（抽象语法树）动态编译，能够自动化构建、推演与筛选 TA-Lib 和动量特征算子，形成 RL 迭代飞轮。
2. **声明式执行引擎 (XML DAG Engine)**：舍弃了重型的第三方工作流工具，采用原生的 XML DAG 声明式配置，驱动全生命周期的 Walk-Forward 滚动优化（包含超参塌陷检测、模型衰减门控、Ray Tune 并发寻优）。

## 2. 技术栈与关键依赖

| 层级 | 库 / 工具 |
|------|----------|
| 语言 | Python `>=3.11,<3.15` |
| 包管理 | Poetry |
| 核心数据结构 | Polars, NumPy, PyArrow |
| 统计与机器学习 | `stumpy` (矩阵轮廓), `dtaidistance` (DTW), `scipy`, `scikit-learn` |
| 编排与执行引擎 | 标准库 `xml.etree` (DAG 解析), 原生 `threading` (队列排队调度) |
| 分布式超参优化 | Ray (`ray[tune]`), Optuna (`OptunaSearch`) |
| 追踪与监控 | MLflow 3, `prometheus-client` |
| 智能体与编译 | OpenAI API, 动态 AST 解析与 C 扩展桥接 (TA-Lib) |
| 可视化与网关 | FastAPI, WebSockets, Streamlit, Bokeh 3, Matplotlib |
| 行情数据 RPC | `bt-sdk`, ZMQ, `uvloop` |

> *注：本项目已在 v5 架构中彻底移除了对 Prefect 的依赖，大幅降低了系统的复杂度和故障点。*

## 3. 仓库结构

```text
bt_studio/
├── dags/                  # XML DAG 声明目录
│   ├── wfo_production.xml # 生产全量 Walk-Forward 编排图
│   └── wfo_hpo.xml        # 单窗口强制 HPO 筛选图 (Agent 评估专用)
├── engine/                # 声明式 DAG 执行引擎
│   ├── dag_parser.py      # XML 解析与语法静态校验
│   └── executor.py        # 滚动窗口迭代、门控执行与 Ray 生命周期管理
├── orchestrator/          # 任务调度与队列管理
│   └── task_manager.py    # 异步 FIFO 队列，负责 API Server 与 Agent 落盘任务对接
├── pipeline/              # 核心算法与流水线节点 (Node 库)
│   ├── agent/             # LLM 智能体: Prompt 构建、AST 编译、TA-Lib 算子门控
│   ├── features/          # 基础特征工程 (如 OFI 生成)
│   ├── patterns/          # 基于 Stumpy 的模式发现与 FSM (有限状态机) 构建
│   ├── preprocess/        # 宏观数据准备与 Tick 数据采样对齐
│   ├── diagnostics/       # 超参空间塌陷检测与 HPO 校验
│   ├── metrics/           # Pareto 前沿与 fANOVA 重要性评估
│   └── inference/         # 样本外 (OOS) 预测与推断
├── plugins/               # 独立插件与外部网关
│   ├── api_server/        # FastAPI / WebSocket 服务端
│   └── xtp_client/        # ZMQ 交易客户端底层
├── utils/                 # 工具类 (路径管理、月份对齐等)
├── visual/                # 数据可视化 (Streamlit App 与 Bokeh 组件)
├── constant.py            # 全局路径及常量注册
├── default_config.py      # 全局生产参数基线 (Single Source of Truth)
└── tune.py                # 供 Engine 调用的纯节点库 (7 大核心 Node)

scripts/                   # 启动与运行入口
├── run_agent.py           # 启动 LLM 自动化特征挖掘飞轮
├── run_tune.py            # 启动生产环境全量 WFO 跑批
├── run_streamlit.py       # 启动可视化看板
└── start_env.sh           # 初始化运行环境变量

tests/                     # 全量测试套件
├── test_dag.py            # DAG 解析、干跑与 intake 断言
├── test_agent_ast.py      # AST 动态编译与静态校验
├── test_arity.py          # TA-Lib 与算子白名单检查
├── test_collapse.py       # 超参塌陷算法回归测试
└── run_all.py             # 快速执行所有本地测试
```

## 4. 运行与测试命令

### 4.1 安装依赖
```bash
poetry install
```

### 4.2 本地全量测试验证 (Fast 模式)
该命令执行 AST 算子验证、架构整洁度验证、DAG 引擎解析等所有不依赖外部数据库的单元测试：
```bash
poetry run python tests/run_all.py fast
```

### 4.3 启动流水线与智能体
- **运行全生命周期生产跑批 (WFO Production)**：
  ```bash
  poetry run python scripts/run_tune.py
  ```
- **启动 LLM 特征挖掘智能体 (Agent Loop)**：
  该智能体将自行与大模型交互，生成、校验 AST 表达式，并将脱颖而出的候选特征落盘至 `pending/` 目录供 Orchestrator 消费。
  ```bash
  poetry run python scripts/run_agent.py
  ```

### 4.4 基础设施与网关启动
- **依赖服务 (MLflow 等)**：
  ```bash
  docker-compose up -d
  ```
- **启动可视化大屏 (Streamlit)**：
  ```bash
  poetry run python scripts/run_streamlit.py
  ```

## 5. 架构与执行流设计

在 v5 架构下，`bt_studio` 实现了解耦的**双环路机制**：

### 🔄 环路 1：LLM Agent (特征飞轮)
1. **生成与推演**：Agent 通过交互生成全新的指标组合公式，并转换为内部的 AST (抽象语法树)。
2. **静态校验**：通过 `safeops` 及 `talib_validate_hook` 进行算子白名单与因果门控验证。
3. **入库对接**：经过基本验证的“获胜”特征被打包为标准的 JSON，落盘至 `result/llm/pending`（File Contract 机制）。Agent 自主继续下一轮探索。

### ⚙️ 环路 2：Orchestrator & DAG Engine (执行引擎)
1. **统一入口**：`AsyncTaskManager` 按顺序从外部 API 或是 Agent 的 `pending` 目录吸纳任务。
2. **DAG 解析**：解析指定的 XML DAG (如 `wfo_production.xml`)，应用全局基线参数。
3. **滚动迭代 (Walk-Forward)**：
   - `node_prepare_macro`: 锁定当月股票池。
   - `node_extract_feature_monthly`: 并发调取 Tick 级数据合成因子。
   - `node_check_decay_monthly`: (门控点) 通过 Mann-Whitney U 检验判定上月模型是否衰减。
   - `node_tune_monthly` / `node_update_fsm_matrix`: 根据衰减门控结果，决定是启动 Ray Tune 并发调优，还是直接继承转移矩阵。
   - `node_oos_inference_monthly`: 生成最终 Out-Of-Sample 得分。

## 6. 开发约定

- **参数透传**：基线参数在 `default_config.py` 中集中定义，一经实例化，全局不可变透传。
- **纯粹的 DAG 设计**：所有的流程控制 (如 Skip、Gate、Branch) 只在 XML 属性上定义，业务节点代码保持纯函数式 (如 `tune.py` 中)。
- **隔离守护**：架构边界由 `test_architecture.py` 静态审查 (确保 Plugins 不能反向侵入 Core，Agent 环与 Engine 环互不阻塞)。