# bt_studio 项目指南

## 1. 项目概述

`bt_studio` 是一个基于 Python 的量化研究与回测工作室，专注于从中国 A 股市场数据中发现日内与个股的预测性模式。项目目前演进至 v5 架构，采用创新的**双环路解耦设计**，核心包含两大引擎：

1. **自动化特征智能体 (LLM Agent)**：基于 LLM 与 AST(抽象语法树)动态编译，能够自动化构建、推演与筛选 TA-Lib 和动量特征算子，形成 RL 迭代飞轮。
2. **声明式执行引擎 (XML DAG Engine)**：舍弃了重型的第三方工作流工具，采用原生的 XML DAG 声明式配置，驱动全生命周期的 Walk-Forward 滚动优化(包含超参塌陷检测、模型衰减门控、Ray Tune 并发寻优)。

## 2. 技术栈与关键依赖

| 层级 | 库 / 工具 |
|------|----------|
| 语言 | Python `>=3.11,<3.14` |
| 包管理 | Poetry |
| 核心数据结构 | Polars, NumPy, PyArrow |
| 统计与机器学习 | `stumpy` (矩阵轮廓), `numba` (无 GIL 子序列 DTW 编译核), `scipy`, `scikit-learn` |
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
├── dags/                  # XML DAG 声明目录 (注意位于仓库根，非 bt_studio/ 下)
│   ├── wfo_production.xml # 生产全量 Walk-Forward 编排图
│   └── wfo_hpo.xml        # 单窗口强制 HPO 筛选图 (Agent 评估专用)
├── agent/                 # LLM 智能体 (harness / flywheel / prompt / vendors)
├── compiler/              # 因子 AST → Polars 表达式编译器 (SAFE_OPS + TA-Lib 桥接)
├── engine/                # 声明式 DAG 执行引擎
│   ├── parser.py          # XML 解析与语法静态校验
│   ├── executor.py        # 滚动窗口迭代、门控执行与节点注册
│   ├── ctx.py             # Ray 生命周期 (ensure_ray)
│   └── task_manager.py    # 异步 FIFO 队列，负责 API Server 与 Agent 落盘任务对接
├── pipeline/              # 核心算法与流水线节点 (Node 库)
│   ├── patterns/          # 基于 Stumpy 的模式发现与 FSM (有限状态机) 构建
│   ├── preprocess/        # 宏观数据准备、Tick 对齐、静态面板与宏观状态计算 (macro.py)
│   ├── metrics/           # Pareto 前沿与 fANOVA 重要性评估
│   └── inference/         # 样本外 (OOS) 预测与推断
├── trade/                 # 实盘交易引擎 (打分/数据/仓位/执行/调度/漂移哨兵)
├── monitor/               # 产物监控域 (MonitorIndex, server 与 visual 共享)
├── plugins/               # 零核心依赖可插拔插件 (zmq_client + xtp)
│   ├── xtp/               # XTP 行情缓存 (TopkQuoteCache)
│   └── zmq_client/        # 通用 ZMQ 传输 (ZmqClient + codec)
├── server/                # FastAPI / WebSocket 服务端
├── utils/                 # 工具类 (原子 IO、月份对齐、数学统计、塌陷检测 diagnostics/)
├── visual/                # 数据可视化 (Streamlit App 与 Bokeh 组件)
├── constant.py            # 全局路径、版本号及常量注册
├── schema.py              # 全框架统一 dataclass / Enum 数据契约定义
├── default_config.py      # 全局生产参数基线 (Single Source of Truth)
└── tune.py                # 供 Engine 调用的纯节点库 (7 大核心 Node)

scripts/                   # 启动与运行入口 (功能级收敛)
├── start_env.sh           # 初始化运行环境变量 (单线程确定性 / 平台并发 / gRPC 防代理劫持)
├── run_agent.py           # 启动 LLM + RL 自动化特征挖掘飞轮 (Walk-Forward 按窗按需加载)
├── run_tune.py            # 启动 WFO / HPO 阶段跑批 (wfo_production / wfo_hpo)
├── run_visual.py          # 在线与离线可视化评测入口 (--mode online|collapse|drift|pdf)
└── run_backtest.py        # bt_core 组合回测引擎入口

tests/                     # pytest 套件 (engine/pipeline/tune/compiler/agent)
├── conftest.py            # NUMBA/POLARS/OMP 单线程 env, 保证 HPO 打分可复现
├── engine/                # DAG parser/executor/task_manager/tasking/resume 水位
├── pipeline/              # panel/pattern/metrics/inference/sampling/math_func
├── tune/                  # collapse/tune 节点/HPO full-chain (Ray 冒烟)
└── test_*.py              # compiler 因果/prompt 金丝雀/default_config/imports/strategy
```

## 4. 运行与测试命令

### 4.1 安装依赖
```bash
poetry install
```

### 4.2 本地全量测试验证
执行 AST 算子、DAG 引擎、pipeline/tune 节点与不依赖外部数据库的单元测试：
```bash
poetry run python -m pytest tests/ -q -p no:cacheprovider
```

### 4.3 启动流水线与智能体
- **前置环境初始化**：
  ```bash
  source scripts/start_env.sh
  ```
- **启动 LLM 特征挖掘智能体 (Agent Loop)**：
  该智能体按 `train_window + oos_step` 滚动窗口按需拉取 Tick 数据（默认 `min_amount_rank=0.90` 即每月因果成交额前 10% 活力池），通过两段式早停（`max_consec_failures=3` + 全局通过率门控）极速筛选 AST/Recipe，并将胜出特征写入 `viable_features.json`（及可选落盘至 `pending/` 目录）。
  ```bash
  poetry run python scripts/run_agent.py --steps 100 --vendor zhipu --resume auto
  ```
- **运行全生命周期生产跑批 (WFO Production / HPO)**：
  ```bash
  poetry run python scripts/run_tune.py --dag wfo_production
  ```
- **运行组合回测 (bt_core)**：
  ```bash
  poetry run python scripts/run_backtest.py
  ```

### 4.4 基础设施与网关启动
- **依赖服务 (MLflow 等)**：
  ```bash
  docker-compose up -d
  ```
- **启动可视化大屏 (Streamlit)**：
  ```bash
  poetry run python scripts/run_visual.py --mode online
  ```

## 5. 架构与执行流设计

在 v5 架构下，`bt_studio` 实现了解耦的**双环路机制**：

### 🔄 环路 1：LLM Agent (特征飞轮)
1. **生成与推演**：Agent 通过交互生成全新的指标组合公式，并转换为内部的 AST (抽象语法树)。
2. **静态校验**：通过 `safeops` 及 `talib_validate_hook` 进行算子白名单与因果门控验证。
3. **入库对接**：经过基本验证的“获胜”特征被打包为标准的 JSON，落盘至 `result/llm/pending`(File Contract 机制)。Agent 自主继续下一轮探索。

### ⚙️ 环路 2：Orchestrator & DAG Engine (执行引擎)
1. **统一入口**：`AsyncTaskManager` 按顺序从外部 API 或是 Agent 的 `pending` 目录吸纳任务。
2. **DAG 解析**：解析指定的 XML DAG (如 `wfo_production.xml`)，应用全局基线参数。
3. **滚动迭代 (Walk-Forward)**：
   - `node_prepare_macro`: 锁定当月股票池。
   - `node_extract_feature_monthly`: 并发调取 Tick 级数据合成因子。
   - `node_check_decay_monthly`: (门控点) 通过按日配对 Wilcoxon 符号秩检验判定上月模型是否衰减。
   - `node_tune_monthly` / `node_update_fsm_matrix`: 根据衰减门控结果，决定是启动 Ray Tune 并发调优，还是直接继承转移矩阵。
   - `node_oos_inference_monthly`: 生成最终 Out-Of-Sample 得分。

## 6. 开发约定

- **参数透传**：基线参数在 `default_config.py` 中集中定义，一经实例化，全局不可变透传。
- **纯粹的 DAG 设计**：流程控制全部以显式依赖边 (`deps` / 条件边 `?`/`?!`) 声明在 XML 中，业务节点代码保持纯函数式 (如 `tune.py` 中)，签名由 executor 按声明注入。
- **结果隔离**：输出按 task_id 分桶 (`utils/tasking.py`)，多特征/多任务互不覆盖，无 task_id 时回落扁平布局。