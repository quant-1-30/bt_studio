# bt_studio 项目指南

## 1. 项目概述

`bt_studio` 是一个基于 Python 的量化研究与回测工作室，专注于从中国 A 股市场数据中发现日内/个股的预测性模式。核心研究范式为：

- 使用矩阵轮廓（`stumpy`）和 DTW（`dtaidistance`）进行 motif 发现。
- 基于 OFI（Order Flow Imbalance）曲线构建每日宏观状态，并建立有限状态机（FSM）转移网络。
- 通过 Prefect 编排、Ray Tune + Optuna 进行滚动优化。
- 使用 `FSMPredictor` 对样本外数据进行打分，输出 Parquet。
- 通过 Bokeh / Matplotlib / Streamlit 进行可视化。
- 通过 `bt_sdk` / `bt_protocol` 市场数据 RPC 接口获取行情。

## 2. 技术栈与关键依赖

| 层级 | 库 / 工具 |
|------|----------|
| 语言 | Python `>=3.11,<3.15` |
| 包管理 | Poetry |
| 核心数据 | Polars、NumPy、PyArrow、pandas、joblib |
| 统计 / 机器学习 | scikit-learn、statsmodels、scipy、stumpy、dtaidistance、optuna |
| 分布式超参优化 | Ray（`ray[tune]`），配合 `OptunaSearch` 与 `ASHAScheduler` |
| 工作流 / MLOps | Prefect 3、MLflow 3、prometheus-client |
| 可视化 | Bokeh 3、Matplotlib、Streamlit、duckdb |
| 行情数据 RPC | `bt-sdk`、`bt-protocol`、`bt-core`、ZeroMQ、asyncio、uvloop、reactivex |
| 部署 | Docker Compose（Postgres、Prefect、MLflow、Prometheus、Grafana） |
| 其他 | coremltools、graphviz、dotenv、asitop、mlops |

`pyproject.toml` 中配置的 PyPI 源：

- 主源：清华 TUNA 镜像
- 辅助源：本地 devpi 服务器 `http://localhost:3141/bt_sdk/dev/+simple/`

## 3. 仓库结构

```
bt_studio/
├── pipeline/          # 核心流水线
│   ├── features/      # ofi.py — 从 tick/分钟线构建 OFI ratio 特征
│   ├── inference/     # predict.py — FSMPredictor，样本外打分
│   ├── patterns/      # astc.py — FSM panel 构建、motif 发现、DTW 打分
│   ├── preprocess/    # macro.py、sample.py — 数据获取、对齐、股票池采样
│   └── tune/          # train.py — Ray Tune + Prefect 滚动优化流水线
├── plugins/           # 数据接入插件
│   ├── zmq_svr.py     # 异步 ROUTER/DEALER ZMQ 代理服务器
│   ├── tcp_client.pyx # 用于 bt_sdk RPC 的 Cython 异步 TCP 客户端
│   └── zmq_client.pyx # 用于 bt_sdk RPC 的 Cython 异步 ZMQ DEALER 客户端
├── utils/             # 公共工具
│   ├── common.py      # 稳健 z-norm、DTW 参数计算、流收集器
│   └── util.py        # 计时装饰器 + z-normalize
└── visual/            # 可视化
    ├── app.py         # Streamlit 日志分析应用（DuckDB + Bokeh）
    ├── vis.py         # Matplotlib 3D 超参数地形图
    ├── bkh/           # Bokeh 绘图（Plot 类、主题、工具）
    └── mpl/           # Matplotlib 绘图（类 backtrader 的 Plot、主题、finance、格式化）

contrib/               # 实验/遗留代码
├── astc.py            # 已注释掉的 ATSC motif chain 辅助代码
├── engine.py          # BayesianOnlineFSM + MotifFSMModel（实验性）
├── preprocess.py      # 大量宏观/tick/GPD 预处理辅助函数
└── sentinel.py        # Sentinel 对象工具

scripts/
└── start_env.sh       # 设置 AIRFLOW_HOME/PYTHONPATH 并迁移 Airflow 数据库

tests/                 # 测试目录（当前内容有限，见第 7 节）
├── logs/
├── test_plot.html
├── test_plot.py
└── test_prefect_fsm.py
```

## 4. 构建、测试与运行命令

项目未在 `pyproject.toml` 中配置 Poetry scripts、Makefile 或 pytest。

- **安装依赖：**
  ```bash
  poetry install
  ```

- **运行核心流水线：**
  ```bash
  poetry run python bt_studio/pipeline/tune/train.py
  ```

- **启动基础设施服务：**
  ```bash
  docker-compose up -d
  ```

- **启动 Airflow 环境（`scripts/start_env.sh` 中的逻辑）：**
  ```bash
  export AIRFLOW_HOME=$(pwd)/airflow_home
  export PYTHONPATH=$(pwd)
  export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/bt_studio/pipeline/dags
  export AIRFLOW__CORE__LOAD_EXAMPLES=False
  poetry run airflow db migrate
  ```

- **Ray 集群（来自 README 笔记）：**
  ```bash
  export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
  ray start --head --port=6379 --include-dashboard=false --metrics-export-port=8080 \
    --num-cpus=8 --object-store-memory 21474836480 --memory 34359738368 \
    --system-config='{"automatic_object_spilling_enabled": false, "metrics_export_port": 8080}'
  ```

- **Prefect 服务：**
  ```bash
  prefect server start --host 0.0.0.0
  ```

- **MLflow 服务：**
  ```bash
  mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db ...
  ```

- **测试：** 当前未配置测试运行器。`pytest` 不是项目的直接依赖，仅在 `poetry.lock` 中作为传递性开发依赖出现。

## 5. 运行时与部署架构

### Docker Compose 服务（`docker-compose.yml`）

| 服务 | 镜像 | 端口 | 说明 |
|------|------|------|------|
| `postgres` | postgres:15 | 5432 | 用户 `quant_user`，密码 `quant_password`，数据库 `studio_db` |
| `prefect-server` | prefecthq/prefect:3-latest | 4200 | 指向本地 Postgres 连接字符串 |
| `mlflow` | ghcr.io/mlflow/mlflow:v2.10.0 | 5000 | 使用 Postgres 后端，本地 artifact 挂载 `/mnt/data/mlflow_artifacts` |
| `prometheus` | prom/prometheus:latest | 9090 | 抓取 `host.docker.internal:8080` 的 Ray 指标 |
| `grafana` | grafana/grafana:latest | 3000 | 默认管理员密码 `admin` |

### 流水线运行流程

1. `prepare_macro()` 通过 `bt_sdk.ctx.external_mdapi_context` 获取股票池和日线数据。
2. `universe_sample()` 按成交额 / 上市时间过滤。
3. `prepare_tick()` 获取 tick 数据，对齐到 240 分钟骨架，构建 OFI 曲线。
4. `discover_fsm_pattern()` 使用 `stumpy` 生成 motif 候选，并通过 DTW 确定触发条件。
5. `evaluate_and_build_fsm()` 构建经 Laplace 平滑的转移矩阵，并执行 Mann-Whitney U 检验。
6. `train.py` 使用 Prefect `@task` / `@flow` 与 Ray Tune（`OptunaSearch`、`ASHAScheduler`）进行编排，结果记录到 MLflow。
7. `FSMPredictor` 对样本外数据打分，输出 `scores_{year}.parquet`。

### 插件 / RPC

- `zmq_svr.py` 是一个 Python ZMQ ROUTER→DEALER 代理，带工作池和连接统计。
- `tcp_client.pyx` / `zmq_client.pyx` 是 Cython 异步客户端，使用 `uvloop`、PyArrow IPC 流和 `reactivex` subjects，依赖 `bt_sdk.core.rpc.client` 中的 `RpcClient`。

## 6. 代码组织与开发约定

- **模块导出：** 每个子包通过 `__init__.py` 重新导出公共符号。
- **惰性求值：** 大量使用 `polars.LazyFrame` 和 `scan_parquet`。
- **函数式风格：** 流水线步骤多为纯函数，输入 `pl.LazyFrame` / `pl.DataFrame` 与 `config: dict`。
- **配置驱动：** 超参数、日期、阈值等均通过嵌套字典传入。
- **注释语言：** 代码标识符为英文，注释主要为中文。
- **文件规模：** 流水线代码较紧凑；`visual/mpl/plot.py`（899 行）和 `contrib/preprocess.py`（562 行）为较大文件。

## 7. 测试策略

当前测试覆盖非常有限，且部分内容已过期：

- `tests/test_plot.py` 并非测试文件，而是 `bt_studio/visual/bkh/plot.py` 的可执行脚本副本。
- `tests/test_prefect_fsm.py` 仅 7 行，引用了不存在的模块 `bt_studio.pipeline.dags.fsm`，且包含无效的配置占位符语法。
- 无 `pytest.ini`、`[tool.pytest.ini_options]` 或 CI 配置。
- 对 FSM 数学、DTW、OFI、ZMQ 插件等核心逻辑暂无单元测试。
