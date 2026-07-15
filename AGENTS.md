# AGENTS.md — bt_studio 项目指南

---

## 1. 项目概述

`bt_studio` 是一个基于 Python 的量化研究与回测工作室，聚焦中国 A 股市场日内/个股的预测性模式挖掘。核心研究范式为：

- 使用 **矩阵轮廓（`stumpy`）** 与 **DTW（`dtaidistance`）** 发现时序 motif；
- 基于 **OFI（Order Flow Imbalance）** 曲线构建每日宏观状态，并建立有限状态机（FSM）转移网络；
- 通过 **Prefect** 编排、**Ray Tune + Optuna** 进行滚动超参优化；
- 使用 `FSMPredictor` 对样本外数据打分，输出 Parquet；
- 通过 **Bokeh / Matplotlib / Streamlit** 可视化结果；
- 通过 `bt_sdk` / `bt_protocol` 市场数据 RPC 接口获取行情。

项目当前为研究与实验性质，部分文档与代码存在过期、损坏或未完成的模块（详见第 8 节“已知问题”）。

---

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

- 主源：清华 TUNA 镜像 `https://pypi.tuna.tsinghua.edu.cn/simple/`
- 辅助源：本地 devpi 服务器 `http://localhost:3141/bt_sdk/dev/+simple/`

> 注意：仓库根目录存在一个 `_version.py`，它是遗留的 versioneer 文件，但末尾被混入了一段无关代码（`subprocess.run(...)`），当前不可作为版本工具使用。

---

## 3. 仓库结构与代码组织

```
bt_studio/
├── bt_trade/           # 交易原语（Pnc / Sizer）
│   ├── __init__.py
│   ├── pncs.py         # TestPnc：PnC 空实现占位
│   └── sizers.py       # FixedSize / WeightedSizer / TurtleSizer / KellySizer
├── pipeline/           # 核心流水线
│   ├── features/       # ofi.py — 从 tick/分钟线构建 OFI ratio 特征
│   ├── inference/      # predict.py — FSMPredictor，样本外打分
│   ├── metrics/        # score.py / pareto.py / fanova.py — HPO 评分与模型选择
│   ├── patterns/       # astc.py / fsm.py / pattern.py — motif 发现、FSM 构建
│   └── preprocess/     # macro.py / sample.py / panel.py — 数据获取、对齐、建面板
├── plugins/            # 数据接入插件
│   ├── tcp_client.pyx  # Cython 异步 TCP 客户端（bt_sdk RPC）
│   ├── zmq_client.pyx  # Cython 异步 ZMQ DEALER 客户端
│   └── zmq_svr.py      # Python 异步 ZMQ ROUTER→DEALER 代理服务器
├── utils/              # 公共工具
│   ├── common.py       # 稳健 z-norm、DTW 参数计算、流收集器、decay 权重
│   └── util.py         # 计时装饰器 + z-normalize
├── visual/             # 可视化
│   ├── app.py          # Streamlit 日志分析应用（当前不可用，见第 8 节）
│   ├── board.py        # 仅含注释，无实际代码
│   ├── vis.py          # Matplotlib 3D 超参数地形图
│   ├── bkh/            # Bokeh 绘图（Plot 类、主题、工具）
│   └── mpl/            # Matplotlib 绘图（类 backtrader 的 Plot、主题、finance 等）
├── sim_engine.py       # 基于 bt_core 的回测/仿真入口
└── tune_train.py       # 当前 Walk-Forward Optimization（WFO）训练与推理入口

contrib/                # 实验性多维 FSM 重写（当前不可直接运行）
├── mastc.py            # 多维 ASTC 辅助
├── mfsm.py             # 多维 FSM 构建
├── mpattern.py         # 多维 motif 发现
└── mpredict.py         # 多维 FSMPredictor

scripts/
└── start_env.sh        # 设置 AIRFLOW_HOME/PYTHONPATH 并迁移 Airflow 数据库

tests/                  # 测试目录（当前内容有限且部分损坏）
├── logs/
├── test.py
├── test_plot.html
├── test_plot.py
└── test_prefect_fsm.py
```

### 3.1 模块导出约定

每个子包通过 `__init__.py` 重新导出公共符号。例如：

- `bt_studio.pipeline.preprocess` 导出 `prepare_macro`、`prepare_tick`、`universe_sample`、`build_fsm_panel`；
- `bt_studio.pipeline.patterns` 导出 `evaluate_and_build_fsm`、`discover_fsm_pattern`；
- `bt_studio.pipeline.inference` 导出 `FSMPredictor`。

### 3.2 代码风格约定

- **标识符语言**：变量、函数、类名为英文；注释主要为中文。
- **惰性求值**：大量使用 `polars.LazyFrame` 与 `scan_parquet`。
- **函数式风格**：流水线步骤多为纯函数，输入 `pl.LazyFrame` / `pl.DataFrame` 与 `config: dict`。
- **配置驱动**：超参数、日期、阈值等均通过嵌套字典 `common_config` / `tune_config` 传入。
- **文件规模**：流水线代码较紧凑；`visual/mpl/plot.py`（约 899 行）为最大单文件。

---

## 4. 构建、测试与运行命令

项目未在 `pyproject.toml` 中配置 Poetry scripts、Makefile 或 pytest 选项。

### 4.1 安装依赖

```bash
poetry install
```

### 4.2 启动基础设施服务

```bash
docker-compose up -d
```

服务清单见 `docker-compose.yml`：

| 服务 | 镜像 | 端口 | 说明 |
|------|------|------|------|
| `postgres` | postgres:15 | 5432 | 用户 `quant_user`，密码 `quant_password`，数据库 `studio_db` |
| `prefect-server` | prefecthq/prefect:3-latest | 4200 | 指向本地 Postgres 连接字符串（注意：配置中的密码 `yourTopSecretPassword` 与 postgres 服务不一致） |
| `mlflow` | ghcr.io/mlflow/mlflow:v2.10.0 | 5000 | 使用 Postgres 后端，本地 artifact 挂载 `/mnt/data/mlflow_artifacts` |
| `prometheus` | prom/prometheus:latest | 9090 | 抓取 `host.docker.internal:8080` 的 Ray 指标 |
| `grafana` | grafana/grafana:latest | 3000 | 默认管理员密码 `admin` |

### 4.3 启动 Prefect 服务

```bash
prefect server start --host 0.0.0.0
```

### 4.4 启动 MLflow 服务

`docker-compose.yml` 已内置 mlflow 服务；若需独立启动：

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql://quant_user:quant_password@localhost:5432/studio_db \
  --default-artifact-root /opt/mlflow/artifacts
```

> 注意：`bt_studio/tune_train.py` 当前硬编码 `MLFLOW_TRACKING_URI=http://127.0.0.1:5001`，与 docker-compose 暴露的 5000 端口不一致。

### 4.5 启动 Ray 集群（本地开发示例）

```bash
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
ray start --head --port=6379 --include-dashboard=false --metrics-export-port=8080 \
  --num-cpus=8 --object-store-memory 21474836480 --memory 34359738368 \
  --system-config='{"automatic_object_spilling_enabled": false, "metrics_export_port": 8080}'
```

### 4.6 运行核心训练/推理流水线

当前实际入口是 `bt_studio/tune_train.py`，而非 `README.md` 中提到的 `bt_studio/pipeline/tune/train.py`（该路径不存在）：

```bash
poetry run python bt_studio/tune_train.py
```

`tune_train.py` 实现了一个滚动月度训练/OOS 循环：

1. `node_prepare_macro`：构建/读取日线股票池 Parquet；
2. `node_extract_feature_monthly`：按月获取 tick 数据并构建 OFI 特征；
3. `node_check_decay_monthly`：检验旧模型在近期样本外数据上是否仍有效；
4. `node_tune_monthly`：使用 Ray Tune + Optuna TPE 进行 HPO，经 fANOVA 与 Pareto 筛选后保存最佳模型；
5. `node_update_fsm_matrix`：复用历史 motif，仅在新数据上更新 FSM 转移矩阵；
6. `node_oos_inference_monthly`：使用 `FSMPredictor` 输出样本外打分 Parquet。

结果默认写入 `result/fsm/{models,features,scores}/`。

### 4.7 启动可视化

- **Streamlit 应用（当前不可用）**：
  ```bash
  streamlit run bt_studio/visual/app.py
  ```
  该文件存在导入错误（`from plot import Plot` 及不存在的 `plot_from_integrated_df`）。

- **Bokeh 独立绘图**：
  ```bash
  python bt_studio/visual/bkh/plot.py
  # 或
  python tests/test_plot.py
  ```
  两者均依赖本地 parquet 路径。

- **Matplotlib 策略绘图**：
  ```python
  from bt_studio.visual.mpl.plot import Plot
  p = Plot()
  figs = p.plot(strategy)
  p.show()
  ```

### 4.8 测试

当前未配置测试运行器，也无 `pytest.ini`/`[tool.pytest.ini_options]`/CI 配置。

```bash
# 可手动安装 pytest（已通过传递依赖存在于 poetry.lock）后运行：
poetry run pytest tests/
```

但现有 `tests/` 内容基本不是有效测试：

- `tests/test.py`：仅加载两个硬编码 parquet 并打印行数/日期，无断言。
- `tests/test_plot.py`：是 `bt_studio/visual/bkh/plot.py` 的可执行脚本副本。
- `tests/test_prefect_fsm.py`：7 行，引用不存在的 `bt_studio.pipeline.dags.fsm`，且配置占位符语法无效。

### 4.9 Airflow 环境初始化（如使用）

```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
export PYTHONPATH=$(pwd)
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/bt_studio/pipeline/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
poetry run airflow db migrate
```

> 注意：`bt_studio/pipeline/dags/` 目录在当前仓库中不存在。

---

## 5. 运行时与流水线架构

### 5.1 数据流

```
外部 MDAPI（bt_sdk/bt_protocol）
        |
        v
prepare_macro() / prepare_tick()      # pipeline/preprocess/macro.py
        |
        v
universe_sample()                     # pipeline/preprocess/sample.py
        |
        v
build_ofi()                           # pipeline/features/ofi.py
        |
        v
build_fsm_panel()                     # pipeline/preprocess/panel.py
        |
        +-- 训练 ----------------------------------+
        |                                         |
        v                                         v
discover_fsm_pattern()              evaluate_and_build_fsm()
# pipeline/patterns/pattern.py      # pipeline/patterns/fsm.py
        |                                         |
        +---> stumpy/DTW 候选 motif               +---> FSM 矩阵 + bin_weights + p-value + score
        |                                         |
        +-----------------------------------------+
                                                      |
                                              metrics.pareto / fanova
                                                      |
                                              模型 checkpoint（pickle）
                                                      |
                                              FSMPredictor.predict()
                                              # pipeline/inference/predict.py
```

### 5.2 关键配置字典

#### `common_config`（跨模块通用）

| Key | 含义 |
|-----|------|
| `start_date`, `end_date` | 研究日期范围 |
| `benchmark` | 基准指数代码 |
| `dtw_window_frac` | DTW Sakoe-Chiba band 占 motif 长度比例 |
| `stats_windows` | 前向收益窗口，如 `[1, 2, 3]` |
| `decay` | 轨迹 decay 权重半衰期 |
| `ranking_window` | 宏观状态分位数滚动窗口 |
| `ranking_ratio` | 收益四分位切割比例 |
| `exclude_bars` | 尾端屏蔽 bar 数，防止 lookahead |
| `days_since_ipo` | 上市时间过滤阈值 |
| `top_k_ratio` | 各板块按成交额保留比例 |
| `alternative` | MWU 检验方向：`"greater"` / `"less"` / `"two-sided"` |
| `win_rate` | HPO 评分硬阈值 |
| `max_points` | `stumpy` 采样预算，默认 20000 |

#### `tune_config`（Optuna 搜索）

| Key | 含义 |
|-----|------|
| `cross_days` | 每条曲线拼接的天数 |
| `motif_minutes` | motif 长度（分钟） |
| `downsample` | 降采样因子 |
| `threshold_r` | DTW 相似度阈值 |
| `m` | 派生：`motif_minutes // downsample` |
| `threshold_d` | 派生：`sqrt(2*m*(1-threshold_r))` |

### 5.3 模型 checkpoint 结构

```python
{
    "config": tune_config,
    "motif": np.ndarray,
    "fsm_matrix": {
        "P(T1|Macro)": list,     # 3 × 4
        "P(T2|T1)": list,        # 4 × 4
        "P(T3|T2)": list,        # 4 × 4
        "bin_weights": {window: [w0, w1, w2, w3]}
    }
}
```

### 5.4 插件 / RPC

- `zmq_svr.py`：ZMQ ROUTER→DEALER 代理，带工作池与连接统计；
- `tcp_client.pyx` / `zmq_client.pyx`：Cython 异步客户端，使用 `uvloop`、PyArrow IPC 流和 `reactivex` subjects；
- 这两个 `.pyx` 文件未在 `pyproject.toml` 中声明构建配置，需要外部 Cython 构建步骤或手动编译。

---

## 6. 安全与敏感信息注意事项

- `.env` 文件存在且被系统标记为敏感文件，不要将其内容提交或打印到日志；
- `docker-compose.yml` 中硬编码了数据库密码（`quant_password`、`yourTopSecretPassword`），仅适用于本地开发；
- `tune_train.py` 硬编码 `MLFLOW_TRACKING_URI=http://127.0.0.1:5001`；
- 多处测试文件与可视化脚本包含本地绝对路径或硬编码文件名，不具可移植性；
- 当前代码未对 RPC 输入、Parquet 路径、反序列化对象做输入校验，生产环境需补充；
- 项目使用 GPL v3 许可证（见 `LICENSE`）。

---

## 7. 开发约定清单

- 使用 **Poetry** 管理依赖；不要直接修改 `poetry.lock`，应通过 `poetry add/lock` 更新。
- 优先使用 **Polars LazyFrame**；需要在 `collect()` 时考虑 `engine="streaming"`。
- 配置通过 `common_config` / `tune_config` 字典传递，避免硬编码阈值。
- 修改 `pipeline/patterns/` 中的 FSM 逻辑时，需同时检查 `pipeline/inference/predict.py` 与 `contrib/` 中对应的多维实现是否同步。
- 新增 Cython 模块时，需要在构建系统中补充 `setup.py` 或 `pyproject.toml` 的 build-ext 配置；当前 `.pyx` 文件无自动构建。
- 注释以中文为主，保持与现有代码一致。

---

## 8. 已知问题与注意事项

1. **README.md 过期且末尾损坏**：`README.md` 第 167–193 行混入了一段 Python 注释（来自 `tune_train.py` 或类似文件），且描述的目录结构（如 `pipeline/tune/`、`contrib/preprocess.py`）与当前文件树不符。
2. **`bt_studio/tune_train.py` 是当前工作入口**，但为未提交修改状态（`git status` 显示 `M`），且代码中大量 Prefect `@task`/`@flow` 装饰器被注释掉。
3. **`pipeline/inference/predict.py` 中 `_calculate_fsm_score` 重复定义**：第一次定义存在 `stats_windows` 未定义及 `bin_weights` 索引错误，第二次定义覆盖第一次，但需清理。**（已修复：删除重复实现、补充 `calculate_decay_weights` 导入、`bin_weights` 保持 dict 类型。）**
4. **`contrib/` 多维实现当前不可导入**：存在多处导入错误、未定义变量（如 `search_config`、`sample_size`、`.m_astc` 等）。
5. **`visual/app.py` 损坏**：引用不存在的函数与类。
6. **`tests/test_prefect_fsm.py` 损坏**：引用不存在的 `bt_studio.pipeline.dags.fsm`。
7. **`_version.py` 末尾混入无关代码**：不可作为版本工具使用。
8. **单元测试覆盖不足**：核心 FSM 数学、DTW、OFI 等已有初步 pytest 覆盖（见 `tests/test_pipeline.py`），ZMQ 插件等仍无测试。
9. **Docker Compose 配置不一致**：`prefect-server` 中的 Postgres 密码与 `postgres` 服务环境变量不匹配。
10. **`tune_train.py` 模型 checkpoint 曾缺少 `m` / `threshold_d`**：`node_tune_monthly` 从 Ray Tune 结果提取的 `best_config` 原本只含搜索空间原始参数，导致 `FSMPredictor` / `build_fsm_panel` 在 OOS 阶段 KeyError。**（已修复：保存前补全派生参数；同时增加 `warmup_paths` 为空保护。）**

---

## 9. 推荐的路径

1. 阅读 `README.md` 
2. 按顺序阅读核心流水线：
   - `bt_studio/pipeline/preprocess/macro.py`
   - `bt_studio/pipeline/preprocess/sample.py`
   - `bt_studio/pipeline/features/ofi.py`
   - `bt_studio/pipeline/preprocess/panel.py`
   - `bt_studio/pipeline/patterns/astc.py`、`pattern.py`、`fsm.py`
   - `bt_studio/pipeline/inference/predict.py`
   - `bt_studio/pipeline/metrics/*.py`
3. 阅读 `bt_studio/tune_train.py` 了解 WFO 编排。
4. 在本地启动 Docker Compose 与 Ray，再运行 `poetry run python bt_studio/tune_train.py`。
5. 可视化部分优先使用 `bt_studio/visual/bkh/plot.py` 或 `bt_studio/visual/vis.py`；`app.py` 需要修复后才能使用。
