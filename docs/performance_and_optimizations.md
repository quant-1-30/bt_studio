# 性能优化与瓶颈分析指南 (Performance & Optimizations)

在 bt_studio 的 v5 (XML DAG + Engine) 架构落地后，系统的执行链路已实现极致的解耦与收敛。随着特征挖掘广度（LLM Agent 飞轮）和回测深度的增加，系统的物理瓶颈主要集中在**内存 (OOM)**、**CPU 并发争用** 以及 **磁盘 I/O** 上。

本文档梳理了目前架构中存在的性能卡点，以及当前/未来的对应解决方案。

---

## 1. 内存瓶颈与 OOM (Out Of Memory) 风险

### 1.1 风险定位：`node_prepare_train_data` 的全量实例化
在 Walk-Forward 的训练阶段，为了让后续成百上千个 Ray Tune 试验 (Trials) 能够极速复用数据，`node_prepare_train_data` 节点会将整个训练窗口（例如连续 12 个月的分钟级或 Tick 级特征数据和收益率数据），通过 Polars 执行 `.collect().to_arrow()`，然后调用 `ray.put()` 送入 Ray 的 Plasma Object Store 共享内存中。

**痛点：**
- 当标的池（Universe）扩大，或高频数据分辨率极高时，这步 `.collect()` 会在主进程中产生极大的瞬时内存峰值。
- Plasma 存储的激增可能导致 Ray 集群触发 Object Spilling（对象溢出到磁盘），一旦发生，HPO 的速度将呈现断崖式下跌，甚至直接 OOM 崩溃。

### 1.2 当前防护与未来解决方案
- **当前防护 (已实现)**：
  - **Static Panel 剥离缓存**：将与 Tune 搜索空间（Search Space）无关的静态特征（如目标收益、Z-Score 截面标准化等）提前做了一次性的 `build_static_panel`。后续所有试验只针对需要动态降采样的曲线（Curves）做计算，大幅降低了每轮 Trial 的内存冗余拷贝。
- **演进方向 (Future Work)**：
  - **前置降采样 (Downsample Before Parquet)**：如果原始数据过大，可考虑在 `node_extract_feature_monthly` 生成特征落盘时，直接输出特定级别的降采样聚合，而不是将全量高频数据留到 HPO 阶段。
  - **Chunked / Lazy execution 深度下推**：避免在主进程一次性 `.to_arrow()`，而是向 Ray Worker 传递 Polars LazyFrame 的查询计划（Query Plan）或文件路径列表，依靠 Polars 自带的 Out-of-Core (流式处理) 边读边算。

---

## 2. CPU 计算风暴与并发控制

### 2.1 风险定位：算子树推演与框架争用
- Agent 生成的 AST 特征树经常包含深层嵌套的滚动算子（如基于 EMA、ROC、MACD 的多重组合）。此类算子即使由底层的 TA-Lib C 语言库和 Polars Rust 核心支持，在海量数据上的计算仍然极为密集。
- Polars 默认会吃满所有可用的 CPU 核心进行计算，这与 Ray Tune 试图并发拉起多个 Trial Worker 的行为产生严重的**线程争用 (Thread Contention)**，导致上下文切换开销（Context Switching Overhead）飙升。

### 2.2 当前防护与未来解决方案
- **当前防护 (已实现)**：
  - **环境变量级硬隔离**：在 `bt_studio/engine/executor.py` 的执行引擎初始化时，强制为 Ray 注入严格的单线程运行时环境（Runtime Environment）：
    ```python
    "env_vars": {
        "POLARS_MAX_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    ```
    这确保了 Ray 在调度 `common_config["num_workers"]` 个并行 Worker 时，每个 Worker 内部的 C/Rust/C++ 计算库均退化为单线程，完全避免了 CPU 踩踏风暴。
  - **`reuse_actors=True`**：在 Tune 的配置中开启，避免了 Python 进程反复启动和 JIT 编译的巨大开销。
- **演进方向 (Future Work)**：
  - **算子融合 (Operator Fusion)**：在 AST 编译器中引入常量折叠和常见算子的融合（例如先平滑再差分），减少中间产生的临时列。

---

## 3. 磁盘 I/O 与特征复用

### 3.1 风险定位：冗余的 gRPC 拉取与重复落盘
在全生命周期的 Walk-Forward 滚动中，多个窗口会有高度重叠的月份（例如，1 月的 OOS 可能是 2 月的 Train 构成）。如果管控不当，网络 I/O (gRPC 拉取 Tick) 和磁盘写入会成为拖慢流水的核心卡点。

### 3.2 当前防护与未来解决方案
- **当前防护 (已实现)**：
  - **基于月份的严格 PIT (Point-In-Time) 缓存机制**：`node_extract_feature_monthly` 会前置检查 `FEATURE_DIR` 下是否已存在 `hf_{feature}_{ym}.parquet`，仅对缺失的月份发起 gRPC 网络请求。
  - **DAG 消费者剪枝 (Consumer Pruning)**：在 Executor 解析 XML DAG 时，引入了 `paths_for_months` 逻辑。针对 OOS 衰减检测 (`decay`)，直接复用已提取的 Train 月份特征文件。对于在 Agent 验证阶段 (如 `wfo_hpo.xml`) 被 Skip 掉的 OOS 节点，Executor 会联动剪枝，跳过对其专属切片的无用拉取。
- **演进方向 (Future Work)**：
  - 目前大量细碎的月度 Parquet 在长时间跨度下会引发较长的寻址时间。如果磁盘吞吐成为瓶颈，可引入基于 DuckDB 或更高效列式格式的聚合冷库设计，或者直接挂载高速的内存盘（tmpfs / ramdisk）应对。

---

## 总结
目前的工程架构通过**环境级线程束缚**、**静态数据与动态逻辑切分共享**、**DAG节点剪枝复用**，已将单机 / 集群性能压榨到一个非常稳定且高效的状态。对于随时可能爆发的 OOM 风险，重点需要管控 `train_window` 的跨度以及 `downsample` 倍率的设计。