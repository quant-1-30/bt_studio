# 幂律分布
# **极值理论 (Extreme Value Theory, EVT)：** 在对策略历史回撤进行建模时，单独把那些超过阈值的极值（Tails）切出来，用**广义帕累托分布 (GPD)** 去拟合。这能准确预测出“黑天鹅降临时，我到底会亏多少”。
# ***小数凯利 (Fractional Kelly)：** 经典的凯利公式 $f = p - q/b$ 也是基于温和分布假设的。在幂律市场中，极端亏损会导致分母爆发。实盘中，顶级机构算出的最佳仓位后，通常只执行 **半凯利 (Half-Kelly) 甚至 1/4 凯利**，留足现金应对幂律左尾的深
# ### 2. 工业级应用方案：稳健缩放（Robust Scaling）
# 面对幂律特征（如成交量、极端收益率），不要用均值和标准差，改用：
# *   **横截面排序 (Cross-sectional Ranking)：** 将 5000 只股票当天的成交量转化为 0~1 之间的百分位排名（Percentile Rank）。无论极端放量有多恐怖，最大值永远是 1.0。
# *   **中位数绝对偏差 (MAD, Median Absolute Deviation)：** 用中位数代替均值，用 MAD 代替标准差。
# *   **对数转换 (Log Transformation)：** 对于价格和成交量，必须先取对数 $\log(x)$，将乘性的幂律爆发转化为线性的加法，再喂给 FSM 或机器学习模型。

# 在您的量化系统中，这表现为两种截然不同的子策略并行：

# 1. **左侧极度保守（90% 资金）：**
#    配置在极低风险、免疫市场黑天鹅的策略上。例如：您的**无风险套利策略**、**严格 Beta 中性的高频残差策略**。这部分提供缓慢但确定性的线性收益。
# 2. **右侧极度激进（10% 资金）：**
#    配置在专门捕获“幂律右尾”的高赔率策略上。例如：您之前设计的**“宏观牛市条件下的微观 VCP 形态突破贝叶斯策略”**。这部分策略平时可能经常因为小止损而流血（Bleeding），但一旦压中一个长达数月的连招主升浪，这 10% 的资金翻 10 倍，足以覆盖整个组合的风险。
# 3. **绝对不碰中间地带：**
#    坚决不把重仓放在那些“胜率 60%，盈亏比 1:1，但可能遭遇黑天鹅一波带走”的平庸策略上。

# 转化为“鲁棒标准差” (Robust SD):** 为了让 MAD 在数值尺度上与传统的标准差兼容（方便套用 Z-Score），统计学上通常乘以一个常数 `1.4826`（在正态分布假设下的渐近缩放因子）。
# $\sigma_{robust} = 1.4826 \times MAD$

# universe / cross section # 拼接由于市场政策变化 创业板10%-20%, 因此标的nbins 适配特定时点变化
# Volatility Clustering 
# revision ---> 14:55 降低日内回撤风险
### **1. 状态机匹配 (Step FSM)**；**2. 贝叶斯大脑更新 (Update Posterior)**；**3. 全概率预测 (Predict)**

嵌套优化问题（Nested Optimization Problem）”**。

**业界顶尖解决方案：“网格化降维 + 代理评估模型 (Surrogate-Assisted Hierarchical Search)”**


# Ray Scale
RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 ray start --address='172.20.10.3:6379' / --head --object-store-memory=****
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

ray stop 

ray status

127.0.0.1:8265 # dashboard


Ray can't initialize sys standard streams due to fd restricted

Mac ulimit -n 256 default to adapt old select program 

select / epoll / kqueue , select traverse / epoll notify (epoll_create / epoll_ctl / epoll_wait) / kqueue (kevent register / wait)

epoll (red-black tree) io callback linux / kqueue macos trigger by lt and et

Ray:
** StoreAgent **基础设施服务（Infrastructure Service）** Detached 模式 (`lifetime="detached"`) **机制**：**全局注册（Pinned by GCS） 引用数 = 0 (来自脚本) + 1 (来自 GCS) = 1 **

1.  **持久化连接**：它持有的 TCP/ZMQ 连接非常昂贵，不能因为你跑了一次 `bt_core.py` 脚本结束了，连接就断开。下次跑还得重新连。
2.  **共享复用**：你可能同时起 5 个不同的回测脚本（Driver），它们都要连接同一个 `StoreAgent`。如果是默认模式，谁创建谁负责销毁，很难共享。
3.  **服务发现**：因为它是 Detached 且有名字（Name），任何后来的脚本只要知道名字 `StoreAgent_NodeID`，就能通过 `ray.get_actor()` 连上它，而不需要重新创建

1.  `self.chan.put`  Actor 无法访问你本地进程的 Queue

*   **原则** **Rx 流操作 (pipe/subscribe)**、**复杂中间态处理** 的逻辑，都应该 **留在 Actor 内部**
*   **接口** Actor 对外只暴露 **“请求 -> 响应”** 的粗粒度接口

# .  **内存极度受限**：你设置了 `object_store_memory=2GB`。
# 2.  **并发压力**：4 个 Agent 同时在跑，每个都在疯狂往里 `ray.put` 数据（PyArrow Tables）。
# 3.  **驱逐机制 (Eviction)**：当 Object Store 满了（2GB 很容易满），Ray 会触发 **LRU 驱逐策略**。它会尝试把“引用计数看似较少”或者“旧的”对象清理掉，或者 Spill（溢出）到磁盘

ValueError: The configured object store size (25.0GiB) exceeds the optimal size on Mac (2.0GiB). This will harm performance! There is a known issue where Ray's performance degrades with object store size greater than 2.0GB on a Mac.To reduce the object store capacity, specify`object_store_memory` when calling ray.init() or ray start.To ignore this warning, set RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1.


# Parallel(n_jobs=pool_size)(delayed(rpc)(meta) for meta in batches) # Parallel(n_jobs=2, return_as="generator")
# export RAY_record_ref_creation_sites=1

ray start --head --object-store-memory 21474836480 --metrics-export-port=8080 # 20G 

ray service struck reason:
1\  allocate memory and cpus
2\ ray start memory = ray actor + 
3\ Ray 的 CPU 资源是「并发上限约束」，而非「任务总数约束」

export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=0 # avoid swap to ssd


ray start --head --num-cpus 12 --memory 34359738368 

✅ ray.put + yield ObjectRef:  The object has already been deleted by the reference counting protocol. This should not happen.


CPU 使用率从 20% 提升到 30% 说明之前的优化（如 Async StoreAgent）生效了，消除了死锁和部分阻塞，但**系统的并发度（Concurrency）依然不足以填满 CPU**。

这通常是因为：**Ray 的 Worker 申请了 CPU 资源（比如 `num_cpus=1`），但大部分时间在等待数据（IO Wait）或等待 Actor 响应，导致物理 CPU 并没有真正在计算。**

要将 CPU 压榨到 80%-90% 以上，你需要实施 **“超额订阅（Oversubscription）”** 和 **“去中心化（Sharding）”**

StoreAgent 占用的 CPU 100% 计入 ray start / ray.init() 启动的 Ray 节点资源池里，它不会是“额外”的系统进程资源

# pytest 

| scope    | 作用范围             |
| -------- | ---------------- |
| function | 每个 test 一次（默认）   |
| class    | 每个类一次            |
| module   | 每个文件一次           |
| session  | 整个 pytest 进程一次 ⭐ |


asyncio.get_running_loop()` 的行为**：
    *   这个函数只有在 **协程内部** 或者已经通过 `asyncio.run()` 启动的上下文中才能调用 

grpc-aio 与 uvloop 冲突
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


## 1. 为什么不能直接用 `get_running_loop`？（死锁陷阱）

Ray 的 Async Actor 或某些 Worker 环境确实自带一个 Event Loop（运行在主线程）。但是，如果你尝试在这个 Loop 上玩“同步桥接”，会引发死锁。

#### 死锁场景复现：
1.  **Ray 主线程**：正在运行 Event Loop。
2.  **你的代码**：调用 `td_api.submit()`（同步接口）。
3.  **你的操作**：
    *   获取主线程 Loop：`loop = asyncio.get_running_loop()`。
    *   提交任务：`fut = asyncio.run_coroutine_threadsafe(coro, loop)`。
    *   **死锁点**：`result = fut.result()`（阻塞主线程，等待结果）。
4.  **后果**：
    *   主线程被 `fut.result()` 卡住了（Block）。
    *   Event Loop 也就被卡住了（因为它跑在主线程）。
    *   提交的 `coro` 永远得不到执行机会（因为 Loop 被卡住了）。
    *   **结果：程序永久挂起（Deadlock）。**

# 性能分析 recursive 是 killer

loop.create_task() 返回的 asyncio.Task 需用 await 等待，而 run_coroutine_threadsafe 返回的 Future 需用 result() 等待（不能 await，需用 asyncio.wrap_future() 转换后才能 await）。

读写架构分离 （ 分布式读取 / 去中心写入）
   **MdApi** -> **Worker 进程级单例**（Factory Pattern）。因为它是读操作，且连接开销相对较小，分散在 Worker 中可以利用多网卡带宽。
   ***Writer** -> **Cluster 级单例**（Ray Actor）。因为它是写操作，数据库连接是瓶颈，必须中心化管理以利用 Batch Insert 的优势。

proxy for batchwriter


Ray Tune 默认会在控制台（CLI）输出参数表。如果你通过 with_parameters 传入了 data_ref 或 actor，控制台可能会显示类似 ObjectRef(xxx) 或 Actor 的字符串表示

*避坑指南：**
永远不要用以下名字命名你的 Python 脚本：
*   `signal.py`
*   `socket.py`
*   `select.py`
*   `threading.py`
*   `queue.py`
*   `io.py`
*   `code.py`
*   `email.py`
*   `random.py`

# ta-lib replace indicator logic




export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
ray start --head --include-dashboard=false --system-config='{"automatic_object_spilling_enabled": false, "metrics_export_port": 8080}' 
ray summary actors


# 按照以上思路结合实际工程落地:  以单个A市场为例（比如创业板）  
# 1\  基于A市场抽取5%标的，2010-2020作为训练数据（chain + FSM, 2020-2026作为验证（通过ray tune 寻找最优参数）
# 2、根据1中的最优chain 和 fsm ， 离线计算每个标每天14:55 是否到达fsm最后一个节点 + 涨的概率，形成一个parquet文件
# 3、将所有标的parquet文件组合起来，用于backtrader 回测的数据。（根据每天所有的标的在14:55 涨区间概率）
# 根据以上方案相当于 按照不同市场执行backtrader , 如果对接实盘 提前读取所有parquet 过滤处于fsm倒数第二个节点, 
# 在当天14:55执行计算判断顺利迁移到最后一个节点的概率 并执行买入或者卖出操作。 应该parquet 应该包含 当前节点顺序、prob, meta应该包含（chain , fsm 等固定数据）


### 1. Parquet 文件的 Schema 设计（保存什么？）

# 对于每一个标的、在每天的 14:55，你只需要拿当天的 $m$ 分钟残差收益曲线，与你 Top-K 的 `learned_motif` 计算一次 Z-Normalized 欧氏距离。
# **只有当 `distance < threshold_d` 时，才生成一条记录写入 Parquet。**

# 这个 Parquet 文件的 Schema（表结构）建议设计如下：

# | 列名 (Column) | 类型 (Type) | 说明 (Description) |
# | :--- | :--- | :--- |
# | `date` | `int32` | 触发日期，例如 `20200102` |
# | `sid` | `binary` / `string` | 股票代码，例如 `b'000001'` |
# | `trial_id` | `string` | 记录是哪个 Top-K 模型触发的（关联特定的 motif 和 FSM） |
# | `distance` | `float32` | 实际匹配的距离（越小说明形态越逼真，可用于后续信号加权） |
# | `macro_state` | `int8` | 触发当日的宏观状态 (0=跌, 1=震荡, 2=涨) |
# | `signal_score` | `float32` | **核心**：基于 FSM 矩阵查表算出的该股票 T+1 预期收益得分 |
# | `bins` 

# 决策树 算法应用于量化投资


# 编译 dtaidistance
1\ export env
export CFLAGS="-I/opt/homebrew/opt/libomp/include"
export CXXFLAGS="-I/opt/homebrew/opt/libomp/include"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"

2\ clear cache
    poetry run pip uninstall -y dtaidistance
    poetry cache clear pypi --all
3\ recompile to install
poetry run pip install --no-cache-dir --no-binary dtaidistance dtaidistance 

4\ test c api
    from dtaidistance import dtw
    print(dtw.try_import_c())

# in cluster a. nfs / b. grpc host / c. resource / d. docker distribute

brew services start redis
caffeinate -i -s python finetune.py

# import resource
# # temporarly avoid init_sys_streams bug
# soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
Ray Tune 的自动解包机制**，不再在下游手动调用 `ray.get()`

# oos_gpd = build_rolling_gpd(oos_panel_all, best_config["loopback"], best_config["gpd_quantiles"], best_config["gpd_freq_month"])
# oos_gpd_ref = ray.put(oos_gpd)
        
# best_config["learned_motif"] = best_trial.metrics["learned_motif"]
# best_config["fsm_prior_matrix"] = best_trial.metrics["fsm_prior_matrix"]
        
# ray_oos_ds = ray.data.from_arrow(oos_panel.to_arrow()) 
        
# scored_ds = ray_oos_ds.map_batches(
#     MotifFSMModel,  
#     fn_constructor_kwargs={
#         "config": best_config, 
#         "macro_ref": macro_ref,   # 传指针！
#         "gpd_ref": oos_gpd_ref    # 传指针！
#     },
#     batch_format="pyarrow", 
#     batch_size=5000,
#     num_cpus=1,
#     concurrency=10
# )

# output_path = f"/data/factors/my_strategy/year={trade_year}"
# scored_ds.write_parquet(output_path)

    
T-1 14:55 / T 14:55 动态分位数界定宏观状态自适应高波/低波周期避免状态太多先验概率支撑模型会严重退化 

内存中 wird (freeze) / commpressed (cold data not release)

RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 # 
