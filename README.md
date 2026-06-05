# 监控协议 opentelemetry-api / opentelemetry-sdk / opentelemetry-distro

 chromedriver / chromium/ 
 | 浏览器     | driver       |
| ------- | ------------ |
| Firefox | geckodriver  |
| Chrome  | chromedriver |
| Edge    | msedgedriver |


poetry cache list
PyPI
_default_cache
devpi
tuna
poetry cache clear devpi --all

brew services start redis
caffeinate -i -s python finetune.py

ps -ef | grep airflow | grep -v grep | awk '{print $2}' | xargs kill -9

ps -ef | grep airflow | awk '{print $2}' | xargs kill -9


# `multiple_outputs=True`（Airflow 最佳实践
Task return dict `multiple_outputs=True`  `[key]` Airflow 会在底层自动解构并解析为**真实的字符串**


# Ray Scale GCS ---> Redis
RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 ray start --head --port=6379 --include-dashboard=false --metrics-export-port=8080  
--num-cpus=8 --object-store-memory 21474836480 --memory 34359738368 
--system-config='{"automatic_object_spilling_enabled": false, "metrics_export_port": 8080}' 

配置这两个参数时，--memory + --object-memory 的总和绝对不能接近或超过物理内存的总量！ 必须至少留下 10% ~ 20% 的空闲物理内存 留给操作系统内核以及我们在前面问题中聊到的 wired（系统常驻）和 compressor 空间。否则，一旦触发操作系统的硬交换（Swap to SSD），整个 Ray 集群的吞吐量将会直接瘫痪。 

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
3.  **服务发现**：因为它是 Detached 且有名字（Name），任何后来的脚本只要知道名字 `StoreAgent_NodeID`，就能通过 `ray.get_actor()` 

# .  **内存极度受限**：你设置了 `object_store_memory=2GB`。
# 2.  **并发压力**：4 个 Agent 同时在跑，每个都在疯狂往里 `ray.put` 数据（PyArrow Tables）。
# 3.  **驱逐机制 (Eviction)**：当 Object Store 满了（2GB 很容易满），Ray 会触发 **LRU 驱逐策略**。它会尝试把“引用计数看似较少”或者“旧的”对象清理掉，或者 Spill（溢出）到磁盘

ValueError: The configured object store size (25.0GiB) exceeds the optimal size on Mac (2.0GiB). This will harm performance! There is a known issue where Ray's performance degrades with object store size greater than 2.0GB on a Mac.To reduce the object store capacity, specify`object_store_memory` when calling ray.init() or ray start.To ignore this warning, set RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1.

# Parallel(n_jobs=pool_size)(delayed(rpc)(meta) for meta in batches) # Parallel(n_jobs=2, return_as="generator")
# export RAY_record_ref_creation_sites=1

ray service struck reason:
1\  allocate memory and cpus
2\ ray start memory = ray actor + 
3\ Ray 的 CPU 资源是「并发上限约束」，而非「任务总数约束」

export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=0 # avoid swap to ssd


✅ ray.put + yield ObjectRef:  The object has already been deleted by the reference counting protocol. This should not happen.

CPU 使用率从 20% 提升到 30% 说明之前的优化（如 Async StoreAgent）生效了，消除了死锁和部分阻塞，但**系统的并发度（Concurrency）依然不足以填满 CPU**。

这通常是因为：**Ray 的 Worker 申请了 CPU 资源（比如 `num_cpus=1`），但大部分时间在等待数据（IO Wait）或等待 Actor 响应，导致物理 CPU 并没有真正在计算。**

要将 CPU 压榨到 80%-90% 以上，你需要实施 **“超额订阅（Oversubscription）”** 和 **“去中心化（Sharding）”**

StoreAgent 占用的 CPU 100% 计入 ray start / ray.init() 启动的 Ray 节点资源池里，它不会是“额外”的系统进程资源


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


export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1

ray summary actors


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


brew services start redis
caffeinate -i -s python finetune.py

# import resource
# # temporarly avoid init_sys_streams bug
# soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
Ray Tune 的自动解包机制**，不再在下游手动调用 `ray.get()`

内存中 wird (freeze) / commpressed (cold data not release)

RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 # 

RayTrainReportCallback

docker-compose up -d

# prefect

prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopSecretPassword@localhost:5432/prefect"


mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  <!-- --backend-store-uri postgresql://user:password@localhost:5432/mlflow \ -->
  --default-artifact-root s3://my-bucket/mlflow

export PREFECT_TELEMETRY_ENABLE=false

unset PREFECT_API_URL

prefect config unset PREFECT_API_URL

# server

prefect server start --host 0.0.0.0


# client

prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

ps -ef | grep prefect | grep -v grep | awk '{print $2}' | xargs kill -9

ps -ef | grep uvicorn | grep -v grep | awk '{print $2}' | xargs kill -9

rm -rf ~/.prefect

unset http_proxy
unset https_proxy
unset all_proxy


export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"

