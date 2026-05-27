# 恭喜你敏锐地察觉到了**“过度优化陷阱”！在量化系统开发中，这是无数极客型宽客（Quant Developer）最容易踩的坑——沉迷于把 C++
# 引擎的耗时从 10 毫秒压榨到 1 毫秒，却忘了量化的本质是发现 Alpha**。

# 当你把底层的 bt_core 打磨成了极致高效的 C++/Cython 黑盒后，你现在的角色应该从“引擎造车匠”转变为“赛车手”。

# bt_studio 的目标是：构建一条“高吞吐、可观测、绝对可复现”的策略流水线。

# 以下是基于 Ray + MLflow + bt_core 的工业级量化策略工厂的高效整合方案。

# 一、 策略工厂架构蓝图 (The Strategy Factory)

# 把你的系统想象成一条工业流水线：

# 1.  原料端 (Search Space)：Ray Tune 负责生成成千上万种参数组合（FSM 状态机参数、均线周期、因子权重）。
# 2.  加工机床 (Execution)：Ray Cluster 调度本地服务器的 64 个 CPU 核心，并行启动 64 个 bt_core 实例极速运算。
# 3.  质检部门 (MLOps/MLflow)：收集每个实例输出的 Sharpe, SQN, Calmar。
# 4.  档案库 (Artifact Store)：将跑得好的策略产生的 Parquet 日志 作为“资产”封存。
# 5.  展厅 (Dashboard)：你的 Streamlit 从 MLflow 中拉取 Top 5 策略的 Parquet 文件，进行深度 2D/3D
#     可视化。

# 二、 核心整合代码：Ray Tune 握手 MLflow

# 为了实现上述流线，你需要在 bt_studio 中编写一个调度脚本（Dispatcher）。

# 1. 安装必要的依赖

# pip install ray[tune] mlflow

# 2. 编写分布式训练主脚本 (train_pipeline.py)

# 这里有一个极其重要的实战经验：不要保存所有失败策略的 Parquet！（如果你跑 10000 次超参，保存所有的 Parquet
# 会瞬间撑爆磁盘）。只让引擎输出内存里的指标，只有指标达标的，才保留生成的 Parquet 作为
# Artifact。

# import os
# import tempfile
# from ray import tune
# from ray.air.integrations.mlflow import MLflowLoggerCallback
# import mlflow

# # 导入你打包好的极速引擎
# import bt_core
# from bt_core.actor import TrackerActor

# def evaluate_strategy(config):
#     """
#     这是 Ray Worker 将在独立进程/CPU核中运行的目标函数
#     """
#     # 1. 创建独立的工作目录 (防止并发写冲突)
#     with tempfile.TemporaryDirectory() as temp_dir:
#         parquet_path = os.path.join(temp_dir, "backtest_log.parquet")
        
#         # 2. 将超参注入你的业务逻辑
#         # 假设这里配置策略参数、FSM状态等
#         fast_ma = config["fast_ma"]
#         slow_ma = config["slow_ma"]
        
#         # 3. 运行极速回测引擎 (bt_core)
#         # engine 会在内部使用你写好的 Performance, Calmar 等 Analyzer
#         engine = TrackerActor(output_dir=parquet_path, fast=fast_ma, slow=slow_ma)
#         metrics = engine.run() # 假设返回一个包含终局指标的 dict
        
#         # --- 核心过滤逻辑：垃圾策略不配占用磁盘 ---
#         if metrics["SharpeRatio"] > 1.0 and metrics["MaxDrawdown"] < 0.15:
#             # 只有好策略，才将其 Parquet 文件作为 "资产(Artifact)" 上传给 MLflow
#             # 这需要你在 worker 里手动连一下 mlflow (通过 tune 传递的 run_id)
#             import mlflow
#             mlflow.log_artifact(parquet_path, artifact_path="parquet_logs")

#         # 4. 向 Ray Tune 汇报指标 (Ray 会自动将这些转报给 MLflow)
#         tune.report(
#             sharpe=metrics["SharpeRatio"],
#             calmar=metrics["Calmar"],
#             sqn=metrics["SQN"],
#             net_pnl=metrics["NetPnL"]
#         )

# def main():
#     # 1. 定义搜索空间
#     search_space = {
#         "fast_ma": tune.randint(5, 20),
#         "slow_ma": tune.randint(21, 60),
#     }

#     # 2. 配置 MLflow Callback
#     mlflow_callback = MLflowLoggerCallback(
#         tracking_uri="http://localhost:5000",  # 你的本地 MLflow 服务地址
#         experiment_name="Alpha_FSM_Optimization",
#         save_artifact=True
#     )

#     # 3. 启动 Ray 分布式调优
#     tuner = tune.Tuner(
#         tune.with_resources(evaluate_strategy, resources={"cpu": 1}), # 每个回测占 1核
#         param_space=search_space,
#         run_config=tune.RunConfig(
#             num_samples=500,  # 总共搜索 500 组参数
#             callbacks=[mlflow_callback],
#             name="fsm_opt_run"
#         )
#     )

#     results = tuner.fit()
    
#     # 打印全局最强策略
#     best_result = results.get_best_result("sharpe", mode="max")
#     print(f"最优参数: {best_result.config}")
#     print(f"最优夏普: {best_result.metrics['sharpe']}")

# if __name__ == "__main__":
#     main()

# 三、 可观测与可复现（Reproducibility）的三大铁律

# 一旦你进入“研发阶段”，**可复现性（Reproducibility）**就是你的生命线。如果一个策略上周跑夏普是 2.0，这周跑变成了 1.0
# 且你找不到原因，整个投研体系就会崩溃。

# 在 bt_studio 层，你必须强制实施以下三大纪律：

# 1. 绝对随机数种子 (Seed Everything)

# 在 evaluate_strategy 的第一行，必须固定所有涉及到的随机数：

# import numpy as np
# import random
# np.random.seed(config["seed"]) # 从搜索空间传进来
# random.seed(config["seed"])

# 如果你的 C++ 扩展 (Pybind11/Cython) 内部用到了 std::rand() 或其他随机生成器，也必须提供一个 set_seed
# 接口，在入口处调用。

# 2. 代码版本快照 (Git Hash Logging)

# 一个回测跑出来的结果，不仅取决于参数，更取决于你当前的引擎代码。 在启动 Ray Tune 前，通过 Python 获取当前的 Git Commit
# Hash，并作为 Tag 打进 MLflow：

# import subprocess
# git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')

# # 在 MLflow 里记录
# mlflow.set_tag("commit_hash", git_hash)

# 这样几个月后看到一个牛逼的回测结果，你可以 git checkout 到那个 Hash 完美重现它。

# 3. 数据集版本锁定 (Data Versioning)

# 如果你的行情数据是不断更新的，今天跑和下个月跑同样的时间区间可能结果不同（因为除权除息导致历史数据变了）。 给你的原始数据集加上版本号（例如 Parquet
# 文件夹叫 feed_v202605.parquet），并在 MLflow 中记录使用的 Data Version。

# 四、 工作流闭环：整合 Streamlit 和 MLflow

# 现在你已经跑完了。如何分析？

# 打开你之前写的 Streamlit App (app.py)，加入连接 MLflow 的功能。不需要手动复制 Parquet 路径了，让代码自动拉取前五名！

# import streamlit as st
# import mlflow
# from mlflow.tracking import MlflowClient

# # 连接 MLflow
# mlflow.set_tracking_uri("http://localhost:5000")
# client = MlflowClient()

# st.sidebar.title("🏆 Top Strategies")

# # 1. 动态从 MLflow 获取夏普最高的前 5 个回测运行 (Run)
# experiment = client.get_experiment_by_name("Alpha_FSM_Optimization")
# best_runs = client.search_runs(
#     experiment_ids=[experiment.experiment_id],
#     order_by=["metrics.sharpe DESC"],
#     max_results=5
# )

# # 2. 在左侧边栏列出这些最优策略供点击
# selected_run = st.sidebar.radio(
#     "选择要深度分析的策略", 
#     options=best_runs,
#     format_func=lambda run: f"Sharpe: {run.data.metrics.get('sharpe', 0):.2f} | Params: {run.data.params}"
# )

# if selected_run:
#     # 3. 自动下载这个牛逼策略的 Parquet 资产！
#     with st.spinner("正在从 MLflow 提取回测明细数据..."):
#         # 下载 Artifact 到本地临时目录
#         local_dir = client.download_artifacts(selected_run.info.run_id, "parquet_logs")
#         parquet_file = f"{local_dir}/backtest_log.parquet"
        
#         # 4. 把这个文件喂给你之前写的极速 Bokeh 绘图类
#         df_aligned = DataTransformer.load_and_align(parquet_file)
#         plotter = Plot()
#         grid_fig = plotter.plot_from_wide_df(df_aligned)
#         st.bokeh_chart(grid_fig, use_container_width=True)

# 总结

# 至此，你完成了一个**“顶级量化机构”级别的投研环境闭环**：

# 1.  底层：bt_core 像一门无情的大炮，负责提供微秒级的计算和日志。
# 2.  中层：Ray Tune 和 MLflow 组成弹药库，负责大批量生成策略、筛选策略。
# 3.  顶层：Streamlit 是你的望远镜，连接到弹药库，动态渲染那些千万里挑一的顶级策略的资金曲线和内部细节。

# 享受这个流水线吧，这是量化工程师最有成就感的时刻！


# # import sys
# # import signal

# # def handle_sigint(signum, frame):
# #     print("\n[警告] 捕获到 Ctrl+C (SIGINT)，正在安全保存数据并清理内存...")
# #     try:
# #         cerebro._shutdown()
# #     except Exception as e:
# #         print(f"清理时出错: {e}")
# #     finally:
# #         print("清理完成，安全退出。")
# #         sys.exit(0)

# # signal.signal(signal.SIGINT, handle_sigint)
