 四、 工作流闭环：整合 Streamlit 和 MLflow

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
