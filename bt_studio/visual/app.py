import streamlit as st
import pandas as pd
import duckdb
import numpy as np
from plot import Plot # 导入你之前的 Plot 类

# 页面配置
st.set_page_config(page_title="自包含日志分析工具", layout="wide")

@st.cache_data
def load_comprehensive_data(file_path, symbol):
    """
    核心：不再读取外部数据库。
    直接从 Parquet 日志文件的不同层/表中读取并拼装。
    """
    con = duckdb.connect(database=':memory:')
    
    # 使用 DuckDB 直接查询 Parquet
    # 1. 读取行情快照
    # 2. Left Join 订单信号 (buy/sell)
    # 3. Left Join 指标数据 (假设记录在 indicator 表中)
    query = f"""
        WITH ohlcv AS (
            SELECT * FROM read_parquet('{file_path}') 
            WHERE sid = '{symbol}' AND table_type = 'vtohlcv'
        ),
        orders AS (
            SELECT 
                datetime, 
                sum(case when direction = 'BUY' then size else 0 end) as buy,
                sum(case when direction = 'SELL' then size else 0 end) as sell
            FROM read_parquet('{file_path}')
            WHERE sid = '{symbol}' AND table_type = 'vtorder'
            GROUP BY datetime
        )
        SELECT 
            o.*, 
            COALESCE(s.buy, 0) as buy, 
            COALESCE(s.sell, 0) as sell
        FROM ohlcv o
        LEFT JOIN orders s ON o.datetime = s.datetime
        ORDER BY o.datetime
    """
    df = con.execute(query).df()
    return df

# --- UI 侧边栏 ---
st.sidebar.header("📁 数据导入")
log_file = st.sidebar.text_input("日志文件路径 (Parquet)", "backtest_result.parquet")

if log_file:
    # 获取 sid 列表（同样从 log 文件直接读）
    symbols = duckdb.query(f"SELECT DISTINCT sid FROM read_parquet('{log_file}')").df()['sid'].tolist()
    selected_sid = st.sidebar.selectbox("选择证券", symbols)
    
    # 绘图配置
    use_candle = st.sidebar.toggle("启用 K线", value=True)
    
    if st.sidebar.button("生成分析报表"):
        df = load_comprehensive_data(log_file, selected_sid)
        
        if not df.empty:
            st.subheader(f"回测详情: {selected_sid}")
            
            # --- 实例化并调用你的 Plot 类 ---
            plotter = Plot()
            
            # 适配你的 datasource 结构
            # 这里由于数据已经自包含在 df 里，我们直接手动填充 plotter.datasource
            from bokeh.models import ColumnDataSource
            
            # 1. 转换时间
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # 2. 构建 Plot 期望的数据源
            plotter.datasource = {
                "main": ColumnDataSource(df),
                "strategy": ColumnDataSource(df[['datetime', 'buy', 'sell']])
            }
            
            # 3. 运行你原来的 plotdata 逻辑
            # 注意：你需要对原 Plot 类做微调，使其 plotdata 接受这个 datasource 字典
            grid_plot = plotter.plot_from_integrated_df(df, candle=use_candle)
            
            # 4. Streamlit 显示
            st.bokeh_chart(grid_plot, use_container_width=True)
            
            # 展示 Account/指标等其他信息
            with st.expander("查看原始数据"):
                st.dataframe(df)
