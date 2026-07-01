import polars as pl

if __name__ == "__main__":

    hf_df = pl.read_parquet("/Users/hengxinliu/startup/bt_studio/result/fsm/train/hf_2004.parquet")
    daily_df = pl.read_parquet("/Users/hengxinliu/startup/bt_studio/result/fsm/global_daily.parquet")

    print("HF 数据行数:", hf_df.height)
    if hf_df.height > 0:
        print("HF 数据日期范围:", hf_df["day"].min(), "到", hf_df["day"].max())

    print("Daily 数据日期范围:", daily_df["day"].min(), "到", daily_df["day"].max())
