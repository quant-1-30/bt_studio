import polars as pl
from bt_sdk.ctx import external_mdapi_context
from bt_protocol._protocol import QueryBody

from bt_studio.utils.common import _collect_stream_sync


def prepare_macro(start_date: int, end_date: int, benchmark: bytes, warm=10000):
    from bt_sdk.ctx import external_mdapi_context
    
    with external_mdapi_context() as mdapi:
        # =======================================================
        # Universe PIT
        # =======================================================
        inst_df = mdapi.get_instrument()
        valid_meta = inst_df.filter(pl.col("delist") > start_date)

        universe_lazy = valid_meta.select([
            pl.col("sid").cast(pl.Binary), 
            pl.col("first_trading").cast(pl.Int32)
        ]).lazy()
        
        # =======================================================
        # Universe Daily
        # =======================================================
        universe = valid_meta["sid"].cast(pl.Binary).to_list() 
        body = QueryBody(start_date=start_date - warm, end_date=end_date, sid=[universe]) 
    
        obs = mdapi.subscribe(body, RpcTopic.Daily) 
        raw = _collect_stream_sync(obs)

        lazy_frames = []
        for sid, df in raw.items():
            if df.height > 0:
                if "sid" not in df.columns:
                    df = df.with_columns(pl.lit(sid).alias("sid")) # Literal 
                lazy_frames.append(df.lazy())
                
        if not lazy_frames:
            raise ValueError("获取到的订阅日频数据为空，无法构建 daily_lazy")
            
        daily_lazy = pl.concat(lazy_frames)
        return universe_lazy, daily_lazy 
 

def prepare_tick(start_date: int, end_date: int, sids: list[bytes], warm=10000):
    body = QueryBody(start_date=start-warm, end_date=end_date, sid=sids)

    obs = mdapi.subscribe(body, RpcTopic.Tick)
    raw_tick_dict = _collect_stream_sync(obs)

    # filter on 14:55  
    snapshot_dict = {}
    for sid_bytes, tick_df in raw_tick_dict.items():

        if tick_df.height == 0 or "tick" not in tick_df.columns:
             pass
        snapshot_dict[sid_bytes] = align_skeleton(tick_df.lazy())
    return snapshot_dict


def align_skeleton(tick_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
        9:30 - 11:30 / 13:00 - 14:59 to ensure 240 minute
    """
    to_minutes = pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute()
    
    minute_idx_expr = (
        pl.when(to_minutes < 11 * 60 + 30)
        .then(to_minutes - (9 * 60 + 30))  
        .otherwise((to_minutes - (13 * 60)) + 120) 
    ).cast(pl.Int32)
    
    base_lf = (
        tick_lf
        .with_columns(minute_idx_expr.alias("minute_idx"))
        .filter((pl.col("minute_idx") >= 0) & (pl.col("minute_idx") < 240))
    )
    
    unique_pairs = base_lf.select(["sid", "day"]).unique()
    
    skeleton_lf = (
        unique_pairs
        .join(
            pl.LazyFrame({"minute_idx": np.arange(240, dtype=np.int32)}), 
            how="cross"
        )
    )
    
    padded_lf = (
        skeleton_lf
        .join(base_lf, on=["day", "sid", "minute_idx"], how="left")
        .sort(["day", "sid", "minute_idx"])
        .with_columns([
            pl.col("close").forward().backward().over(["day", "sid"]),
        ])
        .with_columns([
            pl.col("open").fill_null(pl.col("close")),
            pl.col("high").fill_null(pl.col("close")),
            pl.col("low").fill_null(pl.col("close")),
            pl.col("amount").fill_null(0.0),
            pl.col("volume").fill_null(0.0)
        ])
    )
    
    return padded_lf
