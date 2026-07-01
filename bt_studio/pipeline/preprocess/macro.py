import polars as pl
import numpy as np

from bt_sdk.ctx import external_mdapi_context
from bt_protocol._protocol import QueryBody
from bt_protocol.constant import RpcTopic

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
        body = QueryBody(start_date=start_date - warm, end_date=end_date, sid=universe) 
    
        obs = mdapi.subscribe(body, RpcTopic.Daily) 
        raw = _collect_stream_sync(obs)

        lazy_frames = []
        for sid, df in raw.items():
            if df.height > 0:
                if "sid" not in df.columns:
                    df = df.with_columns(
                        pl.lit(sid).alias("sid").cast(pl.Binary)
                    )
                else:
                    df = df.with_columns(
                        pl.col("sid").cast(pl.Binary)
                    ) 
                lazy_frames.append(df.lazy())
                
        if not lazy_frames:
            raise ValueError("daily is Null")
            
        daily_lazy = pl.concat(lazy_frames)
        return universe_lazy, daily_lazy 
 

def prepare_tick(start_date: int, end_date: int, sids: list[bytes], warm=10000):
    body = QueryBody(start_date=start_date -warm, end_date=end_date, sid=sids)

    with external_mdapi_context() as mdapi:
        obs = mdapi.subscribe(body, RpcTopic.Tick)
        raw_tick_dict = _collect_stream_sync(obs)

        # filter on 14:55  
        snapshot_dict = {}
        for sid_bytes, tick_df in raw_tick_dict.items():
            if tick_df.height == 0 or "tick" not in tick_df.columns:
                 continue

            tick_df = tick_df.with_columns(pl.lit(sid_bytes).alias("sid").cast(pl.Binary))
            snapshot_dict[sid_bytes] = align_skeleton(tick_df.lazy())
            # import pdb; pdb.set_trace()
        return snapshot_dict


def align_skeleton(tick_lf: pl.LazyFrame) -> pl.LazyFrame:
    # ====================================================================
    # Eager Mode avoid Lazy Optimize Bug
    # ====================================================================
    df = tick_lf.collect(engine="streaming") 
    if df.height == 0:
        return df.lazy()

    # TimeStamp to  Date
    df = df.with_columns(
        (pl.col("tick") * 1000).cast(pl.Int64).cast(pl.Datetime("ms")).alias("tick_dt")
    )

    # ====================================================================
    #  UTC + 8 Hour ---> Aisa Shanghai
    # ====================================================================
    median_hour = df.select(pl.col("tick_dt").dt.hour().median()).item()
    if median_hour is not None and median_hour < 8:
        df = df.with_columns(tick_dt = pl.col("tick_dt").dt.offset_by("8h"))

    # default i8 ---> Int32
    df = df.with_columns([
        (
            pl.col("tick_dt").dt.hour().cast(pl.Int32) * 60 + 
            pl.col("tick_dt").dt.minute().cast(pl.Int32)
        ).alias("to_minutes"),
        pl.col("tick_dt").dt.date().alias("day")
    ])

    # filter by A trading 
    df = df.filter(
        ((pl.col("to_minutes") >= 9 * 60 + 30) & (pl.col("to_minutes") < 11 * 60 + 30)) |
        ((pl.col("to_minutes") >= 13 * 60) & (pl.col("to_minutes") < 15 * 60))
    )

    if df.height == 0:
        return df.lazy()

    # minute ---> 0-239 index
    df = df.with_columns(
        minute_idx = pl.when(pl.col("to_minutes") < 11 * 60 + 30)
                       .then(pl.col("to_minutes") - (9 * 60 + 30))  
                       .otherwise((pl.col("to_minutes") - (13 * 60)) + 120)
                       .cast(pl.Int32)
    ).filter((pl.col("minute_idx") >= 0) & (pl.col("minute_idx") < 240))

    # ====================================================================
    # explode replace Cross Join
    # ====================================================================
    unique_pairs = df.select(["sid", "day"]).unique()
    skeleton_df = (
        unique_pairs
        .with_columns(pl.int_ranges(0, 240, dtype=pl.Int32).alias("minute_idx"))
        .explode("minute_idx")
    )
    
    # join skeleton_df with df to fill missing minute_idx, then forward/backward fill close, and fill other columns with default values
    padded_df = (
        skeleton_df
        .join(df, on=["day", "sid", "minute_idx"], how="left")
        .sort(["day", "sid", "minute_idx"])
        .with_columns([
            pl.col("close").forward_fill().backward_fill().over(["day", "sid"]),
        ])
        .with_columns([
            pl.col("open").fill_null(pl.col("close")),
            pl.col("high").fill_null(pl.col("close")),
            pl.col("low").fill_null(pl.col("close")),
            pl.col("amount").fill_null(0.0),
            pl.col("volume").fill_null(0.0)
        ])
        .drop(["tick", "to_minutes"])
        .rename({"tick_dt": "tick"})
    )
    
    return padded_df.lazy()
