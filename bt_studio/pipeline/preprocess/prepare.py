import polars as pl
import numpy as np

from bt_sdk.ctx import external_mdapi_context
from bt_protocol._protocol import QueryBody
from bt_protocol.constant import RpcTopic

from bt_studio.pipeline.utils import _collect_stream_sync


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
        return snapshot_dict


def align_skeleton(tick_lf: pl.LazyFrame) -> pl.LazyFrame:
    processed_lf = tick_lf.with_columns(
        (pl.col("tick") * 1000)
        .cast(pl.Int64)
        .cast(pl.Datetime("ms"))
        .alias("tick_dt")
    ).with_columns(
        pl.col("tick_dt").dt.date().alias("day")
    )

    processed_lf = processed_lf.with_columns(
        pl.when(
            pl.col("tick_dt").dt.hour().median().over(["day"]) < 8 # 9 -15 median > 8
        )  
        .then(pl.col("tick_dt").dt.offset_by("8h"))
        .otherwise(pl.col("tick_dt"))
        .alias("tick_dt")
    )

    processed_lf = processed_lf.with_columns(
        [
            (
                pl.col("tick_dt").dt.hour().cast(pl.Int32) * 60
                + pl.col("tick_dt").dt.minute().cast(pl.Int32)
            ).alias("to_minutes"),
            pl.col("tick_dt").dt.date().alias("day"),
        ]
    ).filter(
        ((pl.col("to_minutes") >= 9 * 60 + 30) & (pl.col("to_minutes") < 11 * 60 + 30)) |
        ((pl.col("to_minutes") >= 13 * 60) & (pl.col("to_minutes") < 15 * 60))
    )

    # timestamp to minute_idx
    processed_lf = processed_lf.with_columns(
        minute_idx=pl.when(pl.col("to_minutes") < 11 * 60 + 30)
        .then(pl.col("to_minutes") - (9 * 60 + 30))
        .otherwise((pl.col("to_minutes") - (13 * 60)) + 120)
        .cast(pl.Int32)
    )

    # ====================================================================
    # 240 skeleton
    # ====================================================================
    unique_pairs = processed_lf.select(["sid", "day"]).unique()

    skeleton_lf = unique_pairs.with_columns(
        pl.int_ranges(0, 240, dtype=pl.Int32).alias("minute_idx")
    ).explode("minute_idx")

    # ====================================================================
    # padding
    # ====================================================================
    padded_lf = (
        skeleton_lf.join(
            processed_lf, on=["day", "sid", "minute_idx"], how="left"
        )
        .sort(["day", "sid", "minute_idx"])
        # ensure padding in sid
        .with_columns(
            [
                pl.col("close")
                .forward_fill() 
                .over(["day", "sid"])
                .alias("close"),
            ]
        )
        .with_columns(
            [
                pl.col("close")
                .backward_fill()
                .over(["day", "sid"])
                .alias("close")
            ]
        )
        .with_columns(
            [
                pl.col("open").fill_null(pl.col("close")),
                pl.col("high").fill_null(pl.col("close")),
                pl.col("low").fill_null(pl.col("close")),
                pl.col("amount").fill_null(0.0),
                pl.col("volume").fill_null(0.0),
            ]
        )
        .drop(["to_minutes", "tick_dt", "tick"])
        # .rename({"tick_dt": "tick"})
    )
    return padded_lf
