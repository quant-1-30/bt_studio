import polars as pl
import numpy as np

from bt_sdk.ctx import external_mdapi_context
from bt_protocol._protocol import QueryBody
from bt_protocol.constant import RpcTopic

from bt_studio.utils.common import _collect_stream_sync
from .panel import align_skeleton


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
