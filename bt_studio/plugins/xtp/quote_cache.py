from __future__ import annotations

import asyncio
from array import array
import math
from typing import Any, Optional, Sequence

# END-labeled
_TRADING_BARS = 240
_MORNING_START = 9 * 60 + 30
_MORNING_END = 11 * 60 + 30
_AFTERNOON_START = 13 * 60
_AFTERNOON_END = 15 * 60
_NAN = float("nan")

# XTP exchange enum (matches XTP_EXCHANGE_TYPE in xtp_api_data_type.h).
XTP_EXCHANGE_SH = 1       
XTP_EXCHANGE_SZ = 2       
XTP_EXCHANGE_UNKNOWN = 3  


def data_time_to_bar_idx(data_time: int) -> int:
    """Map XTP data_time (YYYYMMDDHHMMSSmmm) onto 0..239 END-labeled 1-minute bar index (-1 if outside session)."""
    hhmmss = (int(data_time) // 1000) % 1000000
    hh, mm, ss = hhmmss // 10000, (hhmmss // 100) % 100, hhmmss % 100
    total_min = hh * 60 + mm
    end_min = total_min + (1 if ss > 0 else 0)

    # Exact session opens (the entire open second, incl. sub-second ms) land on
    # the first bar; the strict `>` in the range checks below would drop them.
    if total_min == _MORNING_START and ss == 0:
        return 0
    if total_min == _AFTERNOON_START and ss == 0:
        return 120
    if _MORNING_START < end_min <= _MORNING_END:
        return end_min - (_MORNING_START + 1)
    if _AFTERNOON_START < end_min <= _AFTERNOON_END:
        return 120 + (end_min - (_AFTERNOON_START + 1))
    return -1


class TopkQuoteCache:
    """Zero-allocation, stdlib C-array backed intraday 240-bar quote cache for Top-K PIT universe sids.

    Uses flat contiguous C-double buffers (`array('d')` of length N_sids * 240)
    with zero third-party dependencies, updating OHLCV + Amount in O(1) per tick.

    ``client`` is duck-typed (any ``ZmqClient``-like with ``connect``/``call``/
    ``subscribe``); the XTP-specific market-data schema (``data_time``, ``qty``,
    ``turnover``, limit prices) is the only backend coupling here.
    """

    def __init__(
        self,
        client: "ZmqClient",
        sids: Sequence[str | bytes],
        hfq_scales: Optional[dict[str, float]] = None,
    ):
        self.client = client
        self.tickers: list[str] = [s.decode() if isinstance(s, bytes) else str(s) for s in sids]
        self.sid_to_idx: dict[str, int] = {t: i for i, t in enumerate(self.tickers)}
        self._n = len(self.tickers)

        self.scales = array("d", [1.0]) * self._n
        if hfq_scales:
            for t, sc in hfq_scales.items():
                if t in self.sid_to_idx and sc > 0:
                    self.scales[self.sid_to_idx[t]] = float(sc)

        self.trading_day: int = 0
        self.reset_day(0)
        self.frames_received: int = 0
        self.ticks_ingested: int = 0
        self._pump_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()

    def reset_day(self, ymd: int = 0) -> None:
        """Reset intraday 240-bar and cumulative counters cleanly on session rollover."""
        total_cells = self._n * _TRADING_BARS
        self._open = array("d", [_NAN]) * total_cells
        self._high = array("d", [_NAN]) * total_cells
        self._low = array("d", [_NAN]) * total_cells
        self._close = array("d", [_NAN]) * total_cells
        self._volume = array("d", [0.0]) * total_cells
        self._amount = array("d", [0.0]) * total_cells
        self._prev_cum_vol = array("d", [0.0]) * self._n
        self._prev_cum_amt = array("d", [0.0]) * self._n
        self.preclose = array("d", [0.0]) * self._n
        self.upper_limit = array("d", [0.0]) * self._n
        self.lower_limit = array("d", [0.0]) * self._n
        self.trading_day = int(ymd)

    @staticmethod
    def _exchange_id(ticker: str) -> int:
        # 北交所 (43/83/87/88/92 开头) 本 XTP SDK 无对应枚举, 显式返回 UNKNOWN
        # 而不是误映射到深交所(2), 避免静默订阅到错误市场。
        if ticker.startswith(("43", "83", "87", "88", "92")):
            return XTP_EXCHANGE_UNKNOWN
        # 上交所: 6xx A股/科创板, 5xx 基金/ETF, 9xx B股
        if ticker.startswith(("6", "5", "9")):
            return XTP_EXCHANGE_SH
        # 深交所: 0xx/3xx A股/创业板, 2xx B股, 1xx 基金/ETF
        return XTP_EXCHANGE_SZ

    def on_depth_tick(self, md: dict[str, Any]) -> bool:
        """Ingest one XTP MarketData dict in O(1) time into C-double buffers."""
        ticker = md.get("ticker", "")
        idx = self.sid_to_idx.get(ticker, -1)
        if idx < 0:
            return False

        px = float(md.get("last_price") or 0.0)
        if px <= 0.0:
            return False

        dt_raw = int(md.get("data_time") or 0)
        tick_ymd = dt_raw // 1000000000
        if tick_ymd > 0 and tick_ymd != self.trading_day:
            self.reset_day(tick_ymd)

        scale = self.scales[idx]
        if self.preclose[idx] == 0.0 and md.get("pre_close_price"):
            self.preclose[idx] = float(md["pre_close_price"]) * scale
            self.upper_limit[idx] = float(md.get("upper_limit_price") or 0.0) * scale
            self.lower_limit[idx] = float(md.get("lower_limit_price") or 0.0) * scale

        bar_idx = data_time_to_bar_idx(dt_raw)
        if bar_idx < 0:
            return False

        adj_px = px * scale
        cum_vol = float(md.get("qty") or 0.0)
        cum_amt = float(md.get("turnover") or 0.0)
        prev_v = self._prev_cum_vol[idx]
        prev_a = self._prev_cum_amt[idx]
        delta_vol = (cum_vol - prev_v) if cum_vol > prev_v else 0.0
        delta_amt = (cum_amt - prev_a) if cum_amt > prev_a else 0.0
        if cum_vol >= prev_v:
            self._prev_cum_vol[idx] = cum_vol
        if cum_amt >= prev_a:
            self._prev_cum_amt[idx] = cum_amt

        pos = idx * _TRADING_BARS + bar_idx
        cur_open = self._open[pos]
        if math.isnan(cur_open):
            self._open[pos] = adj_px
            self._high[pos] = adj_px
            self._low[pos] = adj_px
        else:
            if adj_px > self._high[pos]:
                self._high[pos] = adj_px
            if adj_px < self._low[pos]:
                self._low[pos] = adj_px
        self._close[pos] = adj_px
        self._volume[pos] += delta_vol / scale
        self._amount[pos] += delta_amt
        self.ticks_ingested += 1
        return True

    async def start(self) -> dict[str, Any]:
        """Register quote event stream first (zero-gap), then subscribe Top-K tickers."""
        await self.client.connect()
        self._pump_task = asyncio.create_task(self._pump_loop())
        await asyncio.wait_for(self._ready.wait(), timeout=self.client.timeout)
        sub_specs = [{"ticker": t, "exchange_id": self._exchange_id(t)} for t in self.tickers]
        return await self.client.call("quote.subscribe_market_data", {"tickers": sub_specs})

    async def stop(self) -> None:
        if self._pump_task:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None

    async def _pump_loop(self) -> None:
        async for msg in self.client.subscribe("quote.subscribe_events", {}, idle_timeout=0.0):
            self.frames_received += 1
            res = msg.get("result") if isinstance(msg, dict) else None
            if not isinstance(res, dict):
                continue
            if res.get("subscribed"):
                self._ready.set()
                continue
            if res.get("event") == "onDepthMarketData" and isinstance(res.get("data"), dict):
                self.on_depth_tick(res["data"])

    def snapshot_bars(self, up_to_bar_idx: int = 239) -> dict[str, Any]:
        """Export causally forward-filled intraday bars [0..up_to_bar_idx] as columnar buffers."""
        cut = min(_TRADING_BARS, max(1, int(up_to_bar_idx) + 1))
        out_sid: list[bytes] = []
        out_bidx = array("i")
        out_open = array("d")
        out_high = array("d")
        out_low = array("d")
        out_close = array("d")
        out_vol = array("d")
        out_amt = array("d")
        bar_range = array("i", range(cut))

        for i, ticker in enumerate(self.tickers):
            row_base = i * _TRADING_BARS
            has_obs = False
            last_c = _NAN
            for b in range(cut):
                c = self._close[row_base + b]
                if not math.isnan(c):
                    has_obs = True
                    break
            if not has_obs:
                continue

            sid_b = ticker.encode()
            out_sid.extend([sid_b] * cut)
            out_bidx.extend(bar_range)

            for b in range(cut):
                pos = row_base + b
                c = self._close[pos]
                if not math.isnan(c):
                    last_c = c
                    o, h, l = self._open[pos], self._high[pos], self._low[pos]
                else:
                    c = last_c
                    o = h = l = last_c
                out_open.append(o)
                out_high.append(h)
                out_low.append(l)
                out_close.append(c)
                out_vol.append(self._volume[pos])
                out_amt.append(self._amount[pos])

        return {
            "sid": out_sid,
            "bar_idx": out_bidx,
            "open": out_open,
            "high": out_high,
            "low": out_low,
            "close": out_close,
            "volume": out_vol,
            "amount": out_amt,
        }
