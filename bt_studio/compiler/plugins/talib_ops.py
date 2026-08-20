#! /usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    import talib
except ImportError:  # pragma: no cover
    talib = None


# ---------------------------------------------------------------------------
# dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    name: str
    minimum: int = 2
    maximum: int = 250
    fixed: Optional[int] = None
    default: Optional[int] = None

    def __post_init__(self) -> None:
        if self.fixed is not None and self.default is None:
            object.__setattr__(self, "default", self.fixed)


@dataclass(frozen=True)
class TalibOpSpec:
    op: str                                # AST 
    func: str                              # talib C 
    inputs: Tuple[str, ...]               
    params: Tuple[ParamSpec, ...]          
    output_index: Optional[int] = None     
    prior_refs: Tuple[str, ...] = ()       
    primitive_equiv: Optional[Dict[str, Any]] = None  # Primitive AST
    equiv_note: str = ""                  
    summary: str = ""                    


REGISTRY: Dict[str, TalibOpSpec] = {}


def _register(spec: TalibOpSpec) -> TalibOpSpec:
    REGISTRY[spec.op] = spec
    return spec


_register(TalibOpSpec(
    op="rsi",
    func="RSI",
    inputs=("close",),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq038",),
    primitive_equiv=None,
    equiv_note="Wilder recursive smooth ---> ts_rank(delta(close,N))",
    summary="Wilder",
))

_register(TalibOpSpec(
    op="aroon_osc",
    func="AROONOSC",
    inputs=("high", "low"),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq001",),
    primitive_equiv=None,
    equiv_note="AROONOSC = 100*(距最低点天数 - 距最高点天数)/N, 与极值新鲜度线性等价 (WQ#1 同族)",
    summary="新高/新低新鲜度指标",
))

_register(TalibOpSpec(
    op="adx",
    func="ADX",
    inputs=("high", "low", "close"),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq024",),
    primitive_equiv=None,
    equiv_note="DMI 方向运动经双平滑后归一, 用作趋势/横盘 regime 开关 (对应 WQ#24 均线 regime)",
    summary="趋势强度(无量纲 regime 开关)",
))

_register(TalibOpSpec(
    op="mfi",
    func="MFI",
    inputs=("high", "low", "close", "volume"),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq007", "wq043"),
    primitive_equiv=None,
    equiv_note="typical price x volume 的资金流向比, 即成交量加权 RSI (对应 WQ#7/#43 量价确认)",
    summary="量加权动量 (量价确认)",
))

_register(TalibOpSpec(
    op="trix",
    func="TRIX",
    inputs=("close",),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq052",),
    primitive_equiv=None,
    equiv_note="三重 EMA 的百分比变化率, 超平滑长周期趋势方向 (对应 WQ#52 长周期动量)",
    summary="三重平滑趋势动量",
))

_register(TalibOpSpec(
    op="cmo",
    func="CMO",
    inputs=("close",),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq030",),
    primitive_equiv={
        "op": "mul",
        "args": [
            {
                "op": "div",
                "args": [
                    {"op": "sum", "args": [{"op": "delta", "args": [{"col": "close"}, 1]}, "N"]},
                    {"op": "sum", "args": [{"op": "abs", "args": [{"op": "delta", "args": [{"col": "close"}, 1]}]}, "N"]},
                ],
            },
            100,
        ],
    },
    equiv_note="精确等价: CMO = 100 * ΣΔ / Σ|Δ|; talib 版保留作原语层数值校准基准",
    summary="涨跌幅归一差 (WQ#30 符号和的连续化)",
))

_register(TalibOpSpec(
    op="stoch_rsi",
    func="STOCHRSI",
    inputs=("close",),
    params=(
        ParamSpec("timeperiod", 2, 250, default=14),
        ParamSpec("fastk_period", 2, 100, default=5),
        ParamSpec("fastd_period", 2, 100, fixed=3, default=3),
    ),
    output_index=0,
    prior_refs=(),
    primitive_equiv=None,
    equiv_note="RSI 在自身 N 日区间中的相对位置 (嵌套归一化); fastd_period 冻结为 3",
    summary="RSI 区间相对位置 (嵌套归一动量)",
))

# ---- Frozen Vary include fixed restrict ----

_register(TalibOpSpec(
    op="ultosc",
    func="ULTOSC",
    inputs=("high", "low", "close"),
    params=(
        ParamSpec("timeperiod1", 2, 250, fixed=7, default=7),
        ParamSpec("timeperiod2", 2, 250, default=14), 
        ParamSpec("timeperiod3", 2, 250, fixed=28, default=28),
    ),
    prior_refs=(),
    equiv_note="三窗口买卖压力复合动量; p1/p3 冻结 (7/28), 只暴露 p2, 满足单特征至多 1 个自由参数规则",
    summary="终极振荡器 (p1/p3 冻结为 7/28)",
))

for _name, _idx, _note in (
    ("macd", 0, "MACD 快慢线差"),
    ("macd_signal", 1, "MACD 信号线"),
    ("macd_hist", 2, "MACD 柱 (快线-信号线)"),
):
    _register(TalibOpSpec(
        op=_name,
        func="MACD",
        inputs=("close",),
        params=(
            ParamSpec("fastperiod", 2, 250, fixed=12, default=12),
            ParamSpec("slowperiod", 2, 250, fixed=26, default=26),
            ParamSpec("signalperiod", 2, 250, fixed=9, default=9),
        ),
        output_index=_idx,
        prior_refs=(),
        equiv_note=f"{_note}; 12/26/9 参数完全冻结, 不允许自由变参",
        summary=_note + " (参数冻结 12/26/9)",
    ))


# ---------------------------------------------------------------------------
# Static AST Validate
# ---------------------------------------------------------------------------

def validate_node(node: Any) -> List[str]:
    if not isinstance(node, dict):
        return ["节点必须是 JSON 对象"]

    op = node.get("op")
    spec = REGISTRY.get(op)
    if spec is None:
        return [f"未知算子 {op!r}, 当前白名单: {sorted(REGISTRY)}"]

    args = node.get("args")
    if not isinstance(args, list):
        return [f"{op}: args 必须是数组"]

    n_expected = len(spec.inputs) + len(spec.params)
    if len(args) != n_expected:
        return [
            f"{op}: args 应有 {n_expected} 项 "
            f"({len(spec.inputs)} 个字段引用 + {len(spec.params)} 个整数参数), 实际 {len(args)} 项"
        ]

    errors: List[str] = []

    for i, col in enumerate(spec.inputs):
        a = args[i]
        if not (isinstance(a, dict) and set(a) == {"col"} and a.get("col") == col):
            errors.append(f'{op}: args[{i}] 应为字段引用 {{"col": "{col}"}}')

    for j, p in enumerate(spec.params): # p is ParamSpec
        v = args[len(spec.inputs) + j]
        if isinstance(v, bool) or not isinstance(v, int):
            errors.append(f"{op}: 参数 {p.name} 必须为整数, 实际 {v!r}")
            continue

        if p.fixed is not None and v != p.fixed:
            errors.append(f"{op}: 参数 {p.name} 已冻结为 {p.fixed}, 实际传入 {v}")

        if not (p.minimum <= v <= p.maximum):
            errors.append(f"{op}: 参数 {p.name} 应在 [{p.minimum}, {p.maximum}] 区间内, 实际 {v}")

    return errors


_LOOKBACK: Dict[str, Callable[..., int]] = {
    "rsi": lambda t: t,
    "aroon_osc": lambda t: t,
    "adx": lambda t: 2 * t,
    "mfi": lambda t: t,
    "trix": lambda t: 3 * t + 10,
    "cmo": lambda t: t,
    "stoch_rsi": lambda t, k, d: t + k + d,
    "ultosc": lambda p1, p2, p3: p1 + p2 + p3,
    "macd": lambda f, s, g: s + g + 50,
}
_LOOKBACK["macd_signal"] = _LOOKBACK["macd"]
_LOOKBACK["macd_hist"] = _LOOKBACK["macd"]


def warmup_bars(node: Dict[str, Any]) -> int:
    op = node.get("op", "")
    spec = REGISTRY.get(op)
    if not spec:
        return 0

    args = node.get("args", [])
    raw_params = args[len(spec.inputs):]

    params = []
    for p in raw_params:
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            params.append(int(p))

    fn = _LOOKBACK.get(spec.op)
    if fn is None:
        return max(params) if params else 0

    try:
        return int(fn(*params))
    except (TypeError, ValueError):
        return max(params) if params else 0


# ---------------------------------------------------------------------------
# Execute and Prompt Render
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame, node: Dict[str, Any]) -> pd.Series:
    if talib is None:
        raise RuntimeError("pip install TA-Lib")

    spec = REGISTRY[node["op"]]
    arrays = [df[c].to_numpy(dtype=float) for c in spec.inputs]
    params = list(node["args"][len(spec.inputs):])

    raw = getattr(talib, spec.func)(*arrays, *params)
    if isinstance(raw, tuple):
        raw = raw[spec.output_index if spec.output_index is not None else 0]

    return pd.Series(raw, index=df.index, name=spec.op)


def format_ops_help() -> str:
    lines = ["[TA-Lib 白名单算子](参数必须显式给出且为整数; =N 表示已冻结):"]
    for spec in REGISTRY.values():
        cols = ", ".join(f'{{"col": "{c}"}}' for c in spec.inputs)
        ps = ", ".join(
            f"<{p.name}={p.fixed}>" if p.fixed is not None else f"<{p.name}>"
            for p in spec.params
        )
        prior = f" [{','.join(spec.prior_refs)}]" if spec.prior_refs else ""
        lines.append(f"  {spec.op}({cols}, {ps}){prior}  # {spec.summary}")
    lines.append(
        "  严禁堆叠名指标组合 (RSI+MACD+KDJ); "
        "MOM/ROC/APO/BOP/WILLR 等请用原语层组合，不在白名单内."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reject Due to Duplicate and multi dimensions
# ---------------------------------------------------------------------------

REJECTED: Dict[str, str] = {
    "MOM": "delta(close, N) — 原语已有",
    "ROC": "roc — 原语已有",
    "APO": "sub(sma(close,f), sma(close,s))",
    "PPO": "APO / sma(close,s) — 归一化均线差",
    "BOP": "div(sub(close,open), sub(high,low)) — 即 WQ#101",
    "WILLR": "-div(sub(max(high,N),close), sub(max(high,N),min(low,N))) — 区间位置, WQ#24 同族",
    "AROON": "双输出, 只取 AROONOSC",
    "STOCH/STOCHF": "区间位置同 WILLR, %K/%D 平滑可由 sma 组合",
    "MINUS_DI/PLUS_DI/MINUS_DM/PLUS_DM": "ADX 内部组件",
    "DX/ADXR": "ADX 原始值/平滑值",
    "MACDEXT/MACDFIX": "MACD 冗余变体, 参数不显式",
}
