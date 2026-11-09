#! /usr/bin/env python3

from __future__ import annotations

import numpy as np

try:
    import talib
except ImportError:  # pragma: no cover
    talib = None


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data Specifications
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
    op: str                                # AST 算子名称
    func: str                              # TA-Lib 底层 C 函数名
    inputs: Tuple[str, ...]                # 依赖输入列名序列
    params: Tuple[ParamSpec, ...]          # 超参数规范
    output_index: Optional[int] = None     # 多返回值索引
    prior_refs: Tuple[str, ...] = ()       # 经典 Alpha 参照编号
    primitive_equiv: Optional[Dict[str, Any]] = None  # 原语等价 AST 表达式
    equiv_note: str = ""                   # 对应量化物理含义备注
    summary: str = ""                      # 简要物理描述


REGISTRY: Dict[str, TalibOpSpec] = {}


def _register(spec: TalibOpSpec) -> TalibOpSpec:
    REGISTRY[spec.op] = spec
    return spec


# ---------------------------------------------------------------------------
# Whitelist Specifications
# ---------------------------------------------------------------------------

_register(TalibOpSpec(
    op="rsi",
    func="RSI",
    inputs=("close",),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq038",),
    primitive_equiv=None,
    equiv_note="Wilder recursive smooth ---> ts_rank(delta(close,N))",
    summary="Wilder 相对强弱指标",
))

_register(TalibOpSpec(
    op="aroon_osc",
    func="AROONOSC",
    inputs=("high", "low"),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq001",),
    primitive_equiv=None,
    equiv_note="AROONOSC = 100*(距最低点天数 - 距最高点天数)/N, 与极值新鲜度线性等价",
    summary="新高/新低极值新鲜度",
))

_register(TalibOpSpec(
    op="adx",
    func="ADX",
    inputs=("high", "low", "close"),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq024",),
    primitive_equiv=None,
    equiv_note="DMI 方向运动经双平滑后归一, 纯粹的无量纲趋势强度",
    summary="趋势强度 (无量纲 Regime 开关)",
))

_register(TalibOpSpec(
    op="mfi",
    func="MFI",
    inputs=("high", "low", "close", "volume"),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq007", "wq043"),
    primitive_equiv=None,
    equiv_note="typical price x volume 资金流向比, 即成交量加权 RSI",
    summary="量加权动量 (量价确认)",
))

_register(TalibOpSpec(
    op="trix",
    func="TRIX",
    inputs=("close",),
    params=(ParamSpec("timeperiod", 2, 250, default=14),),
    prior_refs=("wq052",),
    primitive_equiv=None,
    equiv_note="三重 EMA 变化率, 超平滑长周期动量",
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
    equiv_note="精确等价: CMO = 100 * ΣΔ / Σ|Δ|; 符号和连续化",
    summary="涨跌幅归一差",
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
    equiv_note="RSI 在自身 N 日区间中的相对位置; fastd_period 冻结为 3",
    summary="RSI 区间相对位置 (嵌套归一动量)",
))

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
    equiv_note="三窗口买卖压力复合动量; p1/p3 冻结为 7/28, 保证至多 1 个自由维度",
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
# Validation & Warmup Calculations
# ---------------------------------------------------------------------------

def validate_node(node: Any) -> List[str]:
    if not isinstance(node, dict):
        return ["节点必须是 JSON 字典对象"]

    op = node.get("op")
    spec = REGISTRY.get(op)
    if spec is None:
        return [f"未知算子 {op!r}, 当前白名单: {sorted(REGISTRY)}"]

    args = node.get("args")
    if not isinstance(args, list):
        return [f"{op}: args 必须是列表"]

    n_expected = len(spec.inputs) + len(spec.params)
    if len(args) != n_expected:
        return [
            f"{op}: args 应有 {n_expected} 项 "
            f"({len(spec.inputs)} 个字段引用 + {len(spec.params)} 个参数), 实际 {len(args)} 项"
        ]

    errors: List[str] = []

    # 1. 字段引用校验
    for i, col in enumerate(spec.inputs):
        a = args[i]
        if not (isinstance(a, dict) and "col" in a and str(a.get("col")).strip() == col):
            errors.append(f'{op}: args[{i}] 应为字段引用 {{"col": "{col}"}}')

    # 2. 参数类型与边界校验
    for j, p in enumerate(spec.params):
        v = args[len(spec.inputs) + j]

        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{op}: 参数 {p.name} 必须为数值, 实际 {v!r}")
            continue

        if isinstance(v, float) and not v.is_integer():
            errors.append(f"{op}: 参数 {p.name} 必须为整数, 实际 {v!r}")
            continue

        v_int = int(v)

        if p.fixed is not None and v_int != p.fixed:
            errors.append(f"{op}: 参数 {p.name} 已冻结为 {p.fixed}, 实际传入 {v_int}")

        if not (p.minimum <= v_int <= p.maximum):
            errors.append(f"{op}: 参数 {p.name} 应在 [{p.minimum}, {p.maximum}] 区间内, 实际 {v_int}")

    return errors


def format_ops_help() -> str:
    lines = ["[TA-Lib 白名单算子](参数必须显式给出且为整数; <p=N> 表示已冻结约束):"]
    for spec in REGISTRY.values():
        cols = ", ".join(f'{{"col": "{c}"}}' for c in spec.inputs)
        ps = ", ".join(
            f"<{p.name}={p.fixed}>" if p.fixed is not None else f"<{p.name}>"
            for p in spec.params
        )
        prior = f" [{','.join(spec.prior_refs)}]" if spec.prior_refs else ""
        lines.append(f"  {spec.op}({cols}, {ps}){prior}  # {spec.summary}")
    lines.append(
        "  约束: 严禁堆叠同质化经典指标; MOM/ROC/APO/BOP/WILLR 等请用原语层组合，不在白名单内."
    )
    return "\n".join(lines)
