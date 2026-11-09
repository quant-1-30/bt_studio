#!/usr/bin/env python3
"""compiler 窗口算子因果性回归测试 (对应 1.2 未来函数修复)。

覆盖:
    1. 截面算子 (cs_rank/cs_demean/cs_zscore) 只聚合同时点 (day, bar_idx)
       的横截面 —— 对未来 bar 的扰动不变 (原先 over("day") 把当日全部
       240 根 bar 含未来 bar 卷入, 是分钟面板上的未来函数);
    2. 时序算子 (ref/delta/mean/cum_* 等) 按标的分区 over("sid") ——
       多标的 concat 帧上不再把上一只票的尾部混入下一只票的开头,
       cum_sum 不再跨标的累计;
    3. talib 桥按标的分组执行, 与逐票调用 talib 结果一致;
    4. 跨族窗口嵌套 (cs 套 ts / ts 套 cs) 在编译期被显式拒绝
       (polars 1.43 对不同分区键的窗口嵌套会静默输出全 null);
       同族嵌套与并列组合、recipe 跨步组合均正常。
"""

import os
import sys
import datetime

import numpy as np
import polars as pl
import pytest
import talib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bt_studio.compiler.ast import (
    compile_ast,
    compile_recipe,
    ASTCompilationError,
)

N_SIDS = 4
N_BARS = 60
DAYS = [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]


def _panel(seed: int = 3) -> pl.DataFrame:
    """(sid, day, bar_idx) 分钟面板; 各标的价值量级分离, 便于检测跨标的污染。"""
    rng = np.random.RandomState(seed)
    rows = []
    for si in range(N_SIDS):
        for d in DAYS:
            for b in range(N_BARS):
                rows.append({
                    "sid": f"60{si:04d}",
                    "day": d,
                    "bar_idx": b,
                    "x": float(si * 1000 + b + 1 + rng.normal(0, 0.1)),
                })
    return pl.DataFrame(rows).sort(["sid", "day", "bar_idx"]).with_columns(
        pl.col("bar_idx").cast(pl.Int32))


def _run(df: pl.DataFrame, ast: dict) -> tuple[pl.DataFrame, str]:
    expr, name = compile_ast(ast, check_causal=True)
    return df.with_columns(expr.alias(name)), name


# ============================================================================
# 1. 截面算子: 对未来数据扰动不变 (核心未来函数回归)
# ============================================================================

@pytest.mark.parametrize("op", ["cs_rank", "cs_demean", "cs_zscore"])
def test_cs_ops_invariant_to_future_bars(op):
    df = _panel()
    cut = 30
    base, name = _run(df, {"op": op, "args": [{"col": "x"}]})

    # 扰动: 第一天 cut 之后的所有 bar, 以及之后全部交易日
    day0 = DAYS[0]
    pert = df.with_columns(
        pl.when((pl.col("day") > day0) | ((pl.col("day") == day0) & (pl.col("bar_idx") > cut)))
        .then(pl.col("x") * -3.0 + 100.0)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    after, _ = _run(pert, {"op": op, "args": [{"col": "x"}]})

    kept_base = base.filter((pl.col("day") == day0) & (pl.col("bar_idx") <= cut)).sort(["sid", "bar_idx"])
    kept_after = after.filter((pl.col("day") == day0) & (pl.col("bar_idx") <= cut)).sort(["sid", "bar_idx"])
    np.testing.assert_allclose(
        kept_base[name].to_numpy(), kept_after[name].to_numpy(), rtol=1e-12,
        err_msg=f"{op} 在 bar<={cut} 的取值受到了未来 bar 的影响 -> 存在未来函数",
    )


def test_cs_ops_cross_section_only_same_bar():
    """cs_rank 只在同时点 (day, bar_idx) 跨标的排名, 与当日其它 bar 无关。"""
    df = _panel()
    out, name = _run(df, {"op": "cs_rank", "args": [{"col": "x"}]})

    bar = 17
    moment = df.filter(pl.col("bar_idx") == bar, pl.col("day") == DAYS[1]).sort("sid")
    expect_rank = moment["x"].rank(method="average") / moment.height
    got = out.filter(pl.col("bar_idx") == bar, pl.col("day") == DAYS[1]).sort("sid")
    np.testing.assert_allclose(got[name].to_numpy(), expect_rank.to_numpy(), rtol=1e-12)


# ============================================================================
# 2. 时序算子: 标的隔离 (concat 帧无跨标的污染)
# ============================================================================

def test_ts_ops_sid_isolation_sorted():
    df = _panel()
    out, name = _run(df, {"op": "ref", "args": [{"col": "x"}, 1]})

    # 每只票的首行必须是 null, 而不是上一只票的最后一个值
    first_rows = out.filter(
        (pl.col("day") == DAYS[0]) & (pl.col("bar_idx") == 0)
    ).sort("sid")
    assert first_rows[name].is_null().all(), "各标的起始 bar 不应引用其它标的的数据"

    # ref 正确引用同标的上一根 bar
    a0 = out.filter(pl.col("sid") == "60000", pl.col("day") == DAYS[0]).sort("bar_idx")
    raw_a0 = df.filter(pl.col("sid") == "60000", pl.col("day") == DAYS[0]).sort("bar_idx")
    np.testing.assert_allclose(a0[name][1:].to_numpy(), raw_a0["x"][:-1].to_numpy())


def test_cum_sum_restarts_per_sid():
    df = _panel()
    out, name = _run(df, {"op": "cum_sum", "args": [{"col": "x"}]})

    per_sid = df.group_by("sid").agg(pl.col("x").sum().alias("sid_total")).sort("sid")
    last_rows = out.group_by("sid").agg(pl.col(name).last().alias("cum_last")).sort("sid")
    joined = per_sid.join(last_rows, on="sid")
    # 各标的累计终值 = 自身求和 (原先不分区时为全场累计, 后面的票会累加前面所有票)
    np.testing.assert_allclose(
        joined["cum_last"].to_numpy(), joined["sid_total"].to_numpy(), rtol=1e-9,
    )


def test_ts_ops_sid_isolation_shuffled():
    """行序乱序 (非按时间排序) 时, shift.over('sid') 也绝不引用其它标的的行:
    每行取的是"同标的在帧序中的上一行", 跨标的相邻行不串数。"""
    df = _panel()
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
    out, name = _run(shuffled, {"op": "ref", "args": [{"col": "x"}, 1]})

    # 手工期望: 按帧序扫描, 每行取同标的上一次出现时的 x, 首次出现为 null
    sids = shuffled["sid"].to_list()
    xs = shuffled["x"].to_list()
    expect: list[float | None] = []
    last_x_by_sid: dict[str, float] = {}
    for i in range(shuffled.height):
        expect.append(last_x_by_sid.get(sids[i]))
        last_x_by_sid[sids[i]] = xs[i]

    got = out[name].to_list()
    for i, (g, e) in enumerate(zip(got, expect)):
        assert (g is None) == (e is None), f"row {i}: null 位形不一致"
        if e is not None:
            assert g == pytest.approx(e), f"row {i}: 引用了其它标的的数据"


# ============================================================================
# 3. talib 桥: 按标的分组
# ============================================================================

def test_talib_bridge_sid_isolated():
    df = _panel().with_columns(pl.col("x").alias("close"))
    ast = {"op": "rsi", "args": [{"col": "close"}, 5]}
    out, name = _run(df, ast)

    for si in range(N_SIDS):
        sid = f"60{si:04d}"
        arr = df.filter(pl.col("sid") == sid)["close"].to_numpy().astype(np.float64)
        expect = talib.RSI(arr, timeperiod=5)
        got = out.filter(pl.col("sid") == sid).sort(["day", "bar_idx"])[name].to_numpy()
        np.testing.assert_allclose(got[5:], expect[5:], rtol=1e-10,
                                   err_msg=f"{sid} 的 talib 结果混入了其它标的的数据")


# ============================================================================
# 4. 跨族窗口嵌套: 编译期拒绝; 同族/并列/recipe 跨步: 正常
# ============================================================================

def test_cross_family_nesting_rejected():
    # cs 套 ts
    with pytest.raises(ASTCompilationError, match="window-family"):
        compile_ast({"op": "cs_zscore", "args": [{"op": "mean", "args": [{"col": "x"}, 5]}]})
    # ts 套 cs
    with pytest.raises(ASTCompilationError, match="window-family"):
        compile_ast({"op": "mean", "args": [{"op": "cs_rank", "args": [{"col": "x"}]}, 5]})
    # 深层嵌套同样拦截: add(mean(cs_rank(x),5), 1)
    with pytest.raises(ASTCompilationError, match="window-family"):
        compile_ast({"op": "add", "args": [
            {"op": "mean", "args": [{"op": "cs_rank", "args": [{"col": "x"}]}, 5]}, 1.0]})


def test_same_key_nesting_and_sibling_combine():
    df = _panel()

    # 同族嵌套: ref(mean(x,3),2) 与两步物化结果一致
    expr, name = compile_ast({"op": "ref", "args": [{"op": "mean", "args": [{"col": "x"}, 3]}, 2]})
    nested = df.with_columns(expr.alias(name))
    two_step = (
        df.with_columns(
            pl.col("x").rolling_mean(3, min_samples=1).over("sid").alias("_m")
        ).with_columns(pl.col("_m").shift(2).over("sid").alias("_r"))
    )
    j = nested.with_columns(two_step["_r"])
    same_null = (j[name].is_null() == j["_r"].is_null()).all()
    assert same_null
    np.testing.assert_allclose(j[name].drop_nulls().to_numpy(), j["_r"].drop_nulls().to_numpy(), rtol=1e-10)

    # 并列组合: add(cs_rank(x), ref(x,1)) 可编译且非全 null
    sib_expr, sib_name = compile_ast({"op": "add", "args": [
        {"op": "cs_rank", "args": [{"col": "x"}]},
        {"op": "ref", "args": [{"col": "x"}, 1]},
    ]})
    sib = df.with_columns(sib_expr.alias(sib_name))
    assert sib[sib_name].null_count() < sib.height


def test_recipe_cross_step_family_mix():
    """recipe 分步是跨族组合的正规通道: 先时序物化, 再做截面。"""
    df = _panel()
    recipe = [
        {"name": "ts_step", "ast": {"op": "mean", "args": [{"col": "x"}, 3]}},
        {"name": "cs_step", "ast": {"op": "cs_zscore", "args": [{"col": "ts_step"}]}},
    ]
    exprs = compile_recipe(recipe, check_causal=True)
    assert len(exprs) == 2

    lf = df.lazy()
    for e in exprs:
        lf = lf.with_columns(e)
    out = lf.collect()

    # 非 null 为主 (仅各标的头部 warmup 为 null), 且确实做了截面标准化
    assert out["cs_step"].null_count() < out.height * 0.1
    per_moment = out.group_by(["day", "bar_idx"]).agg(pl.col("cs_step").median().abs().alias("med"))
    assert per_moment["med"].max() < 1e-6, "cs_zscore 后每个 (day,bar_idx) 横截面中位数应为 0"
