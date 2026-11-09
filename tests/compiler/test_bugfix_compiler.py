#!/usr/bin/env python3
"""compiler 子系统 bugfix 回归测试 (TDD red→green)。

覆盖 2026-09 代码审计定位的逻辑 bug:
    H3  TA-Lib 白名单桥接算子缺 .over("sid") 分区 → 跨标的污染
    H4  白名单算子不在时序族集合 → 跨族嵌套禁令整族失明
    H5  roc 负周期逃过因果守卫 → 未来函数漏网
    H6  canonicalize_ast 默认参数回填错位 → 少参数静默算错 (atr(high,low))
    M14 compile_recipe 的 max_depth=None 意图是"无限制", 实际回落常量 4
    L   非法窗口参数 (mean(x,0) / rank(x,1) / 非整数窗口) 编译期拒绝
"""

import datetime
import os
import sys

import numpy as np
import polars as pl
import pytest
import talib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bt_studio.compiler import SAFE_OPS
from bt_studio.compiler.ast import (
    ASTCompilationError,
    TS_WINDOW_OPS,
    compile_ast,
    compile_recipe,
)


N_SIDS = 3
N_BARS = 80
DAYS = [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]


def _panel(seed: int = 5) -> pl.DataFrame:
    """各标的价值量级分离的分钟面板, 便于检测跨标的污染。"""
    rng = np.random.RandomState(seed)
    rows = []
    for si in range(N_SIDS):
        for d in DAYS:
            for b in range(N_BARS):
                rows.append({
                    "sid": f"60{si:04d}",
                    "day": d,
                    "bar_idx": b,
                    "x": float(si * 500 + b + 1 + rng.normal(0, 0.1)),
                })
    return pl.DataFrame(rows).sort(["sid", "day", "bar_idx"])


# ---------------------------------------------------------------------------
# H3: 白名单 talib 算子必须按 sid 分区
# ---------------------------------------------------------------------------

def test_whitelist_talib_op_sid_isolated():
    assert "rsi" in SAFE_OPS, "TA-Lib 白名单算子应已注册进 SAFE_OPS"
    df = _panel().rename({"x": "close"})
    expr, name = compile_ast({"op": "rsi", "args": [{"col": "close"}, 14]})
    out = df.with_columns(expr.alias(name))

    for si in range(N_SIDS):
        sid = f"60{si:04d}"
        arr = (df.filter(pl.col("sid") == sid).sort(["day", "bar_idx"])
               ["close"].to_numpy().astype(np.float64))
        expect = talib.RSI(arr, timeperiod=14)
        got = (out.filter(pl.col("sid") == sid).sort(["day", "bar_idx"])
               [name].to_numpy())
        np.testing.assert_allclose(
            got[14:], expect[14:], rtol=1e-10,
            err_msg=f"{sid} 的白名单 talib 结果混入了其它标的的数据")


# ---------------------------------------------------------------------------
# H4: 白名单算子必须进入时序族 → 跨族嵌套被拒绝
# ---------------------------------------------------------------------------

def test_whitelist_ops_belong_to_ts_family():
    from bt_studio.compiler.ast import _ts_window_ops
    ts_ops = _ts_window_ops()
    bridge = [op for op, spec in SAFE_OPS.items() if spec.get("family") == "ts"]
    assert set(bridge) <= ts_ops
    assert {"rsi", "adx", "macd"} & ts_ops, "白名单 talib 算子必须是时序族"


def test_cross_family_nesting_with_bridge_op_rejected():
    # 截面包裹白名单时序算子 → 拒绝
    with pytest.raises(ASTCompilationError, match="window-family"):
        compile_ast({"op": "cs_zscore",
                     "args": [{"op": "rsi", "args": [{"col": "close"}, 14]}]})
    # 白名单时序算子包裹截面 → 拒绝
    with pytest.raises(ASTCompilationError, match="window-family"):
        compile_ast({"op": "adx",
                     "args": [{"op": "cs_rank", "args": [{"col": "x"}]},
                              {"col": "low"}, {"col": "close"}, 14]})
    # 同族嵌套 (时序套时序) 仍然合法
    compile_ast({"op": "mean",
                 "args": [{"op": "rsi", "args": [{"col": "close"}, 14]}, 5]})


def test_prompt_ts_family_line_includes_bridge_ops():
    from bt_studio.agent.prompt import _render_layer_one
    from bt_studio.compiler.ast import _ts_window_ops
    text = _render_layer_one()
    ts_ops = _ts_window_ops() & set(SAFE_OPS)
    for op in ("rsi", "adx", "macd"):
        if op in SAFE_OPS:
            assert f'"{op}"' in text, f"prompt 时序族全集缺少白名单算子 {op}"
    assert ts_ops


# ---------------------------------------------------------------------------
# H5: roc 负周期 = 未来函数, 必须被因果守卫拒绝
# ---------------------------------------------------------------------------

def test_roc_negative_period_rejected():
    from bt_studio.compiler.ops import compile_expr
    with pytest.raises(ASTCompilationError):
        compile_ast({"op": "roc", "args": [{"col": "close"}, -3]},
                    check_causal=True)
    # compile_expr 默认同样启用因果守卫 (阻断前视旁路)
    with pytest.raises(ASTCompilationError):
        compile_expr({"op": "ref", "args": [{"col": "close"}, -1]})


# ---------------------------------------------------------------------------
# H6: canonicalize 默认参数回填不得越过 arity 下限
# ---------------------------------------------------------------------------

def test_atr_missing_required_arg_rejected():
    # atr(high, low) 少 close —— 必须被拒绝, 而不是把默认值 14 填进 close 位
    with pytest.raises(ASTCompilationError, match="3 to 4"):
        compile_ast({"op": "atr", "args": [{"col": "high"}, {"col": "low"}]})


def test_atr_three_args_gets_default_window():
    expr, name = compile_ast(
        {"op": "atr", "args": [{"col": "high"}, {"col": "low"}, {"col": "close"}]})
    assert expr is not None and "14" in name


def test_stoch_k_missing_required_arg_rejected():
    with pytest.raises(ASTCompilationError):
        compile_ast({"op": "stoch_k",
                     "args": [{"col": "high"}, {"col": "low"}]})


def test_canonicalize_rejects_invalid_arity_at_boundary():
    """canonicalize 持有 spec, arity 越界当场拒绝 (不再推迟到 _compile_node)。"""
    from bt_studio.compiler.ast import ast_fingerprint, canonicalize_ast

    with pytest.raises(ASTCompilationError, match="3 to 4"):
        canonicalize_ast({"op": "atr", "args": [{"col": "high"}, {"col": "low"}]})
    with pytest.raises(ASTCompilationError, match="exactly 1"):
        canonicalize_ast({"op": "abs", "args": []})

    # 合法回填不受影响: delta(x) -> delta(x, 1)
    node = canonicalize_ast({"op": "delta", "args": [{"col": "close"}]})
    assert node["args"] == [{"col": "close"}, 1]

    # 非法候选的指纹必须恒可得 (失败池去重依赖), 不得抛错
    assert ast_fingerprint({"op": "atr", "args": [{"col": "high"}, {"col": "low"}]})


# ---------------------------------------------------------------------------
# M14: compile_recipe 单步不受 MAX_AST_DEPTH=4 误伤
# ---------------------------------------------------------------------------

def test_recipe_deep_single_step_compiles():
    """单步固有深度 5 (≤ MAX_RECIPE_DEPTH=6) 的步骤必须可编译;
    旧实现内层 compile_ast(None) 回落常量 4, 与累计深度守卫自相矛盾。"""
    deep = {"op": "mean", "args": [
        {"op": "mean", "args": [
            {"op": "mean", "args": [
                {"op": "mean", "args": [
                    {"op": "mean", "args": [{"col": "close"}, 3]}, 3]}, 3]}, 3]}, 3]}
    # 深度 = 5 层嵌套 (单棵 AST 默认上限 4, recipe 累计上限 6)
    recipe = [{"name": "deep_feat", "ast": deep}]
    exprs = compile_recipe(recipe, check_causal=True)
    assert len(exprs) == 1


# ---------------------------------------------------------------------------
# L: 窗口参数编译期校验
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ast", [
    {"op": "mean", "args": [{"col": "x"}, 0]},
    {"op": "mean", "args": [{"col": "x"}, -5]},
    {"op": "rank", "args": [{"col": "x"}, 1]},
    {"op": "ref", "args": [{"col": "x"}, 0]},
    {"op": "mean", "args": [{"col": "x"}, 2.5]},
])
def test_invalid_window_args_rejected(ast):
    with pytest.raises(ASTCompilationError):
        compile_ast(ast)


@pytest.mark.parametrize("ast", [
    {"op": "mean", "args": [{"col": "x"}, 2]},
    {"op": "mean", "args": [{"col": "x"}, 5.0]},   # 整数值浮点可接受
    {"op": "ema", "args": [{"col": "x"}, 1]},
    {"op": "ref", "args": [{"col": "x"}, 1]},
])
def test_valid_window_args_accepted(ast):
    expr, name = compile_ast(ast)
    assert expr is not None


def test_rank_rolling_map_polars14_compat():
    """Polars 1.44 移除 rolling_map(return_dtype=...) 后 rank 仍可编译求值;
    且当前值为 NaN/Null 时必须返回 Null (L1 修复: 绝不借用上一有效值排名)。"""
    expr, name = compile_ast({"op": "rank", "args": [{"col": "x"}, 3]})
    df = pl.DataFrame({"sid": ["a"] * 6, "x": [1.0, 3.0, 2.0, 4.0, 5.0, 6.0]})
    out = df.lazy().with_columns(expr.alias(name)).collect()
    vals = out[name].to_list()
    # mid-rank 口径: 3 窗内唯一最大值 = (2 小于 + 0.5 等于) / 3 = 5/6
    assert vals[-1] == 5.0 / 6.0, vals
    assert all(v is None or 0.0 <= v <= 1.0 for v in vals), vals

    # L1 回归: 输入 [1.0, 2.0, None, 4.0], window=3 -> 第 3 位 (index 2, 值为 None) 必须为 None
    df_nan = pl.DataFrame({"sid": ["a"] * 4, "x": [1.0, 2.0, None, 4.0]})
    out_nan = df_nan.lazy().with_columns(expr.alias(name)).collect()
    nan_vals = out_nan.get_column(name).to_list()
    assert nan_vals[2] is None, f"当前值为 None 时 rank 必须返回 None, 实际: {nan_vals}"
    assert nan_vals[1] == 0.75 and nan_vals[3] == 0.75

    # L7 回归: roc 在 warmup 期 (prev 为 Null) 必须返回 None 而非假 0.0
    roc_expr, roc_name = compile_ast({"op": "roc", "args": [{"col": "x"}, 2]})
    out_roc = df.lazy().with_columns(roc_expr.alias(roc_name)).collect()
    roc_vals = out_roc[roc_name].to_list()
    assert roc_vals[0] is None and roc_vals[1] is None
    assert roc_vals[2] is not None


def test_talib_generic_op_enforces_whitelist():
    """通用 talib 算子已从 SAFE_OPS 移除并由 talib_validate_hook 拦截, 强制走精选白名单算子。"""
    with pytest.raises(ASTCompilationError):
        compile_ast({"op": "talib", "args": ["COSH", {"col": "close"}]})
    with pytest.raises(ASTCompilationError):
        compile_ast({"op": "talib", "args": ["RSI", {"col": "close"}], "kwargs": {"timeperiod": 14}})


def test_new_ts_ops_corr_cov_zscore():
    """新增的 corr / cov / zscore 算子按 sid 分区计算且支持默认窗口回填。"""
    df = _panel()
    df = df.with_columns((pl.col("x") * 2.0 + 1.0).alias("y"))

    for ast in (
        {"op": "zscore", "args": [{"col": "x"}, 10]},
        {"op": "zscore", "args": [{"col": "x"}]},
        {"op": "corr", "args": [{"col": "x"}, {"col": "y"}, 10]},
        {"op": "corr", "args": [{"col": "x"}, {"col": "y"}]},
        {"op": "cov", "args": [{"col": "x"}, {"col": "y"}, 10]},
    ):
        expr, name = compile_ast(ast, check_causal=True)
        out = df.lazy().with_columns(expr.alias(name)).collect()
        assert out[name].null_count() < out.height

    # corr(x, 2x+1, 10) 在稳定期应接近 1.0
    expr_c, name_c = compile_ast({"op": "corr", "args": [{"col": "x"}, {"col": "y"}, 10]})
    out_c = df.lazy().with_columns(expr_c.alias(name_c)).collect()
    valid_c = out_c[name_c].drop_nulls().to_numpy()
    assert np.allclose(valid_c, 1.0, atol=1e-6)

    # 窗口过小 (<3) 应被拒绝
    with pytest.raises(ASTCompilationError):
        compile_ast({"op": "corr", "args": [{"col": "x"}, {"col": "y"}, 2]})


if __name__ == "__main__":
    raise SystemExit("run via pytest")
