#!/usr/bin/env python3
"""prompt.py 与编译器的一致性回归测试。

prompt 是直接喂给 LLM 的系统提示词: 范例教出的每个 AST 都必须能被当前
编译器接受, 否则 agent 产出的候选会成批被 compile guard 拒绝
(曾发生: 模式 A 旗舰范例 cs_zscore(delta(delta(close,3),1)) 在加入
跨族嵌套守卫后变成非法 AST, prompt 却仍在教)。

本测试直接引用 prompt 模块里的范例常量与渲染函数 —— prompt 改了范例,
这里自动跟随; 编译器改了约束, 这里第一时间红。
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from bt_studio.agent.prompt import (
    EXAMPLE_RECIPE,
    EXAMPLE_SINGLE_AST,
    _render_layer_one,
    build_rl_context_prompt,
    build_system_prompt,
)
from bt_studio.compiler import SAFE_OPS, ASTCompilationError, compile_ast, compile_recipe
from bt_studio.compiler.plugins import talib_ops


# ---------------------------------------------------------------------------
# 1. few-shot 范例必须可编译
# ---------------------------------------------------------------------------

def test_single_ast_example_compiles():
    subs = EXAMPLE_SINGLE_AST["sub_features"]
    assert 1 <= len(subs) <= 3, "范例应演示 1~3 个候选"
    for ast in subs:
        expr, name = compile_ast(ast, check_causal=True)  # 不抛异常即通过
        assert expr is not None and name


def test_recipe_example_compiles():
    for recipe in EXAMPLE_RECIPE["sub_features"]:
        assert isinstance(recipe, list) and recipe, "recipe 范例必须是步骤列表"
        exprs = compile_recipe(recipe, check_causal=True)
        assert len(exprs) == len(recipe)
        # 每一步都必须有 name/ast (LLM 依葫芦画瓢)
        for step in recipe:
            assert "name" in step and "ast" in step


def test_examples_only_use_registered_ops():
    def walk(node, acc):
        if isinstance(node, dict):
            if "op" in node:
                acc.add(node["op"])
            for v in node.get("args", []) or []:
                walk(v, acc)
        elif isinstance(node, list):
            for v in node:
                walk(v, acc)

    for payload in (EXAMPLE_SINGLE_AST, EXAMPLE_RECIPE):
        ops: set = set()
        walk(payload["sub_features"], ops)
        unknown = ops - set(SAFE_OPS)
        assert not unknown, f"范例使用了未注册算子: {unknown}"


# ---------------------------------------------------------------------------
# 2. 漂移金丝雀: 跨族嵌套必须被双端拒绝
# ---------------------------------------------------------------------------

def test_canary_cross_family_nesting_rejected():
    """旧 prompt 的旗舰范例 cs_zscore(delta(delta(close,3),1)) 现为非法:
    prompt 不得再教, 编译器必须继续拒绝。"""
    illegal = {"op": "cs_zscore",
               "args": [{"op": "delta",
                         "args": [{"op": "delta", "args": [{"col": "close"}, 3]}, 1]}]}
    with pytest.raises(ASTCompilationError):
        compile_ast(illegal, check_causal=True)

    # 反向也非法 (时序内嵌截面)
    with pytest.raises(ASTCompilationError):
        compile_ast({"op": "mean",
                     "args": [{"op": "cs_rank", "args": [{"col": "close"}]}, 5]},
                    check_causal=True)

    # 同族嵌套与并列组合必须仍然合法 (prompt 明确宣称二者合法)
    compile_ast({"op": "delta",
                 "args": [{"op": "delta", "args": [{"col": "close"}, 1]}, 1]},
                check_causal=True)
    compile_ast({"op": "add",
                 "args": [{"op": "cs_rank", "args": [{"col": "close"}]},
                          {"op": "cs_rank", "args": [{"col": "volume"}]}]},
                check_causal=True)


def test_system_prompt_has_no_illegal_example():
    prompt = build_system_prompt()
    # 两个范例完整嵌入 (以 hypothesis_id 与标志性步骤名为证)
    assert EXAMPLE_SINGLE_AST["hypothesis_id"] in prompt
    assert EXAMPLE_RECIPE["hypothesis_id"] in prompt
    assert "final_alpha" in prompt
    # 跨族禁令作为硬性规则出现
    assert "跨族嵌套禁令" in prompt
    assert "cs_zscore(mean(close, 5))" in prompt
    # 深度限制动态渲染
    assert "4 层" in prompt
    # 范例不得包含非法的 cs 包裹时序结构
    assert "cs_zscore(delta" not in prompt.replace(" ", "")


# ---------------------------------------------------------------------------
# 3. 算子清单渲染与注册表一致
# ---------------------------------------------------------------------------

def test_layer_one_matches_registry():
    text = _render_layer_one()
    bridge = set(talib_ops.REGISTRY) | {"talib"}

    # 算子分桶行: 每行 JSON 数组里的算子都必须真实存在于 SAFE_OPS
    bucket_lines = [ln.strip() for ln in text.splitlines()
                    if ln.strip().startswith("- ") and "族算子全集" not in ln]
    assert bucket_lines
    bucket_ops = set()
    for line in bucket_lines:
        for arr in _json_arrays(line):
            for op in arr:
                # 多输入行带 arity 后缀, 如 "atr(3-4)"
                bare = op.split("(")[0]
                assert bare in SAFE_OPS, f"prompt 渲染了不存在的算子: {op}"
                bucket_ops.add(bare)

    # TA-Lib 桥接算子不得泄漏进第一层分桶 (其签名/冻结参数由第二层说明);
    # 族全集行除外 —— talib 本身就是时序族窗口算子, 跨族禁令对它同样生效
    leaked = bridge & bucket_ops
    assert not leaked, f"桥接算子泄漏进第一层分桶: {leaked}"

    # 时序滚动桶非空且带缺省窗口说明 (此前因 arity 误判渲染为空列表)
    assert '"mean"' in text and '"delta"' in text
    assert "缺省值" in text


def _json_arrays(line: str):
    out = []
    i = 0
    while i < len(line):
        if line[i] == "[":
            j = line.find("]", i)
            if j > i:
                try:
                    out.append(json.loads(line[i:j + 1]))
                except json.JSONDecodeError:
                    pass
                i = j + 1
            else:
                break
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------
# 4. RL 上下文 prompt 兼容性
# ---------------------------------------------------------------------------

def test_rl_context_prompt():
    p = build_rl_context_prompt(
        "测试目标",
        [{"score": 1.5, "reasoning": "r", "ast": {"op": "sign", "args": [{"col": "close"}]}}],
        [{"reason": "compile error", "ast": {"bad": 1}}],
    )
    assert "测试目标" in p and "成功案例 1" in p and "失败案例 1" in p

    # NaN 分数不得崩溃
    p2 = build_rl_context_prompt("t", [{"score": float("nan")}], [])
    assert "N/A" in p2

    # 空历史也能构建
    assert "目标" in build_rl_context_prompt("t", [], [])


if __name__ == "__main__":
    raise SystemExit("run via pytest")
