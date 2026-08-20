#! /usr/bin/env python3
from __future__ import annotations

import json
from typing import Any, Dict, List

from .safeops import SAFE_OPS
from .plugins import talib_ops
from bt_studio.constant import RL_TOPK


_KNOWN_ROLLING_OPS = {
    "ref", "delta", "mean", "std", "var", "max", "min",
    "sum", "ema", "rank", "ts_rank", "ts_mean", "ts_std",
    "ts_max", "ts_min", "decay_linear", "wma", "ts_argmax", "ts_argmin"
}

_TALIB_BRIDGE_OPS = {"talib", "rsi"}  # 桥接/外挂算子排除项


def _render_layer_one() -> str:
    """从 SAFE_OPS 动态渲染第一层基础算子帮助信息"""
    unary: List[str] = []
    rolling: List[str] = []
    pairwise: List[str] = []
    ternary: List[str] = []
    variadic: List[str] = []

    for op in sorted(SAFE_OPS.keys()):
        if op in _TALIB_BRIDGE_OPS:
            continue

        spec = SAFE_OPS[op]
        lo, hi = spec["arity"]

        if (lo, hi) == (1, 1):
            unary.append(op)
        elif (lo, hi) == (2, 2):
            if op.startswith("ts_") or op in _KNOWN_ROLLING_OPS:
                rolling.append(op)
            else:
                pairwise.append(op)

        elif (lo, hi) == (3, 3):
            ternary.append(op)
        else:
            variadic.append(op)

    lines = [
        "[第一层: 基础算子 (Primitives)]",
        f"- 单目/截面算子 (1个参数: [输入]): {json.dumps(unary, ensure_ascii=False)}",
        "  *截面算子 (cs_*) 对全市场截面进行去极值、标准化或中性化*",
        f"- 时序滚动算子 (2个参数: [表达式, 窗口N]): {json.dumps(rolling, ensure_ascii=False)}",
        "  *特别提醒: `delta(expr, N)` 代表时序变动量; `rank(expr, N)` / `ts_rank` 代表滚动分位数排名*",
        f"- 二元代数算子 (2个参数: [左表达式, 右表达式]): {json.dumps(pairwise, ensure_ascii=False)}",
    ]
    if ternary:
        lines.append(f"- 三元算子 (3个参数): {json.dumps(ternary, ensure_ascii=False)}")
    if variadic:
        lines.append(f"- 变参/多参数算子: {json.dumps(variadic, ensure_ascii=False)}")
    return "\n".join(lines)


def _format_ops_help() -> str:
    layer_one = _render_layer_one()
    talib_content = ""
    if talib_ops.talib is None:
        talib_content = "\n (# TA-Lib C底层库未启用: 仅支持第一层基础算子，禁止调用 TA-Lib 算子)"
    else:
        talib_content = f"\n{talib_ops.format_ops_help()}"

    return f"""
        {layer_one}

        【第二层: TA-Lib 白名单精选算子 (Layer 2)】
        系统内置了极速的 C 底层实现。**为了防止参数过拟合，我们极其克制地只开放了以下白名单**.
        语法一律采用位置参数传递: `{{"op": "算子名", "args": [输入列, 周期N]}}`，绝对禁止使用 kwargs!
        冻结参数(如 MACD 的 12/26/9)必须显式按顺序写出，否则编译期直接拒绝。{talib_content}
    """


def build_system_prompt(max_depth: int = 4) -> str:
    return f"""你是一名顶尖的量化对冲基金资深 Alpha 研究员（如 WorldQuant / Two Sigma 风格）。
你的任务是根据金融微观结构逻辑与量价行为金融学，构建极具预测能力的高频/日频因子表达式，并输出为严格规范的 JSON AST(抽象语法树)。

【WorldQuant 101 Alphas 级别的动能与微观结构指导原则】:
1. **动量加速度 (Acceleration)**: 动能不仅看一阶方向，更要看其二阶导数变化率。如 `delta(delta(close, 1), 1)` 捕捉了价格的加速度；加速度过度透支通常预示短期均值回归。
2. **趋势一致性 (Consistency)**: 利用多周期方向符号累加衡量动能连贯性。如 `sign(delta(close, 1)) + sign(delta(close, 2)) + sign(delta(close, 3))`，一致性极高时易出现反转或突破。
3. **量价确认 (Volume Confirmation)**: 纯价格突破极易形成假信号。务必结合 `amount` 或成交量异常度（如当前成交额除以均值）进行确认，确保信号拥有流动性支撑。
4. **横截面相对强弱 (Cross-Sectional Neutralization)**: 单股时间序列信号易受大盘 Beta 污染，务必在最外层合理运用 `cs_zscore` / `cs_demean` / `cs_rank` 提取纯粹截面 Alpha。
5. **拒绝参数过拟合**: 逻辑必须具备清晰的经济学直觉，绝对禁止使用无实际意义的浮点“魔数”常数（如 0.7321)。

【AST 结构语法规则】:
1. 顶层必须为合法 JSON, 包含 `"hypothesis_id"`, `"economic_reasoning"` 和 `"sub_features"`。
2. 基础行情列: `{{"col": "字段名"}}`，可用字段仅限: `["open", "high", "low", "close", "volume", "amount"]`。
3. 算子节点: `{{"op": "算子名", "args": [...]}}`。全部参数必须放入 `args` 列表中，**绝对禁止使用 `kwargs` 或 `params`**!
4. **最大树深硬限制在 {max_depth} 层以内！** 超过 {max_depth} 层的过度嵌套树会被编译器直接拒绝。
5. 严禁未来数据操作（如负向位移 `shift(-1)` 或 `backward_fill`）。
6. 每一轮迭代可在 `sub_features` 中提出 1~3 个候选特征（支持单行 AST 或 多步 Recipe)。

{_format_ops_help()}

[两种特征构建模式的范例]:

**模式 A: 单行复合 AST (适合紧凑型公式)**
```json
{{
  "hypothesis_id": "hyp_accel_reversion",
  "economic_reasoning": "利用 delta 提取 3 日动量加速度。若资产加速上涨，短期面临超买回调风险；通过截面标准化提取相对强弱。",
  "sub_features": [
    {{
      "op": "cs_zscore",
      "args": [
        {{
          "op": "delta",
          "args": [
            {{"op": "delta", "args": [{{"col": "close"}}, 3]}},
            1
          ]
        }}
      ]
    }}
  ]
}}

**模式 B: 多步 Recipe / Let-binding (适合需要中间变量的复杂特征)**
{{
  "hypothesis_id": "hyp_trend_consistency_volume",
  "economic_reasoning": "计算连续 3 天的涨跌符号和衡量趋势一致性。利用当前成交额与 20 日均值的比值确认量能，二者结合并做截面去均值。",
  "sub_features": [
    [
      {{"name": "dir1", "ast": {{"op": "sign", "args": [{{"op": "delta", "args": [{{"col": "close"}}, 1]}}]}}}},
      {{"name": "dir2", "ast": {{"op": "sign", "args": [{{"op": "delta", "args": [{{"col": "close"}}, 2]}}]}}}},
      {{"name": "dir3", "ast": {{"op": "sign", "args": [{{"op": "delta", "args": [{{"col": "close"}}, 3]}}]}}}},
      {{"name": "trend_score", "ast": {{"op": "add", "args": [{{"op": "add", "args": [{{"col": "dir1"}}, {{"col": "dir2"}}]}}, {{"col": "dir3"}}]}}}},
      {{"name": "vol_ratio", "ast": {{"op": "div", "args": [{{"col": "amount"}}, {{"op": "mean", "args": [{{"col": "amount"}}, 20]}}]}}}},
      {{"name": "final_alpha", "ast": {{"op": "cs_demean", "args": [{{"op": "mul", "args": [{{"col": "trend_score"}}, {{"col": "vol_ratio"}}]}}]}}}}
    ]
  ]
}}

输出格式强制要求:
请直接输出标准 JSON 文本（允许使用 json  代码块包裹）,不要包含与 JSON 内容无关的闲聊或前后缀解释。
"""


def build_rl_context_prompt(
    target_description: str,
    successful_history: List[Dict[str, Any]],
    failed_history: List[Dict[str, Any]],
) -> str:
    """构建注入强化学习经验回放(Replay Buffer)的 User 提示词"""
    sections: List[str] = [
    f"【当前挖掘目标与市场场景】:\n{target_description}\n"
    ]

    # 1. Reward > 0
    if successful_history:
        success_lines = [
            "[经验池 - 成功案例 (Top-Reward 因子参考)]:",
            "以下是你之前探索出并在 Walk-Forward 检验中获得高分的特征,请提炼其经济学精髓并在新假设中进行变异与升华:"
        ]
        for i, item in enumerate(successful_history[:RL_TOPK]):
            score_val = item.get("score")
            score_str = f"{float(score_val):.2f}" if isinstance(score_val, (int, float)) else "N/A"
            reasoning = item.get("reasoning", "无记录")
            ast_json = json.dumps(item.get("ast", {}), ensure_ascii=False)
            success_lines.append(
                f"- 成功案例 {i + 1} (Score: {score_str}):\n"
                f"  逻辑: {reasoning}\n"
                f"  AST: {ast_json}"
            )
        sections.append("\n".join(success_lines) + "\n")

    # 2. Reward <= 0 
    if failed_history:
        failed_lines = [
            "[经验池 - 失败案例 (避坑与约束提示)]:",
            "以下是你近期提出的被系统拒绝或评测不合格的因子,**绝对不要**再次生成类似结构或犯相同错误:"
        ]
        for i, item in enumerate(failed_history[:RL_TOPK]):
            reason = item.get("reason", "未知原因")
            ast_json = json.dumps(item.get("ast", {}), ensure_ascii=False)
            failed_lines.append(
                f"- 失败案例 {i + 1} (失败诊断: {reason}):\n"
                f"  AST: {ast_json}"
            )
        sections.append("\n".join(failed_lines) + "\n")

    sections.append(
        "请基于上述目标与经验反馈，结合微观结构与量价行为提出一个全新且具备更强预测力的特征假设, 并输出严格规范的 JSON。"
    )

    return "\n".join(sections)
