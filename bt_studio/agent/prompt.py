#! /usr/bin/env python3
from __future__ import annotations

import json

import numpy as np

from typing import Any, Dict, List, Optional

from bt_studio.compiler import SAFE_OPS
from bt_studio.compiler.ast import CS_WINDOW_OPS, _ts_window_ops
from bt_studio.compiler.plugins import talib_ops
from bt_studio.constant import MAX_RECIPE_DEPTH, RL_TOPK


# ---------------------------------------------------------------------------
# Few-shot prompt
# ---------------------------------------------------------------------------

EXAMPLE_SINGLE_AST: Dict[str, Any] = {
    "hypothesis_id": "hyp_accel_reversion",
    "economic_reasoning": "利用二阶差分提取短期动量加速度. 加速拉升面临流动性透支与均值回归风险。注意: 单行 AST 禁止跨族嵌套截面算子，若需截面标准化必须使用模式 B Recipe.",
    "sub_features": [
        {"op": "delta", "args": [{"op": "delta", "args": [{"col": "close"}, 2]}, 1]}
    ],
}

EXAMPLE_RECIPE: Dict[str, Any] = {
    "hypothesis_id": "hyp_trend_volume_breakout",
    "economic_reasoning": "结合短期价格动能方向与成交量放大倍数.时序特征先行计算并物化, 最后一步使用截面去均值算子进行市场中性化.",
    "sub_features": [
        [
            {"name": "ret_dir", "ast": {"op": "sign", "args": [{"op": "delta", "args": [{"col": "close"}, 1]}]}},
            {"name": "vol_ratio", "ast": {"op": "div", "args": [{"col": "amount"}, {"op": "mean", "args": [{"col": "amount"}, 20]}]}},
            {"name": "trend_vol", "ast": {"op": "mul", "args": [{"col": "ret_dir"}, {"col": "vol_ratio"}]}},
            {"name": "final_alpha", "ast": {"op": "cs_demean", "args": [{"col": "trend_vol"}]}},
        ]
    ],
}


def _get_ts_window_ops_list() -> List[str]:
    """dynamic extract time-series window operator set (reusing compiler.ast._ts_window_ops single source of truth)."""
    ts_ops = set(_ts_window_ops())
    if hasattr(talib_ops, "REGISTRY"):
        ts_ops.update(talib_ops.REGISTRY.keys())
    return sorted(ts_ops)


def _render_layer_one() -> str:
    """SAFE_OPS"""
    bridge_ops = set(talib_ops.REGISTRY.keys())

    unary: List[str] = []
    rolling: List[str] = []
    pairwise: List[str] = []
    multi: List[str] = []
    rolling_defaults: List[str] = []

    for op in sorted(SAFE_OPS.keys()):
        if op in bridge_ops:
            continue

        spec = SAFE_OPS[op]
        lo, hi = spec["arity"]
        defaults = spec.get("defaults") or []

        if (lo, hi) == (1, 1):
            unary.append(op)
        elif (lo, hi) == (1, 2):
            rolling.append(op)
            if defaults:
                rolling_defaults.append(f"{op}={defaults[0]}")
        elif (lo, hi) == (2, 2):
            pairwise.append(op)
        else:
            multi.append(f"{op}({lo}-{hi if hi is not None else 'N'})")

    cs_ops = sorted(CS_WINDOW_OPS & set(SAFE_OPS))
    ts_ops = _get_ts_window_ops_list()

    lines = [
        "[第一层: 基础算子 (Primitives)]",
        f"- 单目/截面算子 (1个参数 [输入]): {json.dumps(unary, ensure_ascii=False)}",
        "  *cs_* 开头为横截面算子 (按 day×bar 全市场截面计算), 其余为逐元素单目变换*",
        f"- 时序滚动算子 (参数 [表达式, 窗口N], N 可省略): {json.dumps(rolling, ensure_ascii=False)}",
        f"  *窗口 N 必须为正整数 (回看历史); 缺省值: {', '.join(rolling_defaults)}*",
        "  *`delta(expr, N)` 为 N 期变动量; `rank(expr, N)` 为时序百分比分位数*",
        f"- 二元代数与逻辑比较 (2个参数 [左表达式, 右表达式]): {json.dumps(pairwise, ensure_ascii=False)}",
    ]
    if multi:
        lines.append(
            f"- 多输入算子 (位置参数): {json.dumps(multi, ensure_ascii=False)}"
        )
    lines += [
        f"- 时序族算子全集 (按 sid 标的分区执行): {json.dumps(ts_ops, ensure_ascii=False)}",
        f"- 截面族算子全集 (按 day×bar 分区执行): {json.dumps(cs_ops, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


def _format_ops_help() -> str:
    layer_one = _render_layer_one()
    if talib_ops.talib is None:
        talib_content = "\n(# TA-Lib 底层库未启用: 仅支持第一层基础算子)"
    else:
        talib_content = f"\n{talib_ops.format_ops_help()}"

    res = f"""{layer_one}

【第二层: TA-Lib 白名单精选算子 (Layer 2)】
系统内置了 C 底层高性能实现。为了防止严重过拟合，仅开放以下精选白名单算子。
语法一律采用位置参数传递: `{{"op": "算子名", "args": [输入列, 周期N]}}`，绝对禁止使用 kwargs!
冻结参数(如 MACD 的 12/26/9)必须按定义给出，严禁随意更动。{talib_content}"""
    return res


def build_system_prompt(max_depth: int = 4) -> str:
    """构建用于驱动 LLM 进行 Alpha 特征假设挖掘的 System Prompt。"""
    ops_help = _format_ops_help()
    example_a = json.dumps(EXAMPLE_SINGLE_AST, ensure_ascii=False, indent=2)
    example_b = json.dumps(EXAMPLE_RECIPE, ensure_ascii=False, indent=2)

    prompt = f"""你是一名顶尖量化对冲基金资深 Alpha 研究员(WorldQuant / Two Sigma 风格)。
        你的任务是基于金融微观结构理论、订单流失衡(OFI)与量价行为金融学，构建极具预测能力的高频/日频因子表达式，并输出为严谨合规的 JSON AST。

        【WorldQuant 101 Alphas 级别的动能与微观结构指导原则】:
        1. **动量加速度 (Acceleration)**: 动能不仅看一阶方向，更要看其二阶导数变化率。如 `delta(delta(close, 2), 1)` 捕捉了价格加速度;加速度过度透支通常预示短期超买回调。(同族嵌套合法)
        2. **趋势与量能确认 / 量价背离 (Volume Confirmation & Price-Volume Divergence)**: 纯价格形态容易形成假突破。善用 `corr(close, volume, N)`、`cov` 或成交额放大比率 (`div(amount, mean(amount, 20))`) 捕捉“价涨量增确认”或“价涨量缩背离”，确保信号拥有真实资金沉淀。
        3. **横截面相对强弱 (Cross-Sectional Neutralization)**: 单股时序特征易受宏观 Beta 干扰。**截面标准化算子 (如 cs_demean, cs_zscore, cs_rank) 必须且只能作为 Recipe 模式的最后一步物化**;单行 AST 内直接嵌套截面算子会被编译器严厉拒绝。
        4. **适配下游 Motif + Z-Normalized DTW 形态引擎 & 交易时点契约**:
           - 系统在 T 日尾盘入场截断点 (由 `exclude_bars` 配置控制) 截断日内特征曲线，通过 Matrix Profile 与 Z-Score 归一化 DTW 寻找高频重复波形，预测 **T 日尾盘入场 → T+1 早盘各目标窗口出场** 的超额收益。
           - **波形连续性硬约束**: 最终输出特征必须具有连续、平滑且有局部起伏的日内波形（善用 `ema`、`mean`、`zscore`、`rank`）；**禁止将 `gte`/`lte` 二值阶跃输出作为最终特征**（Z-Score 归一化后退化为死水或尖刺，导致 DTW 无法匹配触发），**禁止直接输出未归一化的纯单调累加列**（如裸 `cum_sum(amount)`）。
           - **`log` 与 `sqrt` 定义域守卫 (`x <= 0 -> Null`)**: `log(x)` 仅对严格正数 `x > 0` 有效（`x <= 0` 会自动置为 `Null`）；**严禁对可取零或负值的列（如 `delta`、`roc`、`cs_demean`、`cs_zscore`、`ofi`）直接套用 `log` 或 `sqrt`**，否则将产生大量 `Null` 被 Stage 1 直接拦截（仅对 `volume`、`amount`、价格比值或 `abs(x)` 正值序列使用）。
        5. **拒绝无意义过拟合**: 严禁拼接无经济学直觉的任意常数浮点“魔数”。

        【AST 结构语法规则 (硬性工业级约束, 违规当场拒绝)】:
        1. 顶层必须输出合法标准 JSON, 包含 `"hypothesis_id"`, `"economic_reasoning"` 和 `"sub_features"`。
        2. 基础行情列定义: `{{"col": "字段名"}}`，仅支持原始量价字段: `["open", "high", "low", "close", "volume", "amount"]`。
           **没有 vwap / ofi 等现成派生列** —— 一切派生量必须由你在表达式内显式定义:
           - `vwap` (bar 内成交均价): `{{"op": "div", "args": [{{"col": "amount"}}, {{"col": "volume"}}]}}`
           - `ofi` (tick-rule 订单流失衡代理): `{{"op": "mul", "args": [{{"op": "sign", "args": [{{"op": "delta", "args": [{{"col": "close"}}, 1]}}]}}, {{"col": "volume"}}]}}`
           引用未定义列会在编译/执行期被直接拒绝。
        3. 算子节点格式: `{{"op": "算子名", "args": [...]}}`。全部参数必须放入 `args` 列表中，**绝对禁止使用 `kwargs` 或 `params`**!
        4. 窗口参数 N 一律为**正整数**，严禁使用负数(负数视为未来函数侵入，直接触发安全防御拦截)。
        5. **跨族嵌套禁令**: 时序族窗口算子 (按个股 sid 分区) 与截面族算子 (按 day×bar 分区) 在单棵 AST 内**禁止纵向嵌套** (如 `cs_zscore(mean(close, 5))` 会导致静默全 Null 错误)。需要时序 + 截面组合时，**必须使用模式 B Recipe 分步**。
        6. **树深硬性约束**: 单行 AST 最大树深 ≤ {max_depth} 层;Recipe 模式每步展开后的累积等效深度 ≤ {MAX_RECIPE_DEPTH} 层。
        7. 严禁未来信息算子与前瞻特征名。
        8. 每一轮迭代可在 `sub_features` 中提出 1~3 个候选特征(单行 AST 或 多步 Recipe)。

        {ops_help}

        【两种特征构建模式的标准范例】:

        **模式 A: 单行复合 AST (适合紧凑型公式, 仅使用同一算子族)**
        ```json
        {example_a}
        ```

        **模式 B: 多步 Recipe / Let-binding (需要中间变量, 或需要时序+截面跨族组合时的唯一合法方式)**
        ```json
        {example_b}
        ```

        输出格式强制要求:
        请直接输出标准 JSON 文本(允许使用 ```json 代码块包裹)，不要包含与 JSON 内容无关的闲聊或前后缀解释。"""

    return prompt


def _family_signature(ast_obj: Any) -> str:
    """从 AST/Recipe 提取确定性"特征族"签名: 算子集 + 基础列集。

    用于同族饱和度提示 —— 族 = (ops, cols) 集合, 同族即同机制结构,
    与具体窗口/常数/嵌套顺序/Recipe 中间步骤变量名无关。
    """
    ops, cols = set(), set()
    step_names = {
        str(s["name"]).strip()
        for s in (ast_obj if isinstance(ast_obj, list) else [])
        if isinstance(s, dict) and isinstance(s.get("name"), str)
    }

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if "op" in node and isinstance(node["op"], str):
                ops.add(node["op"])
            if "col" in node and isinstance(node["col"], str):
                col_name = node["col"].strip()
                if col_name not in step_names:
                    cols.add(col_name)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(ast_obj)
    if not ops and not cols:
        return "(未知结构)"
    return f"ops[{'+'.join(sorted(ops))}] × cols[{'+'.join(sorted(cols))}]"


def _render_family_stats(
    successful_history: List[Dict[str, Any]],
    failed_history: List[Dict[str, Any]],
) -> Optional[str]:
    """同族探索饱和度: 已充分探索的族应换机制, 防止 LLM 在同结构上打转。"""
    fam: Dict[str, Dict[str, Any]] = {}

    def _collect(items: List[Dict[str, Any]], kind: str) -> None:
        for it in items:
            sig = _family_signature(it.get("ast", {}))
            rec = fam.setdefault(sig, {"succ": [], "fail": 0, "best": -float("inf")})
            if kind == "succ":
                rec["succ"].append(it.get("name", "?"))
                try:
                    rec["best"] = max(rec["best"], float(it.get("score", -500)))
                except (TypeError, ValueError):
                    pass
            else:
                rec["fail"] += 1

    _collect(successful_history, "succ")
    _collect(failed_history, "fail")
    if not fam:
        return None

    lines = ["[同族探索饱和度 (族 = 算子集 × 基础列集, 同族 ≈ 同机制结构)]:"]
    # 成功多的族在前; 无成功按失败数
    for sig, rec in sorted(fam.items(), key=lambda kv: (-len(kv[1]["succ"]), -kv[1]["fail"])):
        n_succ = len(rec["succ"])
        tag = "已饱和" if n_succ >= 2 else ("已有突破" if n_succ == 1 else "未突破")
        names = ", ".join(rec["succ"][:3])
        lines.append(
            f"- {sig}: 成功 {n_succ} / 失败 {rec['fail']} → {tag}"
            + (f" (代表: {names}, best={rec['best']:.2f})" if n_succ else "")
        )
    lines.append(
        "探索策略: 标记【已饱和】的族不得再产出结构同形的新候选 (换基础列组合、换算子机制或跨族组合);"
        "优先在【未突破】的族中做机制性变化。"
    )
    return "\n".join(lines)


_FAILURE_PRESCRIPTIONS: Dict[str, str] = {
    "COMPILE_OR_GUARD": "存在语法/树深超限/跨族嵌套/非法窗口参数，请严格遵循 AST 语法与跨族 Recipe 分步规则",
    "DUPLICATE_AST": "重复提交了已探索过的同构公式，必须更换算子机制或改变特征组合结构",
    "STAGE1_NULL": "缺失率过高 (>20%)，请减小滚动窗口 N 或避免过多分母趋零的链式除法",
    "STAGE1_DEAD": "方差接近 0 (死水特征)，请勿将 gte/lte 二值结果作为最终输出，改用连续量价算子",
    "STAGE2_TRIGGER": "DTW 形态触发次数不足 (<30)，说明曲线过于尖锐或噪声过大，请用 ema/mean/zscore/rank 做平滑归一",
    "STAGE2_ECON": "统计触发但真实超额收益 bp 不足，请结合成交额放大 (amount)、波动率 (atr/std) 或订单流 (ofi) 增强尾盘弹性",
    "STAGE2_PVAL": "跨交易日配对显著性不足 (u_pval > 0.10)，建议在 Recipe 末步增加 cs_zscore/cs_demean/cs_rank 剥离市场 Beta",
}


def _render_failure_diagnostics(failed_history: List[Dict[str, Any]]) -> Optional[str]:
    """汇总失败四分类统计并给出定向改进处方，以及提取接近成功的 Near-Miss 潜力特征。"""
    if not failed_history:
        return None

    counts: Dict[str, int] = {}
    near_misses: List[Dict[str, Any]] = []

    for it in failed_history:
        code = str(it.get("code") or "")
        if not code:
            r_str = str(it.get("reason") or "").upper()
            for k in _FAILURE_PRESCRIPTIONS:
                if k in r_str:
                    code = k
                    break
            code = code or "OTHER"
        counts[code] = counts.get(code, 0) + 1

        w_ratio = float(it.get("window_pass_ratio") or 0.0)
        if 0.25 <= w_ratio < 0.60 and it.get("ast"):
            near_misses.append(it)

    lines: List[str] = ["[失败模式统计与定向调优处方]:"]
    for code, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        advice = _FAILURE_PRESCRIPTIONS.get(code, "请检查算子组合的经济学合理性")
        lines.append(f"- {code} ({cnt} 次): {advice}")

    if near_misses:
        near_misses.sort(key=lambda x: -float(x.get("window_pass_ratio") or 0.0))
        lines.append("\n[潜力突破靶标 (Near-Miss: 已在部分月份展现显著 Alpha，建议在此基础上微调平滑窗口或增加截面中性化)]:")
        for nm in near_misses[:2]:
            w_pct = float(nm.get("window_pass_ratio") or 0.0)
            nm_name = nm.get("name", "candidate")
            nm_ast = json.dumps(nm.get("ast"), ensure_ascii=False, default=str)
            lines.append(f"- {nm_name} (窗口通过率 {w_pct:.0%}, 距阈值 60% 仅差临门一脚): {nm_ast}")

    return "\n".join(lines)


def build_rl_context_prompt(
    target_description: str,
    successful_history: List[Dict[str, Any]],
    failed_history: List[Dict[str, Any]],
    ) -> str:
    """构建注入强化学习经验回放池 (Replay Buffer) 的 User Prompt。"""
    sections: List[str] = [
    f"【当前挖掘目标与市场场景】:\n{target_description}\n"
    ]

    def _safe_json_dumps(data: Any) -> str:
        try:
            return json.dumps(data, ensure_ascii=False, default=str)
        except Exception:
            return str(data)

    # 1. Top-Reward 成功样本经验注入
    if successful_history:
        success_lines = [
            "[经验池 - 成功案例 (Top-Reward 优质因子参考)]:",
            "以下是你之前探索出并在 Walk-Forward 检验中获得高分的特征, 请提炼其微观经济学逻辑并在新假设中进行演进:",
        ]
        for i, item in enumerate(successful_history[:RL_TOPK]):
            score_val = item.get("score")
            try:
                if score_val is not None and not np.isnan(float(score_val)):
                    score_str = f"{float(score_val):.2f}"
                else:
                    score_str = "N/A"
            except (ValueError, TypeError):
                score_str = "N/A"

            reasoning = item.get("reasoning", "无记录")
            ast_json = _safe_json_dumps(item.get("ast", {}))
            success_lines.append(
                f"- 成功案例 {i + 1} (Score: {score_str}):\n"
                f"  逻辑: {reasoning}\n"
                f"  AST: {ast_json}"
            )
        sections.append("\n".join(success_lines) + "\n")

    # 2. 负反馈失败样本隔离与避坑提示
    if failed_history:
        failed_lines = [
            "[经验池 - 失败案例 (避坑与编译约束阻断记录)]:",
            "以下是你近期提出并被系统守卫拦截或未通过有效性检验的因子, **绝对不要**重复相同错误或生成类似结构:",
        ]
        for i, item in enumerate(failed_history[:RL_TOPK]):
            reason = item.get("reason", "未知错误")
            ast_json = _safe_json_dumps(item.get("ast", {}))
            failed_lines.append(
                f"- 失败案例 {i + 1} (失败归因: {reason}):\n"
                f"  AST: {ast_json}"
            )
        sections.append("\n".join(failed_lines) + "\n")

        diag_block = _render_failure_diagnostics(failed_history)
        if diag_block:
            sections.append(diag_block + "\n")

    # 3. 同族饱和度: 多样性约束, 防止同机制结构反复探索
    family_block = _render_family_stats(successful_history, failed_history)
    if family_block:
        sections.append(family_block + "\n")

    sections.append(
        "请结合上述场景目标与经验反馈，构建一个逻辑清晰、结构合规的全新特征假设，并输出严格规范的 JSON。"
    )

    return "\n".join(sections)
