#!/usr/bin/env python3
"""agent 子系统 bugfix 回归测试 (TDD red→green)。

覆盖 2026-09 代码审计定位的逻辑 bug:
    H7  vendors/google.py 文件头部误粘贴草稿 → import 即 SyntaxError;
        默认模型名两处定义不一致
    M13 Stage1 无逐候选隔离 (一个坏引用炸整批) + Stage1.5 幸存特征
        应在 Stage2 之前落盘 (docs/architecture.md 数据流: 预过滤→落盘→滚动评估)
    M15 ReplayBuffer.add_failure 只与队首去重 → 失败经验重复入池
    M16 falsy AST ({} / []) 被跳过, 不进失败池 → RL 负反馈盲区
"""

import ast as pyast
import datetime
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import bt_studio.agent.harness as harness_mod
import bt_studio.agent.flywheel as flywheel_mod
from bt_studio.agent.flywheel import ReplayBuffer, RLFeatureFlywheel
from bt_studio.agent.harness import TwoStageAgentHarness
from bt_studio.compiler.ast import ast_fingerprint


VENDORS_DIR = Path(__file__).resolve().parent.parent.parent / "bt_studio" / "agent" / "vendors"


# ---------------------------------------------------------------------------
# H7: google vendor 源码可解析 + 默认模型单一定义
# ---------------------------------------------------------------------------

def test_google_vendor_source_parses():
    src = (VENDORS_DIR / "google.py").read_text(encoding="utf-8")
    tree = pyast.parse(src)  # 旧文件 __future__ import 前有可执行语句 -> SyntaxError
    classes = [n.name for n in pyast.walk(tree) if isinstance(n, pyast.ClassDef)]
    assert "GoogleProvider" in classes
    # 模块顶层不得残留草稿期的 API 调用
    top_levels = [
        n.func.attr if isinstance(n.func, pyast.Attribute) else ""
        for n in tree.body if isinstance(n, pyast.Expr) and isinstance(n.value, pyast.Call)
    ]
    assert not any(a in ("create", "generate_content") for a in top_levels), top_levels


def test_google_default_model_defined_once():
    """默认模型名只能有一处字面量定义 (base 常量), 不得两处漂移。"""
    hits = []
    for p in VENDORS_DIR.glob("*.py"):
        for mo in __import__("re").finditer(r'"gemini-[^"]+"', p.read_text(encoding="utf-8")):
            hits.append((p.name, mo.group(0)))
    literals = {lit for _, lit in hits}
    assert len(literals) <= 1, f"默认模型字面量多于一处: {hits}"


# ---------------------------------------------------------------------------
# M15: ReplayBuffer 失败池全量去重
# ---------------------------------------------------------------------------

def test_replay_buffer_failure_dedup_full_pool():
    buf = ReplayBuffer()
    a = {"reason": "r1", "ast": {"op": "sign", "args": [{"col": "close"}]}}
    b = {"reason": "r2", "ast": {"op": "log", "args": [{"col": "volume"}]}}
    buf.add_failure(a)
    buf.add_failure(b)
    buf.add_failure(a)  # 与队首不相邻的重复 —— 不得再次入池
    assert len(buf.failed_history) == 2, buf.failed_history
    assert ast_fingerprint(buf.failed_history[0]["ast"]) != \
        ast_fingerprint(buf.failed_history[1]["ast"])


# ---------------------------------------------------------------------------
# M16: falsy AST 候选进入失败池 (不再静默跳过)
# ---------------------------------------------------------------------------

def test_flywheel_records_empty_ast_failure(tmp_path, monkeypatch):
    # _persist_step 会真实落盘 LLM_RUN_DIR; 测试必须隔离到 tmp_path,
    # 否则 stub 飞轮会把 h1_s1 写进生产 result/llm/runs。
    monkeypatch.setattr(flywheel_mod, "LLM_RUN_DIR", str(tmp_path))

    class _StubVendor:
        def generate(self, user_prompt, system_prompt):
            return json.dumps({
                "hypothesis_id": "h1",
                "economic_reasoning": "r",
                "sub_features": [{}],
            })

    class _StubHarness:
        def run(self, candidates, hf_lf, dret_lf):
            from bt_studio.agent.harness import FeatureMiningResult
            return [FeatureMiningResult(feature_name="rejected_0", raw_ast={})]

    monkeypatch.setattr(flywheel_mod, "get_llm_provider", lambda vendor=None: _StubVendor())
    fw = RLFeatureFlywheel(harness=_StubHarness())

    results = fw.step("t", None, None)
    assert len(results) == 1
    assert len(fw.buffer.failed_history) == 1, \
        "空 AST 候选必须进入失败池, 否则 RL 负反馈存在盲区"


# ---------------------------------------------------------------------------
# M13a: Stage1 逐候选隔离
# ---------------------------------------------------------------------------

def _tiny_hf() -> pl.LazyFrame:
    days = [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]
    rows = {
        "day": [d for d in days for _ in range(6)],
        "sid": ["600001"] * 12,
        "bar_idx": list(range(6)) * 2,
        "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 2,
    }
    return pl.DataFrame(rows).lazy()


def test_stage1_isolates_bad_candidates():
    """引用不存在列的候选只应淘汰自己, 不得炸掉整批。"""
    cc = {"eps": 1e-8, "min_window_pass_ratio": 0.6}
    harness = TwoStageAgentHarness(common_config=cc)

    candidates = [
        # 正常编译但恒为 0 -> stage1 std 过滤淘汰 (不出局到 stage2)
        {"op": "mul", "args": [{"col": "x"}, 0.0]},
        # 引用不存在的列: 编译期惰性通过, collect 期才爆
        {"op": "abs", "args": [{"col": "ghost_column"}]},
    ]
    results = harness.run(candidates, _tiny_hf(), _tiny_hf())  # 不得抛异常
    assert len(results) == 2
    bad = [r for r in results if "ghost" in (r.stage1_prefilter.reason
                                             if r.stage1_prefilter else "")]
    assert bad, "坏候选应被记为 stage1 失败"
    assert all(r.stage_passed == "rejected" for r in results)


# ---------------------------------------------------------------------------
# M13b: Stage1.5 —— 幸存特征在 Stage2 之前落盘
# ---------------------------------------------------------------------------

def test_stage15_persists_survivors_before_stage2(tmp_path, monkeypatch):
    feat_root = str(tmp_path)
    monkeypatch.setattr(harness_mod, "FEATURE_DIR", feat_root)

    cc = {"eps": 1e-8, "min_window_pass_ratio": 0.99}
    harness = TwoStageAgentHarness(common_config=cc)

    # Stage2 全部失败 (mock), 但 Stage1 幸存者必须已经落盘
    def _failing_windows(self, hf_enhanced, dret_lf, feature_names, months):
        from bt_studio.agent.harness import EvalResult
        return {f: [EvalResult(feature_name=f, status="failed",
                               reason="forced_fail") for _ in months]
                for f in feature_names}

    monkeypatch.setattr(
        TwoStageAgentHarness, "_evaluate_windows", _failing_windows)

    results = harness.run(
        [{"op": "abs", "args": [{"col": "x"}]}], _tiny_hf(), _tiny_hf())

    assert len(results) == 1 and not results[0].stage2_passed
    persisted = list(Path(feat_root).rglob("hf_*.parquet"))   # v2: 直接指纹目录
    assert persisted, "Stage1 幸存特征应在 Stage2 判定前落盘 (Stage1.5)"
    # 列序确定性锚点: day 列固定在首位 (set 迭代序不可作为列序来源)
    schema = pl.read_parquet_schema(persisted[0])
    assert schema.names()[0] == "day"


def test_recipe_final_name_collision_does_not_overwrite_columns():
    """多个 Recipe 末步同名时, 编译期必须先隔离末步再统一 rename。

    生产烟测曾复现: 两个 Recipe 末步都叫 final_alpha, 先物化原名导致后候选
    覆盖先候选, Stage1 bulk 再按 compiled_names 找 final_alpha 时 ColumnNotFound。
    """
    cc = {"eps": 1e-8, "min_window_pass_ratio": 0.6}
    harness = TwoStageAgentHarness(common_config=cc)
    candidates = [
        [
            {"name": "d", "ast": {"op": "delta", "args": [{"col": "x"}, 1]}},
            {"name": "final_alpha", "ast": {"op": "abs", "args": [{"col": "d"}]}},
        ],
        [
            {"name": "d", "ast": {"op": "delta", "args": [{"col": "x"}, 1]}},
            {"name": "final_alpha", "ast": {"op": "mul", "args": [{"col": "d"}, 2.0]}},
        ],
    ]

    hf_enhanced, records, compiled_names = harness._compile_asts(candidates, _tiny_hf())
    assert len(compiled_names) == 2 and all(n.startswith("f_") for n in compiled_names)
    assert [r.get("display_name") for r in records] == ["final_alpha", "final_alpha"]
    assert all(r["error"] is None for r in records)

    df = hf_enhanced.collect()
    name0, name1 = compiled_names
    assert {name0, name1, "_c0_d", "_c1_d"}.issubset(set(df.columns))
    first = df[name0].to_list()
    second = df[name1].to_list()
    pairs = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
    assert pairs, "两列最终特征都必须有非空样本"
    # 第二候选是第一候选的 2 倍缩放; 若发生覆盖, 两列会相等或首列丢失。
    assert any(abs(b - 2.0 * a) < 1e-12 for a, b in pairs)
    assert any(abs(b - a) > 1e-12 for a, b in pairs)


def test_family_signature_ignores_recipe_intermediate_names():
    """Recipe 中间步骤变量名变化不得改变特征族签名 (同族饱和度防打转)。"""
    from bt_studio.agent.prompt import _family_signature

    r1 = [
        {"name": "ret_dir", "ast": {"op": "sign", "args": [{"op": "delta", "args": [{"col": "close"}, 1]}]}},
        {"name": "final_alpha", "ast": {"op": "cs_demean", "args": [{"col": "ret_dir"}]}},
    ]
    r2 = [
        {"name": "renamed_step_x", "ast": {"op": "sign", "args": [{"op": "delta", "args": [{"col": "close"}, 5]}]}},
        {"name": "out_alpha", "ast": {"op": "cs_demean", "args": [{"col": "renamed_step_x"}]}},
    ]
    sig1 = _family_signature(r1)
    sig2 = _family_signature(r2)
    assert sig1 == sig2 == "ops[cs_demean+delta+sign] × cols[close]"
    assert "ret_dir" not in sig1 and "renamed_step_x" not in sig2


def test_stage2_failure_preserves_specific_reason_and_code(tmp_path, monkeypatch):
    """全部窗口失败时, stage2_eval 保留真实失败原因而非退化为 None (STAGE2_NO_EVAL)。"""
    feat_root = str(tmp_path)
    monkeypatch.setattr(harness_mod, "FEATURE_DIR", feat_root)

    cc = {"eps": 1e-8, "min_window_pass_ratio": 0.6, "stage2_max_pval": 0.1}
    harness = TwoStageAgentHarness(common_config=cc)

    def _failing_dtw_windows(self, hf_enhanced, dret_lf, feature_names, months):
        from bt_studio.agent.harness import EvalResult
        return {
            f: [EvalResult(feature_name=f, status="failed", reason="DTW (n=3) < trigger (30)")
                for _ in months]
            for f in feature_names
        }

    monkeypatch.setattr(TwoStageAgentHarness, "_evaluate_windows", _failing_dtw_windows)
    results = harness.run([{"op": "abs", "args": [{"col": "x"}]}], _tiny_hf(), _tiny_hf())
    assert len(results) == 1
    r = results[0]
    assert not r.stage2_passed
    assert r.stage2_eval is not None
    assert RLFeatureFlywheel._failure_code(r) == "STAGE2_TRIGGER"
    assert "DTW" in RLFeatureFlywheel._failure_reason(r)


def test_eval_feature_on_grid_does_not_short_circuit_on_high_pval(monkeypatch):
    """第 1 个网格点 is_success 但 u_pval > gate 时不得提前 break 错失第 2 个显著网格点。"""
    import bt_studio.agent.harness as ew
    from bt_studio.agent.harness import EvalResult

    calls = []

    def _fake_eval(panel_lf, feat, cc, cfg):
        calls.append(cfg["threshold_r"])
        if cfg["threshold_r"] == 0.55:
            # 第 1 点: 物理成功但 pval=0.35 > 0.10
            return EvalResult(feature_name=feat, status="success", score=1.0, u_pval=0.35, config=cfg)
        # 第 2 点: 显著通过 pval=0.02 <= 0.10
        return EvalResult(feature_name=feat, status="success", score=3.5, u_pval=0.02, config=cfg)

    monkeypatch.setattr(ew, "evaluate_feature", _fake_eval)
    monkeypatch.setattr(ew, "extract_curves_from_panel", lambda lf, tc, cc: lf)

    dummy_df = pl.DataFrame({"day": [datetime.date(2024, 1, 2)], "sid": ["a"]})
    combos_by_ds = {
        3: [
            {"downsample": 3, "motif_minutes": 30, "threshold_r": 0.55},
            {"downsample": 3, "motif_minutes": 30, "threshold_r": 0.65},
        ]
    }
    res = ew.eval_feature_on_grid(
        dummy_df.lazy(), dummy_df.lazy(), dummy_df, "f_test",
        {"stage2_max_pval": 0.10}, combos_by_ds,
    )
    assert calls == [0.55, 0.65]
    assert res.u_pval == 0.02 and res.score == 3.5


if __name__ == "__main__":
    raise SystemExit("run via pytest")
