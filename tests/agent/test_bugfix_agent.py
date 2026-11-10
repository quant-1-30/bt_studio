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

    cc = {"eps": 1e-8, "min_window_pass_ratio": 0.6, "u_pval": 0.1}
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
        dummy_df.lazy(), dummy_df, "f_test",
        {"u_pval": 0.10}, combos_by_ds,
    )
    assert calls == [0.55, 0.65]
    assert res.u_pval == 0.02 and res.score == 3.5


def test_recipe_canonical_fingerprint_and_named_ast_wrapper():
    """Verify Recipe list canonicalization, base market column shadowing guard, and named AST wrapper compilation."""
    from bt_studio.compiler.ast import ASTCompilationError, ast_fingerprint, compile_recipe

    r1 = [{"name": " s1 ", "ast": {"op": "mean", "args": [{"col": " close "}]}}]
    r2 = [{"name": "s1", "ast": {"op": "mean", "args": [{"col": "close"}, 5.0]}}]
    assert ast_fingerprint(r1) == ast_fingerprint(r2)

    with pytest.raises(ASTCompilationError, match="shadow"):
        compile_recipe([{"name": "close", "ast": {"op": "abs", "args": [{"col": "open"}]}}])

    harness = TwoStageAgentHarness(common_config={"eps": 1e-8})
    _, records, compiled_names = harness._compile_asts(
        [{"name": "wrapped_alpha", "ast": {"op": "abs", "args": [{"col": "x"}]}}],
        _tiny_hf(),
    )
    assert len(compiled_names) == 1
    assert records[0]["display_name"] == "wrapped_alpha"
    assert records[0]["error"] is None


def test_resolve_walk_forward_windows_warmup_strip_and_tail_align():
    """Verify resolve_walk_forward_windows strips warmup_months before start_date and aligns tail remainder months."""
    from bt_studio.utils.months import resolve_walk_forward_windows, window_tag

    # Warmup months 202310..202312 must be stripped when start_date=20240101
    avail = [202310, 202311, 202312, 202401, 202402, 202403, 202404, 202405]
    wins = resolve_walk_forward_windows(
        avail,
        {"start_date": 20240101, "end_date": 20240531, "train_window": 3, "oos_step": 2},
    )
    assert wins == [[202401, 202402, 202403], [202403, 202404, 202405]]
    assert window_tag(wins[0]) == "202401_202403_3"

    # Tail remainder alignment when (n_months - train_window) % oos_step != 0
    wins_tail = resolve_walk_forward_windows(
        [202401, 202402, 202403, 202404],
        {"start_date": 20240101, "end_date": 20240430, "train_window": 3, "oos_step": 2},
    )
    assert wins_tail == [[202401, 202402, 202403], [202402, 202403, 202404]]


def test_on_demand_walk_forward_lazy_fetch_and_early_stop(tmp_path, monkeypatch):
    """Verify on-demand Walk-Forward mode fetches only active windows, reuses overlapping monthly parquets, and stops downloading on early-stop."""
    monkeypatch.setattr(harness_mod, "FEATURE_DIR", str(tmp_path / "features"))
    monkeypatch.setattr(harness_mod, "EVAL_CACHE_ROOT", str(tmp_path / "eval_cache"))

    fetched_ranges = []

    def _fake_prepare_tick(start_date, end_date, sids):
        fetched_ranges.append((start_date, end_date, tuple(sids)))
        # Synthesize minute bars for warmup + requested months
        days = [
            datetime.date(2023, 12, 15),
            datetime.date(2024, 1, 10),
            datetime.date(2024, 2, 10),
            datetime.date(2024, 3, 10),
            datetime.date(2024, 4, 10),
        ]
        d_lo = datetime.datetime.strptime(str(start_date), "%Y%m%d").date()
        d_hi = datetime.datetime.strptime(str(end_date), "%Y%m%d").date()
        sel_days = [d for d in days if d_lo <= d <= d_hi]
        rows = {
            "day": [d for d in sel_days for _ in range(12)],
            "sid": [b"600001"] * (12 * len(sel_days)),
            "bar_idx": list(range(12)) * len(sel_days),
            "close": [float(i + 1) for i in range(12)] * len(sel_days),
        }
        return pl.DataFrame(rows).lazy()

    monkeypatch.setattr(harness_mod, "prepare_tick", _fake_prepare_tick)
    monkeypatch.setattr(
        harness_mod,
        "build_static_panel",
        lambda w_lf, d_lf, cc, is_train=True: pl.LazyFrame({"day": [datetime.date(2024, 1, 10)], "sid": [b"600001"]}),
    )

    eval_calls = []

    def _fake_eval_on_grid(feat_lf, static_df, feat, cc, combos_by_ds):
        df = feat_lf.collect()
        yms = sorted(df.select((pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).unique()).to_series().to_list())
        eval_calls.append(tuple(yms))
        return harness_mod.EvalResult(feature_name=feat, status="failed", reason="doomed_window")

    monkeypatch.setattr(harness_mod, "eval_feature_on_grid", _fake_eval_on_grid)

    universe_sids = {
        202312: [b"600001"],  # warmup month
        202401: [b"600001"],
        202402: [b"600001"],
        202403: [b"600001"],
        202404: [b"600001"],
    }
    cc = {
        "start_date": 20240101,
        "end_date": 20240430,
        "train_window": 2,
        "oos_step": 1,
        "warmup_months": 1,
        "min_window_pass_ratio": 0.99,  # 0 failures allowed -> early stop after Window 1!
        "eval_procs": 1,
        "eps": 1e-8,
    }
    dret_lf = pl.LazyFrame({"day": [datetime.date(2024, 1, 10)], "sid": [b"600001"], "close": [10.0]})
    harness = TwoStageAgentHarness(common_config=cc, universe_sids=universe_sids)

    results = harness.run([{"op": "abs", "args": [{"col": "close"}]}], None, dret_lf)
    assert len(results) == 1
    # Window 1 ([202401, 202402]) evaluated and failed -> early stop!
    # Windows 2 ([202402, 202403]) and 3 ([202403, 202404]) must NEVER be evaluated or fetched!
    assert eval_calls == [(202401, 202402)]
    assert all(end <= 20240229 for _, end, _ in fetched_ranges)

    # Now test multi-window progression: overlapping month 202402 is NOT re-compiled when advancing to Window 2 ([202402, 202403])
    fetched_ranges.clear()
    eval_calls.clear()
    monkeypatch.setattr(
        harness_mod,
        "eval_feature_on_grid",
        lambda feat_lf, static_df, feat, cc, combos_by_ds: (
            eval_calls.append(tuple(sorted(feat_lf.collect().select((pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).unique()).to_series().to_list())))
            or harness_mod.EvalResult(feature_name=feat, status="success", score=2.5, u_pval=0.01, trigger_count=50)
        ),
    )
    harness2 = TwoStageAgentHarness(common_config=dict(cc, _eval_data_fp="pass_run"), universe_sids=universe_sids)
    results2 = harness2.run([{"op": "mean", "args": [{"col": "close"}, 3]}], None, dret_lf)
    assert len(results2) == 1 and results2[0].stage2_passed
    assert eval_calls == [(202401, 202402), (202402, 202403), (202403, 202404)]
    # Fetch calls: (1) Stage 1: 202401 (+warmup 202312), (2) W1 missing 202402 (+warmup 202401),
    # (3) W2 missing 202403 (+warmup 202402), (4) W3 missing 202404 (+warmup 202403)
    assert [end for _, end, _ in fetched_ranges] == [20240131, 20240229, 20240331, 20240430]


if __name__ == "__main__":
    raise SystemExit("run via pytest")
