#!/usr/bin/env python3
"""flywheel 断点续跑行为锁定: 步边界恢复 / pending 防重烧 / 指纹隔离。"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bt_studio.agent.flywheel import RLFeatureFlywheel
from bt_studio.agent.vendors.dummy import DummyProvider


class StubHarness:
    """记录调用次数的可控 harness; 可注入崩溃。

    进程级崩溃用 SystemExit (BaseException, 不被 step 的 except Exception 吸收)
    —— pending 防的是进程死亡/断电/OOM, Python 级 harness 异常按设计转特征失败。
    """

    def __init__(self, crash_on_step=None):
        self.calls = 0
        self.crash_on_step = crash_on_step

    def run(self, candidates, hf, dret):
        self.calls += 1
        if self.crash_on_step is not None and self.calls == self.crash_on_step:
            raise SystemExit("simulated process kill mid-stage2")
        return []


class CountingDummy(DummyProvider):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, *a, **kw):
        self.calls += 1
        return f"""```json
{{
    "hypothesis_id": "hyp_step_{self.calls}",
    "economic_reasoning": "sign(close.diff)*amount 累计占比,截面去均值还原 OFI。",
    "sub_features": [
        {{"op": "mean", "args": [{{"col": "close"}}, {self.calls + 2}]}}
    ]
}}
```"""


def _fw(tmp, harness, run_dir=None, fp="fp" * 8, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr("bt_studio.agent.flywheel.FEATURE_DIR", str(tmp))
    fw = RLFeatureFlywheel(harness=harness, vendor="dummy",
                           run_dir=run_dir, config_fingerprint=fp)
    fw._seen_candidate_fps.clear()
    fw.llm_vendor = CountingDummy()
    return fw


def test_state_written_each_step(tmp_path, monkeypatch):
    monkeypatch.setattr("bt_studio.agent.flywheel.LLM_RUN_DIR", str(tmp_path))
    h = StubHarness()
    fw = _fw(tmp_path, h, fp="aa" * 8)
    fw.step("t", None, None)
    st = json.load(open(fw._state_path()))
    assert st["step_counter"] == 1 and st["config_fingerprint"] == "aa" * 8
    assert not os.path.exists(fw._pending_path())


def test_pending_resumes_without_llm_reburn(tmp_path, monkeypatch):
    """崩溃在 Stage2 中途: 恢复时用暂存候选, 不再调 LLM。"""
    monkeypatch.setattr("bt_studio.agent.flywheel.LLM_RUN_DIR", str(tmp_path))
    monkeypatch.setattr("bt_studio.agent.flywheel.FEATURE_DIR", str(tmp_path / "features"))
    h = StubHarness(crash_on_step=1)
    fw = _fw(tmp_path, h, fp="bb" * 8)
    with pytest.raises(SystemExit):
        fw.step("t", None, None)
    assert fw.llm_vendor.calls == 1
    assert os.path.exists(fw._pending_path())  # 候选已暂存

    # 新进程: run_dir 恢复 (包括通过 Stage2 的成功特征, 验证 reasoning 不抛 NameError)
    class SuccessStubHarness:
        def run(self, candidates, hf, dret):
            from bt_studio.agent.harness import FeatureMiningResult
            return [
                FeatureMiningResult(
                    feature_name="f_1234567890ab",
                    raw_ast=candidates[0] if candidates else {"op": "abs", "args": [{"col": "close"}]},
                    ast_fingerprint="1234567890ab",
                    stage2_passed=True,
                    final_score=2.5,
                    window_pass_ratio=0.8,
                )
            ]

    h2 = SuccessStubHarness()
    fw2 = _fw(tmp_path, h2, run_dir=fw._run_dir, fp="bb" * 8)
    fw2.llm_vendor = CountingDummy()
    res = fw2.step("t", None, None)  # 走 pending, 无 LLM, 且成功入池不抛 UnboundLocalError
    assert fw2.llm_vendor.calls == 0
    assert fw2._step_counter == 1
    assert len(res) == 1 and res[0].stage2_passed
    assert len(fw2.buffer.successful_history) == 1
    assert "OFI" in fw2.buffer.successful_history[0]["reasoning"]
    assert not os.path.exists(fw2._pending_path())


def test_run_resumes_remaining_steps_only(tmp_path, monkeypatch):
    """断点续跑时 flywheel.run(n_steps=4) 只跑剩余步数 (4 - 2 = 2)。"""
    monkeypatch.setattr("bt_studio.agent.flywheel.LLM_RUN_DIR", str(tmp_path))
    h1 = StubHarness()
    fw1 = _fw(tmp_path, h1, fp="11" * 8)
    fw1.step("t", None, None)
    fw1.step("t", None, None)
    assert fw1._step_counter == 2

    h2 = StubHarness()
    fw2 = _fw(tmp_path, h2, run_dir=fw1._run_dir, fp="11" * 8)
    fw2.run("t", None, None, n_steps=4)
    assert h2.calls == 2
    assert fw2._step_counter == 4


def test_fingerprint_mismatch_starts_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr("bt_studio.agent.flywheel.LLM_RUN_DIR", str(tmp_path))
    h = StubHarness()
    fw = _fw(tmp_path, h, fp="cc" * 8)
    fw.step("t", None, None)
    # 不同指纹: 不识别旧 state → 新 run 目录, 计数器归零
    fw2 = _fw(tmp_path, StubHarness(), run_dir=fw._run_dir, fp="dd" * 8)
    assert fw2._step_counter == 0
    assert fw2._run_dir != fw._run_dir


def test_buffer_restored_across_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("bt_studio.agent.flywheel.LLM_RUN_DIR", str(tmp_path))
    h = StubHarness()
    fw = _fw(tmp_path, h, fp="ee" * 8)
    fw.buffer.add_success({"name": "x", "score": 1.0, "ast": {"op": "abs", "args": [{"col": "close"}]}})
    fw.buffer.add_failure({"reason": "r", "ast": {"op": "neg", "args": [{"col": "open"}]}})
    fw._save_state("t")
    fw2 = _fw(tmp_path, StubHarness(), run_dir=fw._run_dir, fp="ee" * 8)
    assert len(fw2.buffer.successful_history) == 1
    assert len(fw2.buffer.failed_history) == 1
    assert len(fw2.buffer._success_fps) == 1 and len(fw2.buffer._failed_fps) == 1


def test_failure_dedup_uses_same_fp_after_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("bt_studio.agent.flywheel.LLM_RUN_DIR", str(tmp_path))
    fw = _fw(tmp_path, StubHarness(), fp="ff" * 8)
    item = {"reason": "same_gate_reject", "ast": {"op": "neg", "args": [{"col": "open"}]}}
    fw.buffer.add_failure(item)
    fw._save_state("t")

    fw2 = _fw(tmp_path, StubHarness(), run_dir=fw._run_dir, fp="ff" * 8)
    n0 = len(fw2.buffer.failed_history)
    fw2.buffer.add_failure({"reason": "same_gate_reject",
                            "ast": {"op": "neg", "args": [{"col": "open"}]}})
    assert len(fw2.buffer.failed_history) == n0  # resume 后同 AST+同 reason 不重复入池
