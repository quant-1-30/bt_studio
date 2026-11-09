#!/usr/bin/env python3
"""LLM metering 模块行为锁定: 记账/恢复/预检熔断/异常透传/认证豁免。"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bt_studio.agent.vendors.base import (
    BaseLLMProvider,
    BudgetExceeded,
    LlmUsageMeter,
    MeteredProvider,
)
from bt_studio.agent.vendors.google import GoogleProvider
from bt_studio.agent.vendors.openai_compatible import OpenAICompatibleProvider


class FakeInner(BaseLLMProvider):
    model = "fake-model"

    def __init__(self):
        self.calls = 0
        self.last_usage = None

    def generate(self, user_prompt, system_prompt=None, temperature=0.2, json_mode=True):
        self.calls += 1
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        return "ok"


def _meter(tmp_path, budget=None):
    return LlmUsageMeter(budget_total=budget, ledger_dir=str(tmp_path))


def test_meter_records_usage_and_ledger(tmp_path):
    inner = FakeInner()
    m = _meter(tmp_path)
    p = MeteredProvider(inner, m)
    assert p.generate("x") == "ok"
    assert m.totals == {"prompt_tokens": 100, "completion_tokens": 50,
                        "total_tokens": 150, "calls": 1}
    ledger = list(tmp_path.glob("usage_*.jsonl"))[0]
    rec = json.loads(ledger.read_text().splitlines()[0])
    assert rec["status"] == "ok" and rec["usage"]["total_tokens"] == 150


def test_meter_restores_totals_on_restart(tmp_path):
    m1 = _meter(tmp_path)
    m1.record("Fake", "m", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 1.0, "ok")
    m2 = _meter(tmp_path)  # 新进程语义: 重建即恢复当日累计
    assert m2.totals["total_tokens"] == 15 and m2.totals["calls"] == 1


def test_budget_precheck_blocks_before_call(tmp_path):
    inner = FakeInner()
    m = _meter(tmp_path, budget=150)
    p = MeteredProvider(inner, m)
    p.generate("x")  # 恰好用满
    with pytest.raises(BudgetExceeded):
        p.generate("x")  # 预检在调用前熔断
    assert inner.calls == 1  # 没有烧第二发


def test_metered_generate_uses_lock_and_second_process_restores(tmp_path):
    inner1 = FakeInner()
    m1 = _meter(tmp_path, budget=150)
    MeteredProvider(inner1, m1).generate("x")
    assert list(tmp_path.glob("usage_*.jsonl.lock")), "预算临界区必须生成账本锁文件"

    # 第二个进程语义: 从同一账本恢复累计后, 下一次调用前熔断
    inner2 = FakeInner()
    m2 = _meter(tmp_path, budget=150)
    with pytest.raises(BudgetExceeded):
        MeteredProvider(inner2, m2).generate("x")
    assert inner2.calls == 0


def test_error_recorded_and_reraised(tmp_path):
    class Boom(FakeInner):
        def generate(self, *a, **kw):
            self.calls += 1
            raise RuntimeError("net down")

    inner = Boom()
    m = _meter(tmp_path)
    with pytest.raises(RuntimeError):
        MeteredProvider(inner, m).generate("x")
    rec = json.loads(list(tmp_path.glob("usage_*.jsonl"))[0].read_text().splitlines()[0])
    assert rec["status"] == "error" and rec["error_type"] == "RuntimeError"


def test_auth_error_failfast_no_retry(monkeypatch):
    """401/403 豁免重试: 单次尝试即上抛 (修正'所有异常都退避'的重试 bug)。"""
    prov = OpenAICompatibleProvider(api_key="k", base_url="http://localhost:1", model="m")
    attempts = []

    class Err(Exception):
        status_code = 401

    def fake_create(**kw):
        attempts.append(1)
        raise Err("invalid key")

    monkeypatch.setattr(prov.client.chat.completions, "create", fake_create)
    with pytest.raises(Err):
        prov.generate("x")
    assert len(attempts) == 1  # 未重试


def test_meter_rolls_over_at_midnight(tmp_path):
    m = _meter(tmp_path, budget=1)
    # 显式构造昨日账本/昨日累计; 跨午夜 precheck 应切到今天且预算归零
    old = tmp_path / "usage_20000101.jsonl"
    old.write_text(json.dumps({"usage": {"total_tokens": 5}}) + "\n", encoding="utf-8")
    m.today = "20000101"
    m._path = str(old)
    m.totals["total_tokens"] = 5
    m.precheck()
    assert not m._path.endswith("usage_20000101.jsonl")
    assert m.totals["total_tokens"] == 0


def test_google_usage_metadata_mapped_to_last_usage():
    def fake_generate_content(**kw):
        return SimpleNamespace(
            text="{}",
            usage_metadata=SimpleNamespace(
                prompt_token_count=11, candidates_token_count=7, total_token_count=18
            ),
        )

    prov = GoogleProvider.__new__(GoogleProvider)  # 不初始化真实 genai.Client
    prov.client = SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content))
    prov.model = "gemini-test"
    prov.last_usage = None

    assert prov.generate("u", "s") == "{}"
    assert prov.last_usage == {
        "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_bad_request_degrades_response_format_once(monkeypatch):
    prov = OpenAICompatibleProvider(api_key="k", base_url="http://localhost:1", model="m")
    seen = []

    class Err(Exception):
        status_code = 400

    def fake_create(**kw):
        seen.append(dict(kw))
        if "response_format" in kw:
            raise Err("response_format unsupported")
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )

    monkeypatch.setattr(prov.client.chat.completions, "create", fake_create)
    assert prov.generate("x") == "ok"
    assert len(seen) == 2 and "response_format" in seen[0] and "response_format" not in seen[1]


def test_rate_limit_retries_with_backoff(monkeypatch):
    prov = OpenAICompatibleProvider(api_key="k", base_url="http://localhost:1", model="m")
    attempts, sleeps = [], []

    class Err(Exception):
        status_code = 429

    def fake_create(**kw):
        attempts.append(1)
        if len(attempts) < 3:
            raise Err("rate limited")
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )

    monkeypatch.setattr(prov.client.chat.completions, "create", fake_create)
    monkeypatch.setattr("bt_studio.agent.vendors.openai_compatible.time.sleep", sleeps.append)
    assert prov.generate("x") == "ok"
    assert len(attempts) == 3 and sleeps == [1, 2]


def test_failover_provider_circuit_breaker():
    from bt_studio.agent.vendors import FailoverLLMProvider

    class _FailingPrimary(BaseLLMProvider):
        def __init__(self):
            self.calls = 0
            self.model = "primary-m"
        def generate(self, *a, **kw):
            self.calls += 1
            raise RuntimeError("primary down")

    class _GoodFallback(BaseLLMProvider):
        def __init__(self):
            self.calls = 0
            self.model = "fallback-m"
            self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        def generate(self, *a, **kw):
            self.calls += 1
            return '{"ok": 1}'

    prim, fb = _FailingPrimary(), _GoodFallback()
    fo = FailoverLLMProvider(prim, fb, failure_threshold=2, cooldown_s=60.0)

    # 前 2 次调用先试 primary 失败再切 fallback -> 触发断路器打开
    assert fo.generate("u") == '{"ok": 1}'
    assert fo.generate("u") == '{"ok": 1}'
    assert prim.calls == 2 and fb.calls == 2

    # 第 3 次调用处于冷却期内 -> 直接跳过 primary 调 fallback (零延迟)
    assert fo.generate("u") == '{"ok": 1}'
    assert prim.calls == 2 and fb.calls == 3
    assert fo.model == "fallback-m"
