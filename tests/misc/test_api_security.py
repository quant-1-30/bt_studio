#!/usr/bin/env python3
"""api_server 安全最小闭环: CORS 白名单 + 可选 Bearer token。"""

import importlib
import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _client(monkeypatch, token="secret-token"):
    monkeypatch.setenv("BT_STUDIO_API_TOKEN", token)
    monkeypatch.setenv("BT_STUDIO_CORS_ORIGINS", "https://app.example")
    import bt_studio.plugins.api_server.main as m
    importlib.reload(m)

    async def _noop(*a, **kw):
        return None

    monkeypatch.setattr(m.ResultWatcher, "start", _noop)
    monkeypatch.setattr(m.ResultWatcher, "stop", _noop)
    return m, TestClient(m.app)


def test_api_requires_bearer_when_token_configured(monkeypatch):
    _, client = _client(monkeypatch)
    assert client.get("/api/results").status_code == 401
    assert client.get("/api/v2/overview").status_code == 401  # v2 同受 token middleware 保护
    assert client.get("/api/health").status_code == 200  # health 探活放行
    ok = client.get("/api/results", headers={"Authorization": "Bearer secret-token"})
    assert ok.status_code == 200


def test_api_open_in_dev_when_token_absent(monkeypatch):
    monkeypatch.delenv("BT_STUDIO_API_TOKEN", raising=False)
    monkeypatch.setenv("BT_STUDIO_ENV", "dev")
    import bt_studio.plugins.api_server.main as m
    importlib.reload(m)

    async def _noop(*a, **kw):
        return None

    monkeypatch.setattr(m.ResultWatcher, "start", _noop)
    monkeypatch.setattr(m.ResultWatcher, "stop", _noop)
    client = TestClient(m.app)
    assert client.get("/api/results").status_code == 200


def test_api_refuses_prod_without_token(monkeypatch):
    import pytest
    monkeypatch.delenv("BT_STUDIO_API_TOKEN", raising=False)
    monkeypatch.setenv("BT_STUDIO_ENV", "prod")
    import bt_studio.plugins.api_server.main as m
    with pytest.raises(RuntimeError, match="BT_STUDIO_API_TOKEN"):
        importlib.reload(m)


def test_api_rate_limit_and_metrics_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("BT_STUDIO_ENV", "dev")
    monkeypatch.setenv("BT_STUDIO_RATE_LIMIT_PER_MIN", "3")
    from bt_studio.plugins.api_server import monitor_api as MA
    monkeypatch.setattr(
        MA, "_index",
        MA.MonitorIndex(result_root=str(tmp_path), db_path=str(tmp_path / "m.sqlite")),
    )
    _, client = _client(monkeypatch, token="secret-token")
    headers = {"Authorization": "Bearer secret-token"}

    # /metrics 端点返回 Prometheus 文本指标
    m_resp = client.get("/metrics", headers=headers)
    assert m_resp.status_code == 200
    assert "bt_studio_viable_features_total" in m_resp.text

    # 连续请求超过 3 req/min 触发 429 限流 (health 探活豁免)
    client.get("/api/results", headers=headers)
    client.get("/api/results", headers=headers)
    r_limited = client.get("/api/results", headers=headers)
    assert r_limited.status_code == 429
    assert client.get("/api/health").status_code == 200
