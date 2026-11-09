#!/usr/bin/env python3
"""monitor v2 API/WS 行为锁定: 只读边界 + review 写路径 + WS snapshot/背压语义。"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

fastapi_tc = pytest.importorskip("fastapi.testclient")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bt_studio.plugins.api_server import monitor_api as MA


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("bt_studio.plugins.api_server.indexer.RESULT_ROOT", str(tmp_path))
    (tmp_path / "artifacts" / "usage").mkdir(parents=True)
    today_jsonl = tmp_path / "artifacts" / "usage" / "usage_20260101.jsonl"
    today_jsonl.write_text(json.dumps({
        "provider": "OpenAICompatibleProvider", "model": "glm-5.3", "status": "ok",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "latency_ms": 800.0}) + "\n")

    app = FastAPI()
    app.include_router(MA.router, prefix="/api/v2")
    app.include_router(MA.ws_router)
    MA._index = MA.MonitorIndex(result_root=str(tmp_path),
                                db_path=str(tmp_path / "cache" / "m.sqlite"))
    with TestClient(app) as tc:
        yield tc, tmp_path


def test_readonly_endpointsSmoke(client):
    tc, tmp = client
    for path in ("/api/v2/overview", "/api/v2/runs", "/api/v2/usage/daily",
                 "/api/v2/coverage", "/api/v2/alerts", "/api/v2/features"):
        r = tc.get(path)
        assert r.status_code == 200, (path, r.text)
    ov = tc.get("/api/v2/overview").json()
    assert ov["usage_all_time"]["total_tokens"] == 150  # 账本日 ≠ 今日时当日为 0, 全量仍可见
    assert tc.get("/api/v2/usage/daily?from=20260101&to=20260101").json()["items"][0]["total_tokens"] == 150
    assert tc.get("/api/v2/runs/nonexistent").json()["steps"] == []


def test_review_post_writes_both_stores(client):
    tc, tmp = client
    r = tc.post("/api/v2/review", json={
        "fingerprint": "fpXYZ", "decision": "promote_candidate",
        "note": "n", "operator": "bob"})
    assert r.status_code == 200
    assert (tmp / "artifacts" / "review" / "fpXYZ.json").exists()
    r2 = tc.post("/api/v2/review", json={
        "fingerprint": "fpXYZ", "decision": "auto_promote", "note": "", "operator": ""})
    assert r2.status_code == 422  # 非法 decision 拒绝 — 写边界白名单


def test_ws_snapshot_and_auth(client, monkeypatch):
    tc, tmp = client
    with tc.websocket_connect("/ws/v2") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot" and "data" in msg
    # token 保护: 配置后未授权连接被拒 (close code 4401 → TestClient 抛 WebSocketDisconnect)
    monkeypatch.setenv("BT_STUDIO_API_TOKEN", "secret")
    with pytest.raises(Exception):
        with tc.websocket_connect("/ws/v2"):
            pass


def test_screener_and_cross_screen_focus_sync(client):
    import polars as pl

    tc, tmp = client
    score_dir = tmp / "artifacts" / "tune" / "scores"
    score_dir.mkdir(parents=True)
    pl.DataFrame({
        "day": ["2024-12-02", "2024-12-02"],
        "sid": [b"600000", b"688001"],
        "distance": [1.1, 1.4],
        "macro_state": [2, 2],
        "expected_return": [0.004, 0.003],
        "fsm_score": [0.92, 0.75],
    }).write_parquet(score_dir / "scores_202412.parquet")
    MA._index.scan_once()

    scr = tc.get("/api/v2/screener").json()
    assert scr["model_id"] == 202412
    assert len(scr["items"]) == 2
    assert scr["items"][0]["sid"] == "600000"

    r_focus = tc.post("/api/v2/focus", json={"sid": "688001", "model_id": 202412, "feature_id": "f_abc"})
    assert r_focus.status_code == 200
    cur_focus = tc.get("/api/v2/focus").json()
    assert cur_focus["sid"] == "688001"
    assert cur_focus["model_id"] == 202412

