#!/usr/bin/env python3
"""bt_studio.monitor M0 索引行为锁定 (tmp result_root, 不碰真实 result)。"""
import json
import os
from datetime import datetime

from bt_studio.plugins.api_server.indexer import MonitorIndex


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def test_monitor_index_m0(tmp_path, monkeypatch):
    root = str(tmp_path)
    today = datetime.now().strftime("%Y%m%d")

    _write_json(os.path.join(root, "artifacts/runs/run1/state.json"), {
        "step_counter": 1, "config_fingerprint": "fp1", "target": "t",
        "replay_buffer": {"successful_history": [], "failed_history": [{"code": "STAGE2_PVAL"}]},
    })
    _write_json(os.path.join(root, "artifacts/runs/run1/step_20261011_1_s1.json"), {
        "step_index": 1, "hypothesis_id": "h1", "features": [{"feature": "f_a"}],
        "replay_buffer": {"successful_history": [], "failed_history": [1, 2, 3]},
    })
    usage_path = os.path.join(root, "artifacts/usage", f"usage_{today}.jsonl")
    os.makedirs(os.path.dirname(usage_path), exist_ok=True)
    with open(usage_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"provider": "OpenAICompatibleProvider", "model": "glm-5.3",
                            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                            "latency_ms": 10.0, "status": "ok"}) + "\n")

    _write_json(os.path.join(root, "features/agent/viable_features.json"), {
        "features": [{"feature_id": "f_abc123", "display_name": "alpha", "fingerprint": "abc123",
                      "final_score": 1.0, "window_pass_ratio": 0.5}]
    })
    feat_parquet = os.path.join(root, "features/agent/f_abc123-abc12345/hf_f_abc123_202401.parquet")
    os.makedirs(os.path.dirname(feat_parquet), exist_ok=True)
    open(feat_parquet, "wb").write(b"")

    # v2 月包布局: cache/bars/ym=202401.parquet + 邻置 meta
    from bt_studio.constant import RAW_PIT_VERSION as _RPV
    RAW_PIT_DIR = os.path.join(root, "cache", "bars", _RPV)
    os.makedirs(RAW_PIT_DIR, exist_ok=True)
    raw = os.path.join(RAW_PIT_DIR, "ym=202401.parquet")
    with open(os.path.join(RAW_PIT_DIR, "ym=202401.meta.json"), "w") as mf:
        import json as _j; _j.dump({"covered_sids": ["600001"]}, mf)
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    open(raw, "wb").write(b"")

    _write_json(os.path.join(root, "artifacts/tune/t1/models/model_202403.manifest.json"), {
        "model_id": 202403, "feature_id": "f_abc123", "source": "tune",
    })
    open(os.path.join(root, "artifacts/tune/t1/models/model_202403.pkl"), "wb").write(b"")
    os.makedirs(os.path.join(root, "tune/t1/scores"), exist_ok=True)
    open(os.path.join(root, "tune/t1/scores/scores_202403.parquet"), "wb").write(b"")

    monkeypatch.setenv("AGENT_STAGE3", "off")
    monkeypatch.setenv("LLM_DAILY_TOKEN_BUDGET", "1000")
    idx = MonitorIndex(result_root=root, db_path=os.path.join(root, "cache", "idx.sqlite"))
    idx.scan_once()

    ov = idx.overview()
    assert ov["usage"]["total_tokens"] == 150
    assert ov["token_budget_used_ratio"] == 0.15
    assert ov["raw_pit"]["status"] == "ok" and ov["raw_pit"]["existing_cells"] == 1

    runs = idx.list_runs(limit=10)
    assert runs["items"][0]["run_id"] == "run1"
    assert runs["items"][0]["n_failure"] == 3 and runs["items"][0]["has_pending"] is False
    detail = idx.get_run("run1")
    assert detail and detail["state"]["config_fingerprint"] == "fp1"
    assert detail["steps"][0]["hypothesis_id"] == "h1"

    usage = idx.usage_daily(today, today)
    assert usage[0]["total_tokens"] == 150 and usage[0]["latency_avg_ms"] == 10.0

    codes = {a["code"] for a in idx.alerts()}
    assert "forbidden_pending_tune" in codes  # AGENT_STAGE3=off 下 tune 产物必须报警
    assert "token_budget_80" not in codes


# ---------------------------------------------------------------------------
# M1 补全: review 闭环 / features 查询 / repeat_fingerprint 告警
# ---------------------------------------------------------------------------

def _mk_run_step(path, run_id, fps, n_succ=0):
    path.write_text(json.dumps({
        "step_index": 1, "hypothesis_id": "h",
        "features": [{"fingerprint": fp} for fp in fps],
        "replay_buffer": {"successful_history": [1] * n_succ, "failed_history": []},
    }))
    assert run_id  # 目录名即 run_id


def test_review_roundtrip_and_features_join(tmp_path, monkeypatch):
    monkeypatch.setattr("bt_studio.plugins.api_server.indexer.RESULT_ROOT", str(tmp_path))
    (tmp_path / "artifacts" / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "artifacts" / "runs" / "r2").mkdir(parents=True)
    _mk_run_step(tmp_path / "artifacts" / "runs" / "r1" / "step_a.json", "r1", ["fpA"], n_succ=1)
    _mk_run_step(tmp_path / "artifacts" / "runs" / "r2" / "step_b.json", "r2", ["fpA"])  # 跨 run 重探

    agent = tmp_path / "artifacts" / "features"
    agent.mkdir(parents=True)
    (agent / "viable_features.json").write_text(json.dumps(
        {"features": [{"feature_name": "featX", "fingerprint": "feed0001",
                       "final_score": 1.2, "window_pass_ratio": 0.67}]}))

    idx = MonitorIndex(result_root=str(tmp_path),
                       db_path=str(tmp_path / "cache" / "m.sqlite"))
    idx.scan_once()

    # repeat_fingerprint 告警 (fpA 出现在 2 个 run)
    codes = [a["code"] for a in idx.alerts()]
    assert "repeat_fingerprint" in codes

    # review 写入 → sqlite + result/llm/review/*.json 双落
    out = idx.post_review("feed0001", "promote_candidate", "strong", "alice")
    assert out["decision"] == "promote_candidate"
    assert (tmp_path / "artifacts" / "review" / "feed0001.json").exists()
    assert idx.review_mark("feed0001")["operator"] == "alice"

    # features 列表 join review
    feats = idx.list_features()["items"]
    assert feats and feats[0]["feature_id"] == "feed0001"
    assert feats[0]["review"]["decision"] == "promote_candidate"

    # 非法 decision 拒绝
    import pytest as _pytest
    with _pytest.raises(ValueError):
        idx.post_review("feed0001", "auto_promote", "", "")

    # 详情: parquet 只 stat 不读 (造空文件即可索引)
    feat_dir = agent / "feed0001"
    feat_dir.mkdir()
    (feat_dir / "hf_featX_202401.parquet").write_bytes(b"")
    idx.scan_once()
    detail = idx.get_feature("feed0001")
    assert any(p["ym"] == 202401 for p in detail["parquets"])


def test_step_counts_from_history_lists(tmp_path, monkeypatch):
    """flywheel artifact 的 replay_buffer 是 history 列表, 计数应取 len 而非不存在的 n_success 键。"""
    monkeypatch.setattr("bt_studio.plugins.api_server.indexer.RESULT_ROOT", str(tmp_path))
    rd = tmp_path / "artifacts" / "runs" / "r3"
    rd.mkdir(parents=True)
    (rd / "step_c.json").write_text(json.dumps({
        "step_index": 2, "features": [],
        "replay_buffer": {"successful_history": [{"a": 1}, {"b": 2}],
                          "failed_history": [{"c": 3}]}}))
    idx = MonitorIndex(result_root=str(tmp_path),
                       db_path=str(tmp_path / "cache" / "m.sqlite"))
    idx.scan_once()
    run = idx.get_run("r3")
    assert run["steps"][0]["n_success"] == 2
    assert run["steps"][0]["n_failure"] == 1


def test_webhook_retry_on_failure_and_lru_bars_eviction(tmp_path, monkeypatch):
    """验证 A1 (webhook 发送失败不置去重签名, 下轮重试) 与 A3 (LRU 驱逐覆盖 evals + bars 月包)。"""
    import urllib.request

    root = str(tmp_path)
    idx = MonitorIndex(result_root=root, db_path=os.path.join(root, "cache", "m.sqlite"))

    # 1. A1: Webhook 第 1 次失败抛异常 -> _last_webhook_sig 未被设置 -> 第 2 次成功发送并设置签名
    monkeypatch.setenv("BT_STUDIO_ALERT_WEBHOOK", "http://127.0.0.1:9999/hook")
    attempts = []

    class _DummyResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=3.0):
        attempts.append(req.full_url)
        if len(attempts) == 1:
            raise OSError("temporary network failure")
        return _DummyResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    sample_alerts = [{"code": "token_budget_80", "level": "warn", "message": "80%"}]

    idx._notify_webhook_if_needed(sample_alerts)
    assert len(attempts) == 1
    assert getattr(idx, "_last_webhook_sig", None) is None, "发送失败时不得提前写入 _last_webhook_sig"

    idx._notify_webhook_if_needed(sample_alerts)
    assert len(attempts) == 2
    assert getattr(idx, "_last_webhook_sig", None) is not None

    # 第 3 次相同告警命中去重签名，不再重复发送
    idx._notify_webhook_if_needed(sample_alerts)
    assert len(attempts) == 2

    # 2. A3: LRU 驱逐同时清理 cache/evals/ 与 cache/bars/ 月包
    eval_file = os.path.join(root, "cache", "evals", "v2", "a.json")
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    open(eval_file, "w").write("{}")

    os.makedirs(idx.raw_pit_dir, exist_ok=True)
    bar_pack = os.path.join(idx.raw_pit_dir, "ym=202201.parquet")
    bar_meta = os.path.join(idx.raw_pit_dir, "ym=202201.meta.json")
    open(bar_pack, "wb").write(b"parquet")
    open(bar_meta, "w").write("{}")

    removed = idx._evict_eval_cache_lru(max_remove=10, include_bars=True)
    assert removed == 2
    assert not os.path.exists(eval_file)
    assert not os.path.exists(bar_pack) and not os.path.exists(bar_meta)


def test_clean_stale_tmp_files_and_multi_kind_watcher_scan(tmp_path):
    """验证孤儿 .tmp 清理与 watcher 同目录单次 walk 多类型分发。"""
    import time
    from bt_studio.utils.io import clean_stale_tmp_files
    from bt_studio.plugins.api_server.watcher import (
        ArtifactKind, _scan_root_multi_kinds_sync,
    )

    # 1. 陈旧 .tmp 清理 (仅删超过 max_age_s 的孤儿 tmp, 不误删新鲜写入中的 tmp)
    stale_tmp = tmp_path / "a.parquet.123.tmp"
    fresh_tmp = tmp_path / "b.parquet.456.tmp"
    stale_tmp.write_bytes(b"stale")
    fresh_tmp.write_bytes(b"fresh")
    old_ts = time.time() - 7200.0
    os.utime(stale_tmp, (old_ts, old_ts))

    removed = clean_stale_tmp_files(str(tmp_path), max_age_s=3600.0)
    assert removed == 1
    assert not stale_tmp.exists() and fresh_tmp.exists()

    # 2. Watcher 单次 walk 同时分发 MODEL / SCORE / COLLAPSE
    tune_root = tmp_path / "tune" / "task1"
    (tune_root / "models").mkdir(parents=True)
    (tune_root / "scores").mkdir(parents=True)
    (tune_root / "collapse").mkdir(parents=True)
    (tune_root / "models" / "model_202401.pkl").write_bytes(b"pkl")
    (tune_root / "scores" / "scores_202401.parquet").write_bytes(b"pq")
    (tune_root / "collapse" / "collapse_202401_sig.json").write_text('{"verdict": "healthy_plateau"}')

    kinds = [ArtifactKind.MODEL, ArtifactKind.SCORE, ArtifactKind.COLLAPSE]
    res = _scan_root_multi_kinds_sync(str(tmp_path / "tune"), kinds, {})
    assert len(res[ArtifactKind.MODEL][0]) == 1
    assert len(res[ArtifactKind.SCORE][0]) == 1
    assert len(res[ArtifactKind.COLLAPSE][0]) == 1
