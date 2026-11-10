#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
from typing import Any, Dict, Optional

from bt_studio.compiler.ast import ast_fingerprint
from bt_studio.constant import BT_STUDIO_VERSION, EVAL_LOGIC_VERSION, RAW_PIT_VERSION
from bt_studio.utils.io import atomic_save_json, read_json_safe
from bt_studio.utils.math_func import entry_bar_idx


def _digest_file(path: str, secret_key: str = "") -> str:
    if not path or not os.path.exists(path):
        return ""
    hasher = (
        hmac.new(secret_key.encode("utf-8"), digestmod=hashlib.sha256)
        if secret_key
        else hashlib.sha256()
    )
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_sha256(path: str) -> str:
    return _digest_file(path)


def _file_hmac_sha256(path: str, secret_key: str) -> str:
    return _digest_file(path, secret_key) if secret_key else ""


def safe_load_model_ckpt(model_path: str) -> Dict[str, Any]:
    """checkpoint:
    1. BT_STUDIO_MODEL_HMAC_KEY ---> manifest pkl_hmac_sha256 
    2. pkl_sha256 """
    manifest_path = (
        model_path[:-4] + ".manifest.json"
        if model_path.endswith(".pkl")
        else model_path + ".manifest.json"
    )
    hmac_key = os.environ.get("BT_STUDIO_MODEL_HMAC_KEY", "").strip()
    if os.path.exists(manifest_path):
        manifest = read_json_safe(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"模型 manifest 损坏 ({manifest_path})")

        if hmac_key:
            expected_hmac = str(manifest.get("pkl_hmac_sha256", ""))
            actual_hmac = _file_hmac_sha256(model_path, hmac_key)
            if not expected_hmac or not hmac.compare_digest(actual_hmac, expected_hmac):
                raise ValueError(
                    f"模型文件 HMAC-SHA256 密钥签名验证失败 (拒绝反序列化): {model_path}"
                )

        expected_sha = str(manifest.get("pkl_sha256", ""))
        if expected_sha:
            actual_sha = _file_sha256(model_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"模型文件 SHA-256 签名不匹配 (拒绝反序列化): {model_path} "
                    f"(expected={expected_sha[:12]}..., actual={actual_sha[:12]}...)"
                )

    with open(model_path, "rb") as f:
        return pickle.load(f)


def write_model_manifest(
    model_dir: str,
    model_ckpt: Dict[str, Any],
    common_config: dict,
    source: str = "tune",
    gates: Optional[Dict[str, Any]] = None,
) -> None:
    model_id = int(model_ckpt.get("train_end_month"))
    pkl_path = os.path.join(model_dir, f"model_{model_id}.pkl")
    pkl_sha256 = _file_sha256(pkl_path)
    hmac_key = os.environ.get("BT_STUDIO_MODEL_HMAC_KEY", "").strip()
    pkl_hmac = _file_hmac_sha256(pkl_path, hmac_key) if hmac_key else ""
    ast_recipe = model_ckpt.get("ast_recipe") or common_config.get("ast_recipe")
    try:
        ast_fp = ast_fingerprint(ast_recipe) if ast_recipe is not None else ""
    except Exception:
        ast_fp = ""
    feature_id = str(common_config.get("feature_col") or model_ckpt.get("feature_col") or "")
    entry_idx = entry_bar_idx(common_config)

    manifest = {
        "model_id": model_id,
        "source": source,
        "pkl_sha256": pkl_sha256,
        "pkl_hmac_sha256": pkl_hmac,
        "feature_id": feature_id,
        "feature_col": feature_id,  
        "ast_fingerprint": ast_fp or (feature_id[2:] if feature_id.startswith("f_") else ""),
        "data_fp": common_config.get("_eval_data_fp", ""),
        "task_id": common_config.get("_task_id", ""),
        "train_end_month": model_id,
        "config": model_ckpt.get("config", {}),
        "T1_rets": model_ckpt.get("T1_rets", common_config.get("T1_rets")),
        "decay_minutes": model_ckpt.get("decay_minutes", common_config.get("decay_minutes")),
        "label_basis": {
            "signal_cut_bar": entry_idx - 1,
            "entry_price": f"minute_bar_{entry_idx}_close_end_labeled",
            "t1_open_price": "minute_bar_0_open_end_labeled",
            "price_adjustment": "causal_hfq",
            "execution_feasibility": "bt_core",
            "limit_rules": "bt_core",
            "auction": "bt_core",
            "costs": "bt_core",
        },
        "code": {
            "bt_studio_version": BT_STUDIO_VERSION,
            "raw_pit_version": RAW_PIT_VERSION,
            "eval_logic_version": EVAL_LOGIC_VERSION,
        },
        "gates": gates or {},
    }
    atomic_save_json(manifest, os.path.join(model_dir, f"model_{model_id}.manifest.json"))
