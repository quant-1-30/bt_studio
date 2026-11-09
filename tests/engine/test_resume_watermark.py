#!/usr/bin/env python3
"""断点恢复水位回归: model 存在但 score 缺失的窗口不得被跳过。"""

import os

from bt_studio.engine import executor as tasking_mod


def test_window_artifacts_complete_requires_model_and_score(tmp_path):
    model_dir = tmp_path / "models"
    score_dir = tmp_path / "scores"
    model_dir.mkdir()
    score_dir.mkdir()
    common = {
        "tune_model_dir": os.fspath(model_dir),
        "tune_score_dir": os.fspath(score_dir),
    }

    # 旧行为/外部产物: 连 model 都没有时不强行阻断 (保持 resume 兼容)
    assert tasking_mod.window_artifacts_complete(202401, common) is True

    model_path = model_dir / "model_202401.pkl"
    model_path.write_bytes(b"stub-model")
    # tune 后 OOS 前崩溃: model 在、score 缺 -> 必须重放该窗口
    assert tasking_mod.window_artifacts_complete(202401, common) is False

    score_path = score_dir / "scores_202401.parquet"
    score_path.write_bytes(b"stub-score")
    assert tasking_mod.window_artifacts_complete(202401, common) is True
