#! /usr/bin/env python3

import os

from dotenv import load_dotenv

from bt_studio.constant import (
    FEATURE_DIR, TUNE_MODEL_DIR, TUNE_SCORE_DIR, TUNE_COLLAPSE_DIR,
    LLM_RUN_DIR, LLM_PENDING_DIR, LLM_PROFILE_DIR, DAGS_DIR
)


def io_setup():
    """Load .env and lazily create all output directories (no side-effects)."""
    load_dotenv()
    for d in (FEATURE_DIR, TUNE_MODEL_DIR, TUNE_SCORE_DIR, TUNE_COLLAPSE_DIR,
              LLM_RUN_DIR, LLM_PROFILE_DIR):
        os.makedirs(d, exist_ok=True)


def list_dags() -> list[str]:
    if not os.path.isdir(DAGS_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(DAGS_DIR)
                  if f.endswith(".xml"))
