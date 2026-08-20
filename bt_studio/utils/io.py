#! /usr/bin/env python3

from __future__ import annotations

import json
import os
import pickle
import tempfile
from typing import Any
import polars as pl


def atomic_save_json(data: Any, target_path: str, indent: int = 2) -> None:
    """os.replace is atomic ops"""
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    # tempfile and source in the same directory
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_dir,
        delete=False,
        suffix=".tmp",
    ) as f:
        temp_path = f.name
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
        f.flush()
        os.fsync(f.fileno())  # force to sync

    os.replace(temp_path, target_path)


def atomic_save_parquet(df: pl.DataFrame, target_path: str) -> None:
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    temp_path = f"{target_path}.{os.getpid()}.tmp"
    df.write_parquet(temp_path)
    os.replace(temp_path, target_path)


def atomic_save_pickle(obj: Any, target_path: str) -> None:
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)

    temp_path = f"{target_path}.{os.getpid()}.tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, target_path)
