#!/usr/bin/env python3
"""Abstract LLM provider base class, daily token usage ledger, and budget circuit-breaker decorator."""

from __future__ import annotations

import contextlib
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from bt_studio.constant import LLM_USAGE_DIR as _DEFAULT_LLM_USAGE_DIR
from bt_studio.utils.io import exclusive_file_lock as _exclusive_file_lock

DEFAULT_GOOGLE_MODEL = "gemini-3.8-flash"


class BaseLLMProvider(ABC):
    """Abstract base interface for LLM generation providers."""

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        json_mode: bool = True,
    ) -> str:
        raise NotImplementedError


class BudgetExceeded(Exception):
    """Raised when the daily LLM token budget is exhausted."""


class LlmUsageMeter:
    """Append-only daily JSONL token usage ledger with incremental offset restore and budget enforcement."""

    def __init__(self, budget_total: Optional[int] = None, ledger_dir: Optional[str] = None):
        self.budget_total = budget_total
        self.ledger_dir = ledger_dir or os.environ.get("LLM_USAGE_DIR", _DEFAULT_LLM_USAGE_DIR)
        os.makedirs(self.ledger_dir, exist_ok=True)
        self.today = ""
        self._path = ""
        self._read_offset = 0
        self.totals: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        self._rollover_if_needed()

    def _rollover_if_needed(self) -> None:
        day = datetime.now().strftime("%Y%m%d")
        if day != self.today:
            self.today = day
            self._path = os.path.join(self.ledger_dir, f"usage_{day}.jsonl")
            self._read_offset = 0
            self.totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        self._restore_from_ledger()

    def _restore_from_ledger(self) -> None:
        if not os.path.exists(self._path):
            return
        size = os.path.getsize(self._path)
        if self._read_offset > size:
            self._read_offset = 0
            self.totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        if self._read_offset == size:
            return
        with open(self._path, "rb") as f:
            f.seek(self._read_offset)
            chunk = f.read()
        last_nl = chunk.rfind(b"\n")
        if last_nl == -1:
            return
        complete = chunk[: last_nl + 1]
        self._read_offset += len(complete)
        for line in complete.decode("utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = rec.get("usage") or {}
            self.totals["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            self.totals["completion_tokens"] += int(u.get("completion_tokens") or 0)
            self.totals["total_tokens"] += int(u.get("total_tokens") or 0)
            self.totals["calls"] += 1

    @property
    def _budget(self) -> Optional[int]:
        if self.budget_total is not None:
            return int(self.budget_total)
        env_v = os.environ.get("LLM_DAILY_TOKEN_BUDGET")
        return int(env_v) if env_v and env_v.isdigit() else None

    @contextlib.contextmanager
    def exclusive_lease(self):
        with _exclusive_file_lock(f"{self._path}.lock"):
            self._rollover_if_needed()
            yield

    def precheck(self) -> None:
        self._rollover_if_needed()
        budget = self._budget
        if budget is not None and self.totals["total_tokens"] >= budget:
            raise BudgetExceeded(
                f"daily LLM budget exhausted: {self.totals['total_tokens']}/{budget} tokens "
                f"({self.totals['calls']} calls, ledger: {self._path})"
            )

    def record(
        self,
        provider: str,
        model: str,
        usage: Optional[Dict[str, Any]],
        latency_ms: float,
        status: str,
        error_type: str = "",
    ) -> None:
        self._rollover_if_needed()
        rec = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "provider": provider,
            "model": model,
            "usage": usage,
            "latency_ms": round(latency_ms, 1),
            "status": status,
            **({"error_type": error_type} if error_type else {}),
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._restore_from_ledger()


class MeteredProvider(BaseLLMProvider):
    """Decorator wrapping a BaseLLMProvider with token budget prechecks and JSONL usage logging."""

    def __init__(self, inner: BaseLLMProvider, meter: LlmUsageMeter):
        self.inner = inner
        self.meter = meter

    @property
    def last_usage(self) -> Optional[Dict[str, Any]]:
        return getattr(self.inner, "last_usage", None)

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", type(self.inner).__name__)

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        json_mode: bool = True,
    ) -> str:
        pname = type(self.inner).__name__
        t0 = time.perf_counter()
        if self.meter._budget is not None:
            with self.meter.exclusive_lease():
                self.meter.precheck()
                return self._call_and_record(pname, t0, user_prompt, system_prompt, temperature, json_mode)
        self.meter.precheck()
        return self._call_and_record(pname, t0, user_prompt, system_prompt, temperature, json_mode, lock_record=True)

    def _call_and_record(
        self,
        pname: str,
        t0: float,
        user_prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        json_mode: bool,
        lock_record: bool = False,
    ) -> str:
        try:
            out = self.inner.generate(user_prompt, system_prompt, temperature, json_mode)
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            if lock_record:
                with self.meter.exclusive_lease():
                    self.meter.record(pname, self.model, None, elapsed, status="error", error_type=type(e).__name__)
            else:
                self.meter.record(pname, self.model, None, elapsed, status="error", error_type=type(e).__name__)
            raise
        elapsed = (time.perf_counter() - t0) * 1000
        if lock_record:
            with self.meter.exclusive_lease():
                self.meter.record(pname, self.model, self.last_usage, elapsed, status="ok")
        else:
            self.meter.record(pname, self.model, self.last_usage, elapsed, status="ok")
        return out
