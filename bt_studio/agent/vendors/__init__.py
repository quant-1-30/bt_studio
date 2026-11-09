#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import time
from typing import Optional
from .base import (
    BaseLLMProvider,
    BudgetExceeded,
    DEFAULT_GOOGLE_MODEL,
    LlmUsageMeter,
    MeteredProvider,
)
from .dummy import DummyProvider
from .openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


def _build_single_provider(name: str, **kwargs) -> BaseLLMProvider:
    if name == "dummy":
        return DummyProvider()
    if name == "zhipu" or name.startswith("glm"):
        model = kwargs.get("model") or (name if name.startswith("glm-") else "glm-5.3")
        return OpenAICompatibleProvider(
            api_key=kwargs.get("api_key")
            or os.environ.get("ZHIPU_API_KEY")
            or os.environ.get("LLM_API_KEY"),
            base_url=kwargs.get("base_url")
            or os.environ.get("ZHIPU_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://open.bigmodel.cn/api/paas/v4/",
            model=model,
        )
    if name == "kimi":
        return OpenAICompatibleProvider(
            api_key=kwargs.get("api_key") or os.environ.get("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.cn/v1",
            model=kwargs.get("model", "moonshot-v1-8k"),
        )
    if name == "google":
        from .google import GoogleProvider
        return GoogleProvider(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", DEFAULT_GOOGLE_MODEL),
        )
    raise ValueError(f"未知的 LLM 提供商: {name}")


class FailoverLLMProvider(BaseLLMProvider):
    """主备多供应商自动故障转移包装器 (带连续失败熔断冷却期 Circuit Breaker)。

    - 主供应商连续失败达到 failure_threshold (默认 2 次) 后，打开断路器进入
      cooldown_s (默认 300s, 可通过 LLM_FAILOVER_COOLDOWN_S 配置) 冷却期；
    - 冷却期内直接路由至 fallback 备用供应商，避免每次都耗尽主供应商多轮退避重试；
    - 冷却期结束后自动半开探测主供应商，成功则重置失败计数。
    """

    def __init__(
        self,
        primary: BaseLLMProvider,
        fallback: BaseLLMProvider,
        failure_threshold: int = 2,
        cooldown_s: Optional[float] = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_s = (
            float(cooldown_s)
            if cooldown_s is not None
            else float(os.environ.get("LLM_FAILOVER_COOLDOWN_S", "300.0") or 300.0)
        )
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._using_fallback = False
        self.last_usage = None

    @property
    def active_provider(self) -> BaseLLMProvider:
        return self.fallback if self._using_fallback else self.primary

    @property
    def model(self) -> str:
        active = self.active_provider
        return getattr(active, "model", type(active).__name__)

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> str:
        now = time.monotonic()
        if now < self._circuit_open_until:
            self._using_fallback = True
            self.last_usage = None
            out = self.fallback.generate(user_prompt, system_prompt, temperature, json_mode)
            self.last_usage = getattr(self.fallback, "last_usage", None)
            return out

        self.last_usage = None
        try:
            out = self.primary.generate(user_prompt, system_prompt, temperature, json_mode)
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._using_fallback = False
            self.last_usage = getattr(self.primary, "last_usage", None)
            return out
        except Exception as primary_err:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._circuit_open_until = time.monotonic() + self.cooldown_s
                logger.warning(
                    f"[FailoverLLMProvider] 主供应商连续失败 {self._consecutive_failures} 次, "
                    f"开启熔断直连备用供应商 ({self.cooldown_s:.0f}s): {primary_err}"
                )
            else:
                logger.warning(f"[FailoverLLMProvider] 主供应商调用异常, 切换备用供应商: {primary_err}")

            try:
                self._using_fallback = True
                out = self.fallback.generate(user_prompt, system_prompt, temperature, json_mode)
                self.last_usage = getattr(self.fallback, "last_usage", None)
                return out
            except Exception as fb_err:
                self.last_usage = None
                logger.error(f"[FailoverLLMProvider] 备用供应商同样失败: {fb_err}")
                raise primary_err


def get_llm_provider(provider_name: str | None = None, **kwargs) -> BaseLLMProvider:
    name = (provider_name or os.environ.get("LLM_PROVIDER", "dummy")).lower()
    if name == "dummy":
        return DummyProvider()

    inner = _build_single_provider(name, **kwargs)
    fallback_name = os.environ.get("LLM_FALLBACK_PROVIDER", "").strip().lower()
    if fallback_name and fallback_name != name:
        try:
            fb_inner = _build_single_provider(fallback_name)
            inner = FailoverLLMProvider(inner, fb_inner)
        except Exception as e:
            logger.warning(
                f"[vendors] LLM_FALLBACK_PROVIDER={fallback_name!r} 初始化失败, "
                f"降级为单供应商模式: {e}"
            )

    # [metering] 统一包裹: 记账 + 预算熔断; 预算未配置时零行为变化
    return MeteredProvider(inner, LlmUsageMeter())
