#!/usr/bin/env python3

from __future__ import annotations

import os
from .base import BaseLLMProvider
from .dummy import DummyProvider
from .openai_compatible import OpenAICompatibleProvider


def get_llm_provider(provider_name: str | None = None, **kwargs) -> BaseLLMProvider:
    name = (provider_name or os.environ.get("LLM_PROVIDER", "dummy")).lower()

    if name == "dummy":
        return DummyProvider()
    
    if name == "zhipu":
        return OpenAICompatibleProvider(
            api_key=kwargs.get("api_key") or os.environ.get("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            model=kwargs.get("model", "glm-5.3"),
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
            model=kwargs.get("model", "gemini-3.5-flash"),
        )

    raise ValueError(f"未知的 LLM 提供商: {name}")
