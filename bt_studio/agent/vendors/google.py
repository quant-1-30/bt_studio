#!/usr/bin/env python3
from __future__ import annotations

import os

from typing import Optional
from google import genai
from google.genai import types
from .base import BaseLLMProvider, DEFAULT_GOOGLE_MODEL


class GoogleProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None,
                 model: str = DEFAULT_GOOGLE_MODEL):
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model = model

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> str:
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
            response_mime_type="application/json" if json_mode else None,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        # [metering] 与 openai_compatible 同一 last_usage 契约; google-genai 的
        # usage_metadata 字段名不同, 映射到统一 (prompt/completion/total)。
        u = getattr(response, "usage_metadata", None)
        self.last_usage = (
            {
                "prompt_tokens": getattr(u, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(u, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(u, "total_token_count", 0) or 0,
            }
            if u else None
        )
        return response.text or ""
