#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Optional
from openai import OpenAI
from .base import BaseLLMProvider


class OpenAICompatibleProvider(BaseLLMProvider):
    # kimi / deepseek / zai 

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "default",
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("LLM_API_KEY"),
            base_url=base_url or os.environ.get("LLM_BASE_URL"),
        )
        self.model = model

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""
