#!/usr/bin/env python3

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """LLMs abstract api"""

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        json_mode: bool = True,
    ) -> str:
        """JSON / TEXT"""
        raise NotImplementedError
