import os
from google import genai

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

tools = [
    {
        'type': 'google_search',
    },
]

generation_config = {
    'max_output_tokens': 65536,
    'top_p': 0.95,
    'thinking_level': 'medium',
}

interaction = client.interactions.create(
    model='models/gemini-3.7-flash',
    input='',
    tools=tools,
    generation_config=generation_config,
)

print(interaction.steps[-1])






from __future__ import annotations

import os
from typing import Optional
from google import genai
from google.genai import types
from .base import BaseLLMProvider


class GoogleProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.7-flash"):
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
        return response.text or ""

