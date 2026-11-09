#!/usr/bin/env python3

from __future__ import annotations

import os
import time

from typing import Optional
from openai import OpenAI
from .base import BaseLLMProvider


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI 兼容端点通用 Provider (zhipu glm-5.3 / kimi / deepseek)。

    glm-5.3 注意事项:
    - 端点: https://open.bigmodel.cn/api/paas/v4/ (OpenAI 兼容)
    - json_mode 经 response_format json_object 支持; 部分模型对
      response_format + temperature 组合有约束, 失败时自动降级重试一次 (去 response_format)
    - 指数退避网络重试 (429/5xx)
    """

    MAX_RETRIES = 3
    DEFAULT_TIMEOUT_S = 300.0  # glm-5.3 reasoning 长生成; 无超时会挂 ~30min (600s×3 重试)

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "default",
        timeout_s: Optional[float] = None,
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("LLM_API_KEY"),
            base_url=base_url or os.environ.get("LLM_BASE_URL"),
            timeout=timeout_s if timeout_s is not None else float(
                os.environ.get("LLM_TIMEOUT_S", self.DEFAULT_TIMEOUT_S)),
            max_retries=0,  # 重试/退避/降级由 generate 统一管理
        )
        self.model = model

    @staticmethod
    def _status_code(err: Exception) -> Optional[int]:
        code = getattr(err, "status_code", None)
        if code is None:
            resp = getattr(err, "response", None)
            code = getattr(resp, "status_code", None)
        return code

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

        # 参数降级阶梯: json_mode → 纯文本; 再降 → 去 temperature
        # (glm-5.3 等 reasoning 模型对部分参数组合有限制)
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        self.last_usage = None  # 防上次成功 usage 在本次失败后被外层误记
        last_err: Exception | None = None
        retriable_http = {408, 409, 425, 429, 500, 502, 503, 504}
        param_http = {400, 404, 415, 422}

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )
                # [metering 接触点] usage 透出给外层计量 wrap (2 行契约之一)
                u = getattr(response, "usage", None)
                self.last_usage = (
                    {"prompt_tokens": u.prompt_tokens,
                     "completion_tokens": u.completion_tokens,
                     "total_tokens": u.total_tokens}
                    if u else None
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                status = self._status_code(e)

                # 认证/权限/资源不存在: 配置错误, 重试只烧时间与限流额度
                if status in (401, 403):
                    raise

                # 参数不兼容: 只允许确定性降级 (response_format → 去 temperature)
                if status in param_http:
                    if "response_format" in kwargs:
                        kwargs = {}
                        continue
                    if temperature is not None:
                        temperature = None  # type: ignore[assignment]
                        continue
                    raise

                # 其他 4xx 多为请求语义错误, 不做盲重试
                if status is not None and 400 <= status < 500 and status not in retriable_http:
                    raise

                # 网络异常 (status=None) 与 408/409/425/429/5xx: 指数退避
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        raise last_err  # type: ignore[misc]
