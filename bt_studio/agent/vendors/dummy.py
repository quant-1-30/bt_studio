#!/usr/bin/env python3

from .base import BaseLLMProvider


class DummyProvider(BaseLLMProvider):

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> str:
        return """```json
{
    "hypothesis_id": "hyp_reconstructed_ofi",
    "economic_reasoning": "sign(close.diff)*amount 累计占比,截面去均值还原 OFI。",
    "sub_features": [
        [
            {"name": "ofi_dir", "ast": {"op": "sign", "args": [{"op": "delta", "args": [{"col": "close"}, 1]}]}},
            {"name": "ofi_signed_amt", "ast": {"op": "mul", "args": [{"col": "ofi_dir"}, {"col": "amount"}]}},
            {"name": "ofi_cum_sa", "ast": {"op": "cum_sum", "args": [{"col": "ofi_signed_amt"}]}},
            {"name": "ofi_cum_amt", "ast": {"op": "cum_sum", "args": [{"col": "amount"}]}},
            {"name": "final_ofi", "ast": {"op": "cs_demean", "args": [{"op": "div", "args": [{"col": "ofi_cum_sa"}, {"col": "ofi_cum_amt"}]}]}}
        ]
    ]
}
```"""
