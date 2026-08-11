"""Feature specification and registry for the Feature Agent pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List
import polars as pl


@dataclass
class FeatureSpec:
    name: str
    build_fn: Callable
    output_col: str
    description: str = ""
    causal: bool = True


class FeatureRegistry:
    def __init__(self):
        self._specs: Dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec):
        if spec.name in self._specs:
            raise ValueError(f"Feature {spec.name!r} already registered")
        if not spec.causal:
            print(f"  WARNING [Registry] Feature {spec.name!r} declares causal=False")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def list_features(self) -> List[FeatureSpec]:
        return list(self._specs.values())

    def names(self) -> List[str]:
        return list(self._specs.keys())

    def __len__(self):
        return len(self._specs)

    def __contains__(self, name):
        return name in self._specs


def default_registry() -> FeatureRegistry:
    from .ofi import build_ofi
    from .vol import build_vol
    r = FeatureRegistry()
    r.register(FeatureSpec("ofi", build_ofi, "ofi_ratio",
        "Order Flow Imbalance: weighted SA/Impact/Liquidity ratio.", True))
    r.register(FeatureSpec("vol", build_vol, "vol_ratio",
        "Intraday volatility regime: cumulative high-low range.", True))
    return r
