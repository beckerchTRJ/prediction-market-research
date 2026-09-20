"""Long-horizon ensemble divergence (horizon A/B against the same-day book).

Identical in model to `ensemble_divergence` -- same fat-tailed multi-model
bucket distribution, market shrinkage, fractional Kelly and risk caps -- but it
only trades a market while that market is still `min_horizon_days` (default 1.0)
or more from close, skipping the same-day session entirely.

The motivation is to test a recurring claim about weather markets: informed
participants with private nowcasting models hold their largest edge *near close*,
so trading the day before sidesteps that flow. Kalshi only lists these
daily-high markets ~1.5 days out at most, so this variant trades a narrow
1.0-1.7d window and will accumulate resolved markets (and therefore reach an edge
verdict) more slowly than the same-day strategies. That low volume is inherent
to the contract listing schedule, not a defect -- it runs on its own bankroll so
its edge/Brier verdict stays directly comparable to `ensemble_divergence`.
"""
from __future__ import annotations

from .ensemble_divergence import EnsembleDivergenceStrategy


class EnsembleDivergenceLongHorizonStrategy(EnsembleDivergenceStrategy):
    name = "ensemble_divergence_lh"

    def generate(self, ctxs, book):
        min_h = self.params.get("min_horizon_days", 1.0)
        far = [c for c in ctxs if c.horizon_days >= min_h]
        return super().generate(far, book)
