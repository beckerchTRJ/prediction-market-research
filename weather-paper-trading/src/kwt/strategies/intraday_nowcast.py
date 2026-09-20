"""Intraday nowcast — exploit the within-day collapse of high-temperature risk.

Temperatures peak in mid-afternoon. Once the peak has passed, the day's high is
nearly determined: it cannot fall below the max already observed, and the
remaining upside is a few hours of cooling. A market that still prices the high
diffusely late in the day is slow. This strategy builds a tightened predictive
distribution from:
    nowcast_member = max(observed_so_far, ensemble_member_daily_max)
which floors every member at what has already happened, then floors the whole
distribution's location at max(observed_so_far, remaining_forecast_max). As the
day advances and observed_so_far rises past the members, the distribution
collapses onto the realized high and the strategy takes large, high-confidence
positions in the correct bucket while the market lags.

Only active for same-day markets with enough of the day elapsed (configurable),
since before peak heating the floor carries little information.
"""
from __future__ import annotations

from ..distributions import floored_nowcast
from .base import Order, Signal, Strategy
from .climatology import _maybe_trade


class IntradayNowcastStrategy(Strategy):
    name = "intraday_nowcast"

    def generate(self, ctxs, book):
        orders: list[Order] = []
        signals: list[Signal] = []
        edge_thr = self.params.get("edge_threshold", 0.06)
        max_c = self.params.get("max_contracts", 50)
        frac = self.params.get("kelly_fraction", 0.35)
        # This strategy's own aggression gate: only act once enough of the local
        # day has elapsed for the observed-so-far floor to be informative. (The
        # context now attaches obs_so_far all day — the shared conditioning is
        # self-gating — so the hours gate lives here, not in collect.py.)
        min_hours = self.params.get("min_hours_elapsed", 11)
        max_horizon = self.params.get("max_horizon_days", 1.0)
        floor_buffer = self.params.get("floor_buffer", 0.0)    # °F slack on the floor

        for c in ctxs:
            if (c.nwp is None or c.obs_so_far is None or c.horizon_days > max_horizon
                    or (c.hours_elapsed or 0) < min_hours):
                continue
            # The realized high cannot be below what's already been observed today.
            # Floor every ensemble member at observed-so-far (a CERTAIN bound); the
            # members already carry the remaining-hours uncertainty, so we do NOT
            # also hard-floor at remaining_max (a mere deterministic forecast). Note
            # c.nwp is already intraday-conditioned upstream; the extra floor below
            # only applies the optional floor_buffer slack and is otherwise a no-op.
            try:
                nowcast = floored_nowcast(c.nwp, c.obs_so_far, floor_buffer=floor_buffer)
            except ValueError:
                continue
            p_yes = nowcast.p_bucket(c.low, c.high, blend_empirical=0.85)
            new_orders, sig = _maybe_trade(self, c, p_yes, edge_thr, max_c, frac, book,
                                           "nowcast")
            sig.meta["obs_so_far"] = round(c.obs_so_far, 1)
            sig.meta["nowcast_mean"] = round(nowcast.mean, 2)
            sig.meta["nowcast_std"] = round(nowcast.std, 2)
            orders.extend(new_orders)
            signals.append(sig)
        return orders, signals
