"""The inventory skew that leans quotes against one-sided inventory (the direct
antidote to the adverse-selection pile-ups seen on live day 1) was inert: the
live maker fed `plan_quote` the paper default `max_inventory=60`, while the live
per-market net cap is 3. So skew = inventory_skew*(net/60) <= 0.1c at the live
cap and rounded to zero on the 1c exchange grid. The live quote params must scale
max_inventory to the actual risk cap so the skew engages.
"""
from __future__ import annotations

import types

from kwt import live_engine as le
from kwt.risk import LiveRiskLimits
from kwt.strategies.market_making import plan_quote


class _NWP:
    def p_bucket(self, low, high, blend_empirical=0.6):
        return 0.50


class _Ctx:
    """Minimal market context: fair lands at 0.50, simple (non-touch) quoting."""
    nwp = _NWP()
    horizon_days = 1.0            # * 24h > min_quote_hours, so we quote
    low, high = 70, 75
    yes_mid = 0.50
    yes_bid = None               # no book -> plan_quote takes the simple fair+/-half path
    yes_ask = None

    def feasible_yes_bounds(self):
        return (0.0, 1.0)


BASE = {"half_spread": 0.03, "inventory_skew": 0.02,
        "fair_market_weight": 0.7, "disagree_widen": 0.5}


def _ask(net, max_inventory):
    return plan_quote(_Ctx(), net, {**BASE, "max_inventory": max_inventory}).new_ask


def test_skew_is_inert_at_paper_scale_but_engages_at_live_cap():
    """Characterizes the bug: at max_inventory=60 a full live inventory barely
    moves the quote; scaled to the live cap of 3 it leans a meaningful ~2c."""
    flat = _ask(0, 60)
    bugged = _ask(3, 60)          # paper-scaled skew: negligible
    fixed = _ask(3, 3)            # live-cap-scaled skew: engages
    assert abs(bugged - flat) < 0.005          # ~0.1c -> effectively off
    assert (flat - fixed) >= 0.015             # leans the ask down ~2c to shed inventory


def test_live_quote_params_scale_max_inventory_to_risk_cap():
    """The wiring fix: the live maker's quote params must set max_inventory to the
    per-market net cap so the skew above actually fires in production."""
    limits = LiveRiskLimits(max_position_per_market=3)
    qp = le._quote_params({"inventory_skew": 0.02, "max_inventory": 60},
                          {"near_decided_band": 0.15}, limits)
    assert qp["max_inventory"] == 3                 # scaled to the real cap
    assert qp["near_decided_band"] == 0.15          # overrides preserved
    assert qp["inventory_skew"] == 0.02             # base params preserved
