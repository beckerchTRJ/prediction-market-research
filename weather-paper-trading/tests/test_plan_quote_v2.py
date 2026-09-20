"""W3 quoting-logic upgrades: all param-gated, default = current behavior.

Covers: nowcast fair floor (#3), FLB haircut (#1), per-side touch edges (#2),
bucket-side selection / one-sided quotes (#1,#8), and the simulate_fills
one-sided guard. The first test class is the load-bearing regression guard:
with every new param absent, plan_quote + simulate_fills must be unchanged.
"""
from __future__ import annotations

import pytest

from kwt.strategies.market_making import (
    QuotePlan,
    _flb_haircut,
    plan_quote,
    simulate_fills,
)


class FakeNWP:
    def __init__(self, p):
        self.p = p

    def p_bucket(self, low, high, blend_empirical=0.6):
        return self.p


class Ctx:
    def __init__(self, p=0.5, horizon_days=2.0, yes_bid=0.45, yes_ask=0.55,
                 bounds=(0.0, 1.0), nwp=True, obs_so_far=None):
        self.nwp = FakeNWP(p) if nwp else None
        self.horizon_days = horizon_days
        self.low, self.high = 70, 71
        self.yes_bid, self.yes_ask = yes_bid, yes_ask
        self._bounds = bounds
        self.obs_so_far = obs_so_far

    @property
    def yes_mid(self):
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return None

    def feasible_yes_bounds(self):
        return self._bounds


PARAMS = {"half_spread": 0.03, "fair_market_weight": 0.7, "min_quote_hours": 3.0,
          "max_inventory": 60, "quote_size": 5, "participation_rate": 0.05,
          "extreme_participation": 0.02, "max_adverse_through": 0.05,
          "max_fills_per_run": 10, "min_edge_vs_market": 0.0, "disagree_widen": 0.5,
          "inventory_skew": 0.02}


def _trade(yp, cnt, ct="t"):
    return {"yes_price_dollars": yp, "count": cnt, "created_time": ct}


# --------------------------------------------------------------------------
# Regression: defaults reproduce the current behavior exactly (byte-identical).
# --------------------------------------------------------------------------
class TestRegressionDefaults:
    def test_two_sided_basic(self):
        p = plan_quote(Ctx(p=0.5, yes_bid=0.45, yes_ask=0.55), 0, PARAMS)
        assert (p.new_bid, p.new_ask) == (0.47, 0.53)
        assert p.fair == 0.5 and p.disagree == 0.0
        assert p.quotable is True and p.skip_reason is None
        assert p.model_fair == 0.5

    def test_inventory_skew(self):
        # skew = -0.02 * (30/60) = -0.01
        p = plan_quote(Ctx(p=0.5, yes_bid=0.45, yes_ask=0.55), 30, PARAMS)
        assert p.new_bid == pytest.approx(0.46) and p.new_ask == pytest.approx(0.52)
        assert p.quotable is True

    def test_model_disagreement_widens(self):
        # model 0.7 vs mid 0.5 -> fair 0.56, disagree 0.2, eff_half 0.13
        p = plan_quote(Ctx(p=0.7, yes_bid=0.45, yes_ask=0.55), 0, PARAMS)
        assert p.fair == pytest.approx(0.56)
        assert p.disagree == pytest.approx(0.2)
        assert p.new_bid == pytest.approx(0.43) and p.new_ask == pytest.approx(0.69)

    def test_quote_at_touch_wide_book(self):
        p = plan_quote(Ctx(p=0.5, yes_bid=0.40, yes_ask=0.60), 0,
                       dict(PARAMS, quote_at_touch=True))
        assert (p.new_bid, p.new_ask) == (0.40, 0.60)
        assert p.quotable is True

    def test_nowcast_floor_off_by_default_noop(self):
        base = plan_quote(Ctx(p=0.5, obs_so_far=70.0), 0, PARAMS)
        # enabling with obs None short-circuits -> still unchanged
        same = plan_quote(Ctx(p=0.5, obs_so_far=None), 0,
                          dict(PARAMS, nowcast_floor_enabled=True))
        assert (base.new_bid, base.new_ask) == (same.new_bid, same.new_ask)
        assert base.fair == same.fair

    def test_simulate_fills_two_sided_unchanged(self):
        trades = [_trade(0.45, 100, "t1"), _trade(0.55, 100, "t2")]
        sim = simulate_fills(0.45, 0.55, trades, 0, PARAMS, "")
        assert sim.buy_yes == 5 and sim.buy_no == 5
        assert sim.fills == 2 and sim.newest == "t2"

    def test_simulate_fills_first_cycle_unchanged(self):
        sim = simulate_fills(None, None, [_trade(0.45, 100, "t1")], 0, PARAMS, "")
        assert sim.buy_yes == 0 and sim.buy_no == 0 and sim.fills == 0
        assert sim.newest == "t1"

    def test_simulate_fills_adverse_unchanged(self):
        trades = [_trade(0.30, 100, "t1")]  # sweeps below bid-through
        sim = simulate_fills(0.45, 0.55, trades, 0, PARAMS, "")
        assert sim.buy_yes == 0 and sim.adverse_skipped == 100


# --------------------------------------------------------------------------
# #1 FLB haircut helper + application
# --------------------------------------------------------------------------
class TestFlbHaircut:
    def test_value_at_10c(self):
        p = {"flb_haircut_at_10c": 0.03, "flb_haircut_zero_at": 0.40}
        assert _flb_haircut(0.10, p) == pytest.approx(0.03)

    def test_zero_at_and_above(self):
        p = {"flb_haircut_at_10c": 0.03, "flb_haircut_zero_at": 0.40}
        assert _flb_haircut(0.40, p) == 0.0
        assert _flb_haircut(0.55, p) == 0.0

    def test_linear_between(self):
        p = {"flb_haircut_at_10c": 0.03, "flb_haircut_zero_at": 0.40}
        # halfway (price 0.25) -> half of 0.03
        assert _flb_haircut(0.25, p) == pytest.approx(0.015)

    def test_held_flat_below_10c(self):
        p = {"flb_haircut_at_10c": 0.03, "flb_haircut_zero_at": 0.40}
        assert _flb_haircut(0.05, p) == pytest.approx(0.03)
        assert _flb_haircut(0.01, p) == pytest.approx(0.03)

    def test_never_negative_and_default_off(self):
        assert _flb_haircut(0.10, {}) == 0.0            # default at10=0 -> no-op
        assert _flb_haircut(0.10, {"flb_haircut_at_10c": 0.0}) == 0.0
        assert _flb_haircut(0.9, {"flb_haircut_at_10c": 0.03}) == 0.0

    def test_applied_lowers_fair(self):
        base = plan_quote(Ctx(p=0.10, yes_bid=0.08, yes_ask=0.12), 0, PARAMS)
        hc = plan_quote(Ctx(p=0.10, yes_bid=0.08, yes_ask=0.12), 0,
                        dict(PARAMS, flb_haircut_at_10c=0.03, flb_haircut_zero_at=0.40))
        assert hc.fair < base.fair
        assert hc.fair == pytest.approx(0.07)

    def test_applied_never_below_p_lo(self):
        # huge haircut clips at p_lo
        hc = plan_quote(Ctx(p=0.10, yes_bid=0.08, yes_ask=0.12, bounds=(0.09, 1.0)),
                        0, dict(PARAMS, flb_haircut_at_10c=0.5, flb_haircut_zero_at=0.40))
        assert hc.fair == pytest.approx(0.09)


# --------------------------------------------------------------------------
# #2 per-side touch edges
# --------------------------------------------------------------------------
class TestPerSideEdges:
    def test_asymmetric_edges(self):
        # book near mid so fair±edge binds; fair=0.5
        p = plan_quote(Ctx(p=0.5, yes_bid=0.49, yes_ask=0.51), 0,
                       dict(PARAMS, quote_at_touch=True,
                            touch_min_edge_bid=0.04, touch_min_edge_ask=0.01))
        assert p.new_bid == pytest.approx(0.46)   # 4c below fair
        assert p.new_ask == pytest.approx(0.51)   # 1c above fair

    def test_none_falls_back_to_touch_min_edge(self):
        ctx_args = dict(p=0.5, yes_bid=0.49, yes_ask=0.51)
        fallback = plan_quote(Ctx(**ctx_args), 0,
                              dict(PARAMS, quote_at_touch=True, touch_min_edge=0.01))
        explicit = plan_quote(Ctx(**ctx_args), 0,
                              dict(PARAMS, quote_at_touch=True, touch_min_edge=0.01,
                                   touch_min_edge_bid=None, touch_min_edge_ask=None))
        assert (fallback.new_bid, fallback.new_ask) == (explicit.new_bid, explicit.new_ask)


# --------------------------------------------------------------------------
# #1,#8 bucket-side selection (one-sided quotes)
# --------------------------------------------------------------------------
class TestBucketSideSelection:
    def test_ask_only_below(self):
        p = plan_quote(Ctx(p=0.2, yes_bid=0.18, yes_ask=0.22), 0,
                       dict(PARAMS, ask_only_below=0.25))
        assert p.new_bid is None
        assert p.new_ask is not None
        assert p.quotable is True and p.skip_reason is None

    def test_bid_only_above(self):
        p = plan_quote(Ctx(p=0.8, yes_bid=0.78, yes_ask=0.82), 0,
                       dict(PARAMS, bid_only_above=0.75))
        assert p.new_ask is None
        assert p.new_bid is not None
        assert p.quotable is True and p.skip_reason is None

    def test_no_trigger_when_mid_outside(self):
        p = plan_quote(Ctx(p=0.5, yes_bid=0.45, yes_ask=0.55), 0,
                       dict(PARAMS, ask_only_below=0.25, bid_only_above=0.75))
        assert p.new_bid is not None and p.new_ask is not None
        assert p.quotable is True

    def test_both_none_guard(self):
        p = plan_quote(Ctx(p=0.5, yes_bid=0.45, yes_ask=0.55), 0,
                       dict(PARAMS, ask_only_below=0.9, bid_only_above=0.1))
        assert p.new_bid is None and p.new_ask is None
        assert p.quotable is False and p.skip_reason == "one_sided_empty"


# --------------------------------------------------------------------------
# simulate_fills one-sided resting quotes
# --------------------------------------------------------------------------
class TestSimulateFillsOneSided:
    def test_ask_only_fills_asks_not_bids(self):
        trades = [_trade(0.56, 100, "t1"), _trade(0.45, 100, "t2")]
        sim = simulate_fills(None, 0.55, trades, 0, PARAMS, "")
        assert sim.buy_no == 5      # ask filled
        assert sim.buy_yes == 0     # no bid resting
        assert sim.newest == "t2"

    def test_bid_only_fills_bids_not_asks(self):
        trades = [_trade(0.44, 100, "t1"), _trade(0.56, 100, "t2")]
        sim = simulate_fills(0.45, None, trades, 0, PARAMS, "")
        assert sim.buy_yes == 5     # bid filled
        assert sim.buy_no == 0      # no ask resting
        assert sim.newest == "t2"

    def test_both_none_still_no_fills(self):
        sim = simulate_fills(None, None, [_trade(0.45, 100, "t1")], 0, PARAMS, "")
        assert sim.buy_yes == 0 and sim.buy_no == 0
