"""The extracted MM planner/simulator behave as the live + paper paths expect."""
from __future__ import annotations

from kwt.strategies.market_making import FillSim, plan_quote, simulate_fills


class FakeNWP:
    def __init__(self, p):
        self.p = p

    def p_bucket(self, low, high, blend_empirical=0.6):
        return self.p


class Ctx:
    def __init__(self, p=0.5, horizon_days=2.0, yes_bid=0.45, yes_ask=0.55,
                 bounds=(0.0, 1.0), nwp=True):
        self.nwp = FakeNWP(p) if nwp else None
        self.horizon_days = horizon_days
        self.low, self.high = 70, 71
        self.yes_bid, self.yes_ask = yes_bid, yes_ask
        self._bounds = bounds

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


def test_plan_quote_skip_no_forecast():
    assert plan_quote(Ctx(nwp=False), 0, PARAMS).skip_reason == "no_forecast"


def test_plan_quote_skip_too_close():
    p = plan_quote(Ctx(horizon_days=2.0 / 24.0), 0, PARAMS)
    assert p.skip_reason == "too_close_to_settle"


def test_plan_quote_skip_decided_by_obs():
    p = plan_quote(Ctx(bounds=(1.0, 1.0)), 0, PARAMS)
    assert p.skip_reason == "decided_by_obs" and p.fair == 1.0


def test_plan_quote_quotable_brackets_fair():
    p = plan_quote(Ctx(p=0.5, yes_bid=0.45, yes_ask=0.55), 0, PARAMS)
    assert p.quotable
    assert p.new_bid < p.fair < p.new_ask


def test_plan_quote_no_edge_still_caches():
    # fair == mid (model agrees with market) but require a 0.10 edge -> skip,
    # yet bid/ask are still populated for caching.
    params = dict(PARAMS, min_edge_vs_market=0.10)
    p = plan_quote(Ctx(p=0.5, yes_bid=0.45, yes_ask=0.55), 0, params)
    assert p.skip_reason == "no_edge_vs_mkt"
    assert p.new_bid is not None and p.new_ask is not None


def test_plan_quote_near_decided_gate_is_opt_in():
    # A same-day bucket the observation has NEARLY decided (feasible band 0.1).
    near = Ctx(p=0.5, bounds=(0.0, 0.1))
    # Off by default (paper): still quotes.
    assert plan_quote(near, 0, PARAMS).quotable
    # Live pilot enables it -> skip as 'near_decided'.
    p = plan_quote(near, 0, dict(PARAMS, near_decided_band=0.15))
    assert p.skip_reason == "near_decided" and not p.quotable
    # A wide, undecided market is unaffected even with the gate on.
    assert plan_quote(Ctx(p=0.5, bounds=(0.0, 1.0)), 0,
                      dict(PARAMS, near_decided_band=0.15)).quotable


def test_min_market_spread_skips_tight_books():
    # 1c-wide book, require 2c -> don't make a market here.
    tight = Ctx(p=0.45, yes_bid=0.45, yes_ask=0.46)
    p = plan_quote(tight, 0, dict(PARAMS, min_market_spread=0.02))
    assert p.skip_reason == "market_too_tight" and not p.quotable
    # Off by default (paper) -> still quotes the tight book.
    assert plan_quote(tight, 0, PARAMS).quotable


def test_quote_at_touch_joins_wide_book():
    # 7c book 0.40/0.47, fair ~0.436 sits inside -> quotes JOIN the touch (capture
    # the market's spread) instead of a fixed 3c offset off fair.
    wide = Ctx(p=0.44, yes_bid=0.40, yes_ask=0.47)
    p = plan_quote(wide, 0, dict(PARAMS, quote_at_touch=True))
    assert p.quotable
    assert p.new_bid == 0.40 and p.new_ask == 0.47
    # The old fixed-offset path would have quoted ~fair +/- 3c, well inside 0.40/0.47.
    off = plan_quote(wide, 0, PARAMS)
    assert off.new_bid > 0.40 and off.new_ask < 0.47


def test_quote_at_touch_pulls_back_when_fair_outside_touch():
    # Model says YES ~0.20 while the market bids 0.40 -> our bid pulls BELOW the
    # touch (we won't pay the market's price for something we think is worth less).
    ctx = Ctx(p=0.20, yes_bid=0.40, yes_ask=0.47)
    p = plan_quote(ctx, 0, dict(PARAMS, quote_at_touch=True))
    assert p.new_bid < 0.40           # behind the touch, defensive
    assert p.new_ask == 0.47          # ask still joins (market ask >= fair+edge)


def _trade(price, count, t):
    return {"yes_price_dollars": price, "count": count, "created_time": t}


def test_simulate_fills_bid_and_ask():
    # bid 0.45 / ask 0.55. A print at 0.45 lifts our bid (buy yes); a print at
    # 0.55 hits our ask (buy no). Participation 5% of 200 = 10 cap; size 5 each.
    trades = [_trade(0.45, 100, "t1"), _trade(0.55, 100, "t2")]
    sim = simulate_fills(0.45, 0.55, trades, 0, PARAMS, "")
    assert isinstance(sim, FillSim)
    assert sim.buy_yes == 5 and sim.buy_no == 5
    assert sim.print_vol == 200
    assert sim.newest == "t2"


def test_simulate_fills_first_cycle_no_fills():
    sim = simulate_fills(None, None, [_trade(0.45, 100, "t1")], 0, PARAMS, "")
    assert sim.buy_yes == 0 and sim.buy_no == 0
    assert sim.newest == "t1"        # watermark still advances


def test_simulate_fills_adverse_through_skipped():
    # A print far below our bid (news) sweeps through; it's skipped from fills
    # but counted as adverse.
    trades = [_trade(0.30, 50, "t1")]   # 0.30 < bid 0.45 - 0.05
    sim = simulate_fills(0.45, 0.55, trades, 0, PARAMS, "")
    assert sim.buy_yes == 0
    assert sim.adverse_skipped == 50


def test_simulate_fills_respects_inventory_cap():
    # Already at +60 net (max_inventory) -> no more YES buys even on a bid print.
    sim = simulate_fills(0.45, 0.55, [_trade(0.45, 100, "t1")], 60, PARAMS, "")
    assert sim.buy_yes == 0


def test_simulate_fills_max_fills_per_side_caps_at_one():
    # Live audit rests ONE bid; two bid prints must fill it at most once, not twice.
    params = dict(PARAMS, quote_size=1, max_fills_per_side=1)
    trades = [_trade(0.45, 100, "t1"), _trade(0.44, 100, "t2")]  # both hit the bid
    sim = simulate_fills(0.45, 0.55, trades, 0, params, "")
    assert sim.buy_yes == 1                    # one fill event of size 1, not 2
    assert sim.buy_no == 0
    # Without the per-side cap the same tape fills more (participation-limited).
    loose = simulate_fills(0.45, 0.55, trades, 0, dict(PARAMS, quote_size=1), "")
    assert loose.buy_yes > sim.buy_yes


def test_simulate_fills_default_ignores_new_knobs():
    # Absent the live-audit knobs, behavior is byte-for-byte the original.
    trades = [_trade(0.45, 100, "t1"), _trade(0.55, 100, "t2")]
    sim = simulate_fills(0.45, 0.55, trades, 0, PARAMS, "")
    assert sim.buy_yes == 5 and sim.buy_no == 5


def test_simulate_fills_pessimistic_counts_adverse_sweep():
    # count_adverse_as_fills: a sweep below our bid picks off our resting bid.
    params = dict(PARAMS, quote_size=1, max_fills_per_side=1,
                  count_adverse_as_fills=True)
    trades = [_trade(0.30, 50, "t1")]          # 0.30 < bid 0.45 - 0.05 (adverse)
    sim = simulate_fills(0.45, 0.55, trades, 0, params, "")
    assert sim.buy_yes == 1                     # picked off (optimistic sim = 0)
    assert sim.adverse_skipped == 50            # still tallied
    optimistic = simulate_fills(0.45, 0.55, trades, 0,
                                dict(params, count_adverse_as_fills=False), "")
    assert optimistic.buy_yes == 0              # excludes the sweep
