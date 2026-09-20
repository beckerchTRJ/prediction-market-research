"""One-sided edge quotes must be EXEMPT from the two-sided spread gates.

The ~1c-wide weather books make `min_market_spread` (market_too_tight) filter
out ~half of all quote decisions — including the cheap-longshot buckets whose
ASK side is the validated favorite-longshot fade. A one-sided quote isn't making
a two-sided market, so it should not be gated by the two-sided spread checks.
Prod-neutral: with ask_only_below/bid_only_above unset, the gates apply as before.
"""
from kwt.strategies.market_making import plan_quote


class _NWP:
    def __init__(self, p): self.p = p
    def p_bucket(self, low, high, blend_empirical=0.6): return self.p


class Ctx:
    """Minimal MarketContext stub for plan_quote."""
    def __init__(self, mid, bid, ask, p=0.02):
        self.ticker = "KXHIGHNY-26JUL02-T120"
        self.low, self.high = 119, None
        self.horizon_days = 1.0
        self.yes_bid, self.yes_ask, self.yes_mid = bid, ask, mid
        self.hours_elapsed = None
        self.obs_so_far = None
        self.nwp = _NWP(p)
    def feasible_yes_bounds(self): return (0.01, 0.99)


BASE = {"quote_at_touch": True, "min_market_spread": 0.02, "touch_min_edge": 0.01,
        "fair_market_weight": 0.7, "max_inventory": 3}


def test_ask_only_exempt_from_market_too_tight():
    # Cheap longshot, 1c book (would be market_too_tight two-sided). ask_only fires.
    p = plan_quote(Ctx(mid=0.035, bid=0.02, ask=0.03), 0, {**BASE, "ask_only_below": 0.12})
    assert p.quotable is True
    assert p.new_bid is None          # toxic long side dropped
    assert p.new_ask is not None      # the fade sell rests despite the 1c book
    assert p.skip_reason is None


def test_two_sided_1c_book_still_market_too_tight():
    # Same tight book but mid in the two-sided region -> gate still applies.
    p = plan_quote(Ctx(mid=0.50, bid=0.49, ask=0.50, p=0.50), 0,
                   {**BASE, "ask_only_below": 0.12})
    assert p.quotable is False
    assert p.skip_reason == "market_too_tight"


def test_no_flags_market_too_tight_unchanged():
    # Prod-neutral: without the side-selection flags, a 1c book is filtered.
    p = plan_quote(Ctx(mid=0.035, bid=0.02, ask=0.03), 0, BASE)
    assert p.quotable is False
    assert p.skip_reason == "market_too_tight"


def test_bid_only_exempt_from_market_too_tight():
    p = plan_quote(Ctx(mid=0.90, bid=0.895, ask=0.905, p=0.90), 0,
                   {**BASE, "bid_only_above": 0.80})
    assert p.quotable is True
    assert p.new_ask is None
    assert p.new_bid is not None


def test_touch_min_edge_ask_zero_is_honored():
    # Regression for the `0.0 or fallback` bug: edge_ask=0.0 must join the touch,
    # not silently fall back to touch_min_edge. Book straddles fair tightly so that
    # fair+edge exceeds the touch and the edge actually binds: edge_ask=0 floors the
    # ask at fair (~join the touch), a positive fallback would push it higher.
    ctx = Ctx(mid=0.30, bid=0.29, ask=0.31, p=0.30)
    p0 = plan_quote(ctx, 0, {**BASE, "touch_min_edge": 0.05, "touch_min_edge_ask": 0.0})
    pf = plan_quote(ctx, 0, {**BASE, "touch_min_edge": 0.05})   # ask falls back to 0.05
    assert p0.new_ask < pf.new_ask   # 0.0 honored -> ask sits lower (nearer fair)


def test_self_cross_skipped_when_one_sided():
    # Force a case where the two-sided quote would self-cross (new_bid >= new_ask)
    # yet ask_only keeps only the ask -> must still be quotable, not spread_too_tight.
    # Wide book so we pass market_too_tight regardless; tiny book width via fair.
    p = plan_quote(Ctx(mid=0.035, bid=0.01, ask=0.99), 0,
                   {**BASE, "min_market_spread": 0.0, "ask_only_below": 0.12,
                    "touch_min_edge": 0.5})   # huge edge would cross a two-sided quote
    assert p.quotable is True
    assert p.new_bid is None and p.new_ask is not None
