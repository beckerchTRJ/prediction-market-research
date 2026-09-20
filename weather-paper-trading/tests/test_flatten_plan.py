"""plan_flatten decides the inventory regime per market (Fable §3): normal
two-sided quoting, reduce_only (shed only), or cross (pay to exit). This is the
'brain' of the flatten ladder; execution is gated separately.
"""
from __future__ import annotations

from kwt.strategies.base import MarketCtx
from kwt.strategies.market_making import plan_flatten


class _NWP:
    def __init__(self, p):
        self.p = p

    def p_bucket(self, low, high, blend_empirical=0.6):
        return self.p


def _ctx(p_yes, yes_mid=0.50, hours_elapsed=None):
    bid = None if yes_mid is None else round(yes_mid - 0.02, 2)
    ask = None if yes_mid is None else round(yes_mid + 0.02, 2)
    return MarketCtx(
        ticker="KXHIGHNY-26JUL04-B80", city="nyc", target_date="2026-07-04",
        low=79, high=80, bucket_kind="range", horizon_days=0.4,
        yes_bid=bid, yes_ask=ask, no_bid=bid, no_ask=ask,
        last_price=yes_mid, open_interest=500.0, nwp=_NWP(p_yes),
        obs_so_far=None, hours_elapsed=hours_elapsed)


P = {"flatten_reduce_at": 2, "flatten_min_win_prob": 0.35,
     "flatten_deep_itm": 0.90, "same_day_cutoff_hour": 13.0}


def test_flat_book_is_normal():
    assert plan_flatten(_ctx(0.5), 0.0, P).regime == "normal"


def test_single_contract_benign_is_normal():
    # net +1, model agrees it's a coin flip -> no pressure yet.
    assert plan_flatten(_ctx(0.5), 1.0, P).regime == "normal"


def test_inventory_band_triggers_reduce_only_on_correct_side():
    # net +2 long YES -> shed by RESTING THE ASK (sell YES).
    plan = plan_flatten(_ctx(0.5), 2.0, P)
    assert plan.regime == "reduce_only" and plan.reduce_side == "ask"
    # net -2 short (long NO) -> shed by RESTING THE BID (buy YES back).
    plan = plan_flatten(_ctx(0.5), -2.0, P)
    assert plan.regime == "reduce_only" and plan.reduce_side == "bid"


def test_feasibility_squeeze_crosses_even_at_one_lot():
    # Long 1 YES but the model now says YES only wins 20% -> toxic, cross out.
    plan = plan_flatten(_ctx(0.20, yes_mid=0.20), 1.0, P)
    assert plan.regime == "cross" and plan.reduce_side == "ask"


def test_time_stop_crosses_a_coinflip_past_local_cutoff():
    # Long 1, held win-prob 0.5 (<0.6), 14:00 local (past cutoff) -> cross.
    plan = plan_flatten(_ctx(0.50, yes_mid=0.50, hours_elapsed=14.0), 1.0, P)
    assert plan.regime == "cross" and plan.reason == "time_stop"
    # Same position BEFORE the cutoff is only reduce-pressure, not a cross.
    assert plan_flatten(_ctx(0.50, yes_mid=0.50, hours_elapsed=10.0), 1.0, P).regime == "normal"


def test_deep_itm_position_never_crosses():
    # Short YES (long NO) net -3; NO is deep ITM (yes_mid 0.05 -> NO worth 0.95).
    # Even though it's a big position, don't pay to exit a near-certain winner.
    plan = plan_flatten(_ctx(0.05, yes_mid=0.05), -3.0, P)
    assert plan.regime == "reduce_only"     # not "cross"
