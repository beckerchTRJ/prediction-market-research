"""Two adverse-selection time gates (Fable §2c/2d — the cheapest big win):

1. Local-time cutoff: for SAME-DAY markets, stop two-sided quoting after ~13:00
   station-local (post-peak-heating). `min_quote_hours` measures hours-to-close
   (~21:00 local for these) — five hours too late; by then flow is nowcast-informed.
2. NWP blackout: cancel + don't re-quote in the ~45-min windows around synoptic
   model releases (GFS 00/06/12/18Z, available ~4h later), when counterparties
   re-price minutes ahead of the cycle-lagged fair.
"""
from __future__ import annotations

from kwt.strategies.base import MarketCtx
from kwt.strategies.market_making import plan_quote
from kwt import live_engine as le


class _NWP:
    def p_bucket(self, low, high, blend_empirical=0.6):
        return 0.50


def _same_day_ctx(hours_elapsed):
    return MarketCtx(
        ticker="KXHIGHNY-26JUL04-B80", city="nyc", target_date="2026-07-04",
        low=79, high=80, bucket_kind="range", horizon_days=0.4,
        yes_bid=0.45, yes_ask=0.55, no_bid=0.45, no_ask=0.55,
        last_price=0.50, open_interest=500.0, nwp=_NWP(),
        obs_so_far=None, hours_elapsed=hours_elapsed)


PARAMS = {"half_spread": 0.03, "same_day_cutoff_hour": 13.0}


def test_same_day_quoting_stops_after_local_cutoff():
    plan = plan_quote(_same_day_ctx(14.0), 0.0, PARAMS)   # 14:00 local -> past cutoff
    assert plan.quotable is False and plan.skip_reason == "past_local_cutoff"


def test_same_day_quoting_allowed_before_cutoff():
    plan = plan_quote(_same_day_ctx(10.0), 0.0, PARAMS)   # 10:00 local -> fine
    assert plan.quotable is True


def test_cutoff_does_not_touch_multiday_markets():
    # No intraday clock (hours_elapsed None) => not a same-day market => gate off.
    c = _same_day_ctx(None)
    assert plan_quote(c, 0.0, PARAMS).quotable is True


def test_nwp_blackout_window_detection():
    windows = [["03:30", "04:15"], ["21:30", "22:15"]]
    assert le._in_nwp_blackout("2026-07-04T21:45:00Z", windows) is True    # inside 18Z window
    assert le._in_nwp_blackout("2026-07-04T04:00:00Z", windows) is True
    assert le._in_nwp_blackout("2026-07-04T12:00:00Z", windows) is False   # clear
    assert le._in_nwp_blackout("2026-07-04T12:00:00Z", []) is False        # disabled
