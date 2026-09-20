"""The tuned day-before fade variant restricts to the harvestable band+window and
caps cross-city same-date exposure — while the original longshot_fade is unchanged.
"""
from __future__ import annotations

import types

from kwt.strategies.base import Book, MarketCtx
from kwt.strategies.longshot_fade import (LongshotFadeDayBeforeStrategy,
                                          LongshotFadeStrategy)

SERVICES = types.SimpleNamespace(fee_cfg={"taker_rate": 0.07})

DAYB = {"min_yes_price": 0.03, "max_yes_price": 0.20, "min_horizon_days": 0.5,
        "short_horizon_days": 1.5, "flat_contracts": 10, "max_entry_premium": 0.02,
        "max_date_contracts": 30, "require_model_confirm": False}


def _ctx(ticker, yes_mid, horizon, date="2026-07-03"):
    yb, ya = round(yes_mid - 0.01, 2), round(yes_mid + 0.01, 2)
    return MarketCtx(
        ticker=ticker, city="nyc", target_date=date, low=79, high=80,
        bucket_kind="range", horizon_days=horizon, yes_bid=yb, yes_ask=ya,
        no_bid=round(1 - ya, 2), no_ask=round(1 - yb, 2), last_price=yes_mid,
        open_interest=500.0, nwp=None)


def _faded(strat, ctxs):
    orders, _ = strat.generate(ctxs, Book(cash=1000.0))
    return {o.ticker: o.contracts for o in orders if o.side == "no" and o.action == "buy"}


def test_dayb_fades_only_band_and_window():
    s = LongshotFadeDayBeforeStrategy(DAYB, SERVICES)
    ctxs = [
        _ctx("IN_BAND", 0.12, 1.0),        # 12c, 1 day out -> FADE
        _ctx("TOO_CHEAP", 0.02, 1.0),      # 2c  -> below min_yes_price, skip
        _ctx("TOO_RICH", 0.25, 1.0),       # 25c -> above max_yes_price, skip
        _ctx("SAME_DAY", 0.12, 0.2),       # <0.5d -> informed-flow window, skip
        _ctx("TOO_FAR", 0.12, 3.0),        # >1.5d -> far horizon, skip
    ]
    faded = _faded(s, ctxs)
    assert set(faded) == {"IN_BAND"}
    assert faded["IN_BAND"] == 10


def test_dayb_cross_city_date_cap():
    # 5 fadeable buckets on the SAME target_date, flat 10 each, cap 30 -> total
    # NO across the date is capped at 30 (heat-dome correlated-blowout guard).
    s = LongshotFadeDayBeforeStrategy(DAYB, SERVICES)
    ctxs = [_ctx(f"B{i}", 0.12, 1.0, date="2026-07-03") for i in range(5)]
    faded = _faded(s, ctxs)
    assert sum(faded.values()) == 30


def test_original_longshot_fade_unchanged():
    # Default params: no price floor, max 0.10, window <= 2d, no date cap.
    s = LongshotFadeStrategy({"max_yes_price": 0.10, "short_horizon_days": 2.0,
                              "flat_contracts": 10, "max_entry_premium": 0.02}, SERVICES)
    # A 5c bucket same-day (0.2d) is STILL faded by the original (no lower bounds).
    faded = _faded(s, [_ctx("CHEAP_SAMEDAY", 0.05, 0.2)])
    assert faded.get("CHEAP_SAMEDAY") == 10
