"""Deterministic unit tests for the load-bearing math (no network)."""
from __future__ import annotations

import numpy as np
import pytest

from kwt.clients.kalshi import parse_bucket, parse_event_date
from kwt.collect import _taker_tradable
from kwt.distributions import Forecast, condition_members, feasible_yes_bounds
from kwt.fees import taker_fee, maker_fee
from kwt.metrics import diebold_mariano, diebold_mariano_blocked
from kwt.strategies.base import MarketCtx, Strategy, Services
from kwt.strategies.calibration_overlay import CalibrationOverlayStrategy, recalibration_slope


# --- Kalshi market parsing -------------------------------------------------
def test_parse_event_date():
    assert parse_event_date("KXHIGHNY-26JUN08") == "2026-06-08"
    assert parse_event_date("KXHIGHCHI-26DEC31") == "2026-12-31"
    assert parse_event_date("garbage") is None


def test_parse_bucket_kinds():
    assert parse_bucket({"floor_strike": 79, "cap_strike": None}) == ("above", 80, None)
    assert parse_bucket({"floor_strike": None, "cap_strike": 72}) == ("below", None, 71)
    assert parse_bucket({"floor_strike": 78, "cap_strike": 79}) == ("range", 78, 79)


# --- fee model -------------------------------------------------------------
def test_taker_fee_known_values():
    # round_up(0.07 * 100 * 0.5 * 0.5) = round_up(1.75) = 1.75
    assert taker_fee(100, 0.50) == pytest.approx(1.75)
    # round_up(0.07 * 1 * 0.5 * 0.5) = round_up(0.0175) = 0.02
    assert taker_fee(1, 0.50) == pytest.approx(0.02)
    # fee vanishes toward the extremes
    assert taker_fee(10, 0.02) < taker_fee(10, 0.50)
    assert taker_fee(0, 0.5) == 0.0


def test_maker_fee_smaller_than_taker():
    assert maker_fee(100, 0.5) < taker_fee(100, 0.5)


# --- distribution ----------------------------------------------------------
def _members(mean=75.0, sd=4.0, n=200):
    rng = np.random.default_rng(0)
    return rng.normal(mean, sd, n)


def test_partition_probabilities_sum_to_one():
    fc = Forecast.from_members(_members())
    # a full partition of the daily-high line
    buckets = [(None, 70), (71, 72), (73, 74), (75, 76), (77, 78), (79, None)]
    total = sum(fc.p_bucket(lo, hi, blend_empirical=1.0) for lo, hi in buckets)
    assert total == pytest.approx(1.0, abs=0.02)


def test_probabilities_in_range_and_central_bucket_largest():
    fc = Forecast.from_members(_members(mean=75, sd=3))
    p_center = fc.p_bucket(74, 76)
    p_tail = fc.p_bucket(90, None)
    assert 0 < p_tail < p_center < 1
    assert p_center > 0.1


def test_student_t_has_fatter_tails_than_normal():
    fc = Forecast.from_members(_members(mean=75, sd=4))
    far = (95, None)
    assert fc.p_student_t(*far, df=4) > fc.p_normal(*far)


# --- sizing ----------------------------------------------------------------
def test_kelly_size_positive_and_zero_edge():
    s = Strategy({}, Services())
    # strong edge: model 0.7 vs price 0.5 -> positive size
    assert s.kelly_size(0.70, 0.50, 1000, 0.25, 100) > 0
    # no edge: model below price -> zero
    assert s.kelly_size(0.40, 0.50, 1000, 0.25, 100) == 0
    # capped
    assert s.kelly_size(0.99, 0.50, 1_000_000, 1.0, 30) == 30


def test_recalibration_slope_monotone_in_horizon():
    p = {"slopes": {"le_1d": 0.8, "le_2d": 0.9, "le_1w": 1.0, "le_long": 1.1}}
    assert recalibration_slope(0.5, p) == 0.8
    assert recalibration_slope(1.5, p) == 0.9
    assert recalibration_slope(5, p) == 1.0
    assert recalibration_slope(20, p) == 1.1


# --- intraday conditioning (Spec #1: time-of-day awareness) -----------------
def test_condition_members_high_floors_at_observed_max():
    members = np.array([70.0, 75.0, 80.0])
    out = condition_members(members, obs_so_far=78.0, metric="high")
    # realized high can't be below what's already been observed today
    assert list(out) == [78.0, 78.0, 80.0]


def test_condition_members_low_ceils_at_observed_min():
    members = np.array([40.0, 45.0, 50.0])
    out = condition_members(members, obs_so_far=42.0, metric="low")
    # realized low can't exceed the morning minimum already observed
    assert list(out) == [40.0, 42.0, 42.0]


def test_condition_members_noop_when_obs_none_or_weak():
    members = np.array([70.0, 75.0, 80.0])
    assert list(condition_members(members, None, "high")) == [70.0, 75.0, 80.0]
    # a weak early-morning floor below every member changes nothing (self-gating)
    assert list(condition_members(members, 60.0, "high")) == [70.0, 75.0, 80.0]


def test_feasible_bounds_high_impossible_certain_undecided():
    # observed high already 80: a "76-77" bucket can no longer occur
    assert feasible_yes_bounds(76, 77, obs_so_far=80.0, metric="high") == (0.0, 0.0)
    # a "below 79" bucket is impossible too (high >= 80 > 79.5)
    assert feasible_yes_bounds(None, 79, obs_so_far=80.0, metric="high") == (0.0, 0.0)
    # an "80 or above" bucket (low=80, high=None) is already certain
    assert feasible_yes_bounds(80, None, obs_so_far=80.0, metric="high") == (1.0, 1.0)
    # an undecided bucket straddling the observed max stays open
    assert feasible_yes_bounds(79, 82, obs_so_far=80.0, metric="high") == (0.0, 1.0)
    # no observation -> no constraint
    assert feasible_yes_bounds(76, 77, obs_so_far=None, metric="high") == (0.0, 1.0)


def test_feasible_bounds_low_symmetric():
    # observed morning low already 42: a "45-46" bucket can no longer occur
    assert feasible_yes_bounds(45, 46, obs_so_far=42.0, metric="low") == (0.0, 0.0)
    # a "42 or below" bucket (low=None, high=42) is already certain
    assert feasible_yes_bounds(None, 42, obs_so_far=42.0, metric="low") == (1.0, 1.0)
    # undecided bucket straddling the observed min
    assert feasible_yes_bounds(40, 43, obs_so_far=42.0, metric="low") == (0.0, 1.0)


def _ctx(**kw):
    base = dict(ticker="T", city="NY", target_date="2026-06-08", low=76, high=77,
                bucket_kind="range", horizon_days=0.2, yes_bid=0.01, yes_ask=0.02,
                no_bid=0.97, no_ask=0.98, last_price=0.02, open_interest=10.0)
    base.update(kw)
    return MarketCtx(**base)


# --- correlation-aware edge test (block by city-day) ------------------------
def test_blocked_dm_reports_effective_n_not_observation_count():
    # 3 "days", many correlated buckets per day. The naive DM sees 12 obs; the
    # blocked DM must collapse to 3 independent blocks (effective N).
    blocks = ["d1"] * 4 + ["d2"] * 4 + ["d3"] * 4
    # model perfectly predicts each outcome; market is always wrong by a lot
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], float)
    model = y.copy()
    market = np.full(12, 0.5)
    naive = diebold_mariano(model, market, y)
    blocked = diebold_mariano_blocked(model, market, y, blocks)
    assert naive["n"] == 12
    assert blocked["n_blocks"] == 3        # effective sample size, not 12
    assert blocked["n_obs"] == 12
    assert blocked["mean_diff"] < 0        # model has lower loss -> beats market


def test_blocked_dm_insufficient_blocks_returns_nan_pvalue():
    # one independent day -> cannot establish significance no matter how many buckets
    blocks = ["d1"] * 10
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
    res = diebold_mariano_blocked(y.copy(), np.full(10, 0.5), y, blocks)
    assert res["n_blocks"] == 1
    assert np.isnan(res["p_value"])        # honest: no cross-block variation to test


# --- taker-side liquidity gate (hygiene, not a strategy change) -------------
def test_taker_tradable_gate():
    flt = {"max_spread": 0.15, "min_open_interest": 20}
    ok = dict(yes_bid=0.40, yes_ask=0.42, no_bid=0.58, no_ask=0.60, open_interest=100.0)
    # normal, tight, liquid market -> tradable on either side
    assert _taker_tradable(_ctx(**ok), "yes", flt) is True
    assert _taker_tradable(_ctx(**ok), "no", flt) is True
    # no ask on the side we want to buy -> can't take
    assert _taker_tradable(_ctx(**{**ok, "yes_ask": None}), "yes", flt) is False
    # dead market (open interest below floor) -> skip
    assert _taker_tradable(_ctx(**{**ok, "open_interest": 5.0}), "yes", flt) is False
    # absurd quoted spread on the bought side -> skip
    assert _taker_tradable(_ctx(**{**ok, "yes_bid": 0.10, "yes_ask": 0.42}), "yes", flt) is False
    # one-sided cheap longshot (no bid) is still takable -> longshot_fade preserved
    assert _taker_tradable(_ctx(**{**ok, "no_bid": None}), "no", flt) is True


def test_feasibility_clamp_protects_all_forecast_strategies():
    # An UNconditioned forecast centered at 75 still puts ~20% on the 74-75
    # bucket, but the observed high is already 80 -> the bucket is impossible.
    # The clamp must override that residual model mass for every forecast
    # strategy, not just calibration_overlay (Codex High finding).
    from kwt.strategies.base import Book
    from kwt.strategies.ensemble_divergence import EnsembleDivergenceStrategy
    from kwt.strategies.market_making import MarketMakingStrategy

    fc = Forecast.from_members(np.random.default_rng(1).normal(75, 4, 300))
    assert fc.p_bucket(74, 75) > 0.1                      # naive model still likes it
    c = _ctx(low=74, high=75, obs_so_far=80.0, nwp=fc, clim=fc,
             yes_ask=0.20, yes_bid=0.18, no_ask=0.82, no_bid=0.80)

    ed, _ = EnsembleDivergenceStrategy({}, Services()).generate([c], Book(cash=1000.0))
    assert [o for o in ed if o.side == "yes"] == []       # never buys the impossible bucket
    assert any(o.side == "no" for o in ed)                # correctly fades it instead

    # market-maker must refuse to quote a decided bucket (skips before using kalshi)
    mm_o, mm_s = MarketMakingStrategy({}, Services(kalshi=object())).generate([c], Book(cash=1000.0))
    assert mm_o == []
    assert mm_s and mm_s[0].meta.get("reason") == "decided_by_obs"


def test_long_horizon_variant_gates_on_horizon():
    # The long-horizon ensemble variant is identical to ensemble_divergence
    # except it only trades a market while it is still >= min_horizon_days from
    # close. It must (a) drop same-day contexts entirely -- no signal, no order --
    # and (b) on day-before contexts produce exactly what the parent would.
    from kwt.strategies.base import Book
    from kwt.strategies.ensemble_divergence import EnsembleDivergenceStrategy
    from kwt.strategies.ensemble_divergence_lh import (
        EnsembleDivergenceLongHorizonStrategy,
    )

    fc = Forecast.from_members(np.random.default_rng(2).normal(74.5, 1.5, 400))
    # body-priced market the model disagrees with -> a real entry, not a tail bet
    common = dict(low=74, high=75, nwp=fc, yes_bid=0.20, yes_ask=0.25,
                  no_bid=0.75, no_ask=0.80, last_price=0.22, open_interest=100.0)
    near = _ctx(horizon_days=0.5, **common)   # same-day -> gated out
    far = _ctx(horizon_days=1.5, **common)    # day-before -> traded

    params = {"min_horizon_days": 1.0}
    lh = EnsembleDivergenceLongHorizonStrategy(params, Services())
    parent = EnsembleDivergenceStrategy({}, Services())

    # (a) same-day context is dropped before the parent logic runs at all
    o_near, s_near = lh.generate([near], Book(cash=1000.0))
    assert o_near == [] and s_near == []
    # ...while the parent would have emitted a signal on that same context
    assert parent.generate([near], Book(cash=1000.0))[1] != []

    # (b) on a day-before context the variant matches the parent's trade exactly
    # (the only field that differs is the strategy label, which is intended)
    o_lh, s_lh = lh.generate([far], Book(cash=1000.0))
    o_par, s_par = parent.generate([far], Book(cash=1000.0))
    decision = lambda o: (o.ticker, o.side, o.action, o.contracts, o.limit_price,
                          o.role, o.reason)
    assert [decision(o) for o in o_lh] == [decision(o) for o in o_par]
    assert [s.ticker for s in s_lh] == [s.ticker for s in s_par]
    assert any(o.side == "yes" for o in o_lh)   # the entry actually fires


def test_calibration_overlay_does_not_fade_decided_bucket():
    # Late in the day, observed high is 80 -> the "76-77" bucket is impossible,
    # so its market YES is ~0.02. Calibration would normally pull it toward 0.5
    # (slope<1) and buy the YES -- a bet against the already-realized outcome.
    # The feasibility clamp must force p_cal to 0 and suppress the entry.
    strat = CalibrationOverlayStrategy(
        {"edge_threshold": 0.05, "max_horizon_days": 2.0,
         "slopes": {"le_1d": 0.8}}, Services())
    ctx = _ctx(obs_so_far=80.0)
    orders, signals = strat.generate([ctx], book=__import__("kwt.strategies.base",
                                      fromlist=["Book"]).Book(cash=1000.0))
    yes_orders = [o for o in orders if o.side == "yes"]
    assert yes_orders == []                       # never buys the impossible bucket
    assert signals and signals[0].meta["p_cal"] <= 0.001
