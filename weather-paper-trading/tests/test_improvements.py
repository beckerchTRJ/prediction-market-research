"""Tests for the strategy-viability fixes: logit recalibration, sell/exit path,
fee-net edges, cheap-price gates, and the market-only longshot fade."""
from __future__ import annotations

import sqlite3

import pytest

from kwt.db import SCHEMA
from kwt import engine
from kwt.strategies.base import Book, MarketCtx, Order, Services, Strategy
from kwt.strategies.calibration_overlay import recalibrate
from kwt.strategies.longshot_fade import LongshotFadeStrategy


def _ctx(**kw):
    base = dict(ticker="T", city="NY", target_date="2026-06-10", low=76, high=77,
                bucket_kind="range", horizon_days=1.0, yes_bid=0.02, yes_ask=0.03,
                no_bid=0.97, no_ask=0.98, last_price=None, open_interest=100.0)
    base.update(kw)
    return MarketCtx(**base)


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    c.execute("INSERT INTO strategies (name, enabled, bankroll0, cash, realized_pnl, "
              "params_json, created_ts) VALUES ('s',1,1000,1000,0,'{}','t')")
    return c


# --- logit-space recalibration ----------------------------------------------
def test_recalibrate_is_logistic_not_linear():
    # The linear bug mapped a 0.5c price to ~10.4% (a claimed 20x mispricing).
    # In log-odds space, slope 0.8 maps 0.5c to ~1.4% — a subtle tail tilt.
    p = recalibrate(0.005, 0.80)
    assert 0.005 < p < 0.02            # softened, but still a deep longshot
    # slope 1 is the identity
    assert recalibrate(0.30, 1.0) == pytest.approx(0.30, abs=1e-9)
    # symmetric around 0.5
    assert recalibrate(0.5, 0.8) == pytest.approx(0.5, abs=1e-9)
    assert recalibrate(0.995, 0.8) == pytest.approx(1 - recalibrate(0.005, 0.8), abs=1e-9)


# --- sell / exit accounting ---------------------------------------------------
def test_sell_realizes_pnl_and_frees_position():
    conn = _conn()
    book = engine.load_book(conn, "s")
    engine.execute_order(conn, Order("s", "MKT", "yes", "buy", 100, 0.40),
                         {"yes_ask": 0.40, "no_ask": 0.60}, book, {"taker_rate": 0.07})
    cost = conn.execute("SELECT cost FROM positions WHERE ticker='MKT'").fetchone()["cost"]
    # sell all 100 at bid 0.55
    got = engine.execute_order(conn, Order("s", "MKT", "yes", "sell", 100, 0.55),
                               {"yes_bid": 0.55, "no_bid": 0.45}, book, {"taker_rate": 0.07})
    assert got == 100
    assert conn.execute("SELECT 1 FROM positions WHERE ticker='MKT'").fetchone() is None
    row = conn.execute("SELECT cash, realized_pnl FROM strategies WHERE name='s'").fetchone()
    fee = 0.07 * 100 * 0.55 * 0.45
    expected_pnl = 100 * 0.55 - fee - cost
    assert row["realized_pnl"] == pytest.approx(expected_pnl, abs=0.02)
    assert row["cash"] == pytest.approx(1000 + expected_pnl, abs=0.02)
    # sell trade is settled at insert so settle.py can't re-mark it
    t = conn.execute("SELECT settled, pnl FROM trades WHERE action='sell'").fetchone()
    assert t["settled"] == 1 and t["pnl"] is not None


def test_sell_partial_keeps_proportional_cost():
    conn = _conn()
    book = engine.load_book(conn, "s")
    engine.execute_order(conn, Order("s", "MKT", "no", "buy", 50, 0.30),
                         {"yes_ask": 0.70, "no_ask": 0.30}, book, {"taker_rate": 0.07})
    cost0 = conn.execute("SELECT cost FROM positions WHERE side='no'").fetchone()["cost"]
    engine.execute_order(conn, Order("s", "MKT", "no", "sell", 20, 0.35),
                         {"yes_bid": 0.65, "no_bid": 0.35}, book, {"taker_rate": 0.07})
    pos = conn.execute("SELECT contracts, cost FROM positions WHERE side='no'").fetchone()
    assert pos["contracts"] == 30
    assert pos["cost"] == pytest.approx(cost0 * 30 / 50, abs=1e-6)


def test_exit_orders_fire_only_on_reversed_edge():
    strat = Strategy({"exit_edge": 0.05}, Services())
    strat.name = "x"
    book = Book(cash=100.0, positions={("T", "yes"): {"contracts": 10, "cost": 3.0}})
    # market bid 0.60 vs our p_yes 0.30 -> selling nets ~0.58 vs 0.30 EV: exit
    c = _ctx(yes_bid=0.60, yes_ask=0.62, no_bid=0.38, no_ask=0.40)
    out = strat.exit_orders(c, p_yes=0.30, book=book)
    assert len(out) == 1 and out[0].action == "sell" and out[0].side == "yes"
    # bid barely above p -> hold
    assert strat.exit_orders(_ctx(yes_bid=0.33, yes_ask=0.35), 0.30, book) == []


# --- fee-net edge + cheap gates ----------------------------------------------
def test_net_edge_subtracts_fee():
    s = Strategy({}, Services(fee_cfg={"taker_rate": 0.07}))
    assert s.net_edge(0.60, 0.50) == pytest.approx(0.10 - 0.07 * 0.25, abs=1e-9)


def test_cheap_gates_block_longshot_buys_for_model_strategies():
    s = Strategy({"min_buy_price": 0.10}, Services())
    assert s.passes_cheap_gates(0.20, 0.05) is False        # below min_buy_price
    s2 = Strategy({}, Services())
    assert s2.passes_cheap_gates(0.06, 0.05) is False       # <1.5x ratio when cheap
    assert s2.passes_cheap_gates(0.12, 0.05) is True
    assert s2.passes_cheap_gates(0.45, 0.40) is True        # body untouched


# --- market-only longshot fade -------------------------------------------------
def test_longshot_fade_trades_without_model_confirmation():
    strat = LongshotFadeStrategy({"max_yes_price": 0.10, "flat_contracts": 10,
                                  "max_entry_premium": 0.03}, Services())
    # cheap longshot, tight NO entry, no ensemble attached -> still fades
    c = _ctx(yes_bid=0.04, yes_ask=0.06, no_bid=0.94, no_ask=0.96, nwp=None)
    orders, signals = strat.generate([c], Book(cash=1000.0))
    assert len(orders) == 1 and orders[0].side == "no" and orders[0].contracts == 10
    # but never fades an 'above' bucket the day's observation has decided YES
    c4 = _ctx(low=70, high=None, bucket_kind="above", yes_bid=0.04, yes_ask=0.06,
              no_bid=0.94, no_ask=0.96, obs_so_far=80.0)
    o4, _ = strat.generate([c4], Book(cash=1000.0))
    assert o4 == []                     # decided YES -> never fade


def test_longshot_fade_blocks_expensive_entry():
    strat = LongshotFadeStrategy({"max_yes_price": 0.10, "flat_contracts": 10,
                                  "max_entry_premium": 0.02}, Services())
    # no_ask 5c over fair NO mid -> entry premium too high
    c = _ctx(yes_bid=0.02, yes_ask=0.04, no_bid=0.90, no_ask=0.99, nwp=None)
    orders, _ = strat.generate([c], Book(cash=1000.0))
    assert orders == []
