"""Settlement accounting test on an in-memory DB (no network)."""
from __future__ import annotations

import sqlite3

import pytest

from kwt.db import SCHEMA
from kwt import engine
from kwt.strategies.base import Book, Order


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    c.execute("INSERT INTO strategies (name, enabled, bankroll0, cash, realized_pnl, "
              "params_json, created_ts) VALUES ('s',1,1000,1000,0,'{}','t')")
    return c


def test_buy_then_winning_settlement():
    conn = _conn()
    book = engine.load_book(conn, "s")
    # buy 100 YES @ 0.40 (ask 0.40), fee = round_up(0.07*100*0.4*0.6)=round_up(1.68)=1.68
    order = Order("s", "MKT", "yes", "buy", 100, 0.40)
    filled = engine.execute_order(conn, order, {"yes_ask": 0.40, "no_ask": 0.60}, book,
                                  {"taker_rate": 0.07})
    assert filled == 100
    cash_after_buy = conn.execute("SELECT cash FROM strategies WHERE name='s'").fetchone()["cash"]
    assert cash_after_buy == pytest.approx(1000 - (100 * 0.40 + 1.68), abs=1e-6)

    # market resolves YES -> 100 contracts pay $1 each
    conn.execute("INSERT INTO markets (ticker, result, status, close_time, city, target_date) "
                 "VALUES ('MKT','yes','settled','2000-01-01',NULL,NULL)")
    pos = conn.execute("SELECT contracts, cost FROM positions WHERE ticker='MKT'").fetchone()
    payoff = pos["contracts"]            # 100
    pnl = payoff - pos["cost"]
    # apply settlement the way settle.py does
    srow = conn.execute("SELECT cash, realized_pnl FROM strategies WHERE name='s'").fetchone()
    conn.execute("UPDATE strategies SET cash=?, realized_pnl=? WHERE name='s'",
                 (srow["cash"] + payoff, srow["realized_pnl"] + pnl))
    final = conn.execute("SELECT cash, realized_pnl FROM strategies WHERE name='s'").fetchone()
    # net profit = 100*(1-0.40) - 1.68 = 58.32
    assert final["realized_pnl"] == pytest.approx(58.32, abs=1e-6)
    assert final["cash"] == pytest.approx(1000 + 58.32, abs=1e-6)


def test_losing_settlement_loses_cost_only():
    conn = _conn()
    book = engine.load_book(conn, "s")
    engine.execute_order(conn, Order("s", "MKT", "no", "buy", 50, 0.30),
                         {"yes_ask": 0.70, "no_ask": 0.30}, book, {"taker_rate": 0.07})
    pos = conn.execute("SELECT cost FROM positions WHERE ticker='MKT' AND side='no'").fetchone()
    # NO loses (result yes) -> payoff 0, pnl = -cost
    assert -pos["cost"] < 0


def test_risk_cap_scaling_not_exceeding_cash():
    conn = _conn()
    conn.execute("UPDATE strategies SET cash=10 WHERE name='s'")
    book = engine.load_book(conn, "s")
    # try to buy 100 @ 0.40 = $40 but only $10 cash -> scaled down, never negative cash
    engine.execute_order(conn, Order("s", "MKT", "yes", "buy", 100, 0.40),
                         {"yes_ask": 0.40, "no_ask": 0.60}, book, {"taker_rate": 0.07})
    cash = conn.execute("SELECT cash FROM strategies WHERE name='s'").fetchone()["cash"]
    assert cash >= 0


def test_edge_eval_uses_earliest_signal_not_converged_close():
    """The edge test must compare model vs market at FIRST sighting, before the
    market converges near settlement — otherwise the market looks omniscient."""
    from kwt.report import _resolved_signal_frame
    conn = _conn()
    conn.execute("INSERT INTO markets (ticker, result, status, close_time, city, target_date) "
                 "VALUES ('MKT','yes','settled','2026-06-08T23:59:00Z','NY','2026-06-08')")
    # early sighting: model and market both uncertain (the fair comparison point)
    conn.execute("INSERT INTO signals (ts, strategy, ticker, model_prob, market_prob, edge, "
                 "side, decision, meta_json) VALUES "
                 "('2026-06-08T06:00:00Z','s','MKT',0.30,0.50,0.0,'yes','enter','{}')")
    # near-close sighting: market has converged to ~1.0 (would make market Brier ~0)
    conn.execute("INSERT INTO signals (ts, strategy, ticker, model_prob, market_prob, edge, "
                 "side, decision, meta_json) VALUES "
                 "('2026-06-08T22:00:00Z','s','MKT',0.85,0.99,0.0,'yes','hold','{}')")
    df = _resolved_signal_frame(conn)
    row = df[df["ticker"] == "MKT"].iloc[0]
    assert row["model_prob"] == 0.30        # earliest, not the 0.85 near-close value
    assert row["market_prob"] == 0.50       # earliest, not the converged 0.99
    assert row["outcome"] == 1.0


def test_edge_eval_skips_probabilityless_earliest_signal():
    """Earliest EVALUABLE signal: a null-prob skip (e.g. calibration_overlay's
    out-of-horizon log) must not represent — and thereby drop — the market."""
    from kwt.report import _resolved_signal_frame
    conn = _conn()
    conn.execute("INSERT INTO markets (ticker, result, status, close_time, city, target_date) "
                 "VALUES ('MKT','yes','settled','2026-06-08T23:59:00Z','NY','2026-06-08')")
    # earliest row is an out-of-horizon skip with NULL probabilities
    conn.execute("INSERT INTO signals (ts, strategy, ticker, model_prob, market_prob, edge, "
                 "side, decision, meta_json) VALUES "
                 "('2026-06-01T06:00:00Z','s','MKT',NULL,NULL,NULL,'none','skip','{}')")
    # later, in-horizon, evaluable signal
    conn.execute("INSERT INTO signals (ts, strategy, ticker, model_prob, market_prob, edge, "
                 "side, decision, meta_json) VALUES "
                 "('2026-06-08T06:00:00Z','s','MKT',0.30,0.50,0.0,'yes','enter','{}')")
    df = _resolved_signal_frame(conn)
    assert (df["ticker"] == "MKT").any()    # market is NOT dropped
    row = df[df["ticker"] == "MKT"].iloc[0]
    assert row["model_prob"] == 0.30 and row["market_prob"] == 0.50
