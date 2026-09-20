"""Kalshi returns fixed-point / decimal-suffixed field names on the portfolio
endpoints (`position_fp`, `count_fp`, `yes_price_dollars`, `fee_cost`), NOT the
bare names the first cut of the code read. Reading the bare names silently
yielded 0 for every position and every fill quantity, which:

  * marked the whole open book to $0 value (mtm = -market_exposure), so growing
    inventory read as a growing INTRADAY LOSS and tripped the daily-loss kill
    switch on a purely phantom loss; and
  * stored every real fill with count=0, so the participation metric (the entire
    point of the live experiment) read 0%.

These tests pin the real Kalshi payload shape (captured from a live read-only
`/portfolio/positions` and `/portfolio/fills` dump on 2026-07-03).
"""
from __future__ import annotations

import sqlite3
import types

from kwt.db import SCHEMA
from kwt import live_engine as le

PREFIXES = ("KXHIGH",)


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def _cfg():
    return types.SimpleNamespace(raw={"live": {"funded_capital": 20.0}})


class FakeTrading:
    def __init__(self, *, positions=None, orders=None, fills=None):
        self.place_orders = True
        self._positions = positions or []
        self._orders = orders or []
        self._fills = fills or []

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        return self._positions

    def get_orders(self, status=None):
        return self._orders

    def get_fills(self):
        return self._fills


# Real Kalshi position object shape (from live read-only dump 2026-07-03).
def _kalshi_pos(ticker, position_fp, exposure_dollars, realized_dollars="0.000000"):
    return {
        "ticker": ticker,
        "position_fp": position_fp,                       # signed contracts, STRING
        "market_exposure_dollars": exposure_dollars,       # dollars, STRING
        "realized_pnl_dollars": realized_dollars,          # dollars, STRING
        "fees_paid_dollars": "0.000000",
        "resting_orders_count": 0,
    }


# Real Kalshi fill object shape (from live read-only dump 2026-07-03).
def _kalshi_fill(fill_id, ticker, count_fp, yes_price_dollars, fee_cost="0.000000"):
    return {
        "trade_id": fill_id,
        "fill_id": fill_id,
        "ticker": ticker,
        "side": "yes",
        "action": "buy",
        "count_fp": count_fp,                 # STRING, e.g. "1.00"
        "yes_price_dollars": yes_price_dollars,  # STRING dollars, e.g. "0.1400"
        "no_price_dollars": "0.8600",
        "fee_cost": fee_cost,                 # STRING dollars
        "is_taker": False,
        "created_time": "2026-07-03T15:09:59Z",
    }


def test_reconcile_reads_signed_position_from_position_fp():
    """The net position must come from `position_fp` — reading bare `position`
    (absent) made every position read as 0 (flat), the root of the phantom loss."""
    conn = _conn()
    trading = FakeTrading(positions=[
        _kalshi_pos("KXHIGHNY-26JUL03-B100.5", "-2.00", "0.99"),
        _kalshi_pos("KXHIGHCHI-26JUL04-T85", "1.00", "0.14"),
    ])
    state = le.reconcile(conn, trading, _cfg(), "2026-07-03T00:00:00Z", {}, PREFIXES, False)
    assert state.net_by_market["KXHIGHNY-26JUL03-B100.5"] == -2.0
    assert state.net_by_market["KXHIGHCHI-26JUL04-T85"] == 1.0


def test_day_pnl_is_not_a_phantom_loss_as_inventory_grows():
    """Building inventory at fair value must be ~P&L-neutral. With the bug,
    positions read as 0 so mtm = -market_exposure, and each new fill deepened a
    fake intraday loss until the kill switch fired. Regression: two cycles, more
    inventory in the second, all bought at mid -> day P&L stays ~0, not -exposure.
    """
    conn = _conn()
    # Long 2 YES at 0.50 each -> cost $1.00, mid 0.50 -> value $1.00 -> mtm 0.
    pos1 = [_kalshi_pos("KXHIGHNY-26JUL03-B50", "2.00", "1.00")]
    mid = {"KXHIGHNY-26JUL03-B50": 0.50}
    # Cycle 1 sets the day baseline (returns 0 by construction).
    le.reconcile(conn, trading_from(pos1), _cfg(), "2026-07-03T00:00:00Z", {}, PREFIXES, False, mid)
    # Cycle 2: maker added more inventory, still at fair value (long 6 YES, cost $3.00).
    pos2 = [_kalshi_pos("KXHIGHNY-26JUL03-B50", "6.00", "3.00")]
    state = le.reconcile(conn, trading_from(pos2), _cfg(), "2026-07-03T01:00:00Z", {}, PREFIXES, False, mid)
    # Fair-value inventory growth is NOT a loss. Bug produced ~ -2.00 here.
    assert abs(state.day_realized_pnl) < 0.01, state.day_realized_pnl


def trading_from(positions):
    return FakeTrading(positions=positions)


def test_ingest_fills_reads_count_price_and_fee_from_kalshi_fields():
    """Fills must store real count/price/fee from `count_fp`/`yes_price_dollars`/
    `fee_cost`; the bare names are absent, so the buggy read stored 0 for all."""
    conn = _conn()
    trading = FakeTrading(fills=[
        _kalshi_fill("f1", "KXHIGHCHI-26JUL04-T85", "1.00", "0.1400"),
        _kalshi_fill("f2", "KXHIGHNY-26JUL03-B100.5", "3.00", "0.5500", fee_cost="0.010000"),
    ])
    le._ingest_fills(conn, trading, "t", PREFIXES, False)
    rows = {r["fill_id"]: r for r in conn.execute(
        "SELECT fill_id, count, price, fee FROM live_fills").fetchall()}
    assert rows["f1"]["count"] == 1
    assert abs(rows["f1"]["price"] - 0.14) < 1e-9
    assert rows["f2"]["count"] == 3
    assert abs(rows["f2"]["price"] - 0.55) < 1e-9
    assert abs(rows["f2"]["fee"] - 0.01) < 1e-9


def test_connect_sets_busy_timeout():
    """Concurrent crons (collect / settle / live-mm) all write data/kwt.db; without
    a busy_timeout SQLite raises 'database is locked' immediately instead of
    waiting, crashing live-mm cycles. connect() must set a non-zero busy_timeout.
    """
    from kwt.db import connect
    conn = connect(":memory:")
    (timeout_ms,) = conn.execute("PRAGMA busy_timeout").fetchone()
    # Must be explicitly raised above Python's 5s default, since a build can hold
    # the write lock longer than 5s across a city's network fetches.
    assert timeout_ms >= 30000
