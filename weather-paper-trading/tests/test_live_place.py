"""Placement records the REAL order from Kalshi's V2 create-order response.

Kalshi's `POST /portfolio/events/orders` returns a FLAT body (order_id,
client_order_id, fill_count, remaining_count, ts_ms) — not nested under an
"order" key. These lock in that `_place` reads that shape: it must persist the
real order_id (not None) and derive a sane status, so the live_orders audit is
usable before any real-money run. dry_run must still record a planned no-op.
"""
from __future__ import annotations

import sqlite3
import types

from kwt.db import SCHEMA
from kwt import live_engine as le


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


class _Trading:
    """Minimal create_order stub returning a scripted response dict."""
    def __init__(self, resp):
        self._resp = resp

    def create_order(self, **kwargs):
        return dict(self._resp)


def _row(conn):
    return conn.execute(
        "SELECT ticker, status, order_id, client_order_id FROM live_orders").fetchone()


def test_flat_v2_resting_response_records_order_id_and_status():
    conn = _conn()
    trading = _Trading({
        "order_id": "ord-123", "client_order_id": "cli-abc",
        "fill_count": "0.00", "remaining_count": "1.00", "ts_ms": 1,
    })
    ok = le._place(conn, trading, "t", "prod", "KXHIGHNY-26JUL01-B80",
                   "bid", 0.55, 1, False)
    assert ok
    r = _row(conn)
    assert r["order_id"] == "ord-123"          # real id persisted (was None before)
    assert r["client_order_id"] == "cli-abc"   # from response, not the local coid
    assert r["status"] == "resting"            # 0 filled, 1 remaining


def test_flat_v2_immediate_fill_marks_filled():
    conn = _conn()
    trading = _Trading({
        "order_id": "ord-9", "client_order_id": "cli-9",
        "fill_count": "1.00", "remaining_count": "0.00", "ts_ms": 2,
    })
    le._place(conn, trading, "t", "prod", "KXHIGHNY-X", "ask", 0.60, 1, False)
    assert _row(conn)["status"] == "filled"


def test_dry_run_records_planned_noop():
    conn = _conn()
    trading = _Trading({"dry_run": True, "order": {"order_id": None,
                        "client_order_id": "x", "status": "dry_run"}})
    le._place(conn, trading, "t", "dry_run", "KXHIGHNY-X", "bid", 0.50, 1, False)
    r = _row(conn)
    assert r["status"] == "planned"
    assert r["order_id"] is None


def test_nested_response_fallback_still_reads_order_id():
    conn = _conn()  # defensive: if a future schema nests under "order"
    trading = _Trading({"order": {"order_id": "nested-1", "client_order_id": "n",
                        "fill_count": "0.00", "remaining_count": "1.00"}})
    le._place(conn, trading, "t", "prod", "KXHIGHNY-X", "bid", 0.50, 1, False)
    assert _row(conn)["order_id"] == "nested-1"
