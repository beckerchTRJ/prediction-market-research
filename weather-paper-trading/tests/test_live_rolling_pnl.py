"""The daily-loss kill switch used a UTC-midnight baseline, but the trading
evening (22Z-03Z) straddles that reset, so a single bad session split its loss
across two "days", each under the -$5 limit, and the switch never fired. A
rolling-window baseline measures the loss as one continuous drawdown.
"""
from __future__ import annotations

import sqlite3

from kwt.db import SCHEMA
from kwt import live_engine as le


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def test_rolling_pnl_captures_loss_across_utc_midnight():
    conn = _conn()
    # Session opens 23:00Z flat; first checkpoint baselines to itself -> 0.
    assert le._rolling_pnl(conn, 0.0, "2026-07-03T23:00:00Z") == 0.0
    # 01:00Z next UTC day, equity has dropped to -6. A UTC-midnight reset would
    # have measured only the post-midnight slice; the rolling window sees -6.
    assert le._rolling_pnl(conn, -6.0, "2026-07-04T01:00:00Z") == -6.0


def test_rolling_pnl_baselines_against_the_24h_old_checkpoint():
    conn = _conn()
    le._rolling_pnl(conn, 5.0, "2026-07-01T00:00:00Z")     # stale, >24h old
    le._rolling_pnl(conn, 0.0, "2026-07-02T11:30:00Z")     # ~24h before the eval
    le._rolling_pnl(conn, -1.0, "2026-07-03T02:00:00Z")    # inside the window (not baseline)
    # Eval at 2026-07-03T12:00Z: cutoff is 07-02T12:00; the most recent checkpoint
    # at/least-24h old is the 07-02T11:30 one (equity 0), NOT the stale 07-01 point.
    assert le._rolling_pnl(conn, -2.0, "2026-07-03T12:00:00Z") == -2.0
