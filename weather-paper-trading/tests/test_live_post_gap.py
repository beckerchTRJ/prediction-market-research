"""Post-gap safety: the first cycle after a long gap (host slept, cron stalled)
must reconcile + cancel only, and place NO quotes — otherwise the maker re-quotes
off pre-sleep state before validating the world. On live day 1 a 5-hour gap was
followed by an immediate pick-off on wake.
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


def test_no_gap_on_first_ever_cycle():
    conn = _conn()
    assert le._gap_since_last_cycle(conn, "2026-07-04T00:00:00Z", 1200) is False


def test_normal_cadence_is_not_a_gap():
    conn = _conn()
    le._gap_since_last_cycle(conn, "2026-07-04T00:00:00Z", 1200)
    # 10 min later — normal cron cadence, not a gap.
    assert le._gap_since_last_cycle(conn, "2026-07-04T00:10:00Z", 1200) is False


def test_long_gap_is_flagged_then_clears():
    conn = _conn()
    le._gap_since_last_cycle(conn, "2026-07-04T00:00:00Z", 1200)
    # 5-hour gap (host slept) -> flagged.
    assert le._gap_since_last_cycle(conn, "2026-07-04T05:00:00Z", 1200) is True
    # The watermark advanced, so the very next normal cycle is clean again.
    assert le._gap_since_last_cycle(conn, "2026-07-04T05:10:00Z", 1200) is False
