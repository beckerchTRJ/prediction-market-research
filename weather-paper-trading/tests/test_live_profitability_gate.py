"""Settlement gate and staged cohort activation use independent target dates."""
from __future__ import annotations

from kwt.config import load_config
from kwt.db import connect
from kwt.live_engine import _active_live_cfg
from kwt.markout import net_markout_evidence


def _add_dates(conn, cohort, values):
    for i, value in enumerate(values, start=1):
        day = f"2026-07-{i:02d}"
        ticker = f"KXHIGHNY-{cohort[:4].upper()}26JUL{i:02d}-B80"
        conn.execute("INSERT INTO markets (ticker,target_date,result) VALUES (?,?,?)", (ticker, day, "yes"))
        conn.execute("INSERT INTO snapshots (ts,ticker,yes_bid,yes_ask) VALUES (?,?,?,?)", (day + "T00:00:00Z", ticker, .49, .51))
        conn.execute("INSERT INTO live_fills (ts,fill_id,ticker,book_side,count,price,fee,is_taker,created_time,cohort_id) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (day, f"{cohort}-{i}", ticker, "bid", 1, .50, 0, 0, day + "T00:00:00Z", cohort))
        conn.execute("INSERT INTO live_fill_markouts (fill_id,ticker,created_time,direction,mid_at_fill,mo_60,mo_settle) "
                     "VALUES (?,?,?,?,?,?,?)", (f"{cohort}-{i}", ticker, day + "T00:00:00Z", 1, .50, value, value))
    conn.commit()


def test_settlement_gate_stops_negative_and_expands_positive():
    c = connect(":memory:")
    _add_dates(c, "negative", [-.02, -.01] * 5)
    assert net_markout_evidence(c, "negative")["settlement"]["status"] == "futility"
    _add_dates(c, "positive", [.01, .02] * 10)
    assert net_markout_evidence(c, "positive")["settlement"]["status"] == "expand"


def test_successor_remains_inactive_until_ten_settled_dates():
    c = connect(":memory:")
    cfg = load_config()
    live, cohort, dates, status = _active_live_cfg(c, cfg)
    assert cohort == "ask15_v2" and dates == 0 and status == "collecting"
    assert live["risk"]["max_capital_at_risk"] == 12.0
    _add_dates(c, "ask15_v2", [.01, .02] * 5)
    live, cohort, dates, status = _active_live_cfg(c, cfg)
    assert cohort == "capacity16_v3" and dates == 10 and status == "collecting"
    assert live["risk"]["max_capital_at_risk"] == 16.0
    assert live["risk"]["max_net_inventory"] == 19
    assert live["quote_overrides"]["min_market_spread"] == .02


def test_futility_prevents_successor_activation():
    c = connect(":memory:")
    _add_dates(c, "ask15_v2", [-.02, -.01] * 5)
    _, cohort, dates, status = _active_live_cfg(c, load_config())
    assert cohort == "ask15_v2" and dates == 10 and status == "futility"
