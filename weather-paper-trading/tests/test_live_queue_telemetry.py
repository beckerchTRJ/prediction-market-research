"""Passive queue telemetry must never influence the live execution path."""
from __future__ import annotations

import kwt.live_engine as le
from kwt.db import connect, init_db, kv_set
from kwt.risk import LiveState


class _QueueClient:
    authenticated = True

    def __init__(self):
        self.requested_tickers = None

    def get_queue_positions(self, market_tickers):
        self.requested_tickers = list(market_tickers)
        return [
            {"order_id": "ours", "queue_position_fp": "2.00"},
            {"order_id": "foreign", "queue_position_fp": "99.00"},
        ]


class _BrokenQueueClient:
    authenticated = True

    def get_queue_positions(self, market_tickers):
        raise le.KalshiTradingError(503, "queue unavailable")


def _state():
    s = LiveState(reconciled=True)
    s.resting_orders = {
        "ours": {"ticker": "KXHIGHNY-26JUL20-B80", "side": "bid", "price": .40,
                 "remaining": 1.0, "created_time": "2026-07-20T00:00:00Z"},
    }
    return s


def test_queue_snapshot_is_firewalled_and_idempotent(tmp_path):
    db = str(tmp_path / "kwt.db"); init_db(db)
    c = connect(db); kv_set(c, "mmlive:cohort", "ask15_v2")
    client = _QueueClient()
    n = le._capture_queue_observations(c, client, _state(), "2026-07-20T00:10:00Z", ("KXHIGH",))
    assert n == 1
    # The read must be scoped to our experiment's resting tickers — Kalshi
    # rejects an unscoped queue_positions call.
    assert client.requested_tickers == ["KXHIGHNY-26JUL20-B80"]
    # The primary key makes a retry within the same reconcile cycle idempotent.
    assert le._capture_queue_observations(c, _QueueClient(), _state(), "2026-07-20T00:10:00Z", ("KXHIGH",)) == 1
    rows = c.execute("SELECT cohort_id,order_id,queue_ahead,quote_age_sec FROM live_queue_observations").fetchall()
    assert len(rows) == 1
    assert tuple(rows[0]) == ("ask15_v2", "ours", 2.0, 600.0)


def test_no_resting_experiment_orders_skips_the_queue_read(tmp_path):
    db = str(tmp_path / "kwt.db"); init_db(db)
    c = connect(db)
    client = _QueueClient()
    state = LiveState(reconciled=True)
    state.resting_orders = {
        "foreign": {"ticker": "KXMIDTERM-X", "side": "bid", "price": .40,
                    "remaining": 1.0, "created_time": "2026-07-20T00:00:00Z"},
    }
    assert le._capture_queue_observations(c, client, state, "2026-07-20T00:10:00Z", ("KXHIGH",)) == 0
    assert client.requested_tickers is None  # nothing of ours resting -> no API call


def test_queue_failure_only_logs_a_diagnostic(tmp_path):
    db = str(tmp_path / "kwt.db"); init_db(db)
    c = connect(db); kv_set(c, "mmlive:cohort", "ask15_v2")
    assert le._capture_queue_observations(
        c, _BrokenQueueClient(), _state(), "2026-07-20T00:10:00Z", ("KXHIGH",)) == 0
    assert c.execute("SELECT kind FROM live_risk_events").fetchone()[0] == "queue_error"


def test_unauthenticated_queue_collection_is_a_noop(tmp_path):
    db = str(tmp_path / "kwt.db"); init_db(db)
    c = connect(db)
    client = _QueueClient(); client.authenticated = False
    assert le._capture_queue_observations(c, client, _state(), "2026-07-20T00:10:00Z", ("KXHIGH",)) == 0
    assert c.execute("SELECT COUNT(*) FROM live_queue_observations").fetchone()[0] == 0


def test_monitoring_alerts_on_stale_queue_and_collapsed_fill_ratio(tmp_path):
    db = str(tmp_path / "kwt.db"); init_db(db)
    c = connect(db); kv_set(c, "mmlive:cohort", "ask15_v2")
    c.execute("INSERT INTO live_queue_observations (ts,cohort_id,order_id,quote_age_sec) VALUES (?,?,?,?)",
              ("2026-07-20T00:10:00Z", "ask15_v2", "o1", 1801))
    c.execute("INSERT INTO mm_fill_audit (interval_end,ticker,print_vol,sim_fill_qty,actual_fill_qty,cohort_id) "
              "VALUES (?,?,?,?,?,?)", ("2026-07-20T00:10:00Z", "KXHIGHNY-A", 25, 20, 0, "ask15_v2"))
    le._monitor_experiment(c, "2026-07-20T00:10:00Z", "ask15_v2", 1800, 5)
    alerts = [r[0] for r in c.execute("SELECT detail FROM live_risk_events WHERE kind='monitor_alert'")]
    assert any("TTL" in x for x in alerts)
    assert any("fill ratio" in x for x in alerts)
