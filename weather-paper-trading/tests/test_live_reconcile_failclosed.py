"""Fail-CLOSED reconcile.

If the authenticated maker can't get a successful account read (auth/API error
mid-session, e.g. EXPIRED_TIMESTAMP), it previously returned an EMPTY LiveState
(zero inventory, zero risk) and then cancel-and-replaced + quoted the whole
universe as if flat. On live day 1 that is how the per-market cap got breached
(net -4 vs 3) during the EXPIRED_TIMESTAMP cluster. The maker must instead treat
an incomplete reconcile as "I am blind" -> cancel-only, place NOTHING.
"""
from __future__ import annotations

import sqlite3
import types

import kwt.live_engine as le
from kwt.config import load_config
from kwt.clients.kalshi_trading import KalshiTradingError
from kwt.db import SCHEMA, init_db, connect
from kwt.strategies.base import MarketCtx


class _NWP:
    def p_bucket(self, low, high, blend_empirical=0.6):
        return 0.50


def _ctx():
    return MarketCtx(
        ticker="KXHIGHNY-26JUL02-B80", city="nyc", target_date="2026-07-02",
        low=79, high=80, bucket_kind="range", horizon_days=1.0,
        yes_bid=0.45, yes_ask=0.55, no_bid=0.45, no_ask=0.55,
        last_price=0.50, open_interest=500.0, nwp=_NWP())


class _Kalshi:
    def trades_since(self, ticker, since, **kw):
        return []


class _FailingReconcileTrading:
    """Authenticated prod client whose account read fails mid-session. Placing
    anything here would be a bug — the maker is blind and must not quote."""
    def __init__(self):
        self.place_orders = True
        self.authenticated = True
        self.created = []

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        raise KalshiTradingError(401, "EXPIRED_TIMESTAMP")

    def get_orders(self, status=None):
        return []          # nothing resting to cancel

    def get_fills(self):
        return []

    def create_order(self, **kw):
        self.created.append(kw)
        return {"order_id": "x", "client_order_id": kw.get("client_order_id"),
                "fill_count": "0.00", "remaining_count": "1.00"}

    def cancel_order(self, oid):
        return {"order_id": oid, "status": "canceled"}


def test_reconcile_failure_blocks_all_quoting(monkeypatch, tmp_path):
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _FailingReconcileTrading()
    monkeypatch.setattr(le, "utcnow_iso", lambda: "2026-07-02T00:01:00Z")
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([_ctx()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi())
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)

    s = le.run_live_mm(mode="prod", db_path=db, cfg=load_config(), verbose=False)

    assert trading.created == []                      # placed NOTHING
    assert s["placed"] == 0 and s["quoted_both"] == 0
    conn = connect(db)
    n = conn.execute("SELECT COUNT(*) FROM live_risk_events "
                     "WHERE kind='reconcile_failed'").fetchone()[0]
    conn.close()
    assert n >= 1                                     # logged the blind-state stop


def _reconcile_conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


class _OkTrading:
    authenticated = True
    place_orders = True

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        return []

    def get_orders(self, status=None):
        return []

    def get_fills(self):
        return []


def test_reconcile_flag_true_on_success_false_on_error():
    cfg = types.SimpleNamespace(raw={"live": {"funded_capital": 20.0}})
    ts = "2026-07-04T00:00:00Z"
    ok = le.reconcile(_reconcile_conn(), _OkTrading(), cfg, ts, {}, ("KXHIGH",), False)
    assert ok.reconciled is True

    bad = le.reconcile(_reconcile_conn(), _FailingReconcileTrading(), cfg, ts, {},
                       ("KXHIGH",), False)
    assert bad.reconciled is False
