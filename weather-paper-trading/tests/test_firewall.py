"""The live-MM experiment is firewalled from the rest of the Kalshi account.

These lock in that positions/orders/fills NOT matching the experiment ticker
prefixes (e.g. election/turnout markets sharing the same account) are never
summed into risk, never cancelled by the kill switch / cancel-replace, and
never ingested into the fill audit.
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
    def __init__(self, *, place_orders=True, positions=None, orders=None, fills=None):
        self.place_orders = place_orders
        self._positions = positions or []
        self._orders = orders or []
        self._fills = fills or []
        self.cancelled = []

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        return self._positions

    def get_orders(self, status=None):
        return self._orders

    def get_fills(self):
        return self._fills

    def cancel_all(self, ids):
        self.cancelled.extend(ids)
        return [{"order_id": i, "status": "canceled"} for i in ids]


def test_predicate_and_default_prefix():
    assert le._in_experiment("KXHIGHNY-26JUL01-B80", PREFIXES)
    assert not le._in_experiment("KXMIDTERM-CONTROL", PREFIXES)
    assert not le._in_experiment(None, PREFIXES)
    assert le._experiment_prefixes(types.SimpleNamespace(raw={"live": {}})) == ("KXHIGH",)


def test_reconcile_ignores_turnout_positions_and_orders():
    conn = _conn()
    trading = FakeTrading(
        positions=[
            {"ticker": "KXHIGHNY-26JUL01-B80", "position": 3,
             "market_exposure_dollars": 2.0, "realized_pnl": 0.0},
            {"ticker": "KXMIDTERM-CONTROL", "position": 100,
             "market_exposure_dollars": 2200.0, "realized_pnl": 0.0},  # turnout
        ],
        orders=[
            {"order_id": "a", "ticker": "KXHIGHNY-26JUL01-B80"},
            {"order_id": "b", "ticker": "KXSENATE-XYZ"},  # turnout
        ])
    state = le.reconcile(conn, trading, _cfg(), "2026-07-01T00:00:00Z", {}, PREFIXES, False)
    # the $2,200 turnout exposure must NOT count — only the $2 weather position
    assert round(state.capital_at_risk, 2) == 2.0
    assert "KXHIGHNY-26JUL01-B80" in state.net_by_market
    assert "KXMIDTERM-CONTROL" not in state.net_by_market
    assert state.open_order_count == 1  # only the weather resting order
    # ground-truth resting order IDs must ALSO be firewalled to experiment tickers
    assert state.resting_order_ids == {"a"}


def test_cancel_all_resting_never_touches_turnout_orders():
    conn = _conn()
    trading = FakeTrading(orders=[
        {"order_id": "weather1", "ticker": "KXHIGHCHI-26JUL01-B70"},
        {"order_id": "turnout1", "ticker": "KXMIDTERM-CONTROL"},
        {"order_id": "turnout2", "ticker": "KXLAMAYOR-XYZ"},
    ])
    le._cancel_all_resting(conn, trading, "t", PREFIXES, False)
    assert trading.cancelled == ["weather1"]  # turnout orders left resting


def test_ingest_fills_drops_turnout_fills():
    conn = _conn()
    trading = FakeTrading(fills=[
        {"trade_id": "f1", "ticker": "KXHIGHNY-26JUL01-B80", "side": "yes",
         "action": "buy", "count": 1, "yes_price": 50, "fee": 1, "created_time": "t"},
        {"trade_id": "f2", "ticker": "KXMIDTERM-CONTROL", "side": "yes",
         "action": "buy", "count": 1, "yes_price": 50, "fee": 1, "created_time": "t"},
    ])
    le._ingest_fills(conn, trading, "t", PREFIXES, False)
    tickers = {r[0] for r in conn.execute("SELECT ticker FROM live_fills").fetchall()}
    assert tickers == {"KXHIGHNY-26JUL01-B80"}


def test_dry_run_cancel_is_noop():
    conn = _conn()
    trading = FakeTrading(place_orders=False,
                          orders=[{"order_id": "w", "ticker": "KXHIGHNY-X"}])
    le._cancel_all_resting(conn, trading, "t", PREFIXES, False)
    assert trading.cancelled == []


def test_kill_raises_loudly_when_it_cannot_list_orders():
    """The panic button must never silently no-op: if it can't even LIST resting
    orders (e.g. no credentials), raise instead of returning as if all clear."""
    import pytest
    from kwt.clients.kalshi_trading import KalshiAuthError

    class _Failing:
        place_orders = True

        def get_orders(self, status=None):
            raise KalshiAuthError("Missing KALSHI_API_KEY_ID")

    with pytest.raises(le.KillSwitchError):
        le._cancel_all_resting(_conn(), _Failing(), "t", PREFIXES, False,
                               raise_on_error=True)
    # Default (cancel/replace path) must NOT raise — it logs and returns 0.
    assert le._cancel_all_resting(_conn(), _Failing(), "t", PREFIXES, False) == 0
