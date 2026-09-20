"""Flatten is now ON by default (config.yaml), not just available via override --
these lock that in using the REAL loaded config, unlike test_flatten_wiring.py
and test_flatten_on_kill.py which explicitly force the flag either way.
"""
from __future__ import annotations

import kwt.live_engine as le
from kwt.config import load_config
from kwt.strategies.base import MarketCtx

TICKER = "KXHIGHNY-26JUL02-B80"


class _NWP:
    def p_bucket(self, low, high, blend_empirical=0.6):
        return 0.50


def _ctx():
    return MarketCtx(
        ticker=TICKER, city="nyc", target_date="2026-07-02",
        low=79, high=80, bucket_kind="range", horizon_days=1.0,
        yes_bid=0.45, yes_ask=0.55, no_bid=0.45, no_ask=0.55,
        last_price=0.50, open_interest=500.0, nwp=_NWP())


class _Kalshi:
    def trades_since(self, ticker, since, **kw):
        return []


class _LongTrading:
    def __init__(self):
        self.place_orders = True
        self.authenticated = True
        self.placed_sides = []

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        return [{"ticker": TICKER, "position_fp": "2.00",
                 "market_exposure_dollars": "1.00", "realized_pnl_dollars": "0.0"}]

    def get_orders(self, status=None):
        return []

    def get_fills(self):
        return []

    def create_order(self, *, ticker, side, price, count, client_order_id,
                     post_only=True, expiration_time=None, **kw):
        self.placed_sides.append(side)
        return {"order_id": f"{side}-1", "client_order_id": client_order_id,
                "fill_count": "0.00", "remaining_count": "1.00"}

    def cancel_order(self, oid):
        return {"order_id": oid, "status": "canceled"}


def test_stock_config_flattens_held_inventory_without_override(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _LongTrading()
    cfg = load_config()      # NO flatten override -- proves the default is on
    monkeypatch.setattr(le, "utcnow_iso", lambda: "2026-07-02T00:01:00Z")
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([_ctx()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi())
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)
    le.run_live_mm(mode="prod", db_path=db, cfg=cfg, verbose=False)
    # Long +2 -> reduce by resting the ask only, without any explicit override.
    assert trading.placed_sides == ["ask"]
