"""The live engine's reserved_cost must use side_cost (1-price for asks), not raw
price, or a cycle full of cheap asks can under-reserve capital and let a later
market in the SAME cycle over-commit before the next reconcile catches it — the
mechanism behind the 2026-07-07 capital-cap breach (12.00 -> 14.14).
"""
from __future__ import annotations

import kwt.live_engine as le
from kwt.config import load_config
from kwt.strategies.base import MarketCtx


class _NWP:
    def __init__(self, p):
        self.p = p

    def p_bucket(self, low, high, blend_empirical=0.6):
        return self.p


def _cheap_ctx(ticker):
    # mid ~0.05 -> routes through ask_only_below (ask-only, cheap longshot fade).
    return MarketCtx(
        ticker=ticker, city="nyc", target_date="2026-07-02",
        low=119, high=None, bucket_kind="above", horizon_days=1.0,
        yes_bid=0.03, yes_ask=0.07, no_bid=0.93, no_ask=0.97,
        last_price=0.05, open_interest=500.0, nwp=_NWP(0.05))


class _Kalshi:
    def trades_since(self, ticker, since, **kw):
        return []


class _Trading:
    def __init__(self):
        self.place_orders = True
        self.authenticated = True
        self.placed = []       # (ticker, side, price, count)

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        return []

    def get_orders(self, status=None):
        return []

    def get_fills(self):
        return []

    def create_order(self, *, ticker, side, price, count, client_order_id,
                     post_only=True, expiration_time=None, **kw):
        self.placed.append((ticker, side, price, count))
        return {"order_id": f"{side}-{ticker}", "client_order_id": client_order_id,
                "fill_count": "0.00", "remaining_count": "1.00"}

    def cancel_order(self, oid):
        return {"order_id": oid, "status": "canceled"}


def _cfg(max_capital_at_risk):
    cfg = load_config()
    cfg.raw["live"]["risk"]["max_capital_at_risk"] = max_capital_at_risk
    cfg.raw["live"]["risk"]["max_open_orders"] = 99
    cfg.raw["live"]["risk"]["max_net_inventory"] = 99
    cfg.raw["live"]["risk"]["max_position_per_market"] = 99
    cfg.raw["live"]["risk"]["max_event_exposure"] = 99
    cfg.raw["live"]["risk"]["max_order_size"] = 1
    return cfg


def test_cheap_asks_reserve_one_minus_price_across_the_cycle(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    # Two cheap-longshot markets, each wants an ask at ~0.07 (ask-only fade).
    # cost/contract = 1-0.07 = 0.93. With a $1.00 cap, only ONE of the two asks
    # should fit (2*0.93 > 1.00); the old price-based reservation (2*0.07=0.14)
    # would have let both through.
    ctxs = [_cheap_ctx("KXHIGHNY-26JUL02-T120"), _cheap_ctx("KXHIGHCHI-26JUL02-T120")]
    trading = _Trading()
    monkeypatch.setattr(le, "utcnow_iso", lambda: "2026-07-02T00:01:00Z")
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: (ctxs, {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi())
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)
    le.run_live_mm(mode="prod", db_path=db, cfg=_cfg(1.0), verbose=False)
    assert len(trading.placed) == 1     # only one ask fit under the corrected reservation
