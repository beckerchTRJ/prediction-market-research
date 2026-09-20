"""Gated flatten execution in run_live_mm: when live.flatten.enabled and the maker
holds one-sided inventory, it rests ONLY the reducing side (relaxing both-or-neither)
instead of re-quoting two-sided and adding to the toxic position. Default-off: with
the flag disabled the maker quotes normally.
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
    """Prod client holding net +2 YES in the quoted market; records placed sides."""
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


def _cfg(flatten_enabled):
    cfg = load_config()
    cfg.raw.setdefault("live", {}).setdefault("flatten", {})["enabled"] = flatten_enabled
    return cfg


def _drive(monkeypatch, db, trading, cfg):
    monkeypatch.setattr(le, "utcnow_iso", lambda: "2026-07-02T00:01:00Z")
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([_ctx()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi())
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)
    return le.run_live_mm(mode="prod", db_path=db, cfg=cfg, verbose=False)


def test_flatten_rests_only_the_reducing_side(monkeypatch, tmp_path):
    from kwt.db import init_db, connect
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _LongTrading()
    _drive(monkeypatch, db, trading, _cfg(flatten_enabled=True))
    # Long +2 -> shed by resting the ASK only; the bid (accumulating side) is pulled.
    assert trading.placed_sides == ["ask"]
    conn = connect(db)
    n = conn.execute("SELECT COUNT(*) FROM live_risk_events WHERE kind='flatten'").fetchone()[0]
    conn.close()
    assert n >= 1


def test_flatten_disabled_quotes_two_sided(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _LongTrading()
    _drive(monkeypatch, db, trading, _cfg(flatten_enabled=False))
    # Flag off: normal two-sided quoting even while holding inventory.
    assert sorted(trading.placed_sides) == ["ask", "bid"]


class _TrackingLongTrading:
    """Like _LongTrading, but get_orders reflects what's actually resting across
    cycles (ground truth), so a bug in flatten's oid caching would surface as
    accumulated resting orders instead of being masked by an always-empty list."""
    def __init__(self):
        self.place_orders = True
        self.authenticated = True
        self.placed_sides = []
        self._resting = {}   # order_id -> (ticker, side, price)
        self._n = 0

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        return [{"ticker": TICKER, "position_fp": "2.00",
                 "market_exposure_dollars": "1.00", "realized_pnl_dollars": "0.0"}]

    def get_orders(self, status=None):
        return [{"order_id": oid, "ticker": t} for oid, (t, s, p) in self._resting.items()]

    def get_fills(self):
        return []

    def create_order(self, *, ticker, side, price, count, client_order_id,
                     post_only=True, expiration_time=None, **kw):
        self._n += 1
        oid = f"o{self._n}"
        self._resting[oid] = (ticker, side, price)
        self.placed_sides.append(side)
        return {"order_id": oid, "client_order_id": client_order_id,
                "fill_count": "0.00", "remaining_count": "1.00"}

    def cancel_order(self, oid):
        self._resting.pop(oid, None)
        return {"order_id": oid, "status": "canceled"}


def test_flatten_does_not_stack_orders_across_cycles(monkeypatch, tmp_path):
    """A persistent held position (position unchanged, order never fills) must
    rest at most ONE reduce-only order at a time -- not accumulate a new one
    every cycle on top of the still-resting prior one."""
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)

    trading = _TrackingLongTrading()
    cfg = _cfg(flatten_enabled=True)

    def _drive_at(ts):
        monkeypatch.setattr(le, "utcnow_iso", lambda: ts)
        monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([_ctx()], {}))
        monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi())
        monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
        monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)
        return le.run_live_mm(mode="prod", db_path=db, cfg=cfg, verbose=False)

    _drive_at("2026-07-02T00:00:00Z")
    _drive_at("2026-07-02T00:10:00Z")
    _drive_at("2026-07-02T00:20:00Z")

    # Exactly one order resting at the end, not three stacked on top of each other.
    assert len(trading._resting) == 1


def test_cross_flatten_uses_bounded_ioc_and_cooldown(tmp_path):
    from kwt.db import init_db, connect
    from kwt.risk import LiveRiskLimits, LiveState
    from kwt.strategies.market_making import FlattenPlan

    class Trading(_LongTrading):
        def __init__(self):
            super().__init__(); self.calls = []

        def create_order(self, **kw):
            self.calls.append(kw)
            return {"order_id": "ioc-1", "client_order_id": kw["client_order_id"],
                    "fill_count": "1.00", "remaining_count": "0.00"}

    db = str(tmp_path / "kwt.db"); init_db(db); conn = connect(db)
    trading = Trading(); summary = {"placed": 0}
    params = {"cross_enabled": True, "cross_max_slippage_ticks": 1,
              "cross_max_contracts_per_cycle": 1, "cross_cooldown_seconds": 600}
    flat = FlattenPlan("cross", "ask", "squeeze")
    state = LiveState(open_order_count=0)
    le._flatten_market(conn, trading, "2026-07-02T00:00:00Z", "prod", _ctx(), 2,
                       flat, state, LiveRiskLimits(), params, None, summary, False)
    call = trading.calls[0]
    assert call["side"] == "ask" and call["price"] == 0.44 and call["count"] == 1
    assert call["post_only"] is False
    assert call["time_in_force"] == "immediate_or_cancel"
    le._flatten_market(conn, trading, "2026-07-02T00:05:00Z", "prod", _ctx(), 2,
                       flat, state, LiveRiskLimits(), params, None, summary, False)
    assert len(trading.calls) == 1
    conn.close()
