"""Cancel-and-replace every 10 min was unconditional, even when the price hadn't
changed -- sending the quote to the back of the exchange's price-time queue for
free. These lock in the price-unchanged fast path: an order confirmed still
resting (ground truth) at the identical price is left alone; anything else
(price changed, order already gone, market no longer quotable) is shed and
replaced, scoped to that one ticker.
"""
from __future__ import annotations

import kwt.live_engine as le
from kwt.config import load_config
from kwt.strategies.base import MarketCtx

TICKER = "KXHIGHNY-26JUL02-B80"
TICKER2 = "KXHIGHCHI-26JUL02-B80"


class _NWP:
    def __init__(self, p=0.50):
        self.p = p

    def p_bucket(self, low, high, blend_empirical=0.6):
        return self.p


def _ctx(ticker=TICKER, city="nyc", oi=500.0, nwp=None):
    return MarketCtx(
        ticker=ticker, city=city, target_date="2026-07-02",
        low=79, high=80, bucket_kind="range", horizon_days=1.0,
        yes_bid=0.45, yes_ask=0.55, no_bid=0.45, no_ask=0.55,
        last_price=0.50, open_interest=oi, nwp=nwp or _NWP())


class _Kalshi:
    def trades_since(self, ticker, since, **kw):
        return []


class _Trading:
    """Prod client that remembers what it currently has resting (order_id ->
    (ticker, side, price)) so get_orders reflects reality across cycles, and
    records every create/cancel call."""
    def __init__(self, positions=None):
        self.place_orders = True
        self.authenticated = True
        self.created = []      # (ticker, side, price)
        self.canceled = []     # order_id
        self._resting = {}     # order_id -> (ticker, side, price)
        self._n = 0
        self._positions = positions or []

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        return self._positions

    def get_orders(self, status=None):
        return [{"order_id": oid, "ticker": t} for oid, (t, s, p) in self._resting.items()]

    def get_fills(self):
        return []

    def create_order(self, *, ticker, side, price, count, client_order_id,
                     post_only=True, expiration_time=None, **kw):
        self._n += 1
        oid = f"o{self._n}"
        self._resting[oid] = (ticker, side, price)
        self.created.append((ticker, side, price))
        return {"order_id": oid, "client_order_id": client_order_id,
                "fill_count": "0.00", "remaining_count": "1.00"}

    def cancel_order(self, oid):
        self._resting.pop(oid, None)
        self.canceled.append(oid)
        return {"order_id": oid, "status": "canceled"}


def _drive(monkeypatch, db, trading, ctxs, ts, cfg=None):
    monkeypatch.setattr(le, "utcnow_iso", lambda: ts)
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: (ctxs, {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi())
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)
    return le.run_live_mm(mode="prod", db_path=db, cfg=cfg or load_config(), verbose=False)


def test_unchanged_quote_is_left_resting(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    ctxs = [_ctx()]
    s1 = _drive(monkeypatch, db, trading, ctxs, "2026-07-02T00:00:00Z")
    assert s1["quoted_both"] == 1
    assert sorted(x[1] for x in trading.created) == ["ask", "bid"]
    assert trading.canceled == []

    # Cycle 2: identical market/plan -> both sides unchanged.
    s2 = _drive(monkeypatch, db, trading, ctxs, "2026-07-02T00:10:00Z")
    assert s2["quoted_both"] == 1          # coverage row still shows a two-sided quote
    assert len(trading.created) == 2       # NO new orders created in cycle 2
    assert trading.canceled == []          # NO cancels in cycle 2


def test_price_change_replaces_only_that_ticker(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    ctxs = [_ctx(TICKER, "nyc"), _ctx(TICKER2, "chi")]
    _drive(monkeypatch, db, trading, ctxs, "2026-07-02T00:00:00Z")
    assert len(trading.created) == 4       # 2 tickers x 2 sides

    # Cycle 2: TICKER's forecast moves (new fair -> new prices); TICKER2 unchanged.
    moved_ctxs = [_ctx(TICKER, "nyc", nwp=_NWP(0.30)), _ctx(TICKER2, "chi")]
    _drive(monkeypatch, db, trading, moved_ctxs, "2026-07-02T00:10:00Z")
    # Only TICKER's two orders (from cycle 1, "o1"/"o2") were canceled; TICKER2's
    # cycle-1 orders ("o3"/"o4") are untouched since its price didn't change.
    assert len(trading.canceled) == 2
    assert all(oid in ("o1", "o2") for oid in trading.canceled)   # TICKER's cycle-1 orders
    assert len(trading.created) == 6       # 4 from cycle 1 + 2 fresh for TICKER in cycle 2


def test_filled_order_forces_full_replace(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    ctxs = [_ctx()]
    _drive(monkeypatch, db, trading, ctxs, "2026-07-02T00:00:00Z")
    assert len(trading.created) == 2

    # Simulate the bid having FILLED between cycles: remove it from _resting (so
    # get_orders no longer reports it) and reflect the resulting position.
    bid_oid = next(oid for oid, (t, s, p) in trading._resting.items() if s == "bid")
    del trading._resting[bid_oid]
    trading._positions = [{"ticker": TICKER, "position_fp": "1.00",
                           "market_exposure_dollars": "0.45", "realized_pnl_dollars": "0.0"}]

    _drive(monkeypatch, db, trading, ctxs, "2026-07-02T00:10:00Z")
    # The still-resting ask (unaffected by the fill) is defensively replaced too
    # (this coarser per-ticker design always replaces the WHOLE pair once either
    # side needs it, rather than tracking mixed keep/replace state) -- both a
    # fresh bid and fresh ask are placed in cycle 2.
    assert len(trading.created) == 4


def test_market_no_longer_quotable_sheds_stale_orders(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    ctxs = [_ctx(oi=500.0)]
    _drive(monkeypatch, db, trading, ctxs, "2026-07-02T00:00:00Z")
    assert len(trading.created) == 2
    assert trading.canceled == []

    # Cycle 2: open interest drops below min_open_interest (50) -> gated.
    gated_ctxs = [_ctx(oi=10.0)]
    _drive(monkeypatch, db, trading, gated_ctxs, "2026-07-02T00:10:00Z")
    assert sorted(trading.canceled) == ["o1", "o2"]   # both stale orders shed
    assert len(trading.created) == 2                  # nothing new placed
