"""W4 gated live-engine levers: resting passive exits, METAR requote blackout,
the flatten log line, and honoring the W3 one-sided-plan contract in the
placement path.

Every new behavior is config-gated and defaults to current behavior: the first
test locks in that a dry-run cycle with ALL new flags off is byte-identical to
the existing engine (no new orders, no new cancels, no new risk events).
"""
from __future__ import annotations

import kwt.live_engine as le
from kwt.config import load_config
from kwt.strategies.base import MarketCtx
from kwt.strategies.market_making import QuotePlan

TICKER = "KXHIGHNY-26JUL02-B80"


class _NWP:
    def p_bucket(self, low, high, blend_empirical=0.6):
        return 0.50


def _ctx(oi=500.0):
    return MarketCtx(
        ticker=TICKER, city="nyc", target_date="2026-07-02",
        low=79, high=80, bucket_kind="range", horizon_days=1.0,
        yes_bid=0.45, yes_ask=0.55, no_bid=0.45, no_ask=0.55,
        last_price=0.50, open_interest=oi, nwp=_NWP())


class _Kalshi:
    def __init__(self, prints=None):
        self._prints = prints or []

    def trades_since(self, ticker, since, **kw):
        return [t for t in self._prints if not since or t.get("created_time", "") > since]


class _Trading:
    """Prod client; records (side, price) of every order placed. Optional held
    positions drive the reconciled inventory."""
    def __init__(self, positions=None):
        self.place_orders = True
        self.authenticated = True
        self.placed = []            # (side, price)
        self._positions = positions or []

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        return self._positions

    def get_orders(self, status=None):
        return []

    def get_fills(self):
        return []

    def create_order(self, *, ticker, side, price, count, client_order_id,
                     post_only=True, expiration_time=None, **kw):
        self.placed.append((side, price))
        return {"order_id": f"{side}-{len(self.placed)}",
                "client_order_id": client_order_id,
                "fill_count": "0.00", "remaining_count": "1.00"}

    def cancel_order(self, oid):
        return {"order_id": oid, "status": "canceled"}


def _cfg(**live_overrides):
    cfg = load_config()
    live = cfg.raw.setdefault("live", {})
    for k, v in live_overrides.items():
        if isinstance(v, dict):
            live.setdefault(k, {})
            live[k].update(v)
        else:
            live[k] = v
    return cfg


def _drive(monkeypatch, db, trading, cfg, ctxs=None, kalshi=None,
           ts="2026-07-02T00:01:00Z", mode="prod"):
    monkeypatch.setattr(le, "utcnow_iso", lambda: ts)
    monkeypatch.setattr(le, "build_contexts",
                        lambda *a, **k: (ctxs if ctxs is not None else [_ctx()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: kalshi or _Kalshi())
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)
    return le.run_live_mm(mode=mode, db_path=db, cfg=cfg, verbose=False)


# --------------------------------------------------------------------------- #
# 1) Regression: all new flags off -> identical to current behavior.
# --------------------------------------------------------------------------- #
def test_all_flags_off_is_prod_neutral(monkeypatch, tmp_path):
    from kwt.db import init_db, connect
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    s = _drive(monkeypatch, db, trading, _cfg())      # stock config, nothing enabled
    # Unchanged two-sided quote; no new-lever orders.
    assert sorted(x[0] for x in trading.placed) == ["ask", "bid"]
    assert s["quoted_both"] == 1 and s["placed"] == 2 and s["blocked"] == 0
    conn = connect(db)
    kinds = {r[0] for r in conn.execute("SELECT DISTINCT kind FROM live_risk_events").fetchall()}
    conn.close()
    assert "resting_exit" not in kinds
    assert "metar_blackout" not in kinds


# --------------------------------------------------------------------------- #
# 2) Resting passive exits.
# --------------------------------------------------------------------------- #
def _held_long():
    # net +2 YES, cost basis $1.00 -> entry 0.50 YES.
    return [{"ticker": TICKER, "position_fp": "2.00",
             "market_exposure_dollars": "1.00", "realized_pnl_dollars": "0.0"}]


def test_resting_exit_off_places_nothing_extra(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading(positions=_held_long())
    # Held market is GATED (oi below min_open_interest) so the normal loop skips it.
    # Flatten explicitly off to isolate resting_exit behavior.
    _drive(monkeypatch, db, trading, _cfg(flatten={"enabled": False}), ctxs=[_ctx(oi=10.0)])
    assert trading.placed == []          # gated + resting_exit off -> nothing


def test_resting_exit_on_rests_one_reduce_only_maker(monkeypatch, tmp_path):
    from kwt.db import init_db, connect
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading(positions=_held_long())
    cfg = _cfg(resting_exit={"enabled": True, "scratch_ticks": 2},
               flatten={"enabled": False})
    _drive(monkeypatch, db, trading, cfg, ctxs=[_ctx(oi=10.0)])
    # Long +2 -> ONE post-only ask to shed, at entry(0.50)+2c = 0.52, never crossing
    # (0.52 > yes_bid 0.45).
    assert len(trading.placed) == 1
    side, price = trading.placed[0]
    assert side == "ask"
    assert abs(price - 0.52) < 1e-9
    assert price > 0.45                  # post-only: does not cross the bid
    conn = connect(db)
    n = conn.execute("SELECT COUNT(*) FROM live_risk_events WHERE kind='resting_exit'").fetchone()[0]
    conn.close()
    assert n == 1


def test_resting_exit_short_rests_a_bid(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    # net -2 YES (short), NO cost basis $1.00 -> YES-equiv entry = 1 - 0.50 = 0.50.
    positions = [{"ticker": TICKER, "position_fp": "-2.00",
                  "market_exposure_dollars": "1.00", "realized_pnl_dollars": "0.0"}]
    trading = _Trading(positions=positions)
    cfg = _cfg(resting_exit={"enabled": True, "scratch_ticks": 2},
               flatten={"enabled": False})
    _drive(monkeypatch, db, trading, cfg, ctxs=[_ctx(oi=10.0)])
    assert len(trading.placed) == 1
    side, price = trading.placed[0]
    assert side == "bid"                 # shed a short by buying YES back below entry
    assert abs(price - 0.48) < 1e-9      # 0.50 - 2c
    assert price < 0.55                  # post-only: does not cross the ask


def test_resting_exit_firewall_ignores_foreign_position(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    # A non-experiment (turnout) position must never be exited, even with the flag on.
    positions = [{"ticker": "KXMIDTERM-CONTROL", "position_fp": "5.00",
                  "market_exposure_dollars": "2.00", "realized_pnl_dollars": "0.0"}]
    trading = _Trading(positions=positions)
    cfg = _cfg(resting_exit={"enabled": True, "scratch_ticks": 2})
    # The only quotable ctx is flat (no position) and gated, so nothing to exit.
    _drive(monkeypatch, db, trading, cfg, ctxs=[_ctx(oi=10.0)])
    assert trading.placed == []          # firewall: foreign inventory untouched


def test_resting_exit_skipped_when_opposing_side_already_quoted(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading(positions=_held_long())
    cfg = _cfg(resting_exit={"enabled": True, "scratch_ticks": 2},
               flatten={"enabled": False})
    # oi high -> the market IS two-sided quoted (bid+ask). The ask is the opposing
    # (reducing) side, so the resting-exit pass must NOT add a second ask.
    _drive(monkeypatch, db, trading, cfg, ctxs=[_ctx(oi=500.0)])
    assert sorted(x[0] for x in trading.placed) == ["ask", "bid"]   # exactly the quote


# --------------------------------------------------------------------------- #
# 3) METAR requote blackout.
# --------------------------------------------------------------------------- #
def test_metar_blackout_suppresses_inside_window(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    cfg = _cfg(requote={"metar_blackout_seconds": 420})
    # :55 is 5 min (300s) before the top of the hour -> inside the ±420s window.
    s = _drive(monkeypatch, db, trading, cfg, ts="2026-07-02T00:55:00Z")
    assert s.get("blackout") is True
    assert trading.placed == []


def test_metar_blackout_allows_outside_window(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    cfg = _cfg(requote={"metar_blackout_seconds": 420})
    # :30 is far from any top-of-hour -> quotes normally.
    s = _drive(monkeypatch, db, trading, cfg, ts="2026-07-02T00:30:00Z")
    assert not s.get("blackout")
    assert s["quoted_both"] == 1


def test_metar_blackout_off_by_default(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    # seconds=0 (default) -> even at :55 we quote (no METAR blackout).
    s = _drive(monkeypatch, db, trading, _cfg(), ts="2026-07-02T00:55:00Z")
    assert not s.get("blackout")
    assert s["quoted_both"] == 1


def test_metar_blackout_helper():
    assert le._in_metar_blackout("2026-07-02T00:00:10Z", 420)    # just after :00
    assert le._in_metar_blackout("2026-07-02T00:56:00Z", 420)    # 4 min before :00
    assert not le._in_metar_blackout("2026-07-02T00:30:00Z", 420)
    assert not le._in_metar_blackout("2026-07-02T00:56:00Z", 0)  # disabled


# --------------------------------------------------------------------------- #
# 4) One-sided-plan contract (W3): place only the populated side.
# --------------------------------------------------------------------------- #
def _plan(new_bid, new_ask, quotable=True):
    return QuotePlan(new_bid, new_ask, 0.50, 0.0, quotable, None, 0.50)


def test_one_sided_ask_only(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    monkeypatch.setattr(le, "plan_quote", lambda *a, **k: _plan(None, 0.55))
    s = _drive(monkeypatch, db, trading, _cfg())
    assert [x[0] for x in trading.placed] == ["ask"]     # only the ask rested
    assert s["quoted_both"] == 0                          # not a two-sided quote


def test_one_sided_bid_only(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    monkeypatch.setattr(le, "plan_quote", lambda *a, **k: _plan(0.45, None))
    s = _drive(monkeypatch, db, trading, _cfg())
    assert [x[0] for x in trading.placed] == ["bid"]
    assert s["quoted_both"] == 0


def test_both_none_quotable_places_nothing(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    monkeypatch.setattr(le, "plan_quote", lambda *a, **k: _plan(None, None))
    _drive(monkeypatch, db, trading, _cfg())
    assert trading.placed == []                           # nothing to rest


def test_two_sided_plan_unchanged(monkeypatch, tmp_path):
    from kwt.db import init_db
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading()
    monkeypatch.setattr(le, "plan_quote", lambda *a, **k: _plan(0.45, 0.55))
    s = _drive(monkeypatch, db, trading, _cfg())
    assert sorted(x[0] for x in trading.placed) == ["ask", "bid"]
    assert s["quoted_both"] == 1


# --------------------------------------------------------------------------- #
# 5) Flatten log line (#1): the gate still wires through and a reduce runs.
# --------------------------------------------------------------------------- #
def test_flatten_still_wires_and_logs(monkeypatch, tmp_path, capsys):
    from kwt.db import init_db, connect
    db = str(tmp_path / "kwt.db")
    init_db(db)
    trading = _Trading(positions=_held_long())
    cfg = _cfg(flatten={"enabled": True})
    monkeypatch.setattr(le, "utcnow_iso", lambda: "2026-07-02T00:01:00Z")
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([_ctx()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi())
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)
    le.run_live_mm(mode="prod", db_path=db, cfg=cfg, verbose=True)
    # Long +2 -> reduce by resting the ask only (accumulating bid pulled).
    assert [x[0] for x in trading.placed] == ["ask"]
    out = capsys.readouterr().out
    assert "FLATTEN" in out
    conn = connect(db)
    n = conn.execute("SELECT COUNT(*) FROM live_risk_events WHERE kind='flatten'").fetchone()[0]
    conn.close()
    assert n >= 1
