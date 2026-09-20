"""End-to-end drive of run_live_mm over two dry-run cycles.

Cycle 1 rests a two-sided quote (dry_run: recorded, nothing sent) and caches it;
cycle 2 audits real trade prints against that resting quote. Asserts the new
columns/tables are populated: dual sim baselines, per-side sim, persisted
model_fair + mid_at_placement, and a coverage row per (ticker, cycle).
"""
from __future__ import annotations

import types

import kwt.live_engine as le
from kwt.config import load_config
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
    def __init__(self, prints):
        self._prints = prints

    def trades_since(self, ticker, since, **kw):
        return [t for t in self._prints if not since or t.get("created_time", "") > since]


def _print(price, count, t, tid):
    return {"yes_price_dollars": price, "count": count, "created_time": t, "trade_id": tid}


def _drive(monkeypatch, db, kalshi, ts, ctx_fn=_ctx):
    # Pin the cycle clock so interval boundaries are deterministic (the audit
    # windows fills/prints by created_time between placement ts and this ts).
    monkeypatch.setattr(le, "utcnow_iso", lambda: ts)
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([ctx_fn()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: kalshi)
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    cfg = load_config()
    return le.run_live_mm(mode="dry_run", db_path=db, cfg=cfg, verbose=False)


def test_two_cycles_audit_and_coverage(monkeypatch, tmp_path):
    db = str(tmp_path / "kwt.db")
    from kwt.db import init_db, connect
    init_db(db)

    # Cycle 1: no prior quote -> no audit row, but a quote is planned+cached and a
    # coverage row is written.
    k1 = _Kalshi([_print(0.50, 10, "2026-07-02T00:00:00Z", "seed")])
    s1 = _drive(monkeypatch, db, k1, ts="2026-07-02T00:01:00Z")
    assert s1["quoted_both"] == 1 and s1["audited"] == 0

    conn = connect(db)
    cov = conn.execute("SELECT rested_any, rested_both, mkt_bid, mkt_ask, model_fair "
                       "FROM mm_quote_log").fetchall()
    assert len(cov) == 1 and cov[0]["rested_any"] == 1 and cov[0]["rested_both"] == 1
    assert cov[0]["mkt_bid"] == 0.45 and cov[0]["mkt_ask"] == 0.55   # touch recorded
    conn.close()

    # Cycle 2: prints cross the resting bid (0.45) and ask (0.55) -> the audit
    # measures sim fills against them.
    prints = [_print(0.45, 100, "2026-07-02T00:05:00Z", "p1"),
              _print(0.55, 100, "2026-07-02T00:06:00Z", "p2")]
    s2 = _drive(monkeypatch, db, _Kalshi(prints), ts="2026-07-02T00:10:00Z")
    assert s2["audited"] == 1

    conn = connect(db)
    row = conn.execute(
        "SELECT rested_any, rested_bid, rested_ask, print_vol_bid, print_vol_ask, "
        "sim_fill_qty, sim_fill_qty_incl_adverse, sim_buy_yes, sim_buy_no, "
        "actual_fill_qty, model_fair, mid_at_placement FROM mm_fill_audit").fetchone()
    # size 1 per side, capped once per side -> 1 YES + 1 NO simulated.
    assert row["sim_buy_yes"] == 1 and row["sim_buy_no"] == 1
    assert row["sim_fill_qty"] == 2
    assert row["sim_fill_qty_incl_adverse"] >= row["sim_fill_qty"]  # pessimistic >= optimistic
    assert row["actual_fill_qty"] == 0                              # dry_run: no real fills
    assert row["rested_any"] == 1 and row["rested_bid"] == 1 and row["rested_ask"] == 1
    assert row["print_vol_bid"] == 100 and row["print_vol_ask"] == 100
    assert row["model_fair"] == 0.5                                 # persisted, not NULL
    assert row["mid_at_placement"] == 0.5                           # persisted from cycle 1
    conn.close()


def _ask_only_ctx():
    # Cheap YES longshot: the live treatment intentionally rests only the ask.
    return MarketCtx(
        ticker="KXHIGHNY-26JUL02-T120", city="nyc", target_date="2026-07-02",
        low=119, high=None, bucket_kind="above", horizon_days=1.0,
        yes_bid=0.02, yes_ask=0.06, no_bid=0.94, no_ask=0.98,
        last_price=0.035, open_interest=500.0, nwp=_NWPp(0.02))


def test_one_sided_quote_is_audited_with_side_specific_volume(monkeypatch, tmp_path):
    db = str(tmp_path / "kwt.db")
    from kwt.db import init_db, connect
    init_db(db)

    # Cycle 1 rests only the ask and seeds the public-trade watermark.
    _drive(monkeypatch, db,
           _Kalshi([_print(0.04, 10, "2026-07-02T00:00:00Z", "seed")]),
           ts="2026-07-02T00:01:00Z", ctx_fn=_ask_only_ctx)
    # Cycle 2 has public flow at the resting ask. There is no bid-side quote to
    # audit, so this must still produce a normal interval rather than disappear.
    s2 = _drive(monkeypatch, db,
                _Kalshi([_print(0.06, 100, "2026-07-02T00:05:00Z", "ask-print")]),
                ts="2026-07-02T00:10:00Z", ctx_fn=_ask_only_ctx)
    assert s2["audited"] == 1 and s2["quoted_both"] == 0

    conn = connect(db)
    row = conn.execute(
        "SELECT rested_any, rested_bid, rested_ask, print_vol, print_vol_bid, "
        "print_vol_ask, sim_buy_yes, sim_buy_no FROM mm_fill_audit").fetchone()
    assert row["rested_any"] == 1 and row["rested_bid"] == 0 and row["rested_ask"] == 1
    assert row["print_vol"] == 100 and row["print_vol_bid"] == 0
    assert row["print_vol_ask"] == 100
    assert row["sim_buy_yes"] == 0 and row["sim_buy_no"] == 1
    cov = conn.execute("SELECT SUM(rested_any), SUM(rested_both) FROM mm_quote_log").fetchone()
    assert cov[0] == 2 and cov[1] == 0
    conn.close()


def test_overlong_cycle_is_cancel_only(monkeypatch, tmp_path):
    db = str(tmp_path / "kwt.db")
    from kwt.db import init_db, connect
    init_db(db)
    times = iter(["2026-07-02T00:01:00Z", "2026-07-02T00:21:01Z"])
    end = "2026-07-02T00:21:01Z"
    monkeypatch.setattr(le, "utcnow_iso", lambda: next(times, end))
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([_ctx()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi([]))
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())

    summary = le.run_live_mm(mode="dry_run", db_path=db,
                             cfg=load_config(), verbose=False)
    assert summary["stale_cycle"] is True
    assert summary["placed"] == 0
    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) FROM mm_quote_log").fetchone()[0] == 0
    assert conn.execute("SELECT kind FROM live_risk_events ORDER BY id DESC LIMIT 1").fetchone()[0] == "stale_cycle"
    conn.close()


def _edge_ctx():
    # Cheap-longshot bucket whose best bid is already at the $0.01 floor (wide
    # enough to pass the min-spread filter, so we isolate the book-edge skip).
    return MarketCtx(
        ticker="KXHIGHNY-26JUL02-T120", city="nyc", target_date="2026-07-02",
        low=119, high=None, bucket_kind="above", horizon_days=1.0,
        yes_bid=0.01, yes_ask=0.06, no_bid=0.94, no_ask=0.99,
        last_price=0.02, open_interest=500.0, nwp=_NWPp(0.02))


class _NWPp:
    def __init__(self, p):
        self.p = p

    def p_bucket(self, low, high, blend_empirical=0.6):
        return self.p


def test_book_edge_quote_is_skipped(monkeypatch, tmp_path):
    db = str(tmp_path / "kwt.db")
    from kwt.db import init_db, connect
    init_db(db)
    monkeypatch.setattr(le, "utcnow_iso", lambda: "2026-07-02T00:01:00Z")
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([_edge_ctx()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi([]))
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    # Isolate the book-edge skip: this cheap-longshot fixture (mid ~0.035) would
    # otherwise route down the ask_only_below=0.12 path (rest the ask, drop the
    # bid) — that one-sided behavior is covered by the W3/W4 tests. Pin it off so
    # this test exercises the both-or-neither book-edge skip it's named for.
    cfg = load_config()
    cfg.raw["live"]["quote_overrides"]["ask_only_below"] = None
    s = le.run_live_mm(mode="dry_run", db_path=db, cfg=cfg, verbose=False)
    assert s["quoted_both"] == 0 and s["placed"] == 0    # never touched the book
    conn = connect(db)
    assert conn.execute("SELECT skip_reason FROM mm_quote_log").fetchone()[0] == "at_book_edge"
    conn.close()


class _PartialTrading:
    """place_orders=True; the bid raises (post-only reject), the ask rests. The
    engine must cancel the orphaned ask to preserve both-or-neither."""
    def __init__(self):
        self.place_orders = True
        self.authenticated = True
        self.canceled = []

    def get_balance(self):
        return {"balance_dollars": 20.0}

    def get_positions(self):
        return []

    def get_orders(self, status=None):
        return []

    def get_fills(self):
        return []

    def create_order(self, *, ticker, side, price, count, client_order_id,
                     post_only=True, **kw):
        from kwt.clients.kalshi_trading import KalshiTradingError
        if side == "bid":
            raise KalshiTradingError(400, "post only would cross")
        return {"order_id": "ask-1", "client_order_id": client_order_id,
                "fill_count": "0.00", "remaining_count": "1.00"}

    def cancel_order(self, oid):
        self.canceled.append(oid)
        return {"order_id": oid, "status": "canceled"}


def test_partial_pair_cancels_orphan(monkeypatch, tmp_path):
    db = str(tmp_path / "kwt.db")
    from kwt.db import init_db, connect
    init_db(db)
    trading = _PartialTrading()
    monkeypatch.setattr(le, "utcnow_iso", lambda: "2026-07-02T00:01:00Z")
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([_ctx()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi([]))
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)
    s = le.run_live_mm(mode="prod", db_path=db, cfg=load_config(), verbose=False)

    assert trading.canceled == ["ask-1"]        # the rested ask was cancelled
    assert s["quoted_both"] == 0                 # not a clean two-sided quote
    conn = connect(db)
    assert conn.execute("SELECT rested_both FROM mm_quote_log").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM live_orders WHERE status='canceled' AND reason='orphan_pair'"
    ).fetchone()[0] == 1
    conn.close()


class _CaptureTrading:
    """Successful two-sided placement that records the expiration_time sent."""
    def __init__(self):
        self.place_orders = True
        self.authenticated = True
        self.exp_times = []

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
        self.exp_times.append(expiration_time)
        return {"order_id": f"{side}-{ticker}", "client_order_id": client_order_id,
                "fill_count": "0.00", "remaining_count": "1.00"}

    def cancel_order(self, oid):
        return {"order_id": oid, "status": "canceled"}


def test_orders_carry_deadman_expiry(monkeypatch, tmp_path):
    db = str(tmp_path / "kwt.db")
    from kwt.db import init_db
    from kwt.live_engine import _epoch
    init_db(db)
    trading = _CaptureTrading()
    ts = "2026-07-02T00:01:00Z"
    monkeypatch.setattr(le, "utcnow_iso", lambda: ts)
    monkeypatch.setattr(le, "build_contexts", lambda *a, **k: ([_ctx()], {}))
    monkeypatch.setattr(le, "KalshiClient", lambda *a, **k: _Kalshi([]))
    monkeypatch.setattr(le, "OpenMeteoClient", lambda *a, **k: object())
    monkeypatch.setattr(le, "KalshiTradingClient", lambda *a, **k: trading)
    le.run_live_mm(mode="prod", db_path=db, cfg=load_config(), verbose=False)
    expected = _epoch(ts) + 1800    # config order_ttl_seconds default
    assert trading.exp_times and all(e == expected for e in trading.exp_times)


def test_own_prints_excluded_from_sim(monkeypatch, tmp_path):
    db = str(tmp_path / "kwt.db")
    from kwt.db import init_db, connect
    init_db(db)
    _drive(monkeypatch, db, _Kalshi([]), ts="2026-07-02T00:01:00Z")  # cycle 1

    # A print that would fill our bid, but its trade_id matches one of our own
    # recorded fills -> it must be excluded, leaving sim = 0.
    conn = connect(db)
    conn.execute("INSERT INTO live_fills (ts, fill_id, ticker, side, action, count, "
                 "price, fee, is_taker, created_time) VALUES "
                 "(?,?,?,?,?,?,?,?,?,?)",
                 ("2026-07-02T00:05:00Z", "mine1", "KXHIGHNY-26JUL02-B80", "yes",
                  "buy", 1, 0.45, 0.0, 0, "2026-07-02T00:05:00Z"))
    conn.commit(); conn.close()

    prints = [_print(0.45, 100, "2026-07-02T00:05:00Z", "mine1")]  # our own print
    _drive(monkeypatch, db, _Kalshi(prints), ts="2026-07-02T00:10:00Z")
    conn = connect(db)
    row = conn.execute("SELECT sim_fill_qty FROM mm_fill_audit").fetchone()
    assert row["sim_fill_qty"] == 0     # own print filtered out
    conn.close()


def test_capacity_blocked_pair_still_rests_the_risk_reducing_side(monkeypatch, tmp_path):
    # Book-level scenario: portfolio net inventory is pinned at max_net_inventory
    # (long across OTHER markets); this market itself is flat. The bid would add
    # inventory and is rightly vetoed, but the ask REDUCES book exposure and was
    # previously thrown away by the all-or-neither pair block. It must rest.
    db = str(tmp_path / "kwt.db")
    from kwt.db import init_db, connect
    from kwt.risk import LiveState
    init_db(db)

    def _pinned_reconcile(conn, trading, cfg, ts, event_of, prefixes, verbose, mid_of=None):
        lim = cfg.raw["live"]["risk"]["max_net_inventory"]
        s = LiveState(reconciled=True, funded_capital=20.0, balance=20.0)
        # Long book, from a market outside this cycle's quoting loop.
        s.net_by_market = {"KXHIGHCHI-26JUL02-B80": float(lim)}
        return s

    monkeypatch.setattr(le, "reconcile", _pinned_reconcile)
    k = _Kalshi([_print(0.50, 10, "2026-07-02T00:00:00Z", "seed")])
    s = _drive(monkeypatch, db, k, ts="2026-07-02T00:01:00Z")

    assert s["placed"] == 1        # the ask rested
    assert s["quoted_both"] == 0
    assert s["blocked"] == 1       # only the bid was blocked

    conn = connect(db)
    rows = conn.execute("SELECT side, status, reason FROM live_orders ORDER BY side").fetchall()
    by_side = {r["side"]: r for r in rows}
    assert by_side["bid"]["status"] == "blocked"
    assert by_side["bid"]["reason"] == "max_net_inventory"
    assert by_side["ask"]["status"] != "blocked"
    # No companion side may be discarded as paired_side_blocked in this scenario.
    assert not any(r["reason"] == "paired_side_blocked" for r in rows)
    conn.close()


def test_capital_blocked_pair_still_blocks_both_sides(monkeypatch, tmp_path):
    # max_capital_at_risk is direction-blind (an ask fill also costs capital),
    # so a capital-bound book must NOT get a one-sided rescue.
    db = str(tmp_path / "kwt.db")
    from kwt.db import init_db, connect
    from kwt.risk import LiveState
    init_db(db)

    def _capital_pinned(conn, trading, cfg, ts, event_of, prefixes, verbose, mid_of=None):
        s = LiveState(reconciled=True, funded_capital=20.0, balance=20.0)
        # Long book so the ask would otherwise qualify as risk-reducing.
        s.net_by_market = {"KXHIGHCHI-26JUL02-B80": 1.0}
        s.capital_at_risk = float(cfg.raw["live"]["risk"]["max_capital_at_risk"])
        return s

    monkeypatch.setattr(le, "reconcile", _capital_pinned)
    k = _Kalshi([_print(0.50, 10, "2026-07-02T00:00:00Z", "seed")])
    s = _drive(monkeypatch, db, k, ts="2026-07-02T00:01:00Z")

    assert s["placed"] == 0
    assert s["blocked"] == 2
    conn = connect(db)
    rows = conn.execute("SELECT status FROM live_orders").fetchall()
    assert all(r["status"] == "blocked" for r in rows) and len(rows) == 2
    conn.close()
