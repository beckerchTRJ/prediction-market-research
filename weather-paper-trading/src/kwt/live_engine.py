"""Live market-making execution loop (the real-money experiment harness).

One cycle of `kwt live-mm` does, in order:

  1. Build the same market contexts the paper harness uses (build_contexts).
  2. Reconcile balance / positions / resting orders / fills from Kalshi.
  3. Kill-switch check (daily loss, capital at risk). If tripped: cancel all
     resting orders, log, stop.
  4. Per market: record the fill audit (simulated 5%/2% fill qty vs the ACTUAL
     fills since our quote went up) — this is the data the experiment exists for.
  5. Cancel-and-replace: cancel our resting orders, then place fresh post-only
     quotes, each vetted (and scaled) by the RiskEngine.

Modes:
  dry_run  - default. No credentials required. Records the audit from public
             trade prints and records *planned* orders, but SENDS NOTHING.
  demo     - places real orders on Kalshi's demo environment (fake money).
  prod     - places real orders with REAL money. Requires explicit opt-in.
"""
from __future__ import annotations

import uuid
import copy
from datetime import datetime, timedelta

from .clients.kalshi import KalshiClient
from .clients.kalshi_trading import KalshiAuthError, KalshiTradingClient, KalshiTradingError
from .clients.openmeteo import OpenMeteoClient
from .collect import build_contexts
from .config import Config, DEFAULT_DB, load_config, utcnow_iso
from .db import connect, ensure_experiment_cohort, kv_get, kv_set, log_run
from .risk import LiveRiskLimits, LiveState, RiskEngine, release_order, reserve_order
from .strategies.market_making import plan_flatten, plan_quote, simulate_fills

KV = "mmlive"  # kv namespace for live resting quotes / watermarks (separate from paper)


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _dollars(p: dict, key: str) -> float:
    """A Kalshi portfolio money field in dollars. The `<key>_dollars` variant is
    already dollars; the bare `<key>` is integer cents. Mixing them (the old code
    divided market_exposure by 100 but not realized_pnl) overstated realized P&L
    100x and corrupted every P&L / kill-switch calculation.
    """
    dv = p.get(key + "_dollars")
    if dv is not None:
        return _f(dv)
    return _f(p.get(key)) / 100.0


def _log_risk(conn, ts, kind, ticker, detail):
    conn.execute("INSERT INTO live_risk_events (ts, kind, ticker, detail, cohort_id) "
                 "VALUES (?,?,?,?,?)", (ts, kind, ticker, detail,
                                         kv_get(conn, f"{KV}:cohort", "legacy")))


def _experiment_prefixes(cfg: Config) -> tuple[str, ...]:
    """Ticker prefixes that belong to THIS experiment (the weather daily-high
    series the maker quotes). This is the firewall: any account ticker NOT
    matching one of these prefixes — e.g. election/turnout positions and orders
    living in the same Kalshi account — is never summed into risk, never
    cancelled, and never ingested by the live-MM engine.
    """
    raw = (cfg.raw.get("live", {}) or {}).get("experiment_prefixes")
    if not raw:
        raw = ["KXHIGH"]
    return tuple(str(p) for p in raw)


def _in_experiment(ticker, prefixes: tuple[str, ...]) -> bool:
    return bool(ticker) and any(ticker.startswith(p) for p in prefixes)


def _rotate_by_city(ctxs: list, offset: int) -> list:
    """Rotate which city's contexts sort first, by CITY BLOCK (not raw index), so
    a binding per-cycle cap doesn't always starve the same tail cities in
    cities.yaml's fixed order. `offset` selects the starting city; the caller
    advances it once per cycle so coverage round-robins fairly over time.
    """
    order = []
    by_city = {}
    for c in ctxs:
        if c.city not in by_city:
            order.append(c.city)
            by_city[c.city] = []
        by_city[c.city].append(c)
    if not order:
        return ctxs
    start = offset % len(order)
    rotated_cities = order[start:] + order[:start]
    return [c for city in rotated_cities for c in by_city[city]]


def _quote_params(strategy_params: dict, overrides: dict | None,
                  limits: LiveRiskLimits) -> dict:
    """Build the live maker's quoting params: shared strategy params + live-only
    quote_overrides, with the inventory-skew scale anchored to the LIVE cap.

    plan_quote computes skew = inventory_skew*(net/max_inventory). The shared
    strategy default is max_inventory=60 (the paper size), but the live risk
    engine caps net inventory per market at `max_position_per_market` (3 in the
    $20 profile). Left at 60, skew <= 0.1c at the live cap and rounds to zero on
    the 1c grid, so the anti-adverse-selection lean never engages. Anchor it to
    the real per-market net cap so the skew reaches its full magnitude there.
    """
    qp = {**strategy_params, **(overrides or {})}
    qp["max_inventory"] = limits.max_position_per_market
    return qp


def _shed_stale_orders(conn, trading, ts, ticker, state) -> None:
    """Cancel whichever of this ticker's own cached resting orders are still live
    (per the ground-truth reconcile read), then clear the cache. Called whenever
    this cycle's plan for `ticker` is anything OTHER than "identical to what's
    already resting" -- gated, book-edge, no side populated, flatten, or a real
    price change all shed through here before anything new is placed.
    """
    for s in ("bid", "ask"):
        oid = kv_get(conn, f"{KV}:{s}_oid:{ticker}", "") or None
        if oid and oid in state.resting_order_ids:
            p = _maybe_float(kv_get(conn, f"{KV}:{s}:{ticker}", "")) or 0.0
            n = _maybe_float(kv_get(conn, f"{KV}:{s}_cnt:{ticker}", "")) or 0.0
            try:
                trading.cancel_order(oid)
            except (KalshiAuthError, KalshiTradingError) as e:
                _log_risk(conn, ts, "api_error", ticker, f"cancel stale {s}: {e}")
            else:
                state.open_order_count = max(0, state.open_order_count - 1)
                er = conn.execute(
                    "SELECT city,target_date FROM markets WHERE ticker=?", (ticker,)).fetchone()
                event = (er["city"], er["target_date"]) if er else None
                release_order(state, ticker, s, p, n, event)
                conn.execute(
                    "UPDATE live_orders SET status='canceled', updated_ts=? "
                    "WHERE order_id=? AND status='resting'", (ts, oid))
        kv_set(conn, f"{KV}:{s}_oid:{ticker}", "")


def _quote_age_seconds(created_ts: str | None, now_ts: str) -> float | None:
    """Age of a resting quote for diagnostics; bad exchange timestamps are NULL."""
    if not created_ts:
        return None
    try:
        age = (datetime.fromisoformat(now_ts.replace("Z", "+00:00"))
               - datetime.fromisoformat(created_ts.replace("Z", "+00:00"))).total_seconds()
        return max(0.0, age)
    except (TypeError, ValueError):
        return None


def _capture_queue_observations(conn, trading, state: LiveState, ts: str,
                                prefixes: tuple[str, ...]) -> int:
    """Persist one passive queue snapshot after a clean reconcile.

    Queue failures deliberately do not affect orders: this is measurement-only
    and an unavailable diagnostic must never turn into an execution outage.
    """
    if not state.reconciled or not getattr(trading, "authenticated", False):
        return 0
    fetch = getattr(trading, "get_queue_positions", None)
    if not callable(fetch):
        return 0                         # compatibility with old test/demo clients
    tickers = sorted({d["ticker"] for d in state.resting_orders.values()
                      if _in_experiment(d["ticker"], prefixes)})
    if not tickers:
        return 0                         # unscoped queue reads are rejected upstream
    try:
        positions = fetch(tickers)
    except Exception as e:  # diagnostic-only read must never interrupt execution
        _log_risk(conn, ts, "queue_error", None, f"queue positions failed: {e}")
        return 0
    rows = 0
    for q in positions:
        oid = q.get("order_id")
        detail = state.resting_orders.get(oid) if oid else None
        if not detail or not _in_experiment(detail["ticker"], prefixes):
            continue
        conn.execute(
            "INSERT OR IGNORE INTO live_queue_observations "
            "(ts,cohort_id,order_id,ticker,side,price,remaining_count,queue_ahead,quote_age_sec) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (ts, kv_get(conn, f"{KV}:cohort", "legacy"), oid, detail["ticker"],
             detail["side"], detail["price"], detail["remaining"],
             _f(q.get("queue_position_fp", q.get("queue_position"))),
             _quote_age_seconds(detail.get("created_time"), ts)))
        rows += 1
    return rows


def _settled_dates(conn, cohort_id: str) -> int:
    """Independent v2 date blocks with actual contracts and settlement P&L."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT m.target_date) n FROM live_fills f "
        "JOIN live_fill_markouts mo ON mo.fill_id=f.fill_id "
        "JOIN markets m ON m.ticker=f.ticker "
        "WHERE COALESCE(f.cohort_id,'legacy')=? AND COALESCE(f.count,0)>0 "
        "AND m.target_date IS NOT NULL AND mo.mo_settle IS NOT NULL", (cohort_id,)).fetchone()
    return int(row["n"] or 0)


def _active_live_cfg(conn, cfg: Config) -> tuple[dict, str, int, str]:
    """Select the immutable active policy, holding v2 until its checkpoint.

    ``experiment.successor`` is configuration for a future cohort, not a live
    policy change.  It only takes effect after the predecessor has the required
    number of independently settled target dates.
    """
    base = copy.deepcopy((cfg.raw.get("live", {}) or {}))
    exp = base.get("experiment", {}) or {}
    cohort_id = str(exp.get("cohort_id", "legacy"))
    successor = exp.get("successor", {}) or {}
    settled = _settled_dates(conn, cohort_id)
    required = int(successor.get("activate_after_settled_dates", 0) or 0)
    checkpoint_status = "collecting"
    if required and settled >= required:
        from .markout import net_markout_evidence
        checkpoint_status = net_markout_evidence(conn, cohort_id)["settlement"]["status"]
    if successor and required and settled >= required:
        if checkpoint_status == "futility":
            return base, cohort_id, settled, checkpoint_status
        next_id = str(successor["cohort_id"])
        overrides = successor.get("risk_overrides", {}) or {}
        base["risk"] = {**(base.get("risk", {}) or {}), **overrides}
        base["experiment"] = {k: v for k, v in successor.items()
                              if k not in ("risk_overrides", "activate_after_settled_dates")}
        base["experiment"].setdefault("primary_metric", "settlement_pnl")
        base["experiment"].setdefault("horizon_minutes", 60)
        return base, next_id, settled, checkpoint_status
    return base, cohort_id, settled, checkpoint_status


def _monitor_alert_once(conn, ts: str, cohort_id: str, key: str, detail: str) -> None:
    """Persist a non-blocking operational alert only once per cohort and condition."""
    marker = f"{KV}:monitor:{cohort_id}:{key}"
    if kv_get(conn, marker, ""):
        return
    _log_risk(conn, ts, "monitor_alert", None, detail)
    kv_set(conn, marker, ts)


def _monitor_experiment(conn, ts: str, cohort_id: str, ttl: int,
                        daily_max_loss: float) -> None:
    """Emit passive alerts; never alter the normal risk or order path."""
    if ttl > 0:
        age = conn.execute(
            "SELECT MAX(quote_age_sec) a FROM live_queue_observations WHERE cohort_id=?",
            (cohort_id,)).fetchone()["a"]
        if age is not None and float(age) > ttl:
            _monitor_alert_once(conn, ts, cohort_id, "quote_age",
                                f"queue snapshot age {float(age):.0f}s exceeded TTL {ttl}s")
    row = conn.execute(
        "SELECT COALESCE(SUM(sim_fill_qty),0) sim,COALESCE(SUM(actual_fill_qty),0) actual "
        "FROM mm_fill_audit WHERE cohort_id=? AND COALESCE(print_vol,0)>0", (cohort_id,)).fetchone()
    sim, actual = _f(row["sim"]), _f(row["actual"])
    if sim >= 20 and actual / sim < .5:
        _monitor_alert_once(conn, ts, cohort_id, "fill_ratio",
                            f"actual/sim fill ratio {actual / sim:.2f} below 0.50 on {sim:.1f} simulated contracts")
    pnl = conn.execute(
        "SELECT COALESCE(SUM(v.pnl_per_contract*f.count),0) pnl FROM mm_fill_pnl v "
        "JOIN live_fills f ON f.id=v.id WHERE COALESCE(f.cohort_id,'legacy')=? "
        "AND COALESCE(f.count,0)>0 AND v.pnl_per_contract IS NOT NULL", (cohort_id,)).fetchone()["pnl"]
    if float(pnl or 0) <= -abs(daily_max_loss):
        _monitor_alert_once(conn, ts, cohort_id, "settlement_drawdown",
                            f"settled P&L {float(pnl):.2f} breached -{abs(daily_max_loss):.2f}")


def reconcile(conn, trading: KalshiTradingClient, cfg: Config, ts: str,
              event_of: dict[str, tuple], prefixes: tuple[str, ...],
              verbose: bool, mid_of: dict[str, float] | None = None) -> LiveState:
    """Pull live account state from Kalshi into a LiveState. Empty if unauthed.

    FIREWALL: only positions/orders on the experiment's own markets (`prefixes`)
    are counted. The rest of the account is invisible here, so turnout positions
    cannot trip the kill switch, inflate the open-order count, or move P&L.
    """
    live_cfg = cfg.raw.get("live", {})
    funded = _f(live_cfg.get("funded_capital", 0.0))
    state = LiveState(funded_capital=funded)
    # YES-equivalent average entry price per experiment market, derived from the
    # reconciled cost basis. Consumed only by the gated resting-exit pass; empty
    # (and unused) otherwise, so this is prod-neutral when the flag is off.
    state.entry_by_market = {}
    try:
        bal = trading.get_balance()
        positions = trading.get_positions()
        orders = trading.get_orders(status="resting")
    except KalshiAuthError as e:
        if verbose:
            print(f"  reconcile: no live credentials ({e}); pure-simulation mode")
        return state
    except KalshiTradingError as e:
        _log_risk(conn, ts, "api_error", None, f"reconcile failed: {e}")
        if verbose:
            print(f"  reconcile: API error {e}")
        return state

    state.balance = _f(bal.get("balance")) / 100.0 if bal.get("balance") else _f(bal.get("balance_dollars"))
    # firewall: only OUR resting orders count toward the open-order cap
    state.open_order_count = sum(1 for o in orders
                                 if _in_experiment(o.get("ticker"), prefixes))
    state.resting_order_ids = {o.get("order_id") for o in orders
                               if o.get("order_id") and _in_experiment(o.get("ticker"), prefixes)}
    state.resting_orders = {}
    # Preload every exchange-confirmed resting obligation before quote iteration.
    # Prefer our own acknowledged order record for bid/ask semantics; portfolio
    # order payloads can encode legs as yes/no rather than book side.
    for o in orders:
        ticker = o.get("ticker")
        oid = o.get("order_id")
        if not oid or not _in_experiment(ticker, prefixes):
            continue
        local = conn.execute(
            "SELECT ts,side,price,count FROM live_orders WHERE order_id=? "
            "ORDER BY id DESC LIMIT 1", (oid,)).fetchone()
        raw_side = o.get("book_side") or o.get("side")
        side = raw_side if raw_side in ("bid", "ask") else (local["side"] if local else None)
        if o.get("yes_price_dollars") is not None:
            price = _f(o.get("yes_price_dollars"))
        elif o.get("yes_price") is not None:
            price = _f(o.get("yes_price")) / 100.0
        else:
            price = _f(local["price"] if local else 0)
        remaining = _f(o.get("remaining_count_fp", o.get("remaining_count")),
                       _f(local["count"] if local else 0))
        state.resting_orders[oid] = {
            "ticker": ticker, "side": side, "price": price, "remaining": remaining,
            "created_time": o.get("created_time") or (local["ts"] if local else None),
        }
        er = event_of.get(ticker)
        if er is None:
            mr = conn.execute(
                "SELECT city,target_date FROM markets WHERE ticker=?", (ticker,)).fetchone()
            er = (mr["city"], mr["target_date"]) if mr else None
        if side in ("bid", "ask"):
            reserve_order(state, ticker, side, price, remaining, er)
        else:
            # Unknown exchange encoding: reserve a full dollar and both directional
            # possibilities rather than allowing an unclassified order to create room.
            reserve_order(state, ticker, "bid", 0.5, remaining, er)
            reserve_order(state, ticker, "ask", 0.5, remaining, er)
    mid_of = mid_of or {}
    exp_realized = 0.0
    mtm = 0.0                                  # unrealized P&L of open positions
    for p in positions:
        ticker = p.get("ticker")
        if not _in_experiment(ticker, prefixes):
            continue                          # firewall: skip non-experiment positions
        # Kalshi returns the signed contract count as `position_fp` (a decimal
        # STRING, e.g. "-2.00"); the bare `position` key is absent, so reading it
        # made every position read as 0 (flat). That collapsed the mark-to-market
        # to -market_exposure and tripped the daily-loss kill switch on a phantom
        # loss. Prefer position_fp; keep bare `position` as a test/back-compat path.
        pos = _f(p.get("position_fp", p.get("position")))  # signed: + net YES, - net NO
        exposure = _dollars(p, "market_exposure")
        realized = _dollars(p, "realized_pnl")
        state.net_by_market[ticker] = pos
        # YES-equivalent entry: a long of `pos` YES cost `exposure`, so entry =
        # exposure/pos; a short holds |pos| NO at cost `exposure`, so its
        # YES-equivalent entry is 1 - exposure/|pos|. None when cost is unknown.
        if pos > 0:
            state.entry_by_market[ticker] = (exposure / pos) if exposure else None
        elif pos < 0:
            state.entry_by_market[ticker] = (1.0 - exposure / (-pos)) if exposure else None
        ev = event_of.get(ticker)
        if ev:
            state.net_by_event[ev] = state.net_by_event.get(ev, 0.0) + pos
            state.event_positions.setdefault(ev, []).append((pos, exposure))
        state.capital_at_risk += abs(exposure)
        exp_realized += realized
        mid = mid_of.get(ticker)
        if mid is not None:
            # mark to market: YES contracts worth `mid`, NO contracts worth `1-mid`
            value = pos * mid if pos >= 0 else (-pos) * (1.0 - mid)
            mtm += value - abs(exposure)
        conn.execute(
            "INSERT OR REPLACE INTO live_positions (ts, ticker, position, "
            "market_exposure, realized_pnl, fees_paid) VALUES (?,?,?,?,?,?)",
            (ts, ticker, int(pos), exposure, realized, _dollars(p, "fees_paid")))

    state.realized_pnl_total = exp_realized
    # Day P&L = realized + unrealized change since the first cycle of the UTC day,
    # scoped to the experiment ONLY. Buying raises cost AND mark-to-market value
    # together, so a fill is P&L-neutral here — unlike the old capital_at_risk
    # proxy, under which every fill read as intraday PROFIT and the daily-loss kill
    # switch could not fire until settlement. Falls back to realized-only when no
    # mids are supplied (mtm stays 0).
    state.day_realized_pnl = _rolling_pnl(conn, exp_realized + mtm, ts)
    # We got balance + positions + resting orders without an exception: the state
    # is trustworthy. On any read error above we returned early with reconciled
    # still False, and run_live_mm will refuse to quote (fail-closed).
    state.reconciled = True
    return state


def _in_nwp_blackout(ts: str, windows) -> bool:
    """True if this cycle's UTC time falls in a configured [start,end) HH:MM window
    (synoptic model-release blackout). Windows that don't cross midnight only."""
    if not windows:
        return False
    hhmm = ts[11:16]                       # "HH:MM" from ISO ...THH:MM:SSZ
    for w in windows:
        if len(w) == 2 and str(w[0]) <= hhmm < str(w[1]):
            return True
    return False


def _in_metar_blackout(ts: str, seconds: float) -> bool:
    """True within ±`seconds` of the top of the hour — the window when the hourly
    METAR observation prints (obs updates ~:53-:00) and station-informed flow
    re-prices ahead of our cycle-lagged fair. `seconds<=0` disables (default 0),
    so this is prod-neutral until explicitly turned on via requote.metar_blackout_seconds.
    """
    if not seconds or seconds <= 0:
        return False
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return False
    secs_into_hour = dt.minute * 60 + dt.second
    dist = min(secs_into_hour, 3600 - secs_into_hour)   # distance to nearest :00
    return dist <= seconds


def _gap_since_last_cycle(conn, ts: str, max_gap_seconds: float) -> bool:
    """True if the previous cycle ran longer than max_gap_seconds ago (host slept,
    cron stalled). Advances the watermark to `ts`, so the flag self-clears on the
    next normal cycle. First-ever cycle is never a gap.
    """
    last = kv_get(conn, f"{KV}:last_cycle_ts", "")
    kv_set(conn, f"{KV}:last_cycle_ts", ts)
    if not last:
        return False
    try:
        gap = (datetime.fromisoformat(ts.replace("Z", "+00:00"))
               - datetime.fromisoformat(last.replace("Z", "+00:00"))).total_seconds()
    except ValueError:
        return False
    return gap > max_gap_seconds


def _cycle_age_seconds(start: str, end: str) -> float:
    """Elapsed wall-clock seconds for one live cycle, or 0 for bad timestamps."""
    try:
        return max(0.0, (datetime.fromisoformat(end.replace("Z", "+00:00"))
                         - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds())
    except (TypeError, ValueError):
        return 0.0


def _clear_quote_state(conn) -> None:
    """Drop cached quote/watermark state at an immutable cohort cutover.

    Historical audit rows remain untouched, but the next cohort must not audit or
    keep an order created under the previous policy as if it were its own quote.
    """
    prefixes = ("mmlive:bid:%", "mmlive:ask:%", "mmlive:ts:%", "mmlive:wm:%",
                "mmlive:pmid:%", "mmlive:mfair:%", "mmlive:bid_oid:%",
                "mmlive:ask_oid:%", "mmlive:bid_cnt:%", "mmlive:ask_cnt:%")
    conn.execute("DELETE FROM kv WHERE " + " OR ".join("k LIKE ?" for _ in prefixes), prefixes)


def _iso_shift_hours(ts: str, hours: float) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) + timedelta(hours=hours)
    return dt.isoformat().replace("+00:00", "Z")


def _rolling_pnl(conn, equity_now: float, ts: str, window_hours: float = 24.0) -> float:
    """Experiment equity change over a TRAILING window, not a calendar-day reset.

    Records this cycle's equity, then baselines against the most recent checkpoint
    at least `window_hours` old (or the earliest checkpoint when the history is
    shorter). A UTC-midnight baseline split an evening's loss (22Z-03Z straddles
    the reset) across two sub-limit "days" so the daily-loss kill switch never
    fired; a rolling window measures the drawdown as one continuous session.
    """
    conn.execute("INSERT OR REPLACE INTO live_equity (ts, equity) VALUES (?,?)",
                 (ts, equity_now))
    cutoff = _iso_shift_hours(ts, -window_hours)
    row = conn.execute("SELECT equity FROM live_equity WHERE ts <= ? "
                       "ORDER BY ts DESC LIMIT 1", (cutoff,)).fetchone()
    if row is None:
        row = conn.execute("SELECT equity FROM live_equity ORDER BY ts ASC LIMIT 1").fetchone()
    base = _f(row["equity"], equity_now) if row is not None else equity_now
    return equity_now - base


def _ingest_fills(conn, trading: KalshiTradingClient, ts: str,
                  prefixes: tuple[str, ...], verbose: bool) -> None:
    """Pull recent fills into live_fills (idempotent via UNIQUE fill_id).

    FIREWALL: only fills on the experiment's own markets are stored; account
    fills elsewhere (turnout etc.) are ignored so they never pollute the fill
    audit or participation numbers.
    """
    try:
        fills = trading.get_fills()
    except (KalshiAuthError, KalshiTradingError):
        return
    for f in fills:
        if not _in_experiment(f.get("ticker"), prefixes):
            continue
        # Kalshi fills use decimal-STRING fields (`count_fp`, `yes_price_dollars`,
        # `fee_cost`); the bare `count`/`yes_price`/`fee` keys are absent, so the
        # first cut stored 0 for every quantity/price and the participation metric
        # (the whole point of this experiment) read 0%. Prefer the real fields;
        # keep the bare (cents) names as a back-compat/test fallback.
        count = int(_f(f.get("count_fp", f.get("count"))))
        if f.get("yes_price_dollars") is not None:
            price = _f(f.get("yes_price_dollars"))
        elif f.get("yes_price"):
            price = _f(f.get("yes_price")) / 100.0
        else:
            price = _f(f.get("price"))
        fee = _f(f.get("fee_cost")) if f.get("fee_cost") is not None else _f(f.get("fee")) / 100.0
        conn.execute(
            "INSERT OR IGNORE INTO live_fills (ts, fill_id, order_id, client_order_id, "
            "ticker, side, action, book_side, count, price, fee, is_taker, created_time,cohort_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,COALESCE((SELECT cohort_id FROM live_orders "
            "WHERE order_id=? ORDER BY id DESC LIMIT 1),'legacy'))",
            (ts, f.get("trade_id") or f.get("fill_id"), f.get("order_id"),
             f.get("client_order_id"), f.get("ticker"), f.get("side"),
             # book_side ('bid'/'ask') is the reliable fill direction; (side, action)
             # can't distinguish our bid- from ask-fills (an ask fill is (no, sell)).
             f.get("action"), f.get("book_side"), count, price, fee,
             1 if f.get("is_taker") else 0, f.get("created_time"), f.get("order_id")))


def _actual_fills(conn, ticker: str, start: str, end: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(count),0) c FROM live_fills WHERE ticker=? "
        "AND created_time > ? AND created_time <= ?", (ticker, start, end)).fetchone()
    return _f(row["c"])


def _actual_fills_side(conn, ticker: str, start: str, end: str) -> tuple[float, float]:
    """Real fills in the interval split by which of OUR orders they hit (bid vs
    ask), attributed via live_fills.order_id -> live_orders.side. Adverse
    selection is directional, so a symmetric total would hide it. (0,0) in
    dry_run — no real order_ids exist to join.
    """
    rows = conn.execute(
        "SELECT lo.side s, COALESCE(SUM(lf.count),0) c FROM live_fills lf "
        "JOIN live_orders lo ON lo.order_id = lf.order_id "
        "WHERE lf.ticker=? AND lf.order_id IS NOT NULL "
        "AND lf.created_time > ? AND lf.created_time <= ? GROUP BY lo.side",
        (ticker, start, end)).fetchall()
    bid = ask = 0.0
    for r in rows:
        if r["s"] == "bid":
            bid = _f(r["c"])
        elif r["s"] == "ask":
            ask = _f(r["c"])
    return bid, ask


def _own_trade_ids(conn, ticker: str, start: str, end: str) -> set:
    """trade_ids of our OWN fills in the interval (== the print's trade_id)."""
    rows = conn.execute(
        "SELECT fill_id FROM live_fills WHERE ticker=? "
        "AND created_time > ? AND created_time <= ?", (ticker, start, end)).fetchall()
    return {r["fill_id"] for r in rows if r["fill_id"]}


def _log_quote(conn, ts, mode, c, net_yes, *, rested_both, skip_reason, plan) -> None:
    """One coverage row per (ticker, cycle), including successful one-sided rests.

    ``skip_reason=''`` is the existing success marker, so it also identifies a
    one-sided quote when ``rested_both=0``. This keeps all existing call sites
    correct while adding the broader ``rested_any`` denominator.
    """
    rested_any = 1 if rested_both or (skip_reason or "") == "" else 0
    conn.execute(
        "INSERT OR REPLACE INTO mm_quote_log (ts, mode, ticker, rested_any, rested_both, "
        "skip_reason, bid, ask, model_fair, market_mid, mkt_bid, mkt_ask, net_yes,cohort_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ts, mode, c.ticker, rested_any, rested_both, skip_reason or "",
         (plan.new_bid if plan else None), (plan.new_ask if plan else None),
         (plan.model_fair if plan else None), c.yes_mid, c.yes_bid, c.yes_ask, net_yes,
         kv_get(conn, f"{KV}:cohort", "legacy")))


def _flatten_market(conn, trading, ts, mode, c, net_yes, flat, state, limits,
                    flatten_params, order_expiry, summary, verbose) -> None:
    """Gated inventory exit for ONE market (Fable §3). Rests only the reducing side
    at the touch (post-only) and pulls the accumulating side, so the maker sheds a
    one-sided position instead of holding it to settlement. A gated ``cross`` uses
    a one-contract, one-tick-bounded IOC; reduce_only remains post-only at the touch.

    A reducing order strictly shrinks |net|, so it is NOT gated by the per-market
    position cap (that both-or-neither trap is exactly what starved the reducing
    side on day 1); only open-order room is respected.
    """
    side = flat.reduce_side
    if flat.regime == "cross" and flatten_params.get("cross_enabled", False):
        cooldown = int(flatten_params.get("cross_cooldown_seconds", 600))
        last = kv_get(conn, f"{KV}:cross_ts:{c.ticker}", "") or ""
        if last:
            try:
                if _epoch(ts) - _epoch(last) < cooldown:
                    _log_quote(conn, ts, mode, c, net_yes, rested_both=0,
                               skip_reason="flatten_cross_cooldown", plan=None)
                    kv_set(conn, f"{KV}:bid:{c.ticker}", "")
                    kv_set(conn, f"{KV}:ask:{c.ticker}", "")
                    return
            except ValueError:
                pass
        if c.yes_bid is None or c.yes_ask is None:
            _log_quote(conn, ts, mode, c, net_yes, rested_both=0,
                       skip_reason="flatten_no_book", plan=None)
            kv_set(conn, f"{KV}:bid:{c.ticker}", "")
            kv_set(conn, f"{KV}:ask:{c.ticker}", "")
            return
        ticks = int(flatten_params.get("cross_max_slippage_ticks", 1))
        price = (max(0.01, c.yes_bid - ticks * .01) if side == "ask"
                 else min(0.99, c.yes_ask + ticks * .01))
        count = min(int(abs(net_yes)), int(limits.max_order_size),
                    int(flatten_params.get("cross_max_contracts_per_cycle", 1)))
        if count > 0:
            oid = _place(conn, trading, ts, mode, c.ticker, side, price, count, verbose,
                         post_only=False, time_in_force="immediate_or_cancel",
                         reason=f"flatten_cross:{flat.reason}")
            if oid is not None:
                summary["placed"] += 1
                kv_set(conn, f"{KV}:cross_ts:{c.ticker}", ts)
                _log_risk(conn, ts, "flatten_cross", c.ticker,
                          f"{flat.reason}: {side}@{price:.2f} count={count} net={net_yes:.0f}")
        _log_quote(conn, ts, mode, c, net_yes, rested_both=0,
                   skip_reason="flatten_cross", plan=None)
        kv_set(conn, f"{KV}:bid:{c.ticker}", "")
        kv_set(conn, f"{KV}:ask:{c.ticker}", "")
        return
    price = (c.yes_ask if side == "ask" else c.yes_bid)
    if price is None:
        price = c.last_price
    if flat.regime == "cross":
        _log_risk(conn, ts, "flatten_cross_recommended", c.ticker,
                  f"{flat.reason}: net={net_yes:.0f} — taker exit deferred, resting reduce")
    if price is None or price <= 0.01 or price >= 0.99:
        _log_quote(conn, ts, mode, c, net_yes, rested_both=0,
                   skip_reason="flatten_no_book", plan=None)
    elif state.open_order_count <= limits.max_open_orders - 1:
        oid = _place(conn, trading, ts, mode, c.ticker, side, price,
                     limits.max_order_size, verbose, expiration_time=order_expiry)
        if oid is not None:
            summary["placed"] += 1
            state.open_order_count += 1
            kv_set(conn, f"{KV}:{side}_oid:{c.ticker}", oid or "")
        _log_risk(conn, ts, "flatten", c.ticker,
                  f"{flat.regime} {side}@{price:.2f} net={net_yes:.0f}")
        if verbose:
            print(f"  FLATTEN {c.ticker}: {flat.regime} rest {side}@{price:.2f} "
                  f"net={net_yes:.0f}")
        _log_quote(conn, ts, mode, c, net_yes, rested_both=0,
                   skip_reason=f"flatten_{flat.regime}", plan=None)
    # One-sided this cycle: clear both watermarks so the next cycle's two-sided
    # audit doesn't fire on a stale pair.
    kv_set(conn, f"{KV}:bid:{c.ticker}", "")
    kv_set(conn, f"{KV}:ask:{c.ticker}", "")


def _rest_exit(conn, trading, ts, mode, c, net_yes, entry, state, limits,
               order_expiry, scratch_ticks, summary, verbose) -> None:
    """Rest ONE post-only, reduce-only passive exit for a held market (W4 #5).

    Long (net YES>0): rest an ASK at entry+scratch_ticks to shed above cost.
    Short (net YES<0): rest a BID at entry-scratch_ticks to buy YES back below the
    YES-equivalent entry. The order strictly shrinks |net| (size capped at |net|),
    so it can never increase inventory or breach a directional cap. Skipped if it
    would cross the book (a passive exit that crosses is a taker, not a maker) or
    if there is no open-order room.
    """
    long = net_yes > 0
    side = "ask" if long else "bid"
    if entry is None:                        # no cost basis reconciled: fall back to mid
        entry = c.yes_mid if c.yes_mid is not None else c.last_price
    if entry is None:
        return
    tick = scratch_ticks * 0.01
    price = round(entry + tick, 2) if long else round(entry - tick, 2)
    price = min(0.98, max(0.02, price))
    # Post-only: an exit that crosses the resting book would execute as a taker.
    if long and c.yes_bid is not None and price <= c.yes_bid:
        return
    if not long and c.yes_ask is not None and price >= c.yes_ask:
        return
    # Reduce-only: never rest more than the position we hold, so a fill cannot flip
    # |net| upward. Order size is also bounded by the risk cap.
    count = min(int(limits.max_order_size), int(abs(net_yes)))
    if count < 1:
        return
    if state.open_order_count > limits.max_open_orders - 1:
        return
    oid = _place(conn, trading, ts, mode, c.ticker, side, price, count, verbose,
                 expiration_time=order_expiry)
    if oid is not None:
        summary["placed"] += 1
        state.open_order_count += 1
        _log_risk(conn, ts, "resting_exit", c.ticker,
                  f"{side}@{price:.2f} net={net_yes:.0f} entry={entry:.2f}")
        if verbose:
            print(f"  RESTING EXIT {c.ticker}: rest {side}@{price:.2f} net={net_yes:.0f}")


def run_live_mm(mode: str = "dry_run", db_path=DEFAULT_DB, cfg: Config | None = None,
                verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    conn = connect(db_path)
    ts = utcnow_iso()
    live_cfg, selected_cohort_id, v2_settled_dates, checkpoint_status = _active_live_cfg(conn, cfg)
    policy_cfg = Config(raw={**cfg.raw, "live": live_cfg}, cities=cfg.cities)
    exp_cfg = live_cfg.get("experiment", {}) or {}
    cohort_id = selected_cohort_id
    previous_cohort = kv_get(conn, f"{KV}:cohort", "") or ""
    cohort_changed = previous_cohort not in ("", "legacy", cohort_id)
    frozen = {"quote_overrides": live_cfg.get("quote_overrides", {}),
              "flatten": live_cfg.get("flatten", {}), "risk": live_cfg.get("risk", {}),
              "live_cities": sorted(k for k, v in policy_cfg.cities.items()
                                    if v.get("live_enabled", True))}
    ensure_experiment_cohort(
        conn, cohort_id, frozen,
        primary_metric=str(exp_cfg.get("primary_metric", "net_markout")),
        horizon_minutes=int(exp_cfg.get("horizon_minutes", 60)), activated_ts=ts)
    kv_set(conn, f"{KV}:cohort", cohort_id)
    params = policy_cfg.strategies.get("market_making", {})
    # Live-only quoting overrides (e.g. near_decided_band) layered on the shared MM
    # params, so the live maker can be more conservative than the paper backtest
    # without changing paper behavior.
    limits = LiveRiskLimits.from_cfg(live_cfg)
    quote_params = _quote_params(params, live_cfg.get("quote_overrides", {}), limits)
    # Dead-man's switch: every resting order carries an exchange-side expiry so a
    # stalled loop (crash, sleep, cron disabled) can't leave stale quotes live to
    # be picked off. Refreshed each cycle; 0 disables. Default 30 min = 3 cron
    # intervals of slack before the exchange auto-cancels on its own.
    ttl = int(live_cfg.get("order_ttl_seconds", 1800) or 0)
    # Filled from the post-reconcile clock below. Using the cycle-start `ts`
    # here could submit an already-expired order after a slow context build.
    order_expiry = None
    # Flatten config is read up here (not just in the quote loop) so the kill switch
    # can run the flatten ladder instead of freezing and holding toxic inventory.
    flatten_cfg = live_cfg.get("flatten", {}) or {}
    flatten_on = bool(flatten_cfg.get("enabled", False))
    flatten_params = {**quote_params, **flatten_cfg}
    # Resting passive exits (gated): rest a single reduce-only maker at entry±ticks
    # on held markets we are NOT otherwise working out this cycle. Default off.
    # CAVEAT if ever enabled alongside the price-unchanged fast path (see the main
    # loop): a KEPT (unchanged) quote writes no live_orders row this cycle, so the
    # "already have an opposing resting order" check below could miss it and layer
    # a redundant order. Not addressed here since resting_exit defaults off.
    resting_exit_cfg = live_cfg.get("resting_exit", {}) or {}
    resting_exit_on = bool(resting_exit_cfg.get("enabled", False))
    scratch_ticks = _f(resting_exit_cfg.get("scratch_ticks", 2), 2)
    # Requote cadence levers (gated). obs_triggered would force a stale market to
    # re-quote when a newer station obs arrived; the live loop already CANCELS AND
    # REPLACES every resting quote each cycle (no unchanged-skip state exists), so
    # every cycle is already a fresh quote and this flag is an inert, no-op-safe
    # hook — it changes nothing while on, and invents no staleness state that could
    # perturb prod when off. metar_blackout_seconds is handled in the gate above.
    requote_cfg = live_cfg.get("requote", {}) or {}
    obs_triggered = bool(requote_cfg.get("obs_triggered", False))  # noqa: F841 (documented no-op hook)
    risk = RiskEngine(limits)
    # The live maker rests orders capped to max_order_size (risk), NOT the paper
    # `quote_size`. The fill audit's "sim" baseline must simulate that SAME size,
    # or actual/sim compares a 1-contract reality against a 5-contract simulation
    # and understates the true fill rate. Everything else (participation_rate,
    # spreads, skip logic) stays identical to paper.
    audit_params = dict(params)
    audit_params["quote_size"] = limits.max_order_size
    # The live book rests exactly ONE order per side, so the sim baseline must fill
    # at most once per side per interval — otherwise it over-counts fills the real
    # book cannot produce and biases actual/sim low.
    audit_params["max_fills_per_side"] = 1
    # Pessimistic baseline: a resting order cannot dodge a sweep through its quote.
    adverse_params = dict(audit_params)
    adverse_params["count_adverse_as_fills"] = True

    kalshi = KalshiClient()          # read-only public data (markets, prints)
    om = OpenMeteoClient()
    trading = KalshiTradingClient(mode=mode)

    prefixes = _experiment_prefixes(policy_cfg)   # firewall: MM only ever touches these
    if cohort_changed:
        # A cohort id is an immutable policy boundary, not a cosmetic label. Stop
        # the old policy's experiment orders before v2 can place anything, and
        # discard its quote watermarks so v2 cannot audit or keep them as its own.
        try:
            _cancel_all_resting(conn, trading, ts, prefixes, verbose, raise_on_error=True)
        except KillSwitchError:
            conn.commit()
            conn.close()
            raise
        _clear_quote_state(conn)
        _log_risk(conn, ts, "cohort_cutover", None,
                  f"{previous_cohort} -> {cohort_id}; old quotes canceled, quote state cleared")
        conn.commit()
    ctxs, counts = build_contexts(conn, policy_cfg, kalshi, om, ts, verbose)
    # New cities collect paper evidence immediately but cannot receive live orders
    # until their 14-day readiness review flips live_enabled in cities.yaml.
    ctxs = [c for c in ctxs if (policy_cfg.cities.get(c.city, {}) or {}).get("live_enabled", True)]
    city_rotation = int(kv_get(conn, f"{KV}:city_rotation", "0") or 0)
    ctxs = _rotate_by_city(ctxs, city_rotation)
    kv_set(conn, f"{KV}:city_rotation", city_rotation + 1)
    event_of = {c.ticker: (c.city, c.target_date) for c in ctxs}
    mid_of = {c.ticker: c.yes_mid for c in ctxs if c.yes_mid is not None}

    _ingest_fills(conn, trading, ts, prefixes, verbose)
    state = reconcile(conn, trading, policy_cfg, ts, event_of, prefixes, verbose, mid_of)
    # Complete any now-observable 15/30/60-minute horizons on every cycle,
    # including cancel-only/blackout/kill cycles that return before quote placement.
    from .markout import compute_markouts
    compute_markouts(conn, ts=ts)

    summary = {"ts": ts, "mode": mode, "contexts": len(ctxs),
               "audited": 0, "planned": 0, "placed": 0, "blocked": 0,
               "quoted_both": 0, "killed": False,
               "authenticated": trading.authenticated, "stale_cycle": False,
               "cohort_cutover": cohort_changed, "v2_settled_dates": v2_settled_dates,
               "checkpoint_status": checkpoint_status, "queue_observations": 0}

    # FAIL-CLOSED: an authenticated maker that could not fully read its account
    # (auth/API error mid-session) is blind — inventory and risk read as zero, so
    # quoting would rebuild the whole book as if flat and can breach caps. Cancel
    # whatever we can and place NOTHING until a clean reconcile. Unauthenticated
    # dry-run is exempt (no account to read; it sends nothing anyway).
    if trading.authenticated and not state.reconciled:
        summary["killed"] = True
        summary["reconcile_failed"] = True
        _log_risk(conn, ts, "reconcile_failed", None,
                  "incomplete account read — cancel-only, no quotes placed")
        _cancel_all_resting(conn, trading, ts, prefixes, verbose)
        conn.commit()
        log_run(conn, "live-mm", _jdump(summary))
        conn.commit()
        conn.close()
        if verbose:
            print("RECONCILE FAILED: blind to account — cancelled resting orders, no quotes.")
        return summary

    # A successful reconciliation is the sole precondition for queue telemetry.
    # This call is intentionally before the cancellation-only gates: a blackout
    # cycle still contributes a useful final observation of the resting queue.
    summary["queue_observations"] = _capture_queue_observations(
        conn, trading, state, ts, prefixes)

    # A statistically pre-registered futility result stops the experiment before
    # it can silently roll into the higher-capacity successor cohort.
    if checkpoint_status == "futility":
        summary["stopped_for_futility"] = True
        _monitor_alert_once(conn, ts, cohort_id, "futility", "settlement profitability gate: futility")
        _cancel_all_resting(conn, trading, ts, prefixes, verbose)
        conn.commit()
        log_run(conn, "live-mm", _jdump(summary))
        conn.commit()
        conn.close()
        if verbose:
            print("FUTILITY STOP: settlement profitability gate failed — cancelled resting orders.")
        return summary

    # STALE-CYCLE: build_contexts and reconciliation happen before this check so
    # the account is still brought current, but an overlong cycle never places
    # quotes whose exchange expiry was computed from stale start-of-cycle time.
    cycle_completed_ts = utcnow_iso()
    cycle_age = _cycle_age_seconds(ts, cycle_completed_ts)
    summary["cycle_age_seconds"] = round(cycle_age, 1)
    max_gap = _f(live_cfg.get("max_gap_seconds", 1200.0), 1200.0)
    if max_gap and cycle_age > max_gap:
        summary["stale_cycle"] = True
        _log_risk(conn, ts, "stale_cycle", None,
                  f"cycle age {cycle_age:.0f}s exceeded {max_gap:.0f}s — cancel-only")
        _cancel_all_resting(conn, trading, ts, prefixes, verbose)
        conn.commit()
        log_run(conn, "live-mm", _jdump(summary))
        conn.commit()
        conn.close()
        if verbose:
            print(f"STALE CYCLE: age {cycle_age:.0f}s > {max_gap:.0f}s; "
                  "cancelled resting orders, no quotes.")
        return summary

    # POST-GAP: the first cycle after a long gap (laptop wake / cron stall) must
    # reconcile + cancel only. Re-quoting off pre-gap state before a second cycle
    # validates the world is how the day-1 wake got picked off. Self-clears next cycle.
    if max_gap and _gap_since_last_cycle(conn, ts, max_gap):
        summary["post_gap"] = True
        _log_risk(conn, ts, "post_gap", None,
                  "first cycle after a long gap — cancel-only, no quotes placed")
        _cancel_all_resting(conn, trading, ts, prefixes, verbose)
        conn.commit()
        log_run(conn, "live-mm", _jdump(summary))
        conn.commit()
        conn.close()
        if verbose:
            print("POST-GAP: cancelled resting orders, no quotes (revalidating after a gap).")
        return summary

    # NWP BLACKOUT: around synoptic model releases, informed counterparties re-price
    # minutes ahead of our cycle-lagged fair. Cancel and sit out these UTC windows.
    blackout = live_cfg.get("nwp_blackout_windows", []) or []
    if _in_nwp_blackout(ts, blackout):
        summary["blackout"] = True
        _log_risk(conn, ts, "nwp_blackout", None,
                  "synoptic release window — cancel-only, no quotes placed")
        _cancel_all_resting(conn, trading, ts, prefixes, verbose)
        conn.commit()
        log_run(conn, "live-mm", _jdump(summary))
        conn.commit()
        conn.close()
        if verbose:
            print("NWP BLACKOUT: cancelled resting orders, no quotes (model-release window).")
        return summary

    # METAR BLACKOUT (gated, requote.metar_blackout_seconds; default 0 = off): the
    # hourly station observation prints around the top of the hour and moves the
    # informed fair before our next cycle. Sit out ±window, same cancel-only shape
    # as the NWP blackout above.
    metar_secs = _f((live_cfg.get("requote", {}) or {})
                    .get("metar_blackout_seconds", 0.0), 0.0)
    if metar_secs and _in_metar_blackout(ts, metar_secs):
        summary["blackout"] = True
        _log_risk(conn, ts, "metar_blackout", None,
                  "top-of-hour METAR print window — cancel-only, no quotes placed")
        _cancel_all_resting(conn, trading, ts, prefixes, verbose)
        conn.commit()
        log_run(conn, "live-mm", _jdump(summary))
        conn.commit()
        conn.close()
        if verbose:
            print("METAR BLACKOUT: cancelled resting orders, no quotes (obs-print window).")
        return summary

    # Orders expire from the time they are actually placed, not from the cycle's
    # initial timestamp. The stale-cycle guard above prevents this from being used
    # after a slow run has already exceeded the TTL/cadence safety envelope.
    order_expiry = (_epoch(cycle_completed_ts) + ttl) if ttl > 0 else None

    kill = risk.check_kill(state)
    if kill:
        summary["killed"] = True
        _log_risk(conn, ts, "kill_switch", None, kill)
        _cancel_all_resting(conn, trading, ts, prefixes, verbose)
        state.open_order_count = 0
        # FLATTEN, don't freeze: a kill switch that cancels quotes but HOLDS the
        # toxic inventory to settlement is a loss-MAXIMIZER (Fable §1e). When the
        # flatten ladder is enabled, run one reducing pass over every held market;
        # otherwise fall back to the old freeze-and-hold. Never place NEW two-sided
        # quotes under a kill.
        if flatten_on:
            for c in ctxs:
                net_yes = state.net_by_market.get(c.ticker, 0.0)
                if abs(net_yes) < 1:
                    continue
                flat = plan_flatten(c, net_yes, flatten_params)
                if flat.regime == "normal":
                    flat.regime, flat.reduce_side = "reduce_only", (
                        "ask" if net_yes > 0 else "bid")
                _flatten_market(conn, trading, ts, mode, c, net_yes, flat, state,
                                limits, flatten_params, order_expiry, summary, verbose)
        conn.commit()
        log_run(conn, "live-mm", _jdump(summary))
        conn.commit()
        conn.close()
        if verbose:
            action = "flattening held inventory" if flatten_on else "no quotes placed"
            print(f"KILL SWITCH: {kill} — cancelled resting orders, {action}.")
        return summary

    # NOTE: no blanket cancel-all here. Cancellation is now per-ticker and
    # price-conditional (see the price-unchanged fast path and
    # _shed_stale_orders below) so an unchanged quote keeps its exchange queue
    # position instead of being sent to the back of the line every cycle.
    # state.open_order_count stays at its reconciled (ground-truth) value.
    quote_size = params.get("quote_size", 5)
    for c in ctxs:
        net_yes = state.net_by_market.get(c.ticker, 0.0)

        # --- 1) fill audit for the interval the prior quote was resting -----
        prev_bid = _maybe_float(kv_get(conn, f"{KV}:bid:{c.ticker}", ""))
        prev_ask = _maybe_float(kv_get(conn, f"{KV}:ask:{c.ticker}", ""))
        wm = kv_get(conn, f"{KV}:wm:{c.ticker}", "") or ""
        interval_start = kv_get(conn, f"{KV}:ts:{c.ticker}", "") or ts
        if prev_bid is not None or prev_ask is not None:
            trades = kalshi.trades_since(c.ticker, wm)
            # Exclude OUR OWN fills from the tape: once live, our fills print
            # publicly and — being inside our own band — would inflate BOTH sim
            # and actual and drag the ratio toward 1. live_fills.fill_id == the
            # print's trade_id, so drop them exactly.
            own = _own_trade_ids(conn, c.ticker, interval_start, ts)
            tape = [t for t in trades if t.get("trade_id") not in own] if own else trades
            sim = simulate_fills(prev_bid, prev_ask, tape, net_yes, audit_params, wm)
            sim_adv = simulate_fills(prev_bid, prev_ask, tape, net_yes, adverse_params, wm)
            actual = _actual_fills(conn, c.ticker, interval_start, ts)
            act_bid, act_ask = _actual_fills_side(conn, c.ticker, interval_start, ts)
            placement_mid = _maybe_float(kv_get(conn, f"{KV}:pmid:{c.ticker}", ""))
            model_fair = _maybe_float(kv_get(conn, f"{KV}:mfair:{c.ticker}", ""))
            conn.execute(
                "INSERT OR IGNORE INTO mm_fill_audit (interval_start, interval_end, "
                "mode, ticker, bid, ask, rested_any, rested_bid, rested_ask, size, "
                "market_mid, mid_at_placement, model_fair, horizon_days, print_vol, "
                "print_vol_bid, print_vol_ask, sim_fill_qty, sim_fill_qty_incl_adverse, "
                "sim_buy_yes, sim_buy_no, actual_fill_qty, actual_bid_fills, "
                "actual_ask_fills, adverse_skipped,cohort_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (interval_start, ts, mode, c.ticker, prev_bid, prev_ask, 1,
                 int(prev_bid is not None), int(prev_ask is not None),
                 int(audit_params.get("quote_size", 1)), c.yes_mid, placement_mid,
                 model_fair, c.horizon_days, sim.print_vol, sim.print_vol_bid,
                 sim.print_vol_ask, sim.buy_yes + sim.buy_no,
                 sim_adv.buy_yes + sim_adv.buy_no, sim.buy_yes, sim.buy_no,
                 actual, act_bid, act_ask, sim.adverse_skipped, cohort_id))
            summary["audited"] += 1
            if sim.newest:
                kv_set(conn, f"{KV}:wm:{c.ticker}", sim.newest)

        # --- 1b) inventory flatten (gated) ----------------------------------
        # If we hold one-sided inventory in this market, shed it instead of
        # re-quoting two-sided and deepening a toxic position. Skips the normal
        # quote below for this market.
        if flatten_on:
            flat = plan_flatten(c, net_yes, flatten_params)
            if flat.regime != "normal":
                _shed_stale_orders(conn, trading, ts, c.ticker, state)
                _flatten_market(conn, trading, ts, mode, c, net_yes, flat, state,
                                limits, flatten_params, order_expiry, summary, verbose)
                continue

        # --- 2) plan + place this cycle's quote -----------------------------
        gate = risk.market_allowed(c.ticker, c.city, c.open_interest, c.horizon_days)
        plan = plan_quote(c, net_yes, quote_params) if not gate else None
        skip_reason = gate or (plan.skip_reason if plan and not plan.quotable else "")
        if gate or plan is None or not plan.quotable:
            _shed_stale_orders(conn, trading, ts, c.ticker, state)
            _log_quote(conn, ts, mode, c, net_yes, rested_both=0, skip_reason=skip_reason,
                       plan=plan)
            kv_set(conn, f"{KV}:bid:{c.ticker}", "")
            kv_set(conn, f"{KV}:ask:{c.ticker}", "")
            continue

        event = event_of.get(c.ticker, (c.city, c.target_date))
        # Never quote AT the book edge: a post-only bid clipped to the $0.01 floor
        # (or ask to the $0.99 ceiling) is rejected by Kalshi ("post only"), which
        # would place only the OTHER side and orphan a one-sided quote. If we can't
        # rest both sides inside the book, don't quote this market at all.
        # One-sided contract (W3): a quotable plan may deliberately populate only
        # ONE of new_bid/new_ask (the other is None = "do not rest that side"); the
        # default two-sided plan populates both. Every deref below is None-guarded so
        # a suppressed side never crashes (no f"{None:.2f}") and never gets placed.
        # With both sides populated this path is byte-identical to the old
        # both-or-neither logic.
        if (plan.new_bid is not None and plan.new_bid <= 0.01) or \
           (plan.new_ask is not None and plan.new_ask >= 0.99):
            _shed_stale_orders(conn, trading, ts, c.ticker, state)
            _log_quote(conn, ts, mode, c, net_yes, rested_both=0,
                       skip_reason="at_book_edge", plan=plan)
            kv_set(conn, f"{KV}:bid:{c.ticker}", "")
            kv_set(conn, f"{KV}:ask:{c.ticker}", "")
            continue

        # Rest only the populated side(s). A two-sided plan is placed all-or-neither
        # (the audit needs two live sides); a one-sided plan rests its single side.
        # A previously-rested order on the now-suppressed side was already cancelled
        # by the cancel-and-replace above, so it is never left stranded.
        sides = [(s, p) for s, p in (("bid", plan.new_bid), ("ask", plan.new_ask))
                 if p is not None]
        if not sides:                         # quotable but no side populated: skip
            _shed_stale_orders(conn, trading, ts, c.ticker, state)
            _log_quote(conn, ts, mode, c, net_yes, rested_both=0,
                       skip_reason="not_quotable", plan=plan)
            kv_set(conn, f"{KV}:bid:{c.ticker}", "")
            kv_set(conn, f"{KV}:ask:{c.ticker}", "")
            continue
        two_sided = len(sides) == 2
        summary["planned"] += len(sides)

        # --- price-unchanged fast path: keep a still-resting quote in place -----
        # Cancel-and-replace every cycle sends an UNCHANGED quote to the back of
        # the exchange's price-time queue for nothing -- on these ~1-2c books
        # that's a prime suspect for the low fill-realization rate. Only take
        # this path when EVERY needed side's order is confirmed still resting
        # against the ground-truth reconcile read (an order can fill or
        # TTL-expire between cycles -- trusting our own cache without this check
        # could "keep" an order that's actually already gone) AND at the
        # identical price, AND no side we no longer want is still resting (that
        # must still be shed via the full-replace path below).
        wanted = {s for s, _ in sides}
        prev_price = {"bid": prev_bid, "ask": prev_ask}
        prev_oid = {s: (kv_get(conn, f"{KV}:{s}_oid:{c.ticker}", "") or None)
                    for s in ("bid", "ask")}
        orphan = next((s for s in ("bid", "ask") if s not in wanted and prev_oid[s]
                       and prev_oid[s] in state.resting_order_ids), None)
        unchanged = orphan is None and all(
            prev_oid[s] and prev_oid[s] in state.resting_order_ids
            and prev_price[s] is not None and round(prev_price[s], 2) == round(p, 2)
            for s, p in sides)
        if unchanged:
            if two_sided:
                summary["quoted_both"] += 1
            _log_quote(conn, ts, mode, c, net_yes, rested_both=1 if two_sided else 0,
                       skip_reason="", plan=plan)
            continue

        # Plan changed (or a cached order already filled/expired): shed whichever
        # of THIS TICKER's own previously-resting orders are still live before
        # placing fresh ones. Scoped to this ticker only, never a blanket sweep,
        # so an unrelated market's queue position is never disturbed by a change
        # elsewhere in the book.
        _shed_stale_orders(conn, trading, ts, c.ticker, state)

        room = state.open_order_count <= limits.max_open_orders - len(sides)
        # Vet the pair sequentially against a planning copy so side two sees the
        # cost and directional obligation reserved by side one. Both are still
        # placed all-or-neither; this closes the same-cycle paired-cost gap.
        planning_state = copy.deepcopy(state)
        verdicts = {}
        for s, p in sides:
            verdicts[s] = risk.vet_order(
                ticker=c.ticker, side=s, price=p, count=quote_size,
                event=event, state=planning_state)
            if verdicts[s].allowed > 0:
                reserve_order(planning_state, c.ticker, s, p, verdicts[s].allowed, event)
        all_ok = room and all(v.allowed > 0 for v in verdicts.values())

        # Capacity rescue: when one side is vetoed purely because the BOOK is at
        # its net-inventory cap, the companion side reduces book exposure — its
        # fill frees capacity. Discarding it (the old all-or-neither pair block)
        # was strictly counterproductive. Rescue only the side whose direction
        # opposes the book's net; capital blocks are direction-blind (an ask
        # fill also spends capital), so they never qualify.
        if not all_ok and room and len(sides) == 2:
            reduce_side = ("ask" if state.net_inventory > 0
                           else "bid" if state.net_inventory < 0 else None)
            vetoed = {s: v for s, v in verdicts.items() if v.allowed <= 0}
            if (reduce_side and reduce_side not in vetoed
                    and all(v.reason == "max_net_inventory" for v in vetoed.values())):
                for s, p in sides:
                    if s in vetoed:
                        summary["blocked"] += 1
                        _log_risk(conn, ts, "block", c.ticker, f"{s}: {vetoed[s].reason}")
                        _record_order(conn, ts, mode, c.ticker, s, p, 0, status="blocked",
                                      reason=vetoed[s].reason, resp=None)
                sides = [(s, p) for s, p in sides if s == reduce_side]
                two_sided = False
                if reduce_side == "bid":
                    plan.new_ask = None   # suppressed side must not be cached
                else:
                    plan.new_bid = None
                all_ok = True

        rested_both = 0
        fully_rested = False
        if not all_ok:
            for s, p in sides:
                reason = verdicts[s].reason if verdicts[s].allowed <= 0 else (
                    "" if room else "max_open_orders")
                summary["blocked"] += 1
                _log_risk(conn, ts, "block", c.ticker, f"{s}: {reason or 'paired_side_blocked'}")
                _record_order(conn, ts, mode, c.ticker, s, p, 0, status="blocked",
                              reason=reason or "paired_side_blocked", resp=None)
            kv_set(conn, f"{KV}:bid:{c.ticker}", "")
            kv_set(conn, f"{KV}:ask:{c.ticker}", "")
        else:
            placed = []      # (side, price, count, order_id) that actually rested
            for s, p in sides:
                n = verdicts[s].allowed
                oid = _place(conn, trading, ts, mode, c.ticker, s, p, n, verbose,
                             expiration_time=order_expiry)
                if oid is not None:
                    summary["placed"] += 1
                    state.open_order_count += 1
                    reserve_order(state, c.ticker, s, p, n, event)
                    placed.append((s, p, n, oid))
            if len(placed) == len(sides):
                fully_rested = True
                if two_sided:
                    rested_both = 1
                    summary["quoted_both"] += 1
                # Cache each populated side (the audit reads the pair next cycle);
                # clear a suppressed side so a one-sided cycle never fires a stale
                # two-sided audit. Also cache the order_id/count so an UNCHANGED
                # next cycle can recognize this order and leave it resting.
                kv_set(conn, f"{KV}:bid:{c.ticker}",
                       f"{plan.new_bid:.4f}" if plan.new_bid is not None else "")
                kv_set(conn, f"{KV}:ask:{c.ticker}",
                       f"{plan.new_ask:.4f}" if plan.new_ask is not None else "")
                kv_set(conn, f"{KV}:ts:{c.ticker}", ts)
                for s, p, n, oid in placed:
                    kv_set(conn, f"{KV}:{s}_oid:{c.ticker}", oid or "")
                    kv_set(conn, f"{KV}:{s}_cnt:{c.ticker}", n)
                # Persist placement context so the NEXT cycle's audit isn't blind:
                # market mid AT PLACEMENT (effective-spread base) and the model fair.
                kv_set(conn, f"{KV}:pmid:{c.ticker}",
                       f"{c.yes_mid:.4f}" if c.yes_mid is not None else "")
                kv_set(conn, f"{KV}:mfair:{c.ticker}",
                       f"{plan.model_fair:.4f}" if plan.model_fair is not None else "")
                if not kv_get(conn, f"{KV}:wm:{c.ticker}", ""):
                    # First interval resting this ticker: watermark to "now" so the
                    # next interval measures only prints arriving while we rest.
                    latest = kalshi.trades_since(c.ticker, "")
                    if latest:
                        kv_set(conn, f"{KV}:wm:{c.ticker}", latest[-1].get("created_time", ""))
            else:
                # Partial pair (one side errored, e.g. a post-only reject): CANCEL
                # the side that did rest, so we never leave a one-sided quote live.
                for s, p, n, oid in placed:
                    if oid and trading.place_orders:
                        try:
                            trading.cancel_order(oid)
                        except (KalshiAuthError, KalshiTradingError) as e:
                            _log_risk(conn, ts, "api_error", c.ticker,
                                      f"orphan cancel {s}: {e}")
                        else:
                            # Update the row _place just wrote (don't append a
                            # duplicate) so live_orders shows one terminal status.
                            conn.execute(
                                "UPDATE live_orders SET status='canceled', "
                                "reason='orphan_pair', updated_ts=? WHERE order_id=? "
                                "AND ts=?", (ts, oid, ts))
                    state.open_order_count = max(0, state.open_order_count - 1)
                    release_order(state, c.ticker, s, p, n, event)
                kv_set(conn, f"{KV}:bid:{c.ticker}", "")
                kv_set(conn, f"{KV}:ask:{c.ticker}", "")
                kv_set(conn, f"{KV}:bid_oid:{c.ticker}", "")
                kv_set(conn, f"{KV}:ask_oid:{c.ticker}", "")

        _log_quote(conn, ts, mode, c, net_yes, rested_both=rested_both,
                   skip_reason="" if fully_rested else "not_rested", plan=plan)

    # --- 3) resting passive exits (gated) -----------------------------------
    # For each held experiment market we did NOT already work out this cycle
    # (no opposing reducing order placed above), rest ONE post-only reduce-only
    # maker at entry ± scratch_ticks to shed inventory passively. Default off.
    if resting_exit_on:
        for c in ctxs:
            if not _in_experiment(c.ticker, prefixes):
                continue                    # firewall: never touch foreign inventory
            net_yes = state.net_by_market.get(c.ticker, 0.0)
            if abs(net_yes) < 1:
                continue                    # flat: nothing to exit
            opposing = "ask" if net_yes > 0 else "bid"   # the side that reduces |net|
            already = conn.execute(
                "SELECT COUNT(*) FROM live_orders WHERE ticker=? AND side=? AND ts=? "
                "AND status IN ('resting','planned','partial','filled')",
                (c.ticker, opposing, ts)).fetchone()[0]
            if already:
                continue                    # already have an opposing resting order
            _rest_exit(conn, trading, ts, mode, c, net_yes,
                       state.entry_by_market.get(c.ticker), state, limits,
                       order_expiry, scratch_ticks, summary, verbose)

    _monitor_experiment(conn, ts, cohort_id, ttl, limits.daily_max_loss)
    conn.commit()
    log_run(conn, "live-mm", _jdump(summary))
    conn.commit()
    conn.close()
    if verbose:
        print(f"live-mm @ {ts} [{mode}]: {summary['contexts']} ctxs, "
              f"{summary['audited']} audited, {summary['quoted_both']} quoted, "
              f"{summary['placed']} placed, {summary['blocked']} blocked"
              + (" (NOT authenticated — sim only)" if not trading.authenticated else ""))
    return summary


def _epoch(ts_iso: str) -> int:
    """UTC ISO8601 (…Z) -> Unix seconds, for Kalshi's expiration_time field."""
    from datetime import datetime
    return int(datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp())


def _place(conn, trading, ts, mode, ticker, side, price, count, verbose,
           expiration_time=None, *, post_only=True,
           time_in_force="good_till_canceled", reason=""):
    """Place one order. Returns the resting order_id on live success, "" on
    dry_run success (nothing to cancel), or None on failure — so the caller can
    tell success from failure AND cancel an orphaned side of a partial pair.

    expiration_time (Unix seconds) makes the EXCHANGE auto-cancel the order if we
    never get back to re-quote it — a dead-man's switch against a stalled loop.
    """
    coid = str(uuid.uuid4())
    try:
        resp = trading.create_order(ticker=ticker, side=side, price=price, count=count,
                                    client_order_id=coid, post_only=post_only,
                                    time_in_force=time_in_force,
                                    expiration_time=expiration_time)
    except (KalshiAuthError, KalshiTradingError) as e:
        _log_risk(conn, ts, "api_error", ticker, f"place {side}: {e}")
        _record_order(conn, ts, mode, ticker, side, price, count,
                      status="error", reason=str(e), resp=None, client_order_id=coid)
        return None
    if resp.get("dry_run"):
        _record_order(conn, ts, mode, ticker, side, price, count, status="planned",
                      reason=reason, resp=None, client_order_id=coid, order_id=None)
        return ""
    # Kalshi's V2 create-order response (POST /portfolio/events/orders) is a FLAT
    # body — order_id / client_order_id / fill_count / remaining_count / ts_ms at
    # the top level, NOT nested under an "order" key (verified vs docs.kalshi.com,
    # Jul 2026). Read flat; fall back to a nested "order" object only if a future
    # schema drift reintroduces one. There is no `status` field, so derive it from
    # the fill/remaining counts (fixed-point strings like "0.00"/"1.00").
    order = resp.get("order") if isinstance(resp.get("order"), dict) else resp
    filled = _f(order.get("fill_count"))
    remaining = _f(order.get("remaining_count"))
    if order.get("status"):
        status = order["status"]
    elif filled > 0 and remaining <= 0:
        status = "filled"
    elif filled > 0:
        status = "partial"
    else:
        status = "resting"
    oid = order.get("order_id")
    _record_order(conn, ts, mode, ticker, side, price, count, status=status,
                  reason=reason, resp=_jdump(resp), order_id=oid,
                  client_order_id=order.get("client_order_id") or coid)
    return oid or ""    # "" = placed but no id returned (rare) — success, uncancelable


def _record_order(conn, ts, mode, ticker, side, price, count, *, status, reason,
                  resp, client_order_id=None, order_id=None):
    conn.execute(
        "INSERT OR IGNORE INTO live_orders (ts, mode, strategy, ticker, side, price, "
        "count, client_order_id, order_id, status, reason, resp_json, updated_ts,cohort_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ts, mode, "market_making", ticker, side, price, int(count),
         client_order_id or str(uuid.uuid4()), order_id, status, reason, resp, ts,
         kv_get(conn, f"{KV}:cohort", "legacy")))


class KillSwitchError(RuntimeError):
    """Raised when the panic button cannot list/cancel orders — must be LOUD, never
    a silent no-op, so 'kill' failing to cancel real orders can't be mistaken for
    success."""


def _cancel_all_resting(conn, trading: KalshiTradingClient, ts: str,
                        prefixes: tuple[str, ...], verbose: bool,
                        raise_on_error: bool = False) -> int:
    """Cancel the experiment's resting orders (kill switch + cancel/replace).
    Returns the number cancelled.

    FIREWALL: only orders on the experiment's own markets (`prefixes`) are
    cancelled. Orders elsewhere in the account — e.g. resting turnout election
    orders — are never touched, even when the kill switch fires.

    raise_on_error=True (the standalone panic button) turns a failed list/cancel
    into a raised KillSwitchError instead of a logged-and-return, so the operator
    is never told nothing when orders may still be live.
    """
    if not trading.place_orders:
        return 0  # dry_run: nothing was ever placed
    try:
        orders = trading.get_orders(status="resting")
    except (KalshiAuthError, KalshiTradingError) as e:
        _log_risk(conn, ts, "api_error", None, f"list resting failed: {e}")
        if raise_on_error:
            raise KillSwitchError(f"could not list resting orders: {e}") from e
        return 0
    ids = [o.get("order_id") for o in orders
           if o.get("order_id") and _in_experiment(o.get("ticker"), prefixes)]
    if not ids:
        if verbose:
            print("  no experiment orders resting — nothing to cancel")
        return 0
    results = trading.cancel_all(ids)
    failed = [r for r in results if r.get("error")]
    # Reflect successful cancels in the local order log so live-status doesn't keep
    # counting cancelled orders as resting (results are in the same order as ids).
    for oid, r in zip(ids, results):
        if not r.get("error"):
            conn.execute("UPDATE live_orders SET status='canceled', updated_ts=? "
                         "WHERE order_id=? AND status='resting'", (ts, oid))
    if failed:
        _log_risk(conn, ts, "api_error", None, f"{len(failed)} cancels failed")
        if raise_on_error:
            raise KillSwitchError(f"{len(failed)}/{len(ids)} cancels FAILED — "
                                  "orders may still be live")
    if verbose:
        print(f"  cancelled {len(ids) - len(failed)}/{len(ids)} resting orders")
    return len(ids) - len(failed)


def _maybe_float(s):
    try:
        return float(s) if s else None
    except (TypeError, ValueError):
        return None


def _jdump(obj):
    import json
    return json.dumps(obj, separators=(",", ":"))
