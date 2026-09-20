"""Collection cycle: snapshot markets + forecasts, then run every strategy.

Run this on a schedule (e.g. hourly, and a few times in the final hours before
each market closes). Each cycle:
  1. pull open Kalshi weather markets -> upsert markets + price snapshots
  2. pull the multi-model ensemble + climatology -> per (city, date) distribution
  3. build market contexts and let each enabled strategy trade on paper
  4. record signals + equity for performance tracking
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from .clients.kalshi import KalshiClient
from .clients.openmeteo import OpenMeteoClient
from .config import Config, DEFAULT_DB, load_config, now_utc, utcnow_iso
from .db import connect, jdump, log_run, upsert
from .distributions import Forecast, condition_members
from .fees import maker_fee, taker_fee
from . import engine
from .strategies import REGISTRY, MarketCtx, Services


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _taker_tradable(cx, side: str, flt: dict) -> bool:
    """Liquidity hygiene for TAKER entries: don't lift the ask on a dead or
    blown-out book. Gates on the side being bought so it never blocks a one-sided
    cheap longshot (longshot_fade buys NO on a YES-only book). Market-making is
    exempt — it quotes inside the spread rather than crossing it.

    Skips when: the bought side has no ask (can't take); open interest is below
    the floor (dead market); or a two-sided quote on the bought side is wider
    than max_spread (crossing it would eat any edge)."""
    ask = cx.yes_ask if side == "yes" else cx.no_ask
    bid = cx.yes_bid if side == "yes" else cx.no_bid
    if ask is None:
        return False
    min_oi = flt.get("min_open_interest", 0)
    if min_oi and cx.open_interest is not None and cx.open_interest < min_oi:
        return False
    max_spread = flt.get("max_spread")
    if max_spread is not None and bid is not None and (ask - bid) > max_spread + 1e-9:
        return False
    return True


def ensure_strategies(conn: sqlite3.Connection, cfg: Config) -> None:
    ts = utcnow_iso()
    for name, params in cfg.strategies.items():
        if not params.get("enabled"):
            continue
        exists = conn.execute("SELECT 1 FROM strategies WHERE name=?", (name,)).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO strategies (name, enabled, bankroll0, cash, realized_pnl, "
                "params_json, created_ts) VALUES (?,?,?,?,?,?,?)",
                (name, 1, cfg.starting_bankroll, cfg.starting_bankroll, 0.0,
                 jdump(params), ts),
            )
        else:
            conn.execute("UPDATE strategies SET enabled=1, params_json=? WHERE name=?",
                         (jdump(params), name))
    conn.commit()


class ClimatologyCache:
    """One archive pull per city; serves windowed day-of-year member sets."""

    def __init__(self, om: OpenMeteoClient, cfg: Config):
        self.om = om
        self.cfg = cfg
        self._series: dict[str, dict[str, float]] = {}

    def _load(self, code: str) -> dict[str, float]:
        if code in self._series:
            return self._series[code]
        c = self.cfg.cities[code]
        lookback = self.cfg.strategies.get("climatology", {}).get("lookback_years", 15)
        end = (now_utc() - timedelta(days=2)).strftime("%Y-%m-%d")
        start = f"{now_utc().year - lookback}-01-01"
        try:
            series = self.om.archive_daily_highs(
                c["lat"], c["lon"], c["tz"], start, end,
                unit=self.cfg.forecast["temperature_unit"])
        except Exception:
            series = {}
        self._series[code] = series
        return series

    def members(self, code: str, target_date: str, window: int = 7) -> list[float]:
        series = self._load(code)
        if not series:
            return []
        try:
            tgt = datetime.fromisoformat(target_date)
        except ValueError:
            return []
        mmdd = (tgt.month, tgt.day)
        out = []
        for d, v in series.items():
            try:
                dt = datetime.fromisoformat(d)
            except ValueError:
                continue
            # day-of-year distance, ignoring year, within +/- window
            delta = abs((datetime(2000, dt.month, dt.day) - datetime(2000, *mmdd)).days)
            delta = min(delta, 366 - delta)
            if delta <= window:
                out.append(v)
        return out


def build_contexts(conn: sqlite3.Connection, cfg: Config, kalshi: KalshiClient,
                   om: OpenMeteoClient, ts: str,
                   verbose: bool = True) -> tuple[list[MarketCtx], dict]:
    """Pull markets/snapshots/forecasts and assemble the per-market contexts.

    This is steps 1-3 of a collection cycle with NO trading. The paper `collect`
    loop and the live `kwt.live_engine` both call this so they operate on an
    identical view of the market. Snapshots are persisted as a side effect
    (desirable for both paths). Returns (contexts, {markets, snapshots}).
    """
    clim_cache = ClimatologyCache(om, cfg)
    series_to_city = cfg.series_to_city()
    unit = cfg.forecast["temperature_unit"]

    # 1) markets + snapshots ------------------------------------------------
    n_markets = n_snaps = 0
    market_rows: list[dict] = []
    needed_dates: dict[str, set[str]] = {}
    for code, c in cfg.cities.items():
        try:
            raws = list(kalshi.iter_markets(c["series"], status="open"))
        except Exception as e:
            if verbose:
                print(f"  ! {code} markets fetch failed: {e}")
            continue
        for m in raws:
            norm = kalshi.normalize_market(m, series_to_city)
            if norm["target_date"] is None:
                continue
            close = _parse_iso(norm["close_time"])
            horizon = ((close - now_utc()).total_seconds() / 86400.0) if close else 999
            norm_db = dict(norm)
            norm_db["first_seen"] = ts
            norm_db["last_seen"] = ts
            # don't overwrite first_seen on update
            existing = conn.execute("SELECT first_seen FROM markets WHERE ticker=?",
                                    (norm["ticker"],)).fetchone()
            if existing:
                norm_db["first_seen"] = existing["first_seen"]
            upsert(conn, "markets", norm_db, keys=["ticker"])
            n_markets += 1
            snap = kalshi.snapshot_row(m, ts)
            conn.execute(
                "INSERT OR IGNORE INTO snapshots (ts,ticker,yes_bid,yes_ask,no_bid,no_ask,"
                "last_price,volume,open_interest,liquidity) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (snap["ts"], snap["ticker"], snap["yes_bid"], snap["yes_ask"],
                 snap["no_bid"], snap["no_ask"], snap["last_price"], snap["volume"],
                 snap["open_interest"], snap["liquidity"]))
            n_snaps += 1
            if cfg.min_horizon_days <= horizon <= cfg.max_horizon_days:
                norm["_horizon"] = horizon
                norm["_snap"] = snap
                market_rows.append(norm)
                needed_dates.setdefault(code, set()).add(norm["target_date"])
        # Commit per city so the write lock is released before the NEXT city's
        # network fetch — otherwise the lock is held across every city's HTTP call
        # and concurrent cycles (live-mm every 10 min) hit 'database is locked'.
        conn.commit()
    conn.commit()

    # 2) forecasts (nwp ensemble + climatology) ----------------------------
    fc_cache: dict[tuple[str, str], dict] = {}
    fdays = cfg.forecast["forecast_days"]
    for code, dates in needed_dates.items():
        c = cfg.cities[code]
        try:
            ens = om.ensemble_daily_highs(c["lat"], c["lon"], c["tz"],
                                          cfg.forecast["ensemble_models"], fdays, unit)
        except Exception as e:
            ens = {}
            if verbose:
                print(f"  ! {code} ensemble fetch failed: {e}")
        try:
            det = om.deterministic_daily_highs(c["lat"], c["lon"], c["tz"],
                                               cfg.forecast["deterministic_models"], fdays, unit)
        except Exception:
            det = {}
        for date in dates:
            nwp_fc = clim_fc = None
            members = ens.get(date, [])
            if members:
                nwp_fc = Forecast.from_members(members)
                q = nwp_fc.quantiles()
                horizon = (datetime.fromisoformat(date) - datetime.fromisoformat(ts[:10])).days
                upsert(conn, "forecasts", {
                    "ts": ts, "city": code, "target_date": date, "horizon_days": horizon,
                    "mean": nwp_fc.mean, "std": nwp_fc.std, **q,
                    "n_members": len(members), "members_json": jdump([round(m, 2) for m in members]),
                    "models_json": jdump(det.get(date, {})), "source": "nwp-ensemble",
                }, keys=["ts", "city", "target_date", "source"])
            clim_members = clim_cache.members(code, date)
            if clim_members:
                clim_fc = Forecast.from_members(clim_members)
                q = clim_fc.quantiles()
                upsert(conn, "forecasts", {
                    "ts": ts, "city": code, "target_date": date, "horizon_days": -1,
                    "mean": clim_fc.mean, "std": clim_fc.std, **q,
                    "n_members": len(clim_members), "members_json": "[]",
                    "models_json": "{}", "source": "climatology",
                }, keys=["ts", "city", "target_date", "source"])
            fc_cache[(code, date)] = {"nwp": nwp_fc, "clim": clim_fc}
        # Release the lock between cities (see the markets loop above): the next
        # city's ensemble/deterministic HTTP fetch must not run holding the lock.
        conn.commit()
    conn.commit()

    # 2b) intraday observed-so-far max for every near-term city. The relevant
    # "today" is the STATION-LOCAL date (from the API's current_time), which can
    # differ from the UTC date near midnight — we attach the floor only to the
    # market whose target_date matches that local date.
    intraday: dict[str, dict] = {}
    for code in needed_dates:
        c = cfg.cities[code]
        try:
            intraday[code] = om.intraday_state(c["lat"], c["lon"], c["tz"], unit)
        except Exception:
            intraday[code] = {}

    # 3) build contexts -----------------------------------------------------
    ctxs: list[MarketCtx] = []
    flt = cfg.filters
    for r in market_rows:
        fc = fc_cache.get((r["city"], r["target_date"]), {})
        if not fc.get("nwp") and not fc.get("clim"):
            continue
        s = r["_snap"]
        ya = s["yes_ask"]
        # liquidity / sanity filters
        if ya is not None and (ya < flt["min_yes_ask"] or ya > flt["max_yes_ask"]):
            pass  # still allow; longshot_fade wants cheap YES. filter spread only.
        if flt.get("require_two_sided") and (s["yes_bid"] is None or s["yes_ask"] is None):
            continue
        if (s["yes_bid"] is not None and s["yes_ask"] is not None
                and (s["yes_ask"] - s["yes_bid"]) > flt["max_spread"] + 1e-9
                and r["bucket_kind"] != "unknown"):
            # too wide for taker entry, but keep for MM which quotes inside
            pass
        intr_all = intraday.get(r["city"], {})
        local_today = (intr_all.get("current_time") or "")[:10]
        # Attach the intraday observation to the market for the station's LOCAL
        # current day (computed in the city's own tz, so it rolls over correctly).
        # No hours gate here: the shared conditioning below is self-gating (a weak
        # morning floor is a no-op). intraday_nowcast applies its own hours gate.
        intr = intr_all if r["target_date"] == local_today else {}
        metric = r.get("metric", "high")   # Spec #2 will set this per series; all 'high' today
        # Fail-safe: only the high path has an intraday observation wired
        # (intraday_state returns observed_max). Until the low seam (Spec #2)
        # supplies observed_min, leave any non-high market UNCONDITIONED rather
        # than floor a low against the max and decide it the wrong way.
        obs_so_far = intr.get("observed_max") if metric == "high" else None

        # Condition the forecasts on the observed-so-far extreme so every strategy
        # sees a physically-possible distribution; keep the raw forecast available.
        nwp_raw, clim_raw = fc.get("nwp"), fc.get("clim")
        if obs_so_far is not None:
            nwp = (Forecast.from_members(condition_members(nwp_raw.members, obs_so_far, metric))
                   if nwp_raw else None)
            clim = (Forecast.from_members(condition_members(clim_raw.members, obs_so_far, metric))
                    if clim_raw else None)
        else:
            nwp, clim = nwp_raw, clim_raw

        ctxs.append(MarketCtx(
            ticker=r["ticker"], city=r["city"], target_date=r["target_date"],
            low=r["low"], high=r["high"], bucket_kind=r["bucket_kind"], metric=metric,
            horizon_days=r["_horizon"], yes_bid=s["yes_bid"], yes_ask=s["yes_ask"],
            no_bid=s["no_bid"], no_ask=s["no_ask"], last_price=s["last_price"],
            open_interest=s["open_interest"], nwp=nwp, clim=clim,
            nwp_raw=nwp_raw, clim_raw=clim_raw, obs_so_far=obs_so_far,
            remaining_max=intr.get("remaining_max"),
            hours_elapsed=intr.get("hours_elapsed")))

    return ctxs, {"markets": n_markets, "snapshots": n_snaps}


def collect(db_path=DEFAULT_DB, cfg: Config | None = None, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    conn = connect(db_path)
    ensure_strategies(conn, cfg)
    ts = utcnow_iso()
    kalshi = KalshiClient()
    om = OpenMeteoClient()

    ctxs, counts = build_contexts(conn, cfg, kalshi, om, ts, verbose)
    n_markets, n_snaps = counts["markets"], counts["snapshots"]

    # 4) run strategies -----------------------------------------------------
    services = Services(kalshi=kalshi, conn=conn, bankroll0=cfg.starting_bankroll,
                        fee_cfg=cfg.fees)
    summary = {"ts": ts, "markets": n_markets, "snapshots": n_snaps,
               "contexts": len(ctxs), "strategies": {}}
    for name in cfg.enabled_strategies():
        strat = REGISTRY[name](cfg.strategies[name], services)
        book = engine.load_book(conn, name)
        orders, signals = strat.generate(ctxs, book)
        filled = 0
        ctx_by_ticker = {c.ticker: c for c in ctxs}
        event_of = {c.ticker: (c.city, c.target_date) for c in ctxs}

        # Exits first: sells free cash and risk room before this cycle's entries.
        sells = [o for o in orders if o.action == "sell"]
        entries = [o for o in orders if o.action != "sell"]
        for o in sells:
            cx = ctx_by_ticker.get(o.ticker)
            if cx is None:
                continue
            snap = {"yes_bid": cx.yes_bid, "no_bid": cx.no_bid}
            if engine.execute_order(conn, o, snap, book, cfg.fees) > 0:
                filled += 1

        # portfolio risk caps (cost-basis deployed)
        # Per-strategy overrides take precedence over global risk config.
        strat_params = cfg.strategies.get(name, {})
        bankroll0 = cfg.starting_bankroll
        max_deployed = strat_params.get(
            "max_deployed_frac", cfg.risk.get("max_deployed_frac", 0.6)) * bankroll0
        max_market = strat_params.get(
            "max_market_frac", cfg.risk.get("max_market_frac", 0.08)) * bankroll0
        # Per-EVENT cap: the ~18 buckets of one city-day are a single multinomial
        # outcome, so per-market caps alone let a strategy stack correlated bets
        # on the same weather realization.
        max_event = strat_params.get(
            "max_event_frac", cfg.risk.get("max_event_frac", 0.15)) * bankroll0
        deployed = engine.position_cost(conn, name)
        market_cost = {}
        for r in conn.execute(
                "SELECT ticker, COALESCE(SUM(cost),0) c FROM positions WHERE strategy=? "
                "GROUP BY ticker", (name,)):
            market_cost[r["ticker"]] = r["c"]
        event_cost: dict[tuple[str, str], float] = {}
        for r in conn.execute(
                "SELECT m.city ci, m.target_date td, COALESCE(SUM(p.cost),0) c "
                "FROM positions p JOIN markets m ON m.ticker = p.ticker "
                "WHERE p.strategy=? GROUP BY m.city, m.target_date", (name,)):
            event_cost[(r["ci"], r["td"])] = r["c"]
        for o in entries:
            cx = ctx_by_ticker.get(o.ticker)
            if cx is None:
                continue
            # Liquidity hygiene on taker entries (MM quotes inside, so it uses
            # action='fill_at' and is exempt; event_arbitrage needs the whole
            # partition to fill atomically, so it manages its own completeness).
            if (o.action == "buy" and name != "event_arbitrage"
                    and not _taker_tradable(cx, o.side, cfg.filters)):
                continue
            est_price = (cx.yes_ask if o.side == "yes" else cx.no_ask) \
                if o.action == "buy" else o.limit_price
            if est_price is None:
                continue

            def est_cost_of(n: float) -> float:
                fee = (maker_fee(n, est_price, cfg.fees.get("maker_rate", 0.0025))
                       if o.role == "maker"
                       else taker_fee(n, est_price, cfg.fees.get("taker_rate", 0.07)))
                return n * est_price + fee

            # per-market, per-event and portfolio caps, fee-inclusive to match
            # real cost basis
            ev = event_of.get(o.ticker)
            room = min(max_market - market_cost.get(o.ticker, 0.0),
                       max_event - event_cost.get(ev, 0.0),
                       max_deployed - deployed)
            if est_cost_of(o.contracts) > room:   # scale order down to fit caps
                o.contracts = float(int(room / (est_price * 1.08)))  # headroom for fee
                if o.contracts < 1 or est_cost_of(o.contracts) > room:
                    continue
            snap = {"yes_ask": cx.yes_ask, "no_ask": cx.no_ask,
                    "yes_bid": cx.yes_bid, "no_bid": cx.no_bid}
            got = engine.execute_order(conn, o, snap, book, cfg.fees)
            if got > 0:
                filled += 1
                spent = est_cost_of(got)
                deployed += spent
                market_cost[o.ticker] = market_cost.get(o.ticker, 0.0) + spent
                event_cost[ev] = event_cost.get(ev, 0.0) + spent
        for sig in signals:
            engine.record_signal(conn, ts, sig)
        engine.record_equity(conn, ts, name)
        summary["strategies"][name] = {"orders": len(orders), "filled": filled,
                                       "signals": len(signals)}
        if verbose:
            print(f"  {name:>20}: {len(orders):>3} orders, {filled:>3} filled, "
                  f"cash=${book.cash:,.2f}")
    conn.commit()
    log_run(conn, "collect", jdump(summary))
    conn.commit()
    conn.close()
    if verbose:
        print(f"collect @ {ts}: {n_markets} markets, {len(ctxs)} in-horizon contexts")
    return summary
