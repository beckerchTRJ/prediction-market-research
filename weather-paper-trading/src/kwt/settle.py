"""Settlement: resolve matured markets and realize P&L.

Reads Kalshi's authoritative `result` for any market we hold an open position in
(or any tracked market past its close). Pays $1 per winning contract, books the
realized P&L, frees the position, and records the realized daily high (from
Kalshi's grading via Open-Meteo actuals) for model evaluation.
"""
from __future__ import annotations

import sqlite3

from .clients.kalshi import KalshiClient
from .clients.nws import NWSClient
from .clients.openmeteo import OpenMeteoClient
from .config import Config, DEFAULT_DB, load_config, utcnow_iso
from .db import connect, jdump, log_run, upsert
from . import engine


def _resolved_result(market: dict | None) -> str | None:
    if not market:
        return None
    res = (market.get("result") or "").lower()
    status = (market.get("status") or "").lower()
    if res in ("yes", "no") and status in ("settled", "finalized", "determined", "closed"):
        return res
    if res in ("yes", "no"):
        return res
    return None


def settle(db_path=DEFAULT_DB, cfg: Config | None = None, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    conn = connect(db_path)
    kalshi = KalshiClient()
    om = OpenMeteoClient()
    nws = NWSClient()
    ts = utcnow_iso()

    # tickers we hold OR that are tracked-but-unsettled and past close
    rows = conn.execute("""
        SELECT DISTINCT m.ticker, m.city, m.target_date
        FROM markets m
        WHERE m.ticker IN (SELECT DISTINCT ticker FROM positions)
           OR (m.result IS NULL AND m.close_time < ?)
    """, (ts,)).fetchall()

    settled_markets = 0
    realized_positions = 0
    pnl_total = 0.0
    obs_pending: dict[tuple[str, str], None] = {}

    for r in rows:
        ticker = r["ticker"]
        market = kalshi.get_market(ticker)
        result = _resolved_result(market)
        if result is None:
            continue
        # update market record
        conn.execute("UPDATE markets SET result=?, status=?, last_seen=? WHERE ticker=?",
                     (result, (market.get("status") or "settled"), ts, ticker))
        settled_markets += 1
        obs_pending[(r["city"], r["target_date"])] = None

        # settle every open position on this ticker
        positions = conn.execute(
            "SELECT strategy, side, contracts, cost FROM positions WHERE ticker=?",
            (ticker,)).fetchall()
        for p in positions:
            payoff = p["contracts"] if p["side"] == result else 0.0
            pnl = payoff - p["cost"]
            pnl_total += pnl
            realized_positions += 1
            # credit payoff to cash, book realized pnl
            srow = conn.execute("SELECT cash, realized_pnl FROM strategies WHERE name=?",
                                (p["strategy"],)).fetchone()
            conn.execute("UPDATE strategies SET cash=?, realized_pnl=? WHERE name=?",
                         (srow["cash"] + payoff, srow["realized_pnl"] + pnl, p["strategy"]))
            conn.execute(
                "UPDATE trades SET settled=1, pnl=CASE WHEN side=? THEN contracts*(1-price)-fee "
                "ELSE -(contracts*price)-fee END WHERE strategy=? AND ticker=? AND side=? AND settled=0",
                (result, p["strategy"], ticker, p["side"]))
            conn.execute("DELETE FROM positions WHERE strategy=? AND ticker=? AND side=?",
                         (p["strategy"], ticker, p["side"]))
    conn.commit()

    # record realized highs for evaluation (Kalshi grades local-day max)
    obs_written = 0
    for (city, date) in obs_pending:
        c = cfg.cities.get(city)
        if not c:
            continue
        existing = conn.execute(
            "SELECT 1 FROM observations WHERE city=? AND target_date=?", (city, date)).fetchone()
        if existing:
            continue
        observed, source = None, None
        # prefer the official settlement station's reading
        try:
            observed = nws.observed_high_f(c["lat"], c["lon"], date, c["tz"],
                                           station=c.get("station"))
            if observed is not None:
                source = f"nws-{c.get('station', 'station')}"
        except Exception:
            observed = None
        if observed is None:  # fall back to Open-Meteo reanalysis actuals
            try:
                highs = om.recent_observed_highs(c["lat"], c["lon"], c["tz"], past_days=10,
                                                 unit=cfg.forecast["temperature_unit"])
                if date in highs:
                    observed, source = highs[date], "open-meteo-actual"
            except Exception:
                observed = None
        if observed is not None:
            upsert(conn, "observations", {
                "city": city, "target_date": date, "observed_high": observed,
                "source": source, "ts": ts}, keys=["city", "target_date"])
            obs_written += 1

    # refresh equity for all strategies
    for name in cfg.enabled_strategies():
        engine.record_equity(conn, ts, name)

    summary = {"ts": ts, "settled_markets": settled_markets,
               "realized_positions": realized_positions,
               "pnl_total": round(pnl_total, 2), "observations": obs_written}
    log_run(conn, "settle", jdump(summary))
    conn.commit()
    conn.close()
    if verbose:
        print(f"settle @ {ts}: {settled_markets} markets resolved, "
              f"{realized_positions} positions, P&L ${pnl_total:,.2f}")
    return summary
