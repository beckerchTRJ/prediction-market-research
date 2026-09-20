"""Performance report: per-strategy P&L + the model-vs-market edge test.

Pulls everything from the DB and prints a console report; optionally writes CSV
artifacts. The edge verdict per strategy compares the strategy's last-pre-close
probability against the market's implied probability on resolved markets.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import ARTIFACTS_DIR, DEFAULT_DB, load_config
from .db import connect
from . import metrics

# Minimum independent blocks (city-days) before the edge verdict is trusted.
# Below this, the loss-differential has too few independent weather realizations
# to distinguish skill from one or two synoptic regimes — report "inconclusive".
MIN_EDGE_BLOCKS = 8
MIN_FADE_DATES = 20


def _categorical_trade_events(conn: sqlite3.Connection, strategy: str) -> tuple[list[dict], int]:
    """Build complete city-day categorical portfolios for the rare-loss null test."""
    traded_events = conn.execute(
        "SELECT m.city,m.target_date,MIN(t.ts) entry_ts "
        "FROM trades t JOIN markets m ON m.ticker=t.ticker "
        "WHERE t.strategy=? AND t.settled=1 AND m.target_date IS NOT NULL "
        "GROUP BY m.city,m.target_date ORDER BY m.target_date,m.city", (strategy,)).fetchall()
    events: list[dict] = []
    excluded = 0
    for ev in traded_events:
        markets = conn.execute(
            "SELECT ticker,result FROM markets WHERE city=? AND target_date=? "
            "AND result IN ('yes','no') ORDER BY COALESCE(low,-999),COALESCE(high,999)",
            (ev["city"], ev["target_date"])).fetchall()
        if not markets or sum(m["result"] == "yes" for m in markets) != 1:
            excluded += 1
            continue
        probs: list[float] = []
        complete = True
        for m in markets:
            snap = conn.execute(
                "SELECT yes_bid,yes_ask FROM snapshots WHERE ticker=? AND ts<=? "
                "AND yes_bid IS NOT NULL AND yes_ask IS NOT NULL ORDER BY ts DESC LIMIT 1",
                (m["ticker"], ev["entry_ts"])).fetchone()
            if not snap:
                complete = False
                break
            probs.append((float(snap["yes_bid"]) + float(snap["yes_ask"])) / 2.0)
        if not complete or sum(probs) <= 0:
            excluded += 1
            continue
        trades = conn.execute(
            "SELECT t.ticker,t.side,t.action,t.contracts,t.price,t.fee,t.pnl "
            "FROM trades t JOIN markets m ON m.ticker=t.ticker "
            "WHERE t.strategy=? AND t.settled=1 AND m.city=? AND m.target_date=?",
            (strategy, ev["city"], ev["target_date"])).fetchall()
        pnl_by_outcome = []
        tickers = [m["ticker"] for m in markets]
        for winner in tickers:
            pnl = 0.0
            for t in trades:
                wins = t["ticker"] == winner
                payout = float(t["contracts"]) * (
                    (1.0 if wins else 0.0) if t["side"] == "yes" else
                    (0.0 if wins else 1.0))
                cost = float(t["contracts"]) * float(t["price"])
                if t["action"] == "buy":
                    pnl += payout - cost - float(t["fee"] or 0)
                else:
                    pnl += cost - payout - float(t["fee"] or 0)
            pnl_by_outcome.append(pnl)
        events.append({"target_date": ev["target_date"],
                       "event": f"{ev['city']}|{ev['target_date']}",
                       "probs": probs, "pnl_by_outcome": pnl_by_outcome,
                       "actual_pnl": sum(float(t["pnl"] or 0) for t in trades)})
    return events, excluded


def _resolved_signal_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """EARLIEST (first-sighting) signal per (strategy, ticker) for resolved markets.

    The edge test must compare the model against the market at the SAME, early
    information set. Taking the last pre-close signal instead lets the market
    converge toward the realized outcome (market Brier -> ~0), which makes the
    market look omniscient and the model look hopeless — an apples-to-oranges
    comparison. First sighting is the fair point: both probabilities reflect what
    was knowable when we first quoted the market. The close_time guard still
    excludes any (degenerate) post-close signal so no lookahead can leak in.
    """
    q = """
    WITH joined AS (
        SELECT s.strategy, s.ticker, s.model_prob, s.market_prob, s.side, s.decision, s.ts,
               m.result, m.city, m.target_date, m.close_time,
               ROW_NUMBER() OVER (
                   PARTITION BY s.strategy, s.ticker ORDER BY s.ts ASC) AS rn
        FROM signals s
        JOIN markets m ON m.ticker = s.ticker
        WHERE m.result IN ('yes','no')
          AND (m.close_time IS NULL OR s.ts <= m.close_time)
          -- earliest EVALUABLE signal: ignore probability-less skips (e.g.
          -- calibration_overlay's out-of-horizon logs) so a market isn't
          -- represented by — then dropped for — a null-prob first row.
          AND s.model_prob IS NOT NULL AND s.market_prob IS NOT NULL
    )
    SELECT strategy, ticker, model_prob, market_prob, side, decision,
           result, city, target_date
    FROM joined WHERE rn = 1
    """
    df = pd.read_sql_query(q, conn)
    if not df.empty:
        df["outcome"] = (df["result"] == "yes").astype(float)
    return df


def build_report(db_path=DEFAULT_DB, write_csv: bool = True) -> dict:
    cfg = load_config()
    conn = connect(db_path)
    report: dict = {"strategies": {}, "edge": {}}

    strat_rows = conn.execute(
        "SELECT name, bankroll0, cash, realized_pnl FROM strategies").fetchall()
    sig_df = _resolved_signal_frame(conn)
    cfg_strats = cfg.strategies

    lines = ["", "=" * 78, "KALSHI WEATHER PAPER-TRADING REPORT", "=" * 78]

    for s in strat_rows:
        name = s["name"]
        # equity time series
        eq = pd.read_sql_query(
            "SELECT ts, equity FROM equity WHERE strategy=? ORDER BY ts", conn, params=(name,))
        # settled trades
        tr = pd.read_sql_query(
            "SELECT pnl FROM trades WHERE strategy=? AND settled=1", conn, params=(name,))
        n_open = conn.execute(
            "SELECT COUNT(*) c, COALESCE(SUM(contracts),0) k FROM positions WHERE strategy=?",
            (name,)).fetchone()
        n_trades = conn.execute(
            "SELECT COUNT(*) c FROM trades WHERE strategy=?", (name,)).fetchone()["c"]
        # Fee drag, reported separately: friction is a parameter of the harness,
        # not a verdict on the strategy's probabilistic skill (in the first days
        # fees were ~half of some strategies' losses).
        fees_paid = float(conn.execute(
            "SELECT COALESCE(SUM(fee),0) f FROM trades WHERE strategy=?",
            (name,)).fetchone()["f"])

        equity_now = s["cash"] + (conn.execute(
            "SELECT COALESCE(SUM(cost),0) c FROM positions WHERE strategy=?",
            (name,)).fetchone()["c"])
        total_return = (equity_now - s["bankroll0"]) / s["bankroll0"] if s["bankroll0"] else 0
        wins = int((tr["pnl"] > 0).sum()) if not tr.empty else 0
        win_rate = wins / len(tr) if len(tr) else float("nan")
        sharpe = (metrics.sharpe_from_equity(eq["ts"].tolist(), eq["equity"].tolist())
                  if len(eq) > 2 else float("nan"))

        # edge test vs market
        sd = sig_df[(sig_df["strategy"] == name) & sig_df["model_prob"].notna()
                    & sig_df["market_prob"].notna()] if not sig_df.empty else pd.DataFrame()
        edge = {}
        if len(sd) >= 8:
            mb = metrics.brier(sd["model_prob"], sd["outcome"])
            kb = metrics.brier(sd["market_prob"], sd["outcome"])
            # Correlation-aware: cluster the loss differential by city-day so the
            # verdict reflects independent weather realizations, not bucket count.
            blocks = sd["city"].astype(str) + "|" + sd["target_date"].astype(str)
            dm = metrics.diebold_mariano_blocked(
                sd["model_prob"], sd["market_prob"], sd["outcome"], blocks)
            edge = {"n_resolved": int(len(sd)), "n_blocks": dm["n_blocks"],
                    "model_brier": round(mb, 4), "market_brier": round(kb, 4),
                    "brier_skill_vs_market": round(metrics.brier_skill_score(mb, kb), 4),
                    "dm_stat": (round(dm["dm_stat"], 3) if dm["dm_stat"] == dm["dm_stat"]
                                else None),
                    "dm_p": (round(dm["p_value"], 4) if dm["p_value"] == dm["p_value"]
                             else None)}
        report["edge"][name] = edge

        # Graduation gate: clustered P&L significance + concentration + (forecast
        # strategies) Brier skill. Guards a concentrated no-skill run from being
        # promoted on raw P&L. verdict_metric='pnl' => market-bias play, P&L only.
        pnl_rows = conn.execute(
            "SELECT t.pnl, m.target_date FROM trades t JOIN markets m ON m.ticker=t.ticker "
            "WHERE t.strategy=? AND t.settled=1 AND m.target_date IS NOT NULL",
            (name,)).fetchall()
        if pnl_rows:
            pnls = [r["pnl"] for r in pnl_rows]
            blocks = [r["target_date"] for r in pnl_rows]
            psig = metrics.pnl_cluster_significance(pnls, blocks)
            conc = metrics.pnl_concentration(pnls)
            vm = (cfg_strats.get(name, {}) or {}).get("verdict_metric", "skill")
            if vm == "pnl":
                null_events, excluded = _categorical_trade_events(conn, name)
                psig = metrics.categorical_portfolio_null_test(null_events)
                psig["excluded_events"] = excluded
            bskill = report["edge"].get(name, {}).get("brier_skill_vs_market")
            verdict = metrics.graduation_verdict(
                pnl_sig=psig, concentration=conc, brier_skill=bskill, verdict_metric=vm,
                min_blocks=MIN_FADE_DATES if vm == "pnl" else 0)
            report["edge"].setdefault(name, {})["graduation"] = {
                "graduated": verdict["graduated"], "reasons": verdict["reasons"],
                "pnl_p_value": psig["p_value"], "concentration": conc,
                "n_blocks": psig["n_blocks"]}
            if vm == "pnl":
                report["edge"][name]["fade_null"] = psig

        report["strategies"][name] = {
            "bankroll0": s["bankroll0"], "cash": round(s["cash"], 2),
            "realized_pnl": round(s["realized_pnl"], 2), "equity": round(equity_now, 2),
            "total_return_pct": round(total_return * 100, 2),
            "open_positions": n_open["c"], "open_contracts": n_open["k"],
            "settled_trades": len(tr), "total_trades": n_trades,
            "win_rate": round(win_rate, 3) if win_rate == win_rate else None,
            "sharpe": round(sharpe, 2) if sharpe == sharpe else None,
            "fees_paid": round(fees_paid, 2),
        }

        st = report["strategies"][name]
        lines.append("")
        lines.append(f"▸ {name}")
        lines.append(f"    equity ${st['equity']:,.2f}  (start ${s['bankroll0']:,.0f}, "
                     f"return {st['total_return_pct']:+.2f}%)   realized P&L ${st['realized_pnl']:,.2f}"
                     f"   fees ${st['fees_paid']:,.2f}")
        lines.append(f"    trades: {st['total_trades']} total, {st['settled_trades']} settled, "
                     f"win-rate {st['win_rate']}   open: {st['open_positions']} mkts / "
                     f"{st['open_contracts']:.0f} contracts   sharpe {st['sharpe']}")
        if "n_resolved" in edge:
            nb = edge["n_blocks"]
            if edge["dm_p"] is None or nb < MIN_EDGE_BLOCKS:
                verdict = f"inconclusive ({nb} indep. city-days)"
            elif edge["dm_stat"] < 0 and edge["dm_p"] < 0.10:
                verdict = "BEATS market"
            else:
                verdict = "no sig. edge"
            dm_txt = f"DM={edge['dm_stat']}, p={edge['dm_p']}" if edge["dm_p"] is not None \
                else "DM=n/a"
            lines.append(f"    EDGE: model Brier {edge['model_brier']} vs market "
                         f"{edge['market_brier']}  (skill {edge['brier_skill_vs_market']:+.3f}, "
                         f"{dm_txt}) -> {verdict}  "
                         f"[n={edge['n_resolved']} buckets / {nb} indep. city-days]")
        else:
            lines.append("    EDGE: not enough resolved markets yet (need >=8)")

        g = report["edge"].get(name, {}).get("graduation")
        if g:
            tag = "GRADUATED" if g["graduated"] else "HOLD"
            report["strategies"][name]["graduation_line"] = (
                f"  gate: {tag}  (clustered P&L p={g['pnl_p_value']}, "
                f"top3={g['concentration']}, blocks={g['n_blocks']})"
                + ("" if g["graduated"] else "  — " + "; ".join(g["reasons"])))
            lines.append("    " + report["strategies"][name]["graduation_line"])

    # data coverage footer
    cov = conn.execute("""
        SELECT (SELECT COUNT(*) FROM markets) m,
               (SELECT COUNT(*) FROM markets WHERE result IS NOT NULL) settled,
               (SELECT COUNT(*) FROM snapshots) snaps,
               (SELECT COUNT(*) FROM forecasts) fc,
               (SELECT COUNT(*) FROM runs WHERE kind='collect') collects
    """).fetchone()
    lines += ["", "-" * 78,
              f"data: {cov['m']} markets ({cov['settled']} resolved), {cov['snaps']} snapshots, "
              f"{cov['fc']} forecasts, {cov['collects']} collect cycles", "=" * 78, ""]
    report["coverage"] = dict(cov)

    if write_csv:
        Path(ARTIFACTS_DIR).mkdir(exist_ok=True)
        pd.DataFrame(report["strategies"]).T.to_csv(ARTIFACTS_DIR / "strategy_performance.csv")
        if not sig_df.empty:
            sig_df.to_csv(ARTIFACTS_DIR / "resolved_signals.csv", index=False)
        pd.read_sql_query("SELECT * FROM equity ORDER BY ts", conn).to_csv(
            ARTIFACTS_DIR / "equity_curve.csv", index=False)

    conn.close()
    report["text"] = "\n".join(lines)
    return report
