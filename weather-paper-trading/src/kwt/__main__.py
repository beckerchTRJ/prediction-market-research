"""Command-line interface.

    python -m kwt initdb              # create the tracking database
    python -m kwt collect             # one collection cycle (snapshot + trade)
    python -m kwt settle              # resolve matured markets, realize P&L
    python -m kwt report              # print performance + edge report
    python -m kwt status              # quick coverage / liveness check
    python -m kwt live-mm             # ONE live market-making cycle (dry_run default)
    python -m kwt live-status         # live orders / fills / fill-audit summary
    python -m kwt kill-live-mm        # cancel all resting live orders (panic button)
"""
from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_DB, load_config
from .db import connect, init_db


def cmd_initdb(args):
    init_db(args.db)
    print(f"initialized {args.db}")


def cmd_collect(args):
    from .collect import collect
    collect(db_path=args.db, verbose=not args.quiet)


def cmd_settle(args):
    from .settle import settle
    settle(db_path=args.db, verbose=not args.quiet)


def cmd_report(args):
    from .report import build_report
    rep = build_report(db_path=args.db, write_csv=not args.no_csv)
    print(rep["text"])


def cmd_calibrate(args):
    from .calibrate import calibration_check
    print(calibration_check(db_path=args.db)["text"])


def cmd_dashboard(args):
    from .dashboard import serve
    serve(db_path=args.db, host=args.host, port=args.port)


def cmd_live_mm(args):
    from .live_engine import run_live_mm
    if args.mode == "prod" and not args.i_understand_real_money:
        print("REFUSING: --mode prod places REAL-MONEY orders. Re-run with "
              "--i-understand-real-money once you have funded a small account, "
              "set KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH, and reviewed the "
              "live.risk caps in config/config.yaml.")
        return
    run_live_mm(mode=args.mode, db_path=args.db)


def cmd_live_status(args):
    conn = connect(args.db)

    def scalar(q):
        try:
            return conn.execute(q).fetchone()[0]
        except Exception:
            return "—"

    total = scalar("SELECT COUNT(*) FROM live_orders")
    live = scalar("SELECT COUNT(*) FROM live_orders WHERE status IN ('resting','filled')")
    blocked = scalar("SELECT COUNT(*) FROM live_orders WHERE status='blocked'")
    print(f"  live_orders : {total} total (resting/filled: {live}, blocked: {blocked})")
    fill_rows = scalar("SELECT COUNT(*) FROM live_fills WHERE COALESCE(count,0)>0")
    fill_contracts = scalar("SELECT COALESCE(SUM(count),0) FROM live_fills WHERE COALESCE(count,0)>0")
    print(f"  live_fills  : {fill_rows} nonzero rows / {fill_contracts} contracts")
    print(f"  fill_audit  : {scalar('SELECT COUNT(*) FROM mm_fill_audit')} intervals")

    # Aggregate (volume-weighted) participation — kept, but note it's dominated by
    # a few busy intervals, so ALSO report the per-interval distribution, which is
    # the honest read of "does a resting quote fill as the sim predicts?".
    sim = scalar("SELECT COALESCE(SUM(sim_fill_qty),0) FROM mm_fill_audit")
    act = scalar("SELECT COALESCE(SUM(actual_fill_qty),0) FROM mm_fill_audit")
    sim_adv = scalar("SELECT COALESCE(SUM(sim_fill_qty_incl_adverse),0) FROM mm_fill_audit")
    ratio = f"  (ratio {act / sim:.2f})" if isinstance(sim, (int, float)) and sim else ""
    print(f"  participation: sim {sim} (pessimistic {sim_adv}) vs actual {act} ctr{ratio}")

    rows = conn.execute(
        "SELECT sim_fill_qty s, actual_fill_qty a FROM mm_fill_audit "
        "WHERE COALESCE(sim_fill_qty,0) > 0").fetchall()
    if rows:
        ratios = sorted((r["a"] or 0) / r["s"] for r in rows)
        med = ratios[len(ratios) // 2]
        hit = sum(1 for r in rows if (r["a"] or 0) >= 1) / len(rows)
        print(f"  per-interval: median actual/sim {med:.2f}, "
              f"P(filled | sim>0) {hit:.0%}  [n={len(rows)} intervals w/ sim>0]")
    print(f"  print_volume: {scalar('SELECT COUNT(*) FROM mm_fill_audit WHERE COALESCE(print_vol,0)>0')} "
          "quote intervals with public trade volume")

    queue = conn.execute(
        "SELECT COUNT(*) n,AVG(queue_ahead) ahead,AVG(quote_age_sec) age "
        "FROM live_queue_observations").fetchone()
    if queue and queue["n"]:
        print(f"  queue       : {queue['n']} snapshots, ahead {queue['ahead']:.2f} ctr, "
              f"quote age {queue['age']:.0f}s")

    # Coverage: report one-sided and two-sided resting separately. The old
    # rested_both denominator hides the active ask-only treatment.
    cov = conn.execute("SELECT COUNT(*) n, "
                       "COALESCE(SUM(COALESCE(rested_any, CASE WHEN rested_both=1 OR "
                       "COALESCE(skip_reason,'')='' THEN 1 ELSE 0 END)),0) a, "
                       "COALESCE(SUM(rested_both),0) r "
                       "FROM mm_quote_log").fetchone()
    if cov and cov["n"]:
        print(f"  coverage    : rested any side {cov['a']}/{cov['n']} decisions "
              f"({cov['a'] / cov['n']:.0%}); both sides {cov['r']}/{cov['n']} "
              f"({cov['r'] / cov['n']:.0%})")

    # Fill quality (the adverse-selection read): effective spread captured vs mid,
    # and realized settlement P&L per contract where markets have resolved.
    q = conn.execute(
        "SELECT AVG(effective_spread) es, COUNT(effective_spread) nes, "
        "AVG(pnl_per_contract) pnl, COUNT(pnl_per_contract) npnl FROM mm_fill_pnl "
        "WHERE is_taker = 0").fetchone()
    if q and q["nes"]:
        print(f"  fill_quality: eff.spread {q['es']:+.3f}/ctr [n={q['nes']}]"
              + (f", settled P&L {q['pnl']:+.3f}/ctr [n={q['npnl']}]" if q["npnl"] else ""))
    from .markout import net_markout_evidence
    active = conn.execute("SELECT cohort_id FROM experiment_cohorts ORDER BY activated_ts DESC LIMIT 1").fetchone()
    if active:
        evidence = net_markout_evidence(conn, active["cohort_id"])
        settlement = evidence["settlement"]
        mean = settlement["mean"]
        print(f"  profit_gate : {settlement['status']}  settled dates {settlement['n_dates']}/20"
              + (f", mean {mean:+.3f}/ctr" if mean is not None else ""))
    print(f"  risk_events : {scalar('SELECT COUNT(*) FROM live_risk_events')}")
    last = conn.execute("SELECT ts, detail FROM runs WHERE kind='live-mm' "
                        "ORDER BY id DESC LIMIT 1").fetchone()
    if last:
        print(f"  last live-mm: {last['ts']}  {last['detail']}")
    conn.close()


def cmd_kill_live_mm(args):
    import os
    from .clients.kalshi_trading import KalshiTradingClient
    from .live_engine import (_cancel_all_resting, _experiment_prefixes,
                              KillSwitchError)
    from .config import utcnow_iso, load_config
    if args.mode == "dry_run":
        print("Nothing to cancel in dry_run (no orders are ever placed).")
        return
    # The panic button must work STANDALONE. If creds aren't already in the env,
    # load them from the standard key files (same as run_live_mm.sh) — otherwise a
    # bare `kwt kill-live-mm --mode prod` silently cancels nothing while orders
    # stay live, which is exactly what you must never let a kill switch do.
    key_id = os.path.expanduser("~/.kalshi/key_id")
    key_pem = os.path.expanduser("~/.kalshi/kalshi_key.pem")
    if not os.environ.get("KALSHI_API_KEY_ID") and os.path.exists(key_id):
        with open(key_id) as fh:
            os.environ["KALSHI_API_KEY_ID"] = fh.read().strip()
    if not os.environ.get("KALSHI_PRIVATE_KEY_PATH") and os.path.exists(key_pem):
        os.environ["KALSHI_PRIVATE_KEY_PATH"] = key_pem

    trading = KalshiTradingClient(mode=args.mode)
    if not trading.authenticated:
        print("ERROR: not authenticated — NO ORDERS CANCELLED. Set KALSHI_API_KEY_ID "
              "/ KALSHI_PRIVATE_KEY_PATH, or place keys at ~/.kalshi/key_id and "
              "~/.kalshi/kalshi_key.pem, then retry.")
        raise SystemExit(2)
    conn = connect(args.db)
    # FIREWALL: the panic button cancels ONLY the experiment's own resting orders,
    # never other positions in the same account (e.g. turnout election orders).
    prefixes = _experiment_prefixes(load_config())
    try:
        n = _cancel_all_resting(conn, trading, utcnow_iso(), prefixes, verbose=True,
                                raise_on_error=True)
    except KillSwitchError as e:
        conn.commit()
        conn.close()
        print(f"ERROR: kill FAILED — {e}. Retry, or cancel manually on Kalshi NOW.")
        raise SystemExit(3)
    conn.commit()
    conn.close()
    print(f"kill-live-mm complete: {n} order(s) cancelled.")


def cmd_markouts(args):
    from .markout import compute_markouts, markout_summary, net_markout_evidence
    from .config import utcnow_iso
    conn = connect(args.db)
    n = compute_markouts(conn, tol_sec=int(args.tol_min * 60), ts=utcnow_iso())
    s = markout_summary(conn)
    print(f"markout: computed {n} KXHIGH fills")

    def line(label, a):
        def f(x):
            return "  —  " if x is None else f"{x:+.3f}"
        print(f"  {label:12} n={a['n']:<4} 15m {f(a['mean_15'])}  30m {f(a['mean_30'])}  "
              f"60m {f(a['mean_60'])}  settle {f(a['mean_settle'])}")

    line("OVERALL", s["overall"])
    for grp in ("by_direction", "by_city", "by_hour"):
        print(f"— {grp.replace('by_', 'by ')} —")
        for label in sorted(s[grp]):
            line(label, s[grp][label])
    cohort = conn.execute(
        "SELECT cohort_id FROM experiment_cohorts ORDER BY activated_ts DESC LIMIT 1"
    ).fetchone()
    cohort_id = cohort["cohort_id"] if cohort else "legacy"
    net = net_markout_evidence(conn, cohort_id)
    settlement = net["settlement"]
    print(f"  NET {cohort_id}: 15m {net['mean_net_15'] if net['mean_net_15'] is not None else '—'}  "
              f"60m {net['mean_net_60'] if net['mean_net_60'] is not None else '—'}  "
              f"dates {net['n_dates']}/10  status {net['status']}")
    print(f"  PROFIT {cohort_id}: settle {net['mean_net_settle'] if net['mean_net_settle'] is not None else '—'}  "
          f"dates {settlement['n_dates']}/20  status {settlement['status']}")
    conn.close()


def cmd_status(args):
    conn = connect(args.db)
    for tbl in ("markets", "snapshots", "forecasts", "signals", "trades", "positions"):
        try:
            n = conn.execute(f"SELECT COUNT(*) c FROM {tbl}").fetchone()["c"]
        except Exception:
            n = "—"
        print(f"  {tbl:>12}: {n}")
    last = conn.execute("SELECT ts, kind FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    if last:
        print(f"  last run: {last['kind']} @ {last['ts']}")
    conn.close()


def main():
    p = argparse.ArgumentParser(prog="kwt", description="Kalshi weather paper-trading harness")
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("initdb").set_defaults(func=cmd_initdb)
    c = sub.add_parser("collect"); c.add_argument("--quiet", action="store_true"); c.set_defaults(func=cmd_collect)
    s = sub.add_parser("settle"); s.add_argument("--quiet", action="store_true"); s.set_defaults(func=cmd_settle)
    r = sub.add_parser("report"); r.add_argument("--no-csv", action="store_true"); r.set_defaults(func=cmd_report)
    sub.add_parser("calibrate").set_defaults(func=cmd_calibrate)
    d = sub.add_parser("dashboard")
    d.add_argument("--host", default="127.0.0.1"); d.add_argument("--port", type=int, default=8787)
    d.set_defaults(func=cmd_dashboard)
    sub.add_parser("status").set_defaults(func=cmd_status)

    mo = sub.add_parser("markouts", help="compute + summarize per-fill markouts")
    mo.add_argument("--tol-min", type=float, default=10.0,
                    help="snapshot-match tolerance in minutes (default 10)")
    mo.set_defaults(func=cmd_markouts)

    lm = sub.add_parser("live-mm", help="one live market-making cycle")
    lm.add_argument("--mode", choices=["dry_run", "demo", "prod"], default="dry_run")
    lm.add_argument("--i-understand-real-money", action="store_true",
                    help="required to actually place real-money orders in --mode prod")
    lm.set_defaults(func=cmd_live_mm)

    sub.add_parser("live-status").set_defaults(func=cmd_live_status)

    kl = sub.add_parser("kill-live-mm", help="cancel all resting live orders")
    kl.add_argument("--mode", choices=["dry_run", "demo", "prod"], default="prod")
    kl.set_defaults(func=cmd_kill_live_mm)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
