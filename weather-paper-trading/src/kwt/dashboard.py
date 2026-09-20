"""Read-only analytics dashboard for paper and live weather experiments."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from .config import DEFAULT_DB, load_config
from .db import connect, kv_get
from .markout import net_markout_evidence
from .report import build_report

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _rows(conn, sql, params=()):
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


def _scalar(conn, sql, default=0, params=()):
    try:
        value = conn.execute(sql, params).fetchone()[0]
        return default if value is None else value
    except Exception:
        return default


def _live_mm(conn) -> dict:
    sim = float(_scalar(conn, "SELECT COALESCE(SUM(sim_fill_qty),0) FROM mm_fill_audit"))
    actual = float(_scalar(conn, "SELECT COALESCE(SUM(actual_fill_qty),0) FROM mm_fill_audit"))
    coverage = _rows(
        conn,
        "SELECT COUNT(*) decisions, "
        "COALESCE(SUM(COALESCE(rested_any, CASE WHEN rested_both=1 OR "
        "COALESCE(skip_reason,'')='' THEN 1 ELSE 0 END)),0) rested_any, "
        "COALESCE(SUM(rested_both),0) rested_both FROM mm_quote_log")
    coverage = coverage[0] if coverage else {
        "decisions": 0, "rested_any": 0, "rested_both": 0}
    queue = _rows(conn, "SELECT COUNT(*) observations,AVG(queue_ahead) queue_ahead,"
                  "AVG(quote_age_sec) quote_age_sec FROM live_queue_observations")
    queue = queue[0] if queue else {"observations": 0, "queue_ahead": None, "quote_age_sec": None}
    # "Time at cap": share of quote cycles (distinct order timestamps) where a
    # capacity limit blocked at least one order — the capacity-utilization
    # number that says when raising max_capital_at_risk would actually matter.
    cycles = int(_scalar(conn, "SELECT COUNT(DISTINCT ts) FROM live_orders", default=0))
    cycles_at_cap = int(_scalar(
        conn, "SELECT COUNT(DISTINCT ts) FROM live_orders WHERE status='blocked' "
        "AND reason IN ('max_capital_at_risk','max_net_inventory')", default=0))
    return {
        "summary": {
            "orders_total": int(_scalar(conn, "SELECT COUNT(*) FROM live_orders")),
            "orders_live": int(_scalar(conn, "SELECT COUNT(*) FROM live_orders WHERE status IN ('resting','partial')")),
            "orders_blocked": int(_scalar(conn, "SELECT COUNT(*) FROM live_orders WHERE status='blocked'")),
            "intervals": int(_scalar(conn, "SELECT COUNT(*) FROM mm_fill_audit")),
            "sim": round(sim, 2), "actual": round(actual, 2),
            "ratio": round(actual / sim, 3) if sim else None,
            "decisions": int(coverage["decisions"]),
            "rested_any": int(coverage["rested_any"]),
            "rested_both": int(coverage["rested_both"]),
            "intervals_with_print_volume": int(_scalar(
                conn, "SELECT COUNT(*) FROM mm_fill_audit WHERE COALESCE(print_vol,0)>0")),
            "cycles": cycles,
            "cycles_at_cap": cycles_at_cap,
            "at_cap_pct": round(cycles_at_cap / cycles, 3) if cycles else None,
            "queue_observations": int(queue["observations"] or 0),
            "mean_queue_ahead": round(float(queue["queue_ahead"]), 2)
                                if queue["queue_ahead"] is not None else None,
            "mean_quote_age_sec": round(float(queue["quote_age_sec"]), 1)
                                  if queue["quote_age_sec"] is not None else None,
        },
        "by_mode": _rows(conn, "SELECT mode,COUNT(*) intervals,ROUND(SUM(sim_fill_qty),2) sim,"
                         "ROUND(SUM(actual_fill_qty),2) actual FROM mm_fill_audit GROUP BY mode"),
        "by_city": _rows(conn, "SELECT m.city,COUNT(*) intervals,ROUND(SUM(a.sim_fill_qty),2) sim,"
                         "ROUND(SUM(a.actual_fill_qty),2) actual FROM mm_fill_audit a "
                         "LEFT JOIN markets m ON m.ticker=a.ticker GROUP BY m.city ORDER BY m.city"),
        "recent_orders": _rows(conn, "SELECT ts,cohort_id,mode,ticker,side,price,count,status,reason "
                               "FROM live_orders ORDER BY id DESC LIMIT 40"),
        "flatten": _rows(conn, "SELECT ts,cohort_id,kind,ticker,detail FROM live_risk_events "
                         "WHERE kind LIKE 'flatten%' ORDER BY id DESC LIMIT 40"),
    }


def overview(db_path: str) -> dict:
    rep = build_report(db_path=db_path, write_csv=False)
    conn = connect(db_path)
    equity_rows = _rows(conn, "SELECT ts,strategy,equity FROM equity ORDER BY ts")
    equity: dict[str, list] = {}
    for row in equity_rows:
        equity.setdefault(row["strategy"], []).append(
            {"ts": row["ts"], "equity": round(float(row["equity"]), 2)})
    latest = conn.execute("SELECT MAX(ts) FROM runs").fetchone()[0]
    cohorts = _rows(conn, "SELECT cohort_id,activated_ts,primary_metric,horizon_minutes,status "
                    "FROM experiment_cohorts ORDER BY activated_ts")
    latest_pos_ts = _scalar(conn, "SELECT MAX(ts) FROM live_positions", None)
    pos = _rows(conn, "SELECT ticker,position,market_exposure,realized_pnl FROM live_positions "
                "WHERE ts=? ORDER BY ticker", (latest_pos_ts,)) if latest_pos_ts else []
    conn.close()
    return {"generated_at": latest, "active_cohort": cohorts[-1]["cohort_id"] if cohorts else "legacy",
            "cohorts": cohorts, "coverage": rep["coverage"],
            "strategies": rep["strategies"], "edge": rep["edge"],
            "equity": equity, "positions": pos}


def evidence(db_path: str) -> dict:
    rep = build_report(db_path=db_path, write_csv=False)
    fades = {}
    for name, edge in rep["edge"].items():
        if "fade_null" in edge:
            fades[name] = {"null": edge["fade_null"],
                           "graduation": edge.get("graduation", {})}
    return {"fades": fades, "edge": rep["edge"]}


def markouts(db_path: str) -> dict:
    conn = connect(db_path)
    cohorts = [r[0] for r in conn.execute(
        "SELECT cohort_id FROM experiment_cohorts ORDER BY activated_ts").fetchall()]
    if "legacy" not in cohorts:
        cohorts.insert(0, "legacy")
    data = {cohort: net_markout_evidence(conn, cohort) for cohort in cohorts}
    conn.close()
    return {"cohorts": data}


def execution(db_path: str) -> dict:
    conn = connect(db_path)
    data = _live_mm(conn)
    data["settlement"] = _rows(
        conn, "SELECT COALESCE(f.cohort_id,'legacy') cohort_id,f.book_side,COUNT(*) fills,"
        "SUM(f.count) contracts,ROUND(SUM(v.pnl_per_contract*f.count),4) pnl "
        "FROM mm_fill_pnl v JOIN live_fills f ON f.id=v.id WHERE v.result IS NOT NULL "
        "AND f.book_side IS NOT NULL AND COALESCE(f.count,0)>0 GROUP BY cohort_id,f.book_side")
    conn.close()
    return data


def risk(db_path: str) -> dict:
    conn = connect(db_path)
    latest_ts = _scalar(conn, "SELECT MAX(ts) FROM live_positions", None)
    positions = _rows(conn, "SELECT ticker,position,market_exposure,realized_pnl FROM live_positions "
                      "WHERE ts=? ORDER BY ABS(position) DESC", (latest_ts,)) if latest_ts else []
    resting = _rows(conn, "SELECT ticker,side,price,count FROM live_orders WHERE status IN ('resting','partial')")
    reserved_cost = sum((float(o["price"]) if o["side"] == "bid" else 1-float(o["price"]))
                        * float(o["count"]) for o in resting)
    reserved_bid = sum(float(o["count"]) for o in resting if o["side"] == "bid")
    reserved_ask = sum(float(o["count"]) for o in resting if o["side"] == "ask")
    deployed = sum(abs(float(p["market_exposure"] or 0)) for p in positions)
    net = sum(float(p["position"] or 0) for p in positions)
    blocks = _rows(conn, "SELECT kind,detail,COUNT(*) n FROM live_risk_events "
                   "WHERE ts>=strftime('%Y-%m-%dT%H:%M:%SZ','now','-7 days') "
                   "GROUP BY kind,detail ORDER BY n DESC LIMIT 30")
    recent = _rows(conn, "SELECT ts,cohort_id,kind,ticker,detail FROM live_risk_events "
                   "ORDER BY id DESC LIMIT 60")
    conn.close()
    return {"latest_position_ts": latest_ts, "deployed_capital": deployed,
            "reserved_cost": reserved_cost, "reserved_bid": reserved_bid,
            "reserved_ask": reserved_ask, "net_inventory": net,
            "positions": positions, "blocks": blocks, "recent": recent}


def cities(db_path: str) -> dict:
    cfg = load_config()
    conn = connect(db_path)
    out = []
    for code, city in cfg.cities.items():
        market_dates = int(_scalar(conn, "SELECT COUNT(DISTINCT target_date) FROM markets WHERE city=?", 0, (code,)))
        resolved = int(_scalar(conn, "SELECT COUNT(DISTINCT target_date) FROM markets WHERE city=? AND result IS NOT NULL", 0, (code,)))
        forecast_dates = int(_scalar(conn, "SELECT COUNT(DISTINCT target_date) FROM forecasts WHERE city=?", 0, (code,)))
        markets_n = int(_scalar(conn, "SELECT COUNT(*) FROM markets WHERE city=?", 0, (code,)))
        snapped = int(_scalar(conn, "SELECT COUNT(DISTINCT s.ticker) FROM snapshots s JOIN markets m "
                              "ON m.ticker=s.ticker WHERE m.city=?", 0, (code,)))
        oi = [float(r[0]) for r in conn.execute(
            "SELECT open_interest FROM snapshots s JOIN markets m ON m.ticker=s.ticker "
            "WHERE m.city=? AND open_interest IS NOT NULL ORDER BY open_interest", (code,)).fetchall()]
        median_oi = oi[len(oi)//2] if oi else None
        first_seen = _scalar(conn, "SELECT MIN(first_seen) FROM markets WHERE city=?", None, (code,))
        try:
            burn_in_days = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(
                str(first_seen).replace("Z", "+00:00"))).days) if first_seen else 0
        except (ValueError, TypeError):
            burn_in_days = 0
        parse_failures = int(_scalar(
            conn, "SELECT COUNT(*) FROM markets WHERE series_ticker=? "
            "AND (target_date IS NULL OR bucket_kind IS NULL)", 0, (city["series"],)))
        fc_cov = min(1.0, forecast_dates / market_dates) if market_dates else 0.0
        snap_cov = snapped / markets_n if markets_n else 0.0
        ready = burn_in_days >= 14 and resolved >= 10 and fc_cov >= .95 and snap_cov >= .95 \
            and parse_failures == 0 and median_oi is not None \
            and median_oi >= cfg.raw["live"]["risk"]["min_open_interest"]
        out.append({"code": code, "name": city["name"], "series": city["series"],
                    "station": city["station"], "cli": city["cli"],
                    "live_enabled": city.get("live_enabled", True), "resolved_dates": resolved,
                    "burn_in_days": burn_in_days, "parse_failures": parse_failures,
                    "forecast_coverage": fc_cov, "snapshot_coverage": snap_cov,
                    "median_open_interest": median_oi, "ready": ready})
    conn.close()
    return {"cities": out}


def legacy_payload(db_path: str) -> dict:
    data = overview(db_path)
    conn = connect(db_path)
    data["live_mm"] = _live_mm(conn)
    data["recent_trades"] = _rows(conn, "SELECT ts,strategy,ticker,side,action,contracts,price,fee,reason "
                                  "FROM trades ORDER BY id DESC LIMIT 40")
    data["recent_enters"] = _rows(
        conn, "SELECT ts,strategy,ticker,model_prob,market_prob,edge,side FROM signals "
        "WHERE decision='enter' ORDER BY id DESC LIMIT 40")
    data["open_positions"] = _rows(
        conn, "SELECT strategy,COUNT(*) markets,COALESCE(SUM(contracts),0) contracts,"
        "COALESCE(SUM(cost),0) cost FROM positions GROUP BY strategy")
    conn.close()
    return data


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


API = {"/api/data": legacy_payload, "/api/overview": overview,
       "/api/evidence": evidence, "/api/markouts": markouts,
       "/api/execution": execution, "/api/risk": risk, "/api/cities": cities}


class Handler(BaseHTTPRequestHandler):
    db_path = str(DEFAULT_DB)

    def log_message(self, *args):
        pass

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in API:
            try:
                body = json.dumps(_clean(API[path](self.db_path)), separators=(",", ":")).encode()
                self._send(200, body, "application/json")
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
            return
        if path in ("/", "/index.html"):
            path = "/static/dashboard.html"
        if path.startswith("/static/"):
            name = path[len("/static/"):]
            target = (STATIC_DIR / name).resolve()
            if target.is_file() and STATIC_DIR in target.parents:
                types = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                         ".js": "application/javascript"}
                self._send(200, target.read_bytes(), types.get(target.suffix, "application/octet-stream"))
            else:
                self._send(404, b"not found", "text/plain")
            return
        self._send(404, b"not found", "text/plain")

    def _readonly(self):
        self._send(405, b'{"error":"dashboard is read-only"}', "application/json")

    do_POST = do_PUT = do_PATCH = do_DELETE = _readonly


def serve(db_path: str = str(DEFAULT_DB), host: str = "127.0.0.1", port: int = 8787) -> None:
    Handler.db_path = db_path
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"dashboard: http://{host}:{port}  (read-only; Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
