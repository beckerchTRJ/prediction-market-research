"""Per-fill markout: the signed change in the market mid at fixed horizons after a
fill. Positive = the market moved in our favor after we traded (benign flow);
negative = it moved against us (adverse selection). Measured against the mid AT the
fill (isolating the market's move, independent of the spread we paid).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from scipy import stats


def fill_direction(book_side) -> int | None:
    """Fill direction from Kalshi's reliable side-of-book: `bid` = our resting bid was
    hit = we bought YES = +1; `ask` = our ask was lifted = we sold YES = -1. None if
    book_side is missing (can't attribute).

    NOTE: (side, action) does NOT work here — Kalshi encodes a maker ASK fill as
    (side=no, action=sell), which naively reads as long-YES but is actually short.
    Empirically confirmed: every ask fill is (no, sell), every bid fill is (yes, buy).
    """
    if book_side == "bid":
        return 1
    if book_side == "ask":
        return -1
    return None


def snapshot_mid(row) -> float | None:
    yb, ya = row["yes_bid"], row["yes_ask"]
    if yb and ya:
        return (yb + ya) / 2.0
    lp = row["last_price"]
    return lp if lp else None


def signed_markout(direction: int, mid_at_fill, later_mid) -> float | None:
    if mid_at_fill is None or later_mid is None:
        return None
    return direction * (later_mid - mid_at_fill)


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _shift_iso(iso: str, seconds: float) -> str:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00")) + timedelta(seconds=seconds)
    return dt.isoformat().replace("+00:00", "Z")


def nearest_mid(conn, ticker: str, target_iso: str, tol_sec: int = 600) -> float | None:
    """yes-mid of the snapshot whose ts is closest to target_iso within tol_sec;
    None if the nearest is outside tolerance or has no usable price."""
    lo, hi = _shift_iso(target_iso, -tol_sec), _shift_iso(target_iso, tol_sec)
    rows = conn.execute(
        "SELECT ts, yes_bid, yes_ask, last_price FROM snapshots "
        "WHERE ticker=? AND ts>=? AND ts<=?", (ticker, lo, hi)).fetchall()
    if not rows:
        return None
    target = _epoch(target_iso)
    best = min(rows, key=lambda r: abs(_epoch(r["ts"]) - target))
    return snapshot_mid(best)


def latest_mid(conn, ticker: str, target_iso: str, tol_sec: int = 600) -> float | None:
    """Latest usable mid at or before a timestamp; never look forward from a fill."""
    lo = _shift_iso(target_iso, -tol_sec)
    rows = conn.execute(
        "SELECT ts,yes_bid,yes_ask,last_price FROM snapshots WHERE ticker=? "
        "AND ts>=? AND ts<=? ORDER BY ts DESC", (ticker, lo, target_iso)).fetchall()
    return snapshot_mid(rows[0]) if rows else None


def upsert_markout(conn, row: dict) -> None:
    """Idempotent write by fill_id — the single path shared by the offline batch
    and (future) live per-cycle tracker."""
    conn.execute(
        "INSERT OR REPLACE INTO live_fill_markouts (fill_id, ticker, created_time, "
        "direction, mid_at_fill, mo_15, mo_30, mo_60, mo_settle, computed_ts) "
        "VALUES (:fill_id,:ticker,:created_time,:direction,:mid_at_fill,:mo_15,:mo_30,"
        ":mo_60,:mo_settle,:computed_ts)", row)


def compute_markouts(conn, horizons=(15, 30, 60), tol_sec: int = 600, ts: str = "") -> int:
    """Batch-compute markouts for every KXHIGH fill into live_fill_markouts. Returns
    the number of fills processed. `ts` stamps computed_ts (inject for determinism)."""
    fills = conn.execute(
        "SELECT lf.fill_id, lf.ticker, lf.book_side, lf.created_time, m.result "
        "FROM live_fills lf LEFT JOIN markets m ON m.ticker=lf.ticker "
        "WHERE lf.ticker LIKE 'KXHIGH%' AND lf.created_time IS NOT NULL "
        "AND COALESCE(lf.count,0)>0").fetchall()
    hcols = {15: "mo_15", 30: "mo_30", 60: "mo_60"}
    processed = 0
    for f in fills:
        d = fill_direction(f["book_side"])
        if d is None:
            continue                         # no book_side -> can't attribute; skip
        processed += 1
        maf = latest_mid(conn, f["ticker"], f["created_time"], tol_sec)
        row = {"fill_id": f["fill_id"], "ticker": f["ticker"],
               "created_time": f["created_time"], "direction": d, "mid_at_fill": maf,
               "mo_15": None, "mo_30": None, "mo_60": None, "mo_settle": None,
               "computed_ts": ts}
        for h in horizons:
            later = nearest_mid(conn, f["ticker"],
                                _shift_iso(f["created_time"], h * 60), tol_sec)
            row[hcols.get(h, f"mo_{h}")] = signed_markout(d, maf, later)
        if f["result"] in ("yes", "no"):
            outcome = 1.0 if f["result"] == "yes" else 0.0
            row["mo_settle"] = signed_markout(d, maf, outcome)
        upsert_markout(conn, row)
    conn.commit()
    return processed


def _agg(rows) -> dict:
    def m(col):
        vals = [r[col] for r in rows if r[col] is not None]
        return float(np.mean(vals)) if vals else None
    return {"n": len(rows), "mean_15": m("mo_15"), "mean_30": m("mo_30"),
            "mean_60": m("mo_60"), "mean_settle": m("mo_settle")}


def markout_summary(conn) -> dict:
    """Aggregate live_fill_markouts overall and sliced by direction (long/short
    YES-equiv), city (ticker series), and UTC hour-of-day. Means ignore NULLs."""
    rows = conn.execute(
        "SELECT mo.* FROM live_fill_markouts mo LEFT JOIN live_fills f ON f.fill_id=mo.fill_id "
        "WHERE f.fill_id IS NULL OR COALESCE(f.count,0)>0").fetchall()
    def group(keyfn):
        buckets: dict = {}
        for r in rows:
            buckets.setdefault(keyfn(r), []).append(r)
        return {k: _agg(v) for k, v in buckets.items()}
    return {
        "overall": _agg(rows),
        "by_direction": group(lambda r: "long" if r["direction"] == 1 else "short"),
        "by_city": group(lambda r: r["ticker"].split("-")[0].replace("KXHIGH", "")),
        "by_hour": group(lambda r: (r["created_time"] or "")[11:13]),
    }


def net_markout_evidence(conn, cohort_id: str | None = None,
                         min_dates: int = 10) -> dict:
    """Spread-plus-markout evidence, descriptive by contract and tested by date."""
    params: tuple = ()
    where = "WHERE v.book_side IS NOT NULL AND COALESCE(f.count,0)>0"
    if cohort_id:
        where += " AND COALESCE(f.cohort_id,'legacy')=?"
        params = (cohort_id,)
    rows = conn.execute(
        "SELECT f.fill_id,f.ticker,f.book_side,f.count,f.created_time,"
        "COALESCE(f.cohort_id,'legacy') cohort_id,m.city,m.target_date,v.yes_price,v.effective_spread,"
        "COALESCE(v.fee_per_ct,0) fee_per_ct,mo.mid_at_fill mark_mid,"
        "mo.mo_15,mo.mo_30,mo.mo_60,mo.mo_settle "
        "FROM mm_fill_pnl v JOIN live_fills f ON f.id=v.id "
        "LEFT JOIN live_fill_markouts mo ON mo.fill_id=f.fill_id "
        "LEFT JOIN markets m ON m.ticker=f.ticker " + where + " ORDER BY f.created_time",
        params).fetchall()
    detail = []
    for r in rows:
        rec = dict(r)
        spread = None
        if r["mark_mid"] is not None:
            spread = ((float(r["yes_price"]) - float(r["mark_mid"]))
                      if r["book_side"] == "ask" else
                      (float(r["mark_mid"]) - float(r["yes_price"])))
        for h in (15, 30, 60, "settle"):
            mo = r[f"mo_{h}"]
            rec[f"net_{h}"] = (spread + float(mo) - float(r["fee_per_ct"])) \
                if spread is not None and mo is not None else None
        price = float(r["yes_price"])
        rec["price_band"] = ("<.12" if price < .12 else ".12-.25" if price < .25
                             else ".25-.50" if price < .50 else ">=.50")
        detail.append(rec)

    def weighted(field, subset):
        valid = [x for x in subset if x[field] is not None]
        denom = sum(float(x["count"] or 0) for x in valid)
        return (sum(float(x[field]) * float(x["count"] or 0) for x in valid) / denom
                if denom else None)

    by_side = {side: {f"net_{h}": weighted(f"net_{h}",
                                            [x for x in detail if x["book_side"] == side])
                      for h in (15, 30, 60, "settle")}
               for side in ("ask", "bid")}
    def date_stats(field):
        values = []
        dates = sorted({x["target_date"] for x in detail
                        if x["target_date"] and x[field] is not None})
        for day in dates:
            values.append({"target_date": day,
                           field: weighted(field, [x for x in detail if x["target_date"] == day])})
        vals = np.asarray([x[field] for x in values], float)
        mean = float(vals.mean()) if vals.size else None
        se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size >= 2 and vals.std(ddof=1) > 0 else None
        p = float(stats.t.sf(mean / se, df=vals.size - 1)) if se else None
        upper80 = float(mean + stats.t.ppf(.80, vals.size - 1) * se) if se else None
        projected = None
        if vals.size >= 5 and mean is not None and mean > 0 and vals.std(ddof=1) > 0:
            projected = int(np.ceil(
                ((1.644854 + .841621) * vals.std(ddof=1) / mean) ** 2))
        return values, {"n_dates": int(vals.size), "mean": mean, "p_value": p,
                        "upper_80": upper80, "projected_dates_80pct": projected}

    date_values, net60 = date_stats("net_60")
    settlement_values, settle = date_stats("net_settle")
    status = "collecting" if net60["n_dates"] < min_dates else (
        "positive" if net60["p_value"] is not None and net60["p_value"] < .05
        and net60["mean"] > 0 else "inconclusive")
    if settle["n_dates"] >= 20 and settle["mean"] is not None and settle["mean"] > 0 \
            and settle["p_value"] is not None and settle["p_value"] < .05:
        profitability_status = "expand"
    elif settle["n_dates"] >= 10 and settle["upper_80"] is not None and settle["upper_80"] <= 0:
        profitability_status = "futility"
    else:
        profitability_status = "collecting" if settle["n_dates"] < 20 else "inconclusive"
    return {"cohort_id": cohort_id, "n_fills": len(detail), "n_dates": net60["n_dates"],
            "mean_net_15": weighted("net_15", detail),
            "mean_net_30": weighted("net_30", detail),
            "mean_net_60": weighted("net_60", detail),
            "mean_net_settle": weighted("net_settle", detail),
            "p_value": net60["p_value"], "status": status,
            "projected_dates_80pct": net60["projected_dates_80pct"], "by_side": by_side,
            "by_date": date_values, "settlement": {**settle, "by_date": settlement_values,
                                                       "status": profitability_status},
            "fills": detail}
