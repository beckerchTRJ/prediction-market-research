"""Offline forecast-calibration check: verify the bucket machinery BEFORE
trusting it with (paper) money.

Every collected forecast distribution is scored against the realized settlement
high via the probability integral transform (PIT): the fraction of ensemble
members at or below the observed value. If the distributions are calibrated,
PIT values are uniform on [0,1]. Systematic deviations diagnose exactly the
failure modes that hurt trading:

  * U-shaped PIT (mass piled at 0 and 1)  -> distribution too NARROW.
  * hump-shaped PIT (mass piled mid)      -> distribution too WIDE — fat tails
    manufacturing probability in buckets the market prices at pennies, i.e.
    the cheap-YES bleed.
  * skewed PIT                            -> biased mean.

Run `python -m kwt calibrate` as observations accrue; tune tail_fatten_df /
blend_empirical until the PIT is flat rather than guessing.
"""
from __future__ import annotations

import json
import math
import sqlite3

from .config import DEFAULT_DB
from .db import connect


def _pit(members: list[float], mean: float | None, std: float | None,
         observed: float) -> float | None:
    if members:
        n = len(members)
        below = sum(1 for m in members if m < observed)
        equal = sum(1 for m in members if m == observed)
        return (below + 0.5 * equal) / n
    if mean is not None and std and std > 0:   # climatology rows store no members
        z = (observed - mean) / std
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return None


def _horizon_bucket(h: float) -> str:
    if h <= 1:
        return "<=1d"
    if h <= 3:
        return "2-3d"
    return ">3d"


def calibration_check(db_path=DEFAULT_DB) -> dict:
    """PIT histogram + central-interval coverage per forecast source/horizon."""
    conn = connect(db_path)
    rows = conn.execute("""
        WITH first_fc AS (
            SELECT f.*, ROW_NUMBER() OVER (
                PARTITION BY f.city, f.target_date, f.source, f.horizon_days
                ORDER BY f.ts ASC) rn
            FROM forecasts f
        )
        SELECT f.city, f.target_date, f.source, f.horizon_days,
               f.mean, f.std, f.members_json, o.observed_high
        FROM first_fc f
        JOIN observations o ON o.city = f.city AND o.target_date = f.target_date
        WHERE f.rn = 1 AND o.observed_high IS NOT NULL
    """).fetchall()
    conn.close()

    groups: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        try:
            members = json.loads(r["members_json"] or "[]")
        except (TypeError, ValueError):
            members = []
        pit = _pit(members, r["mean"], r["std"], r["observed_high"])
        if pit is None:
            continue
        hb = "clim" if r["source"] == "climatology" else _horizon_bucket(r["horizon_days"])
        groups.setdefault((r["source"], hb), []).append(pit)

    out: dict = {"groups": {}}
    lines = ["", "FORECAST CALIBRATION (PIT vs realized settlement highs)",
             "-" * 70,
             "flat histogram = calibrated; ends heavy = too narrow; middle heavy",
             "= too wide (fat tails -> the cheap-longshot bleed)", ""]
    for (source, hb), pits in sorted(groups.items()):
        n = len(pits)
        bins = [0] * 10
        for p in pits:
            bins[min(int(p * 10), 9)] += 1
        hist = " ".join(f"{b / n:.2f}" for b in bins)
        cov80 = sum(1 for p in pits if 0.10 <= p <= 0.90) / n
        tail_mass = (bins[0] + bins[9]) / n
        out["groups"][f"{source}|{hb}"] = {
            "n": n, "pit_hist": [round(b / n, 3) for b in bins],
            "coverage_10_90": round(cov80, 3), "tail_mass": round(tail_mass, 3)}
        lines.append(f"{source:>16} {hb:>5}  n={n:<4} 10-90% coverage={cov80:.2f} "
                     f"(target 0.80)  ends={tail_mass:.2f} (target 0.20)")
        lines.append(f"{'':>23} PIT [{hist}]")
    if not groups:
        lines.append("no resolved (forecast, observation) pairs yet — run settle first")
    out["text"] = "\n".join(lines)
    return out
