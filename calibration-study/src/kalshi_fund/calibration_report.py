from __future__ import annotations

import pandas as pd


def render_report(edge_map: pd.DataFrame, n_markets: int, n_observations: int) -> str:
    lines = [
        "# Kalshi Category Calibration Report",
        "",
        f"Universe: {n_markets:,} markets, {n_observations:,} price observations "
        "(bid/ask midpoints; spreads > $0.05 excluded from headline cells).",
        "",
        "Graduation rule: a (category, bucket, horizon) cell graduates only if the same",
        "side's bootstrap 95% EV interval (clustered by event) excludes zero in the same",
        "direction in both time halves.",
        "",
        "## Graduated cells",
        "",
    ]
    graduated = edge_map[edge_map["graduated_side"].isin(["yes", "no"])]
    if graduated.empty:
        lines.append("None. No cell shows stable, fee-surviving miscalibration —")
        lines.append("the experiment's pre-registered failure condition is met.")
    else:
        lines.append("| Category | Bucket | Horizon (d) | Half | n | Win rate | Side | EV | EV 95% CI |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, row in graduated.sort_values(
            ["category", "bucket", "horizon_days", "half"]
        ).iterrows():
            side = row["graduated_side"]
            ev = row["ev_yes"] if side == "yes" else row["ev_no"]
            low = row["ev_yes_low"] if side == "yes" else row["ev_no_low"]
            high = row["ev_yes_high"] if side == "yes" else row["ev_no_high"]
            lines.append(
                f"| {row['category']} | {row['bucket']} | {row['horizon_days']} "
                f"| {row['half']} | {row['n_obs']} | {row['win_rate']:.3f} "
                f"| buy {side.upper()} | {ev:+.3f} | [{low:+.3f}, {high:+.3f}] |"
            )
    lines += [
        "",
        "## Full edge map",
        "",
        f"{len(edge_map)} cells with enough observations; see calibration_edge_map.csv.",
        "",
    ]
    return "\n".join(lines)
