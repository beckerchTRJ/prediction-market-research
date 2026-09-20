from __future__ import annotations

import math

import numpy as np
import pandas as pd

from kalshi_fund.fees import taker_fee_usd

BUCKET_EDGES = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 1.0]


def assign_price_bucket(mid: float) -> str:
    for low, high in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        if low <= mid < high or (high == 1.0 and mid == 1.0):
            return f"{low:.2f}-{high:.2f}"
    msg = f"mid price {mid} outside [0, 1]"
    raise ValueError(msg)


def profit_buy_yes(won: bool, ask: float, fee_rate: float = 0.07) -> float:
    fee = taker_fee_usd(ask, rate=fee_rate)
    return (1.0 - ask - fee) if won else (-ask - fee)


def profit_buy_no(won: bool, yes_bid: float, fee_rate: float = 0.07) -> float:
    no_price = 1.0 - yes_bid
    fee = taker_fee_usd(no_price, rate=fee_rate)
    return (-no_price - fee) if won else (1.0 - no_price - fee)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p_hat = successes / n
    denominator = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def build_observation_frame(
    markets: pd.DataFrame, snapshots: pd.DataFrame, max_spread: float = 0.05
) -> pd.DataFrame:
    observations = snapshots.merge(
        markets[["ticker", "event_ticker", "category", "result", "close_ts"]],
        on="ticker",
        how="inner",
    )
    observations = observations.dropna(subset=["yes_bid", "yes_ask", "close_ts"]).copy()
    observations["won"] = observations["result"] == "yes"
    observations["wide_spread"] = observations["spread"] > max_spread
    observations["bucket"] = observations["mid"].map(assign_price_bucket)
    median_close = observations["close_ts"].median()
    observations["half"] = np.where(observations["close_ts"] <= median_close, "early", "late")
    observations["profit_yes"] = [
        profit_buy_yes(won, ask) for won, ask in zip(observations["won"], observations["yes_ask"])
    ]
    observations["profit_no"] = [
        profit_buy_no(won, bid) for won, bid in zip(observations["won"], observations["yes_bid"])
    ]
    return observations[
        [
            "ticker", "event_ticker", "category", "close_ts", "horizon_days",
            "yes_bid", "yes_ask", "mid", "spread", "wide_spread", "won", "bucket",
            "half", "profit_yes", "profit_no",
        ]
    ]


def cluster_bootstrap_mean(
    values, clusters, n_boot: int = 500, seed: int = 0
) -> tuple[float, float]:
    frame = pd.DataFrame({"value": values, "cluster": clusters})
    cluster_means = frame.groupby("cluster")["value"].mean()
    cluster_sizes = frame.groupby("cluster")["value"].size()
    rng = np.random.default_rng(seed)
    n_clusters = len(cluster_means)
    boot_means = np.empty(n_boot)
    means_array = cluster_means.to_numpy()
    sizes_array = cluster_sizes.to_numpy()
    for b in range(n_boot):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        boot_means[b] = np.average(means_array[picks], weights=sizes_array[picks])
    return (float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975)))


def compute_edge_map(
    observations: pd.DataFrame, min_obs: int = 50, n_boot: int = 500, seed: int = 0
) -> pd.DataFrame:
    headline = observations[~observations["wide_spread"]]
    rows = []
    for (category, bucket, horizon, half), group in headline.groupby(
        ["category", "bucket", "horizon_days", "half"]
    ):
        n_obs = len(group)
        if n_obs < min_obs:
            continue
        wins = int(group["won"].sum())
        wilson_low, wilson_high = wilson_interval(wins, n_obs)
        ev_yes_low, ev_yes_high = cluster_bootstrap_mean(
            group["profit_yes"], group["event_ticker"], n_boot=n_boot, seed=seed
        )
        ev_no_low, ev_no_high = cluster_bootstrap_mean(
            group["profit_no"], group["event_ticker"], n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "category": category, "bucket": bucket, "horizon_days": horizon, "half": half,
                "n_obs": n_obs, "win_rate": wins / n_obs,
                "wilson_low": wilson_low, "wilson_high": wilson_high,
                "ev_yes": float(group["profit_yes"].mean()),
                "ev_yes_low": ev_yes_low, "ev_yes_high": ev_yes_high,
                "ev_no": float(group["profit_no"].mean()),
                "ev_no_low": ev_no_low, "ev_no_high": ev_no_high,
            }
        )
    edge_map = pd.DataFrame(
        rows,
        columns=[
            "category", "bucket", "horizon_days", "half", "n_obs", "win_rate",
            "wilson_low", "wilson_high", "ev_yes", "ev_yes_low", "ev_yes_high",
            "ev_no", "ev_no_low", "ev_no_high",
        ],
    )
    if edge_map.empty:
        edge_map["graduated_side"] = pd.Series(dtype=str)
        return edge_map

    def graduated_side(cell: pd.DataFrame) -> str:
        if len(cell) < 2:  # need both halves
            return ""
        if (cell["ev_yes_low"] > 0).all():
            return "yes"
        if (cell["ev_no_low"] > 0).all():
            return "no"
        return ""

    grades = (
        edge_map.groupby(["category", "bucket", "horizon_days"])
        .apply(graduated_side, include_groups=False)
        .rename("graduated_side")
        .reset_index()
    )
    return edge_map.merge(grades, on=["category", "bucket", "horizon_days"], how="left")
