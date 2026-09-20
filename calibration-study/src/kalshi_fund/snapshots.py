from __future__ import annotations

import random

import pandas as pd

DAY_SECONDS = 86400


def _quote(candle: dict, side: str) -> float | None:
    value = (candle.get(side) or {}).get("close_dollars")
    if value is None or value == "":
        return None
    return float(value)


def extract_horizon_snapshots(
    candles: list[dict],
    close_ts: int,
    horizons: tuple[int, ...] = (1, 3, 7, 30),
    tolerance_seconds: int = 129600,
) -> list[dict]:
    snapshots = []
    for horizon in horizons:
        target_ts = close_ts - horizon * DAY_SECONDS
        best = None
        best_distance = None
        for candle in candles:
            distance = abs(candle["end_period_ts"] - target_ts)
            if distance > tolerance_seconds:
                continue
            if best_distance is None or distance < best_distance:
                best, best_distance = candle, distance
        if best is None:
            continue
        bid = _quote(best, "yes_bid")
        ask = _quote(best, "yes_ask")
        if bid is None or ask is None:
            continue
        volume_raw = best.get("volume_fp")
        snapshots.append(
            {
                "horizon_days": horizon,
                "snapshot_ts": best["end_period_ts"],
                "yes_bid": bid,
                "yes_ask": ask,
                "mid": (bid + ask) / 2.0,
                "spread": ask - bid,
                "volume": float(volume_raw) if volume_raw not in (None, "") else None,
            }
        )
    return snapshots


def sample_tickers_per_category(
    markets: pd.DataFrame, per_category_cap: int, seed: int = 0
) -> list[str]:
    sampled: list[str] = []
    for category, group in markets.groupby("category", dropna=False, sort=True):
        tickers = sorted(group["ticker"])
        if len(tickers) > per_category_cap:
            rng = random.Random(f"{seed}:{category}")
            tickers = sorted(rng.sample(tickers, per_category_cap))
        sampled.extend(tickers)
    return sampled
