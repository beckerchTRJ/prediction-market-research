from __future__ import annotations

import json
from datetime import datetime

import pandas as pd


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _iso_to_ts(value: str | None) -> int | None:
    if not value:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def resolve_series_ticker(event_series: dict[str, str], event_ticker: str) -> str:
    return event_series.get(event_ticker) or event_ticker.split("-")[0]


def build_settled_market_frame(
    markets: list[dict],
    event_series: dict[str, str],
    series_categories: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for market in markets:
        if market.get("result") not in ("yes", "no"):
            continue
        event_ticker = market.get("event_ticker", "")
        series_ticker = resolve_series_ticker(event_series, event_ticker)
        rows.append(
            {
                "ticker": market["ticker"],
                "event_ticker": event_ticker,
                "series_ticker": series_ticker,
                "category": series_categories.get(series_ticker) or "unknown",
                "title": market.get("title"),
                "market_type": market.get("market_type"),
                "result": market["result"],
                "settlement_value_dollars": _to_float(market.get("settlement_value_dollars")),
                "open_time": market.get("open_time"),
                "close_time": market.get("close_time"),
                "close_ts": _iso_to_ts(market.get("close_time")),
                "volume": _to_float(market.get("volume_fp") or market.get("volume")),
                "raw_json": json.dumps(market, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)
