from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kalshi_fund.kalshi_api import KalshiApiError, KalshiPublicClient
from kalshi_fund.settled_markets import build_settled_market_frame, resolve_series_ticker
from kalshi_fund.storage import upsert_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull settled Kalshi markets into the cache DB.")
    parser.add_argument("--db", required=True, help="SQLite cache database path.")
    parser.add_argument("--min-settled-ts", type=int, default=None,
                        help="Unix ts lower bound on settlement time (limits pull size).")
    parser.add_argument("--max-markets", type=int, default=None,
                        help="Safety cap on number of markets pulled.")
    parser.add_argument("--request-delay", type=float, default=0.15)
    parser.add_argument("--list-categories", action="store_true",
                        help="List series categories with counts and exit (no DB writes).")
    parser.add_argument("--categories", default=None,
                        help="Comma-separated list of categories to pull (exact match). "
                             "Pulls settled markets per-series instead of the global feed.")
    return parser.parse_args()


def run_list_categories(client: KalshiPublicClient) -> None:
    series_list = client.get_series_list()
    counts: dict[str, int] = {}
    for series in series_list:
        category = series.get("category") or "unknown"
        counts[category] = counts.get(category, 0) + 1
    for category, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{category}: {count}")


def run_categories_pull(
    client: KalshiPublicClient, args: argparse.Namespace, categories: list[str]
) -> None:
    print(f"Fetching series list for categories: {categories}")
    series_list = client.get_series_list()
    selected = [s for s in series_list if s.get("category") in categories]
    matched_categories = {s.get("category") for s in selected}
    for category in categories:
        if category not in matched_categories:
            print(f"  warning: category '{category}' matched 0 series")
    print(f"  {len(selected)} matching series")

    total_markets = 0
    cap_hit = False
    for i, series in enumerate(selected, start=1):
        series_ticker = series.get("ticker", "")
        if not series_ticker:
            print("  warning: skipping series entry with empty ticker "
                  f"(title={series.get('title')!r})")
            continue
        series_markets: list[dict] = []
        try:
            for market in client.iter_settled_markets_for_series(series_ticker):
                series_markets.append(market)
                if args.max_markets and total_markets + len(series_markets) >= args.max_markets:
                    cap_hit = True
                    break
        except KalshiApiError as error:
            print(f"  skipping series {series_ticker}: {error}")
            continue

        event_series = {
            m.get("event_ticker", ""): series_ticker for m in series_markets
        }
        series_categories = {series_ticker: series.get("category") or "unknown"}
        frame = build_settled_market_frame(series_markets, event_series, series_categories)
        upsert_frame(args.db, "kalshi_settled_markets", frame, key_columns=["ticker"])
        total_markets += len(series_markets)

        if cap_hit or i % 25 == 0 or i == len(selected):
            print(f"  {i}/{len(selected)} series, {total_markets} markets so far")

        if cap_hit:
            print(f"  hit --max-markets cap of {args.max_markets}")
            break

    print(f"Done. {total_markets} markets pulled across {len(selected)} series.")


def main() -> None:
    args = parse_args()
    client = KalshiPublicClient(request_delay_seconds=args.request_delay)

    if args.list_categories:
        run_list_categories(client)
        return

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        run_categories_pull(client, args, categories)
        return

    print("Fetching settled events for series mapping...")
    event_series: dict[str, str] = {}
    try:
        for event in client.iter_settled_events():
            event_series[event["event_ticker"]] = event.get("series_ticker", "")
    except KalshiApiError as error:
        print(f"  pagination aborted early: {error} — continuing with what was collected")
    print(f"  {len(event_series)} settled events")

    print("Fetching settled markets...")
    markets: list[dict] = []
    try:
        for market in client.iter_settled_markets(min_settled_ts=args.min_settled_ts):
            markets.append(market)
            if args.max_markets and len(markets) >= args.max_markets:
                print(f"  hit --max-markets cap of {args.max_markets}")
                break
    except KalshiApiError as error:
        print(f"  pagination aborted early: {error} — continuing with what was collected")
    print(f"  {len(markets)} settled markets")

    series_tickers = sorted(
        {resolve_series_ticker(event_series, m.get("event_ticker", "")) for m in markets}
    )
    print(f"Fetching categories for {len(series_tickers)} series...")
    series_categories: dict[str, str] = {}
    for series_ticker in series_tickers:
        if not series_ticker:
            continue
        try:
            series = client.get_series(series_ticker)
        except KalshiApiError as error:
            print(f"  skipping {series_ticker}: {error}")
            continue
        series_categories[series_ticker] = series.get("category", "unknown")

    frame = build_settled_market_frame(markets, event_series, series_categories)
    upsert_frame(args.db, "kalshi_settled_markets", frame, key_columns=["ticker"])
    print(f"Upserted {len(frame)} binary settled markets into {args.db}")


if __name__ == "__main__":
    main()
