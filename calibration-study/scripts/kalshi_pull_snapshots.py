from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kalshi_fund.kalshi_api import KalshiPublicClient
from kalshi_fund.snapshots import DAY_SECONDS, extract_horizon_snapshots, sample_tickers_per_category
from kalshi_fund.storage import append_frame, insert_record, query_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull pre-settlement price snapshots for settled markets.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--per-category-cap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-markets", type=int, default=None, help="Cap for this run (resumable).")
    parser.add_argument("--request-delay", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markets = query_frame(
        args.db,
        """
        SELECT m.ticker, m.series_ticker, m.category, m.close_ts
        FROM kalshi_settled_markets m
        LEFT JOIN kalshi_snapshot_attempts a ON a.ticker = m.ticker
        WHERE a.ticker IS NULL AND m.close_ts IS NOT NULL
        """,
    )
    todo = sample_tickers_per_category(markets, args.per_category_cap, seed=args.seed)
    lookup = markets.set_index("ticker")
    if args.max_markets:
        todo = todo[: args.max_markets]
    print(f"{len(todo)} markets to snapshot")

    client = KalshiPublicClient(request_delay_seconds=args.request_delay)
    for index, ticker in enumerate(todo, start=1):
        row = lookup.loc[ticker]
        close_ts = int(row["close_ts"])
        try:
            candles = client.get_market_candlesticks(
                row["series_ticker"], ticker,
                start_ts=close_ts - 31 * DAY_SECONDS, end_ts=close_ts,
            )
            snapshots = extract_horizon_snapshots(candles, close_ts)
            status = "ok" if snapshots else "no_data"
            if snapshots:
                frame = pd.DataFrame(snapshots)
                frame.insert(0, "ticker", ticker)
                append_frame(args.db, "kalshi_price_snapshots", frame)
        except Exception as error:
            status = f"error: {error}"
        insert_record(args.db, "kalshi_snapshot_attempts", {"ticker": ticker, "status": status})
        if index % 100 == 0:
            print(f"  {index}/{len(todo)} done")
    print("Snapshot pull complete.")


if __name__ == "__main__":
    main()
