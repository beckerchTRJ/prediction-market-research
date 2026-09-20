from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kalshi_fund.panel import build_panel_frame
from kalshi_fund.storage import query_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the modeling panel from market and anchor data.")
    parser.add_argument("--output", required=True, help="Output CSV path for the panel.")
    parser.add_argument("--db", help="SQLite database containing raw observations.")
    parser.add_argument("--market-input", help="CSV file for market observations if not loading from DB.")
    parser.add_argument("--anchor-input", help="CSV file for anchor observations if not loading from DB.")
    parser.add_argument("--horizon-days", type=int, default=7, help="Future correction horizon in days.")
    return parser.parse_args()


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.db:
        markets = query_frame(args.db, "SELECT * FROM raw_market_observations")
        anchors = query_frame(args.db, "SELECT * FROM raw_anchor_observations")
        return markets, anchors

    if not args.market_input or not args.anchor_input:
        msg = "either provide --db or both --market-input and --anchor-input"
        raise SystemExit(msg)
    return pd.read_csv(args.market_input), pd.read_csv(args.anchor_input)


def main() -> None:
    args = parse_args()
    market_frame, anchor_frame = _load_inputs(args)
    panel = build_panel_frame(market_frame, anchor_frame, correction_horizon_days=args.horizon_days)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    print(f"Wrote panel with {len(panel)} rows to {output_path}")


if __name__ == "__main__":
    main()
