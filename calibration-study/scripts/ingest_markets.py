from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kalshi_fund.schemas import MarketObservation, validate_dataframe
from kalshi_fund.storage import append_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and ingest market snapshots.")
    parser.add_argument("--input", required=True, help="CSV file containing market observations.")
    parser.add_argument("--db", help="Optional SQLite database path.")
    parser.add_argument("--output", help="Optional normalized CSV output path.")
    parser.add_argument("--source-name", help="Fill `source` when the input file omits it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    if "source" not in frame.columns:
        if not args.source_name:
            msg = "input is missing `source`; pass --source-name to fill it"
            raise SystemExit(msg)
        frame["source"] = args.source_name

    validated = validate_dataframe(frame, MarketObservation)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        validated.to_csv(args.output, index=False)

    if args.db:
        append_frame(args.db, "raw_market_observations", validated)

    print(f"Ingested {len(validated)} validated market observations")


if __name__ == "__main__":
    main()
