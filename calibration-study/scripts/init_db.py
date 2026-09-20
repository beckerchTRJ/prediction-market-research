from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kalshi_fund.storage import init_sqlite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the trading journal SQLite database.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument(
        "--schema",
        default=str(PROJECT_ROOT / "sql" / "schema.sql"),
        help="Path to the SQL schema file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_sqlite(args.db, args.schema)
    print(f"Initialized database at {args.db}")


if __name__ == "__main__":
    main()
