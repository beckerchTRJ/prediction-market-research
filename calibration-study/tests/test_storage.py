from pathlib import Path

import pandas as pd

from kalshi_fund.storage import init_sqlite, query_frame, upsert_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_kalshi_cache_schema_creates_tables(tmp_path):
    db = tmp_path / "cache.db"
    init_sqlite(db, PROJECT_ROOT / "sql" / "kalshi_cache.sql")
    tables = query_frame(db, "SELECT name FROM sqlite_master WHERE type='table'")
    names = set(tables["name"])
    assert {"kalshi_settled_markets", "kalshi_price_snapshots", "kalshi_snapshot_attempts"} <= names


def test_upsert_frame_replaces_existing_keys(tmp_path):
    db = tmp_path / "cache.db"
    init_sqlite(db, PROJECT_ROOT / "sql" / "kalshi_cache.sql")
    first = pd.DataFrame([{"ticker": "T1", "event_ticker": "E1", "result": "yes"}])
    upsert_frame(db, "kalshi_settled_markets", first, key_columns=["ticker"])
    second = pd.DataFrame([{"ticker": "T1", "event_ticker": "E1", "result": "no"}])
    upsert_frame(db, "kalshi_settled_markets", second, key_columns=["ticker"])
    out = query_frame(db, "SELECT ticker, result FROM kalshi_settled_markets")
    assert len(out) == 1
    assert out.loc[0, "result"] == "no"
