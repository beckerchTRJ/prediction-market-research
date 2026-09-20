CREATE TABLE IF NOT EXISTS kalshi_settled_markets (
    ticker TEXT PRIMARY KEY,
    event_ticker TEXT,
    series_ticker TEXT,
    category TEXT,
    title TEXT,
    market_type TEXT,
    result TEXT,
    settlement_value_dollars REAL,
    open_time TEXT,
    close_time TEXT,
    close_ts INTEGER,
    volume REAL,
    raw_json TEXT,
    inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_settled_markets_category
    ON kalshi_settled_markets(category);

CREATE TABLE IF NOT EXISTS kalshi_price_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    snapshot_ts INTEGER NOT NULL,
    yes_bid REAL,
    yes_ask REAL,
    mid REAL,
    spread REAL,
    volume REAL,
    UNIQUE(ticker, horizon_days)
);

CREATE TABLE IF NOT EXISTS kalshi_snapshot_attempts (
    ticker TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    attempted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
