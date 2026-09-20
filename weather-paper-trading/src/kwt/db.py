"""SQLite data layer: schema, connection, and small upsert helpers.

One file = the entire tracking store. Tables:
  markets       - one row per Kalshi bucket market (upserted, carries result)
  snapshots     - price/volume time series for each market
  forecasts     - per city/target_date forecast distribution at each pull
  observations  - realized daily high (for model evaluation only)
  signals       - per-strategy decision log for every market each run
  trades        - immutable paper fills (settlement writes pnl back)
  positions     - open positions per (strategy, ticker, side)
  equity        - per-strategy equity time series
  strategies    - registry + frozen params per strategy

Live trading (real-money market-making experiment, see live_engine.py):
  live_orders     - every order we PLAN/PLACE live (dry_run records, never sends)
  live_fills      - real fills pulled from Kalshi /portfolio/fills
  live_positions  - reconciled Kalshi position snapshots
  live_risk_events- blocked orders, kill-switch trips, API/reconcile failures
  mm_fill_audit   - THE experiment: simulated 5% fill qty vs actual fill qty
                    per quote interval, so the participation assumption is testable
"""
from __future__ import annotations

import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_DB, utcnow_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    ticker        TEXT PRIMARY KEY,
    series_ticker TEXT,
    event_ticker  TEXT,
    city          TEXT,
    target_date   TEXT,
    bucket_kind   TEXT,            -- 'above' | 'below' | 'range'
    low           INTEGER,         -- inclusive integer low  (NULL = -inf)
    high          INTEGER,         -- inclusive integer high (NULL = +inf)
    yes_sub_title TEXT,
    status        TEXT,
    result        TEXT,            -- 'yes' | 'no' | NULL until settled
    open_time     TEXT,
    close_time    TEXT,
    first_seen    TEXT,
    last_seen     TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT,
    ticker        TEXT,
    yes_bid       REAL, yes_ask REAL, no_bid REAL, no_ask REAL,
    last_price    REAL, volume REAL, open_interest REAL, liquidity REAL,
    UNIQUE(ts, ticker)
);
CREATE INDEX IF NOT EXISTS idx_snap_ticker ON snapshots(ticker, ts);

CREATE TABLE IF NOT EXISTS forecasts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT,
    city          TEXT,
    target_date   TEXT,
    horizon_days  REAL,
    mean          REAL, std REAL, p05 REAL, p10 REAL, p50 REAL, p90 REAL, p95 REAL,
    n_members     INTEGER,
    members_json  TEXT,            -- raw per-member daily max (JSON list)
    models_json   TEXT,            -- per-model deterministic daily max (JSON dict)
    source        TEXT,            -- 'nwp-ensemble' | 'climatology'
    UNIQUE(ts, city, target_date, source)
);
CREATE INDEX IF NOT EXISTS idx_fc_city_date ON forecasts(city, target_date, ts);

CREATE TABLE IF NOT EXISTS observations (
    city          TEXT,
    target_date   TEXT,
    observed_high REAL,
    source        TEXT,
    ts            TEXT,
    PRIMARY KEY (city, target_date)
);

CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT,
    strategy      TEXT,
    ticker        TEXT,
    model_prob    REAL,            -- strategy fair YES probability
    market_prob   REAL,            -- market mid YES probability
    edge          REAL,            -- expected edge on chosen side after fees
    side          TEXT,            -- 'yes' | 'no' | 'none'
    decision      TEXT,            -- 'enter' | 'hold' | 'skip'
    meta_json     TEXT,
    UNIQUE(ts, strategy, ticker)
);
CREATE INDEX IF NOT EXISTS idx_sig ON signals(strategy, ticker, ts);

CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT,
    strategy      TEXT,
    ticker        TEXT,
    side          TEXT,            -- 'yes' | 'no'
    action        TEXT,            -- 'buy' | 'sell'
    contracts     REAL,
    price         REAL,            -- execution price (dollars, that side)
    fee           REAL,
    role          TEXT,            -- 'taker' | 'maker'
    reason        TEXT,
    settled       INTEGER DEFAULT 0,
    pnl           REAL
);
CREATE INDEX IF NOT EXISTS idx_trade ON trades(strategy, ticker);

CREATE TABLE IF NOT EXISTS positions (
    strategy      TEXT,
    ticker        TEXT,
    side          TEXT,            -- 'yes' | 'no'
    contracts     REAL,
    cost          REAL,            -- total cash paid (incl fees)
    fees          REAL,
    opened_ts     TEXT,
    PRIMARY KEY (strategy, ticker, side)
);

CREATE TABLE IF NOT EXISTS equity (
    ts             TEXT,
    strategy       TEXT,
    cash           REAL,
    position_cost  REAL,           -- cash tied up in open positions
    realized_pnl   REAL,
    equity         REAL,           -- cash + open position cost basis
    PRIMARY KEY (ts, strategy)
);

CREATE TABLE IF NOT EXISTS strategies (
    name        TEXT PRIMARY KEY,
    enabled     INTEGER,
    bankroll0   REAL,
    cash        REAL,
    realized_pnl REAL DEFAULT 0,
    params_json TEXT,
    created_ts  TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT,
    kind      TEXT,                -- 'collect' | 'settle'
    detail    TEXT
);

CREATE TABLE IF NOT EXISTS kv (
    k TEXT PRIMARY KEY,
    v TEXT
);

-- Live trading (real-money experiment) -------------------------------------
-- These tables are inert until `kwt live-mm` runs. The paper harness never
-- writes them. dry_run mode records planned orders here but sends nothing.
CREATE TABLE IF NOT EXISTS live_orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT,            -- when we planned/placed it (UTC iso)
    mode            TEXT,            -- 'dry_run' | 'demo' | 'prod'
    strategy        TEXT,
    ticker          TEXT,
    side            TEXT,            -- 'bid' (buy YES) | 'ask' (sell YES = buy NO)
    price           REAL,            -- yes-price in dollars
    count           INTEGER,
    client_order_id TEXT UNIQUE,
    order_id        TEXT,            -- Kalshi id (NULL in dry_run / until ack)
    status          TEXT,            -- planned|resting|canceled|filled|rejected|blocked|error
    reason          TEXT,
    resp_json       TEXT,            -- raw API response (NULL in dry_run)
    updated_ts      TEXT
    ,cohort_id      TEXT DEFAULT 'legacy'
);
CREATE INDEX IF NOT EXISTS idx_live_orders ON live_orders(ticker, status);

-- Passive queue telemetry.  One row per exchange-confirmed resting order per
-- successful reconcile cycle; it cannot change live risk or order behaviour.
CREATE TABLE IF NOT EXISTS live_queue_observations (
    ts              TEXT NOT NULL,
    cohort_id       TEXT NOT NULL DEFAULT 'legacy',
    order_id        TEXT NOT NULL,
    ticker          TEXT,
    side            TEXT,
    price           REAL,
    remaining_count REAL,
    queue_ahead     REAL,
    quote_age_sec   REAL,
    PRIMARY KEY (ts, order_id)
);
CREATE INDEX IF NOT EXISTS idx_live_queue_obs ON live_queue_observations(cohort_id, ticker, ts);

CREATE TABLE IF NOT EXISTS live_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT,
    fill_id         TEXT UNIQUE,     -- Kalshi fill id (idempotent ingest)
    order_id        TEXT,
    client_order_id TEXT,
    ticker          TEXT,
    side            TEXT,            -- 'yes' | 'no'  (Kalshi's normalized leg)
    action          TEXT,            -- 'buy' | 'sell'
    book_side       TEXT,            -- 'bid' | 'ask' — which of OUR resting orders filled.
                                     -- The reliable direction: bid=bought YES(+1), ask=sold
                                     -- YES(-1). (side, action) does NOT distinguish these
                                     -- (an ask fill is encoded (no, sell)); use book_side.
    count           INTEGER,
    price           REAL,
    fee             REAL,
    is_taker        INTEGER,
    created_time    TEXT
    ,cohort_id      TEXT DEFAULT 'legacy'
);
CREATE INDEX IF NOT EXISTS idx_live_fills ON live_fills(ticker, created_time);

CREATE TABLE IF NOT EXISTS live_positions (
    ts              TEXT,
    ticker          TEXT,
    position        INTEGER,         -- signed (+YES / -NO) per Kalshi portfolio
    market_exposure REAL,
    realized_pnl    REAL,
    fees_paid       REAL,
    PRIMARY KEY (ts, ticker)
);

CREATE TABLE IF NOT EXISTS live_risk_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT,
    kind            TEXT,            -- block | kill_switch | api_error | reconcile_mismatch
    ticker          TEXT,
    detail          TEXT
    ,cohort_id      TEXT DEFAULT 'legacy'
);
CREATE INDEX IF NOT EXISTS idx_live_risk ON live_risk_events(kind, ts);

-- Per-cycle experiment equity (realized + mark-to-market), experiment-scoped.
-- Backs a ROLLING-window daily-loss kill switch: baselining at UTC midnight split
-- an evening's loss (22Z-03Z straddles the reset) across two sub-limit "days".
CREATE TABLE IF NOT EXISTS live_equity (
    ts              TEXT PRIMARY KEY,
    equity          REAL
);

-- Per-fill markout: signed mid change at fixed horizons after each fill. Measures
-- adverse selection directly. Populated offline now (kwt markouts) and reused by a
-- future live tracker via the same idempotent upsert (INSERT OR REPLACE by fill_id).
CREATE TABLE IF NOT EXISTS live_fill_markouts (
    fill_id       TEXT PRIMARY KEY,   -- matches live_fills.fill_id
    ticker        TEXT,
    created_time  TEXT,               -- the fill time (copied for slicing)
    direction     INTEGER,            -- +1 long-YES-equiv, -1 short-YES-equiv
    mid_at_fill   REAL,               -- yes-mid nearest the fill (NULL if none in tol)
    mo_15         REAL,               -- signed markout at +15 min (NULL if no snap in tol)
    mo_30         REAL,
    mo_60         REAL,
    mo_settle     REAL,               -- signed markout to settlement (NULL until settled)
    computed_ts   TEXT
);

-- The core experiment: per quote-interval, the simulated fill quantity under the
-- current 5%/2% participation rule vs the ACTUAL filled quantity. The whole
-- point of going live is to learn whether sim_fill_qty tracks actual_fill_qty.
CREATE TABLE IF NOT EXISTS mm_fill_audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    interval_start  TEXT,            -- prior cycle ts (quote was resting since)
    interval_end    TEXT,            -- this cycle ts
    mode            TEXT,
    ticker          TEXT,
    bid             REAL,            -- resting bid during the interval
    ask             REAL,            -- resting ask during the interval
    rested_any      INTEGER,         -- 1 = at least one side rested during the interval
    rested_bid      INTEGER,         -- 1 = our bid was the resting side
    rested_ask      INTEGER,         -- 1 = our ask was the resting side
    size            INTEGER,
    market_mid      REAL,            -- YES mid at interval END (kept for continuity)
    mid_at_placement REAL,           -- YES mid when the quote was placed (spread base)
    model_fair      REAL,            -- model bucket prob at placement (now persisted)
    horizon_days    REAL,
    print_vol       REAL,            -- public contracts printed in the interval
    print_vol_bid   REAL,            -- public prints at/below our resting bid
    print_vol_ask   REAL,            -- public prints at/above our resting ask
    sim_fill_qty    REAL,            -- optimistic sim (excludes adverse sweeps)
    sim_fill_qty_incl_adverse REAL,  -- pessimistic sim (a resting order is picked off)
    sim_buy_yes     REAL,            -- per-side optimistic sim
    sim_buy_no      REAL,
    actual_fill_qty REAL,            -- real fills in the interval (0 in dry_run)
    actual_bid_fills REAL,           -- real fills attributed to our bid (buy YES)
    actual_ask_fills REAL,           -- real fills attributed to our ask (sell YES)
    adverse_skipped REAL,            -- contracts that swept through our quote
    cohort_id       TEXT DEFAULT 'legacy',
    UNIQUE(interval_end, ticker)
);
CREATE INDEX IF NOT EXISTS idx_mm_audit ON mm_fill_audit(ticker, interval_end);

-- Per (ticker, cycle) quote coverage: written EVERY cycle for EVERY candidate
-- market, quoted or not. This is the go/no-go denominator — "fraction of
-- intervals we actually rested a two-sided quote" — plus the placement context
-- (touch + fair) the fill-rate number is uninterpretable without.
CREATE TABLE IF NOT EXISTS mm_quote_log (
    ts            TEXT,
    mode          TEXT,
    ticker        TEXT,
    rested_any    INTEGER,          -- 1 = at least one side placed and resting
    rested_both   INTEGER,          -- 1 = both sides placed and resting this cycle
    skip_reason   TEXT,             -- '' when quoted; else why we did not quote
    bid           REAL,
    ask           REAL,
    model_fair    REAL,
    market_mid    REAL,
    mkt_bid       REAL,             -- best book bid at placement (were we behind the touch?)
    mkt_ask       REAL,
    net_yes       REAL,
    cohort_id     TEXT DEFAULT 'legacy',
    PRIMARY KEY (ts, ticker)
);
CREATE INDEX IF NOT EXISTS idx_mm_quote_log ON mm_quote_log(ticker, ts);

-- Immutable registrations for live experiment policy revisions. Historical rows
-- are assigned to `legacy`; each new policy gets a frozen config hash so a cohort
-- can never silently change meaning after evidence starts accumulating.
CREATE TABLE IF NOT EXISTS experiment_cohorts (
    cohort_id       TEXT PRIMARY KEY,
    created_ts      TEXT NOT NULL,
    activated_ts    TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    config_sha256   TEXT NOT NULL,
    primary_metric  TEXT NOT NULL,
    horizon_minutes INTEGER,
    status          TEXT NOT NULL DEFAULT 'active'
);

-- Per-fill quality: effective spread captured vs the mid at fill, and realized
-- settlement P&L, side-aware (a NO fill's cost is 1 - yes_price; it pays on
-- result='no'). This is what answers "is adverse selection eating the spread?".
DROP VIEW IF EXISTS mm_fill_pnl;
CREATE VIEW mm_fill_pnl AS
SELECT *,
  -- Direction from book_side (reliable): bid = bought YES, ask = sold YES. (side,
  -- action) can't distinguish these — Kalshi encodes an ask fill as (no, sell),
  -- which the old CASE read as long-NO and sign-flipped every ask fill.
  CASE WHEN mid_at_fill IS NULL THEN NULL
       WHEN book_side='bid' THEN mid_at_fill - yes_price   -- bought below mid = good
       WHEN book_side='ask' THEN yes_price - mid_at_fill   -- sold above mid = good
       END AS effective_spread,
  CASE WHEN result IS NULL THEN NULL
       WHEN book_side='bid' THEN (CASE WHEN result='yes' THEN 1.0 ELSE 0.0 END) - yes_price - fee_per_ct
       WHEN book_side='ask' THEN yes_price - (CASE WHEN result='yes' THEN 1.0 ELSE 0.0 END) - fee_per_ct
       END AS pnl_per_contract
FROM (
  SELECT f.id, f.ts, f.ticker, f.side, f.action, f.book_side, f.count, f.price AS yes_price,
         f.fee, f.created_time, f.is_taker, m.result,
         f.fee / NULLIF(f.count, 0) AS fee_per_ct,
         (SELECT (s.yes_bid + s.yes_ask) / 2.0 FROM snapshots s
            WHERE s.ticker = f.ticker AND s.yes_bid IS NOT NULL AND s.yes_ask IS NOT NULL
              AND s.ts <= f.created_time ORDER BY s.ts DESC LIMIT 1) AS mid_at_fill
  FROM live_fills f LEFT JOIN markets m ON m.ticker = f.ticker
);
"""

# Additive live-schema migration for existing databases (data/kwt.db already has
# the old mm_fill_audit shape). Bump LIVE_SCHEMA_VERSION when the block below
# changes; connect() re-runs it once per bump. SCHEMA above is the source of
# truth for fresh DBs; this only patches pre-existing ones.
LIVE_SCHEMA_VERSION = "9"
_MM_AUDIT_ADDED_COLS = [
    ("mid_at_placement", "REAL"), ("sim_fill_qty_incl_adverse", "REAL"),
    ("sim_buy_yes", "REAL"), ("sim_buy_no", "REAL"),
    ("actual_bid_fills", "REAL"), ("actual_ask_fills", "REAL"),
    ("rested_any", "INTEGER"), ("rested_bid", "INTEGER"),
    ("rested_ask", "INTEGER"), ("print_vol_bid", "REAL"),
    ("print_vol_ask", "REAL"),
]
_MM_QUOTE_LOG_ADDED_COLS = [("rested_any", "INTEGER")]


def kv_get(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
    return row["v"] if row else default


def kv_set(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute("INSERT INTO kv (k, v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                 (key, str(value)))


def _migrate_live(conn: sqlite3.Connection) -> None:
    """Idempotently bring a pre-existing DB up to the current live schema.

    Fresh DBs get everything from SCHEMA via init_db; this only patches DBs that
    already have the older mm_fill_audit shape. Runs at most once per version bump
    (guarded by a kv marker), so it is cheap on every connect after that.
    """
    has_audit = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mm_fill_audit'"
    ).fetchone()
    has_kv = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kv'").fetchone()
    if not has_audit or not has_kv:
        return  # brand-new (or partial) DB — SCHEMA/init_db will build it current
    if kv_get(conn, "live_schema_version", "1") == LIVE_SCHEMA_VERSION:
        return
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(mm_fill_audit)")}
    for col, decl in _MM_AUDIT_ADDED_COLS:
        if col not in existing:
            conn.execute(f"ALTER TABLE mm_fill_audit ADD COLUMN {col} {decl}")
    existing_quote = {r["name"] for r in conn.execute("PRAGMA table_info(mm_quote_log)")}
    for col, decl in _MM_QUOTE_LOG_ADDED_COLS:
        if col not in existing_quote:
            conn.execute(f"ALTER TABLE mm_quote_log ADD COLUMN {col} {decl}")
    # Backfill only what can be inferred without replaying historical tapes.
    # Side-specific print volumes remain NULL for legacy intervals rather than
    # pretending that total tape volume was observed on each side.
    conn.execute(
        "UPDATE mm_fill_audit SET rested_any=CASE WHEN bid IS NOT NULL OR ask IS NOT NULL "
        "THEN 1 ELSE 0 END WHERE rested_any IS NULL")
    conn.execute(
        "UPDATE mm_fill_audit SET rested_bid=CASE WHEN bid IS NOT NULL THEN 1 ELSE 0 END "
        "WHERE rested_bid IS NULL")
    conn.execute(
        "UPDATE mm_fill_audit SET rested_ask=CASE WHEN ask IS NOT NULL THEN 1 ELSE 0 END "
        "WHERE rested_ask IS NULL")
    conn.execute(
        "UPDATE mm_quote_log SET rested_any=CASE WHEN rested_both=1 OR "
        "COALESCE(skip_reason,'')='' THEN 1 ELSE 0 END WHERE rested_any IS NULL")
    # live_fills gained book_side (v5): the reliable fill direction (bid/ask).
    fill_cols = {r["name"] for r in conn.execute("PRAGMA table_info(live_fills)")}
    if "book_side" not in fill_cols:
        conn.execute("ALTER TABLE live_fills ADD COLUMN book_side TEXT")
    cohort_tables = ("live_orders", "live_fills", "live_risk_events",
                     "mm_fill_audit", "mm_quote_log")
    for table in cohort_tables:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "cohort_id" not in cols:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN cohort_id TEXT DEFAULT 'legacy'")
    # Create any new tables + (re)create the view from the canonical SCHEMA text.
    conn.executescript(SCHEMA)
    kv_set(conn, "live_schema_version", LIVE_SCHEMA_VERSION)
    conn.commit()


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    # Several crons write this DB (collect / settle / live-mm every 10 min). Without
    # a generous busy_timeout a writer that can't grab the lock raises
    # 'database is locked' immediately (Python's default is only 5s) and crashes the
    # cycle. Wait instead; combined with shorter write transactions in build_contexts
    # (commit per city) this makes concurrent cycles cooperate.
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
    ).fetchone():
        # Brand-new DB (including ":memory:") — build the schema immediately so
        # connect() alone is sufficient; SCHEMA is idempotent (IF NOT EXISTS
        # throughout), so this is safe even if init_db() also runs it later.
        conn.executescript(SCHEMA)
        conn.commit()
    _migrate_live(conn)
    return conn


def init_db(db_path: Path | str = DEFAULT_DB) -> None:
    conn = connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def ensure_experiment_cohort(conn: sqlite3.Connection, cohort_id: str,
                             config: dict[str, Any], *, primary_metric: str,
                             horizon_minutes: int | None = None,
                             activated_ts: str | None = None) -> dict[str, Any]:
    """Register an immutable live policy cohort, or verify its frozen config.

    Reusing a cohort id with different parameters fails closed: otherwise a GUI
    slice bearing one name could combine observations from two trading policies.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    row = conn.execute(
        "SELECT * FROM experiment_cohorts WHERE cohort_id=?", (cohort_id,)).fetchone()
    if row:
        if row["config_sha256"] != digest:
            raise ValueError(
                f"experiment cohort {cohort_id!r} already exists with a different config")
        return dict(row)
    ts = activated_ts or utcnow_iso()
    conn.execute(
        "INSERT INTO experiment_cohorts (cohort_id,created_ts,activated_ts,config_json,"
        "config_sha256,primary_metric,horizon_minutes,status) VALUES (?,?,?,?,?,?,?,'active')",
        (cohort_id, ts, ts, payload, digest, primary_metric, horizon_minutes))
    conn.commit()
    return dict(conn.execute(
        "SELECT * FROM experiment_cohorts WHERE cohort_id=?", (cohort_id,)).fetchone())


def upsert(conn: sqlite3.Connection, table: str, row: dict[str, Any], keys: Iterable[str]) -> None:
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in keys)
    conflict = ",".join(keys)
    sql = (
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
    )
    conn.execute(sql, [row[c] for c in cols])


def insert_ignore(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    conn.execute(sql, [row[c] for c in cols])


def log_run(conn: sqlite3.Connection, kind: str, detail: str) -> None:
    conn.execute(
        "INSERT INTO runs (ts, kind, detail) VALUES (?,?,?)",
        (utcnow_iso(), kind, detail),
    )


def jdump(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))
