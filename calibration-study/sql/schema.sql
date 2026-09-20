CREATE TABLE IF NOT EXISTS raw_market_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    market_id TEXT NOT NULL,
    race_id TEXT NOT NULL,
    contract TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    office TEXT,
    state TEXT,
    event_date TEXT NOT NULL,
    p_mkt REAL NOT NULL,
    bid REAL,
    ask REAL,
    volume REAL,
    open_interest REAL,
    liquidity_score REAL,
    metadata_json TEXT,
    inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_anchor_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    race_id TEXT NOT NULL,
    contract TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    office TEXT,
    state TEXT,
    event_date TEXT NOT NULL,
    p_anchor REAL NOT NULL,
    anchor_low REAL,
    anchor_high REAL,
    sample_size REAL,
    anchor_quality REAL,
    metadata_json TEXT,
    inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    target_name TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    train_start TEXT,
    train_end TEXT,
    validation_end TEXT,
    alpha REAL,
    sigma REAL,
    z_score REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id INTEGER,
    timestamp TEXT NOT NULL,
    race_id TEXT NOT NULL,
    contract TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    state TEXT,
    p_mkt REAL NOT NULL,
    p_anchor REAL,
    p_fair_model REAL NOT NULL,
    p_blend REAL,
    sigma REAL,
    delta_hat REAL,
    target_probability REAL,
    metadata_json TEXT,
    FOREIGN KEY(model_run_id) REFERENCES model_runs(model_run_id)
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id INTEGER,
    timestamp TEXT NOT NULL,
    race_id TEXT NOT NULL,
    contract TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    state TEXT,
    strategy TEXT NOT NULL DEFAULT 'political_residual',
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    conservative_probability REAL NOT NULL,
    net_edge REAL NOT NULL,
    suggested_notional_usd REAL NOT NULL,
    thesis_id TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN',
    metadata_json TEXT,
    FOREIGN KEY(model_run_id) REFERENCES model_runs(model_run_id)
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    timestamp TEXT NOT NULL,
    race_id TEXT NOT NULL,
    contract TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    state TEXT,
    strategy_type TEXT NOT NULL,
    thesis_id TEXT,
    side TEXT NOT NULL,
    displayed_price REAL NOT NULL,
    expected_fill_price REAL,
    actual_fill_price REAL,
    notional_usd REAL NOT NULL,
    fees_usd REAL DEFAULT 0,
    slippage_usd REAL DEFAULT 0,
    bankroll_usd REAL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    notes TEXT,
    FOREIGN KEY(signal_id) REFERENCES signals(signal_id)
);

CREATE TABLE IF NOT EXISTS bankroll_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    bankroll_usd REAL NOT NULL,
    available_cash_usd REAL,
    open_exposure_usd REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS post_trade_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL,
    review_timestamp TEXT NOT NULL,
    thesis_score INTEGER,
    execution_score INTEGER,
    process_score INTEGER,
    outcome_label TEXT,
    lessons_learned TEXT,
    process_change TEXT,
    FOREIGN KEY(trade_id) REFERENCES trades(trade_id)
);
