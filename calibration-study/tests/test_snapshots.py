import pandas as pd

from kalshi_fund.snapshots import extract_horizon_snapshots, sample_tickers_per_category

DAY = 86400
CLOSE_TS = 1_767_625_200


def candle(ts_offset_days: float, bid: str, ask: str, volume: str = "10"):
    return {
        "end_period_ts": int(CLOSE_TS - ts_offset_days * DAY),
        "yes_bid": {"close_dollars": bid},
        "yes_ask": {"close_dollars": ask},
        "volume_fp": volume,
    }


def test_picks_nearest_candle_per_horizon():
    candles = [candle(7.3, "0.40", "0.44"), candle(6.8, "0.41", "0.45"), candle(1.1, "0.60", "0.62")]
    snapshots = extract_horizon_snapshots(candles, CLOSE_TS, horizons=(1, 7))
    by_horizon = {s["horizon_days"]: s for s in snapshots}
    assert by_horizon[7]["yes_bid"] == 0.41  # 6.8d is nearer to 7d than 7.2d
    assert by_horizon[1]["mid"] == 0.61
    assert abs(by_horizon[1]["spread"] - 0.02) < 1e-9


def test_skips_horizon_when_no_candle_within_tolerance():
    candles = [candle(1.0, "0.50", "0.52")]
    snapshots = extract_horizon_snapshots(candles, CLOSE_TS, horizons=(1, 30))
    assert [s["horizon_days"] for s in snapshots] == [1]


def test_skips_candle_with_missing_quote():
    candles = [{"end_period_ts": CLOSE_TS - DAY, "yes_bid": {"close_dollars": None},
                "yes_ask": {"close_dollars": "0.50"}, "volume_fp": "0"}]
    assert extract_horizon_snapshots(candles, CLOSE_TS, horizons=(1,)) == []


def test_sampling_is_deterministic_and_capped():
    markets = pd.DataFrame(
        {"ticker": [f"T{i}" for i in range(10)],
         "category": ["A"] * 6 + ["B"] * 4}
    )
    first = sample_tickers_per_category(markets, per_category_cap=3, seed=42)
    second = sample_tickers_per_category(markets, per_category_cap=3, seed=42)
    assert first == second
    assert len(first) == 6  # 3 from A + 3 from B
