import numpy as np
import pandas as pd

from kalshi_fund.calibration import (
    assign_price_bucket,
    build_observation_frame,
    cluster_bootstrap_mean,
    compute_edge_map,
    profit_buy_no,
    profit_buy_yes,
    wilson_interval,
)


def test_price_buckets():
    assert assign_price_bucket(0.07) == "0.05-0.10"
    assert assign_price_bucket(0.05) == "0.05-0.10"  # left edge inclusive
    assert assign_price_bucket(0.97) == "0.95-1.00"


def test_profit_functions_net_of_fees():
    # win: gain (1 - ask) minus fee at ask
    assert abs(profit_buy_yes(True, 0.50) - (0.50 - 0.02)) < 1e-9
    # lose: lose ask plus fee
    assert abs(profit_buy_yes(False, 0.50) - (-0.50 - 0.02)) < 1e-9
    # NO side entry price is 1 - yes_bid
    assert abs(profit_buy_no(False, 0.90) - (0.90 - 0.01)) < 1e-9


def test_wilson_interval_contains_rate():
    low, high = wilson_interval(30, 100)
    assert low < 0.30 < high
    assert 0.0 <= low and high <= 1.0


def test_cluster_bootstrap_shrinks_with_more_clusters():
    # NOTE: deviates from the brief's literal snippet (plain iid normal draws
    # split into 4 vs. 400 "clusters"). With genuinely iid data and only 4
    # clusters, percentile cluster-bootstrap is known to be unstable/biased
    # downward for small G (Cameron/Gelbach/Miller 2008) -- empirically the
    # brief's literal test failed the vast majority of runs across many seeds
    # regardless of the (correct, spec-following) implementation, because the
    # 4 cluster means can land close together by chance. Adding a genuine
    # between-cluster effect makes the "few clusters -> wider CI" property
    # reliably observable, which is what this test is meant to demonstrate.
    rng = np.random.default_rng(0)
    cluster_effects = rng.normal(0, 0.3, size=4)
    base = rng.normal(0.1, 0.5, size=400)
    values = pd.Series(base + np.repeat(cluster_effects, 100))
    few = pd.Series(np.repeat(np.arange(4), 100))
    many = pd.Series(np.arange(400))
    few_low, few_high = cluster_bootstrap_mean(values, few, seed=1)
    many_low, many_high = cluster_bootstrap_mean(values, many, seed=1)
    assert (few_high - few_low) > (many_high - many_low)


def _synthetic_inputs(n_events=400, cheap_yes_win_rate=0.01):
    """Category 'Biased': 8-cent YES contracts that win only 1% of the time
    (heavily overpriced tails -> buying NO is profitable)."""
    rng = np.random.default_rng(7)
    markets, snapshots = [], []
    for i in range(n_events):
        won = rng.random() < cheap_yes_win_rate
        markets.append({
            "ticker": f"B-{i}", "event_ticker": f"EV-{i}", "category": "Biased",
            "result": "yes" if won else "no", "close_ts": 1_700_000_000 + i * 3600,
        })
        snapshots.append({
            "ticker": f"B-{i}", "horizon_days": 7, "snapshot_ts": 0,
            "yes_bid": 0.07, "yes_ask": 0.09, "mid": 0.08, "spread": 0.02, "volume": 100,
        })
    return pd.DataFrame(markets), pd.DataFrame(snapshots)


def test_observation_frame_and_edge_map_graduate_known_bias():
    markets, snapshots = _synthetic_inputs()
    observations = build_observation_frame(markets, snapshots)
    assert set(observations["half"]) == {"early", "late"}
    assert not observations["wide_spread"].any()

    edge_map = compute_edge_map(observations, min_obs=50, n_boot=200, seed=3)
    cell = edge_map[(edge_map["category"] == "Biased") & (edge_map["horizon_days"] == 7)]
    assert (cell["graduated_side"] == "no").all()
    assert (cell["ev_no_low"] > 0).all()
