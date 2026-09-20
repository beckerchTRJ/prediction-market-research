from kwt.config import load_config


def test_controls_retired_and_fade_sharpened():
    s = load_config().strategies
    assert s["longshot_fade"]["enabled"] is False          # retired (tension resolved)
    assert s["calibration_overlay"]["enabled"] is False     # refuted (cheap buckets overpriced)
    assert s["longshot_fade_dayb"]["enabled"] is True       # the surviving fade
    assert s["longshot_fade_dayb"]["min_yes_price"] == 0.05  # 0.02-0.05 loses after cost
    assert s["longshot_fade_dayb"]["verdict_metric"] == "pnl"  # market-bias play, judge on P&L
