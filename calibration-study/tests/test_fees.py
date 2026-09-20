from kalshi_fund.fees import taker_fee_usd


def test_fee_rounds_up_to_next_cent():
    # 0.07 * 0.5 * 0.5 = 0.0175 -> 0.02
    assert taker_fee_usd(0.50) == 0.02


def test_fee_scales_with_contracts_before_rounding():
    # 0.07 * 10 * 0.5 * 0.5 = 0.175 -> 0.18 (not 10 * 0.02)
    assert taker_fee_usd(0.50, contracts=10) == 0.18


def test_fee_zero_at_price_extremes():
    assert taker_fee_usd(0.0) == 0.0
    assert taker_fee_usd(1.0) == 0.0


def test_fee_custom_rate():
    # 0.035 * 0.5 * 0.5 = 0.00875 -> 0.01
    assert taker_fee_usd(0.50, rate=0.035) == 0.01
