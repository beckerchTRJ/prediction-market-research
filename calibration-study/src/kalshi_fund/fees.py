from __future__ import annotations

import math


def taker_fee_usd(price: float, contracts: int = 1, rate: float = 0.07) -> float:
    """Kalshi general taker fee per order, rounded up to the next cent."""
    raw = rate * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0 - 1e-9) / 100.0
