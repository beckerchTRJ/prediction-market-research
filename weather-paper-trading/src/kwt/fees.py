"""Kalshi fee model.

Kalshi's published general trading fee is charged on taker fills:

    fee = round_up( rate * C * P * (1 - P) )      [in dollars]

where C = number of contracts, P = price per contract in dollars (0..1), and the
result is rounded up to the next cent. The fee is maximized at P=0.50 and
vanishes at the extremes, which is why fading deep longshots is relatively
fee-cheap. Maker fills on weather markets are charged a small maker rate (some
series are maker-free; we model a configurable rate to stay conservative).

These are modeled costs for paper trading; verify against Kalshi's current fee
schedule before trading real capital.
"""
from __future__ import annotations

import math


def round_up_cents(dollars: float) -> float:
    return math.ceil(round(dollars, 10) * 100) / 100.0


def taker_fee(contracts: float, price: float, rate: float = 0.07) -> float:
    """Per-fill taker fee in dollars."""
    if contracts <= 0:
        return 0.0
    price = min(max(price, 0.0), 1.0)
    return round_up_cents(rate * contracts * price * (1.0 - price))


def maker_fee(contracts: float, price: float, rate: float = 0.0025) -> float:
    if contracts <= 0 or rate <= 0:
        return 0.0
    price = min(max(price, 0.0), 1.0)
    return round_up_cents(rate * contracts * price * (1.0 - price))
