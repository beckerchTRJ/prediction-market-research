"""Event-arbitrage (structural / near-guaranteed return).

The bucket markets in one event partition the daily high: exactly one resolves
YES. Two static dutch-book conditions are therefore tradable independent of any
forecast:
  * sum of YES asks across all buckets < $1  -> buy 1 YES of each; exactly one
    pays $1, locking 1 - sum(asks) - fees.
  * sum of NO  asks across all buckets < (n-1) -> buy 1 NO of each; n-1 pay $1.

Kept SEPARATE from the forecast strategies (per the kalshi_fund policy of not
mixing guaranteed-return trades with alpha trades). These are rare after the
exchange's overround + fees, but when they appear they are close to riskless, so
the harness flags and books them. Requires the full bucket set for an event, so
this strategy consumes the whole context list and groups by event.
"""
from __future__ import annotations

from collections import defaultdict

from ..fees import taker_fee
from .base import Order, Signal, Strategy


class EventArbitrageStrategy(Strategy):
    name = "event_arbitrage"

    def generate(self, ctxs, book):
        orders: list[Order] = []
        signals: list[Signal] = []
        min_edge = self.params.get("min_edge", 0.02)   # dollars per $1 of guaranteed payoff
        size = self.params.get("contracts", 10)

        events: dict[tuple[str, str], list] = defaultdict(list)
        for c in ctxs:
            events[(c.city, c.target_date)].append(c)

        for (city, date), buckets in events.items():
            yes_asks = [b.yes_ask for b in buckets if b.yes_ask is not None]
            no_asks = [b.no_ask for b in buckets if b.no_ask is not None]
            n = len(buckets)
            # need a (near) complete partition for the guarantee to hold
            if len(yes_asks) < n or n < 3:
                continue
            sum_yes = sum(yes_asks)
            fees = sum(taker_fee(size, b.yes_ask) for b in buckets) / size  # per-contract
            yes_edge = 1.0 - sum_yes - fees
            meta = {"n_buckets": n, "sum_yes_ask": round(sum_yes, 4),
                    "yes_edge": round(yes_edge, 4)}
            if yes_edge >= min_edge:
                for b in buckets:
                    orders.append(Order(self.name, b.ticker, "yes", "buy", size, b.yes_ask,
                                        "taker", f"event_arb:edge={yes_edge:.3f}"))
                signals.append(Signal(self.name, buckets[0].ticker, None, None, yes_edge,
                                      "yes", "enter", meta))
            else:
                signals.append(Signal(self.name, buckets[0].ticker, None, None, yes_edge,
                                      "none", "skip", meta))
        return orders, signals
