"""Copy-trading (Polymarket wallet mirroring) — scaffold, disabled by default.

Polymarket trades are public on-chain, so a high-performing weather trader's
fills can be mirrored. This is wiring, not a turnkey edge: it requires seeding
`follow_wallets` in config with addresses you've identified as skilled (e.g. via
the Polymarket leaderboard / on-chain PnL analysis). Kalshi does NOT expose
per-wallet trades, so copy-trading is Polymarket-only.

Because Polymarket weather markets do not map onto the Kalshi temperature-bucket
schema, copied positions are tracked in their own namespace and are not directly
comparable to the Kalshi strategies' Brier scores — they are tracked purely on
realized P&L. Left as a clearly-marked extension point.
"""
from __future__ import annotations

from .base import Order, Signal, Strategy


class CopyTradingStrategy(Strategy):
    name = "copy_trading"

    def generate(self, ctxs, book):
        # Intentionally inert until target wallets are configured and a Polymarket
        # market<->position mapping is implemented. Emits no orders so it can be
        # enabled safely without polluting P&L.
        wallets = self.params.get("follow_wallets", [])
        signals: list[Signal] = []
        if not wallets:
            return [], signals
        # Placeholder: real implementation would fetch recent wallet_trades via
        # PolymarketClient, dedupe against already-copied trade_ids, scale by
        # copy_fraction, and emit fill_at orders against Polymarket prices.
        return [], signals
