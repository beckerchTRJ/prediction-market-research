"""Climatology baseline.

Builds the bucket distribution purely from ~15 years of historical daily highs
around the target calendar day (no live NWP). It is the honest null model: if a
strategy that uses real forecasts can't beat climatology, it has no weather
skill. It will also opportunistically trade when the market drifts far from the
climatological base rate (e.g. mispriced long-horizon markets).
"""
from __future__ import annotations

from .base import Book, MarketCtx, Order, Signal, Strategy


class ClimatologyStrategy(Strategy):
    name = "climatology"

    def generate(self, ctxs, book):
        orders: list[Order] = []
        signals: list[Signal] = []
        edge_thr = self.params.get("edge_threshold", 0.07)
        max_c = self.params.get("max_contracts", 25)
        frac = self.params.get("kelly_fraction", 0.25)
        for c in ctxs:
            if c.clim is None:
                continue
            p_yes = c.clim.p_bucket(c.low, c.high, blend_empirical=1.0)
            new_orders, sig = _maybe_trade(self, c, p_yes, edge_thr, max_c, frac, book,
                                           "climatology")
            orders.extend(new_orders)
            signals.append(sig)
        return orders, signals


def _maybe_trade(strat: Strategy, c: MarketCtx, p_yes: float, edge_thr: float,
                 max_c: int, frac: float, book: Book, tag: str) -> tuple[list[Order], Signal]:
    """Shared 'buy the cheaper side if model edge clears threshold' logic.

    Edges are measured net of the per-contract taker fee against the ask we
    would actually pay; cheap-price gates keep model strategies out of the
    longshot buckets (owned by the tail strategies); and held positions whose
    edge has reversed are exited at the bid (see Strategy.exit_orders)."""
    # Physical-feasibility clamp (defense-in-depth): member conditioning leaves
    # residual tail mass via the std floor and Student-t tails, so a bucket that
    # today's observation has already decided can still get a non-0/1 fair value.
    # Clamp every strategy's fair p_yes to what's physically possible before trading.
    p_lo, p_hi = c.feasible_yes_bounds()
    p_yes = min(max(p_yes, p_lo), p_hi)
    p_no = 1.0 - p_yes
    yes_ask, no_ask = c.yes_ask, c.no_ask
    edge_yes = strat.net_edge(p_yes, yes_ask)
    edge_no = strat.net_edge(p_no, no_ask)
    if yes_ask is not None and not strat.passes_cheap_gates(p_yes, yes_ask):
        edge_yes = -1
    if no_ask is not None and not strat.passes_cheap_gates(p_no, no_ask):
        edge_no = -1

    exits = strat.exit_orders(c, p_yes, book)

    side, price, p, edge = "none", None, None, max(edge_yes, edge_no)
    if edge_yes >= edge_no and edge_yes >= edge_thr:
        side, price, p = "yes", yes_ask, p_yes
    elif edge_no > edge_yes and edge_no >= edge_thr:
        side, price, p = "no", no_ask, p_no

    meta = {"p_yes": round(p_yes, 4), "edge_yes": round(edge_yes, 4),
            "edge_no": round(edge_no, 4), "horizon_d": round(c.horizon_days, 2)}
    if exits:
        meta["exits"] = [f"{o.side}x{o.contracts:.0f}" for o in exits]
    if side == "none" or price is None:
        decision = "exit" if exits else "skip"
        return exits, Signal(strat.name, c.ticker, p_yes, c.yes_mid, edge, "none",
                             decision, meta)

    target = strat.kelly_size(p, price, book.equity, frac, max_c)
    have = book.contracts(c.ticker, side)
    add = target - have
    if add <= 0:
        return exits, Signal(strat.name, c.ticker, p_yes, c.yes_mid, edge, side, "hold", meta)
    order = Order(strat.name, c.ticker, side, "buy", add, price, "taker",
                  f"{tag}:edge={edge:.3f}")
    return exits + [order], Signal(strat.name, c.ticker, p_yes, c.yes_mid, edge, side,
                                   "enter", meta)
