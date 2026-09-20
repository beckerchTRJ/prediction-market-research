"""Ensemble-divergence (the primary 'do I have a model edge' strategy).

Builds a fat-tailed bucket distribution from the pooled multi-model ensemble
(GFS + ECMWF + ICON + GEM members), shrinks it toward the market price by a
trust parameter (so we only fight the market when we strongly disagree), then
buys whichever side clears the post-fee edge threshold. This is the direct test
of whether public NWP, properly turned into probabilities, beats Kalshi prices.
"""
from __future__ import annotations

from .base import Order, Signal, Strategy
from .climatology import _maybe_trade


class EnsembleDivergenceStrategy(Strategy):
    name = "ensemble_divergence"

    def generate(self, ctxs, book):
        orders: list[Order] = []
        signals: list[Signal] = []
        edge_thr = self.params.get("edge_threshold", 0.05)
        max_c = self.params.get("max_contracts", 40)
        frac = self.params.get("kelly_fraction", 0.30)
        blend = self.params.get("blend_empirical", 0.6)
        shrink = self.params.get("shrink_to_market", 0.25)
        shrink_tail = self.params.get("shrink_to_market_tail", 0.75)
        tail_below = self.params.get("tail_price_below", 0.10)
        df = self.params.get("tail_fatten_df", 6.0)

        for c in ctxs:
            if c.nwp is None:
                continue
            p_raw = c.nwp.p_bucket(c.low, c.high, blend_empirical=blend, df=df)
            mkt = c.yes_mid
            # Shrink toward the market when we have a usable mid (trust param).
            # In the tails (market near 0 or 1) the Student-t fattening makes the
            # ensemble systematically heavier than reality — early losses were
            # almost entirely cheap-YES tail buys — so trust the market much more
            # there; keep the lower body shrinkage where the ensemble plausibly
            # knows something.
            if mkt is not None:
                s = shrink_tail if (mkt <= tail_below or mkt >= 1 - tail_below) else shrink
                p_yes = (1 - s) * p_raw + s * mkt
            else:
                p_yes = p_raw
            new_orders, sig = _maybe_trade(self, c, p_yes, edge_thr, max_c, frac, book,
                                           "ens_div")
            sig.meta["p_raw"] = round(p_raw, 4)
            sig.meta["nwp_mean"] = round(c.nwp.mean, 2)
            sig.meta["nwp_std"] = round(c.nwp.std, 2)
            orders.extend(new_orders)
            signals.append(sig)
        return orders, signals
