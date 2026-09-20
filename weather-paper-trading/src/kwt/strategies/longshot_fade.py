"""Longshot-fade (favorite-longshot bias exploitation).

The literature (Burgi-Deng-Whelan; Le 2026) finds Kalshi longshots are
overpriced and weather markets are *overconfident* at short horizons — prices
on tail buckets are too high relative to base rates. This strategy shorts those
overpriced tails: for cheap YES buckets (<= max_yes_price) within the
short-horizon window it buys NO (equivalently sells the overpriced YES).
Fading deep longshots is also fee-cheap, since the taker fee ~ P(1-P) is small
near 0/1.

The thesis is a pure MARKET bias, so by default the trade gate is market-only
(price + horizon + physical feasibility) with flat sizing — the original
ensemble-confirmation gate let the fat-tailed ensemble (which overprices tails
itself) veto nearly every fade, starving the strategy to ~1 settled trade while
its counter-hypothesis twin accrued 80+. Flat size keeps the bet stream
comparable across markets; set `require_model_confirm: true` to restore the
old behavior. The ensemble probability, when available, is still recorded as
the signal's model_prob for Brier/calibration tracking.
"""
from __future__ import annotations

from collections import defaultdict

from .base import Order, Signal, Strategy


class LongshotFadeStrategy(Strategy):
    name = "longshot_fade"

    def generate(self, ctxs, book):
        orders: list[Order] = []
        signals: list[Signal] = []
        max_yes_price = self.params.get("max_yes_price", 0.10)
        # Optional PRICE-BAND floor: sub-few-cent buckets have no_ask ~ $0.999 and
        # leave no room after spread/fee, so a floor concentrates the fade on the
        # band where the bias is actually harvestable. Default 0 keeps old behavior.
        min_yes_price = self.params.get("min_yes_price", 0.0)
        min_over = self.params.get("min_overpricing", 0.03)
        short_h = self.params.get("short_horizon_days", 2.0)
        # Optional HORIZON-WINDOW floor: same-day (<~0.5d) is dominated by informed
        # nowcast flow (we get picked off); the edge lives day-before. Default 0
        # keeps old behavior (no lower bound).
        min_h = self.params.get("min_horizon_days", 0.0)
        flat_c = self.params.get("flat_contracts", 10)
        require_model = self.params.get("require_model_confirm", False)
        # Optional CROSS-CITY same-date exposure cap: one heat-dome event flips
        # "above" tails to YES across many cities on the SAME date at once — a
        # correlated blowout the per-market cap misses. 0 = off.
        max_date_contracts = self.params.get("max_date_contracts", 0)
        date_no: dict[str, float] = defaultdict(float)
        if max_date_contracts:
            for cc in ctxs:
                date_no[cc.target_date] += book.contracts(cc.ticker, "no")

        for c in ctxs:
            if c.no_ask is None or c.yes_ask is None:
                continue
            market_yes = c.yes_mid if c.yes_mid is not None else c.yes_ask
            # Candidate only if YES is a cheap longshot in the price band + window.
            is_longshot = (market_yes is not None
                           and min_yes_price <= market_yes <= max_yes_price)
            in_window = min_h <= c.horizon_days <= short_h
            # Ensemble probability, clamped to what today's observation has
            # physically decided — recorded for skill tracking, and used as a
            # veto only when require_model_confirm is on.
            p_lo, p_hi = c.feasible_yes_bounds()
            p_yes = None
            if c.nwp is not None:
                p_yes = c.nwp.p_bucket(c.low, c.high, blend_empirical=0.6)
                p_yes = min(max(p_yes, p_lo), p_hi)
            overprice = (market_yes - p_yes) if (market_yes is not None
                                                 and p_yes is not None) else None
            # Never fade a bucket the day has already decided YES.
            feasible = p_lo < 1.0
            # Entry-cost gate: the market's own price can never show positive
            # edge against itself, so the market-only gate bounds what we PAY —
            # don't cross more than `max_entry_premium` (spread + fee) over the
            # NO mid to put the fade on. The thesis (true p_no > 1 - market_yes)
            # supplies the edge; this just stops execution costs from eating it.
            max_prem = self.params.get("max_entry_premium", 0.02)
            premium = c.no_ask - (1.0 - market_yes) + self.fee_per_contract(c.no_ask)
            edge_no = (1.0 - p_yes) - c.no_ask if p_yes is not None else -premium
            meta = {"market_yes": round(market_yes, 4) if market_yes else None,
                    "p_yes": round(p_yes, 4) if p_yes is not None else None,
                    "overprice": round(overprice, 4) if overprice is not None else None,
                    "entry_premium": round(premium, 4),
                    "edge_no": round(edge_no, 4), "horizon_d": round(c.horizon_days, 2)}

            ok = is_longshot and in_window and feasible and premium <= max_prem
            if require_model:
                ok = ok and overprice is not None and overprice >= min_over

            if not ok:
                signals.append(Signal(self.name, c.ticker, p_yes, market_yes,
                                      edge_no, "no", "skip", meta))
                continue

            have = book.contracts(c.ticker, "no")
            add = flat_c - have
            if max_date_contracts:
                add = min(add, max(0.0, max_date_contracts - date_no[c.target_date]))
            if add <= 0:
                signals.append(Signal(self.name, c.ticker, p_yes, market_yes,
                                      edge_no, "no", "hold", meta))
                continue
            if max_date_contracts:
                date_no[c.target_date] += add
            orders.append(Order(self.name, c.ticker, "no", "buy", add, c.no_ask,
                                "taker", f"fade_longshot:mkt={market_yes:.3f}"))
            signals.append(Signal(self.name, c.ticker, p_yes, market_yes,
                                  edge_no, "no", "enter", meta))
        return orders, signals


class LongshotFadeDayBeforeStrategy(LongshotFadeStrategy):
    """Favorite-longshot fade tuned to the day-before window and the 3-20c band
    where the bias is harvestable at executable prices (per the 4-week data
    review), with a cross-city same-date exposure cap. Same logic as
    `longshot_fade`; only the config differs. Runs ALONGSIDE the original so its
    experiment is not disturbed."""
    name = "longshot_fade_dayb"
