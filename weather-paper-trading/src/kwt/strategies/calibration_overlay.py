"""Calibration-overlay (overconfidence de-biasing) — a market-only strategy.

Le (2026) finds Kalshi/Polymarket weather contracts are OVERCONFIDENT at short
horizons: logistic recalibration slopes of ~0.7-0.95 within 48h, i.e. prices are
too extreme relative to realized base rates. The fix is to recalibrate the
market's implied probability in LOG-ODDS space by the empirical slope, then
trade the gap.

The recalibration MUST be logistic (log-odds), matching how the slopes are
estimated: p_cal = sigmoid(intercept + slope * logit(p_mkt)). A linear pull
toward 0.5 (the original implementation) maps a 0.5¢ price to ~10% — a claimed
20x mispricing on every deep longshot — and turns the strategy into a longshot
vacuum (it bought 100+ cheap YES at a ~1% win rate in the first two days). In
logit space the same slope of 0.8 maps 0.5¢ to ~1.4%: a subtle tail tilt, which
is what the literature actually documents.

This is the deliberate COUNTER-HYPOTHESIS to `longshot_fade`:
  * longshot_fade (Burgi-Deng-Whelan favorite-longshot bias) says cheap YES
    longshots are OVERpriced  -> buy NO.
  * calibration_overlay (Le overconfidence) says extreme prices are too extreme,
    so a cheap YES longshot is UNDERpriced -> buy YES (and fade rich favorites).
They bet opposite directions on the same tail buckets. Running both lets the
paper trading reveal which effect actually dominates out-of-sample — uses no
weather model, only the market price and the published recalibration slope.
"""
from __future__ import annotations

import math

from .base import Order, Signal, Strategy


def recalibration_slope(horizon_days: float, params: dict) -> float:
    """Le (2026) Table 3 weather slopes, coarsened by horizon bucket."""
    slopes = params.get("slopes", {})
    if horizon_days <= 1:
        return slopes.get("le_1d", 0.80)
    if horizon_days <= 2:
        return slopes.get("le_2d", 0.90)
    if horizon_days <= 7:
        return slopes.get("le_1w", 1.0)
    return slopes.get("le_long", 1.10)


def _clip(p: float) -> float:
    return min(max(p, 1e-3), 1 - 1e-3)


def recalibrate(p_mkt: float, slope: float, intercept: float = 0.0) -> float:
    """Logistic recalibration: sigmoid(intercept + slope * logit(p_mkt)).

    slope < 1 pulls extreme prices toward 0.5 — but proportionally in log-odds,
    so deep longshots stay deep longshots."""
    p_mkt = _clip(p_mkt)
    z = intercept + slope * math.log(p_mkt / (1.0 - p_mkt))
    return _clip(1.0 / (1.0 + math.exp(-z)))


class CalibrationOverlayStrategy(Strategy):
    name = "calibration_overlay"

    def generate(self, ctxs, book):
        orders: list[Order] = []
        signals: list[Signal] = []
        edge_thr = self.params.get("edge_threshold", 0.05)
        max_c = self.params.get("max_contracts", 30)
        frac = self.params.get("kelly_fraction", 0.25)
        max_h = self.params.get("max_horizon_days", 2.0)
        intercept = self.params.get("intercept", 0.0)

        for c in ctxs:
            mkt = c.yes_mid
            if mkt is None or c.yes_ask is None or c.no_ask is None:
                continue
            if c.horizon_days > max_h:
                signals.append(Signal(self.name, c.ticker, None, mkt, None, "none", "skip",
                                      {"reason": "horizon"}))
                continue
            slope = recalibration_slope(c.horizon_days, self.params)
            p_cal = recalibrate(mkt, slope, intercept)
            # Time-of-day guardrail: never recalibrate past what's already
            # physically determined today. If observed-so-far makes this bucket
            # certain (1) or impossible (0), clamp p_cal to that — so we don't
            # pull a decided market back toward 0.5 and fade the realized outcome.
            p_lo, p_hi = c.feasible_yes_bounds()
            p_cal = min(max(p_cal, p_lo), p_hi)
            exits = self.exit_orders(c, p_cal, book)
            orders.extend(exits)
            edge_yes = self.net_edge(p_cal, c.yes_ask)
            edge_no = self.net_edge(1 - p_cal, c.no_ask)
            if not self.passes_cheap_gates(p_cal, c.yes_ask):
                edge_yes = -1
            if not self.passes_cheap_gates(1 - p_cal, c.no_ask):
                edge_no = -1
            side, price, p, edge = "none", None, None, max(edge_yes, edge_no)
            if edge_yes >= edge_no and edge_yes >= edge_thr:
                side, price, p = "yes", c.yes_ask, p_cal
            elif edge_no > edge_yes and edge_no >= edge_thr:
                side, price, p = "no", c.no_ask, 1 - p_cal
            meta = {"market": round(mkt, 4), "slope": slope, "p_cal": round(p_cal, 4),
                    "edge_yes": round(edge_yes, 4), "edge_no": round(edge_no, 4),
                    "horizon_d": round(c.horizon_days, 2)}
            if exits:
                meta["exits"] = [f"{o.side}x{o.contracts:.0f}" for o in exits]
            if side == "none":
                decision = "exit" if exits else "skip"
                signals.append(Signal(self.name, c.ticker, p_cal, mkt, edge, "none",
                                      decision, meta))
                continue
            target = self.kelly_size(p, price, book.equity, frac, max_c)
            add = target - book.contracts(c.ticker, side)
            if add <= 0:
                signals.append(Signal(self.name, c.ticker, p_cal, mkt, edge, side, "hold", meta))
                continue
            orders.append(Order(self.name, c.ticker, side, "buy", add, price, "taker",
                                f"calib:slope={slope}"))
            signals.append(Signal(self.name, c.ticker, p_cal, mkt, edge, side, "enter", meta))
        return orders, signals
