"""Strategy interface and the shared per-market context strategies operate on."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..distributions import Forecast, feasible_yes_bounds


@dataclass
class Order:
    """A paper order. Two fill modes (see engine.execute):

      action='buy'      -> taker; fills only if that side's ask <= limit_price,
                           executes at the ask.
      action='fill_at'  -> explicit fill of `contracts` at exactly limit_price
                           (used by the market-maker, which controls its own
                           fills against real trade prints). role sets the fee.
      action='sell'     -> taker exit of a held position; fills only if that
                           side's bid >= limit_price, executes at the bid and
                           realizes P&L immediately.
    """
    strategy: str
    ticker: str
    side: str                  # 'yes' | 'no'
    action: str                # 'buy' | 'fill_at' | 'sell'
    contracts: float
    limit_price: float
    role: str = "taker"        # 'taker' | 'maker'
    reason: str = ""


@dataclass
class MarketCtx:
    ticker: str
    city: str
    target_date: str
    low: int | None
    high: int | None
    bucket_kind: str
    horizon_days: float
    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None
    last_price: float | None
    open_interest: float | None
    metric: str = "high"               # 'high' | 'low' — which daily extreme this market grades
    nwp: Forecast | None = None        # intraday-conditioned ensemble forecast
    clim: Forecast | None = None       # intraday-conditioned climatology forecast
    nwp_raw: Forecast | None = None    # unconditioned ensemble (pre-intraday floor/ceiling)
    clim_raw: Forecast | None = None   # unconditioned climatology
    obs_so_far: float | None = None    # today's observed extreme (same-day markets only)
    remaining_max: float | None = None # forecast extreme over remaining hours today
    hours_elapsed: float | None = None # hours of the local day observed so far (same-day only)

    @property
    def yes_mid(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.last_price

    def feasible_yes_bounds(self) -> tuple[float, float]:
        """Physical (p_lo, p_hi) for this bucket's YES given the observed-so-far
        extreme. (0,0)=impossible, (1,1)=certain, (0,1)=undecided. Used by
        market-only strategies that have no forecast to condition."""
        return feasible_yes_bounds(self.low, self.high, self.obs_so_far, self.metric)


@dataclass
class Book:
    """A strategy's current cash and open positions."""
    cash: float
    # (ticker, side) -> {'contracts': float, 'cost': float}
    positions: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)

    def contracts(self, ticker: str, side: str) -> float:
        return self.positions.get((ticker, side), {}).get("contracts", 0.0)

    @property
    def equity(self) -> float:
        """Cash plus cost-basis of open positions — the right base for Kelly
        sizing (sizing off cash alone shrinks bets as positions accumulate)."""
        return self.cash + sum(p["cost"] for p in self.positions.values())


@dataclass
class Signal:
    strategy: str
    ticker: str
    model_prob: float | None
    market_prob: float | None
    edge: float | None
    side: str
    decision: str
    meta: dict[str, Any] = field(default_factory=dict)


class Strategy:
    name: str = "base"

    def __init__(self, params: dict[str, Any], services: "Services"):
        self.params = params
        self.services = services

    def generate(self, ctxs: list[MarketCtx], book: Book) -> tuple[list[Order], list[Signal]]:
        raise NotImplementedError

    # --- shared helpers ---------------------------------------------------
    def kelly_size(self, p: float, price: float, bankroll: float,
                   fraction: float, max_contracts: int) -> int:
        """Fractional-Kelly contract count for a binary YES bet at `price`.

        Edge per contract = p - price; odds b = (1-price)/price. Kelly f* =
        (p*b - (1-p)) / b. Sized off bankroll, capped, floored at 0.
        """
        price = min(max(price, 1e-3), 1 - 1e-3)
        b = (1 - price) / price
        f_star = (p * b - (1 - p)) / b
        if f_star <= 0:
            return 0
        stake = fraction * f_star * bankroll
        contracts = int(stake / price)
        return max(0, min(contracts, max_contracts))

    def fee_per_contract(self, price: float) -> float:
        """Expected taker fee per contract at `price` (rate * P * (1-P)).

        Used to compute the NET edge a taker entry must clear: edge measured
        against the ask alone ignores the fee, which at mid-prices is ~1.75¢
        per contract and was a large share of early paper losses."""
        rate = self.services.fee_cfg.get("taker_rate", 0.07)
        price = min(max(price, 0.0), 1.0)
        return rate * price * (1.0 - price)

    def net_edge(self, p: float, ask: float | None) -> float:
        """Model probability minus the ask minus the per-contract taker fee."""
        if ask is None:
            return -1.0
        return p - ask - self.fee_per_contract(ask)

    def passes_cheap_gates(self, p: float, ask: float) -> bool:
        """Extra evidence gates for low-priced buys.

        A 5¢ edge on a 5¢ contract is a claim of a 2x mispricing — demand a
        minimum model/price ratio there, and (for model strategies) refuse to
        buy below `min_buy_price` outright: the documented favorite-longshot
        bias means cheap tails are systematically overpriced, and the model
        strategies were all bleeding through exactly that trade. Tail bets are
        owned by the dedicated tail strategies."""
        min_buy = self.params.get("min_buy_price", 0.0)
        if ask < min_buy:
            return False
        cheap_below = self.params.get("cheap_price_max", 0.15)
        min_ratio = self.params.get("min_edge_ratio", 1.5)
        if ask < cheap_below and p < ask * min_ratio:
            return False
        return True

    def exit_orders(self, c: MarketCtx, p_yes: float, book: Book) -> list[Order]:
        """Cut or take profit on held positions when the edge has reversed.

        Sells a held side at the bid when the market now overprices that side
        by more than `exit_edge` AFTER the spread-crossing fee — i.e. when
        selling has positive expected value vs holding to settlement."""
        exit_edge = self.params.get("exit_edge")
        if exit_edge is None:
            return []
        out: list[Order] = []
        for side, bid, p_side in (("yes", c.yes_bid, p_yes), ("no", c.no_bid, 1.0 - p_yes)):
            held = book.contracts(c.ticker, side)
            if held < 1 or bid is None:
                continue
            if (bid - self.fee_per_contract(bid)) - p_side >= exit_edge:
                out.append(Order(self.name, c.ticker, side, "sell", held, bid, "taker",
                                 f"exit:p={p_side:.3f},bid={bid:.2f}"))
        return out


@dataclass
class Services:
    """Shared clients/config passed to strategies (e.g. for MM trade fills)."""
    kalshi: Any = None
    conn: Any = None
    bankroll0: float = 1000.0
    fee_cfg: dict[str, Any] = field(default_factory=dict)
