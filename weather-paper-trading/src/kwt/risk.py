"""Hard-coded risk controls for the live market-making experiment.

The plan's most important requirement: risk limits must be enforced in code, not
just sitting in YAML. This module is pure (no DB, no network) so every limit is
unit-testable. `live_engine` builds a `LiveState` from reconciled Kalshi data
each cycle, asks `RiskEngine.check_kill` first, then runs every planned order
through `vet_order`, which returns the largest count allowed (possibly 0).

All sizes are in YES-equivalent contracts; capital figures are in dollars.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def worst_case_event_loss(positions: list[tuple[float, float]]) -> float:
    """Max settlement loss across which bucket of a city-day event wins.

    `positions` is the (net_yes, cost) vector over the event's held buckets — the
    ~18 buckets settle off ONE realized high, so exactly one wins. For a winning
    bucket w, a long-YES bucket pays its size iff it IS w; a short (net NO) bucket
    pays |size| iff it is NOT w. Loss(w) = total_cost - payout(w); we return the
    worst over every held bucket winning AND an unheld bucket winning (w = None).
    """
    if not positions:
        return 0.0
    total_cost = sum(cost for _, cost in positions)

    def payout(winner) -> float:
        p = 0.0
        for i, (q, _) in enumerate(positions):
            if q > 0 and i == winner:
                p += q                     # our YES in the winning bucket pays
            elif q < 0 and i != winner:
                p += -q                    # our NO in every losing bucket pays
        return p

    scenarios = list(range(len(positions))) + [None]   # +None: an unheld bucket wins
    return max(total_cost - payout(w) for w in scenarios)


def side_cost(side: str, price: float) -> float:
    """Dollar cost per contract to REST this side, if it fills: a bid buys YES at
    `price`; an ask sells YES you don't hold, which Kalshi settles as a NO buy at
    `1-price`. Using `price` for both understates ask cost — a cheap ask (e.g.
    0.10) actually commits ~0.90/contract — and let capital_at_risk breach the
    hard cap (12.00 -> 14.14) on 2026-07-07 before the next reconcile caught up.
    """
    return price if side == "bid" else (1.0 - price)


@dataclass
class LiveRiskLimits:
    max_capital_at_risk: float = 50.0      # total cost-basis deployed across book
    max_order_size: int = 1               # contracts per single order
    max_position_per_market: int = 5      # net YES-equiv per market
    max_net_inventory: int = 25           # net YES-equiv across all markets
    max_event_exposure: int = 15          # net YES-equiv per (city, target_date)
    max_event_scenario_loss: float = 0.0  # $ worst-case settlement loss per event (0=off)
    daily_max_loss: float = 10.0          # realized loss that trips the kill switch
    max_open_orders: int = 20             # resting orders at once
    min_open_interest: float = 50.0       # skip illiquid markets
    min_quote_hours: float = 3.0          # no quoting inside this window (belt+braces)
    allowed_tickers: list[str] | None = None   # None = no allowlist
    allowed_cities: list[str] | None = None    # None = no allowlist

    @classmethod
    def from_cfg(cls, live_cfg: dict | None) -> "LiveRiskLimits":
        live_cfg = live_cfg or {}
        r = dict(live_cfg.get("risk", {}))
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in r.items() if k in fields})


@dataclass
class LiveState:
    """Reconciled live account state for one cycle (built from Kalshi reads)."""
    funded_capital: float = 0.0
    balance: float = 0.0
    reconciled: bool = False          # True only after a SUCCESSFUL account read.
                                      # False => blind; the maker must cancel-only.
    day_realized_pnl: float = 0.0     # realized + unrealized P&L change since day start
    realized_pnl_total: float = 0.0   # cumulative realized P&L (dollars), experiment-only
    capital_at_risk: float = 0.0
    reserved_cost: float = 0.0        # cost-basis of THIS cycle's resting quotes
    reserved_bid: float = 0.0
    reserved_ask: float = 0.0
    reserved_bid_by_market: dict[str, float] = field(default_factory=dict)
    reserved_ask_by_market: dict[str, float] = field(default_factory=dict)
    reserved_bid_by_event: dict[tuple, float] = field(default_factory=dict)
    reserved_ask_by_event: dict[tuple, float] = field(default_factory=dict)
    open_order_count: int = 0
    net_by_market: dict[str, float] = field(default_factory=dict)
    net_by_event: dict[tuple, float] = field(default_factory=dict)
    # Per event: the (net_yes, cost) vector over held buckets, for the scenario cap.
    event_positions: dict[tuple, list] = field(default_factory=dict)
    # Ground-truth order_ids (firewalled to experiment tickers) currently resting
    # per Kalshi's own account read — NOT our local kv cache, which can go stale
    # if an order fills or TTL-expires between cycles. Consumed by the
    # price-unchanged fast path in live_engine.py to confirm a cached order is
    # still really there before deciding to leave it alone.
    resting_order_ids: set = field(default_factory=set)
    # Exchange-confirmed order details used only for passive queue telemetry.
    resting_orders: dict[str, dict] = field(default_factory=dict)

    @property
    def net_inventory(self) -> float:
        """Signed net YES-equivalent across all markets — the directional book size
        the max_net_inventory cap is meant to bound. Offsetting positions across
        markets net out (a breadth maker holds many small hedged buckets); GROSS
        exposure is bounded separately by max_capital_at_risk and the per-event
        scenario-loss cap. (Was sum(abs(...)) — that conflated net with gross and
        throttled maker breadth; see risk.py directional room below.)"""
        return sum(self.net_by_market.values())


def reserve_order(state: LiveState, ticker: str, side: str, price: float,
                  count: float, event: tuple | None) -> None:
    """Reserve cost and worst-direction inventory for a resting obligation."""
    n = max(float(count), 0.0)
    state.reserved_cost += side_cost(side, price) * n
    if side == "bid":
        state.reserved_bid += n
        state.reserved_bid_by_market[ticker] = state.reserved_bid_by_market.get(ticker, 0) + n
        if event is not None:
            state.reserved_bid_by_event[event] = state.reserved_bid_by_event.get(event, 0) + n
    else:
        state.reserved_ask += n
        state.reserved_ask_by_market[ticker] = state.reserved_ask_by_market.get(ticker, 0) + n
        if event is not None:
            state.reserved_ask_by_event[event] = state.reserved_ask_by_event.get(event, 0) + n


def release_order(state: LiveState, ticker: str, side: str, price: float,
                  count: float, event: tuple | None) -> None:
    """Release a reservation after an exchange-confirmed cancellation."""
    n = max(float(count), 0.0)
    state.reserved_cost = max(0.0, state.reserved_cost - side_cost(side, price) * n)
    if side == "bid":
        state.reserved_bid = max(0.0, state.reserved_bid - n)
        state.reserved_bid_by_market[ticker] = max(
            0.0, state.reserved_bid_by_market.get(ticker, 0) - n)
        if event is not None:
            state.reserved_bid_by_event[event] = max(
                0.0, state.reserved_bid_by_event.get(event, 0) - n)
    else:
        state.reserved_ask = max(0.0, state.reserved_ask - n)
        state.reserved_ask_by_market[ticker] = max(
            0.0, state.reserved_ask_by_market.get(ticker, 0) - n)
        if event is not None:
            state.reserved_ask_by_event[event] = max(
                0.0, state.reserved_ask_by_event.get(event, 0) - n)


@dataclass
class OrderVerdict:
    allowed: int          # contracts permitted (0 = blocked)
    reason: str           # why scaled/blocked ('' when fully allowed)


class RiskEngine:
    def __init__(self, limits: LiveRiskLimits):
        self.limits = limits

    def check_kill(self, state: LiveState) -> str | None:
        """Return a non-None reason if all quoting must stop and orders cancel."""
        lim = self.limits
        if state.day_realized_pnl <= -abs(lim.daily_max_loss):
            return (f"daily loss {state.day_realized_pnl:.2f} <= "
                    f"-{lim.daily_max_loss:.2f}")
        if state.capital_at_risk > lim.max_capital_at_risk:
            return (f"capital at risk {state.capital_at_risk:.2f} > "
                    f"{lim.max_capital_at_risk:.2f}")
        return None

    def market_allowed(self, ticker: str, city: str | None,
                       open_interest: float | None, horizon_days: float) -> str | None:
        """Pre-quote gate independent of order size. None = allowed."""
        lim = self.limits
        if lim.allowed_tickers is not None and ticker not in lim.allowed_tickers:
            return "ticker_not_allowlisted"
        if lim.allowed_cities is not None and city not in lim.allowed_cities:
            return "city_not_allowlisted"
        if open_interest is not None and open_interest < lim.min_open_interest:
            return "below_min_open_interest"
        if horizon_days * 24.0 < lim.min_quote_hours:
            return "inside_no_quote_window"
        return None

    def vet_order(self, *, ticker: str, side: str, price: float, count: float,
                  event: tuple, state: LiveState) -> OrderVerdict:
        """Largest count permitted for this order given all caps.

        side 'bid' increases net YES; 'ask' increases net NO (decreases net YES).
        We bound the *resulting* absolute net per market / per event / overall,
        plus per-order size, open-order count, and remaining capital.
        """
        lim = self.limits
        if state.open_order_count >= lim.max_open_orders:
            return OrderVerdict(0, "max_open_orders")

        # Scenario-loss cap: if this event's worst-case settlement loss is already at
        # the limit, don't grow it with new quotes (reducing orders bypass vet_order
        # via the flatten path). Guards the concentrated multinomial loss an abs-sum
        # event cap misses.
        if lim.max_event_scenario_loss and event in state.event_positions:
            if worst_case_event_loss(state.event_positions[event]) >= lim.max_event_scenario_loss:
                return OrderVerdict(0, "max_event_scenario_loss")

        signed = 1.0 if side == "bid" else -1.0
        cur_mkt = state.net_by_market.get(ticker, 0.0)
        cur_evt = state.net_by_event.get(event, 0.0)

        # Room to grow |net| in the order's direction (caps are on absolute net).
        def directional_room(cur: float, cap: float) -> float:
            if signed > 0:
                return cap - cur          # headroom up to +cap
            return cap + cur              # headroom down to -cap

        if signed > 0:
            cur_mkt += state.reserved_bid_by_market.get(ticker, 0.0)
            cur_evt += state.reserved_bid_by_event.get(event, 0.0)
            cur_inv = state.net_inventory + state.reserved_bid
        else:
            cur_mkt -= state.reserved_ask_by_market.get(ticker, 0.0)
            cur_evt -= state.reserved_ask_by_event.get(event, 0.0)
            cur_inv = state.net_inventory - state.reserved_ask
        room_mkt = directional_room(cur_mkt, lim.max_position_per_market)
        room_evt = directional_room(cur_evt, lim.max_event_exposure)
        # Directional, like the per-market/per-event rooms: how far the SIGNED net
        # book can grow in this order's direction before hitting +/- the cap.
        room_inv = directional_room(cur_inv, lim.max_net_inventory)
        # Reserve the cost of quotes ALREADY placed this cycle: a resting maker
        # order can fill, so it commits capital against the hard cap even before
        # it's a reconciled position. Without this, a whole cycle of orders can be
        # placed with capital_at_risk still reading ~0 and blow past the cap on a
        # correlated sweep before the next reconcile sees it.
        room_cap = (lim.max_capital_at_risk - state.capital_at_risk
                    - state.reserved_cost) / max(side_cost(side, price), 0.01)

        allowed = min(count, lim.max_order_size, room_mkt, room_evt,
                      room_inv, room_cap)
        allowed = int(max(0, allowed))
        if allowed <= 0:
            # Name the binding constraint for the risk-event log.
            binding = min(
                ("max_position_per_market", room_mkt),
                ("max_event_exposure", room_evt),
                ("max_net_inventory", room_inv),
                ("max_capital_at_risk", room_cap),
                ("max_order_size", float(lim.max_order_size)),
                key=lambda kv: kv[1])[0]
            return OrderVerdict(0, binding)
        reason = "" if allowed >= int(min(count, lim.max_order_size)) else "scaled_to_caps"
        return OrderVerdict(allowed, reason)
