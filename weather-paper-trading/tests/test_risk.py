"""Risk-control enforcement for the live experiment (pure, no network)."""
from __future__ import annotations

from kwt.risk import LiveRiskLimits, LiveState, RiskEngine


def _eng(**kw):
    return RiskEngine(LiveRiskLimits(**kw))


def test_kill_switch_on_daily_loss():
    eng = _eng(daily_max_loss=10.0)
    assert eng.check_kill(LiveState(day_realized_pnl=-9.99)) is None
    assert eng.check_kill(LiveState(day_realized_pnl=-10.0)) is not None
    assert eng.check_kill(LiveState(day_realized_pnl=-25.0)) is not None


def test_kill_switch_on_capital_at_risk():
    eng = _eng(max_capital_at_risk=50.0)
    assert eng.check_kill(LiveState(capital_at_risk=50.0)) is None
    assert eng.check_kill(LiveState(capital_at_risk=50.01)) is not None


def test_order_capped_to_max_order_size():
    eng = _eng(max_order_size=1, max_position_per_market=5, max_event_exposure=15,
               max_net_inventory=25, max_capital_at_risk=50.0)
    v = eng.vet_order(ticker="T", side="bid", price=0.5, count=5, event=("nyc", "d"),
                      state=LiveState())
    assert v.allowed == 1


def test_order_blocked_at_max_open_orders():
    eng = _eng(max_open_orders=2)
    v = eng.vet_order(ticker="T", side="bid", price=0.5, count=1, event=("nyc", "d"),
                      state=LiveState(open_order_count=2))
    assert v.allowed == 0 and v.reason == "max_open_orders"


def test_per_market_directional_room():
    # Already long 5 YES at the per-market cap -> a further bid is blocked, but an
    # ask (which reduces net YES) is allowed.
    eng = _eng(max_order_size=5, max_position_per_market=5, max_event_exposure=99,
               max_net_inventory=99, max_capital_at_risk=999.0)
    st = LiveState(net_by_market={"T": 5.0})
    assert eng.vet_order(ticker="T", side="bid", price=0.5, count=5,
                         event=("nyc", "d"), state=st).allowed == 0
    assert eng.vet_order(ticker="T", side="ask", price=0.5, count=5,
                         event=("nyc", "d"), state=st).allowed == 5


def test_capital_room_scales_order_down():
    # $2 of remaining capital at $0.50/contract -> at most 4 contracts.
    eng = _eng(max_order_size=10, max_position_per_market=99, max_event_exposure=99,
               max_net_inventory=99, max_capital_at_risk=2.0)
    v = eng.vet_order(ticker="T", side="bid", price=0.5, count=10, event=("nyc", "d"),
                      state=LiveState(capital_at_risk=0.0))
    assert v.allowed == 4 and v.reason == "scaled_to_caps"


def test_reserved_cost_counts_against_capital_room():
    # $12 hard cap; $10 already reserved by this cycle's resting quotes leaves $2,
    # so at $0.50/contract only 4 more contracts fit — resting orders commit
    # capital even before they reconcile as positions.
    eng = _eng(max_order_size=10, max_position_per_market=99, max_event_exposure=99,
               max_net_inventory=99, max_capital_at_risk=12.0)
    v = eng.vet_order(ticker="T", side="bid", price=0.5, count=10, event=("nyc", "d"),
                      state=LiveState(capital_at_risk=0.0, reserved_cost=10.0))
    assert v.allowed == 4 and v.reason == "scaled_to_caps"
    # Fully reserved -> no room, blocked on the capital cap.
    v2 = eng.vet_order(ticker="T", side="bid", price=0.5, count=1, event=("nyc", "d"),
                       state=LiveState(capital_at_risk=6.0, reserved_cost=6.0))
    assert v2.allowed == 0 and v2.reason == "max_capital_at_risk"


def test_net_inventory_is_net_not_gross():
    # Offsetting positions across markets (net 0, gross 12) must NOT saturate the
    # net-inventory cap — it bounds directional book size, not gross breadth.
    eng = _eng(max_order_size=99, max_position_per_market=99, max_event_exposure=99,
               max_net_inventory=8, max_capital_at_risk=999.0)
    st = LiveState(net_by_market={"A": 6.0, "B": -6.0})   # net 0, gross 12
    v = eng.vet_order(ticker="C", side="bid", price=0.5, count=99,
                      event=("x", "d"), state=st)
    assert v.allowed == 8    # full net room; a gross cap would block (8-12 < 0)


def test_net_inventory_room_is_directional():
    # Net short 6 across the book. Growing the short (ask) has little room; reducing
    # it (bid) has lots — mirrors the per-market/per-event directional logic.
    eng = _eng(max_order_size=99, max_position_per_market=99, max_event_exposure=99,
               max_net_inventory=8, max_capital_at_risk=999.0)
    st = LiveState(net_by_market={"HELD": -6.0})
    ask = eng.vet_order(ticker="NEW", side="ask", price=0.5, count=99,
                        event=("x", "d"), state=st)
    bid = eng.vet_order(ticker="NEW", side="bid", price=0.5, count=99,
                        event=("x", "d"), state=st)
    assert ask.allowed == 2     # room down to -8 from -6
    assert bid.allowed == 14    # room up to +8 from -6


def test_market_allowed_gates():
    eng = _eng(min_open_interest=50, min_quote_hours=3.0,
               allowed_cities=["nyc"])
    assert eng.market_allowed("T", "nyc", 100, 1.0) is None
    assert eng.market_allowed("T", "nyc", 10, 1.0) == "below_min_open_interest"
    assert eng.market_allowed("T", "mia", 100, 1.0) == "city_not_allowlisted"
    # 2h to settle with a 3h window -> blocked
    assert eng.market_allowed("T", "nyc", 100, 2.0 / 24.0) == "inside_no_quote_window"


def test_limits_from_cfg_ignores_unknown_keys():
    lim = LiveRiskLimits.from_cfg({"risk": {"max_order_size": 3, "bogus": 1}})
    assert lim.max_order_size == 3


def test_side_cost_ask_is_one_minus_price():
    from kwt.risk import side_cost
    assert side_cost("bid", 0.10) == 0.10
    assert abs(side_cost("ask", 0.10) - 0.90) < 1e-9


def test_ask_room_cap_uses_one_minus_price_not_price():
    # A cheap ask (0.10) actually costs 1-0.10=0.90/contract on fill. With a $1.00
    # cap and 10 requested contracts, only 1 fits ($0.90) — the OLD buggy formula
    # (room_cap = cap/price = 1.00/0.10 = 10) would have allowed all 10, letting
    # capital_at_risk breach the cap by ~9x once fills reconciled (the real
    # 2026-07-07 incident: capital_at_risk hit 14.14 against a 12.00 cap).
    eng = _eng(max_order_size=10, max_position_per_market=99, max_event_exposure=99,
               max_net_inventory=99, max_capital_at_risk=1.0)
    v = eng.vet_order(ticker="T", side="ask", price=0.10, count=10, event=("nyc", "d"),
                      state=LiveState())
    assert v.allowed == 1

    # A bid at the same price is unaffected (cost=price for bids): $1.00/$0.10 = 10.
    v_bid = eng.vet_order(ticker="T", side="bid", price=0.10, count=10, event=("nyc", "d"),
                          state=LiveState())
    assert v_bid.allowed == 10


def test_outstanding_orders_reduce_directional_room_before_new_orders():
    from kwt.risk import reserve_order
    eng = _eng(max_order_size=10, max_position_per_market=3,
               max_event_exposure=4, max_net_inventory=5,
               max_capital_at_risk=100)
    state = LiveState()
    reserve_order(state, "T", "bid", .10, 2, ("nyc", "d"))
    v = eng.vet_order(ticker="T", side="bid", price=.10, count=3,
                      event=("nyc", "d"), state=state)
    assert v.allowed == 1  # 2 resting + 1 new reaches per-market cap 3


def test_bid_and_ask_reservations_track_worst_direction_separately():
    from kwt.risk import reserve_order
    eng = _eng(max_order_size=10, max_position_per_market=10,
               max_event_exposure=10, max_net_inventory=3,
               max_capital_at_risk=100)
    state = LiveState()
    reserve_order(state, "A", "bid", .50, 2, ("nyc", "d"))
    reserve_order(state, "B", "ask", .50, 2, ("chi", "d"))
    assert eng.vet_order(ticker="C", side="bid", price=.5, count=3,
                         event=("den", "d"), state=state).allowed == 1
    assert eng.vet_order(ticker="C", side="ask", price=.5, count=3,
                         event=("den", "d"), state=state).allowed == 1
