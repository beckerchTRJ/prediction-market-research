"""A city-day's ~18 buckets settle off ONE realized high, so the event is a single
multinomial outcome, not independent bets (Fable §2e). The abs-sum event cap can
wave through a book with a large concentrated loss if a particular bucket wins.
worst_case_event_loss computes the max settlement loss across which bucket wins,
from the (net, cost) vector.
"""
from __future__ import annotations

from kwt.risk import worst_case_event_loss, RiskEngine, LiveRiskLimits, LiveState

EVENT = ("nyc", "2026-07-04")


def test_vet_blocks_new_quotes_when_event_at_scenario_cap():
    eng = RiskEngine(LiveRiskLimits(max_event_scenario_loss=1.5))
    state = LiveState()
    state.event_positions[EVENT] = [(2.0, 1.0), (-2.0, 1.0)]   # worst case $2 >= $1.5
    v = eng.vet_order(ticker="KXHIGHNY-26JUL04-B80", side="bid", price=0.5,
                      count=1, event=EVENT, state=state)
    assert v.allowed == 0 and v.reason == "max_event_scenario_loss"


def test_vet_allows_new_quotes_below_scenario_cap():
    eng = RiskEngine(LiveRiskLimits(max_event_scenario_loss=5.0))
    state = LiveState()
    state.event_positions[EVENT] = [(2.0, 1.0)]               # worst case $1 < $5
    v = eng.vet_order(ticker="KXHIGHNY-26JUL04-B80", side="bid", price=0.5,
                      count=1, event=EVENT, state=state)
    assert v.allowed > 0


def test_scenario_cap_off_by_default():
    eng = RiskEngine(LiveRiskLimits())                        # max_event_scenario_loss=0
    state = LiveState()
    state.event_positions[EVENT] = [(2.0, 1.0), (-2.0, 1.0)]
    v = eng.vet_order(ticker="KXHIGHNY-26JUL04-B80", side="bid", price=0.5,
                      count=1, event=EVENT, state=state)
    assert v.allowed > 0                                       # cap disabled -> allowed


def test_no_positions_no_loss():
    assert worst_case_event_loss([]) == 0.0


def test_single_long_yes_worst_case_is_its_cost():
    # 2 YES bought for $1: lose the $1 if the bucket misses.
    assert worst_case_event_loss([(2.0, 1.0)]) == 1.0


def test_single_short_no_worst_case_is_its_cost():
    # 2 NO bought for $1: lose the $1 if THIS bucket wins.
    assert worst_case_event_loss([(-2.0, 1.0)]) == 1.0


def test_concentrated_cross_bucket_loss_is_caught():
    # Long 2 YES in bucket A ($1) AND long 2 NO in bucket B ($1): if B wins, YES-A
    # is worthless and NO-B is worthless -> lose both $2. abs-sum net = 0 would hide it.
    assert worst_case_event_loss([(2.0, 1.0), (-2.0, 1.0)]) == 2.0


def test_no_across_two_buckets_is_self_hedging():
    # 2 NO in A ($1) + 2 NO in B ($1): only one bucket can win, so the OTHER bucket's
    # NO always pays. If A wins: A's NO worthless (-$1) but B's NO pays $2 (+$1) ->
    # break even. Worst case is $0, not a loss — the scenario cap correctly sees this
    # where an abs-sum cap (|net| = 4) would look maximally exposed.
    assert worst_case_event_loss([(-2.0, 1.0), (-2.0, 1.0)]) == 0.0
