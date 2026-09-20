import math
from kwt.markout import fill_direction, snapshot_mid, signed_markout


def test_fill_direction_from_book_side():
    # book_side is Kalshi's reliable side-of-book. bid = our resting bid was hit =
    # we bought YES = +1; ask = our ask was lifted = we sold YES = -1. (side, action)
    # does NOT distinguish these — Kalshi encodes our ask fill as (no, sell), which
    # naively reads as long-YES but is actually short.
    assert fill_direction("bid") == 1
    assert fill_direction("ask") == -1
    assert fill_direction(None) is None
    assert fill_direction("") is None


def test_snapshot_mid_prefers_bidask_then_last_then_none():
    assert snapshot_mid({"yes_bid": 0.40, "yes_ask": 0.60, "last_price": 0.99}) == 0.50
    assert snapshot_mid({"yes_bid": 0, "yes_ask": 0, "last_price": 0.42}) == 0.42
    assert snapshot_mid({"yes_bid": 0, "yes_ask": 0, "last_price": 0}) is None


def test_signed_markout_sign_and_none_propagation():
    # long YES, market rose after fill -> favorable (+).
    assert abs(signed_markout(1, 0.50, 0.55) - 0.05) < 1e-9
    # short YES, market rose -> adverse (-).
    assert abs(signed_markout(-1, 0.50, 0.55) - (-0.05)) < 1e-9
    assert signed_markout(1, None, 0.55) is None
    assert signed_markout(1, 0.50, None) is None
