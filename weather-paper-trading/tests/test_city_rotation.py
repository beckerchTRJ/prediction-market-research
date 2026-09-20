"""Rotate which city's contexts sort first each cycle, so a binding per-cycle cap
(max_open_orders, event exposure) doesn't always starve the same tail cities in
cities.yaml's fixed order (DEN/PHIL were starved solid at offset 0 every cycle).
"""
from __future__ import annotations

import types

import kwt.live_engine as le


def _ctx(ticker, city):
    return types.SimpleNamespace(ticker=ticker, city=city)


def _ctxs():
    return [_ctx("NY-1", "nyc"), _ctx("NY-2", "nyc"),
            _ctx("CHI-1", "chi"), _ctx("DEN-1", "den")]


def test_offset_zero_is_original_order():
    out = le._rotate_by_city(_ctxs(), 0)
    assert [c.ticker for c in out] == ["NY-1", "NY-2", "CHI-1", "DEN-1"]


def test_offset_one_rotates_by_whole_city_block():
    out = le._rotate_by_city(_ctxs(), 1)
    # CHI's block moves first, then DEN, then NY's block (both NY contexts stay
    # adjacent -- rotation is by city, not by raw index).
    assert [c.ticker for c in out] == ["CHI-1", "DEN-1", "NY-1", "NY-2"]


def test_offset_wraps_with_modulo():
    # 3 distinct cities (nyc, chi, den) -> offset 3 wraps back to offset 0.
    out = le._rotate_by_city(_ctxs(), 3)
    assert [c.ticker for c in out] == ["NY-1", "NY-2", "CHI-1", "DEN-1"]


def test_empty_list_is_a_noop():
    assert le._rotate_by_city([], 5) == []
