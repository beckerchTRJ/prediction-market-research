"""Tests for kwt.distributions.floored_nowcast — the obs-floor helper extracted
from IntradayNowcastStrategy.generate (behavior-preserving refactor)."""
import numpy as np
import pytest

from kwt.distributions import Forecast, floored_nowcast


def test_members_below_floor_lifted_above_unchanged():
    nwp = Forecast.from_members([60.0, 65.0, 70.0, 75.0])
    result = floored_nowcast(nwp, obs_so_far=68.0)
    # members below 68 get lifted to 68; members already >= 68 stay put
    assert result.members.tolist() == [68.0, 68.0, 70.0, 75.0]


def test_obs_so_far_none_returns_unchanged():
    nwp = Forecast.from_members([60.0, 65.0, 70.0, 75.0])
    result = floored_nowcast(nwp, obs_so_far=None)
    assert result is nwp


def test_p_bucket_below_obs_is_near_zero():
    nwp = Forecast.from_members([60.0, 65.0, 70.0, 75.0])
    result = floored_nowcast(nwp, obs_so_far=80.0)
    # Bucket [50, 55] sits entirely below the observed-so-far floor of 80, so
    # every member has been lifted above it and p_bucket should be ~0.
    p = result.p_bucket(50, 55)
    assert p < 1e-3


def test_equivalent_to_old_inline_flooring():
    members_in = [55.5, 61.2, 66.0, 70.9, 74.4]
    obs_so_far = 63.0
    floor_buffer = 1.5

    # Reproduce the OLD inline computation that used to live in
    # intraday_nowcast.py (lines ~54-57) prior to the refactor.
    old_floor = obs_so_far - floor_buffer
    old_members = np.maximum(np.asarray(members_in, dtype=float), old_floor)
    old_nowcast = Forecast.from_members(old_members)

    nwp = Forecast.from_members(members_in)
    new_nowcast = floored_nowcast(nwp, obs_so_far, floor_buffer=floor_buffer)

    assert new_nowcast.members.tolist() == old_nowcast.members.tolist()
    assert new_nowcast.mean == old_nowcast.mean
    assert new_nowcast.std == old_nowcast.std


def test_empty_forecast_raises_value_error():
    empty = Forecast(members=np.array([]), mean=0.0, std=1.0)
    with pytest.raises(ValueError):
        floored_nowcast(empty, obs_so_far=50.0)
