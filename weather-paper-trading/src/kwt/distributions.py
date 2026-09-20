"""Turn a cloud of forecast daily-high samples into bucket probabilities.

A Kalshi temperature bucket is an inclusive integer range [low, high] on the
official daily high (None = open-ended). Because the graded high is an integer,
the continuous event for bucket [low, high] is:

    temp in [low - 0.5, high + 0.5)

We expose three estimators of P(bucket):
  * empirical   - fraction of ensemble members landing in the interval
  * normal      - Gaussian fit to the members (thin tails)
  * student_t   - Student-t fit (fatter tails; better for extreme buckets)
and a configurable blend. Fat tails matter because the favorite-longshot bias
literature says markets misprice the extremes, and a thin Gaussian would too.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _bounds(low: int | None, high: int | None) -> tuple[float, float]:
    lo = -np.inf if low is None else low - 0.5
    hi = np.inf if high is None else high + 0.5
    return lo, hi


def condition_members(members, obs_so_far: float | None,
                      metric: str = "high") -> np.ndarray:
    """Condition a forecast member cloud on today's observed-so-far extreme.

    The realized daily extreme is physically bounded by what has already been
    observed in the local day: the final HIGH can never fall below the max seen
    so far, and the final LOW can never rise above the min seen so far. Flooring
    (high) / ceiling (low) every member at that certain bound removes the
    impossible mass while leaving the remaining-hours uncertainty (already
    carried by the members) intact.

    Self-gating: early in the day `obs_so_far` is a mild morning value that lies
    outside the member cloud, so the clamp is a no-op until the observation
    starts cutting into the members. `obs_so_far is None` -> members unchanged.
    """
    arr = np.asarray(members, dtype=float)
    if obs_so_far is None:
        return arr
    if metric == "low":
        return np.minimum(arr, obs_so_far)
    return np.maximum(arr, obs_so_far)


def feasible_yes_bounds(low: int | None, high: int | None,
                        obs_so_far: float | None,
                        metric: str = "high") -> tuple[float, float]:
    """Physical feasibility interval for a bucket's YES probability given the
    observed-so-far extreme — for market-only strategies that have no forecast
    to condition.

    Bucket is the inclusive integer range [low, high] (None = open-ended); the
    continuous event is temp in [low-0.5, high+0.5). Returns (p_lo, p_hi):
      (0.0, 0.0) -> bucket already impossible
      (1.0, 1.0) -> bucket already certain
      (0.0, 1.0) -> still undecided (or no observation).
    """
    if obs_so_far is None:
        return 0.0, 1.0
    lo, hi = _bounds(low, high)
    if metric == "low":
        # final low L <= obs; impossible if the whole bucket sits above obs,
        # certain if the bucket is open below and its top is at/above obs.
        if low is not None and obs_so_far <= lo:
            return 0.0, 0.0
        if low is None and high is not None and obs_so_far <= hi:
            return 1.0, 1.0
        return 0.0, 1.0
    # metric == 'high': final high H >= obs; impossible if the whole bucket sits
    # below obs, certain if the bucket is open above and its floor is at/below obs.
    if high is not None and obs_so_far >= hi:
        return 0.0, 0.0
    if high is None and low is not None and obs_so_far >= lo:
        return 1.0, 1.0
    return 0.0, 1.0


@dataclass
class Forecast:
    members: np.ndarray          # ensemble member daily highs
    mean: float
    std: float

    @classmethod
    def from_members(cls, members) -> "Forecast":
        arr = np.asarray([m for m in members if m is not None], dtype=float)
        if arr.size == 0:
            raise ValueError("empty member set")
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 1.0
        return cls(members=arr, mean=float(arr.mean()), std=max(sd, 0.5))

    def p_empirical(self, low: int | None, high: int | None) -> float:
        lo, hi = _bounds(low, high)
        return float(np.mean((self.members >= lo) & (self.members < hi)))

    def p_normal(self, low: int | None, high: int | None) -> float:
        lo, hi = _bounds(low, high)
        return float(stats.norm.cdf(hi, self.mean, self.std)
                     - stats.norm.cdf(lo, self.mean, self.std))

    def p_student_t(self, low: int | None, high: int | None, df: float = 6.0) -> float:
        lo, hi = _bounds(low, high)
        # scale so the t-distribution keeps the ensemble variance
        scale = self.std * np.sqrt((df - 2.0) / df) if df > 2 else self.std
        return float(stats.t.cdf(hi, df, self.mean, scale)
                     - stats.t.cdf(lo, df, self.mean, scale))

    def p_bucket(self, low: int | None, high: int | None,
                 blend_empirical: float = 0.6, df: float = 6.0) -> float:
        """Blend of empirical and fat-tailed parametric probability, clipped."""
        emp = self.p_empirical(low, high)
        par = self.p_student_t(low, high, df=df)
        p = blend_empirical * emp + (1.0 - blend_empirical) * par
        return float(min(max(p, 1e-4), 1 - 1e-4))

    def quantiles(self) -> dict[str, float]:
        q = np.percentile(self.members, [5, 10, 50, 90, 95])
        return {"p05": q[0], "p10": q[1], "p50": q[2], "p90": q[3], "p95": q[4]}


def floored_nowcast(nwp: "Forecast", obs_so_far: float | None,
                    floor_buffer: float = 0.0) -> "Forecast":
    """Return a Forecast with members floored at obs_so_far - floor_buffer.

    obs_so_far is a CERTAIN lower bound on the realized daily high (observed
    peak so far): the final high cannot fall below what's already been
    observed today. Flooring every ensemble member there removes impossible
    mass while leaving the remaining-hours uncertainty (already carried by
    the members) intact. floor_buffer is optional °F slack subtracted from
    the floor before clamping.

    Returns nwp UNCHANGED when obs_so_far is None (early in the day the
    floor carries no information). Raises ValueError if the forecast has no
    members.
    """
    if obs_so_far is None:
        return nwp
    floor = obs_so_far - floor_buffer
    members = np.maximum(nwp.members, floor)
    return Forecast.from_members(members)


def normalize_event(probs: dict[str, float]) -> dict[str, float]:
    """Normalize bucket probs within one event so they sum to 1 (buckets partition)."""
    total = sum(probs.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in probs.items()}
