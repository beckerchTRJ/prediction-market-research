from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

EPSILON = 1e-6
GROUP_COLS = ["race_id", "contract", "cycle"]
NUMERIC_FEATURE_CANDIDATES = [
    "p_mkt",
    "p_anchor",
    "anchor_gap",
    "abs_anchor_gap",
    "logit_anchor_gap",
    "days_to_event",
    "spread",
    "volume_log1p",
    "open_interest_log1p",
    "liquidity_score",
    "market_quality_score",
    "anchor_width",
    "anchor_quality",
]
CATEGORICAL_FEATURE_CANDIDATES = [
    "market_source",
    "anchor_source",
    "office",
    "state",
    "contract",
    "cycle",
]


def clip_prob(values: pd.Series | np.ndarray, eps: float = EPSILON) -> pd.Series:
    series = pd.Series(values, copy=False)
    return series.clip(lower=eps, upper=1.0 - eps)


def logit_prob(values: pd.Series | np.ndarray, eps: float = EPSILON) -> pd.Series:
    clipped = clip_prob(values, eps=eps)
    return np.log(clipped / (1.0 - clipped))


def inv_logit(values: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.Series(1.0 / (1.0 + np.exp(-values.to_numpy())), index=values.index)
    array = np.asarray(values)
    return pd.Series(1.0 / (1.0 + np.exp(-array)))


def _sort_for_asof(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    return frame.sort_values([*GROUP_COLS, time_col]).reset_index(drop=True)


def _market_quality_score(frame: pd.DataFrame) -> pd.Series:
    spread_reference = frame["spread"].dropna().median()
    if pd.isna(spread_reference):
        spread_reference = 0.05
    spread_penalty = 1.0 - frame["spread"].fillna(spread_reference).clip(0.0, 1.0)
    depth_signal = frame["volume_log1p"].fillna(0.0) + frame["open_interest_log1p"].fillna(0.0)
    depth_reference = depth_signal.quantile(0.95)
    if pd.isna(depth_reference) or depth_reference <= 0.0:
        depth_reference = 1.0
    depth_scaled = depth_signal / depth_reference
    explicit = frame["liquidity_score"].fillna(0.0)
    return 0.4 * spread_penalty + 0.4 * depth_scaled.clip(0.0, 1.0) + 0.2 * explicit.clip(0.0, 1.0)


def build_panel_frame(
    market_frame: pd.DataFrame,
    anchor_frame: pd.DataFrame,
    correction_horizon_days: int = 7,
) -> pd.DataFrame:
    if market_frame.empty or anchor_frame.empty:
        msg = "market and anchor inputs must both be non-empty"
        raise ValueError(msg)

    market = market_frame.copy()
    anchor = anchor_frame.copy()

    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=False)
    market["event_date"] = pd.to_datetime(market["event_date"], utc=False)
    anchor["timestamp"] = pd.to_datetime(anchor["timestamp"], utc=False)
    anchor["event_date"] = pd.to_datetime(anchor["event_date"], utc=False)

    market = market.rename(columns={"source": "market_source"})
    anchor = anchor.rename(
        columns={
            "source": "anchor_source",
            "state": "anchor_state",
            "office": "anchor_office",
            "event_date": "anchor_event_date",
        }
    )

    market = _sort_for_asof(market, "timestamp")
    anchor = _sort_for_asof(anchor, "timestamp")

    panel = pd.merge_asof(
        market,
        anchor,
        by=GROUP_COLS,
        left_on="timestamp",
        right_on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    panel["target_timestamp"] = panel["timestamp"] + pd.to_timedelta(correction_horizon_days, unit="D")
    future_market = market[GROUP_COLS + ["timestamp", "p_mkt"]].rename(
        columns={"timestamp": "future_timestamp", "p_mkt": "p_mkt_future"}
    )
    future_market = _sort_for_asof(future_market, "future_timestamp")

    panel = panel.sort_values([*GROUP_COLS, "target_timestamp"]).reset_index(drop=True)
    panel = pd.merge_asof(
        panel,
        future_market,
        by=GROUP_COLS,
        left_on="target_timestamp",
        right_on="future_timestamp",
        direction="forward",
        allow_exact_matches=True,
    )

    panel["state"] = panel["state"].fillna(panel["anchor_state"])
    panel["office"] = panel["office"].fillna(panel["anchor_office"])
    panel["spread"] = (panel["ask"] - panel["bid"]).clip(lower=0.0)
    panel["anchor_width"] = panel["anchor_high"] - panel["anchor_low"]
    panel["volume_log1p"] = np.log1p(panel["volume"].fillna(0.0))
    panel["open_interest_log1p"] = np.log1p(panel["open_interest"].fillna(0.0))
    panel["days_to_event"] = (panel["event_date"] - panel["timestamp"].dt.normalize()).dt.days
    panel["anchor_gap"] = panel["p_anchor"] - panel["p_mkt"]
    panel["abs_anchor_gap"] = panel["anchor_gap"].abs()
    panel["logit_anchor_gap"] = logit_prob(panel["p_anchor"]) - logit_prob(panel["p_mkt"])
    panel["market_quality_score"] = _market_quality_score(panel)
    panel["objective_probability"] = panel["p_mkt_future"]
    panel["target_correction"] = logit_prob(panel["p_mkt_future"]) - logit_prob(panel["p_mkt"])

    return panel.sort_values("timestamp").reset_index(drop=True)


def available_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in NUMERIC_FEATURE_CANDIDATES if column in frame.columns]
    categorical = [column for column in CATEGORICAL_FEATURE_CANDIDATES if column in frame.columns]
    numeric = [column for column in numeric if frame[column].notna().any()]
    categorical = [column for column in categorical if frame[column].notna().any()]
    return numeric, categorical


def required_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"missing required columns: {missing}"
        raise ValueError(msg)
