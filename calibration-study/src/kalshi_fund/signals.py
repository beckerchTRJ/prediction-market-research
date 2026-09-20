from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kalshi_fund.panel import clip_prob


@dataclass(frozen=True)
class SignalConfig:
    alpha: float
    sigma: float
    z_score: float = 1.0
    fee: float = 0.0
    slippage: float = 0.0
    hurdle: float = 0.0
    bankroll_usd: float = 10_000.0
    fractional_kelly: float = 0.25
    race_cap_fraction: float = 0.05
    state_cap_fraction: float = 0.10
    cycle_cap_fraction: float = 0.15
    total_cap_fraction: float = 0.25


def open_exposure_summary(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty or "notional_usd" not in trades.columns:
        return {"total": 0.0, "race": {}, "state": {}, "cycle": {}}

    status = trades.get("status", pd.Series(["OPEN"] * len(trades))).fillna("OPEN").str.upper()
    open_trades = trades.loc[status == "OPEN"].copy()
    if open_trades.empty:
        return {"total": 0.0, "race": {}, "state": {}, "cycle": {}}

    notional = open_trades["notional_usd"].fillna(0.0)
    race = open_trades.groupby("race_id")["notional_usd"].sum().to_dict() if "race_id" in open_trades else {}
    state = open_trades.groupby("state")["notional_usd"].sum().to_dict() if "state" in open_trades else {}
    cycle = open_trades.groupby("cycle")["notional_usd"].sum().to_dict() if "cycle" in open_trades else {}
    return {"total": float(notional.sum()), "race": race, "state": state, "cycle": cycle}


def _remaining_capacity(row: pd.Series, config: SignalConfig, exposure: dict[str, object]) -> float:
    total_remaining = config.bankroll_usd * config.total_cap_fraction - float(exposure["total"])
    race_remaining = config.bankroll_usd * config.race_cap_fraction - float(exposure["race"].get(row["race_id"], 0.0))
    state_remaining = config.bankroll_usd * config.state_cap_fraction - float(exposure["state"].get(row.get("state"), 0.0))
    cycle_remaining = config.bankroll_usd * config.cycle_cap_fraction - float(exposure["cycle"].get(row["cycle"], 0.0))
    return max(0.0, min(total_remaining, race_remaining, state_remaining, cycle_remaining))


def _update_exposure(row: pd.Series, notional: float, exposure: dict[str, object]) -> None:
    exposure["total"] = float(exposure["total"]) + notional
    exposure["race"][row["race_id"]] = float(exposure["race"].get(row["race_id"], 0.0)) + notional
    exposure["state"][row.get("state")] = float(exposure["state"].get(row.get("state"), 0.0)) + notional
    exposure["cycle"][row["cycle"]] = float(exposure["cycle"].get(row["cycle"], 0.0)) + notional


def generate_signal_frame(
    predictions: pd.DataFrame,
    config: SignalConfig,
    open_trades: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = predictions.copy()
    if frame.empty:
        return frame

    exposure = open_exposure_summary(open_trades if open_trades is not None else pd.DataFrame())
    total_cost = config.fee + config.slippage + config.hurdle

    frame["p_blend"] = clip_prob((1.0 - config.alpha) * frame["p_mkt"] + config.alpha * frame["p_fair_model"])
    frame["p_yes_cons"] = clip_prob(frame["p_blend"] - config.z_score * config.sigma)
    frame["p_no_cons"] = clip_prob((1.0 - frame["p_blend"]) - config.z_score * config.sigma)
    frame["price_yes"] = clip_prob(frame["ask"].where(frame["ask"].notna(), frame["p_mkt"]))
    frame["price_no"] = clip_prob(1.0 - frame["bid"].where(frame["bid"].notna(), frame["p_mkt"]))
    frame["net_edge_yes"] = frame["p_yes_cons"] - frame["price_yes"] - total_cost
    frame["net_edge_no"] = frame["p_no_cons"] - frame["price_no"] - total_cost
    frame["kelly_yes"] = config.fractional_kelly * np.maximum(0.0, (frame["p_yes_cons"] - frame["price_yes"]) / (1.0 - frame["price_yes"]))
    frame["kelly_no"] = config.fractional_kelly * np.maximum(0.0, (frame["p_no_cons"] - frame["price_no"]) / (1.0 - frame["price_no"]))
    frame["chosen_side"] = np.where(frame["net_edge_yes"] >= frame["net_edge_no"], "YES", "NO")
    frame["conservative_probability"] = np.where(frame["chosen_side"] == "YES", frame["p_yes_cons"], frame["p_no_cons"])
    frame["entry_price"] = np.where(frame["chosen_side"] == "YES", frame["price_yes"], frame["price_no"])
    frame["net_edge"] = np.where(frame["chosen_side"] == "YES", frame["net_edge_yes"], frame["net_edge_no"])
    frame["selected_kelly_fraction"] = np.where(frame["chosen_side"] == "YES", frame["kelly_yes"], frame["kelly_no"])
    frame["uncapped_notional_usd"] = config.bankroll_usd * frame["selected_kelly_fraction"]
    frame["sigma"] = config.sigma

    allocation_frame = frame.sort_values("net_edge", ascending=False).copy()
    remaining_capacities: list[float] = []
    suggested_notional: list[float] = []
    statuses: list[str] = []
    for _, row in allocation_frame.iterrows():
        remaining = _remaining_capacity(row, config, exposure)
        provisional_notional = min(float(row["uncapped_notional_usd"]), remaining) if float(row["net_edge"]) > 0.0 else 0.0
        status = "OPEN" if provisional_notional > 0.0 else "HOLD"
        remaining_capacities.append(remaining)
        suggested_notional.append(provisional_notional)
        statuses.append(status)
        if provisional_notional > 0.0:
            _update_exposure(row, provisional_notional, exposure)

    allocation_frame["remaining_capacity_usd"] = remaining_capacities
    allocation_frame["suggested_notional_usd"] = suggested_notional
    allocation_frame["status"] = statuses

    signal_columns = [
        "timestamp",
        "race_id",
        "contract",
        "cycle",
        "state",
        "chosen_side",
        "entry_price",
        "conservative_probability",
        "net_edge",
        "selected_kelly_fraction",
        "remaining_capacity_usd",
        "suggested_notional_usd",
        "status",
        "p_mkt",
        "p_fair_model",
        "p_blend",
        "sigma",
    ]
    return allocation_frame[signal_columns].rename(columns={"chosen_side": "side"}).sort_values("net_edge", ascending=False)
