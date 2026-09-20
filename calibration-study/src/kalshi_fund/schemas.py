from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MarketObservation(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: datetime
    source: str
    market_id: str
    race_id: str
    contract: str
    cycle: int
    event_date: date
    office: str | None = None
    state: str | None = None
    p_mkt: float = Field(ge=0.0, le=1.0)
    bid: float | None = Field(default=None, ge=0.0, le=1.0)
    ask: float | None = Field(default=None, ge=0.0, le=1.0)
    volume: float | None = Field(default=None, ge=0.0)
    open_interest: float | None = Field(default=None, ge=0.0)
    liquidity_score: float | None = None
    metadata_json: str | None = None

    @model_validator(mode="after")
    def validate_quotes(self) -> "MarketObservation":
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            msg = "ask must be greater than or equal to bid"
            raise ValueError(msg)
        return self


class AnchorObservation(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: datetime
    source: str
    race_id: str
    contract: str
    cycle: int
    event_date: date
    p_anchor: float = Field(ge=0.0, le=1.0)
    office: str | None = None
    state: str | None = None
    anchor_low: float | None = Field(default=None, ge=0.0, le=1.0)
    anchor_high: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_size: float | None = Field(default=None, ge=0.0)
    anchor_quality: float | None = None
    metadata_json: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "AnchorObservation":
        if self.anchor_low is not None and self.anchor_high is not None:
            if self.anchor_high < self.anchor_low:
                msg = "anchor_high must be greater than or equal to anchor_low"
                raise ValueError(msg)
            if not self.anchor_low <= self.p_anchor <= self.anchor_high:
                msg = "p_anchor must lie inside [anchor_low, anchor_high]"
                raise ValueError(msg)
        return self


class SignalRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: datetime
    race_id: str
    contract: str
    cycle: int
    state: str | None = None
    side: str
    entry_price: float = Field(ge=0.0, le=1.0)
    conservative_probability: float = Field(ge=0.0, le=1.0)
    net_edge: float
    suggested_notional_usd: float = Field(ge=0.0)
    thesis_id: str | None = None
    status: str = "OPEN"
    metadata_json: str | None = None


class TradeRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: datetime
    race_id: str
    contract: str
    cycle: int
    strategy_type: str
    side: str
    displayed_price: float = Field(ge=0.0, le=1.0)
    expected_fill_price: float | None = Field(default=None, ge=0.0, le=1.0)
    actual_fill_price: float | None = Field(default=None, ge=0.0, le=1.0)
    notional_usd: float = Field(ge=0.0)
    fees_usd: float = Field(default=0.0, ge=0.0)
    slippage_usd: float = Field(default=0.0, ge=0.0)
    bankroll_usd: float | None = Field(default=None, ge=0.0)
    status: str = "OPEN"
    state: str | None = None
    thesis_id: str | None = None
    notes: str | None = None


def validate_dataframe(frame: pd.DataFrame, schema: type[BaseModel]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    cleaned = frame.replace({np.nan: None})
    for record in cleaned.to_dict(orient="records"):
        validated = schema.model_validate(record)
        records.append(validated.model_dump(mode="json"))
    return pd.DataFrame(records)

