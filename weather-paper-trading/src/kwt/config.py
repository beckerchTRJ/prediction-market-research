"""Configuration + path resolution."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_DB = DATA_DIR / "kwt.db"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Config:
    raw: dict[str, Any]
    cities: dict[str, dict[str, Any]]

    @property
    def forecast(self) -> dict[str, Any]:
        return self.raw["forecast"]

    @property
    def strategies(self) -> dict[str, dict[str, Any]]:
        return self.raw["strategies"]

    @property
    def filters(self) -> dict[str, Any]:
        return self.raw["filters"]

    @property
    def fees(self) -> dict[str, Any]:
        return self.raw["fees"]

    @property
    def risk(self) -> dict[str, Any]:
        return self.raw.get("risk", {"max_deployed_frac": 0.6, "max_market_frac": 0.08})

    @property
    def starting_bankroll(self) -> float:
        return float(self.raw["starting_bankroll"])

    @property
    def max_horizon_days(self) -> float:
        return float(self.raw["max_horizon_days"])

    @property
    def min_horizon_days(self) -> float:
        return float(self.raw["min_horizon_days"])

    def enabled_strategies(self) -> list[str]:
        return [name for name, c in self.strategies.items() if c.get("enabled")]

    def series_to_city(self) -> dict[str, str]:
        return {c["series"]: code for code, c in self.cities.items()}


def load_config(config_path: Path | None = None, cities_path: Path | None = None) -> Config:
    config_path = config_path or CONFIG_DIR / "config.yaml"
    cities_path = cities_path or CONFIG_DIR / "cities.yaml"
    raw = yaml.safe_load(Path(config_path).read_text())
    cities = yaml.safe_load(Path(cities_path).read_text())["cities"]
    DATA_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    return Config(raw=raw, cities=cities)
