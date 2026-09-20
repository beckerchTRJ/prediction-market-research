"""Polymarket client (Gamma markets API + data-API trade prints).

Scaffold for the copy-trading strategy: Polymarket trades are public on-chain, so
a high-performing weather trader's fills can be mirrored. Weather markets on
Polymarket are sparse and intermittent, so this is disabled by default and
provided as wiring for when target wallets are known.
"""
from __future__ import annotations

import time
from typing import Any

import requests

GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


class PolymarketClient:
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, url: str, params: dict | None = None):
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                if attempt == 2:
                    return None
                time.sleep(1.0 * (attempt + 1))
        return None

    def weather_markets(self, limit: int = 200) -> list[dict[str, Any]]:
        """Open markets whose question/slug looks weather/temperature related."""
        data = self._get(f"{GAMMA}/markets", params={"limit": limit, "closed": "false"})
        if not isinstance(data, list):
            return []
        kw = ("weather", "temperature", "rain", "snow", "hurricane", "degrees",
              "high temp", "warmest", "coldest", "heat")
        out = []
        for m in data:
            text = f"{m.get('question','')} {m.get('slug','')}".lower()
            if any(k in text for k in kw):
                out.append(m)
        return out

    def wallet_trades(self, wallet: str, limit: int = 100) -> list[dict[str, Any]]:
        """Recent trades for a wallet address (newest first)."""
        data = self._get(f"{DATA_API}/trades",
                          params={"user": wallet, "limit": limit, "takerOnly": "false"})
        return data if isinstance(data, list) else []
