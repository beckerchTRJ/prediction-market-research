"""National Weather Service observation client (api.weather.gov, no key).

Used only to record the *realized* daily high for model evaluation/calibration.
P&L settlement reads Kalshi's authoritative `result`, so this is supplementary.
Falls back gracefully; if the official obs aren't retrievable, the collector
uses Open-Meteo recent actuals instead.
"""
from __future__ import annotations

import time
from collections import defaultdict

import requests

UA = {"User-Agent": "kalshi-weather-trading/0.1 (paper research)", "Accept": "application/geo+json"}


class NWSClient:
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(UA)
        self._station_cache: dict[tuple[float, float], str] = {}

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

    def nearest_station(self, lat: float, lon: float) -> str | None:
        key = (round(lat, 3), round(lon, 3))
        if key in self._station_cache:
            return self._station_cache[key]
        pts = self._get(f"https://api.weather.gov/points/{lat},{lon}")
        if not pts:
            return None
        url = pts.get("properties", {}).get("observationStations")
        if not url:
            return None
        stations = self._get(url)
        feats = (stations or {}).get("features", [])
        if not feats:
            return None
        sid = feats[0]["properties"]["stationIdentifier"]
        self._station_cache[key] = sid
        return sid

    def observed_high_f(self, lat: float, lon: float, date: str, tz: str,
                        station: str | None = None) -> float | None:
        """Max observed temperature (°F) over the local calendar day `date`.

        Uses the official settlement station when known, else the nearest.
        """
        sid = station or self.nearest_station(lat, lon)
        if not sid:
            return None
        data = self._get(
            f"https://api.weather.gov/stations/{sid}/observations",
            params={"start": f"{date}T00:00:00{_tz_offset(tz, date)}",
                    "end": f"{date}T23:59:59{_tz_offset(tz, date)}"},
        )
        if not data:
            return None
        highs: dict[str, float] = defaultdict(lambda: -999.0)
        best = None
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            temp_c = (props.get("temperature") or {}).get("value")
            if temp_c is None:
                continue
            f = temp_c * 9 / 5 + 32
            best = f if best is None else max(best, f)
        return round(best) if best is not None else None


def _tz_offset(tz: str, date: str) -> str:
    """Best-effort fixed offset string for the request window."""
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        dt = datetime.fromisoformat(date + "T12:00:00").replace(tzinfo=ZoneInfo(tz))
        off = dt.utcoffset()
        if off is None:
            return "Z"
        total = int(off.total_seconds())
        sign = "+" if total >= 0 else "-"
        total = abs(total)
        return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"
    except Exception:
        return "Z"
