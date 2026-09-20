"""Open-Meteo forecast client (no API key required).

Two products per city:
  * ensemble  -> every member's hourly temperature, grouped into a per-member
                 daily HIGH for each local calendar day. This member cloud is
                 the empirical predictive distribution used for bucket
                 probabilities and fat-tail work.
  * deterministic -> each model's single daily-max run, used for cross-model
                 ensembling / model-agreement features.

Daily high is computed in the station's LOCAL timezone (Kalshi grades the local
calendar-day max), by requesting timezone-localized hourly times and grouping by
date.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import requests

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoClient:
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, url: str, params: dict) -> dict[str, Any]:
        for attempt in range(4):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return {}

    def ensemble_daily_highs(self, lat: float, lon: float, tz: str,
                             models: list[str], forecast_days: int,
                             unit: str = "fahrenheit") -> dict[str, list[float]]:
        """-> {local_date: [member daily-max, ...]} pooled across all members/models."""
        params = {
            "latitude": lat, "longitude": lon, "timezone": tz,
            "hourly": "temperature_2m", "models": ",".join(models),
            "forecast_days": forecast_days, "temperature_unit": unit,
        }
        data = self._get(ENSEMBLE_URL, params)
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        member_keys = [k for k in hourly if k.startswith("temperature_2m")]
        # date -> member_key -> running max
        by_date: dict[str, dict[str, float]] = defaultdict(dict)
        for key in member_keys:
            series = hourly[key]
            for t, val in zip(times, series):
                if val is None:
                    continue
                date = t[:10]
                cur = by_date[date].get(key)
                if cur is None or val > cur:
                    by_date[date][key] = val
        return {date: list(m.values()) for date, m in by_date.items()}

    def deterministic_daily_highs(self, lat: float, lon: float, tz: str,
                                  models: list[str], forecast_days: int,
                                  unit: str = "fahrenheit") -> dict[str, dict[str, float]]:
        """-> {local_date: {model: daily-max}}."""
        params = {
            "latitude": lat, "longitude": lon, "timezone": tz,
            "daily": "temperature_2m_max", "models": ",".join(models),
            "forecast_days": forecast_days, "temperature_unit": unit,
        }
        data = self._get(FORECAST_URL, params)
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        out: dict[str, dict[str, float]] = {d: {} for d in dates}
        for key, series in daily.items():
            if not key.startswith("temperature_2m_max"):
                continue
            model = key.replace("temperature_2m_max_", "").replace("temperature_2m_max", "default")
            for d, val in zip(dates, series):
                if val is not None:
                    out[d][model] = val
        return out

    def archive_daily_highs(self, lat: float, lon: float, tz: str,
                            start: str, end: str,
                            unit: str = "fahrenheit") -> dict[str, float]:
        """Historical observed daily highs (ERA5) over [start, end] (YYYY-MM-DD)."""
        params = {
            "latitude": lat, "longitude": lon, "timezone": tz,
            "start_date": start, "end_date": end,
            "daily": "temperature_2m_max", "temperature_unit": unit,
        }
        data = self._get(ARCHIVE_URL, params)
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        vals = daily.get("temperature_2m_max", [])
        return {d: v for d, v in zip(dates, vals) if v is not None}

    def intraday_state(self, lat: float, lon: float, tz: str,
                       unit: str = "fahrenheit") -> dict:
        """Today's observed-so-far max and remaining-hours forecast max (local day).

        The daily high can never fall below what's already been observed, so this
        gives a hard floor that tightens as the day progresses. Returns
        {observed_max, remaining_max, current_temp, current_time, hours_elapsed}.
        """
        params = {
            "latitude": lat, "longitude": lon, "timezone": tz,
            "hourly": "temperature_2m", "current": "temperature_2m",
            "past_days": 1, "forecast_days": 2, "temperature_unit": unit,
        }
        data = self._get(FORECAST_URL, params)
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        cur_time = data.get("current", {}).get("time")
        cur_temp = data.get("current", {}).get("temperature_2m")
        if not cur_time or not times:
            return {}
        today = cur_time[:10]
        obs, rem = [], []
        for t, v in zip(times, temps):
            if v is None or t[:10] != today:
                continue
            (obs if t <= cur_time else rem).append(v)
        if cur_temp is not None:
            obs.append(cur_temp)
        return {
            "observed_max": max(obs) if obs else None,
            "remaining_max": max(rem) if rem else None,
            "current_temp": cur_temp,
            "current_time": cur_time,
            "hours_elapsed": len(obs),
        }

    def recent_observed_highs(self, lat: float, lon: float, tz: str,
                              past_days: int = 7, unit: str = "fahrenheit") -> dict[str, float]:
        """Recent realized daily highs (for settlement-eval fallback)."""
        params = {
            "latitude": lat, "longitude": lon, "timezone": tz,
            "daily": "temperature_2m_max", "past_days": past_days, "forecast_days": 1,
            "temperature_unit": unit,
        }
        data = self._get(FORECAST_URL, params)
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        vals = daily.get("temperature_2m_max", [])
        return {d: v for d, v in zip(dates, vals) if v is not None}
