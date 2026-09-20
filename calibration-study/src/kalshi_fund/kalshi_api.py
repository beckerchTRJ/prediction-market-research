from __future__ import annotations

import time
from collections.abc import Iterator

import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class KalshiApiError(RuntimeError):
    pass


class KalshiPublicClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        request_delay_seconds: float = 0.15,
        max_retries: int = 5,
    ) -> None:
        self.session = session if session is not None else requests.Session()
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        for attempt in range(self.max_retries):
            if self.request_delay_seconds:
                time.sleep(self.request_delay_seconds)
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            if response.status_code in RETRYABLE_STATUS:
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 30))
                continue
            msg = f"GET {path} failed with status {response.status_code}"
            raise KalshiApiError(msg)
        msg = f"GET {path} failed after {self.max_retries} retries"
        raise KalshiApiError(msg)

    def _iter_paginated(self, path: str, item_key: str, params: dict) -> Iterator[dict]:
        cursor: str | None = None
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self._get(path, page_params)
            yield from payload.get(item_key, [])
            cursor = payload.get("cursor") or ""
            if not cursor:
                return

    def iter_settled_markets(
        self, min_settled_ts: int | None = None, page_limit: int = 1000
    ) -> Iterator[dict]:
        params: dict = {"status": "settled", "limit": page_limit}
        if min_settled_ts is not None:
            params["min_settled_ts"] = min_settled_ts
        yield from self._iter_paginated("/markets", "markets", params)

    def iter_settled_events(self, page_limit: int = 200) -> Iterator[dict]:
        params = {"status": "settled", "limit": page_limit}
        yield from self._iter_paginated("/events", "events", params)

    def get_series(self, series_ticker: str) -> dict:
        payload = self._get(f"/series/{series_ticker}")
        return payload.get("series", payload)

    def get_series_list(self, category: str | None = None) -> list[dict]:
        params: dict = {}
        if category is not None:
            params["category"] = category
        payload = self._get("/series", params)
        return payload.get("series", [])

    def iter_settled_markets_for_series(
        self, series_ticker: str, page_limit: int = 1000
    ) -> Iterator[dict]:
        params = {"status": "settled", "series_ticker": series_ticker, "limit": page_limit}
        yield from self._iter_paginated("/markets", "markets", params)

    def get_market_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1440,
    ) -> list[dict]:
        payload = self._get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        return payload.get("candlesticks", [])
