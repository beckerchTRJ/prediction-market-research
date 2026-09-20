"""Kalshi market-data client.

Read-only market data (markets, orderbook tops, trade prints) is publicly
accessible with no authentication, which is all the paper-trading harness needs.
Optional RSA-PSS request signing is supported (set KALSHI_API_KEY_ID and
KALSHI_PRIVATE_KEY_PATH) for endpoints/rate-limits that require a key.

Temperature-bucket parsing
--------------------------
Each event (e.g. KXHIGHNY-26JUN08) is a partition of mutually-exclusive bucket
markets. We normalize each into an inclusive integer [low, high] range on the
daily high temperature (None = open-ended):

    floor=79, cap=None   "80° or above"  -> low=80,  high=None   (kind 'above')
    cap=72,   floor=None "71° or below"  -> low=None, high=71     (kind 'below')
    floor=78, cap=79     "78° to 79°"     -> low=78,  high=79      (kind 'range')
"""
from __future__ import annotations

import base64
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterator

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def parse_event_date(event_ticker: str) -> str | None:
    """KXHIGHNY-26JUN08 -> '2026-06-08'."""
    if "-" not in event_ticker:
        return None
    tail = event_ticker.split("-")[1]  # '26JUN08'
    if len(tail) < 7:
        return None
    try:
        yy = int(tail[:2])
        mon = MONTHS[tail[2:5].upper()]
        dd = int(tail[5:7])
        return f"20{yy:02d}-{mon:02d}-{dd:02d}"
    except (ValueError, KeyError):
        return None


def parse_bucket(market: dict[str, Any]) -> tuple[str, int | None, int | None]:
    """Return (kind, low, high) inclusive integer range on the daily high."""
    floor = market.get("floor_strike")
    cap = market.get("cap_strike")
    floor = int(floor) if floor is not None else None
    cap = int(cap) if cap is not None else None
    if floor is not None and cap is not None:
        return "range", min(floor, cap), max(floor, cap)
    if floor is not None and cap is None:
        # "floor+1 or above"
        return "above", floor + 1, None
    if cap is not None and floor is None:
        # "cap-1 or below"
        return "below", None, cap - 1
    return "unknown", None, None


def _f(market: dict[str, Any], *names: str) -> float | None:
    for n in names:
        v = market.get(n)
        if v is not None and v != "":
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


class KalshiClient:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 15.0):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._key_id = os.environ.get("KALSHI_API_KEY_ID")
        self._private_key = self._load_private_key()

    def _load_private_key(self):
        path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not path or not os.path.exists(path):
            return None
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            with open(path, "rb") as fh:
                return load_pem_private_key(fh.read(), password=None)
        except Exception:
            return None

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if not (self._key_id and self._private_key):
            return {}
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        url = self.base_url + path
        headers = self._auth_headers("GET", "/trade-api/v2" + path)
        for attempt in range(4):
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(1.0 * (attempt + 1))
        return {}

    def iter_markets(self, series_ticker: str, status: str = "open",
                     limit: int = 200) -> Iterator[dict[str, Any]]:
        cursor = None
        while True:
            params = {"series_ticker": series_ticker, "status": status, "limit": limit}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets", params)
            for m in data.get("markets", []):
                yield m
            cursor = data.get("cursor")
            if not cursor or not data.get("markets"):
                break

    def get_market(self, ticker: str) -> dict[str, Any] | None:
        try:
            return self._get(f"/markets/{ticker}").get("market")
        except requests.RequestException:
            return None

    def get_trades(self, ticker: str, limit: int = 200,
                   min_ts: int | None = None) -> list[dict[str, Any]]:
        """Recent executed trade prints for a market (newest first).

        Used by the market-making sim to fill quotes against real flow.
        """
        params: dict[str, Any] = {"ticker": ticker, "limit": limit}
        if min_ts is not None:
            params["min_ts"] = min_ts
        try:
            return self._get("/markets/trades", params).get("trades", [])
        except requests.RequestException:
            return []

    def trades_since(self, ticker: str, since_iso: str | None,
                     max_pages: int = 25, page: int = 200) -> list[dict[str, Any]]:
        """All trade prints with created_time > since_iso (paginates via cursor).

        Avoids the single-page watermark gap where >page new trades would be
        permanently skipped. Returned oldest-first.
        """
        out: list[dict[str, Any]] = []
        cursor = None
        for _ in range(max_pages):
            params: dict[str, Any] = {"ticker": ticker, "limit": page}
            if cursor:
                params["cursor"] = cursor
            try:
                data = self._get("/markets/trades", params)
            except requests.RequestException:
                break
            trades = data.get("trades", [])
            if not trades:
                break
            out.extend(trades)
            # newest-first; stop once we've paged past the watermark
            if since_iso and any(t.get("created_time", "") <= since_iso for t in trades):
                break
            cursor = data.get("cursor")
            if not cursor:
                break
        fresh = [t for t in out if not since_iso or t.get("created_time", "") > since_iso]
        return sorted(fresh, key=lambda t: t.get("created_time", ""))

    @staticmethod
    def normalize_market(m: dict[str, Any], series_to_city: dict[str, str]) -> dict[str, Any]:
        kind, low, high = parse_bucket(m)
        series = m.get("series_ticker") or (m.get("ticker", "").split("-")[0])
        event = m.get("event_ticker", "")
        return {
            "ticker": m["ticker"],
            "series_ticker": series,
            "event_ticker": event,
            "city": series_to_city.get(series, series.replace("KXHIGH", "")),
            "target_date": parse_event_date(event),
            "bucket_kind": kind,
            "low": low,
            "high": high,
            "yes_sub_title": m.get("yes_sub_title"),
            "status": m.get("status"),
            "result": (m.get("result") or None) or None,
            "open_time": m.get("open_time"),
            "close_time": m.get("close_time"),
        }

    @staticmethod
    def snapshot_row(m: dict[str, Any], ts: str) -> dict[str, Any]:
        return {
            "ts": ts,
            "ticker": m["ticker"],
            "yes_bid": _f(m, "yes_bid_dollars"),
            "yes_ask": _f(m, "yes_ask_dollars"),
            "no_bid": _f(m, "no_bid_dollars"),
            "no_ask": _f(m, "no_ask_dollars"),
            "last_price": _f(m, "last_price_dollars"),
            "volume": _f(m, "volume", "volume_fp"),
            "open_interest": _f(m, "open_interest_fp", "open_interest"),
            "liquidity": _f(m, "liquidity_dollars"),
        }
