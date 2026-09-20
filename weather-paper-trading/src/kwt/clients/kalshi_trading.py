"""Authenticated Kalshi trading client (portfolio + order placement).

This is the LIVE, real-money counterpart to the read-only `kalshi.KalshiClient`.
It is intentionally a separate module so the paper harness can never accidentally
place an order.

Safety model
------------
- Reads (`get_balance`, `get_positions`, `get_orders`, `get_fills`) always go to
  the API when credentials are present.
- Writes (`create_order`, `cancel_order`, `cancel_all`) are gated by
  `place_orders`. In dry-run the client is constructed with `place_orders=False`
  and every write returns a synthetic, clearly-marked response WITHOUT touching
  the network. Nothing leaves the machine unless you explicitly enable writes.

Auth (verified against https://docs.kalshi.com/getting_started/api_keys):
  headers KALSHI-ACCESS-KEY / -TIMESTAMP / -SIGNATURE; the signed message is
  `timestamp(ms) + METHOD + path` (path WITHOUT query string); RSA-PSS over
  SHA-256, MGF1/SHA-256, salt = digest length; signature base64-encoded.

VERIFY-BEFORE-PROD: the order payload field names / fixed-point formatting and
the exact endpoint paths below should be confirmed against Kalshi's current API
during the demo smoke-test stage before any production order. They are set from
the published docs but Kalshi's schema drifts.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import requests

# Kalshi environments (docs.kalshi.com/getting_started/api_environments).
PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
# Docs-canonical demo host (docs.kalshi.com/getting_started/api_keys). The bare
# demo-api.kalshi.co alias also resolves to the same backend but is undocumented.
DEMO_BASE = "https://external-api.demo.kalshi.co/trade-api/v2"
API_PREFIX = "/trade-api/v2"


class KalshiAuthError(RuntimeError):
    pass


class KalshiTradingError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def fmt_count(count: float) -> str:
    """Kalshi fixed-point contract count (whole contracts)."""
    return str(int(round(count)))


def fmt_price(price: float) -> str:
    """Kalshi order price in dollars, cent-granular (min tick 1c)."""
    price = min(max(price, 0.01), 0.99)
    return f"{price:.2f}"


class KalshiTradingClient:
    def __init__(self, *, mode: str = "dry_run", base_url: str | None = None,
                 api_key_id: str | None = None, private_key_path: str | None = None,
                 timeout: float = 15.0):
        if mode not in ("dry_run", "demo", "prod"):
            raise ValueError(f"mode must be dry_run|demo|prod, got {mode!r}")
        self.mode = mode
        self.place_orders = mode in ("demo", "prod")
        self.base_url = base_url or (PROD_BASE if mode in ("dry_run", "prod") else DEMO_BASE)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._key_id = api_key_id or os.environ.get("KALSHI_API_KEY_ID")
        self._private_key = self._load_private_key(
            private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH"))

    # --- auth -------------------------------------------------------------
    @staticmethod
    def _load_private_key(path: str | None):
        if not path or not os.path.exists(path):
            return None
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        with open(path, "rb") as fh:
            return load_pem_private_key(fh.read(), password=None)

    @property
    def authenticated(self) -> bool:
        return bool(self._key_id and self._private_key)

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if not self.authenticated:
            raise KalshiAuthError(
                "Missing KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH — required "
                "for authenticated portfolio/order endpoints.")
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, *, params: dict | None = None,
                 body: dict | None = None) -> dict[str, Any]:
        url = self.base_url + path
        headers = self._auth_headers(method, API_PREFIX + path)  # sign path w/o query
        for attempt in range(4):
            try:
                r = self.session.request(
                    method, url, params=params,
                    data=json.dumps(body) if body is not None else None,
                    headers=headers, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    raise KalshiTradingError(r.status_code, r.text)
                return r.json() if r.content else {}
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(1.0 * (attempt + 1))
        return {}

    # --- reads (always live when authenticated) ---------------------------
    def get_balance(self) -> dict[str, Any]:
        return self._request("GET", "/portfolio/balance")

    def _paginate(self, path: str, key: str, *, params: dict | None = None,
                  limit: int = 200, max_pages: int = 100) -> list[dict[str, Any]]:
        """Fetch ALL pages of a cursor-paginated portfolio endpoint.

        Firewall-critical: these portfolio reads back the risk math AND the kill
        switch's cancel list. A single unpaginated call returns only the first
        ~100 items, so on a shared account (weather + turnout orders/positions) a
        truncated read could undercount risk or, worse, make the kill switch skip
        our own resting orders on an unfetched page. Following the cursor to
        exhaustion guarantees the full account view. `max_pages` is a runaway
        backstop (100 * 200 = 20k items).
        """
        params = dict(params or {})
        params.setdefault("limit", limit)
        out: list[dict[str, Any]] = []
        cursor = None
        for _ in range(max_pages):
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", path, params=params)
            page = data.get(key, [])
            out.extend(page)
            cursor = data.get("cursor")
            if not cursor or not page:
                break
        return out

    def get_positions(self) -> list[dict[str, Any]]:
        return self._paginate("/portfolio/positions", "market_positions")

    def get_orders(self, status: str = "resting") -> list[dict[str, Any]]:
        return self._paginate("/portfolio/orders", "orders", params={"status": status})

    def get_queue_positions(self, market_tickers: list[str]) -> list[dict[str, Any]]:
        """Return queue-ahead quantities for the account's resting orders.

        This is deliberately a single bulk read rather than one request per
        order. Kalshi rejects an unscoped read (400: "Need to specify
        market_tickers or event_ticker"), so the call is scoped to the given
        tickers and skipped entirely when there are none. The live engine
        filters the result through its experiment firewall before persisting
        it; queue position is diagnostic only and never participates in quote
        placement or risk decisions.
        """
        if not market_tickers:
            return []
        return self._request(
            "GET", "/portfolio/orders/queue_positions",
            params={"market_tickers": ",".join(market_tickers)}).get(
            "queue_positions", [])

    def get_fills(self, min_ts: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if min_ts is not None:
            params["min_ts"] = min_ts
        return self._paginate("/portfolio/fills", "fills", params=params, limit=limit)

    # --- writes (gated by place_orders) -----------------------------------
    def create_order(self, *, ticker: str, side: str, price: float, count: float,
                     client_order_id: str, post_only: bool = True,
                     time_in_force: str = "good_till_canceled",
                     expiration_time: int | None = None) -> dict[str, Any]:
        """Place a resting maker limit order. side='bid' buys YES, 'ask' sells YES.

        In dry-run, returns a synthetic response and sends nothing.
        """
        payload: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "count": fmt_count(count),
            "price": fmt_price(price),
            "time_in_force": time_in_force,
            "post_only": post_only,
            "client_order_id": client_order_id,
            "self_trade_prevention_type": "maker",
        }
        if expiration_time is not None:
            payload["expiration_time"] = expiration_time
        if not self.place_orders:
            return {"dry_run": True, "would_send": payload, "order": {
                "order_id": None, "client_order_id": client_order_id,
                "status": "dry_run"}}
        return self._request("POST", "/portfolio/events/orders", body=payload)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        if not self.place_orders:
            return {"dry_run": True, "order_id": order_id, "status": "dry_run_cancel"}
        return self._request("DELETE", f"/portfolio/events/orders/{order_id}")

    def cancel_all(self, order_ids: list[str]) -> list[dict[str, Any]]:
        """Best-effort cancel of a set of resting orders (used by the kill switch)."""
        out = []
        for oid in order_ids:
            try:
                out.append(self.cancel_order(oid))
            except (KalshiTradingError, KalshiAuthError) as e:
                out.append({"order_id": oid, "error": str(e)})
        return out
