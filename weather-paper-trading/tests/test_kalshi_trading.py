"""Trading client: dry-run safety, formatting, auth gating (no network)."""
from __future__ import annotations

import pytest

from kwt.clients.kalshi_trading import (
    KalshiAuthError, KalshiTradingClient, fmt_count, fmt_price)


def test_fmt_helpers():
    assert fmt_count(5) == "5"
    assert fmt_count(5.0) == "5"
    assert fmt_price(0.5) == "0.50"
    assert fmt_price(0.567) == "0.57"
    assert fmt_price(0.0) == "0.01"     # clipped to min tick
    assert fmt_price(1.5) == "0.99"     # clipped to max tick


def test_dry_run_does_not_place(monkeypatch):
    c = KalshiTradingClient(mode="dry_run")
    assert c.place_orders is False

    # Any network use would be a bug in dry-run: make the session explode.
    def boom(*a, **k):
        raise AssertionError("dry_run must not touch the network")
    monkeypatch.setattr(c.session, "request", boom)

    resp = c.create_order(ticker="KXHIGHNY-26JUN08-T70", side="bid", price=0.50,
                          count=1, client_order_id="abc")
    assert resp["dry_run"] is True
    p = resp["would_send"]
    assert p["ticker"] == "KXHIGHNY-26JUN08-T70"
    assert p["side"] == "bid"
    assert p["count"] == "1"
    assert p["price"] == "0.50"
    assert p["post_only"] is True
    assert p["time_in_force"] == "good_till_canceled"
    assert p["client_order_id"] == "abc"
    # cancel in dry_run is also a no-op
    assert c.cancel_order("xyz")["status"] == "dry_run_cancel"


def test_unauthenticated_reads_raise():
    c = KalshiTradingClient(mode="dry_run")  # no env creds in test
    assert c.authenticated is False
    with pytest.raises(KalshiAuthError):
        c.get_balance()


def test_mode_selects_base_url():
    assert KalshiTradingClient(mode="prod").place_orders is True
    assert KalshiTradingClient(mode="demo").place_orders is True
    assert "demo" in KalshiTradingClient(mode="demo").base_url


def test_portfolio_reads_follow_pagination(monkeypatch):
    # A single unpaginated read would return only page 1 and could hide the
    # experiment's own resting orders from the kill switch. Confirm get_orders
    # follows the cursor across pages until it's exhausted.
    c = KalshiTradingClient(mode="dry_run")
    seq = [
        {"orders": [{"order_id": "1", "ticker": "KXHIGHNY-A"}], "cursor": "c1"},
        {"orders": [{"order_id": "2", "ticker": "KXMIDTERM-B"}], "cursor": "c2"},
        {"orders": [{"order_id": "3", "ticker": "KXHIGHNY-C"}], "cursor": ""},
    ]
    calls = {"n": 0}

    def fake_request(method, path, *, params=None, body=None):
        page = seq[calls["n"]]
        calls["n"] += 1
        return page

    monkeypatch.setattr(c, "_request", fake_request)
    orders = c.get_orders()
    assert [o["order_id"] for o in orders] == ["1", "2", "3"]  # all 3 pages merged
    assert calls["n"] == 3  # stopped when cursor went empty


def test_queue_positions_use_one_bulk_authenticated_read(monkeypatch):
    # Kalshi rejects an unscoped queue_positions read (HTTP 400: "Need to
    # specify market_tickers or event_ticker"), so the bulk read must scope
    # itself to the tickers of our resting orders.
    c = KalshiTradingClient(mode="dry_run")
    calls = []

    def fake_request(method, path, *, params=None, body=None):
        calls.append((method, path, params))
        return {"queue_positions": [{"order_id": "o1", "queue_position_fp": "3.00"}]}

    monkeypatch.setattr(c, "_request", fake_request)
    got = c.get_queue_positions(["KXHIGHNY-A", "KXHIGHCHI-B"])
    assert got == [{"order_id": "o1", "queue_position_fp": "3.00"}]
    assert calls == [("GET", "/portfolio/orders/queue_positions",
                      {"market_tickers": "KXHIGHNY-A,KXHIGHCHI-B"})]


def test_queue_positions_with_no_tickers_skip_the_request(monkeypatch):
    # No resting orders -> nothing to measure; an unscoped call would 400.
    c = KalshiTradingClient(mode="dry_run")

    def fake_request(method, path, *, params=None, body=None):
        raise AssertionError("no request expected without tickers")

    monkeypatch.setattr(c, "_request", fake_request)
    assert c.get_queue_positions([]) == []
