import pytest

from kalshi_fund.kalshi_api import KalshiApiError, KalshiPublicClient


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def make_client(responses):
    session = FakeSession(responses)
    client = KalshiPublicClient(session=session, request_delay_seconds=0.0)
    return client, session


def test_iter_settled_markets_paginates_until_empty_cursor():
    client, session = make_client([
        FakeResponse(200, {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "next1"}),
        FakeResponse(200, {"markets": [{"ticker": "C"}], "cursor": ""}),
    ])
    markets = list(client.iter_settled_markets(min_settled_ts=1700000000))
    assert [m["ticker"] for m in markets] == ["A", "B", "C"]
    first_url, first_params = session.calls[0]
    assert first_url.endswith("/markets")
    assert first_params["status"] == "settled"
    assert first_params["min_settled_ts"] == 1700000000
    assert session.calls[1][1]["cursor"] == "next1"


def test_retries_on_429_then_succeeds():
    client, session = make_client([
        FakeResponse(429, {}),
        FakeResponse(200, {"series": {"ticker": "KXTSA", "category": "Transportation"}}),
    ])
    series = client.get_series("KXTSA")
    assert series["category"] == "Transportation"
    assert len(session.calls) == 2


def test_raises_after_exhausting_retries():
    session = FakeSession([FakeResponse(429, {}), FakeResponse(429, {})])
    client = KalshiPublicClient(session=session, request_delay_seconds=0.0, max_retries=2)
    with pytest.raises(KalshiApiError, match="failed after 2 retries"):
        client.get_series("KXTSA")
    assert len(session.calls) == 2


def test_raises_on_client_error():
    client, _ = make_client([FakeResponse(404, {"error": "not found"})])
    with pytest.raises(KalshiApiError):
        client.get_series("NOPE")


def test_get_series_list_hits_series_endpoint_and_unwraps():
    client, session = make_client([
        FakeResponse(200, {"series": [{"ticker": "KXTSA", "category": "Transportation"}]}),
    ])
    series = client.get_series_list()
    assert series == [{"ticker": "KXTSA", "category": "Transportation"}]
    url, params = session.calls[0]
    assert url.endswith("/series")
    assert params == {}


def test_get_series_list_passes_category_param_when_given():
    client, session = make_client([
        FakeResponse(200, {"series": [{"ticker": "KXTSA", "category": "Transportation"}]}),
    ])
    client.get_series_list(category="Transportation")
    url, params = session.calls[0]
    assert url.endswith("/series")
    assert params == {"category": "Transportation"}


def test_iter_settled_markets_for_series_paginates_until_empty_cursor():
    client, session = make_client([
        FakeResponse(200, {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "next1"}),
        FakeResponse(200, {"markets": [{"ticker": "C"}], "cursor": ""}),
    ])
    markets = list(client.iter_settled_markets_for_series("KXTSA"))
    assert [m["ticker"] for m in markets] == ["A", "B", "C"]
    first_url, first_params = session.calls[0]
    assert first_url.endswith("/markets")
    assert first_params["status"] == "settled"
    assert first_params["series_ticker"] == "KXTSA"
    assert session.calls[1][1]["cursor"] == "next1"


def test_candlesticks_path_and_unwrap():
    client, session = make_client([
        FakeResponse(200, {"ticker": "T1", "candlesticks": [{"end_period_ts": 123}]}),
    ])
    candles = client.get_market_candlesticks("KXTSA", "T1", start_ts=100, end_ts=200)
    assert candles == [{"end_period_ts": 123}]
    url, params = session.calls[0]
    assert url.endswith("/series/KXTSA/markets/T1/candlesticks")
    assert params == {"start_ts": 100, "end_ts": 200, "period_interval": 1440}
