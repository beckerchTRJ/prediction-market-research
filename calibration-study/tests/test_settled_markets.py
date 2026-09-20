from kalshi_fund.settled_markets import build_settled_market_frame, resolve_series_ticker

MARKETS = [
    {
        "ticker": "KXTSAW-26JAN05-B2.5",
        "event_ticker": "KXTSAW-26JAN05",
        "title": "TSA weekly 2.5M",
        "market_type": "binary",
        "result": "yes",
        "open_time": "2025-12-29T15:00:00Z",
        "close_time": "2026-01-05T15:00:00Z",
        "volume_fp": "1234",
    },
    {
        "ticker": "SCALAR-1",
        "event_ticker": "SCALAR-EV",
        "result": "scalar",
        "close_time": "2026-01-05T15:00:00Z",
    },
]


def test_builds_frame_with_category_join_and_binary_filter():
    frame = build_settled_market_frame(
        MARKETS,
        event_series={"KXTSAW-26JAN05": "KXTSAW"},
        series_categories={"KXTSAW": "Transportation"},
    )
    assert len(frame) == 1  # scalar row dropped
    row = frame.iloc[0]
    assert row["series_ticker"] == "KXTSAW"
    assert row["category"] == "Transportation"
    assert row["volume"] == 1234.0
    assert row["close_ts"] == 1767625200  # 2026-01-05T15:00:00Z


def test_series_ticker_falls_back_to_event_prefix():
    frame = build_settled_market_frame(MARKETS[:1], event_series={}, series_categories={})
    assert frame.iloc[0]["series_ticker"] == "KXTSAW"
    assert frame.iloc[0]["category"] == "unknown"


def test_resolve_series_ticker_uses_mapping_when_present():
    assert resolve_series_ticker({"KXTSAW-26JAN05": "KXTSAW"}, "KXTSAW-26JAN05") == "KXTSAW"


def test_resolve_series_ticker_falls_back_on_blank_mapping():
    assert resolve_series_ticker({"KXTSAW-26JAN05": ""}, "KXTSAW-26JAN05") == "KXTSAW"
