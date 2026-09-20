import pytest

from kwt.db import connect, ensure_experiment_cohort


def test_cohort_registration_is_idempotent_and_immutable():
    conn = connect(":memory:")
    first = ensure_experiment_cohort(
        conn, "ask15_v1", {"ask_only_below": .15},
        primary_metric="net_markout", horizon_minutes=60,
        activated_ts="2026-07-10T00:00:00Z")
    again = ensure_experiment_cohort(
        conn, "ask15_v1", {"ask_only_below": .15},
        primary_metric="net_markout", horizon_minutes=60,
        activated_ts="2026-07-11T00:00:00Z")
    assert first["config_sha256"] == again["config_sha256"]
    assert again["activated_ts"] == "2026-07-10T00:00:00Z"
    with pytest.raises(ValueError, match="different config"):
        ensure_experiment_cohort(
            conn, "ask15_v1", {"ask_only_below": .12},
            primary_metric="net_markout", horizon_minutes=60)


def test_live_tables_have_legacy_cohort_defaults():
    conn = connect(":memory:")
    for table in ("live_orders", "live_fills", "live_risk_events",
                  "mm_fill_audit", "mm_quote_log"):
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        assert "cohort_id" in cols

