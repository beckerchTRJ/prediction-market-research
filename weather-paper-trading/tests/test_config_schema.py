"""W1: new live-MM config keys parse and default to prod-neutral (no-op) values.

These keys are scaffolding for upcoming market-making improvements (favorite-
longshot haircut, bucket-side selection, per-side edges, nowcast-floor fair,
resting-exit, obs-triggered/METAR-blackout requoting). None of them are wired
into strategy code yet in this change — this test only locks down that the
config file parses and that every new knob defaults to "off"/unchanged so
paper backtests and live dry-runs are unaffected until a human opts in.
"""
from kwt.config import load_config


def test_quote_overrides_new_keys_present_and_prod_neutral():
    qo = load_config().raw["live"]["quote_overrides"]

    # favorite-longshot haircut: off by default
    assert qo["flb_haircut_at_10c"] == 0.0
    assert qo["flb_haircut_zero_at"] == 0.40

    # bucket-selection by mid price. ask_only_below is ENABLED at 0.15 — the one
    # backtest-validated lever (docs/mm_sideselect_backtest.md): quote ask-only on
    # cheap YES longshots. bid_only_above stays off (no robust backtest signal).
    assert qo["ask_only_below"] == 0.15
    assert qo["bid_only_above"] is None
    assert qo["touch_min_edge_bid"] is None
    assert qo["touch_min_edge_ask"] == 0.0
    assert not qo["nowcast_floor_enabled"]


def test_new_high_cities_are_collect_only_and_firewalled():
    cfg = load_config()
    expected = {
        "HOU": "KXHIGHTHOU", "DAL": "KXHIGHTDAL", "OKC": "KXHIGHTOKC",
        "PHX": "KXHIGHTPHX", "LV": "KXHIGHTLV", "ATL": "KXHIGHTATL",
    }
    for city, series in expected.items():
        assert cfg.cities[city]["series"] == series
        assert cfg.cities[city]["live_enabled"] is False
        assert series.startswith("KXHIGH")
        assert cfg.cities[city]["station"].startswith("K")


def test_live_experiment_cohort_is_frozen_ask15_v2():
    live = load_config().raw["live"]
    exp = live["experiment"]
    assert {k: exp[k] for k in ("cohort_id", "primary_metric", "horizon_minutes")} == {
        "cohort_id": "ask15_v2", "primary_metric": "net_markout", "horizon_minutes": 60}
    assert exp["successor"] == {
        "cohort_id": "capacity16_v3", "primary_metric": "settlement_pnl",
        "horizon_minutes": 60, "activate_after_settled_dates": 10,
        "risk_overrides": {"max_capital_at_risk": 16.0, "max_net_inventory": 19}}

def test_live_resting_exit_and_requote_blocks_prod_neutral():
    live = load_config().raw["live"]

    resting_exit = live["resting_exit"]
    assert not resting_exit["enabled"]
    assert resting_exit["scratch_ticks"] == 2

    requote = live["requote"]
    assert not requote["obs_triggered"]
    assert requote["metar_blackout_seconds"] == 0

    # ENABLED as of 2026-07: a kill switch that cancels quotes but holds toxic
    # inventory to settlement is a loss-maximizer; the flatten ladder sheds it
    # in an orderly way instead. flatten_reduce_at/min_win_prob/deep_itm unchanged.
    assert live["flatten"]["enabled"]


def test_inventory_skew_bumped_for_faster_unwind():
    # 2026-07-19: raised 0.02 -> 0.04. With the book pinned at its inventory/
    # capital caps ~92% of cycles, faster passive unwind (a stronger lean on the
    # reducing side) frees quoting capacity inside the same dollar risk. The
    # config comment always earmarked 0.03-0.05 for exactly this.
    mm = load_config().strategies["market_making"]
    assert mm["inventory_skew"] == 0.04


def test_max_open_orders_raised_for_coverage():
    # Was 20 (the previously-binding, non-capital cap — 85% of blocks were this
    # reason). 56 = 7 cities x 4 two-sided markets/city, chosen for coverage, not
    # capital: capital exposure is independently bounded by max_capital_at_risk
    # (Task 1 fixed the ask-side reservation bug that let it breach 12.00 -> 14.14).
    live = load_config().raw["live"]
    assert live["risk"]["max_open_orders"] == 56
