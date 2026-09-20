import math
from kwt.metrics import pnl_concentration


def test_concentration_flags_a_few_big_winners():
    # +100 total, top-3 (34+33+33) = 100 -> fully concentrated.
    pnls = [34, 33, 33] + [1, -1, 1, -1, 1, -1]
    assert abs(pnl_concentration(pnls) - (100 / sum(pnls))) < 1e-9


def test_concentration_is_nan_for_a_losing_book():
    assert math.isnan(pnl_concentration([-5, 1, 2, -3]))


def test_concentration_low_when_spread_out():
    pnls = [2] * 50            # broad, even P&L
    assert pnl_concentration(pnls) == 6 / 100    # top-3 of 100 total


# append to tests/test_graduation_metrics.py
import math
from kwt.metrics import pnl_cluster_significance


def test_pnl_significant_when_consistently_positive_across_days():
    # 6 days, each cleanly positive -> significant edge.
    pnls, blocks = [], []
    for d in range(6):
        pnls += [3, 4, 2]; blocks += [f"d{d}"] * 3
    r = pnl_cluster_significance(pnls, blocks)
    assert r["n_blocks"] == 6
    assert r["mean_per_trade"] == 3.0
    assert r["p_value"] < 0.05          # one-sided: clearly > 0


def test_pnl_not_significant_when_noisy():
    # alternating winning/losing days -> no reliable edge.
    pnls, blocks = [], []
    for d in range(6):
        v = 10 if d % 2 == 0 else -10
        pnls += [v, v]; blocks += [f"d{d}"] * 2
    r = pnl_cluster_significance(pnls, blocks)
    assert r["p_value"] > 0.10


def test_pnl_significance_needs_two_blocks():
    r = pnl_cluster_significance([1, 2, 3], ["d0", "d0", "d0"])
    assert r["n_blocks"] == 1 and math.isnan(r["p_value"])


# append to tests/test_graduation_metrics.py
from kwt.metrics import graduation_verdict, categorical_portfolio_null_test


def _sig(p):  # helper: a pnl_sig dict with a given p_value
    return {"n_blocks": 10, "mean_per_trade": 1.0, "t_stat": 3.0, "p_value": p}


def test_forecast_strategy_needs_skill_significance_and_low_concentration():
    v = graduation_verdict(pnl_sig=_sig(0.01), concentration=0.2, brier_skill=0.05)
    assert v["graduated"] is True and v["reasons"] == []


def test_no_skill_fails_a_forecast_strategy_even_if_profitable():
    # ensemble_divergence_lh case: significant P&L but model worse than market.
    v = graduation_verdict(pnl_sig=_sig(0.01), concentration=0.48, brier_skill=-0.06)
    assert v["graduated"] is False
    assert any("worse than market" in r for r in v["reasons"])
    assert any("concentrated" in r for r in v["reasons"])


def test_market_bias_strategy_judged_on_pnl_only():
    # the fade: skill is N/A; graduate on significant, unconcentrated P&L.
    v = graduation_verdict(pnl_sig=_sig(0.02), concentration=0.15,
                           brier_skill=None, verdict_metric="pnl")
    assert v["graduated"] is True


def test_insignificant_pnl_never_graduates():
    v = graduation_verdict(pnl_sig=_sig(0.30), concentration=0.1,
                           brier_skill=0.1, verdict_metric="pnl")
    assert v["graduated"] is False
    assert any("not significant" in r for r in v["reasons"])


def test_categorical_null_is_deterministic_and_conservative_dependence_binds():
    events = []
    for day in range(6):
        for city in ("a", "b", "c"):
            events.append({"target_date": f"d{day}", "event": f"{city}{day}",
                           "probs": [0.075, 0.925],
                           "pnl_by_outcome": [-9.25, 0.75], "actual_pnl": 0.75})
    a = categorical_portfolio_null_test(events, simulations=20_000)
    b = categorical_portfolio_null_test(events, simulations=20_000)
    assert a == b
    assert a["n_blocks"] == 6
    assert a["p_value"] == max(a["p_independent"], a["p_comonotonic"])
    assert a["p_value"] > 0.05


def test_fade_graduation_requires_twenty_dates():
    v = graduation_verdict(
        pnl_sig={"p_value": 0.0001, "n_blocks": 6}, concentration=0.2,
        brier_skill=None, verdict_metric="pnl", min_blocks=20)
    assert v["graduated"] is False
    assert "6 < 20" in " ".join(v["reasons"])
