import pandas as pd

from kalshi_fund.calibration_report import render_report


def _edge_map():
    base = {
        "bucket": "0.05-0.10", "horizon_days": 7, "n_obs": 200, "win_rate": 0.01,
        "wilson_low": 0.0, "wilson_high": 0.03,
        "ev_yes": -0.07, "ev_yes_low": -0.09, "ev_yes_high": -0.05,
        "ev_no": 0.04, "ev_no_low": 0.02, "ev_no_high": 0.06,
    }
    return pd.DataFrame([
        {**base, "category": "Biased", "half": "early", "graduated_side": "no"},
        {**base, "category": "Biased", "half": "late", "graduated_side": "no"},
        {**base, "category": "Fair", "half": "early", "graduated_side": "",
         "ev_no": 0.0, "ev_no_low": -0.02, "ev_no_high": 0.02},
        {**base, "category": "Fair", "half": "late", "graduated_side": "",
         "ev_no": 0.0, "ev_no_low": -0.02, "ev_no_high": 0.02},
    ])


def test_report_lists_graduated_cells_and_counts():
    report = render_report(_edge_map(), n_markets=1000, n_observations=3000)
    assert "Biased" in report
    assert "1,000 markets" in report
    assert "## Graduated cells" in report
    # non-graduated categories appear only in the full-map summary, not the graduated table
    graduated_section = report.split("## Graduated cells")[1].split("##")[0]
    assert "Fair" not in graduated_section
