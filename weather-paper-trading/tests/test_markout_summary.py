from kwt.db import connect
from kwt.markout import markout_summary


def _row(c, fid, tkr, direction, mo30):
    c.execute("INSERT INTO live_fill_markouts (fill_id, ticker, created_time, direction, "
              "mid_at_fill, mo_15, mo_30, mo_60, mo_settle, computed_ts) "
              "VALUES (?,?,?,?,?,?,?,?,?,?)",
              (fid, tkr, "2026-07-03T14:00:00Z", direction, 0.5, None, mo30, None, None, "t"))


def test_summary_aggregates_by_direction_and_city():
    c = connect(":memory:")
    _row(c, "a", "KXHIGHNY-26JUL03-B80", -1, -0.10)   # short YES, adverse
    _row(c, "b", "KXHIGHNY-26JUL03-B81", -1, -0.20)   # short YES, adverse
    _row(c, "d", "KXHIGHLAX-26JUL03-B70", 1, 0.04)    # long YES, benign
    c.commit()
    s = markout_summary(c)
    assert s["overall"]["n"] == 3
    assert abs(s["by_direction"]["short"]["mean_30"] - (-0.15)) < 1e-9
    assert abs(s["by_direction"]["long"]["mean_30"] - 0.04) < 1e-9
    assert abs(s["by_city"]["NY"]["mean_30"] - (-0.15)) < 1e-9
    assert s["by_hour"]["14"]["n"] == 3
