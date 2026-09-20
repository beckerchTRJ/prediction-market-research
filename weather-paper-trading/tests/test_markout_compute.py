from kwt.db import connect
from kwt.markout import nearest_mid, compute_markouts


def _snap(c, ticker, ts, yb, ya):
    c.execute("INSERT INTO snapshots (ts, ticker, yes_bid, yes_ask, last_price, "
              "volume, open_interest, liquidity) VALUES (?,?,?,?,?,0,0,0)",
              (ts, ticker, yb, ya, 0))


def test_nearest_mid_picks_closest_within_tolerance_else_none():
    c = connect(":memory:")
    _snap(c, "KXHIGHNY-26JUL03-B80", "2026-07-03T00:00:00Z", 0.40, 0.60)  # mid .50
    _snap(c, "KXHIGHNY-26JUL03-B80", "2026-07-03T00:20:00Z", 0.50, 0.70)  # mid .60
    # target 00:04 -> closest is 00:00 (4 min) within 10-min tol -> .50
    assert nearest_mid(c, "KXHIGHNY-26JUL03-B80", "2026-07-03T00:04:00Z", 600) == 0.50
    # target 00:40 -> nearest is 00:20 (20 min) > 10-min tol -> None
    assert nearest_mid(c, "KXHIGHNY-26JUL03-B80", "2026-07-03T00:40:00Z", 600) is None


def test_compute_markouts_fills_table_with_expected_values():
    c = connect(":memory:")
    tk = "KXHIGHNY-26JUL03-B80"
    # a long-YES fill at 00:00, mid .50; +30min mid .55 (favorable), settles NO (outcome 0).
    c.execute("INSERT INTO markets (ticker, city, target_date, result) VALUES (?,?,?, 'no')",
              (tk, "nyc", "2026-07-03"))
    c.execute("INSERT INTO live_fills (ts, fill_id, ticker, side, action, book_side, count, "
              "price, fee, is_taker, created_time) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
              ("t", "f1", tk, "yes", "buy", "bid", 1, 0.50, 0, 1, "2026-07-03T00:00:00Z"))
    _snap(c, tk, "2026-07-03T00:00:00Z", 0.40, 0.60)   # mid .50 at fill
    _snap(c, tk, "2026-07-03T00:30:00Z", 0.50, 0.60)   # mid .55 at +30
    n = compute_markouts(c, horizons=(15, 30, 60), tol_sec=600, ts="2026-07-03T10:00:00Z")
    assert n == 1
    r = c.execute("SELECT * FROM live_fill_markouts WHERE fill_id='f1'").fetchone()
    assert r["direction"] == 1                     # bid -> bought YES -> +1
    assert r["mid_at_fill"] == 0.50
    assert r["mo_15"] is None                    # no snapshot near +15 -> NULL
    assert abs(r["mo_30"] - 0.05) < 1e-9         # long YES, mid .50 -> .55 = +.05
    assert abs(r["mo_settle"] - (-0.50)) < 1e-9  # long YES, outcome 0 vs .50 = -.50


def test_compute_markouts_is_idempotent():
    c = connect(":memory:")
    tk = "KXHIGHNY-26JUL03-B80"
    c.execute("INSERT INTO markets (ticker, city, target_date) VALUES (?,?,?)",
              (tk, "nyc", "2026-07-03"))
    c.execute("INSERT INTO live_fills (ts, fill_id, ticker, side, action, book_side, count, "
              "price, fee, is_taker, created_time) VALUES ('t','f1',?, 'yes','buy','bid',1,"
              "0.5,0,1,'2026-07-03T00:00:00Z')", (tk,))
    _snap(c, tk, "2026-07-03T00:00:00Z", 0.40, 0.60)
    compute_markouts(c, ts="t1"); compute_markouts(c, ts="t2")
    assert c.execute("SELECT COUNT(*) n FROM live_fill_markouts").fetchone()["n"] == 1


def test_ask_fill_is_short_and_null_book_side_is_skipped():
    c = connect(":memory:")
    tk = "KXHIGHNY-26JUL03-B80"
    # ask fill: Kalshi encodes it (no, sell) but book_side='ask' -> short YES -> -1.
    c.execute("INSERT INTO live_fills (ts, fill_id, ticker, side, action, book_side, count, "
              "price, fee, is_taker, created_time) VALUES ('t','ask1',?, 'no','sell','ask',1,"
              "0.5,0,1,'2026-07-03T00:00:00Z')", (tk,))
    # a fill with no book_side can't be attributed -> skipped entirely.
    c.execute("INSERT INTO live_fills (ts, fill_id, ticker, side, action, count, price, "
              "fee, is_taker, created_time) VALUES ('t','nobk',?, 'yes','buy',1,0.5,0,1,"
              "'2026-07-03T00:00:00Z')", (tk,))
    _snap(c, tk, "2026-07-03T00:00:00Z", 0.40, 0.60)   # mid .50
    _snap(c, tk, "2026-07-03T00:30:00Z", 0.60, 0.60)   # mid .60 at +30 (YES rose)
    n = compute_markouts(c, horizons=(30,), ts="t")
    assert n == 1                                        # only the ask fill processed
    r = c.execute("SELECT * FROM live_fill_markouts WHERE fill_id='ask1'").fetchone()
    assert r["direction"] == -1
    # short YES, mid .50 -> .60: adverse -> negative markout.
    assert abs(r["mo_30"] - (-0.10)) < 1e-9
    assert c.execute("SELECT COUNT(*) n FROM live_fill_markouts "
                     "WHERE fill_id='nobk'").fetchone()["n"] == 0


def test_compute_markouts_ignores_non_kxhigh_fills():
    c = connect(":memory:")
    c.execute("INSERT INTO live_fills (ts, fill_id, ticker, side, action, count, price, "
              "fee, is_taker, created_time) VALUES ('t','e1','KXMAYORLA-26-SPRA','no','buy',"
              "1,0.5,0,1,'2026-07-03T00:00:00Z')")
    assert compute_markouts(c, ts="t") == 0
