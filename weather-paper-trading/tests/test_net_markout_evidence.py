from kwt.db import connect
from kwt.markout import net_markout_evidence


def test_net_markout_includes_spread_move_fee_and_isolates_cohort():
    c = connect(":memory:")
    for i, (side, price, fee, mo, cohort) in enumerate([
            ("ask", .55, .01, .02, "ask15_v1"),
            ("bid", .45, .00, -.03, "legacy")]):
        ticker = f"KXHIGHNY-26JUL0{i+1}-B80"
        c.execute("INSERT INTO markets (ticker,target_date,result) VALUES (?,?,?)",
                  (ticker, f"2026-07-0{i+1}", "no"))
        c.execute("INSERT INTO snapshots (ts,ticker,yes_bid,yes_ask) VALUES (?,?,?,?)",
                  (f"2026-07-0{i+1}T00:00:00Z", ticker, .49, .51))
        c.execute("INSERT INTO live_fills (ts,fill_id,ticker,book_side,count,price,fee,"
                  "is_taker,created_time,cohort_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  ("t", f"f{i}", ticker, side, 2, price, fee, 0,
                   f"2026-07-0{i+1}T00:00:00Z", cohort))
        direction = -1 if side == "ask" else 1
        c.execute("INSERT INTO live_fill_markouts (fill_id,ticker,created_time,direction,"
                  "mid_at_fill,mo_15,mo_30,mo_60) VALUES (?,?,?,?,?,?,?,?)",
                  (f"f{i}", ticker, f"2026-07-0{i+1}T00:00:00Z", direction,
                   .50, mo, mo, mo))
    c.commit()
    out = net_markout_evidence(c, "ask15_v1")
    # ask captured 5c vs mid, then gained 2c markout, less .5c fee/contract.
    assert abs(out["mean_net_60"] - .065) < 1e-9
    assert out["n_fills"] == 1 and out["n_dates"] == 1
    assert out["status"] == "collecting"


def test_zero_contract_rows_do_not_count_as_fills_or_evidence():
    c = connect(":memory:")
    ticker = "KXHIGHNY-26JUL01-B80"
    c.execute("INSERT INTO markets (ticker,target_date,result) VALUES (?,?,?)", (ticker, "2026-07-01", "yes"))
    c.execute("INSERT INTO snapshots (ts,ticker,yes_bid,yes_ask) VALUES (?,?,?,?)",
              ("2026-07-01T00:00:00Z", ticker, .49, .51))
    for fill_id, count in (("real", 1), ("zero", 0)):
        c.execute("INSERT INTO live_fills (ts,fill_id,ticker,book_side,count,price,fee,is_taker,created_time,cohort_id) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?)",
                  ("t", fill_id, ticker, "bid", count, .50, 0, 0, "2026-07-01T00:00:00Z", "ask15_v2"))
        c.execute("INSERT INTO live_fill_markouts (fill_id,ticker,created_time,direction,mid_at_fill,mo_60,mo_settle) "
                  "VALUES (?,?,?,?,?,?,?)", (fill_id, ticker, "2026-07-01T00:00:00Z", 1, .50, .01, .01))
    c.commit()
    out = net_markout_evidence(c, "ask15_v2")
    assert out["n_fills"] == 1
    assert out["n_dates"] == 1
