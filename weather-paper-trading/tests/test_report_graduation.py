from kwt.db import init_db, connect
from kwt.report import build_report


def _seed(db):
    init_db(db)
    c = connect(db)
    c.execute("INSERT INTO strategies (name, enabled, bankroll0, cash, realized_pnl) "
              "VALUES ('fade', 1, 1000, 1010, 10)")
    # 6 target_dates, each a clean +$2/trade fade win -> significant, unconcentrated.
    for d in range(6):
        for i in range(3):
            tk = f"KXHIGHNY-26JUL0{d}-B{80+i}"
            c.execute("INSERT INTO markets (ticker, city, target_date, result) "
                      "VALUES (?,?,?, 'no')", (tk, "nyc", f"2026-07-0{d}"))
            c.execute("INSERT INTO trades (strategy, ticker, side, action, contracts, "
                      "price, fee, settled, pnl) VALUES ('fade',?, 'no','buy',1,0.9,0,1,2.0)", (tk,))
    c.commit(); c.close()


def test_report_emits_graduation_verdict(tmp_path):
    db = str(tmp_path / "kwt.db")
    _seed(db)
    rep = build_report(db_path=db, write_csv=False)
    g = rep["edge"]["fade"]["graduation"]
    assert g["n_blocks"] == 6
    assert g["pnl_p_value"] < 0.05
    assert isinstance(g["graduated"], bool)


def test_rare_loss_fade_holds_at_six_dates_and_reports_incomplete_events(tmp_path):
    db = str(tmp_path / "kwt.db"); init_db(db); c = connect(db)
    c.execute("INSERT INTO strategies (name,enabled,bankroll0,cash,realized_pnl) "
              "VALUES ('longshot_fade_dayb',1,1000,1004.5,4.5)")
    for d in range(7):
        date = f"2026-07-{d+1:02d}"; entry = f"{date}T00:00:00Z"
        tail = f"KXHIGHNY-{d}-TAIL"; rest = f"KXHIGHNY-{d}-REST"
        c.execute("INSERT INTO markets (ticker,city,target_date,low,high,result) "
                  "VALUES (?,?,?,?,?,'no')", (tail, "NY", date, 80, 81))
        c.execute("INSERT INTO markets (ticker,city,target_date,low,high,result) "
                  "VALUES (?,?,?,?,?,'yes')", (rest, "NY", date, 82, 83))
        c.execute("INSERT INTO trades (ts,strategy,ticker,side,action,contracts,price,fee,"
                  "settled,pnl) VALUES (?,?,?,?,?,?,?,?,1,?)",
                  (entry, "longshot_fade_dayb", tail, "no", "buy", 1, .925, 0, .075))
        # The seventh event deliberately lacks a complete entry-time partition.
        if d < 6:
            c.execute("INSERT INTO snapshots (ts,ticker,yes_bid,yes_ask) VALUES (?,?,?,?)",
                      (entry, tail, .07, .08))
            c.execute("INSERT INTO snapshots (ts,ticker,yes_bid,yes_ask) VALUES (?,?,?,?)",
                      (entry, rest, .92, .93))
    c.commit(); c.close()
    rep = build_report(db_path=db, write_csv=False)
    edge = rep["edge"]["longshot_fade_dayb"]
    assert edge["graduation"]["graduated"] is False
    assert edge["graduation"]["n_blocks"] == 6
    assert edge["fade_null"]["excluded_events"] == 1
    assert any("6 < 20" in reason for reason in edge["graduation"]["reasons"])
