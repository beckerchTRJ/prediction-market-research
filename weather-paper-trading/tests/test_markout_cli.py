import types
from kwt.db import connect
from kwt.__main__ import cmd_markouts


def test_cmd_markouts_populates_table(tmp_path, capsys):
    db = str(tmp_path / "kwt.db")
    c = connect(db)
    tk = "KXHIGHNY-26JUL03-B80"
    c.execute("INSERT INTO markets (ticker, city, target_date, result) VALUES (?,?,?, 'no')",
              (tk, "nyc", "2026-07-03"))
    c.execute("INSERT INTO live_fills (ts, fill_id, ticker, side, action, book_side, count, "
              "price, fee, is_taker, created_time) VALUES ('t','f1',?, 'yes','buy','bid',1,"
              "0.5,0,1,'2026-07-03T00:00:00Z')", (tk,))
    c.execute("INSERT INTO snapshots (ts, ticker, yes_bid, yes_ask, last_price, volume, "
              "open_interest, liquidity) VALUES ('2026-07-03T00:00:00Z',?,0.4,0.6,0,0,0,0)", (tk,))
    c.commit(); c.close()

    cmd_markouts(types.SimpleNamespace(db=db, tol_min=10))
    out = capsys.readouterr().out
    assert "markout" in out.lower()
    assert connect(db).execute(
        "SELECT COUNT(*) n FROM live_fill_markouts").fetchone()["n"] == 1
