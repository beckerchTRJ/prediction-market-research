"""mm_fill_pnl (consumed by `kwt live-status`) computed effective_spread and
pnl_per_contract from (side, action). Kalshi encodes a maker ASK fill (sold YES,
short) as (side=no, action=sell), so the view read it as a long-NO position and
FLIPPED the sign on every ask fill — reporting the maker earning spread when it was
being picked off. The view must derive direction from book_side (bid/ask).
"""
from __future__ import annotations

from kwt.db import connect


def _seed_fill(c, fill_id, book_side, yes_price, result):
    tk = f"KXHIGHNY-26JUL03-{fill_id}"   # own ticker/market per fill (own result)
    c.execute("INSERT INTO markets (ticker, city, target_date, result) VALUES (?,?,?,?)",
              (tk, "nyc", "2026-07-03", result))
    c.execute("INSERT INTO live_fills (ts, fill_id, ticker, side, action, book_side, count, "
              "price, fee, is_taker, created_time) VALUES ('t',?,?,?,?,?,1,?,0,1,"
              "'2026-07-03T00:00:00Z')",
              (fill_id, tk, "yes" if book_side == "bid" else "no",
               "buy" if book_side == "bid" else "sell", book_side, yes_price))
    # mid_at_fill = 0.50 (snapshot at/just-before the fill).
    c.execute("INSERT INTO snapshots (ts, ticker, yes_bid, yes_ask, last_price, "
              "volume, open_interest, liquidity) VALUES ('2026-07-03T00:00:00Z',?,0.40,0.60,0,0,0,0)",
              (tk,))


def test_view_signs_bid_and_ask_fills_correctly():
    c = connect(":memory:")
    # BID: bought YES at 0.40, mid 0.50 -> bought 10c below mid (good); settles YES -> +0.60.
    _seed_fill(c, "bid1", "bid", 0.40, "yes")
    # ASK: sold YES at 0.60, mid 0.50 -> sold 10c above mid (good); settles NO -> +0.60.
    _seed_fill(c, "ask1", "ask", 0.60, "no")
    c.commit()
    rows = {r["fill_id"]: r for r in c.execute(
        "SELECT lf.fill_id, v.effective_spread, v.pnl_per_contract "
        "FROM mm_fill_pnl v JOIN live_fills lf ON lf.id=v.id").fetchall()}
    # bid: eff spread = mid - price = 0.10; pnl = 1 - 0.40 = 0.60
    assert abs(rows["bid1"]["effective_spread"] - 0.10) < 1e-9
    assert abs(rows["bid1"]["pnl_per_contract"] - 0.60) < 1e-9
    # ask: eff spread = price - mid = 0.10 (was -0.10 under the bug); pnl = 0.60 (was -0.60)
    assert abs(rows["ask1"]["effective_spread"] - 0.10) < 1e-9
    assert abs(rows["ask1"]["pnl_per_contract"] - 0.60) < 1e-9
