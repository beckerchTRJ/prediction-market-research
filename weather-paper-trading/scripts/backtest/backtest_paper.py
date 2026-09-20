"""Paper MM side-selection counterfactual (larger n). Fill RATES in paper are a
known fantasy (2% live realization), but SETTLEMENT is real, so the directional
P&L-by-band is a useful corroboration of the live signal. Each paper MM trade:
  side='yes' buy = bid fill (long YES);  side='no' buy = ask fill (short YES).
mid at fill = nearest prior snapshot yes-mid for that ticker.
"""
import sqlite3
DB = "data/kwt.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Pull settled MM buys with nearest-prior snapshot mid.
rows = conn.execute("""
  SELECT t.side, t.contracts AS ctr, t.pnl,
         (SELECT (s.yes_bid+s.yes_ask)/2.0 FROM snapshots s
           WHERE s.ticker=t.ticker AND s.yes_bid IS NOT NULL AND s.yes_ask IS NOT NULL
             AND s.ts <= t.ts ORDER BY s.ts DESC LIMIT 1) AS mid
  FROM trades t
  WHERE t.strategy='market_making' AND t.action='buy' AND t.settled=1 AND t.pnl IS NOT NULL
""").fetchall()
rows = [r for r in rows if r["mid"] is not None]
print(f"paper settled MM buys w/ mid: {len(rows)}")

def bookside(side): return "bid" if side == "yes" else "ask"
def band(m):
    if m < 0.25: return "<0.25"
    if m > 0.80: return ">0.80"
    return "0.25-0.80"

agg = {}
for r in rows:
    k = (bookside(r["side"]), band(r["mid"]))
    a = agg.setdefault(k, [0, 0.0])
    a[0] += 1; a[1] += r["pnl"]
print(f"\n{'side':>4} {'band':>10} {'n':>4} {'settledP&L$':>12}")
for k in sorted(agg):
    print(f"{k[0]:>4} {k[1]:>10} {agg[k][0]:>4} {agg[k][1]:>+12.2f}")

def kept(r, A, B):
    bs = bookside(r["side"])
    if A is not None and bs == "bid" and r["mid"] < A: return False
    if B is not None and bs == "ask" and r["mid"] > B: return False
    return True

base = sum(r["pnl"] for r in rows)
print(f"\nbaseline paper settled P&L: {base:+.2f}")
print(f"\n{'A':>6} {'settledP&L':>12} {'d_pnl':>9} {'nkept':>6}   (ask_only_below, B off)")
for A in [0.05,0.10,0.15,0.20,0.25,0.30,0.40]:
    kp = [r for r in rows if kept(r, A, None)]
    s = sum(r["pnl"] for r in kp)
    print(f"{A:>6.2f} {s:>+12.2f} {s-base:>+9.2f} {len(kp):>6}")
conn.close()
