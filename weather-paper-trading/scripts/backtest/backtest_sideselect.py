"""Counterfactual backtest of the side-selection policy (ask_only_below /
bid_only_above) against data we already have.

We CANNOT exactly replay fills (raw market prints aren't persisted), but the
side-selection lever is a deterministic policy filter on (book_side, mid):
  * ask_only_below = A  -> below mid A we quote ASK only  -> DROP bid fills, mid < A
  * bid_only_above = B  -> above mid B we quote BID only  -> DROP ask fills, mid > B
Dropping filtered fills and re-summing realized P&L is an exact, CONSERVATIVE
lower bound on the policy's benefit (we don't model the extra ask-side fills a
better queue position would earn).

Two independent P&L proxies, both reported:
  (1) settled realized P&L per contract (mm_fill_pnl.pnl_per_contract) — what
      actually hits the account, but very noisy at this n.
  (2) settlement markout (live_fill_markouts.mo_settle) — signed vs mid at fill,
      less noisy, the basis of Fable's original diagnosis.
"""
import sqlite3

DB = "data/kwt.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row


def load_live_fills():
    """Real-money maker fills with mid, settled pnl, and settlement markout."""
    rows = conn.execute("""
        SELECT p.book_side, p.count AS ctr, p.mid_at_fill AS mid,
               p.pnl_per_contract AS pnl_ct, p.effective_spread AS eff,
               m.mo_settle
        FROM mm_fill_pnl p
        LEFT JOIN live_fill_markouts m ON m.fill_id = p.id
        WHERE p.is_taker = 0 AND p.mid_at_fill IS NOT NULL
    """).fetchall()
    return rows


def kept(f, A, B):
    """True if the fill survives policy (ask_only_below=A, bid_only_above=B)."""
    if A is not None and f["book_side"] == "bid" and f["mid"] < A:
        return False
    if B is not None and f["book_side"] == "ask" and f["mid"] > B:
        return False
    return True


def pnl_sum(fills, A, B):
    """Settled realized P&L ($) over kept, settled fills; also n kept."""
    tot = 0.0
    n = 0
    for f in fills:
        if f["pnl_ct"] is None:
            continue
        if kept(f, A, B):
            tot += f["pnl_ct"] * f["ctr"]
            n += 1
    return tot, n


def markout_sum(fills, A, B):
    """Sum of signed settlement markout ($/ctr * ctr) over kept fills w/ mo_settle."""
    tot = 0.0
    n = 0
    for f in fills:
        if f["mo_settle"] is None:
            continue
        if kept(f, A, B):
            tot += f["mo_settle"] * f["ctr"]
            n += 1
    return tot, n


fills = load_live_fills()
print(f"LIVE real-money maker fills w/ mid: {len(fills)}")

# --- baseline per-band diagnostic -----------------------------------------
print("\n=== LIVE per (book_side, mid-band): settled P&L and settle-markout ===")
print(f"{'side':>4} {'band':>10} {'n':>3} {'ctr':>4} {'settledP&L':>11} {'mo_settle$':>11}")
def band(mid):
    if mid < 0.25: return "<0.25"
    if mid > 0.80: return ">0.80"
    return "0.25-0.80"
agg = {}
for f in fills:
    k = (f["book_side"], band(f["mid"]))
    a = agg.setdefault(k, [0, 0.0, 0.0, 0.0])
    a[0] += 1
    a[1] += f["ctr"]
    if f["pnl_ct"] is not None: a[2] += f["pnl_ct"] * f["ctr"]
    if f["mo_settle"] is not None: a[3] += f["mo_settle"] * f["ctr"]
for k in sorted(agg):
    n, ctr, pnl, mo = agg[k]
    print(f"{k[0]:>4} {k[1]:>10} {n:>3} {ctr:>4.0f} {pnl:>+11.3f} {mo:>+11.3f}")

base_pnl, base_n = pnl_sum(fills, None, None)
base_mo, base_mon = markout_sum(fills, None, None)
print(f"\nBASELINE (no policy): settled P&L {base_pnl:+.3f} [n={base_n}], "
      f"settle-markout {base_mo:+.3f} [n={base_mon}]")

# --- sweep ask_only_below (bid_only_above off) ----------------------------
print("\n=== Sweep ask_only_below A  (bid_only_above off) ===")
print(f"{'A':>6} {'settledP&L':>11} {'d_pnl':>8} {'nkept':>6} | {'markout$':>9} {'d_mo':>8} {'nkept':>6}")
for A in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    p, np_ = pnl_sum(fills, A, None)
    m, nm = markout_sum(fills, A, None)
    print(f"{A:>6.2f} {p:>+11.3f} {p-base_pnl:>+8.3f} {np_:>6} | "
          f"{m:>+9.3f} {m-base_mo:>+8.3f} {nm:>6}")

# --- sweep bid_only_above (ask_only_below off) ----------------------------
print("\n=== Sweep bid_only_above B  (ask_only_below off) ===")
print(f"{'B':>6} {'settledP&L':>11} {'d_pnl':>8} {'nkept':>6} | {'markout$':>9} {'d_mo':>8} {'nkept':>6}")
for B in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60]:
    p, np_ = pnl_sum(fills, None, B)
    m, nm = markout_sum(fills, None, B)
    print(f"{B:>6.2f} {p:>+11.3f} {p-base_pnl:>+8.3f} {np_:>6} | "
          f"{m:>+9.3f} {m-base_mo:>+8.3f} {nm:>6}")

# --- Fable's specific recommendation --------------------------------------
p, npk = pnl_sum(fills, 0.25, 0.80)
m, nmk = markout_sum(fills, 0.25, 0.80)
print(f"\n=== Fable rec (A=0.25, B=0.80) ===")
print(f"settled P&L {p:+.3f} (base {base_pnl:+.3f}, delta {p-base_pnl:+.3f}) n={npk}")
print(f"settle-markout {m:+.3f} (base {base_mo:+.3f}, delta {m-base_mo:+.3f}) n={nmk}")

conn.close()
