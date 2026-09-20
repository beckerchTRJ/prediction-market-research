# MM side-selection backtest (2026-07-06)

Backtest of the live-maker side-selection levers (`ask_only_below` /
`bid_only_above`, added under `live.quote_overrides`) against data already in
`data/kwt.db`. Motivation: the real-money maker was running ~breakeven (+$0.01
over 4 days / 58 settled fills) — capturing ~+2.7c/ctr of spread at fill and
giving it all back by settlement (adverse selection).

## Method & scope

Raw market trade prints are **not persisted** (the sim fetches them live each
cycle and discards them), so an exact fill-by-fill replay is not possible
offline. But side-selection is a **deterministic policy filter** on
`(book_side, mid)`:

- `ask_only_below = A` → below mid `A` we quote ASK only → drop **bid** fills with `mid < A`
- `bid_only_above = B` → above mid `B` we quote BID only → drop **ask** fills with `mid > B`

So we re-score real fills + their settlements exactly, dropping the filtered
fills and re-summing realized P&L. This is a **conservative lower bound** on the
policy benefit — we don't model the extra ask-side fills a better queue position
would earn. It tests ONLY side-selection; the FLB haircut and per-side touch
edges shift quote *prices* and need the raw prints, so they remain **untested
offline** (enable behind live A/B only).

- `scripts/backtest/backtest_sideselect.py` — live real-money fills (`mm_fill_pnl`), n=58 settled
- `scripts/backtest/backtest_paper.py` — paper MM fills (`trades`), n=1,200 settled (larger n; fill *rates* are a known fantasy but settlements are real, so the by-band direction is informative)

## Results

**Live (n=58 settled) — too thin to trust.** Loss concentrated in mid-band
(0.25–0.80) long (bid) fills (−$1.72); cheap-longshot longs happened to win
(+$1.23). Applying Fable's proposed `ask_only_below=0.25` here would have taken
P&L from +$0.01 → −$1.22. A handful of $1 bucket resolutions dominate at this n
— not decision-grade.

**Paper (n=1,200 settled) — corrects the picture.** The favorite-longshot
structure is clean at scale: selling cheap YES longshots = +$862; buying cheap
YES longshots = −$175. Dropping the cheap-*long* (bid) fills improves settled
P&L, with the optimum **well below** the originally suggested 0.25:

| `ask_only_below` | paper Δ settled P&L |
|---|---|
| 0.10 | +$332 |
| **0.15** | **+$376 (peak)** |
| 0.25 | +$175 |
| 0.40 | +$101 |

`bid_only_above` showed no robust signal.

## Verdict

- **Direction validated** (large-n paper): quote ask-only on cheap buckets — stop taking the long side of cheap YES longshots.
- **Threshold corrected:** use `ask_only_below ≈ 0.12` (near the paper optimum, conservative), **not** 0.25.
- **Live can't tune this yet:** 58 settled fills in 4 days because ~86% of orders block. The binding constraint is fill *volume*, not threshold choice — get more live fills at tiny size before trusting the haircut/edge levers.
- Leave `bid_only_above`, FLB haircut, per-side edges, resting exits, and flatten **off** until there's live evidence.

## Reproduce

```
source .venv/bin/activate
python scripts/backtest/backtest_sideselect.py   # live real-money counterfactual
python scripts/backtest/backtest_paper.py         # paper corroboration (larger n)
```
