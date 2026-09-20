# Can a Model Beat a Prediction Market?

[Kalshi](https://kalshi.com) is a regulated exchange where people trade on real-world outcomes, such as "Will New York's high temperature tomorrow be 83–84°F?" A contract that pays $1 if the answer is yes and trades at 30 cents implies a 30% chance.

This project asked a simple question: **are those prices wrong often enough, and by enough, that a disciplined strategy could make money after trading fees?**

**The answer was no.** I tested six strategies over three months and ran a separate study of 44,135 historical prices. Nothing cleared the bar I had set in advance, so I shut the project down. This repository is the research system and the evidence behind that decision.

## Why a "no" is the useful result

The easiest mistake in this kind of work is to keep adjusting a strategy until the past results look good, then lose money when it meets the future. To guard against that, I wrote down the pass/fail rules **before** collecting results, and committed to stopping if they were not met.

A strategy could graduate to real money only if **all** of these held:

- **Its profit was statistically distinguishable from luck**, treating each day as one observation. Bets placed on the same day tend to win or lose together (one heat wave moves every city's market), so counting each bet separately would overstate the evidence.
- **The profit was not concentrated in a few wins.** If the top three trades made up more than 40% of the profit, it counted as luck, not a repeatable edge.
- **Its forecasts were at least as accurate as the market's own prices.**
- **It had at least 20 settled days** of results behind it.

No strategy passed.

## Part 1: three months of simulated trading on weather markets

`weather-paper-trading/`

Every three hours, from June 10 to September 11, 2026, the system:

1. Recorded the price of every open daily-high-temperature market.
2. Pulled fresh weather forecasts (about 80 runs from four forecasting models), 15 years of weather-station history, and the temperatures observed so far that day.
3. Turned each forecast into a probability for each temperature range.
4. Let each strategy place simulated bets from its own $1,000 play-money account, with fees modeled from Kalshi's published formula.
5. When a market settled, recorded the win or loss, and scored the model's probabilities against the market's.

In total it placed 14,022 simulated trades across 4,884 markets.

| Strategy | The idea | What happened |
|---|---|---|
| Historical averages | Bet against prices that stray far from what is normal for that date | Lost its full $1,000 |
| Forecast disagreement | Bet where weather models disagree with the price (two variants) | Both lost their full $1,000 |
| Same-day observations | Bet late in the day using temperatures already recorded | Lost its full $1,000. Its forecasts were far less accurate than the market's |
| Bet against long shots | Sell cheap, unlikely outcomes, which research says are often overpriced (two variants) | Roughly break-even: one variant down about $6, the other up about $11 |
| Buy long shots | The opposite bet, as a check | Lost money; idea rejected |
| Market making | Post both a buy and a sell price and earn the gap between them | Profitable in simulation, but see below |

**The market-making result did not survive contact with reality.** It was the only strategy that looked profitable on paper, so I tested it with a small real-money pilot ($20). Real orders were filled only about 5% as often as the simulation assumed, and the fills that did happen were disproportionately the ones that went on to lose. After four days and 58 settled fills the pilot was break-even (+$0.01). The simulated profit came from an unrealistic assumption about getting orders filled, not from a real advantage. I stopped the pilot on July 26.

The plain takeaway: on these markets, prices already reflect the public weather forecasts. A model built from the same forecasts has nothing to add.

## Part 2: are prices accurate across all of Kalshi?

`calibration-study/`

If weather markets are efficient, maybe other categories are not. This study pulled 113,697 settled markets and 44,135 price observations across Kalshi's categories and checked **calibration**: do things priced at 20% actually happen about 20% of the time?

Prices were grouped by category, price level, and time remaining before settlement, giving 26 groups with enough data. A group counted as a real opportunity only if the mispricing was large enough to survive fees **and** showed up in the same direction in both the first and second half of the history.

**None of the 26 groups qualified.** The rule I had written in advance said that outcome means stop, so I did. The full result is in [`calibration-study/artifacts/calibration_report.md`](calibration-study/artifacts/calibration_report.md).

## What this project demonstrates

- **Experimental discipline:** success and failure criteria fixed before seeing results, and honored when the answer was disappointing.
- **Statistical care:** accounting for correlated bets, checking that results are not driven by a few outliers, and measuring forecast accuracy against a strong benchmark (the market itself).
- **Skepticism of simulations:** validating a too-good paper result with a small real test before trusting it.
- **Data engineering:** a scheduled pipeline that ran unattended for three months, collecting about 267,000 price snapshots and scoring about 206,000 model signals, with automated tests for both codebases.

## Repository layout

| Path | Contents |
|---|---|
| `weather-paper-trading/src/kwt/` | Data collection, forecast models, strategies, simulated and live order handling, settlement, reporting |
| `weather-paper-trading/tests/` | Automated tests |
| `weather-paper-trading/artifacts/` | Per-strategy performance summary and account-balance history (through mid-July) |
| `weather-paper-trading/docs/` | Analysis of why the market-making pilot broke even |
| `calibration-study/src/`, `scripts/`, `sql/` | Market history download, local database, and the calibration analysis |
| `calibration-study/artifacts/` | The calibration report and the full table of results by group |
| `calibration-study/docs/` | The hypotheses and the research process |

Databases, logs, and raw price history are not included because of their size.

## Run it

Both parts are Python 3.12 packages. Simulated trading needs no account or API key.

```bash
cd weather-paper-trading
pip install -e .
python -m kwt initdb
python -m kwt collect     # record prices and place simulated trades
python -m kwt settle      # score settled markets
python -m kwt report
```

```bash
cd calibration-study
pip install -e .
python3 scripts/init_db.py --db kalshi_cache.db --schema sql/kalshi_cache.sql
python3 scripts/kalshi_pull_settled.py --db kalshi_cache.db
python3 scripts/kalshi_pull_snapshots.py --db kalshi_cache.db --per-category-cap 3000
python3 scripts/run_calibration.py --db kalshi_cache.db
```

This is a research project, not financial advice.

**Tools:** Python, pandas, NumPy, SciPy, scikit-learn, SQLite, pytest
