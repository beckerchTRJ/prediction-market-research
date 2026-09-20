# Research Loop

## 1. Ingest And Freeze Data

- Ingest market snapshots and anchor snapshots with original timestamps.
- Freeze raw inputs before feature engineering.
- Track source metadata so later revisions are auditable.

## 2. Build The Panel

- Join anchors to markets using time-aware matching.
- Compute anchor gaps, liquidity proxies, timing variables, and context features.
- Define a future-correction target at a fixed horizon.

## 3. Run Walk-Forward Training

- Fit the residual model only on past data.
- Export out-of-fold predictions for every validation window.
- Compare model probabilities, market probabilities, and anchor-only baselines.

## 4. Estimate Trading Parameters

Estimate `alpha`, `sigma`, slippage, and hurdle only from historical windows available before live deployment.

### `alpha`

Estimate a shrinkage weight that blends the market with the model:

- objective: minimize out-of-fold error of the blended probability relative to the chosen truth proxy
- implementation: clipped least-squares shrinkage between `p_mkt` and `p_model`

Interpretation:

- low `alpha`: model adds little beyond market price
- high `alpha`: model information survives walk-forward evaluation

### `sigma`

Estimate uncertainty from out-of-fold forecast errors:

- baseline: global RMSE of blended probability against the truth proxy
- stricter option: use upper-quantile absolute error by market-quality bucket

Interpretation:

- higher `sigma` raises the conservatism penalty and reduces trading frequency

### Slippage

Estimate from realized fills versus displayed quotes:

- `realized_slippage = signed_fill_price - signed_displayed_entry_price`
- keep venue-specific and contract-liquidity-specific estimates
- use a pessimistic percentile, not the mean, for live deployment

### Hurdle

Estimate the minimum extra edge required to justify capital usage:

- include operational friction, model uncertainty not captured by `sigma`, and opportunity cost
- benchmark opportunity cost against traditional alternatives such as cash yield, bonds, or broad equity exposure, depending on the capital pool being used
- set higher hurdles in thin, slow, or operationally costly segments

## 5. Generate Conservative Signals

- blend `p_mkt` and `p_model` using `alpha`
- subtract `z * sigma` from the relevant side probability
- subtract fees, slippage, and hurdle
- trade only if conservative net edge is positive

## 6. Size Conservatively

- use fractional Kelly on conservative probabilities
- cap exposure by race, state, cycle, and total bankroll
- require manual review for unusually concentrated or correlated positions

## 7. Review And Recalibrate

- compare expected edge to realized edge
- compare expected slippage to realized slippage
- compare signal confidence to realized calibration
- update parameters only on a scheduled cadence, not trade by trade
