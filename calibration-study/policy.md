# Trading Policy

## Mandate

The mandate is to trade political prediction markets only when there is evidence that market prices are misaligned with stronger public anchors and the expected correction remains positive after uncertainty, execution costs, portfolio constraints, and the opportunity cost of using capital here instead of in more traditional investments.

This is not an election handicapping project. It is a market-error project.

It is also a returns project. Research curiosity is a valid reason to study the space, but live risk-taking must clear a higher bar: the strategy should aim to outperform reasonable alternative uses of capital on a risk-adjusted basis.

## Research Standard

- Every hypothesis must be falsifiable before it is tradable.
- Every feature family must have an ex-ante rationale tied to an identifiable mechanism.
- Every performance claim must come from walk-forward validation.
- Every live trade must map to a logged signal or a separately logged arbitrage thesis.
- Every deployment decision must consider whether expected risk-adjusted return is competitive with passive or otherwise traditional alternatives.

## Hypothesis Discipline

Forecast-driven trades may use only hypotheses that have been written down and tested. Candidate examples:

- partisan or media-salience races exhibit wider and more persistent anchor-market gaps
- low-liquidity or wide-spread contracts show slower correction toward anchors
- public anchor disagreement is more informative when anchor uncertainty is low and market depth is weak
- certain contract types or candidate archetypes are systematically overpriced or underpriced

If a hypothesis fails in out-of-sample evaluation, it is demoted or removed.

## Data Policy

- Use time-indexed snapshots only.
- Preserve the timestamp at which each market and anchor observation was actually available.
- Never leak revised polls, revised models, or settlement information into historical features.
- Prefer explicit missingness over silent imputation in raw layers.
- Keep raw ingestion, panel construction, modeling, and trading outputs separate.

## Baseline Model Framing

The baseline tradable model is a residual-correction model:

- `p_anchor`: public-information anchor
- `p_mkt`: market-implied probability
- `delta_hat`: predicted future correction or mispricing signal
- `p_fair = inv_logit(logit(p_mkt) + delta_hat)`

The model is allowed to learn from:

- anchor-market gap features
- market quality and liquidity features
- race context and event timing
- information-environment features

The model is not allowed to rely on discretionary narrative overrides unless those overrides are explicitly logged as separate experimental interventions.

## Validation Policy

- Primary validation is walk-forward only.
- Model selection must be based on out-of-fold results.
- Calibration matters as much as directional accuracy.
- Parameter estimates used for live trading must come from pre-live historical windows.

Required checks for each model run:

- out-of-fold residual error
- probability calibration
- stability by cycle, state, office, and contract type
- sensitivity to transaction costs
- tail behavior and drawdown concentration
- comparison with a benchmark opportunity-cost assumption for capital

## Live Decision Rule

For a candidate trade, define:

- `p_model`: model-implied fair probability
- `p_blend = (1 - alpha) * p_mkt + alpha * p_model`
- `sigma`: empirical uncertainty estimate from historical out-of-fold errors
- `z`: conservatism multiplier

Conservative probabilities:

- `p_yes_cons = max(eps, min(1 - eps, p_blend - z * sigma))`
- `p_no_cons = max(eps, min(1 - eps, (1 - p_blend) - z * sigma))`

Execution prices:

- use displayed ask or a conservative proxy for entering `YES`
- use displayed `NO` entry price or a conservative proxy derived from the bid for entering `NO`

Net edge:

- `net_edge = conservative_probability - entry_price - fees - slippage - hurdle`

Trade only when:

- `net_edge > 0`
- model diagnostics for the relevant slice remain within approved bounds
- exposure caps are not breached

## Position Sizing

Base sizing uses fractional Kelly on conservative probabilities:

- `f_yes = kelly_fraction * max(0, (p_yes_cons - price_yes) / (1 - price_yes))`
- `f_no = kelly_fraction * max(0, (p_no_cons - price_no) / (1 - price_no))`

Hard caps apply at minimum to:

- per race
- per state
- per cycle
- total bankroll

The recommended position is the smaller of:

- fractional Kelly size
- remaining capacity under all applicable caps

## Trade Logging

Every forecast-driven trade must log:

- timestamp
- race and contract identifiers
- thesis identifier
- signal version / model run id
- displayed and expected execution price
- estimated fees and slippage
- bankroll at time of trade
- target size and actual fill size
- reason for deviation, if any

Every post-trade review must log:

- whether the original thesis was right or wrong
- whether the model signal was calibrated
- whether execution matched assumptions
- what should change in data, modeling, or discipline

## Separation Of Strategies

Guaranteed-return or structural trades are kept outside the forecast book. They must be logged separately and may not be used to justify predictive-model performance claims.

At minimum, maintain distinct labels for:

- forecast-driven
- structural / arbitrage
- discretionary experiment

## Change Control

Changes to:

- features
- targets
- trust parameter estimation
- uncertainty penalties
- execution cost assumptions
- exposure caps

must be documented before being used in live trading. If a change materially affects risk, prior backtests must be rerun.
