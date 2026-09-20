# Initial Hypotheses

These are starting hypotheses, not conclusions.

## H1: Salience-Driven Mispricing

Highly salient national races are more likely to exhibit persistent market-anchor gaps because participant flows are influenced by partisan or media narratives more than by incremental public-information updates.

Test:

- Compare future correction after controlling for anchor gap, liquidity, and time to event.
- Slice by national salience proxies and media-attention features.

Failure condition:

- No stable out-of-sample increment over simpler gap-only models.

## H2: Thin-Market Persistence

Mispricing is more likely to persist in contracts with poor depth, wide spreads, or low open interest.

Test:

- Interact anchor gap with spread, volume, and open-interest features.
- Measure whether model value remains after conservative cost assumptions.

Failure condition:

- Gross predictive power disappears after transaction costs or reverses across cycles.

## H3: Anchor Confidence Matters

Anchor-market disagreement is more informative when the anchor itself is internally stable or narrow.

Test:

- Include anchor interval width, poll dispersion, or model uncertainty as interaction terms.

Failure condition:

- Uncertainty conditioning does not improve calibration or net-edge filtering.

## H4: Candidate Archetype Bias

Some candidate archetypes or contract types may be systematically mispriced due to participant composition and narrative framing.

Test:

- Encode candidate, office, and state context without allowing leakage from settlement data.
- Require persistence across cycles or enough pooled observations to justify the effect.

Failure condition:

- Effect is not stable out of sample or is dominated by one-off events.

## H5: Market Overreaction To New Information

Large one-day moves in market prices may overshoot public anchors and partially mean-revert over the next horizon.

Test:

- Model lagged changes and shock indicators jointly with anchor gaps.

Failure condition:

- Any apparent edge vanishes once spread, fees, and slippage are applied.
