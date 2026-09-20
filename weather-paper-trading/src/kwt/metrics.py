"""Performance metrics: Brier, calibration, Sharpe, and model-vs-market DM test.

The headline question — "do I have an edge?" — is answered by comparing the
strategy's probabilistic forecast (model_prob) against the market's implied
probability (market_prob) on the SAME resolved markets, via Brier score and a
Diebold-Mariano test on the loss differential. Lower Brier = better; a
significantly negative DM statistic means the model beats the market.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def brier(probs, outcomes) -> float:
    p = np.asarray(probs, float)
    y = np.asarray(outcomes, float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def brier_skill_score(model_brier: float, ref_brier: float) -> float:
    if ref_brier <= 0:
        return float("nan")
    return 1.0 - model_brier / ref_brier


def diebold_mariano(model_probs, market_probs, outcomes) -> dict:
    """DM test on Brier-loss differential d = L_model - L_market.

    Negative mean d => model has lower loss (beats market). Returns the loss
    differential mean, DM t-stat, and two-sided p-value.
    """
    p_m = np.asarray(model_probs, float)
    p_k = np.asarray(market_probs, float)
    y = np.asarray(outcomes, float)
    mask = ~(np.isnan(p_m) | np.isnan(p_k) | np.isnan(y))
    p_m, p_k, y = p_m[mask], p_k[mask], y[mask]
    n = p_m.size
    if n < 8:
        return {"n": int(n), "mean_diff": float("nan"), "dm_stat": float("nan"),
                "p_value": float("nan")}
    d = (p_m - y) ** 2 - (p_k - y) ** 2
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0:
        return {"n": n, "mean_diff": mean_d, "dm_stat": float("nan"), "p_value": float("nan")}
    dm = mean_d / (sd / np.sqrt(n))
    p_value = float(2 * stats.t.sf(abs(dm), df=n - 1))
    return {"n": int(n), "mean_diff": mean_d, "dm_stat": float(dm), "p_value": p_value}


def diebold_mariano_blocked(model_probs, market_probs, outcomes, blocks) -> dict:
    """Correlation-aware DM test: cluster the Brier-loss differential by block.

    Weather-bucket outcomes are NOT independent — the ~18 buckets in an event are
    a partition of one daily high, and cities on the same day share the synoptic
    pattern. Treating each bucket as an independent observation (plain
    `diebold_mariano`) overstates significance by ~the cluster size. Here we
    average the loss differential within each block (e.g. a city-day), then run
    the t-test ACROSS block means, so the effective sample size is the number of
    independent blocks, not the bucket count.

    Returns mean_diff (negative => model beats market), the across-block t-stat
    and p-value, plus n_blocks (effective N) and n_obs (raw bucket count).

    Caveats: city-day blocks still treat different cities on the same date as
    independent, though they share the synoptic pattern — so significance here is
    an upper bound on confidence, not the last word (day-only blocks would be more
    conservative but leave too few blocks to test early on). Also, mean_diff is the
    unweighted mean of block means, so it can diverge slightly from a bucket-level
    Brier difference when blocks vary in size; the verdict should come from this
    test, with the displayed Brier read as a descriptive summary.
    """
    p_m = np.asarray(model_probs, float)
    p_k = np.asarray(market_probs, float)
    y = np.asarray(outcomes, float)
    blk = np.asarray(blocks)
    mask = ~(np.isnan(p_m) | np.isnan(p_k) | np.isnan(y))
    p_m, p_k, y, blk = p_m[mask], p_k[mask], y[mask], blk[mask]
    n_obs = int(p_m.size)
    d = (p_m - y) ** 2 - (p_k - y) ** 2
    # mean loss differential per block (preserve first-seen block order)
    uniq = list(dict.fromkeys(blk.tolist()))
    block_means = np.array([d[blk == b].mean() for b in uniq], float)
    n_blocks = int(block_means.size)
    base = {"n_obs": n_obs, "n_blocks": n_blocks,
            "mean_diff": float(block_means.mean()) if n_blocks else float("nan")}
    if n_blocks < 2:
        # one (or zero) independent block: no cross-block variation to test
        return {**base, "dm_stat": float("nan"), "p_value": float("nan")}
    sd = float(block_means.std(ddof=1))
    if sd == 0:
        return {**base, "dm_stat": float("nan"), "p_value": float("nan")}
    dm = block_means.mean() / (sd / np.sqrt(n_blocks))
    p_value = float(2 * stats.t.sf(abs(dm), df=n_blocks - 1))
    return {**base, "dm_stat": float(dm), "p_value": p_value}


def calibration_table(probs, outcomes, bins=10) -> list[dict]:
    p = np.asarray(probs, float)
    y = np.asarray(outcomes, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                    "pred": round(float(p[m].mean()), 4),
                    "actual": round(float(y[m].mean()), 4)})
    return out


def pnl_concentration(pnls, k: int = 3) -> float:
    """Share of NET P&L contributed by the top-k trades. NaN for a non-positive
    book (concentrating a loss is meaningless). > ~0.4 means the record leans on a
    few lucky wins rather than a repeatable edge."""
    p = np.asarray(pnls, float)
    p = p[np.isfinite(p)]
    total = float(p.sum())
    if p.size == 0 or total <= 0:
        return float("nan")
    topk = float(np.sort(p)[::-1][:k].sum())
    return topk / total


def pnl_cluster_significance(pnls, blocks) -> dict:
    """One-sided test that mean per-trade P&L > 0, clustering by block (target_date)
    so correlated same-day trades don't inflate significance. Mirrors
    `diebold_mariano_blocked`: average P&L within each block, then t-test the block
    means against 0. p_value is one-sided (H1: edge > 0); small => real positive edge.
    """
    p = np.asarray(pnls, float)
    blk = np.asarray(blocks)
    mask = np.isfinite(p)
    p, blk = p[mask], blk[mask]
    uniq = list(dict.fromkeys(blk.tolist()))
    block_means = np.array([p[blk == b].mean() for b in uniq], float)
    n_blocks = int(block_means.size)
    base = {"n_blocks": n_blocks,
            "mean_per_trade": float(block_means.mean()) if n_blocks else float("nan")}
    if n_blocks < 2:
        return {**base, "t_stat": float("nan"), "p_value": float("nan")}
    sd = float(block_means.std(ddof=1))
    if sd == 0:
        mean = float(block_means.mean())
        # Zero cross-block variance with a non-zero mean is a deterministic
        # signal (every block agrees), not a degenerate one -- only an
        # all-zero book is truly uninformative.
        if mean == 0:
            return {**base, "t_stat": float("nan"), "p_value": float("nan")}
        return {**base, "t_stat": float("inf") if mean > 0 else float("-inf"),
                "p_value": 0.0 if mean > 0 else 1.0}
    t = block_means.mean() / (sd / np.sqrt(n_blocks))
    p_value = float(stats.t.sf(t, df=n_blocks - 1))     # one-sided upper tail
    return {**base, "t_stat": float(t), "p_value": p_value}


def categorical_portfolio_null_test(events: list[dict], *, simulations: int = 200_000,
                                    seed: int = 20260710) -> dict:
    """Market-implied null test for rare-loss, mutually-exclusive portfolios.

    Each event supplies ``target_date``, normalized ``probs`` and a same-length
    ``pnl_by_outcome`` vector for the portfolio actually traded in that city-day.
    The independent simulation draws each city separately; the comonotonic
    sensitivity uses one shared quantile for all cities on a target date. The
    larger upper-tail p-value binds, guarding against heat-dome dependence.
    """
    if not events:
        return {"n_events": 0, "n_blocks": 0, "observed_pnl": float("nan"),
                "expected_pnl": float("nan"), "p_independent": float("nan"),
                "p_comonotonic": float("nan"), "p_value": float("nan"),
                "simulations": simulations, "seed": seed}
    rng_ind = np.random.default_rng(seed)
    rng_com = np.random.default_rng(seed + 1)
    total_ind = np.zeros(simulations, float)
    total_com = np.zeros(simulations, float)
    dates = list(dict.fromkeys(str(e["target_date"]) for e in events))
    shared = {d: rng_com.random(simulations) for d in dates}
    observed = 0.0
    expected = 0.0
    losses = 0
    for event in events:
        probs = np.asarray(event["probs"], float)
        pnls = np.asarray(event["pnl_by_outcome"], float)
        if probs.size == 0 or probs.size != pnls.size or not np.all(np.isfinite(probs)) \
                or not np.all(np.isfinite(pnls)) or probs.sum() <= 0:
            raise ValueError("each event needs finite, aligned probabilities and P&Ls")
        probs = probs / probs.sum()
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        total_ind += pnls[np.searchsorted(cdf, rng_ind.random(simulations), side="right")]
        total_com += pnls[np.searchsorted(
            cdf, shared[str(event["target_date"])], side="right")]
        actual = float(event["actual_pnl"])
        observed += actual
        expected += float(np.dot(probs, pnls))
        losses += int(actual < 0)
    p_ind = float((1 + np.count_nonzero(total_ind >= observed)) / (simulations + 1))
    p_com = float((1 + np.count_nonzero(total_com >= observed)) / (simulations + 1))
    binding = total_com if p_com >= p_ind else total_ind
    counts, edges = np.histogram(binding, bins=30)
    return {"n_events": len(events), "n_blocks": len(dates),
            "observed_pnl": observed, "expected_pnl": expected,
            "observed_losing_events": losses, "p_independent": p_ind,
            "p_comonotonic": p_com, "p_value": max(p_ind, p_com),
            "simulations": simulations, "seed": seed,
            "binding_model": "comonotonic" if p_com >= p_ind else "independent",
            "null_histogram": {"edges": edges.tolist(), "counts": counts.tolist()}}


def graduation_verdict(*, pnl_sig: dict, concentration: float,
                       brier_skill: float | None, verdict_metric: str = "skill",
                       alpha: float = 0.05, max_concentration: float = 0.40,
                       min_blocks: int = 0) -> dict:
    """Promote/hold verdict for a strategy. Graduates (eligible for scaling / a live
    pilot) only when: P&L is significant (clustered, one-sided p < alpha), NOT
    top-3-concentrated, and — for forecast strategies (verdict_metric='skill') — the
    model is not worse than the market (brier_skill >= 0). Market-bias strategies
    (verdict_metric='pnl', e.g. the fade) are judged on P&L alone."""
    reasons: list[str] = []
    enough_blocks = int(pnl_sig.get("n_blocks") or 0) >= min_blocks
    if not enough_blocks:
        reasons.append(
            f"not enough independent dates ({pnl_sig.get('n_blocks', 0)} < {min_blocks})")
    p = pnl_sig.get("p_value")
    sig = enough_blocks and p is not None and np.isfinite(p) and p < alpha
    if enough_blocks and not sig:
        reasons.append(f"P&L not significant (clustered p={p})")
    over_conc = np.isfinite(concentration) and concentration > max_concentration
    if over_conc:
        reasons.append(f"P&L too concentrated (top-3={concentration:.2f} > {max_concentration:.2f})")
    skill_ok = True
    if verdict_metric == "skill":
        skill_ok = brier_skill is not None and np.isfinite(brier_skill) and brier_skill >= 0
        if not skill_ok:
            reasons.append(f"model worse than market (brier_skill={brier_skill})")
    return {"graduated": bool(sig and not over_conc and skill_ok), "reasons": reasons}


def sharpe_from_equity(ts_list, equity_list, periods_per_year: float = 252.0) -> float:
    eq = np.asarray(equity_list, float)
    if eq.size < 3:
        return float("nan")
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size < 2 or rets.std(ddof=1) == 0:
        return float("nan")
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year))
