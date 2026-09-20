"""Market-making (spread capture with inventory control).

Quotes a two-sided market around the ensemble fair value and earns the spread.
Because we can't actually rest orders in paper mode, fills are *simulated from
real Kalshi trade prints*: a print at or below our bid would have lifted our
resting bid (we buy YES at our bid); a print at or above our ask would have hit
our ask (we sell YES = buy NO at 1-ask). Fills are capped per run and by a net
inventory limit. This yields a realistic, conservative MM P&L: we only capture
flow that actually traded through our quotes.

Two pieces of this strategy are factored into pure, side-effect-free functions
so the LIVE execution path (`kwt.live_engine`) can reuse exactly the same math:

  * `plan_quote(ctx, net_yes, params)` -> the bid/ask/size we would rest.
  * `simulate_fills(prev_bid, prev_ask, trades, net_yes, params)` -> the fills
    the participation model predicts against a resting quote. Live trading
    compares this prediction to real fills (the whole point of the experiment).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..db import kv_get, kv_set
from ..distributions import floored_nowcast
from .base import Order, Signal, Strategy


def _clip(x: float) -> float:
    return min(max(x, 0.01), 0.99)


def _flb_haircut(price: float, params: dict) -> float:
    """Favorite-longshot-bias haircut (dollars) to subtract from fair.

    Cheap longshots are systematically overpriced, so shade fair DOWN there.
    The haircut is `flb_haircut_at_10c` at price=0.10, interpolates LINEARLY to
    0 at `flb_haircut_zero_at`, is 0 at/above that price, and is held FLAT at
    the 0.10 value for prices below 0.10 (no extrapolation). Never negative.
    Default `flb_haircut_at_10c=0.0` -> no-op.
    """
    at10 = params.get("flb_haircut_at_10c", 0.0)
    if at10 <= 0:
        return 0.0
    zero_at = params.get("flb_haircut_zero_at", 0.40)
    if price >= zero_at:
        return 0.0
    if price <= 0.10:
        return max(at10, 0.0)
    denom = zero_at - 0.10
    if denom <= 0:
        return 0.0
    return max(at10 * (zero_at - price) / denom, 0.0)


@dataclass
class QuotePlan:
    """The quote we would rest this cycle for one market.

    `quotable` is False when we deliberately do not quote; `skip_reason` says
    why. When skip_reason is 'no_edge_vs_mkt' the bid/ask are still populated
    (and get cached as the next cycle's resting quote), matching the paper path.
    For 'no_forecast' / 'too_close_to_settle' / 'decided_by_obs' there is no
    quote; `fair` carries the feasible bound for the decided-by-obs signal.
    """
    new_bid: float | None
    new_ask: float | None
    fair: float | None
    disagree: float
    quotable: bool
    skip_reason: str | None = None
    model_fair: float | None = None   # the model's raw bucket prob (pre mid-blend)


@dataclass
class FlattenPlan:
    """Inventory regime for one market given reconciled net YES inventory.

    normal      - two-sided spread quoting (no inventory pressure).
    reduce_only - pull the accumulating side; rest ONLY the side that shrinks |net|.
    cross       - the position is toxic enough to pay to exit (aggressive reduce /
                  deferred taker execution). Deep-ITM positions stay reduce_only.
    """
    regime: str
    reduce_side: str | None    # 'ask' = sell YES (shed long); 'bid' = buy YES (shed short)
    reason: str = ""


def plan_flatten(c, net_yes: float, params: dict) -> FlattenPlan:
    """Decide the inventory regime for a market. Pure: reads only c/net_yes/params.

    Triggers (Fable §3), using data available without markout:
      * inventory band: |net| >= flatten_reduce_at -> reduce_only.
      * feasibility squeeze: the HELD side's model win-prob < flatten_min_win_prob
        -> cross (the market has moved against us / obs is deciding it).
      * time stop: same-day, past same_day_cutoff_hour local, held win-prob < 0.6
        -> cross (don't carry a coin-flip-or-worse into the informed endgame).
      * deep-ITM carve-out: if the held side is already worth >= flatten_deep_itm,
        max remaining loss is tiny -> never pay to cross; stay reduce_only.
    """
    if abs(net_yes) < 1:
        return FlattenPlan("normal", None, "")
    long = net_yes > 0
    reduce_side = "ask" if long else "bid"

    if c.nwp is not None:
        p_yes = c.nwp.p_bucket(c.low, c.high, blend_empirical=0.6)
    elif c.yes_mid is not None:
        p_yes = c.yes_mid
    else:
        p_yes = 0.5
    held_win = p_yes if long else (1.0 - p_yes)
    mid = c.yes_mid if c.yes_mid is not None else p_yes
    held_val = mid if long else (1.0 - mid)          # current worth of a held contract
    deep_itm = held_val >= params.get("flatten_deep_itm", 0.90)

    cutoff_h = params.get("same_day_cutoff_hour", 0.0)
    squeeze = held_win < params.get("flatten_min_win_prob", 0.35)
    time_stop = bool(cutoff_h) and c.hours_elapsed is not None \
        and c.hours_elapsed >= cutoff_h and held_win < 0.6
    if (squeeze or time_stop) and not deep_itm:
        return FlattenPlan("cross", reduce_side, "squeeze" if squeeze else "time_stop")
    if abs(net_yes) >= params.get("flatten_reduce_at", 2):
        return FlattenPlan("reduce_only", reduce_side, "inventory")
    return FlattenPlan("normal", None, "")


@dataclass
class FillSim:
    """Predicted fills of a resting quote against a window of real trade prints."""
    buy_yes: float
    buy_no: float
    fills: int
    print_vol: float
    run_cap: float
    newest: str
    adverse_skipped: float
    print_vol_bid: float = 0.0
    print_vol_ask: float = 0.0


def plan_quote(c, net_yes: float, params: dict) -> QuotePlan:
    """Compute the two-sided quote for market context `c` given net inventory.

    Pure function: reads only `c`, `net_yes`, and `params`. No DB, no network.
    """
    half = params.get("half_spread", 0.03)
    max_inv = params.get("max_inventory", 60)
    min_edge = params.get("min_edge_vs_market", 0.0)
    # Anchor fair value mostly to the market mid: our model fair is stale (one
    # forecast pull per cycle) and tail-biased, so quoting around it alone is
    # pure adverse selection — we get filled exactly when informed flow disagrees.
    w_mkt = params.get("fair_market_weight", 0.7)
    # Widen quotes when model and market disagree (uncertainty proxy).
    disagree_widen = params.get("disagree_widen", 0.5)
    # Skew quotes against inventory so fills mean-revert our book.
    inv_skew = params.get("inventory_skew", 0.02)
    # Don't quote in the final hours before settlement: informed nowcast flow
    # dominates there and the spread can't compensate.
    min_quote_h = params.get("min_quote_hours", 3.0)

    if c.nwp is None:
        return QuotePlan(None, None, None, 0.0, False, "no_forecast")
    if c.horizon_days * 24.0 < min_quote_h:
        return QuotePlan(None, None, None, 0.0, False, "too_close_to_settle")
    # Don't make a market on a bucket today's observation has already decided —
    # quoting a spread around a 0/1 outcome only books losses.
    p_lo, p_hi = c.feasible_yes_bounds()
    if p_lo == p_hi:
        return QuotePlan(None, None, p_lo, 0.0, False, "decided_by_obs")
    # ...and don't quote a NEARLY-decided same-day bucket either. After peak heating
    # the daily high is mostly set hours before close, so a still-wide-looking market
    # is picked off by informed nowcast flow the cycle-old fair can't see. Opt-in
    # (0.0 = off) so paper backtests are unchanged; the live pilot enables it.
    near_band = params.get("near_decided_band", 0.0)
    if near_band and (p_hi - p_lo) < near_band:
        return QuotePlan(None, None, (p_lo + p_hi) / 2.0, 0.0, False, "near_decided")
    # Local-time cutoff: on a SAME-DAY market (has an intraday clock), stop two-sided
    # quoting once the station-local day is past `same_day_cutoff_hour` — after peak
    # heating the flow is nowcast-informed by definition, and hours-to-close (which
    # min_quote_hours uses) fires ~5h too late. Opt-in (0 = off); paper unchanged.
    cutoff_h = params.get("same_day_cutoff_hour", 0.0)
    if cutoff_h and c.hours_elapsed is not None and c.hours_elapsed >= cutoff_h:
        return QuotePlan(None, None, (p_lo + p_hi) / 2.0, 0.0, False, "past_local_cutoff")

    model_fair = min(max(c.nwp.p_bucket(c.low, c.high, blend_empirical=0.6),
                         p_lo), p_hi)
    # (#3) Nowcast fair floor: recompute model_fair from a nowcast whose ensemble
    # members are floored at today's observed extreme (impossible mass removed).
    # Opt-in and only when we have an intraday observation; default off -> unchanged.
    if params.get("nowcast_floor_enabled") and c.obs_so_far is not None:
        model_fair = min(max(
            floored_nowcast(c.nwp, c.obs_so_far).p_bucket(
                c.low, c.high, blend_empirical=0.6), p_lo), p_hi)
    if c.yes_mid is not None:
        fair = w_mkt * c.yes_mid + (1 - w_mkt) * model_fair
        disagree = abs(model_fair - c.yes_mid)
    else:
        fair, disagree = model_fair, 0.0
    # (#1) Favorite-longshot haircut: shade fair DOWN by the price-dependent
    # haircut (0 by default). Anchored on the market price when we have one.
    price = c.yes_mid if c.yes_mid is not None else fair
    fair -= _flb_haircut(price, params)
    fair = min(max(fair, p_lo), p_hi)

    skew = -inv_skew * (net_yes / max_inv) if max_inv else 0.0
    have_book = c.yes_bid is not None and c.yes_ask is not None

    # (#1,#8) Bucket-side selection INTENT, computed before the two-sided spread
    # gates. When the mid sits where the other side is directionally toxic we rest
    # only one side (ask_only_below -> drop the bid; bid_only_above -> drop the ask).
    # An edge-driven one-sided quote (e.g. fading an overpriced cheap longshot) is
    # NOT making a two-sided market, so it is EXEMPT from the min_market_spread and
    # self-cross gates below — otherwise the ~1c weather books filter out exactly
    # the longshot fades we want to rest. Both flags unset (default) -> two-sided,
    # every gate applies as before (prod-neutral).
    ask_only = params.get("ask_only_below")
    bid_only = params.get("bid_only_above")
    drop_bid = ask_only is not None and c.yes_mid is not None and c.yes_mid < ask_only
    drop_ask = bid_only is not None and c.yes_mid is not None and c.yes_mid > bid_only
    one_sided = drop_bid or drop_ask

    # (#2) Only make a TWO-SIDED market wide enough to earn after fees. On these
    # weather books the spread is usually ~1c — quoting a fixed 3c half-spread there
    # just crosses (post-only reject) or sits far behind the touch and never fills.
    # Opt-in (0.0 = off) so paper is unchanged. Skipped for a one-sided quote.
    min_mkt_spread = params.get("min_market_spread", 0.0)
    if min_mkt_spread and have_book and not one_sided \
            and (c.yes_ask - c.yes_bid) < min_mkt_spread:
        return QuotePlan(None, None, fair, disagree, False, "market_too_tight",
                         model_fair=model_fair)

    if params.get("quote_at_touch") and have_book:
        # (#1) Anchor to the live book: join (or step inside) the touch so we're
        # actually competitive for fills, but NEVER quote through our fair — cap the
        # bid at fair-edge and floor the ask at fair+edge. On a wide book with fair
        # inside the touch this joins the touch and captures the spread; when our
        # fair sits outside the touch the quote pulls back to fair and simply
        # doesn't fill, which is the correct behavior (don't trade at prices we
        # think are wrong). Rounded to the 1c exchange grid.
        # (#2) Per-side edges: bids may sit further from fair than asks (the long
        # side is the toxic one on these books). None -> fall back to touch_min_edge.
        # Explicit None check (NOT `or`): a per-side edge of 0.0 is a valid setting
        # (join the touch on the fade side), and `0.0 or x` would wrongly fall back.
        _te = params.get("touch_min_edge", 0.01)
        _tb, _ta = params.get("touch_min_edge_bid"), params.get("touch_min_edge_ask")
        edge_bid = _tb if _tb is not None else _te
        edge_ask = _ta if _ta is not None else _te
        imp = params.get("touch_improve", 0.0)
        new_bid = _clip(round(min(c.yes_bid + imp, fair - edge_bid) + skew, 2))
        new_ask = _clip(round(max(c.yes_ask - imp, fair + edge_ask) + skew, 2))
        # Self-cross guard applies only to a two-sided rest; a one-sided quote
        # discards the crossing side anyway (see one_sided above).
        if new_bid >= new_ask and not one_sided:
            return QuotePlan(new_bid, new_ask, fair, disagree, False,
                             "spread_too_tight", model_fair=model_fair)
    else:
        eff_half = half + disagree_widen * disagree
        new_bid = _clip(fair - eff_half + skew)
        new_ask = _clip(fair + eff_half + skew)

    # Only quote if our fair view is at least min_edge away from the market mid.
    if c.yes_mid is not None and abs(fair - c.yes_mid) < min_edge:
        return QuotePlan(new_bid, new_ask, fair, disagree, False, "no_edge_vs_mkt",
                         model_fair=model_fair)

    # (#1,#8) Apply the bucket-side selection decided above: a one-sided QuotePlan
    # with quotable=True means "rest only the populated side" (see the one-sided
    # contract in generate/simulate_fills). Both flags unset -> two-sided, unchanged.
    if drop_bid:
        new_bid = None
    if drop_ask:
        new_ask = None
    if new_bid is None and new_ask is None:
        return QuotePlan(None, None, fair, disagree, False, "one_sided_empty",
                         model_fair=model_fair)
    return QuotePlan(new_bid, new_ask, fair, disagree, True, None,
                     model_fair=model_fair)


def simulate_fills(prev_bid: float | None, prev_ask: float | None,
                   trades: list[dict], net_yes: float, params: dict,
                   last_seen: str = "") -> FillSim:
    """Replay real trade prints against the previous-cycle resting quote.

    A print at/below our bid would have lifted our bid (we buy YES); a print
    at/above our ask would have hit our ask (we sell YES = buy NO). Total fills
    are capped at our realistic queue share (`participation_rate` of observed
    print volume) and the per-run / inventory limits. Prints that sweep deeper
    than `max_adverse_through` past our quote are skipped (a real resting order
    would have been picked off) and counted in `adverse_skipped`.
    """
    max_inv = params.get("max_inventory", 60)
    max_fills = params.get("max_fills_per_run", 10)
    size = params.get("quote_size", 5)
    participation_rate = params.get("participation_rate", 0.05)
    extreme_participation = params.get("extreme_participation", 0.02)
    max_through = params.get("max_adverse_through", 0.05)
    # Live-audit knobs (absent in paper params -> defaults keep paper identical):
    #  * max_fills_per_side caps fills at N events per side per interval. The live
    #    book rests exactly ONE order per side, so the audit passes 1 — without it
    #    the sim can "fill" the same side repeatedly and overstate the baseline.
    #  * count_adverse_as_fills=True is the PESSIMISTIC baseline: a real resting
    #    order cannot dodge a sweep, so prints through the quote are counted as
    #    fills (picked off) instead of only tallied in adverse_skipped.
    max_side = params.get("max_fills_per_side")
    count_adv = bool(params.get("count_adverse_as_fills", False))

    total_print_vol = sum(
        float(t.get("count_fp", t.get("count", 0)) or 0) for t in trades)
    is_extreme = (prev_bid is not None and prev_bid < 0.05) or \
                 (prev_ask is not None and prev_ask > 0.95)
    rate = extreme_participation if is_extreme else participation_rate
    run_cap = max(size, total_print_vol * rate)

    newest = last_seen
    fills = 0
    yes_fills = no_fills = 0        # fill EVENTS per side (for max_fills_per_side)
    buy_yes = buy_no = 0.0
    contracts_filled = 0.0
    adverse_skipped = 0.0
    print_vol_bid = print_vol_ask = 0.0

    # A resting quote fills its live side(s). One-sided contract: exactly one of
    # prev_bid/prev_ask being None means "rest only the populated side" — that side
    # fills, the None side never does. Both non-None is the two-sided path
    # (byte-identical to the original); both None is the first-cycle watermark path.
    if prev_bid is not None or prev_ask is not None:
        # Volume denominators cover the entire tape, independent of the simulated
        # fill caps below. Otherwise a capped early fill would make later public
        # prints disappear from the side-specific participation denominator.
        for t in trades:
            try:
                yp = float(t.get("yes_price_dollars"))
                cnt = float(t.get("count_fp", t.get("count", 0)))
            except (TypeError, ValueError):
                continue
            if prev_bid is not None and yp <= prev_bid:
                print_vol_bid += cnt
            if prev_ask is not None and yp >= prev_ask:
                print_vol_ask += cnt
        for t in trades:
            ct = t.get("created_time", "")
            if ct:
                newest = max(newest, ct)
            try:
                yp = float(t.get("yes_price_dollars"))
                cnt = float(t.get("count_fp", t.get("count", 0)))
            except (TypeError, ValueError):
                continue
            # News that swept through our resting quote — a real order would have
            # been adversely filled here; we record the magnitude regardless.
            adv_bid = prev_bid is not None and yp < prev_bid - max_through
            adv_ask = prev_ask is not None and yp > prev_ask + max_through
            if adv_bid or adv_ask:
                adverse_skipped += cnt
            # Stop exactly where the paper engine did: break (not continue) so the
            # watermark and fill set are byte-for-byte identical to the original.
            if fills >= max_fills or contracts_filled >= run_cap:
                break
            cur_net = net_yes + buy_yes - buy_no
            remaining_cap = run_cap - contracts_filled
            # A print fills our bid if it prints in [bid-through, bid]; our ask if
            # in [ask, ask+through]. Under the pessimistic baseline a sweep beyond
            # the band picks off the corresponding side too. A None side never fills.
            hit_bid = prev_bid is not None and (
                (prev_bid - max_through <= yp <= prev_bid) or (count_adv and adv_bid))
            hit_ask = prev_ask is not None and (
                (prev_ask <= yp <= prev_ask + max_through) or (count_adv and adv_ask))
            side_open_bid = max_side is None or yes_fills < max_side
            side_open_ask = max_side is None or no_fills < max_side
            if hit_bid and cur_net < max_inv and side_open_bid:
                fill = min(size, cnt, max_inv - cur_net, remaining_cap)
                if fill > 0:
                    buy_yes += fill
                    contracts_filled += fill
                    fills += 1
                    yes_fills += 1
            elif hit_ask and cur_net > -max_inv and side_open_ask:
                fill = min(size, cnt, max_inv + cur_net, remaining_cap)
                if fill > 0:
                    buy_no += fill
                    contracts_filled += fill
                    fills += 1
                    no_fills += 1
    else:
        # First cycle for this ticker — advance the watermark without fills.
        for t in trades:
            ct = t.get("created_time", "")
            if ct:
                newest = max(newest, ct)

    return FillSim(buy_yes, buy_no, fills, total_print_vol, run_cap, newest,
                   adverse_skipped, print_vol_bid, print_vol_ask)


class MarketMakingStrategy(Strategy):
    name = "market_making"

    def generate(self, ctxs, book):
        orders: list[Order] = []
        signals: list[Signal] = []
        conn = self.services.conn
        kalshi = self.services.kalshi
        if kalshi is None:
            return orders, signals

        for c in ctxs:
            net_yes = book.contracts(c.ticker, "yes") - book.contracts(c.ticker, "no")
            plan = plan_quote(c, net_yes, self.params)

            if plan.skip_reason == "no_forecast":
                continue
            if plan.skip_reason == "too_close_to_settle":
                signals.append(Signal(self.name, c.ticker, None, c.yes_mid, 0.0,
                                      "none", "skip", {"reason": "too_close_to_settle"}))
                continue
            if plan.skip_reason == "decided_by_obs":
                signals.append(Signal(self.name, c.ticker, plan.fair, c.yes_mid, 0.0,
                                      "none", "skip", {"reason": "decided_by_obs"}))
                continue

            if plan.skip_reason == "one_sided_empty":
                # Both sides suppressed by bucket-side selection — nothing to rest.
                signals.append(Signal(self.name, c.ticker, plan.fair, c.yes_mid, 0.0,
                                      "none", "skip", {"reason": "one_sided_empty"}))
                continue

            new_bid, new_ask = plan.new_bid, plan.new_ask
            if plan.skip_reason == "no_edge_vs_mkt":
                # Still cache the fresh quote so the next cycle has a resting price.
                kv_set(conn, f"mm:bid:{self.name}:{c.ticker}", f"{new_bid:.4f}")
                kv_set(conn, f"mm:ask:{self.name}:{c.ticker}", f"{new_ask:.4f}")
                signals.append(Signal(self.name, c.ticker, plan.fair, c.yes_mid, 0.0,
                                      "none", "skip", {"reason": "no_edge_vs_mkt"}))
                continue

            # Quotable: simulate fills against the quote that was resting since
            # the last cycle, then cache the freshly computed quote.
            wm_key = f"mm:{self.name}:{c.ticker}"
            last_seen = kv_get(conn, wm_key, "") or ""
            cbid = kv_get(conn, f"mm:bid:{self.name}:{c.ticker}", "")
            cask = kv_get(conn, f"mm:ask:{self.name}:{c.ticker}", "")
            prev_bid = float(cbid) if cbid else None
            prev_ask = float(cask) if cask else None

            trades = kalshi.trades_since(c.ticker, last_seen)  # oldest-first, fresh
            sim = simulate_fills(prev_bid, prev_ask, trades, net_yes, self.params,
                                 last_seen)

            if sim.newest:
                kv_set(conn, wm_key, sim.newest)
            # One-sided contract: cache the populated side; store "" for a None side
            # so next cycle reads prev_bid/prev_ask back as None (that side rests
            # nothing and never fills).
            kv_set(conn, f"mm:bid:{self.name}:{c.ticker}",
                   f"{new_bid:.4f}" if new_bid is not None else "")
            kv_set(conn, f"mm:ask:{self.name}:{c.ticker}",
                   f"{new_ask:.4f}" if new_ask is not None else "")

            if sim.buy_yes > 0:
                orders.append(Order(self.name, c.ticker, "yes", "fill_at", sim.buy_yes,
                                    prev_bid, "maker", f"mm_bid@{prev_bid:.2f}"))
            if sim.buy_no > 0:
                orders.append(Order(self.name, c.ticker, "no", "fill_at", sim.buy_no,
                                    _clip(1 - prev_ask),
                                    "maker", f"mm_ask@{prev_ask:.2f}"))
            spread = (new_ask - new_bid) if (new_bid is not None and new_ask is not None) else None
            signals.append(Signal(self.name, c.ticker, plan.fair, c.yes_mid,
                                  spread, "both",
                                  "enter" if (sim.buy_yes or sim.buy_no) else "hold",
                                  {"bid": round(new_bid, 3) if new_bid is not None else None,
                                   "ask": round(new_ask, 3) if new_ask is not None else None,
                                   "prev_bid": round(prev_bid, 3) if prev_bid is not None else None,
                                   "prev_ask": round(prev_ask, 3) if prev_ask is not None else None,
                                   "fills": sim.fills, "buy_yes": sim.buy_yes,
                                   "buy_no": sim.buy_no, "net_yes": net_yes,
                                   "run_cap": round(sim.run_cap, 1),
                                   "print_vol": round(sim.print_vol, 1)}))
        return orders, signals
