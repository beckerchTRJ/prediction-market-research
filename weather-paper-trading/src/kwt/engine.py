"""Paper-trading engine: fill orders, book positions, track cash & equity.

Accounting convention (all in YES-equivalent dollars):
  * Buying a YES contract costs `price` now; pays $1 at settlement iff result=yes.
  * Buying a NO contract  costs `price` now; pays $1 at settlement iff result=no.
    (Buying NO is how you short an overpriced YES bucket.)
  * cash starts at the strategy's bankroll; a buy debits (contracts*price + fee);
    settlement credits the $1 payoff on winning contracts.
  * equity = cash + cost-basis of open positions (conserved until settlement,
    so the equity curve moves only on realized resolutions — conservative).
"""
from __future__ import annotations

import sqlite3
from typing import Any

from .config import utcnow_iso
from .db import jdump, upsert
from .fees import maker_fee, taker_fee
from .strategies import Book, Order, Signal

EPS = 1e-9


def load_book(conn: sqlite3.Connection, strategy: str) -> Book:
    row = conn.execute("SELECT cash FROM strategies WHERE name=?", (strategy,)).fetchone()
    cash = float(row["cash"]) if row else 0.0
    book = Book(cash=cash)
    for r in conn.execute(
        "SELECT ticker, side, contracts, cost FROM positions WHERE strategy=?", (strategy,)
    ):
        book.positions[(r["ticker"], r["side"])] = {
            "contracts": r["contracts"], "cost": r["cost"]}
    return book


def _side_ask(snap: dict[str, Any], side: str) -> float | None:
    return snap.get("yes_ask") if side == "yes" else snap.get("no_ask")


def _side_bid(snap: dict[str, Any], side: str) -> float | None:
    return snap.get("yes_bid") if side == "yes" else snap.get("no_bid")


def execute_sell(conn: sqlite3.Connection, order: Order, snap: dict[str, Any],
                 book: Book, fee_cfg: dict[str, Any]) -> float:
    """Exit (part of) a held position at the bid; realize P&L immediately.

    Proceeds = contracts*bid - taker fee; realized pnl = proceeds minus the
    proportional cost basis of the contracts sold. The trade row is inserted
    already settled=1 so settle.py's buy-side settlement UPDATE (which targets
    settled=0 rows) never touches it.
    """
    taker_rate = fee_cfg.get("taker_rate", 0.07)
    bid = _side_bid(snap, order.side)
    if bid is None or bid <= 0 or bid < order.limit_price - EPS:
        return 0.0
    key = (order.ticker, order.side)
    pos = book.positions.get(key)
    held = pos["contracts"] if pos else 0.0
    contracts = float(int(min(order.contracts, held)))
    if contracts < 1:
        return 0.0
    avg_cost = pos["cost"] / pos["contracts"]
    fee = taker_fee(contracts, bid, taker_rate)
    proceeds = contracts * bid - fee
    pnl = proceeds - contracts * avg_cost

    ts = utcnow_iso()
    conn.execute(
        "INSERT INTO trades (ts, strategy, ticker, side, action, contracts, price, fee, "
        "role, reason, settled, pnl) VALUES (?,?,?,?,?,?,?,?,?,?,1,?)",
        (ts, order.strategy, order.ticker, order.side, "sell", contracts, bid, fee,
         order.role, order.reason, pnl))
    remaining = pos["contracts"] - contracts
    if remaining < 1:
        book.positions.pop(key, None)
        conn.execute("DELETE FROM positions WHERE strategy=? AND ticker=? AND side=?",
                     (order.strategy, order.ticker, order.side))
    else:
        new_cost = pos["cost"] - contracts * avg_cost
        book.positions[key] = {"contracts": remaining, "cost": new_cost}
        conn.execute(
            "UPDATE positions SET contracts=?, cost=? WHERE strategy=? AND ticker=? AND side=?",
            (remaining, new_cost, order.strategy, order.ticker, order.side))
    book.cash += proceeds
    srow = conn.execute("SELECT realized_pnl FROM strategies WHERE name=?",
                        (order.strategy,)).fetchone()
    conn.execute("UPDATE strategies SET cash=?, realized_pnl=? WHERE name=?",
                 (book.cash, srow["realized_pnl"] + pnl, order.strategy))
    return contracts


def execute_order(conn: sqlite3.Connection, order: Order, snap: dict[str, Any],
                  book: Book, fee_cfg: dict[str, Any]) -> float:
    """Fill an order against the latest snapshot; persist trade + position + cash.

    Returns the number of contracts actually filled (0 if unfilled)."""
    if order.action == "sell":
        return execute_sell(conn, order, snap, book, fee_cfg)
    taker_rate = fee_cfg.get("taker_rate", 0.07)
    maker_rate = fee_cfg.get("maker_rate", 0.0025)
    slip = fee_cfg.get("assume_spread_cost", 0.0)

    if order.action == "buy":
        ask = _side_ask(snap, order.side)
        if ask is None or ask <= 0 or ask > order.limit_price + EPS:
            return 0.0
        price = ask
    elif order.action == "fill_at":
        price = order.limit_price
    else:
        return 0.0

    price = min(max(price + slip, 0.001), 0.999)
    contracts = float(order.contracts)
    if contracts < 1:
        return 0.0

    def cost_of(n: float) -> float:
        fee = (maker_fee(n, price, maker_rate) if order.role == "maker"
               else taker_fee(n, price, taker_rate))
        return n * price + fee, fee

    cost, fee = cost_of(contracts)
    if cost > book.cash:
        # scale down to what cash allows
        contracts = float(int((book.cash * 0.999) / price))
        if contracts < 1:
            return 0.0
        cost, fee = cost_of(contracts)
        if cost > book.cash:
            return 0.0

    ts = utcnow_iso()
    conn.execute(
        "INSERT INTO trades (ts, strategy, ticker, side, action, contracts, price, fee, "
        "role, reason, settled, pnl) VALUES (?,?,?,?,?,?,?,?,?,?,0,NULL)",
        (ts, order.strategy, order.ticker, order.side, "buy", contracts, price, fee,
         order.role, order.reason),
    )
    key = (order.ticker, order.side)
    pos = book.positions.get(key, {"contracts": 0.0, "cost": 0.0})
    pos = {"contracts": pos["contracts"] + contracts, "cost": pos["cost"] + cost}
    book.positions[key] = pos
    book.cash -= cost

    upsert(conn, "positions", {
        "strategy": order.strategy, "ticker": order.ticker, "side": order.side,
        "contracts": pos["contracts"], "cost": pos["cost"],
        "fees": (conn.execute(
            "SELECT COALESCE(SUM(fee),0) f FROM trades WHERE strategy=? AND ticker=? AND side=?",
            (order.strategy, order.ticker, order.side)).fetchone()["f"]),
        "opened_ts": ts,
    }, keys=["strategy", "ticker", "side"])
    conn.execute("UPDATE strategies SET cash=? WHERE name=?", (book.cash, order.strategy))
    return contracts


def record_signal(conn: sqlite3.Connection, ts: str, sig: Signal) -> None:
    upsert(conn, "signals", {
        "ts": ts, "strategy": sig.strategy, "ticker": sig.ticker,
        "model_prob": sig.model_prob, "market_prob": sig.market_prob,
        "edge": sig.edge, "side": sig.side, "decision": sig.decision,
        "meta_json": jdump(sig.meta),
    }, keys=["ts", "strategy", "ticker"])


def position_cost(conn: sqlite3.Connection, strategy: str) -> float:
    r = conn.execute("SELECT COALESCE(SUM(cost),0) c FROM positions WHERE strategy=?",
                     (strategy,)).fetchone()
    return float(r["c"])


def record_equity(conn: sqlite3.Connection, ts: str, strategy: str) -> None:
    row = conn.execute("SELECT cash, realized_pnl FROM strategies WHERE name=?",
                       (strategy,)).fetchone()
    cash = float(row["cash"])
    pcost = position_cost(conn, strategy)
    upsert(conn, "equity", {
        "ts": ts, "strategy": strategy, "cash": cash, "position_cost": pcost,
        "realized_pnl": float(row["realized_pnl"]), "equity": cash + pcost,
    }, keys=["ts", "strategy"])
