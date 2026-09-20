from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kalshi_fund.signals import SignalConfig, generate_signal_frame
from kalshi_fund.storage import append_frame, query_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conservative trading signals from model predictions.")
    parser.add_argument("--predictions", required=True, help="CSV produced by fit_residual_model.py.")
    parser.add_argument("--params", required=True, help="Parameter CSV produced by fit_residual_model.py.")
    parser.add_argument("--output", required=True, help="Output CSV path for generated signals.")
    parser.add_argument("--db", help="Optional SQLite database for reading open trades and logging signals.")
    parser.add_argument("--bankroll-usd", type=float, default=10_000.0)
    parser.add_argument("--z-score", type=float, help="Override the default z-score from model_parameters.csv.")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--hurdle-bps", type=float, default=0.0)
    parser.add_argument("--fractional-kelly", type=float, default=0.25)
    parser.add_argument("--race-cap-fraction", type=float, default=0.05)
    parser.add_argument("--state-cap-fraction", type=float, default=0.10)
    parser.add_argument("--cycle-cap-fraction", type=float, default=0.15)
    parser.add_argument("--total-cap-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions, parse_dates=["timestamp"])
    params = pd.read_csv(args.params)
    alpha = float(params.loc[0, "alpha"])
    sigma = float(params.loc[0, "sigma"])
    z_score = float(args.z_score if args.z_score is not None else params.get("z_score", pd.Series([1.0])).iloc[0])
    model_run_id = params.get("model_run_id", pd.Series([pd.NA])).iloc[0]

    open_trades = query_frame(args.db, "SELECT * FROM trades") if args.db else pd.DataFrame()
    config = SignalConfig(
        alpha=alpha,
        sigma=sigma,
        z_score=z_score,
        fee=args.fee_bps / 10_000.0,
        slippage=args.slippage_bps / 10_000.0,
        hurdle=args.hurdle_bps / 10_000.0,
        bankroll_usd=args.bankroll_usd,
        fractional_kelly=args.fractional_kelly,
        race_cap_fraction=args.race_cap_fraction,
        state_cap_fraction=args.state_cap_fraction,
        cycle_cap_fraction=args.cycle_cap_fraction,
        total_cap_fraction=args.total_cap_fraction,
    )

    signals = generate_signal_frame(predictions, config, open_trades=open_trades)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_path, index=False)

    if args.db:
        db_signals = signals[
            ["timestamp", "race_id", "contract", "cycle", "state", "side", "entry_price", "conservative_probability", "net_edge", "suggested_notional_usd", "status"]
        ].copy()
        db_signals["model_run_id"] = model_run_id
        db_signals["thesis_id"] = pd.NA
        db_signals["metadata_json"] = pd.NA
        append_frame(args.db, "signals", db_signals)

    print(f"Wrote {len(signals)} signals to {output_path}")


if __name__ == "__main__":
    main()
