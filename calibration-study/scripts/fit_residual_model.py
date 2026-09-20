from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kalshi_fund.modeling import (
    WalkForwardConfig,
    estimate_trade_parameters,
    fit_full_sample_predictions,
    walk_forward_predictions,
)
from kalshi_fund.panel import available_feature_columns, clip_prob
from kalshi_fund.storage import append_frame, insert_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the residual model with walk-forward validation.")
    parser.add_argument("--input", required=True, help="Panel CSV created by build_panel.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported artifacts.")
    parser.add_argument("--db", help="Optional SQLite database for model run and prediction logging.")
    parser.add_argument("--min-train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--z-score", type=float, default=1.0, help="Default conservatism multiplier to log with the run.")
    parser.add_argument("--horizon-days", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_csv(args.input, parse_dates=["timestamp", "event_date", "target_timestamp", "future_timestamp"])
    numeric_features, categorical_features = available_feature_columns(panel)
    if not numeric_features and not categorical_features:
        raise SystemExit("panel does not contain usable features")

    config = WalkForwardConfig(
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        ridge_alpha=args.ridge_alpha,
        min_train_rows=args.min_train_rows,
    )

    oof_predictions, metrics = walk_forward_predictions(panel, numeric_features, categorical_features, config)
    params = estimate_trade_parameters(oof_predictions)
    alpha = float(params.loc[0, "alpha"])
    sigma = float(params.loc[0, "sigma"])

    latest_rows = (
        panel.sort_values("timestamp")
        .groupby(["race_id", "contract", "cycle"], group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )
    model_predictions = fit_full_sample_predictions(panel, latest_rows, numeric_features, categorical_features, config)
    model_predictions["p_blend"] = clip_prob((1.0 - alpha) * model_predictions["p_mkt"] + alpha * model_predictions["p_fair_model"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = output_dir / "walk_forward_predictions.csv"
    metrics_path = output_dir / "walk_forward_metrics.csv"
    predictions_path = output_dir / "model_predictions.csv"
    params_path = output_dir / "model_parameters.csv"

    params = params.assign(
        z_score=args.z_score,
        horizon_days=args.horizon_days,
        model_name="ridge_residual_baseline",
    )

    model_run_id = None
    if args.db:
        model_run_id = insert_record(
            args.db,
            "model_runs",
            {
                "target_name": config.target_col,
                "horizon_days": args.horizon_days,
                "model_name": "ridge_residual_baseline",
                "train_start": panel["timestamp"].min().isoformat(),
                "train_end": panel["timestamp"].max().isoformat(),
                "validation_end": oof_predictions["timestamp"].max().isoformat(),
                "alpha": alpha,
                "sigma": sigma,
                "z_score": args.z_score,
                "notes": f"numeric_features={numeric_features}; categorical_features={categorical_features}",
            },
        )
        db_predictions = model_predictions[
            ["timestamp", "race_id", "contract", "cycle", "state", "p_mkt", "p_anchor", "p_fair_model", "p_blend", "delta_hat"]
        ].copy()
        db_predictions["model_run_id"] = model_run_id
        db_predictions["sigma"] = sigma
        db_predictions["target_probability"] = pd.NA
        db_predictions["metadata_json"] = pd.NA
        append_frame(args.db, "model_predictions", db_predictions)

    if model_run_id is not None:
        params["model_run_id"] = model_run_id
        model_predictions["model_run_id"] = model_run_id

    oof_predictions.to_csv(oof_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    model_predictions.to_csv(predictions_path, index=False)
    params.to_csv(params_path, index=False)

    print(f"Wrote walk-forward predictions to {oof_path}")
    print(f"Wrote walk-forward metrics to {metrics_path}")
    print(f"Wrote scored latest rows to {predictions_path}")
    print(f"Wrote model parameters to {params_path}")


if __name__ == "__main__":
    main()
