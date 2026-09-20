from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kalshi_fund.calibration import build_observation_frame, compute_edge_map
from kalshi_fund.calibration_report import render_report
from kalshi_fund.storage import query_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the category calibration edge map.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--min-obs", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markets = query_frame(args.db, "SELECT * FROM kalshi_settled_markets")
    snapshots = query_frame(args.db, "SELECT * FROM kalshi_price_snapshots")
    observations = build_observation_frame(markets, snapshots)
    unknown = int((observations["category"] == "unknown").sum())
    print(f"{unknown} observations with unknown category")
    edge_map = compute_edge_map(
        observations, min_obs=args.min_obs, n_boot=args.n_boot, seed=args.seed
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_map.to_csv(output_dir / "calibration_edge_map.csv", index=False)
    report = render_report(edge_map, n_markets=len(markets), n_observations=len(observations))
    (output_dir / "calibration_report.md").write_text(report, encoding="utf-8")
    graduated = edge_map[edge_map["graduated_side"].isin(["yes", "no"])] if not edge_map.empty else edge_map
    print(f"{len(edge_map)} cells; {len(graduated)} graduated rows")
    print(f"Wrote {output_dir / 'calibration_edge_map.csv'} and {output_dir / 'calibration_report.md'}")


if __name__ == "__main__":
    main()
