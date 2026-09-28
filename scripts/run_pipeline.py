"""
V8 end-to-end pipeline runner (config-driven).

Phase 0: this runs the implemented stages (load + shape report) fully, then
walks the remaining stages and reports each as [done] or [pending]. As each
phase is built, its stage lights up here automatically — so this runner doubles
as the project's live progress indicator.

Usage
-----
    python scripts/run_pipeline.py                      # config.yaml + bundled sample
    python scripts/run_pipeline.py --config config.yaml # explicit config
    python scripts/run_pipeline.py --data path/to/other_matches.csv  # different data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable whether this is run from the root or elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, set_seed
from src import (
    ingest,
    features,
    dixon_coles,
    calibrate,
    staking,
    backtest,
    clv,
    metrics,
)


def _run_stage(name, fn, *args, **kwargs):
    """Run one pipeline stage, reporting done/pending without crashing the run.

    A stage that isn't built yet raises NotImplementedError; we catch it and
    mark the stage pending, so the whole pipeline always completes cleanly.
    """
    try:
        result = fn(*args, **kwargs)
        print(f"  [done]    {name}")
        return result
    except NotImplementedError as exc:
        print(f"  [pending] {name}  ->  {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run the V8 pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--data", default=None, help="Override the input CSV path")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # Resolve the data path (CLI override wins; relative paths resolve to root).
    data_path = args.data or config["data"]["sample_path"]
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path

    print(f"\nV8 pipeline — config loaded, seed={config.get('seed', 42)}")
    print(f"input data: {data_path}\n")

    # --- Stage 1: ingest (implemented in Phase 0) ---
    print("Stage 1 — Ingest")
    df = ingest.load_matches(data_path, config)
    print(ingest.build_shape_report(df))
    print()

    # --- Downstream stages (progressively implemented) ---
    print("Downstream stages:")
    feats = _run_stage("Stage 2 — Features", features.build_features, df, config)
    model = _run_stage("Stage 3 — Dixon-Coles fit", dixon_coles.fit, feats, config)
    _run_stage("Stage 4 — Calibration", calibrate.calibrate, None, None, config)
    _run_stage("Stage 5 — Staking", staking.stake, None, None, config)
    _run_stage("Stage 6 — Backtest", backtest.walk_forward, df, config)
    _run_stage("Stage 6 — CLV", clv.compute_clv, None, None)
    _run_stage("Stage 6 — Metrics", metrics.summarise, None)

    print("\nPhase 0 complete: pipeline runs end-to-end and reports stage status.")


if __name__ == "__main__":
    main()
