"""
V8 end-to-end pipeline runner (config-driven).

Phase 0: this runs the implemented stages (load + shape report) fully, then
walks the remaining stages and reports each as [done] or [pending]. As each
phase is built, its stage lights up here automatically — so this runner doubles
as the project's live progress indicator.

30 Sep 2026: the runner now prefers REAL data. It looks for the league/season
files listed in config.yaml under data/raw/; if none are there it falls back to
the bundled sample and prints the exact download URLs. The sample stays in the
repo on purpose, so a fresh clone (and the test suite) always runs without
needing a download.

Usage
-----
    python scripts/run_pipeline.py                      # real data if present, else sample
    python scripts/run_pipeline.py --config config.yaml # explicit config
    python scripts/run_pipeline.py --data path/to/other_matches.csv  # one specific file
    python scripts/run_pipeline.py --sample             # force the bundled sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

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
    parser.add_argument("--sample", action="store_true",
                        help="Force the bundled sample instead of data/raw")
    parser.add_argument("--no-save", action="store_true",
                        help="Run the checks but don't write the canonical table")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    print(f"\nV8 pipeline — config loaded, seed={config.get('seed', 42)}")

    # --- Stage 1: ingest ---
    # Three ways in, in priority order:
    #   1. --data <file>   : one explicit file
    #   2. data/raw/*      : the real league-season files named in the config
    #   3. the sample      : bundled fallback, so a fresh clone always runs
    print("Stage 1 — Ingest")

    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = REPO_ROOT / data_path
        print(f"  source: explicit file — {data_path}")
        df = ingest.load_matches(data_path, config)

    else:
        raw_files = ingest.expected_raw_files(config)
        found = [f for f in raw_files if f["exists"]]
        missing = [f for f in raw_files if not f["exists"]]

        # Always say what is missing, even if some files were found: a silently
        # half-loaded dataset is exactly the kind of thing that ruins a backtest.
        for entry in missing:
            print(f"  MISSING: {entry['path'].name}  ->  download {entry['url']}")
            print(f"           and save it as {entry['path']}")

        if found and not args.sample:
            print(f"  source: real data — {len(found)} of {len(raw_files)} configured files")
            df = ingest.load_many(found, config)
        else:
            sample_path = REPO_ROOT / config["data"]["sample_path"]
            reason = "forced by --sample" if args.sample else "no raw files found yet"
            print(f"  source: bundled sample ({reason}) — {sample_path.name}")
            df = ingest.load_matches(sample_path, config)

    print()
    print(ingest.build_shape_report(df, config))

    # --- Phase 1: validate, clean, canonicalise -----------------------------
    # Everything downstream reads the canonical table, never the raw frame.
    print()
    matches, report = ingest.run_ingest(df, config, save=not args.no_save)
    print(ingest.quality_report(report, matches))
    if "saved_to" in report:
        print(f"\n  canonical table written to: {report['saved_to']}")
    print()

    # --- Phase 2: features, built strictly out of sample ---------------------
    print("Stage 2 — Feature engineering")
    feature_matrix, feature_rep = features.build_features(matches, config)
    print(features.feature_report(feature_matrix, feature_rep))
    if not args.no_save:
        path = features.save_features(feature_matrix, config)
        print(f"\n  feature matrix written to: {path}")
    print()

    # --- Phase 3: Dixon-Coles fit by penalised, time-weighted MLE ------------
    print("Stage 3 — Dixon-Coles fit")

    # Phase 2's shrunk strengths seed the optimiser: a cheap, stable estimate is
    # a far better starting point than a flat guess. The two phases are not
    # rivals — Phase 2 describes, Phase 3 fits.
    blended, _ = features.blend_goals(matches, matches, config)
    seed_strengths = features.team_strengths(
        blended, as_of=matches["MatchDate"].max() + pd.Timedelta(days=1), config=config
    )

    model = dixon_coles.fit(matches, config, seed_strengths=seed_strengths)
    print(dixon_coles.fit_report(model))

    # A worked example: price the most recent fixture end to end.
    latest = matches.sort_values("MatchDate").tail(1)
    priced = dixon_coles.predict(model, latest, config)
    row = priced.iloc[0]
    print()
    print(f"  Worked example — {row['HomeTeam']} vs {row['AwayTeam']}:")
    print(f"    expected goals   {row['ExpectedHomeGoals']:.2f} - "
          f"{row['ExpectedAwayGoals']:.2f}")
    print(f"    1X2              {row['home_win']:.1%} / {row['draw']:.1%} / "
          f"{row['away_win']:.1%}")
    print(f"    over 2.5 goals   {row['over_2_5']:.1%}")
    print(f"    both teams score {row['btts_yes']:.1%}")
    print(f"    most likely      {row['MostLikelyScore']}")

    if not args.no_save:
        model_path = dixon_coles.save_model(model, config)
        print(f"\n  fitted model written to: {model_path}")
    print()

    # --- Downstream stages (progressively implemented) ---
    print("Downstream stages:")
    _run_stage("Stage 4 — Calibration", calibrate.calibrate, None, None, config)
    _run_stage("Stage 5 — Staking", staking.stake, None, None, config)
    _run_stage("Stage 6 — Backtest", backtest.walk_forward, df, config)
    _run_stage("Stage 6 — CLV", clv.compute_clv, None, None)
    _run_stage("Stage 6 — Metrics", metrics.summarise, None)

    print("\nPhases 0-2 complete: canonical table and feature matrix built;"
          " stages 3-6 report status only.")


if __name__ == "__main__":
    main()
