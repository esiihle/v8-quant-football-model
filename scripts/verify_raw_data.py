#!/usr/bin/env python3
"""
Check whether a candidate data file can actually drive this pipeline.

football-data.co.uk is the intended source, but it is a betting-odds site and
is blocked on some networks. Mirrors exist — Kaggle, GitHub, assorted archives —
and they vary in quality: some drop the closing-odds columns, some cover
different seasons, some have been "cleaned" in ways that silently change the
numbers.

This script answers one question about any candidate file: **will V8 produce a
real result from it, or a plausible-looking fake one?** It checks the columns
the pipeline needs, the shape of the data, and the internal consistency of the
odds — because a file with the right headers and nonsense underneath is the
failure mode that matters.

Usage
-----
    python scripts/verify_raw_data.py data/raw/E0_2425.csv
    python scripts/verify_raw_data.py data/raw/*.csv
    python scripts/verify_raw_data.py some_download.csv --quiet

Exit codes
----------
    0  usable
    1  unusable — the reasons are printed
    2  usable, but with warnings worth reading
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config

# A season of a top league is 380 matches; anything far below this will not
# support a walk-forward backtest with a 200-match minimum training window.
MIN_USABLE_ROWS = 200


def describe(path, config, quiet=False):
    """Return (verdict, failures, warnings, facts) for one candidate file."""
    failures, warnings, facts = [], [], {}
    schema = config.get("schema", {})

    try:
        frame = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        # football-data.co.uk files are occasionally latin-1.
        frame = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    except Exception as error:
        return "UNUSABLE", [f"could not be read as CSV: {error}"], [], {}

    facts["rows"] = len(frame)
    facts["columns"] = len(frame.columns)

    # --- 1. A saved error page is still a file ---------------------------
    size_kb = Path(path).stat().st_size / 1024
    facts["size_kb"] = round(size_kb, 1)
    if size_kb < 10:
        failures.append(
            f"only {size_kb:.1f} KB — this is almost certainly a saved error "
            "page or a redirect, not a season of matches"
        )

    # --- 2. The columns the pipeline cannot work without ------------------
    required = list(schema.get("required_columns", []))
    missing = [c for c in required if c not in frame.columns]
    if missing:
        failures.append(f"missing required columns: {missing}")

    # --- 3. Odds: the bet-into price decides bets ------------------------
    bet_odds = list(schema.get("odds_columns", []))
    missing_bet = [c for c in bet_odds if c not in frame.columns]
    if missing_bet:
        failures.append(
            f"missing the odds bet into {missing_bet} — without a price there "
            "is no edge to measure and no bet to place"
        )

    # --- 4. Closing odds: the entire CLV claim rests on these ------------
    close_odds = list(schema.get("closing_odds_columns", []))
    missing_close = [c for c in close_odds if c not in frame.columns]
    if missing_close:
        failures.append(
            f"missing closing odds {missing_close} — CLV is this project's "
            "headline claim and cannot be computed without them. A mirror "
            "that drops these columns is not a substitute for the real file"
        )

    # --- 5. Enough matches to back-test ----------------------------------
    minimum = int(config.get("backtest", {}).get("min_training_matches", 200))
    if len(frame) < MIN_USABLE_ROWS:
        failures.append(
            f"{len(frame)} rows: the walk-forward backtest needs more than "
            f"{minimum} matches for training alone"
        )

    # --- 6. Does the data say what it should? ----------------------------
    # Headers can be right while the contents are wrong. These checks are what
    # separate "looks like the file" from "is the file".
    if not missing:
        goals = frame[["FTHG", "FTAG"]].apply(pd.to_numeric, errors="coerce")
        facts["mean_goals"] = round(float(goals.sum(axis=1).mean()), 2)
        if not (1.5 <= facts["mean_goals"] <= 4.5):
            warnings.append(
                f"mean total goals is {facts['mean_goals']} — a top league "
                "sits near 2.7. Check the columns are what they claim"
            )

        home_rate = (frame["FTR"].astype(str) == "H").mean()
        facts["home_win_rate"] = round(float(home_rate), 3)
        if not (0.30 <= home_rate <= 0.55):
            warnings.append(
                f"home win rate is {home_rate:.1%} — expect roughly 43-46%"
            )

        try:
            dates = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
            facts["first_date"] = str(dates.min().date())
            facts["last_date"] = str(dates.max().date())
            if dates.isna().mean() > 0.01:
                warnings.append(
                    f"{dates.isna().mean():.1%} of dates did not parse"
                )
        except Exception:
            warnings.append("the Date column could not be parsed")

    # --- 7. Are the odds a real book? ------------------------------------
    # The strongest integrity check available. A real bookmaker's three prices
    # imply probabilities summing to a little OVER 1 — the overround, typically
    # 2-8%. Fabricated or mangled odds rarely get this right.
    for label, columns in (("bet-into", bet_odds), ("closing", close_odds)):
        if columns and all(c in frame.columns for c in columns):
            odds = frame[columns].apply(pd.to_numeric, errors="coerce")
            implied = (1.0 / odds).sum(axis=1)
            usable = implied.replace([np.inf, -np.inf], np.nan).dropna()
            if len(usable) == 0:
                failures.append(f"{label} odds are present but not numeric")
                continue

            median_book = float(usable.median())
            facts[f"{label}_overround"] = f"{(median_book - 1) * 100:.1f}%"
            facts[f"{label}_coverage"] = f"{len(usable) / len(frame):.0%}"

            if not (1.0 <= median_book <= 1.20):
                failures.append(
                    f"{label} odds imply probabilities summing to "
                    f"{median_book:.3f}. A real book sums to between 1.00 and "
                    "1.12; this is not a coherent set of prices"
                )
            elif median_book > 1.12:
                warnings.append(
                    f"{label} overround of {(median_book - 1) * 100:.1f}% is "
                    "high for a top league"
                )

            if len(usable) / len(frame) < 0.90:
                warnings.append(
                    f"only {len(usable) / len(frame):.0%} of rows have usable "
                    f"{label} odds"
                )

    # --- 8. Closing should be sharper than opening -----------------------
    # Not a hard rule, but a real closing line carries a thinner margin than the
    # price offered days earlier. If the two are identical, the "closing" column
    # is a copy and CLV would be measured against itself.
    if (bet_odds and close_odds
            and all(c in frame.columns for c in bet_odds + close_odds)):
        bet = frame[bet_odds].apply(pd.to_numeric, errors="coerce")
        close = frame[close_odds].apply(pd.to_numeric, errors="coerce")
        both = bet.notna().all(axis=1) & close.notna().all(axis=1)
        if both.sum():
            identical = (bet[both].to_numpy() == close[both].to_numpy()).all(axis=1).mean()
            facts["identical_open_close"] = f"{identical:.1%}"
            if identical > 0.5:
                failures.append(
                    f"{identical:.0%} of rows have closing odds identical to "
                    "the opening odds — the closing column is a copy, so CLV "
                    "would be measured against itself and always read zero"
                )

    verdict = ("UNUSABLE" if failures
               else "USABLE WITH WARNINGS" if warnings
               else "USABLE")
    return verdict, failures, warnings, facts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="CSV file(s) to check")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--quiet", action="store_true",
                        help="Print the verdict line only")
    args = parser.parse_args()

    config = load_config(args.config)
    worst = 0

    for raw in args.paths:
        path = Path(raw)
        print()
        if not path.exists():
            print(f"{path}: NOT FOUND")
            worst = max(worst, 1)
            continue

        verdict, failures, warnings, facts = describe(path, config, args.quiet)
        print(f"{path.name}: {verdict}")

        if not args.quiet and facts:
            print("  what it contains:")
            for key, value in facts.items():
                print(f"    {key.replace('_', ' '):<22} {value}")

        for problem in failures:
            print(f"  FAIL  {problem}")
        for problem in warnings:
            print(f"  WARN  {problem}")

        if failures:
            worst = max(worst, 1)
        elif warnings:
            worst = max(worst, 2)

    print()
    if worst == 0:
        print("Every file is usable. Run: python scripts/run_pipeline.py")
    elif worst == 2:
        print("Usable, but read the warnings before trusting the numbers.")
    else:
        print("At least one file cannot drive this pipeline. Do not run the")
        print("backtest on it — a result from the wrong data is worse than no")
        print("result, because it looks exactly like a real one.")
    print()
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
