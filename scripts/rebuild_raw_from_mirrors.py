#!/usr/bin/env python3
"""
Rebuild football-data.co.uk season files from two GitHub mirrors.

Why this exists
---------------
football-data.co.uk is unreachable from the author's network (gambling-category
filtering). GitHub is reachable. No single GitHub mirror has everything V8
needs, but two of them together do:

    ODDS mirror    AnishKhetani/premier-league-data
                   data/processed/results_with_odds.csv
                   -> every bookmaker price, INCLUDING Pinnacle closing odds,
                      renamed from football-data's codes (B365H -> bet365_1x2_home,
                      PSCH -> pinnacle_1x2_home_close). No shots.

    SHOTS mirror   datasets/football-datasets
                   datasets/premier-league/season-YYZZ.csv
                   -> results, shots and shots on target in the ORIGINAL
                      football-data column names. No odds at all.

This script joins them on (date, home team, away team), renames the odds back
to football-data's codes, and writes E0_<season>.csv into data/raw/ - exactly
the file V8 would have read had the download worked.

It is NOT part of the pipeline and does no networking. You clone the mirrors
yourself and pass their paths in; that keeps the project's rule that a run never
depends on a website being up (methodology §7.4).

The joins are checked, not trusted
----------------------------------
Two independent copies of the same results are a free audit. For every match
the two mirrors must agree on the goals. Any disagreement, any match present in
one mirror but not the other, or any missing price stops the script before a
file is written - the same "refuse, don't warn" rule as make_charts.py.

Usage
-----
    git clone --depth 1 https://github.com/AnishKhetani/premier-league-data   ../mirror-odds
    git clone --depth 1 https://github.com/datasets/football-datasets         ../mirror-shots

    python scripts/rebuild_raw_from_mirrors.py \
        --odds  ../mirror-odds  --shots ../mirror-shots          # seasons from config.yaml
    python scripts/rebuild_raw_from_mirrors.py ... --seasons 2223 2324 2425
    python scripts/rebuild_raw_from_mirrors.py ... --dry-run     # check, write nothing

Then, always:
    python scripts/verify_raw_data.py data/raw/*.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config

# Where each mirror keeps its files, relative to the clone's root.
ODDS_FILE = Path("data/processed/results_with_odds.csv")
SHOTS_DIR = Path("datasets/premier-league")

# The football-data columns we take from the SHOTS mirror, unchanged.
# Results come from here because this mirror keeps the original names.
SHOTS_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                 "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST"]

# Odds mirror name -> football-data code. Only what V8 reads, plus Pinnacle's
# opening price (PSH/D/A), which lets verify_raw_data.py prove the closing
# price is not a copy of the opening one.
ODDS_RENAMES = {
    "bet365_1x2_home": "B365H", "bet365_1x2_draw": "B365D", "bet365_1x2_away": "B365A",
    "pinnacle_1x2_home": "PSH", "pinnacle_1x2_draw": "PSD", "pinnacle_1x2_away": "PSA",
    "pinnacle_1x2_home_close": "PSCH",
    "pinnacle_1x2_draw_close": "PSCD",
    "pinnacle_1x2_away_close": "PSCA",
}


class RebuildError(Exception):
    """The mirrors cannot be turned into a trustworthy season file."""


def season_label(code):
    """'2324' -> '2023-24', the odds mirror's season format."""
    return f"20{code[:2]}-{code[2:]}"


def load_season(odds_root, shots_root, code):
    """Join one season from both mirrors and return it in football-data format.

    Raises RebuildError, never returns a partial file.
    """
    shots_path = shots_root / SHOTS_DIR / f"season-{code}.csv"
    if not shots_path.exists():
        raise RebuildError(f"{code}: shots mirror has no {shots_path}")
    shots = pd.read_csv(shots_path)
    missing = [c for c in SHOTS_COLUMNS if c not in shots.columns]
    if missing:
        raise RebuildError(f"{code}: shots mirror is missing {missing}")
    shots = shots[SHOTS_COLUMNS].copy()
    shots["Date"] = pd.to_datetime(shots["Date"])

    odds_all = pd.read_csv(odds_root / ODDS_FILE, low_memory=False)
    odds = odds_all[odds_all["season"].astype(str) == season_label(code)].copy()
    if odds.empty:
        raise RebuildError(f"{code}: odds mirror has no season {season_label(code)}")
    missing = [c for c in ODDS_RENAMES if c not in odds.columns]
    if missing:
        raise RebuildError(f"{code}: odds mirror is missing {missing}")
    odds["Date"] = pd.to_datetime(odds["date"])
    # Goals are kept from the odds mirror's sibling file, purely to audit the join.
    results = pd.read_csv(odds_root / ODDS_FILE.with_name("results.csv"))
    results = results[results["season"].astype(str) == season_label(code)]
    odds = odds.merge(results[["match_id", "fthg", "ftag"]], on="match_id", how="left")
    odds = odds.rename(columns={"home_team": "HomeTeam", "away_team": "AwayTeam"})
    odds = odds[["Date", "HomeTeam", "AwayTeam", "fthg", "ftag", *ODDS_RENAMES]]
    odds = odds.rename(columns=ODDS_RENAMES)

    # --- the join: every match must be in both mirrors ----------------------
    key = ["Date", "HomeTeam", "AwayTeam"]
    joined = shots.merge(odds, on=key, how="outer", indicator=True)
    orphans = joined[joined["_merge"] != "both"]
    if len(orphans):
        raise RebuildError(
            f"{code}: {len(orphans)} matches appear in only one mirror, e.g.\n"
            f"{orphans[key + ['_merge']].head().to_string(index=False)}")

    # --- the audit: two independent copies of the score must agree ----------
    disagree = joined[(joined["FTHG"] != joined["fthg"]) | (joined["FTAG"] != joined["ftag"])]
    if len(disagree):
        raise RebuildError(
            f"{code}: the mirrors disagree on {len(disagree)} scores, e.g.\n"
            f"{disagree[key + ['FTHG', 'fthg', 'FTAG', 'ftag']].head().to_string(index=False)}")

    # --- every price V8 bets into or measures against must be present -------
    needed = ["B365H", "B365D", "B365A", "PSCH", "PSCD", "PSCA"]
    gaps = joined[needed].isna().sum()
    if gaps.any():
        raise RebuildError(f"{code}: missing prices {gaps[gaps > 0].to_dict()}")

    out = joined.drop(columns=["fthg", "ftag", "_merge"]).sort_values(key)
    # football-data writes dd/mm/yyyy; V8's loader was built against that.
    out["Date"] = out["Date"].dt.strftime("%d/%m/%Y")
    out.insert(0, "Div", "E0")
    return out.reset_index(drop=True)


def summarise(code, frame):
    """One line a person can sanity-check by eye."""
    open_ = 1 / frame[["PSH", "PSD", "PSA"]]
    close = 1 / frame[["PSCH", "PSCD", "PSCA"]]
    copied = (frame[["PSH", "PSD", "PSA"]].values == frame[["PSCH", "PSCD", "PSCA"]].values).all(axis=1).mean()
    return (f"  E0_{code}: {len(frame)} matches | "
            f"home-win {(frame.FTR == 'H').mean():.1%} | "
            f"goals/match {(frame.FTHG + frame.FTAG).mean():.2f} | "
            f"Pinnacle book open {open_.sum(axis=1).mean():.3f} close {close.sum(axis=1).mean():.3f} | "
            f"closing identical to opening on {copied:.1%} of matches")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--odds", required=True, type=Path, help="clone of AnishKhetani/premier-league-data")
    parser.add_argument("--shots", required=True, type=Path, help="clone of datasets/football-datasets")
    parser.add_argument("--seasons", nargs="+", help="season codes, e.g. 2324 2425 (default: config.yaml)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: data.raw_dir)")
    parser.add_argument("--dry-run", action="store_true", help="run every check, write nothing")
    args = parser.parse_args()

    config = load_config()
    seasons = args.seasons or list(config["data"]["seasons"])
    out_dir = args.out or (REPO_ROOT / config["data"]["raw_dir"])

    print(f"Rebuilding {len(seasons)} season(s) from the mirrors\n")
    built = {}
    for code in seasons:
        try:
            built[code] = load_season(args.odds, args.shots, code)
        except RebuildError as err:
            # One bad season stops everything: half a dataset is a different dataset.
            print(f"REFUSED - {err}\nNothing was written.")
            return 1
        print(summarise(code, built[code]))

    if args.dry_run:
        print("\n--dry-run: every check passed, nothing written.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for code, frame in built.items():
        frame.to_csv(out_dir / f"E0_{code}.csv", index=False)
    print(f"\nWrote {len(built)} file(s) to {out_dir}")
    print("Next: python scripts/verify_raw_data.py data/raw/*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
