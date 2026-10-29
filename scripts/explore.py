#!/usr/bin/env python3
"""
Measure what the real data says, and write it to exploration.json.

The exploration notebook's numbers used to exist only as notebook output, which
had only ever been produced from the 12-match sample. This script makes them a
build artefact like numbers.json: one command regenerates them, the docs quote
the file, and tests/test_docs.py fails if a quoted figure disagrees with it.

Usage
-----
    python scripts/explore.py                       # headline config -> docs/charts/
    python scripts/explore.py --config configs/epl_2012_2025.yaml \\
        --out docs/charts/epl_2012_2025             # the 13-season long run
    python scripts/explore.py --out somewhere/else  # anywhere, any data

Refuses to write into a published folder (docs/charts or below) from the
bundled sample: the same rule make_charts.py has kept since 22 Oct 2026.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import exploration, logging_setup

PUBLISHED = (REPO_ROOT / "docs" / "charts").resolve()
FILENAME = "exploration.json"


def is_published(out_dir):
    """True for docs/charts and any folder inside it."""
    out_dir = Path(out_dir).resolve()
    return out_dir == PUBLISHED or PUBLISHED in out_dir.parents


def summary_lines(found):
    """The findings, readable in a terminal."""
    lines = [f"  {found['matches']:,} matches, {found['first_match']} to "
             f"{found['last_match']}", ""]
    lines.append("  1. Home advantage (95% Wilson intervals)")
    for key, label in (("home_win", "home win"), ("draw", "draw"), ("away_win", "away win")):
        r = found[key]
        lines.append(f"     {label:<9} {r['share']:6.1%}  [{r['low']:.1%}, {r['high']:.1%}]")
    lines.append(f"     goals per match: home {found['goals_home']:.2f}, away "
                 f"{found['goals_away']:.2f} ({found['goals_ratio']:.2f}x)")
    lines.append("")
    lines.append("  2. Low scores vs Poisson")
    lines.append("     score  per-match null (team strengths known)     league-mean null")
    for score, r in found["low_scores"].items():
        lines.append(f"     {score:<5}  {r['departure_per_match']:+7.1%}  "
                     f"[{r['departure_low']:+.1%}, {r['departure_high']:+.1%}]"
                     f"          {r['departure_league_mean']:+7.1%}")
    lines.append(f"     rho fitted per season: median {found['season_rho_median']:+.3f}, "
                 f"negative in {found['season_rho_negative']} of {len(found['season_rho'])} "
                 f"(sign test p = {found['season_rho_sign_test_p']:.3f})")
    lines.append("")
    lines.append("  3. The market")
    lines.append(f"     overround: bet-into {found['overround_bet_into']:.2%}, "
                 f"closing {found['overround_closing']:.2%}")
    lines.append(f"     closing-line calibration error (home win): "
                 f"{found['market_ece_home']:.4f} on all matches, "
                 f"{found['market_ece_home_later_half']:.4f} on the later half")
    lines.append("")
    lines.append("  4. Shots on target vs goals (95% Fisher intervals)")
    for key, label in (("shots_on_target_r_home", "home"), ("shots_on_target_r_away", "away")):
        r = found[key]
        lines.append(f"     {label:<5} r = {r['r']:.3f}  [{r['low']:.3f}, {r['high']:.3f}]")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "docs" / "charts"))
    args = parser.parse_args()

    config = load_config(args.config)
    fingerprint = logging_setup.run_fingerprint(config)
    print(f"\nV8 — exploration\n  fingerprint: {fingerprint}\n")

    table, real = exploration.load_canonical(config)
    if not real and is_published(args.out):
        # Decided before any measuring: a refusal should cost nothing.
        print("  REFUSED - the raw season files named in the config are not all")
        print("  in data/raw, so this would measure the 12-match bundled sample,")
        print("  and docs/charts is published. Rebuild the files, or pass --out")
        print("  to somewhere unpublished to look at the sample deliberately.")
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = exploration.findings(table, config)
    found["real_data"] = real
    found["run_fingerprint"] = fingerprint

    print("\n".join(summary_lines(found)))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / FILENAME).write_text(json.dumps(found, indent=2, default=float) + "\n")
    print(f"\n  wrote {out_dir / FILENAME}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
