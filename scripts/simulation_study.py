#!/usr/bin/env python3
"""
Does this backtester actually work? A simulation study.

THE QUESTION THIS ANSWERS
-------------------------
Every betting backtest reports a number. Almost none of them establish that the
machinery producing the number can tell signal from noise. A backtester that
reports "+4% CLV" is worthless unless you know it would have reported "0%" had
there been nothing there.

So this study does not ask *is the model profitable*. It asks a prior question:

    **When an edge exists, does this pipeline find it?
    When none exists, does it correctly report none?**

That is instrument validation, and it is the thing you must do before trusting
any backtest — including this project's own.

HOW
---
Three synthetic markets, each with a property known by construction because the
data-generating process is written here in this file:

  A. EFFICIENT      the bookmaker prices the true probabilities, plus a margin.
                    There is no edge to find. The pipeline should report none.

  B. BIASED         the bookmaker systematically underprices home advantage, and
                    the closing line corrects toward the truth. A model that
                    learns the real process should beat the opening price.
                    The pipeline should find positive CLV.

  C. NOISE MODEL    an efficient market, bet by a model whose probabilities are
                    random. A control: the measurement must not manufacture an
                    edge out of a model that knows nothing.

A pass is the pipeline agreeing with the truth in all three. A failure in B
means the instrument is blind; a failure in A or C means it hallucinates edges,
which is far worse.

WHAT THIS IS NOT
----------------
This is simulated data and it makes **no claim about real betting markets**.
Scenario B's edge exists because it was put there deliberately. Nothing here
says such an edge exists in practice — the real-data run is what addresses that,
and it is reported separately.

Usage
-----
    python scripts/simulation_study.py
    python scripts/simulation_study.py --seasons 1 --quick   # faster, noisier
    python scripts/simulation_study.py --scenario biased     # just one
"""
import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config, set_seed
from src import backtest, clv, staking, logging_setup

# This script owns one chart. tests/test_docs.py reads this manifest alongside
# make_charts.CHARTS, so the README may link it without being told twice.
CHARTS = {"detection": "simulation-detection-threshold.png"}

INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
BLUE, GOOD, BAD = "#2a78d6", "#0ca30c", "#d03b3b"

TEAMS = [f"Team {i:02d}" for i in range(1, 21)]

# The data-generating process, stated once so every scenario shares it.
BASE_GOALS = 1.35        # league-average goals for a neutral team at home
HOME_ADVANTAGE = 0.26    # in log space: exp(0.26) ~ 1.30x
RHO = -0.04              # Dixon-Coles low-score dependence
MAX_GOALS = 10


def true_probabilities(home_attack, home_defence, away_attack, away_defence):
    """The true 1X2 probabilities and score grid for one fixture.

    This is a bivariate Poisson with the Dixon-Coles tau correction — the same
    family the model fits. That is deliberate: the study is testing the
    BACKTESTER, not whether Dixon-Coles is the right model for football. Giving
    the model a correctly specified world isolates the question being asked.
    """
    lambda_home = BASE_GOALS * home_attack * away_defence * math.exp(HOME_ADVANTAGE)
    lambda_away = BASE_GOALS * away_attack * home_defence

    k = np.arange(0, MAX_GOALS + 1)
    factorials = np.array([math.factorial(int(i)) for i in k])
    p_home = np.exp(-lambda_home) * lambda_home ** k / factorials
    p_away = np.exp(-lambda_away) * lambda_away ** k / factorials
    grid = np.outer(p_home, p_away)

    for (x, y), tau in (((0, 0), 1 - lambda_home * lambda_away * RHO),
                        ((0, 1), 1 + lambda_home * RHO),
                        ((1, 0), 1 + lambda_away * RHO),
                        ((1, 1), 1 - RHO)):
        grid[x, y] *= tau

    grid = np.clip(grid, 1e-12, None)
    grid /= grid.sum()

    home_win = float(np.tril(grid, -1).sum())
    draw = float(np.trace(grid))
    return np.array([home_win, draw, 1.0 - home_win - draw]), grid


# No real bookmaker prices a single outcome beyond this, however certain the
# result looks: the liability is not worth the counterparty risk. Without the
# cap the simulator emits decimal odds below 1.00 on extreme fixtures — an
# impossible price, and one `validation.min_odds` would reject if it arrived in
# a real file. A simulator that produces data its own pipeline would refuse is
# not simulating the thing it claims to.
MAX_SINGLE_OUTCOME = 0.92
MIN_DECIMAL_ODDS = 1.01


def price(probabilities, margin, noise, rng):
    """Turn probabilities into decimal odds, with a bookmaker's margin.

    `noise` is the bookmaker's own estimation error. A real book is not a
    perfect estimate of the truth; it is a good estimate with a margin on top.
    """
    distorted = np.asarray(probabilities, dtype=float) * np.exp(rng.normal(0, noise, 3))
    distorted = distorted / distorted.sum()

    # Cap, then renormalise, then load the margin — in that order, so the book
    # still sums to 1 + margin after the cap has moved mass around.
    distorted = np.clip(distorted, 0.004, MAX_SINGLE_OUTCOME)
    distorted = distorted / distorted.sum() * (1.0 + margin)

    return np.maximum(np.round(1.0 / distorted, 3), MIN_DECIMAL_ODDS)


def build_market(bias, n_seasons, seed):
    """Generate a league whose mispricing is known by construction.

    `bias` is the factor the bookmaker applies to home attack strength when
    pricing: 1.00 is an efficient book, 0.80 badly underprices home teams.

    Returns (matches, mean_mispricing_in_probability_points) — the second value
    is what makes the result interpretable. "A bias factor of 0.90" means
    nothing to a reader; "the book is wrong by 2.4 probability points on the
    home win" is the number that can be compared against a real market.
    """
    rng = np.random.default_rng(seed)
    mispricing = []

    attack = dict(zip(TEAMS, rng.lognormal(0.0, 0.26, len(TEAMS))))
    defence = dict(zip(TEAMS, rng.lognormal(0.0, 0.22, len(TEAMS))))

    rows = []
    date = pd.Timestamp("2022-08-06")

    for season in range(n_seasons):
        fixtures = [(h, a) for h in TEAMS for a in TEAMS if h != a]
        rng.shuffle(fixtures)
        per_round = max(1, len(fixtures) // 38)

        for round_number in range(38):
            window = fixtures[round_number * per_round:(round_number + 1) * per_round]
            for home, away in window:
                truth, grid = true_probabilities(
                    attack[home], defence[home], attack[away], defence[away])

                flat = grid.ravel()
                pick = rng.choice(len(flat), p=flat / flat.sum())
                home_goals, away_goals = divmod(int(pick), grid.shape[1])

                # --- the scenario's one difference ------------------------
                if bias < 1.0:
                    # The bookmaker underprices home advantage: it quotes a
                    # world with less of it than really exists. The CLOSING
                    # line corrects most of the way back, which is exactly the
                    # pattern CLV is designed to detect. `bias` sets the size.
                    understated, _ = true_probabilities(
                        attack[home] * bias, defence[home],
                        attack[away], defence[away])
                    opening_truth = understated
                    closing_truth = 0.25 * understated + 0.75 * truth
                    mispricing.append(abs(float(truth[0] - understated[0])))
                else:
                    opening_truth = truth
                    closing_truth = truth
                    mispricing.append(0.0)

                opening = price(opening_truth, margin=0.055, noise=0.075, rng=rng)
                closing = price(closing_truth, margin=0.022, noise=0.030, rng=rng)

                rows.append({
                    "MatchID": f"S{season}R{round_number:02d}_{home[-2:]}{away[-2:]}",
                    "MatchDate": date + pd.Timedelta(days=7 * round_number
                                                     + int(rng.integers(0, 3))),
                    "League": "SIM", "Season": f"S{season}",
                    "HomeTeam": home, "AwayTeam": away,
                    "HomeGoals": int(home_goals), "AwayGoals": int(away_goals),
                    "Result": ("H" if home_goals > away_goals
                               else "A" if away_goals > home_goals else "D"),
                    "HomeShots": int(rng.poisson(12)), "AwayShots": int(rng.poisson(10)),
                    "HomeShotsOT": int(rng.poisson(5)), "AwayShotsOT": int(rng.poisson(4)),
                    "OddsH": opening[0], "OddsD": opening[1], "OddsA": opening[2],
                    "CloseH": closing[0], "CloseD": closing[1], "CloseA": closing[2],
                })

        date += pd.Timedelta(days=330)
        # Squads change between seasons.
        for team in TEAMS:
            attack[team] *= math.exp(rng.normal(0, 0.10))
            defence[team] *= math.exp(rng.normal(0, 0.09))

    return pd.DataFrame(rows), float(np.mean(mispricing))


def run_pipeline_scenario(matches, config):
    """Walk-forward the real pipeline over a simulated market."""
    walk = backtest.walk_forward(matches, config)
    bets = walk["bets"]
    if len(bets) == 0:
        return {"bets": 0, "note": "the pipeline placed no bets"}

    ledger, summary = staking.simulate_bankroll(bets, config)
    clv_result = clv.clv_summary(bets, config)
    points = clv_result["clv_probability_points"]

    return {
        "bets": int(summary["bets"]),
        "roi": float(summary["roi"]),
        "yield": float(summary["yield"]),
        "clv_points": float(points["estimate"]),
        "clv_low": float(points["lower"]),
        "clv_high": float(points["upper"]),
        "beat_rate": float(clv_result["beat_rate"]["estimate"]),
    }


def run_noise_control(matches, config, seed):
    """An efficient market bet by a model that knows nothing.

    Bypasses Dixon-Coles on purpose: the question is whether the measurement
    machinery can be made to report an edge by a model with no information. If
    random probabilities produce positive CLV, the measurement is broken.
    """
    rng = np.random.default_rng(seed + 99)
    frame = matches.copy()

    # Random probabilities, normalised. No information of any kind.
    draws = rng.dirichlet(np.ones(3) * 4.0, len(frame))
    frame["home_win"], frame["draw"], frame["away_win"] = draws.T

    bets = staking.find_value_bets(frame, config)
    if bets.empty or (bets["stake_fraction"] > 0).sum() == 0:
        return {"bets": 0, "note": "a model with no information found no value"}

    placed = bets[bets["stake_fraction"] > 0].copy()
    for label, column in (("closing_odds", None), ("closing_home", "CloseH"),
                          ("closing_draw", "CloseD"), ("closing_away", "CloseA")):
        if column:
            placed[label] = frame.set_index("MatchID").loc[
                placed["MatchID"], column].to_numpy()
    outcome_to_close = {"home_win": "CloseH", "draw": "CloseD", "away_win": "CloseA"}
    placed["closing_odds"] = [
        frame.set_index("MatchID").loc[row["MatchID"], outcome_to_close[row["outcome"]]]
        for _, row in placed.iterrows()
    ]

    ledger, summary = staking.simulate_bankroll(placed, config)
    clv_result = clv.clv_summary(placed, config)
    points = clv_result["clv_probability_points"]

    return {
        "bets": int(summary["bets"]),
        "roi": float(summary["roi"]),
        "yield": float(summary["yield"]),
        "clv_points": float(points["estimate"]),
        "clv_low": float(points["lower"]),
        "clv_high": float(points["upper"]),
        "beat_rate": float(clv_result["beat_rate"]["estimate"]),
    }


# The sweep. 1.00 is an efficient book and doubles as the control: if the
# pipeline reports an edge there, it is manufacturing one.
BIAS_LEVELS = [1.00, 0.95, 0.88, 0.80]


def significance(result):
    """Is the CLV significantly positive by this project's own rule?

    The whole bootstrap interval above zero — the same test applied to the
    real-data claim, used here where the right answer is known in advance.
    """
    if result.get("bets", 0) == 0:
        return False
    return result["clv_low"] > 0


def draw_detection_chart(rows, noise, threshold, args, out_dir):
    """CLV against the size of the mispricing the book was given.

    The chart answers the study's question at a glance: the line should rise
    with the planted edge, cross zero somewhere, and sit below zero where no
    edge exists. A flat line would mean the instrument is blind; a line above
    zero at the left-hand end would mean it invents edges.
    """
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT, "grid.color": GRID,
        "legend.frameon": False, "figure.dpi": 150, "font.size": 10,
    })

    usable = [r for r in rows if r.get("bets", 0)]
    if not usable:
        return

    x = np.array([r["mispricing_points"] for r in usable])
    y = np.array([r["clv_points"] for r in usable])
    low = np.array([r["clv_low"] for r in usable])
    high = np.array([r["clv_high"] for r in usable])

    fig, ax = plt.subplots(figsize=(7.4, 4.6), layout="constrained")

    ax.axhspan(ax.get_ylim()[0], 0, color=BAD, alpha=0.05, zorder=0)
    ax.errorbar(x, y, yerr=[y - low, high - y], fmt="o-", color=BLUE,
                linewidth=2.0, markersize=8, capsize=5, zorder=4,
                label="measured CLV, with 95% interval")
    ax.axhline(0, color=INK, linewidth=1.5, zorder=5)

    # Colour each marker by the verdict, so pass/fail reads without the legend.
    for row, xi, yi in zip(usable, x, y):
        ax.scatter([xi], [yi], s=90, zorder=6,
                   color=GOOD if row["detected"] else BAD,
                   edgecolor=SURFACE, linewidth=1.5)

    if threshold is not None:
        ax.axvline(threshold, color=GOOD, linestyle="--", linewidth=1.5,
                   zorder=3)
        ax.annotate(f"detects an edge\nfrom about {threshold:.1%}",
                    xy=(threshold, ax.get_ylim()[1]), xytext=(8, -8),
                    textcoords="offset points", fontsize=9, color=GOOD,
                    fontweight="bold", va="top")

    ax.annotate("no edge exists here —\nthe pipeline correctly reports none",
                xy=(x[0], y[0]), xytext=(14, -6), textcoords="offset points",
                fontsize=9, color=INK_SOFT, va="top")

    ax.set_xlabel("how wrong the bookmaker is  (probability points on the home win)")
    ax.set_ylabel("measured CLV")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], marker="o", color=GOOD, linestyle="",
                          markersize=9, label="edge detected"),
               plt.Line2D([], [], marker="o", color=BAD, linestyle="",
                          markersize=9, label="no edge detected")]
    ax.legend(handles=handles, loc="upper left")

    fig.suptitle("What this backtester can and cannot see",
                 fontsize=12, fontweight="bold", color=INK)
    ax.set_title(f"simulated markets · {args.seasons * 380} matches per point · "
                 f"the control with a random model sits at "
                 f"{noise.get('clv_points', float('nan')):+.1%}",
                 fontsize=9, color=INK_SOFT, pad=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CHARTS["detection"]
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    try:
        shown = path.relative_to(REPO_ROOT)
    except ValueError:
        shown = path
    print(f"  wrote {shown}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--seasons", type=int, default=2,
                        help="Seasons per level (default 2 = 760 matches)")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--out", default=None)
    parser.add_argument("--levels", type=float, action="append",
                        help="Override the bias levels swept")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer bootstrap resamples; noisier intervals")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    if args.quick:
        config.setdefault("backtest", {})["bootstrap_resamples"] = 400

    levels = args.levels or BIAS_LEVELS
    out_dir = Path(args.out) if args.out else REPO_ROOT / "docs" / "charts"

    print("\nV8 — simulation study: how big must an edge be before this")
    print("     backtester can see it?")
    print(f"  fingerprint: {logging_setup.run_fingerprint(config)}")
    print(f"  {args.seasons} season(s) per level, seed {args.seed}\n")

    rows = []
    for bias in levels:
        market, mispricing = build_market(bias, args.seasons, args.seed)
        label = ("efficient book" if bias >= 1.0
                 else f"book underprices home by {mispricing:.1%}")
        print(f"  {label}  (bias factor {bias:.2f})")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_pipeline_scenario(market, config)

        result["bias"] = bias
        result["mispricing_points"] = mispricing
        result["detected"] = significance(result)
        rows.append(result)

        if result.get("bets", 0):
            print(f"    bets {result['bets']:>5}   "
                  f"CLV {result['clv_points']:+.2%} "
                  f"[{result['clv_low']:+.2%}, {result['clv_high']:+.2%}]   "
                  f"{'DETECTED' if result['detected'] else 'not detected'}")
        else:
            print("    no bets placed")
        print()

    # --- the control: a model that knows nothing ------------------------
    print("  CONTROL — an efficient book bet by a model with random probabilities")
    efficient_market, _ = build_market(1.00, args.seasons, args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise = run_noise_control(efficient_market, config, args.seed)
    noise["detected"] = significance(noise)
    if noise.get("bets", 0):
        print(f"    bets {noise['bets']:>5}   "
              f"CLV {noise['clv_points']:+.2%} "
              f"[{noise['clv_low']:+.2%}, {noise['clv_high']:+.2%}]")
    print(f"    -> {'FAIL — invented an edge' if noise['detected'] else 'PASS — no edge reported'}\n")

    # --- the two things that must hold ----------------------------------
    efficient = next(r for r in rows if r["bias"] >= 1.0)
    no_false_positive = (not efficient["detected"]) and (not noise["detected"])
    detected = [r for r in rows if r["detected"]]
    threshold = min((r["mispricing_points"] for r in detected), default=None)

    print("  " + "-" * 66)
    print(f"  No false positive on an efficient book: "
          f"{'PASS' if no_false_positive else 'FAIL'}")
    if threshold is not None:
        print(f"  Detection threshold: the instrument finds an edge once the")
        print(f"  book is wrong by about {threshold:.1%} on the home win,")
        print(f"  given {args.seasons * 380} matches.")
    else:
        print(f"  Detection threshold: NOT REACHED at any level tested "
              f"(largest {max(r['mispricing_points'] for r in rows):.1%}).")
        print(f"  On {args.seasons * 380} matches this instrument cannot")
        print("  establish an edge of the sizes simulated.")
    print("  " + "-" * 66 + "\n")

    draw_detection_chart(rows, noise, threshold, args, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seasons_per_level": args.seasons,
        "matches_per_level": args.seasons * 380,
        "seed": args.seed,
        "run_fingerprint": logging_setup.run_fingerprint(config),
        "no_false_positive": bool(no_false_positive),
        "detection_threshold_points": threshold,
        "levels": rows,
        "noise_control": noise,
        "note": ("Simulated data. The edge in each biased level exists because "
                 "it was planted. This measures what the BACKTESTER can "
                 "detect, and makes no claim about real betting markets."),
    }
    path = out_dir / "simulation_study.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        shown = path.relative_to(REPO_ROOT)
    except ValueError:
        shown = path
    print(f"  wrote {shown}\n")

    return 0 if no_false_positive else 1


if __name__ == "__main__":
    raise SystemExit(main())
