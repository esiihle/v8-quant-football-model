#!/usr/bin/env python3
"""
Phase 8 — generate every chart the README reports, from one real pipeline run.

Why this is a script and not a notebook: the README makes numerical claims, and a
claim is only as good as the command that reproduces it. Running

    python scripts/make_charts.py

rebuilds every PNG in docs/charts/ *and* writes docs/charts/numbers.json, which
records the headline figures together with the Phase 7 run fingerprint. The
README quotes numbers from that file, and tests/test_docs.py checks the two
agree. If a rerun moves a number, the README is wrong and the test says so.

THE SAFETY RULE IN THIS FILE
----------------------------
Charts built from the 12-match bundled sample are meaningless, and a meaningless
chart that looks real is worse than no chart. So when real data is absent, every
chart is stamped ILLUSTRATIVE across its face and numbers.json records
`"real_data": false`. The stamp is deliberately hard to miss, because the mistake
it prevents — committing sample charts as results — is easy to make.

Usage
-----
    python scripts/make_charts.py                # real data from data/raw
    python scripts/make_charts.py --sample       # force the bundled sample
    python scripts/make_charts.py --only clv     # rebuild one chart
    python scripts/make_charts.py --list         # what it would write, and why
"""
import argparse
import json
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # no display in CI; write files directly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, set_seed
from src import (ingest, features, dixon_coles, calibrate, staking,
                 backtest, clv, metrics, logging_setup)


# --- The chart manifest ----------------------------------------------------
# One source of truth for what exists. The README links these filenames and
# tests/test_docs.py reads this dict, so a chart cannot be renamed in one place
# and left stale in the other.
CHARTS = {
    "clv_distribution": "clv-distribution.png",
    "reliability": "reliability-diagram.png",
    "bankroll": "bankroll-curve.png",
    "honesty": "in-sample-vs-walk-forward.png",
    "clv_by_edge": "clv-by-claimed-edge.png",
}

CHART_PURPOSE = {
    "clv_distribution": "the lead chart: per-bet CLV in probability points, "
                        "with the bootstrap interval on the mean",
    "reliability": "out-of-sample predicted vs realised probability",
    "bankroll": "the walk-forward bankroll curve with drawdown from peak",
    "honesty": "in-sample vs walk-forward ROI and yield — the 340-point swing",
    "clv_by_edge": "does the model's own claimed edge predict CLV?",
}


# --- Visual grammar --------------------------------------------------------
# Defined once. Two series maximum per chart, validated for colour-vision
# deficiency; polarity is carried by position relative to a zero line, with
# colour as a second channel and never the only one.
INK = "#0b0b0b"          # primary text
INK_SOFT = "#52514e"     # secondary text
GRID = "#dcdbd6"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"         # the model
ORANGE = "#eb6834"       # the market, or the misleading in-sample figure
GOOD = "#0ca30c"
BAD = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "figure.dpi": 150,
})


def new_figure(figsize):
    """Every figure is created here, with constrained layout.

    Constrained layout is what keeps the title, subtitle, legend and tick labels
    from colliding at any figure size. Hand-placing text in figure coordinates
    works at one size and silently breaks at the next, which is how a chart ends
    up with its subtitle printed through its title.
    """
    return plt.subplots(figsize=figsize, layout="constrained")


def finish(fig, ax_or_axes, path, illustrative, title=None, subtitle=None):
    """Title, stamp and save. Every chart goes through here.

    The title is the figure's suptitle and the subtitle is the axes title, so
    constrained layout stacks them instead of this code guessing offsets.
    """
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
    if subtitle:
        # Wrap rather than trust the figure to be wide enough. An unwrapped
        # subtitle runs off the right edge at narrower figure sizes, and the
        # reader only finds out by looking at the PNG.
        np.ravel(axes)[0].set_title("\n".join(textwrap.wrap(subtitle, 76)),
                                    fontsize=9, color=INK_SOFT,
                                    fontweight="normal", pad=8)
    if illustrative:
        for ax in np.ravel(axes):
            ax.text(0.5, 0.5, "ILLUSTRATIVE\nbundled sample — not results",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=15, fontweight="bold", color=BAD, alpha=0.30,
                    rotation=18, zorder=50)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"    wrote {show(path)}")


def show(path):
    """A path relative to the repo when it is inside it, absolute otherwise.

    --out can point anywhere, so relative_to() is not always defined.
    """
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def pct(value, places=1):
    """Format a fraction as a percentage, surviving NaN."""
    return "n/a" if value is None or not np.isfinite(value) else f"{value:+.{places}%}"


# --- Chart 1: CLV distribution (the lead chart) ----------------------------

def chart_clv_distribution(data, illustrative, out_dir):
    """Per-bet CLV in probability points.

    Probability points, not the odds ratio, because the ratio is asymmetric — a
    halved price scores -50%, a doubled one +100% — so its mean is carried by
    outliers. That bug is documented in methodology §6; this chart shows the
    measure that replaced it.
    """
    summary = data["clv"]
    per_bet = summary["per_bet"]
    values = per_bet["clv_probability_points"].dropna()

    fig, ax = new_figure((7.2, 4.2))
    if len(values) == 0:
        ax.text(0.5, 0.5, "no bets with closing odds", transform=ax.transAxes,
                ha="center", color=INK_SOFT)
        finish(fig, ax, out_dir / CHARTS["clv_distribution"], illustrative)
        return

    # Split the histogram at zero: beating the close is to the right of it.
    # Colour is the second channel; position relative to the zero line is first.
    edges = np.histogram_bin_edges(values, bins=min(30, max(8, len(values) // 10)))
    counts, edges = np.histogram(values, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges) * 0.92
    colours = [GOOD if c >= 0 else BAD for c in centres]
    ax.bar(centres, counts, width=widths, color=colours, alpha=0.75,
           edgecolor=SURFACE, linewidth=1.2)

    interval = summary["clv_probability_points"]
    mean = interval["estimate"]
    low, high = interval["lower"], interval["upper"]

    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.12)          # headroom so labels never sit on bars

    ax.axvline(0, color=INK, linewidth=1.6, zorder=5)
    ax.annotate("the closing line", xy=(0, top * 1.07), xytext=(7, 0),
                textcoords="offset points", fontsize=9, color=INK, va="center")

    # The interval. A 95% CI this tight renders as a hairline, so the number is
    # given in a label rather than left to the eye to measure off the band.
    ax.axvspan(low, high, color=BLUE, alpha=0.20, zorder=1)
    ax.axvline(mean, color=BLUE, linewidth=2.0, zorder=6)
    ax.annotate(f"mean {mean:+.2%}   95% CI [{low:+.2%}, {high:+.2%}]",
                xy=(mean, top * 1.07), xytext=(-10, 0),
                textcoords="offset points", fontsize=9, color=BLUE,
                fontweight="bold", va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                          edgecolor="none"))

    ax.set_xlabel("CLV per bet  (probability points vs the closing line)")
    ax.set_ylabel("bets")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)

    # A legend is present because two colours carry meaning.
    handles = [plt.Rectangle((0, 0), 1, 1, color=GOOD, alpha=0.75),
               plt.Rectangle((0, 0), 1, 1, color=BAD, alpha=0.75)]
    ax.legend(handles, ["beat the close", "lost to the close"],
              loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2)

    finish(fig, ax, out_dir / CHARTS["clv_distribution"], illustrative,
           title="Closing-line value: is the model beating the close?",
           subtitle=f"{int(summary['bets_with_closing_odds'])} bets with closing "
                    f"odds · beat rate {summary['beat_rate']['estimate']:.1%}")


# --- Chart 2: reliability diagram ------------------------------------------

def chart_reliability(data, illustrative, out_dir):
    """Predicted vs realised, out of sample. The 45° line is honesty."""
    table = data["reliability"]

    fig, ax = new_figure((5.6, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_SOFT, linewidth=1.4,
            label="perfect calibration", zorder=2)

    if len(table):
        # Marker area tracks how many matches are in the band, so a band of
        # three matches cannot be over-read.
        sizes = 30 + 220 * (table["matches"] / table["matches"].max())
        ax.scatter(table["predicted"], table["realised"], s=sizes,
                   color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=1.5,
                   zorder=4, label="model (out of sample)")
        ax.plot(table["predicted"], table["realised"], color=BLUE,
                linewidth=1.6, alpha=0.5, zorder=3)
        # Label only the fullest bands: a number on every point is noise.
        for _, row in table.nlargest(3, "matches").iterrows():
            ax.annotate(f"n={int(row['matches'])}",
                        (row["predicted"], row["realised"]),
                        xytext=(14, -16), textcoords="offset points",
                        fontsize=8, color=INK_SOFT,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE,
                                  edgecolor="none", alpha=0.85))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("realised frequency")
    ax.grid(alpha=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    ece = data["numbers"].get("ece")
    subtitle = (f"{data['numbers'].get('calibration_method', 'platt')} scaling · "
                f"expected calibration error {ece:.3f} · judged out of sample"
                ) if ece is not None else None
    finish(fig, ax, out_dir / CHARTS["reliability"], illustrative,
           title="Reliability: is the model honest when it says 70%?",
           subtitle=subtitle)


# --- Chart 3: bankroll curve -----------------------------------------------

def chart_bankroll(data, illustrative, out_dir):
    """The walk-forward bankroll, with drawdown from peak shaded.

    One axis, on purpose. Bankroll and drawdown are different scales, and a
    second y-axis would let the eye read a relationship that is not there — so
    the drawdown is drawn as the gap between the curve and its own peak.
    """
    ledger = data["walk_ledger"]

    fig, ax = new_figure((7.2, 4.2))
    if ledger is None or len(ledger) == 0:
        ax.text(0.5, 0.5, "no bets placed in the walk-forward run",
                transform=ax.transAxes, ha="center", color=INK_SOFT)
        finish(fig, ax, out_dir / CHARTS["bankroll"], illustrative)
        return

    curve = (ledger.groupby("MatchDate", sort=True)["bankroll_after_day"]
             .last().reset_index())
    start = data["numbers"].get("starting_bankroll", 1000.0)
    dates = pd.to_datetime(curve["MatchDate"])
    values = curve["bankroll_after_day"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(values)

    ax.fill_between(dates, values, peak, color=BAD, alpha=0.16,
                    label="drawdown from peak", zorder=2)
    ax.plot(dates, peak, color=BAD, linewidth=1.0, alpha=0.45, zorder=3)
    ax.plot(dates, values, color=BLUE, linewidth=2.0, label="bankroll", zorder=4)
    ax.axhline(start, color=INK_SOFT, linestyle="--", linewidth=1.2,
               label=f"starting bankroll ({start:,.0f})", zorder=1)

    # Mark the worst point, which is the number that actually ends bankrolls.
    # The label goes ABOVE the marker: the trough is by definition at the bottom
    # of the axes, and a label below it is a label off the figure.
    worst = int(np.argmin(values - peak))
    ax.scatter([dates.iloc[worst]], [values[worst]], s=70, color=BAD,
               edgecolor=SURFACE, linewidth=1.5, zorder=6)
    ax.annotate(f"max drawdown {data['numbers']['max_drawdown']:.1%}",
                (dates.iloc[worst], values[worst]),
                xytext=(12, 10), textcoords="offset points",
                fontsize=9, color=BAD, fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("bankroll")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    # Below the axes, not "best": `best` put the legend on top of the drawdown
    # fill, which is the one region of this chart a reader needs to see.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    # NOTE: no fig.autofmt_xdate() here. It calls subplots_adjust internally,
    # which switches constrained layout off and lets the subtitle print through
    # the title. The dates are horizontal and legible without it.

    finish(fig, ax, out_dir / CHARTS["bankroll"], illustrative,
           title="Walk-forward bankroll: fractional Kelly, out of sample",
           subtitle=f"{data['numbers']['walk_forward_bets']} bets · "
                    f"ROI {pct(data['numbers']['walk_forward_roi'])} · "
                    f"{data['numbers']['kelly_fraction']:.0%} Kelly")


# --- Chart 4: the honesty chart --------------------------------------------

def chart_honesty(data, illustrative, out_dir):
    """In-sample vs walk-forward, as two panels.

    This is the chart to lead an interview with. The left pair is what the model
    looked like when it was allowed to see the matches it bet on; the right pair
    is the forecast. Reporting the gap rather than tuning it away is the point.

    TWO PANELS, NOT TWO SERIES ON ONE AXIS. ROI is measured on the bankroll and
    yield on money staked; they routinely differ by an order of magnitude, so a
    shared axis flattens yield into an invisible sliver. Two measures of
    different scale get two scales — never two y-axes on one chart, which lets
    the eye read a relationship that is not there.
    """
    numbers = data["numbers"]
    panels = [
        ("ROI", "return on the bankroll",
         numbers.get("in_sample_roi"), numbers.get("walk_forward_roi")),
        ("Yield", "return on money staked",
         numbers.get("in_sample_yield"), numbers.get("walk_forward_yield")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.4), layout="constrained")

    for ax, (name, explanation, in_sample, walk) in zip(axes, panels):
        values = [in_sample if in_sample is not None else np.nan,
                  walk if walk is not None else np.nan]
        bars = ax.bar([0, 1], values, width=0.6, color=[ORANGE, BLUE],
                      zorder=3)

        for position, value in zip([0, 1], values):
            if np.isfinite(value):
                ax.annotate(pct(value), (position, value), ha="center",
                            va="bottom" if value >= 0 else "top",
                            xytext=(0, 5 if value >= 0 else -5),
                            textcoords="offset points", fontsize=11,
                            fontweight="bold", color=INK)

        ax.axhline(0, color=INK, linewidth=1.4, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["in-sample", "walk-forward"], fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{name} — {explanation}", fontsize=10, color=INK_SOFT,
                     pad=8)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)

        # Headroom for the labels, in both directions.
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            low, high = min(finite + [0.0]), max(finite + [0.0])
            span = (high - low) or 1.0
            ax.set_ylim(low - span * 0.18, high + span * 0.18)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(handles,
               ["in-sample — the model saw the matches it bet on",
                "walk-forward — the forecast"],
               loc="outside lower center", ncol=1, frameon=False, fontsize=9)

    finish(fig, axes, out_dir / CHARTS["honesty"], illustrative,
           title="The difference out-of-sample testing makes",
           subtitle=None)
    return


# --- Chart 5: CLV by claimed edge ------------------------------------------

def chart_clv_by_edge(data, illustrative, out_dir):
    """If the model's confidence were informative, CLV would rise with it."""
    table = data["clv_bands"]

    fig, ax = new_figure((7.2, 4.2))
    if table is None or len(table) == 0:
        ax.text(0.5, 0.5, "not enough bets to band", transform=ax.transAxes,
                ha="center", color=INK_SOFT)
        finish(fig, ax, out_dir / CHARTS["clv_by_edge"], illustrative)
        return

    # Probability points where available — the authoritative measure, and the
    # same units as the lead chart so the two can be read together.
    column = ("mean_clv_points" if "mean_clv_points" in table.columns
              else "mean_clv_ratio")
    values = table[column].to_numpy(dtype=float)
    counts = table["bets"].to_numpy()

    # Label each band by the edge the model actually claimed in it. "Band 3"
    # tells a reader nothing; "+4.1% claimed" tells them what the model believed.
    if "mean_claimed_edge" in table.columns:
        labels = [f"{v:+.1%}" for v in table["mean_claimed_edge"]]
    else:
        labels = [str(v) for v in table["band"]]

    colours = [GOOD if v >= 0 else BAD for v in values]
    ax.bar(np.arange(len(values)), values, width=0.68, color=colours,
           alpha=0.85, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.axhline(0, color=INK, linewidth=1.4, zorder=4)

    # Headroom, so a near-zero bar is still visibly a bar and its label fits.
    span = max(np.nanmax(np.abs(values)), 1e-4)
    ax.set_ylim(min(0, np.nanmin(values)) - span * 0.25,
                max(0, np.nanmax(values)) + span * 0.25)

    for index, (value, count) in enumerate(zip(values, counts)):
        # The count sits on the far side of the baseline from the bar, so it
        # never lands inside the bar it describes.
        ax.annotate(f"n={int(count)}", (index, 0),
                    xytext=(0, 7 if value < 0 else -15),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=INK_SOFT)
        ax.annotate(f"{value:+.2%}", (index, value), ha="center",
                    va="bottom" if value >= 0 else "top",
                    xytext=(0, 4 if value >= 0 else -4),
                    textcoords="offset points", fontsize=9,
                    fontweight="bold", color=INK)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("mean edge the model claimed in the band  (low to high)")
    ax.set_ylabel("mean CLV (probability points)"
                  if column == "mean_clv_points" else "mean CLV (odds ratio)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.1%}")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)

    finish(fig, ax, out_dir / CHARTS["clv_by_edge"], illustrative,
           title="Does the model's own confidence predict CLV?",
           subtitle="a rising pattern would mean the model knows when it is "
                    "right; a flat one means it does not")


BUILDERS = {
    "clv_distribution": chart_clv_distribution,
    "reliability": chart_reliability,
    "bankroll": chart_bankroll,
    "honesty": chart_honesty,
    "clv_by_edge": chart_clv_by_edge,
}


# --- Running the pipeline to get the numbers -------------------------------

def gather(config, use_sample, verbose=False):
    """Run the stages the charts need and collect everything in one dict."""
    numbers = {}

    # --- data ---
    print("  loading data")
    expected = [f for f in ingest.expected_raw_files(config) if f["exists"]]
    if use_sample or not expected:
        raw = ingest.load_matches(REPO_ROOT / config["data"]["sample_path"], config)
        illustrative = True
        if not use_sample:
            print("    NO RAW DATA FOUND — falling back to the bundled sample.")
            for entry in ingest.expected_raw_files(config):
                print(f"      download {entry['url']}")
                print(f"           to {entry['path']}")
    else:
        raw = ingest.load_many(expected, config)
        illustrative = False
    matches, _ = ingest.run_ingest(raw, config, save=False)
    numbers["matches"] = int(len(matches))
    numbers["first_match"] = str(pd.to_datetime(matches["MatchDate"]).min().date())
    numbers["last_match"] = str(pd.to_datetime(matches["MatchDate"]).max().date())

    # --- features, fit, calibration (for the reliability diagram) ---
    print("  features and model fit")
    feature_matrix, _ = features.build_features(matches, config)
    blended, _ = features.blend_goals(matches, matches, config)
    seed = features.team_strengths(
        blended, as_of=pd.to_datetime(matches["MatchDate"]).max() + pd.Timedelta(days=1),
        config=config)
    model = dixon_coles.fit(matches, config, seed_strengths=seed)
    numbers["rho"] = float(model["rho"])
    numbers["home_advantage"] = float(model["home_advantage"])
    numbers["effective_sample_size"] = float(model["effective_sample_size"])

    priceable = feature_matrix[feature_matrix["HasSufficientHistory"]].copy()
    predictions = dixon_coles.predict(model, priceable, config)
    predictions["MatchDate"] = priceable["MatchDate"].to_numpy()

    reliability = pd.DataFrame()
    in_sample_summary = None
    try:
        print("  calibration")
        result = calibrate.evaluate(predictions, priceable["Result"].to_numpy(), config)
        numbers["calibration_method"] = result["calibration"]["method"]
        evaluated = result["calibrated_predictions"]
        # Reliability on the home-win probability: the band structure is clearest
        # on one outcome, and home win is the one with the widest range.
        realised = (priceable.set_index("MatchID")
                    .loc[evaluated["MatchID"], "Result"].to_numpy() == "H").astype(float)
        reliability = calibrate.reliability_table(
            evaluated["home_win"].to_numpy(), realised, n_bins=10)
        numbers["ece"] = float(calibrate.expected_calibration_error(
            evaluated["home_win"].to_numpy(), realised, n_bins=10))

        # --- in-sample staking, which is the number the honesty chart contrasts
        print("  in-sample staking (for the contrast)")
        staking_input = evaluated.copy()
        for column in ("OddsH", "OddsD", "OddsA", "Result"):
            staking_input[column] = priceable.set_index("MatchID").loc[
                staking_input["MatchID"], column].to_numpy()
        bets = staking.find_value_bets(staking_input, config)
        _, in_sample_summary = staking.simulate_bankroll(bets, config)
        numbers["in_sample_roi"] = in_sample_summary["roi"]
        numbers["in_sample_yield"] = in_sample_summary["yield"]
        numbers["in_sample_bets"] = in_sample_summary["bets"]
    except (calibrate.CalibrationError, staking.StakingError) as error:
        print(f"    cannot calibrate or stake in sample: {error}")
        numbers["ece"] = None

    # --- walk-forward, which is the honest number ---
    walk_ledger, clv_summary, clv_bands = None, None, None
    try:
        print("  walk-forward backtest (the slow part)")
        walk = backtest.walk_forward(matches, config, progress=verbose)
        bet_history = walk["bets"]
        if len(bet_history):
            walk_ledger, walk_summary = staking.simulate_bankroll(bet_history, config)
            numbers["walk_forward_bets"] = walk_summary["bets"]
            numbers["walk_forward_roi"] = walk_summary["roi"]
            numbers["walk_forward_yield"] = walk_summary["yield"]
            numbers["hit_rate"] = walk_summary["hit_rate"]
            numbers["max_drawdown"] = walk_summary["max_drawdown"]
            numbers["starting_bankroll"] = walk_summary["starting_bankroll"]
            numbers["windows"] = int(len(walk["windows"]))

            clv_summary = clv.clv_summary(bet_history, config)
            points = clv_summary["clv_probability_points"]
            numbers["clv_points_mean"] = points["estimate"]
            numbers["clv_points_low"] = points["lower"]
            numbers["clv_points_high"] = points["upper"]
            numbers["clv_ratio_mean"] = clv_summary["clv_ratio"]["estimate"]
            numbers["clv_ratio_low"] = clv_summary["clv_ratio"]["lower"]
            numbers["clv_ratio_high"] = clv_summary["clv_ratio"]["upper"]
            numbers["beat_rate"] = clv_summary["beat_rate"]["estimate"]
            numbers["clv_headline"] = clv_summary["headline_measure"]
            numbers["clv_measures_disagree"] = bool(clv_summary["measures_disagree"])
            try:
                clv_bands = clv.clv_by_band(bet_history, config=config)
            except Exception as error:          # banding needs spread in `edge`
                print(f"    cannot band CLV: {error}")
        else:
            print("    the walk-forward run placed no bets at these settings")
    except backtest.BacktestError as error:
        print(f"    cannot back-test: {error}")

    numbers["kelly_fraction"] = float(config["staking"]["kelly_fraction"])
    numbers["real_data"] = not illustrative
    numbers["run_fingerprint"] = logging_setup.run_fingerprint(config)

    return {
        "numbers": numbers,
        "reliability": reliability,
        "walk_ledger": walk_ledger,
        "clv": clv_summary,
        "clv_bands": clv_bands,
        "illustrative": illustrative,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--sample", action="store_true",
                        help="Force the bundled sample (charts get stamped)")
    parser.add_argument("--only", choices=sorted(CHARTS), action="append",
                        help="Build only these charts (repeatable)")
    parser.add_argument("--list", action="store_true",
                        help="Show what would be written, and why, then exit")
    parser.add_argument("--out", default=None,
                        help="Output directory (default docs/charts)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nCharts this script writes into docs/charts/:\n")
        for key, filename in CHARTS.items():
            print(f"  {filename:<34} {CHART_PURPOSE[key]}")
        print("\nPlus numbers.json — the headline figures and the run fingerprint,"
              "\nwhich tests/test_docs.py checks the README against.\n")
        return 0

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    out_dir = Path(args.out) if args.out else REPO_ROOT / "docs" / "charts"

    print("\nV8 — building the README charts")
    print(f"  fingerprint: {logging_setup.run_fingerprint(config)}\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = gather(config, use_sample=args.sample, verbose=args.verbose)

    print("\n  drawing")
    wanted = args.only or list(CHARTS)
    for key in wanted:
        if data["clv"] is None and key in ("clv_distribution", "clv_by_edge"):
            print(f"    skipping {CHARTS[key]} — no CLV data from this run")
            continue
        BUILDERS[key](data, data["illustrative"], out_dir)

    # numbers.json is the contract between this run and the README.
    numbers_path = out_dir / "numbers.json"
    numbers_path.parent.mkdir(parents=True, exist_ok=True)
    numbers_path.write_text(json.dumps(data["numbers"], indent=2, default=str) + "\n")
    print(f"    wrote {show(numbers_path)}")

    print()
    if data["illustrative"]:
        print("  THESE CHARTS ARE STAMPED ILLUSTRATIVE. Do not commit them as")
        print("  results — download the raw data and run again.")
    else:
        print("  Real data. Commit docs/charts/ and run pytest tests/test_docs.py")
        print("  to check the README's numbers against numbers.json.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
