r"""
Stage 6 — Closing-line value (CLV).  [Built in Phase 6, 18 Oct 2026]

**The headline metric of this project.**

CLV compares the odds we took against the **closing** odds — the last price
before kick-off, after every injury report, team sheet and professional opinion
has been absorbed. The closing line is the market's most efficient estimate of a
match's probabilities, and that is precisely why beating it means something.

Why CLV rather than ROI
-----------------------
ROI is what you actually earn, so it is the number that matters. It is also
almost useless for *deciding whether a model works*, because its variance is
enormous. A 3% edge betting into 3.0 odds has a per-bet standard deviation of
roughly 1.4 units; establishing that edge is real to any reasonable confidence
takes **thousands** of bets.

CLV does not depend on results at all. Every bet contributes a measurement of
whether the price moved your way, not whether the ball went in. That removes the
single largest source of noise, so a few hundred bets say more about edge than a
few thousand bets of ROI do.

The logic: if you consistently take prices the market later agrees were too
generous, you are finding information before the market does. That is what an
edge *is*. Results then follow, with a lag and a lot of noise.

Three ways to express it, and what each is for
----------------------------------------------
1. **Odds ratio**, :math:`O_t / O_c - 1`. Took 3.50 where the close was 3.00:
   +16.7%. Intuitive and the usual way it is quoted.
2. **Probability points**, de-vigged closing probability minus de-vigged taken
   probability. The honest version for averaging, because odds ratios are
   asymmetric: a price moving 2.00 → 1.80 and 1.80 → 2.00 are the same size of
   move in probability and look very different as ratios.
3. **Beat rate**, the share of bets where the price moved your way. Robust to
   outliers, and the easiest figure to defend.

All three are reported, with **bootstrap confidence intervals** rather than bare
point estimates — a mean CLV of +1.5% means nothing without knowing whether the
interval includes zero.

A caution that belongs in the open
----------------------------------
CLV is a measure of *beating the closing price*, which is not identical to
profit. Two gaps worth stating:

* the margin still has to be paid, so a small positive CLV can coexist with a
  negative expectation;
* at a real bookmaker, consistently beating the close is how accounts get limited
  or closed, which is a practical constraint rather than a modelling one.

See docs/methodology.md section 6.3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.staking import market_probabilities

# The columns a bet record needs for CLV to be computable.
REQUIRED = ["odds", "closing_odds", "outcome"]

OUTCOME_ORDER = ["home_win", "draw", "away_win"]


class CLVError(Exception):
    """Raised when closing-line value cannot be computed."""


# ===========================================================================
# Per-bet CLV
# ===========================================================================

def clv_per_bet(bets, devig_method="shin"):
    """Compute all three CLV expressions for every bet.

    Parameters
    ----------
    bets : pandas.DataFrame
        Needs ``odds`` (taken), ``closing_odds``, ``outcome``, and — for the
        probability-point version — the full closing market
        (``closing_home``, ``closing_draw``, ``closing_away``) so the closing
        line can be de-vigged properly. Without the full market, only the odds
        ratio and beat rate are computed.

    Returns
    -------
    pandas.DataFrame
        A copy with ``clv_ratio``, ``clv_probability_points`` (where computable)
        and ``beat_close``.
    """
    missing = [c for c in REQUIRED if c not in bets.columns]
    if missing:
        raise CLVError(f"CLV needs columns {missing}")

    out = bets.copy()
    taken = out["odds"].to_numpy(dtype=float)
    closing = out["closing_odds"].to_numpy(dtype=float)

    usable = np.isfinite(closing) & (closing > 1.0) & np.isfinite(taken)
    if not usable.any():
        raise CLVError(
            "No bet has usable closing odds, so CLV cannot be computed. "
            "Check that the data source carries closing prices."
        )

    # 1. Odds ratio: took 3.50 where the close was 3.00 -> +16.7%.
    out["clv_ratio"] = np.where(usable, taken / closing - 1.0, np.nan)

    # 3. Beat rate: did the price move our way at all?
    out["beat_close"] = np.where(usable, taken > closing, np.nan)

    # 2. Probability points — the version worth averaging. Needs the whole
    # closing market to de-vig, because a single closing price carries no
    # information about the closing margin.
    full_market = all(c in out.columns for c in
                      ("closing_home", "closing_draw", "closing_away"))
    if full_market:
        points = np.full(len(out), np.nan)
        for position, (_, bet) in enumerate(out.iterrows()):
            market = np.array([bet["closing_home"], bet["closing_draw"],
                               bet["closing_away"]], dtype=float)
            if not (np.all(np.isfinite(market)) and np.all(market > 1.0)):
                continue
            fair_closing, _ = market_probabilities(market, devig_method)
            index = OUTCOME_ORDER.index(bet["outcome"])

            # We beat the close when the closing market thinks the outcome MORE
            # likely than the price we took implied — i.e. we were paid as though
            # it were less likely than the market finally decided.
            taken_implied = 1.0 / float(bet["odds"])
            points[position] = float(fair_closing[index]) - taken_implied
        out["clv_probability_points"] = points

    return out


# ===========================================================================
# Summary with confidence intervals
# ===========================================================================

def bootstrap_interval(values, statistic=np.mean, n_resamples=2000,
                       confidence=0.95, seed=42):
    """Percentile bootstrap interval for a statistic.

    Resampling rather than a normal approximation, because CLV distributions are
    skewed — a handful of large price moves dominate the tail, and a symmetric
    interval would misrepresent them.

    Returns
    -------
    dict
        ``estimate``, ``lower``, ``upper``, ``n``.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {"estimate": np.nan, "lower": np.nan, "upper": np.nan, "n": 0}
    if len(values) == 1:
        single = float(statistic(values))
        return {"estimate": single, "lower": single, "upper": single, "n": 1}

    rng = np.random.default_rng(seed)
    draws = np.array([
        statistic(values[rng.integers(0, len(values), len(values))])
        for _ in range(n_resamples)
    ])

    tail = (1 - confidence) / 2
    return {
        "estimate": float(statistic(values)),
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1 - tail)),
        "n": int(len(values)),
    }


def clv_summary(bets, config=None):
    """The headline CLV figures, every one with a confidence interval.

    Returns
    -------
    dict
        Mean CLV in both expressions, the beat rate, the distribution, and a
        verdict on whether the interval excludes zero.
    """
    config = config or {}
    cfg = config.get("backtest", {})
    n_resamples = int(cfg.get("bootstrap_resamples", 2000))
    confidence = float(cfg.get("confidence_level", 0.95))
    seed = int(config.get("seed", 42))

    with_clv = clv_per_bet(
        bets, (config.get("staking", {}) or {}).get("devig_method", "shin"))

    ratio = bootstrap_interval(with_clv["clv_ratio"], np.mean, n_resamples,
                               confidence, seed)
    beat = bootstrap_interval(with_clv["beat_close"].astype(float), np.mean,
                              n_resamples, confidence, seed)

    points = None
    if "clv_probability_points" in with_clv.columns:
        points = bootstrap_interval(with_clv["clv_probability_points"], np.mean,
                                    n_resamples, confidence, seed)

    # --- The verdict ------------------------------------------------------
    # Probability points is AUTHORITATIVE where available. The odds ratio is
    # asymmetric — a price halving scores -50% while a doubling scores +100% —
    # so its mean is pulled upward by its own right tail. On the first real
    # walk-forward run the two measures disagreed and both intervals excluded
    # zero: ratio +1.82% [+0.81%, +3.00%], points -0.67% [-0.87%, -0.47%]. The
    # ratio was being carried by a single +108% outlier while the model was in
    # fact losing to the close, and the beat rate (47.5%) agreed with points.
    #
    # So the summary reports BOTH, flags when they disagree, and never lets the
    # flattering one stand as the verdict.
    headline = points if points is not None else ratio
    headline_name = ("probability points" if points is not None
                     else "odds ratio")
    significant = bool(headline["lower"] > 0) if headline["n"] > 1 else False

    def excludes_zero(stats):
        if stats is None or stats["n"] < 2:
            return None
        if stats["lower"] > 0:
            return "positive"
        if stats["upper"] < 0:
            return "negative"
        return "inconclusive"

    ratio_verdict = excludes_zero(ratio)
    points_verdict = excludes_zero(points)
    measures_disagree = bool(
        ratio_verdict in ("positive", "negative")
        and points_verdict in ("positive", "negative")
        and ratio_verdict != points_verdict
    )

    distribution = with_clv["clv_ratio"].dropna()
    return {
        "bets": int(len(with_clv)),
        "bets_with_closing_odds": int(with_clv["clv_ratio"].notna().sum()),
        "clv_ratio": ratio,
        "clv_probability_points": points,
        "beat_rate": beat,
        "distribution": {
            "p10": float(distribution.quantile(0.10)) if len(distribution) else np.nan,
            "median": float(distribution.median()) if len(distribution) else np.nan,
            "p90": float(distribution.quantile(0.90)) if len(distribution) else np.nan,
            "worst": float(distribution.min()) if len(distribution) else np.nan,
            "best": float(distribution.max()) if len(distribution) else np.nan,
        },
        "confidence_level": confidence,
        "positive_and_significant": significant,
        "headline_measure": headline_name,
        "ratio_verdict": ratio_verdict,
        "points_verdict": points_verdict,
        "measures_disagree": measures_disagree,
        "per_bet": with_clv,
    }


def clv_by_band(bets, column="edge", n_bands=5, config=None):
    """CLV broken down by the model's own claimed edge.

    The sharpest diagnostic available. If the model has real information, the
    bets where it claimed the *most* edge should show the *most* CLV. A flat
    profile means the edge estimate carries no signal even if the average CLV is
    positive — the model is right on average and cannot tell you where.
    """
    with_clv = clv_per_bet(
        bets, (config or {}).get("staking", {}).get("devig_method", "shin"))

    if column not in with_clv.columns:
        raise CLVError(f"Cannot band by '{column}': column not present.")

    with_clv["band"] = pd.qcut(with_clv[column], q=n_bands, duplicates="drop",
                               labels=False)

    aggregations = {
        "bets": ("clv_ratio", "size"),
        "mean_claimed_edge": (column, "mean"),
        "mean_clv_ratio": ("clv_ratio", "mean"),
        "beat_rate": ("beat_close", "mean"),
    }
    # Probability points is the authoritative measure (see clv_summary), so it
    # belongs in the band table too. Until 21 Oct 2026 this function reported
    # only the odds ratio — the very measure the 18 Oct bug showed can read
    # positive while the model is losing to the close. A diagnostic that can
    # only be read in the misleading units is not a diagnostic.
    if "clv_probability_points" in with_clv.columns:
        aggregations["mean_clv_points"] = ("clv_probability_points", "mean")

    table = with_clv.groupby("band", observed=True).agg(**aggregations).reset_index()
    table["band"] = table["band"] + 1

    # Put the authoritative measure beside the ratio, in reading order.
    if "mean_clv_points" in table.columns:
        order = ["band", "bets", "mean_claimed_edge", "mean_clv_points",
                 "mean_clv_ratio", "beat_rate"]
        table = table[order]
    return table


def clv_report(summary, band_table=None):
    """The CLV report — the lead chart's data and the project's headline claim."""
    confidence = summary["confidence_level"]
    lines = ["Closing-line value (CLV)", "-" * 24]
    lines.append(f"  {'bets':<34} {summary['bets']:>10,}")
    lines.append(f"  {'with usable closing odds':<34} "
                 f"{summary['bets_with_closing_odds']:>10,}")

    def interval(label, stats, as_percent=True):
        if stats is None or stats["n"] == 0:
            return f"  {label:<34} {'n/a':>10}"
        if as_percent:
            return (f"  {label:<34} {stats['estimate']:>+9.2%}"
                    f"   [{stats['lower']:+.2%}, {stats['upper']:+.2%}]")
        return (f"  {label:<34} {stats['estimate']:>10.2%}"
                f"   [{stats['lower']:.2%}, {stats['upper']:.2%}]")

    lines.append("")
    lines.append(f"  Headline figures, with {confidence:.0%} bootstrap intervals:")
    lines.append(interval("mean CLV (odds ratio)", summary["clv_ratio"]))
    if summary["clv_probability_points"]:
        lines.append(interval("mean CLV (probability points)",
                              summary["clv_probability_points"]))
    lines.append(interval("beat the close", summary["beat_rate"], as_percent=False))

    distribution = summary["distribution"]
    lines.append("")
    lines.append("  CLV distribution (odds ratio):")
    lines.append(f"    worst {distribution['worst']:+.2%}  |  "
                 f"p10 {distribution['p10']:+.2%}  |  "
                 f"median {distribution['median']:+.2%}  |  "
                 f"p90 {distribution['p90']:+.2%}  |  "
                 f"best {distribution['best']:+.2%}")

    lines.append("")
    headline = summary["headline_measure"]

    if summary.get("measures_disagree"):
        lines.append("  THE TWO CLV MEASURES DISAGREE, and that is itself the "
                     "finding.")
        lines.append(f"    odds ratio:         {summary['ratio_verdict']}")
        lines.append(f"    probability points: {summary['points_verdict']}")
        lines.append("")
        lines.append("  Probability points is authoritative. The odds ratio is "
                     "ASYMMETRIC — a price")
        lines.append("  halving scores -50% while a doubling scores +100% — so "
                     "its mean is pulled")
        lines.append("  upward by its own right tail. Check the distribution "
                     "above: if the best")
        lines.append("  bet dwarfs the worst, the ratio mean is being carried "
                     "by a handful of")
        lines.append("  price moves and is not describing the typical bet.")
        lines.append("")
        lines.append("  The beat rate is the tie-breaker, and it needs no "
                     "averaging at all.")

    elif summary["positive_and_significant"]:
        lines.append(f"  POSITIVE CLV on {headline}, interval excludes zero.")
        lines.append("  This is the strongest evidence the project can produce "
                     "that the edge is real:")
        lines.append("  the model is taking prices the market later agreed were "
                     "too generous.")

    elif summary["points_verdict"] == "negative" or (
            summary["points_verdict"] is None
            and summary["ratio_verdict"] == "negative"):
        lines.append(f"  CLV is NEGATIVE on {headline}, and the interval "
                     "excludes zero.")
        lines.append("  The model is taking prices the market later moves AWAY "
                     "from, so there is")
        lines.append("  no edge — whatever the ROI happens to say in either "
                     "direction.")

    else:
        lines.append(f"  CLV on {headline} is NOT distinguishable from zero: "
                     "the interval")
        lines.append("  includes it. That is not evidence of an edge. More bets "
                     "are needed, and")
        lines.append("  reporting the point estimate without this caveat would "
                     "be dishonest.")

    beat = summary["beat_rate"]
    if beat["n"] > 1:
        lines.append("")
        if beat["lower"] > 0.5:
            lines.append(f"  Beat rate {beat['estimate']:.1%} is above 50% with "
                         "the interval clear of it.")
        elif beat["upper"] < 0.5:
            lines.append(f"  Beat rate {beat['estimate']:.1%} is BELOW 50% with "
                         "the interval clear of it:")
            lines.append("  the price moves against these bets more often than "
                         "for them.")
        else:
            lines.append(f"  Beat rate {beat['estimate']:.1%} is not "
                         "distinguishable from a coin flip.")

    if band_table is not None and not band_table.empty:
        lines.append("")
        lines.append("  CLV by the model's own claimed edge:")
        lines.append("    " + band_table.to_string(
            index=False, float_format=lambda v: f"{v:.4f}").replace("\n", "\n    "))
        lines.append("    (a rising profile means the edge ESTIMATE carries "
                     "signal, not just the")
        lines.append("     average; a flat one means the model cannot tell you "
                     "WHERE its edge is)")

    lines.append("")
    lines.append("  Note: CLV measures beating the closing price, which is not "
                 "the same as profit.")
    lines.append("  The margin still has to be paid, so small positive CLV can "
                 "coexist with a")
    lines.append("  negative expectation.")
    return "\n".join(lines)


def compute_clv(bets, closing_odds=None):
    """Backwards-compatible entry point from the Phase 0 scaffold."""
    if closing_odds is not None:
        bets = bets.copy()
        bets["closing_odds"] = closing_odds
    return clv_summary(bets)
