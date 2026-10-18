r"""
Metrics.  [Proper scoring rules built in Phase 4, 13 Oct 2026.
           Betting metrics follow in Phase 6.]

Two families of number live here, and they answer different questions:

* **Proper scoring rules** (this phase) — is the probability any good?
* **Betting performance** (Phase 6) — did acting on it make money?

A scoring rule is called *proper* when the only way to minimise it is to report
your true beliefs. That matters more than it sounds: an improper rule can be
gamed by shading predictions toward the extremes, so a model could score well by
being confidently wrong in a particular pattern. Brier and log-loss cannot be
gamed that way, which is why both are reported.

Why both, when either would do
------------------------------
They punish differently, and the difference is informative.

* **Brier** is a squared error, so it is bounded and forgiving. Being wrong at
  99% confidence costs a little less than twice being wrong at 70%.
* **Log-loss** is unbounded, so a confident miss is brutal: a 1% probability on
  something that happens costs about 4.6, where Brier charges under 1.

A model that looks fine on Brier and terrible on log-loss is making occasional
catastrophic calls. For a staking system that distinction is the whole game,
because one confident wrong bet at full Kelly can end a bankroll.

See docs/methodology.md sections 4 and 6.4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Probabilities are clipped away from 0 and 1 before any logarithm. Without
# this, one prediction of exactly 0 on an event that happened makes log-loss
# infinite and the whole evaluation collapses to a single row.
LOG_LOSS_CLIP = 1e-15


def brier_score(probabilities, outcomes):
    """Mean squared error between predicted probability and realised outcome.

    .. math:: \\text{Brier} = \\frac{1}{N}\\sum_i (p_i - o_i)^2

    with ``o_i`` 1 if the event happened and 0 otherwise. Lower is better. A
    useless model that always predicts the base rate ``b`` scores ``b(1-b)``,
    which is the number any result should be compared against.
    """
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - o) ** 2))


def log_loss(probabilities, outcomes):
    """Negative log-likelihood per observation. Lower is better.

    .. math:: -\\frac{1}{N}\\sum_i \\big[o_i \\ln p_i + (1-o_i)\\ln(1-p_i)\\big]
    """
    p = np.clip(np.asarray(probabilities, dtype=float), LOG_LOSS_CLIP,
                1 - LOG_LOSS_CLIP)
    o = np.asarray(outcomes, dtype=float)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


# The probability columns a model produces are not named after the labels the
# data records — the model says "home_win" while the results file says "H". Every
# multi-outcome metric therefore takes an explicit mapping, so the pairing is
# stated once by the caller instead of being assumed inside each function. The
# first version of this module conflated the two and silently scored the wrong
# thing; an explicit mapping makes that impossible.
def _realised_matrix(labels, column_to_label):
    """One column per outcome, 1 where that outcome happened."""
    labels = np.asarray(labels)
    return np.column_stack([
        (labels == label).astype(float) for label in column_to_label.values()
    ])


def multiclass_brier(probability_frame, outcome_labels, column_to_label):
    """Brier score for a mutually exclusive set of outcomes (here H / D / A).

    .. math:: \\frac{1}{N}\\sum_i \\sum_k (p_{ik} - o_{ik})^2

    Summing over all outcomes rather than scoring one in isolation is the honest
    version: a model can look excellent on home wins alone while being hopeless
    at telling draws from away wins.

    Parameters
    ----------
    probability_frame : pandas.DataFrame
        Carries one probability column per outcome.
    outcome_labels : array-like
        What actually happened, in the data's own coding (e.g. "H"/"D"/"A").
    column_to_label : dict
        ``{probability column: data label}``, e.g.
        ``{"home_win": "H", "draw": "D", "away_win": "A"}``.
    """
    probabilities = probability_frame[list(column_to_label)].to_numpy(dtype=float)
    realised = _realised_matrix(outcome_labels, column_to_label)
    return float(np.mean(np.sum((probabilities - realised) ** 2, axis=1)))


def multiclass_log_loss(probability_frame, outcome_labels, column_to_label):
    """Log-loss for a mutually exclusive set of outcomes.

    Only the probability assigned to what actually happened enters the sum, which
    is what makes this the natural multi-outcome extension.
    """
    probabilities = np.clip(
        probability_frame[list(column_to_label)].to_numpy(dtype=float),
        LOG_LOSS_CLIP, 1.0)
    labels = np.asarray(outcome_labels)

    # Position of each outcome's column, keyed by the label it corresponds to.
    position = {label: index
                for index, label in enumerate(column_to_label.values())}

    unknown = set(labels) - set(position)
    if unknown:
        raise ValueError(
            f"Outcome labels {sorted(unknown)} have no probability column. "
            f"Mapping given: {column_to_label}"
        )

    chosen = np.array([probabilities[row, position[label]]
                       for row, label in enumerate(labels)])
    return float(-np.mean(np.log(chosen)))


def baseline_scores(outcome_labels, column_to_label):
    """What a model that only knows the base rates would score.

    Every reported number should be read against this. Beating it is the minimum
    bar for claiming the model knows anything at all, and it is surprisingly easy
    to fail — which is exactly why it gets printed.
    """
    labels = np.asarray(outcome_labels)
    rates = {column: float(np.mean(labels == label))
             for column, label in column_to_label.items()}

    constant = pd.DataFrame({column: np.full(len(labels), rate)
                             for column, rate in rates.items()})
    return {
        "base_rates": rates,
        "brier": multiclass_brier(constant, labels, column_to_label),
        "log_loss": multiclass_log_loss(constant, labels, column_to_label),
    }


# ===========================================================================
# Betting performance  [completed in Phase 6, 18 Oct 2026]
# ===========================================================================

def bootstrap_ci(values, statistic=np.mean, n_resamples=2000, confidence=0.95,
                 seed=42):
    """Percentile bootstrap interval for any statistic.

    **Nothing in this project reports a betting metric without one.** A yield of
    +3% over 200 bets and a yield of +3% over 20,000 bets are entirely different
    claims, and a bare point estimate hides which one you have.

    Resampling rather than a normal approximation, because bet-level returns are
    violently skewed: most bets lose a unit, a few win several, and a symmetric
    interval misrepresents that shape.
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


def max_drawdown(bankroll_curve):
    """Largest peak-to-trough fall, as a share of the running peak.

    Expressed against the peak rather than the starting figure, because that is
    how a drawdown is experienced: a 30% fall from a high is 30% whatever the
    account started at.
    """
    curve = np.asarray(bankroll_curve, dtype=float)
    if len(curve) == 0:
        return 0.0
    peaks = np.maximum.accumulate(curve)
    return float(np.max((peaks - curve) / peaks))


def sharpe_like(returns, periods_per_year=52):
    """A Sharpe-style ratio on per-period returns.

    Called *Sharpe-like* on purpose, and the caveats matter more than the number:

    * no risk-free rate is subtracted — betting returns are not an excess over
      cash in a meaningful sense;
    * Sharpe assumes roughly normal returns, and bet returns are anything but —
      so this understates tail risk, which is the risk that actually ends
      bankrolls.

    It is reported because it is asked for, and read alongside maximum drawdown,
    which does not make that assumption.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def performance_by_band(bets, column="edge", n_bands=5):
    """Hit rate and yield by band of the model's claimed edge.

    The diagnostic that separates "right on average" from "knows where it is
    right". If the model has real information, the bets where it claimed the most
    edge should perform best. A flat profile means the edge estimate is noise
    even when the average is positive.
    """
    frame = bets.copy()
    if column not in frame.columns or "profit" not in frame.columns:
        raise ValueError(f"performance_by_band needs '{column}' and 'profit'.")

    frame["band"] = pd.qcut(frame[column], q=n_bands, duplicates="drop",
                            labels=False)

    table = frame.groupby("band", observed=True).agg(
        bets=("profit", "size"),
        mean_claimed_edge=(column, "mean"),
        hit_rate=("won", "mean"),
        staked=("stake", "sum"),
        profit=("profit", "sum"),
    ).reset_index()
    table["yield"] = table["profit"] / table["staked"].replace(0, np.nan)
    table["band"] = table["band"] + 1
    return table


def summarise(ledger, summary=None, config=None):
    """The headline metrics table, every figure with a confidence interval.

    Parameters
    ----------
    ledger : pandas.DataFrame
        The per-bet ledger from `staking.simulate_bankroll`.
    summary : dict | None
        That function's summary, reused rather than recomputed where available.
    config : dict | None
        Reads ``backtest.bootstrap_resamples`` and ``backtest.confidence_level``.
    """
    config = config or {}
    cfg = config.get("backtest", {})
    n_resamples = int(cfg.get("bootstrap_resamples", 2000))
    confidence = float(cfg.get("confidence_level", 0.95))
    seed = int(config.get("seed", 42))

    if ledger is None or len(ledger) == 0:
        return {"bets": 0, "note": "no bets placed, so there is nothing to report"}

    # Per-bet return on stake: the unit the bootstrap resamples.
    returns = (ledger["profit"] / ledger["stake"].replace(0, np.nan)).to_numpy()

    curve = ledger["bankroll_after_day"].to_numpy(dtype=float)
    weekly = (ledger.set_index("MatchDate")["profit"].resample("W").sum()
              / max(float(summary["starting_bankroll"]) if summary else 1000.0, 1e-9)
              ) if "MatchDate" in ledger.columns else pd.Series(dtype=float)

    return {
        "bets": int(len(ledger)),
        "yield": bootstrap_ci(returns, np.mean, n_resamples, confidence, seed),
        "hit_rate": bootstrap_ci(ledger["won"].astype(float), np.mean,
                                 n_resamples, confidence, seed),
        "average_odds": float(ledger["odds"].mean()),
        "turnover": float(ledger["stake"].sum()),
        "profit": float(ledger["profit"].sum()),
        "roi": float(summary["roi"]) if summary else np.nan,
        "max_drawdown": max_drawdown(curve),
        "sharpe_like": sharpe_like(weekly.to_numpy()) if len(weekly) > 2
                       else float("nan"),
        "confidence_level": confidence,
        # The verdict that matters: does the yield interval exclude zero?
        "yield_significant": bool(
            bootstrap_ci(returns, np.mean, n_resamples, confidence,
                         seed)["lower"] > 0),
    }


def metrics_report(summary, band_table=None):
    """The headline metrics table, formatted with its intervals."""
    if summary.get("bets", 0) == 0:
        return f"Metrics\n-------\n  {summary.get('note', 'no bets')}"

    confidence = summary["confidence_level"]
    lines = ["Headline metrics", "-" * 16]
    lines.append(f"  {'bets':<30} {summary['bets']:>10,}")
    lines.append(f"  {'turnover':<30} {summary['turnover']:>10,.2f}")
    lines.append(f"  {'profit':<30} {summary['profit']:>+10,.2f}")
    lines.append(f"  {'average odds':<30} {summary['average_odds']:>10.2f}")

    lines.append("")
    lines.append(f"  With {confidence:.0%} bootstrap intervals:")
    for label, key, as_signed in (("yield (per unit staked)", "yield", True),
                                  ("hit rate", "hit_rate", False)):
        stats = summary[key]
        if as_signed:
            lines.append(f"    {label:<28} {stats['estimate']:>+9.2%}"
                         f"   [{stats['lower']:+.2%}, {stats['upper']:+.2%}]")
        else:
            lines.append(f"    {label:<28} {stats['estimate']:>10.2%}"
                         f"   [{stats['lower']:.2%}, {stats['upper']:.2%}]")

    lines.append("")
    lines.append(f"  {'ROI (on bankroll)':<30} {summary['roi']:>+10.2%}")
    lines.append(f"  {'maximum drawdown':<30} {summary['max_drawdown']:>10.2%}")
    if np.isfinite(summary.get("sharpe_like", np.nan)):
        lines.append(f"  {'Sharpe-like (weekly)':<30} "
                     f"{summary['sharpe_like']:>10.2f}")
        lines.append("    (no risk-free rate, and it assumes near-normal returns "
                     "which bet returns")
        lines.append("     are not — read it beside the drawdown, which makes no "
                     "such assumption)")

    lines.append("")
    if summary["yield_significant"]:
        lines.append("  Yield interval excludes zero.")
    else:
        lines.append("  Yield interval INCLUDES ZERO: this is not statistically")
        lines.append("  distinguishable from no edge. CLV is the better-powered")
        lines.append("  test on this many bets — see the CLV report.")

    if band_table is not None and not band_table.empty:
        lines.append("")
        lines.append("  Performance by the model's claimed edge:")
        lines.append("    " + band_table.to_string(
            index=False, float_format=lambda v: f"{v:.4f}").replace("\n", "\n    "))

    return "\n".join(lines)
