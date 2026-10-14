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


def summarise(results):
    """Summarise backtest results into the headline metrics table.

    Not yet implemented — arrives in Phase 6, together with ROI, yield, hit rate
    by confidence band, a Sharpe-like ratio and maximum drawdown.
    """
    raise NotImplementedError("metrics.summarise is scheduled for Phase 6.")
