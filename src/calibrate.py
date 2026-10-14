r"""
Stage 4 — Probability calibration.  [Built in Phase 4, 13 Oct 2026]

A calibrated probability means what it says: of all the matches where the model
said 60%, about 60% should happen. The Dixon-Coles fit produces probabilities
that are close but not quite that, for an ordinary reason — it is a simplified
model of a messy process, and the shape of its error is systematic rather than
random. Calibration maps that systematic part away.

Why this phase exists at all
----------------------------
Phase 5 stakes money by comparing the model's probability with the market's. A
probability that is 5 points too high across a whole band of matches does not
look wrong — it looks like edge. The staking rule will dutifully bet it, every
time, and lose. **Calibration is what earns the right to treat the model's
output as a probability**, and therefore the right to compare it to a price.

The discipline that makes it meaningful
---------------------------------------
A calibration map fitted on the same predictions it is then scored on will
always look like an improvement, because it has seen the answers. The
improvement is not real and will not survive contact with new matches.

So calibration here is always fitted on an **earlier** period and evaluated on a
**later** one (`split_by_time`). That is the same walk-forward discipline Phase 6
applies to the whole pipeline, and the roadmap's "done when" for this phase
depends on it: calibrated probabilities must beat raw ones *out of sample*.

The two methods, and when each is the right choice
--------------------------------------------------
* **Platt scaling** fits a one-dimensional logistic regression on the model's
  log-odds — two parameters, a slope and an intercept. It can only stretch or
  shift the existing ordering, so it cannot overfit much, and it is the safer
  choice on a few hundred matches.
* **Isotonic regression** fits any non-decreasing step function. It is far more
  flexible, so it can correct a shape Platt cannot — and on a small sample it
  will happily fit noise, producing flat steps that are artefacts of a handful
  of matches.

With one or two seasons of data, Platt is usually the honest choice. The
comparison is run out of sample and reported either way, rather than the method
being chosen by assertion.

Three outcomes, not two
-----------------------
Calibration is a binary technique and football has three outcomes. Each is
calibrated separately (home / draw / away) and the three are then renormalised
to sum to 1. That renormalisation is a genuine approximation and
`renormalisation_cost` measures how big it is, so the compromise is visible
rather than buried. For the two-sided markets (over/under, BTTS) no such
approximation is needed.

See docs/methodology.md section 4.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics

REPO_ROOT = Path(__file__).resolve().parents[1]

# The 1X2 market, in the order used throughout.
OUTCOMES = ["home_win", "draw", "away_win"]
RESULT_CODES = {"home_win": "H", "draw": "D", "away_win": "A"}

# Probabilities are clipped before taking log-odds, so a 0 or a 1 cannot produce
# an infinite input to the Platt regression.
PROB_CLIP = 1e-6


class CalibrationError(Exception):
    """Raised when a calibration map cannot be fitted or applied."""


# ===========================================================================
# Measuring calibration
# ===========================================================================

def reliability_table(probabilities, outcomes, n_bins=10):
    """Predicted vs realised frequency, bucketed. The reliability diagram's data.

    Each row is a band of predicted probability, with how many matches fell in
    it, what the model said on average, and what actually happened. A well
    calibrated model has ``predicted`` and ``realised`` close in every row that
    has enough matches to mean anything.

    Equal-width bins are used rather than equal-count, because the question is
    "is the model honest when it says 70%", and that band has to exist as a band.
    The count column is what stops a row of three matches being over-read.
    """
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(outcomes, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=False puts a prediction of exactly 1.0 in the top bin rather than
    # off the end of the array.
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for bin_number in range(n_bins):
        mask = index == bin_number
        if not mask.any():
            continue
        rows.append({
            "bin": f"{edges[bin_number]:.1f}-{edges[bin_number + 1]:.1f}",
            "matches": int(mask.sum()),
            "predicted": float(p[mask].mean()),
            "realised": float(o[mask].mean()),
            "gap": float(p[mask].mean() - o[mask].mean()),
        })

    return pd.DataFrame(rows)


def expected_calibration_error(probabilities, outcomes, n_bins=10):
    """One number for how far off the reliability diagram is.

    .. math:: \\text{ECE} = \\sum_b \\frac{n_b}{N}\\,\\big|\\bar p_b - \\bar o_b\\big|

    The average gap between what the model said and what happened, weighted by
    how many matches sit in each band. 0 is perfect.

    ECE is a diagnostic, not a scoring rule: it can be driven to zero by a model
    that always predicts the base rate, which would be perfectly calibrated and
    perfectly useless. It is always read next to Brier or log-loss, never alone.
    """
    table = reliability_table(probabilities, outcomes, n_bins)
    if table.empty:
        return float("nan")

    weights = table["matches"] / table["matches"].sum()
    return float(np.sum(weights * table["gap"].abs()))


# ===========================================================================
# Fitting a calibration map
# ===========================================================================

def _log_odds(probabilities):
    """Convert probabilities to log-odds, with clipping so 0 and 1 are survivable."""
    p = np.clip(np.asarray(probabilities, dtype=float), PROB_CLIP, 1 - PROB_CLIP)
    return np.log(p / (1 - p))


def fit_calibrator(probabilities, outcomes, method="platt"):
    """Fit a one-dimensional calibration map.

    Parameters
    ----------
    probabilities : array-like
        Raw model probabilities, from the FITTING period only.
    outcomes : array-like
        1 if the event happened, 0 otherwise.
    method : {"platt", "isotonic"}

    Returns
    -------
    dict
        A serialisable description of the map. Serialisable on purpose: a
        calibration map is part of the model and has to survive to Phase 6 and
        beyond without a pickle.
    """
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(outcomes, dtype=float)

    if len(p) != len(o):
        raise CalibrationError(
            f"{len(p)} probabilities but {len(o)} outcomes.")
    if len(p) == 0:
        raise CalibrationError("Nothing to fit a calibration map on.")
    if len(np.unique(o)) < 2:
        raise CalibrationError(
            "The fitting period contains only one outcome class, so there is "
            "nothing to calibrate against. Use a longer fitting window."
        )

    if method == "platt":
        # A logistic regression of the outcome on the model's log-odds. Two
        # parameters: slope a (how much to trust the spread of predictions) and
        # intercept b (a level shift). a < 1 means the model is overconfident.
        from sklearn.linear_model import LogisticRegression

        X = _log_odds(p).reshape(-1, 1)
        fit = LogisticRegression(solver="lbfgs", C=1e6)   # C large = unpenalised
        fit.fit(X, o)

        return {
            "method": "platt",
            "slope": float(fit.coef_[0][0]),
            "intercept": float(fit.intercept_[0]),
            "n_fitted": int(len(p)),
        }

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        fit = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        fit.fit(p, o)

        # Stored as the step function's knots rather than the fitted object, so
        # the map is plain data and can be read, diffed and reviewed.
        return {
            "method": "isotonic",
            "x": [float(v) for v in fit.X_thresholds_],
            "y": [float(v) for v in fit.y_thresholds_],
            "n_fitted": int(len(p)),
        }

    raise CalibrationError(
        f"Unknown calibration method '{method}'. Use 'platt' or 'isotonic'.")


def apply_calibrator(calibrator, probabilities):
    """Apply a fitted map to new probabilities."""
    p = np.asarray(probabilities, dtype=float)

    if calibrator["method"] == "platt":
        z = calibrator["slope"] * _log_odds(p) + calibrator["intercept"]
        return 1.0 / (1.0 + np.exp(-z))

    if calibrator["method"] == "isotonic":
        # np.interp reproduces the fitted step function and clamps outside the
        # fitted range, which matches IsotonicRegression(out_of_bounds="clip").
        return np.interp(p, calibrator["x"], calibrator["y"])

    raise CalibrationError(f"Unknown method '{calibrator['method']}'.")


# ===========================================================================
# Three outcomes at once
# ===========================================================================

def renormalisation_cost(calibrated_frame):
    """How far the three calibrated outcome probabilities were from summing to 1.

    Each outcome is calibrated independently, so nothing forces them to add up.
    Renormalising is the practical fix; this measures what that fix had to do.
    A mean absolute deviation of a couple of points is unremarkable; a large one
    means the per-outcome maps disagree with each other and the whole approach
    needs rethinking rather than patching.
    """
    totals = calibrated_frame[OUTCOMES].sum(axis=1)
    return {
        "mean_total_before_renormalising": float(totals.mean()),
        "mean_absolute_deviation": float((totals - 1.0).abs().mean()),
        "worst_deviation": float((totals - 1.0).abs().max()),
    }


def fit_multiclass(predictions, results, method="platt"):
    """Fit one calibration map per 1X2 outcome.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Must carry the `OUTCOMES` columns — the output of `dixon_coles.predict`.
    results : array-like
        Realised results as "H" / "D" / "A".
    """
    labels = np.asarray(results)
    maps = {}

    for outcome in OUTCOMES:
        if outcome not in predictions.columns:
            raise CalibrationError(f"Predictions are missing '{outcome}'.")
        happened = (labels == RESULT_CODES[outcome]).astype(float)
        maps[outcome] = fit_calibrator(predictions[outcome], happened, method)

    return {
        "method": method,
        "outcomes": maps,
        "n_fitted": int(len(predictions)),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def apply_multiclass(predictions, calibration, renormalise=True):
    """Apply per-outcome maps and (by default) renormalise to sum to 1.

    Returns
    -------
    (pandas.DataFrame, dict)
        A copy of `predictions` with calibrated 1X2 columns, and the
        renormalisation cost measured BEFORE the renormalising, so the
        approximation stays visible.
    """
    out = predictions.copy()
    for outcome in OUTCOMES:
        out[outcome] = apply_calibrator(calibration["outcomes"][outcome],
                                        predictions[outcome])

    cost = renormalisation_cost(out)

    if renormalise:
        totals = out[OUTCOMES].sum(axis=1)
        if (totals <= 0).any():
            raise CalibrationError(
                "Calibrated outcome probabilities sum to zero for some matches; "
                "the maps cannot be renormalised."
            )
        for outcome in OUTCOMES:
            out[outcome] = out[outcome] / totals

    return out, cost


# ===========================================================================
# Out-of-sample evaluation
# ===========================================================================

def split_by_time(frame, fit_fraction=0.5, date_column="MatchDate"):
    """Split chronologically: earlier matches to fit on, later ones to judge on.

    Not a random split. Calibration is applied forward in time, so it has to be
    *tested* forward in time; a random split would let the map learn from matches
    that had not happened yet and would flatter it.

    Returns
    -------
    (fit_frame, evaluation_frame)
    """
    ordered = frame.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column])
    ordered = ordered.sort_values(date_column).reset_index(drop=True)

    cut = int(len(ordered) * fit_fraction)
    if cut < 10 or len(ordered) - cut < 10:
        raise CalibrationError(
            f"{len(ordered)} matches split at {fit_fraction:.0%} leaves too few "
            "on one side to fit or judge a calibration map. Use more data."
        )
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def evaluate(predictions, results, config=None, method=None):
    """Fit calibration on the earlier half and judge it on the later half.

    This is the Phase 4 deliverable: Brier and log-loss before and after
    calibration, out of sample, plus the reliability tables behind them.

    Returns
    -------
    dict
        Scores for the raw and calibrated probabilities, the comparison against
        a base-rate-only model, the fitted maps, and the reliability tables.
    """
    config = config or {}
    cal_cfg = config.get("calibration", {})
    method = method or cal_cfg.get("method", "platt")
    fit_fraction = float(cal_cfg.get("fit_fraction", 0.5))
    n_bins = int(cal_cfg.get("reliability_bins", 10))

    frame = predictions.copy()
    frame["Result"] = np.asarray(results)

    fit_frame, eval_frame = split_by_time(frame, fit_fraction)

    calibration = fit_multiclass(fit_frame, fit_frame["Result"], method)
    calibrated, cost = apply_multiclass(eval_frame, calibration)

    # {probability column: result label} — stated once, used everywhere below.
    mapping = RESULT_CODES

    raw_scores = {
        "brier": metrics.multiclass_brier(eval_frame, eval_frame["Result"], mapping),
        "log_loss": metrics.multiclass_log_loss(eval_frame, eval_frame["Result"],
                                                mapping),
    }
    calibrated_scores = {
        "brier": metrics.multiclass_brier(calibrated, eval_frame["Result"], mapping),
        "log_loss": metrics.multiclass_log_loss(calibrated, eval_frame["Result"],
                                                mapping),
    }

    # Reliability for the home-win market, before and after — the clearest single
    # picture of what calibration did.
    home_happened = (eval_frame["Result"] == "H").astype(float)
    reliability = {
        "raw": reliability_table(eval_frame["home_win"], home_happened, n_bins),
        "calibrated": reliability_table(calibrated["home_win"], home_happened,
                                        n_bins),
    }
    ece = {
        "raw": expected_calibration_error(eval_frame["home_win"], home_happened,
                                          n_bins),
        "calibrated": expected_calibration_error(calibrated["home_win"],
                                                 home_happened, n_bins),
    }

    return {
        "method": method,
        "n_fitted": len(fit_frame),
        "n_evaluated": len(eval_frame),
        "fit_period": [str(fit_frame["MatchDate"].min().date()),
                       str(fit_frame["MatchDate"].max().date())],
        "eval_period": [str(eval_frame["MatchDate"].min().date()),
                        str(eval_frame["MatchDate"].max().date())],
        "raw": raw_scores,
        "calibrated": calibrated_scores,
        "baseline": metrics.baseline_scores(eval_frame["Result"], mapping),
        "improvement": {
            "brier": raw_scores["brier"] - calibrated_scores["brier"],
            "log_loss": raw_scores["log_loss"] - calibrated_scores["log_loss"],
        },
        "ece": ece,
        "renormalisation": cost,
        "calibration": calibration,
        "reliability": reliability,
        "calibrated_predictions": calibrated,
    }


def compare_methods(predictions, results, config=None):
    """Score Platt against isotonic out of sample, rather than assuming.

    On a few hundred matches Platt usually wins because isotonic has enough
    freedom to fit noise. Usually is not always, and the point of this function
    is that the choice is made by measurement.
    """
    rows = []
    for method in ("platt", "isotonic"):
        try:
            result = evaluate(predictions, results, config, method=method)
        except CalibrationError as error:
            rows.append({"method": method, "error": str(error)})
            continue
        rows.append({
            "method": method,
            "brier_raw": result["raw"]["brier"],
            "brier_calibrated": result["calibrated"]["brier"],
            "brier_improvement": result["improvement"]["brier"],
            "log_loss_raw": result["raw"]["log_loss"],
            "log_loss_calibrated": result["calibrated"]["log_loss"],
            "log_loss_improvement": result["improvement"]["log_loss"],
            "ece_calibrated": result["ece"]["calibrated"],
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Reporting and persistence
# ===========================================================================

def calibration_report(result):
    """Readable before/after, with the honest verdict spelled out."""
    lines = ["Probability calibration", "-" * 23]
    lines.append(f"  {'method':<30} {result['method']:>12}")
    lines.append(f"  {'fitted on':<30} {result['n_fitted']:>12,} matches "
                 f"({result['fit_period'][0]} to {result['fit_period'][1]})")
    lines.append(f"  {'judged on':<30} {result['n_evaluated']:>12,} matches "
                 f"({result['eval_period'][0]} to {result['eval_period'][1]})")

    lines.append("")
    lines.append("  Out-of-sample scores (lower is better):")
    lines.append(f"    {'':<14}{'Brier':>10}{'log-loss':>12}")
    lines.append(f"    {'base rates':<14}{result['baseline']['brier']:>10.4f}"
                 f"{result['baseline']['log_loss']:>12.4f}")
    lines.append(f"    {'raw model':<14}{result['raw']['brier']:>10.4f}"
                 f"{result['raw']['log_loss']:>12.4f}")
    lines.append(f"    {'calibrated':<14}{result['calibrated']['brier']:>10.4f}"
                 f"{result['calibrated']['log_loss']:>12.4f}")
    lines.append(f"    {'improvement':<14}{result['improvement']['brier']:>+10.4f}"
                 f"{result['improvement']['log_loss']:>+12.4f}")

    lines.append("")
    lines.append(f"  Expected calibration error (home win): "
                 f"{result['ece']['raw']:.4f} -> {result['ece']['calibrated']:.4f}")
    cost = result["renormalisation"]
    lines.append(f"  Renormalisation: outcomes summed to "
                 f"{cost['mean_total_before_renormalising']:.4f} on average "
                 f"(worst deviation {cost['worst_deviation']:.4f})")

    # --- The verdict -----------------------------------------------------
    lines.append("")
    beats_baseline = result["raw"]["log_loss"] < result["baseline"]["log_loss"]
    helped = result["improvement"]["log_loss"] > 0

    if not beats_baseline:
        lines.append("  FAIL — the raw model does not beat a base-rates-only "
                     "predictor out of sample. Calibration cannot rescue that; "
                     "the problem is upstream in Phase 3.")
    elif helped:
        lines.append("  PASS — calibration improves the model out of sample on "
                     "both scoring rules. Phase 5 may treat these as probabilities.")
    else:
        lines.append("  Calibration did NOT improve the model out of sample. "
                     "That is a real result, not a bug: the raw probabilities "
                     "may already be well calibrated, or the fitting window may "
                     "be too short. Report it rather than tuning until it passes.")

    lines.append("")
    lines.append("  Reliability, home-win market (calibrated):")
    table = result["reliability"]["calibrated"]
    if not table.empty:
        lines.append("    " + table.to_string(
            index=False, float_format=lambda v: f"{v:.3f}").replace("\n", "\n    "))

    return "\n".join(lines)


def save_calibration(calibration, config, filename=None):
    """Write the fitted maps to data/processed/ as JSON."""
    out_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / (filename or config.get("calibration", {}).get(
        "calibration_filename", "calibration.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, default=float)
    return path


def load_calibration(path):
    """Read fitted maps back. Plain data, so no version pinning required."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calibrate(probs, outcomes, config=None):
    """Backwards-compatible entry point: fit and apply in one call.

    Kept because the Phase 0 scaffold and the pipeline runner wired against this
    name. New code should use `evaluate` (out-of-sample, with the report) or
    `fit_multiclass` / `apply_multiclass` directly.
    """
    return evaluate(probs, outcomes, config)
