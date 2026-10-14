"""
Phase 4 tests — probability calibration and proper scoring rules.

Two groups matter most:

* **Identities** — Brier and log-loss computed by hand on tiny examples, so the
  numbers everything else is judged by are themselves checked.
* **Discipline** — a calibration map must be fitted on earlier matches and
  judged on later ones. `test_in_sample_calibration_flatters_itself` shows what
  happens when that rule is broken, which is the whole reason the rule exists.

Run with:  pytest tests/test_calibrate.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import calibrate, metrics


@pytest.fixture
def config():
    return load_config()


def make_predictions(n=600, seed=0, overconfidence=1.0, bias=0.0):
    """Predictions with a KNOWN calibration fault, so a fix is measurable.

    ``overconfidence > 1`` stretches the model's log-odds away from the middle,
    which is the classic fault: the model says 85% when it should say 70%.
    ``bias`` shifts every home-win probability, which is the other classic one.
    """
    rng = np.random.default_rng(seed)

    # True probabilities for each match, then the actual result drawn from them.
    true_home = rng.uniform(0.2, 0.65, n)
    true_draw = rng.uniform(0.18, 0.30, n)
    true_away = 1 - true_home - true_draw
    keep = true_away > 0.05
    true_home, true_draw, true_away = true_home[keep], true_draw[keep], true_away[keep]
    n = len(true_home)

    truth = np.column_stack([true_home, true_draw, true_away])
    truth = truth / truth.sum(axis=1, keepdims=True)

    results = np.array([
        rng.choice(["H", "D", "A"], p=row) for row in truth
    ])

    # The model's (mis-calibrated) version of the same probabilities.
    def distort(p):
        odds = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return 1.0 / (1.0 + np.exp(-(overconfidence * odds + bias)))

    model = np.column_stack([distort(truth[:, k]) for k in range(3)])
    model = model / model.sum(axis=1, keepdims=True)

    return pd.DataFrame({
        "MatchID": [f"M{i:04d}" for i in range(n)],
        "MatchDate": pd.date_range("2024-08-01", periods=n, freq="6h"),
        "home_win": model[:, 0],
        "draw": model[:, 1],
        "away_win": model[:, 2],
    }), results


# =========================================================================
# Scoring rules, checked by hand
# =========================================================================

def test_brier_score_by_hand():
    """Three predictions, worked out on paper.

        (0.9 - 1)^2 = 0.01
        (0.2 - 0)^2 = 0.04
        (0.6 - 1)^2 = 0.16
        mean = 0.21 / 3 = 0.07
    """
    assert metrics.brier_score([0.9, 0.2, 0.6], [1, 0, 1]) == pytest.approx(0.07)


def test_log_loss_by_hand():
    """Two predictions:  -(ln 0.8 + ln 0.6) / 2 = 0.3712"""
    expected = -(np.log(0.8) + np.log(0.6)) / 2
    assert metrics.log_loss([0.8, 0.4], [1, 0]) == pytest.approx(expected)


def test_a_perfect_prediction_scores_zero():
    assert metrics.brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert metrics.log_loss([1.0, 0.0], [1, 0]) == pytest.approx(0.0, abs=1e-10)


def test_log_loss_punishes_confident_errors_far_harder_than_brier():
    """The reason both are reported. One confident wrong bet at full Kelly can
    end a bankroll, and Brier alone would barely notice it."""
    confident_miss_brier = metrics.brier_score([0.01], [1])
    confident_miss_log = metrics.log_loss([0.01], [1])

    assert confident_miss_brier < 1.0          # bounded, forgiving
    assert confident_miss_log > 4.0            # unbounded, brutal


def test_log_loss_survives_a_probability_of_zero():
    """Without clipping, one prediction of exactly 0 makes the whole evaluation
    infinite and the result becomes a single row's story."""
    value = metrics.log_loss([0.0, 0.5, 0.9], [1, 0, 1])
    assert np.isfinite(value)


def test_multiclass_brier_by_hand():
    """One match, model says 0.5/0.3/0.2, home wins.

        (0.5-1)^2 + (0.3-0)^2 + (0.2-0)^2 = 0.25 + 0.09 + 0.04 = 0.38
    """
    frame = pd.DataFrame({"home_win": [0.5], "draw": [0.3], "away_win": [0.2]})
    score = metrics.multiclass_brier(frame, ["H"], calibrate.RESULT_CODES)
    assert score == pytest.approx(0.38)


def test_multiclass_log_loss_uses_only_what_happened():
    """A draw occurred, so only the draw column should enter the sum."""
    frame = pd.DataFrame({"home_win": [0.5], "draw": [0.3], "away_win": [0.2]})
    score = metrics.multiclass_log_loss(frame, ["D"], calibrate.RESULT_CODES)
    assert score == pytest.approx(-np.log(0.3))


def test_mismatched_labels_are_rejected():
    """The bug this interface was redesigned to prevent: scoring 'H'/'D'/'A'
    results against columns named 'home_win'/'draw'/'away_win' without saying
    which maps to which."""
    frame = pd.DataFrame({"home_win": [0.5], "draw": [0.3], "away_win": [0.2]})
    with pytest.raises(ValueError):
        metrics.multiclass_log_loss(frame, ["home_win"], calibrate.RESULT_CODES)


def test_baseline_is_what_knowing_nothing_scores(config):
    """Every reported number is read against this floor."""
    _, results = make_predictions(400)
    baseline = metrics.baseline_scores(results, calibrate.RESULT_CODES)

    assert sum(baseline["base_rates"].values()) == pytest.approx(1.0)
    assert 0 < baseline["brier"] < 1
    assert baseline["log_loss"] > 0


# =========================================================================
# Measuring calibration
# =========================================================================

def test_reliability_table_matches_reality_for_a_calibrated_model():
    """If the model is honest, predicted and realised agree in every band.

    "Agree" has to be judged against sampling noise, not a flat tolerance. A
    band holding 200 matches at a realised rate near 0.5 has a standard error of
    sqrt(0.25/200) = 0.035, so a gap of 0.06 there is ordinary while the same gap
    in a 2,000-match band would be a real miscalibration. Each band is therefore
    checked against three standard errors of its own count — which is also why a
    reliability diagram is always read with the counts beside it.
    """
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 4000)
    outcomes = (rng.random(4000) < p).astype(int)      # truly calibrated

    table = calibrate.reliability_table(p, outcomes, n_bins=10)
    assert table["matches"].sum() == 4000

    populated = table[table["matches"] >= 100]
    standard_error = np.sqrt(
        populated["realised"] * (1 - populated["realised"]) / populated["matches"]
    )
    tolerance = 3 * standard_error

    too_far = populated[populated["gap"].abs() > tolerance]
    assert too_far.empty, (
        f"bands outside 3 standard errors of their own count:\n{too_far}"
    )


def test_ece_is_near_zero_for_a_calibrated_model():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 4000)
    outcomes = (rng.random(4000) < p).astype(int)

    assert calibrate.expected_calibration_error(p, outcomes) < 0.03


def test_ece_detects_a_systematic_bias():
    """Add 15 points to every prediction; ECE must notice."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.75, 3000)
    outcomes = (rng.random(3000) < p).astype(int)

    honest = calibrate.expected_calibration_error(p, outcomes)
    biased = calibrate.expected_calibration_error(np.clip(p + 0.15, 0, 1), outcomes)

    assert biased > honest + 0.08


# =========================================================================
# Fitting and applying
# =========================================================================

def test_platt_corrects_overconfidence():
    """The classic fault: the model says 85% when it should say 70%."""
    rng = np.random.default_rng(4)
    true_p = rng.uniform(0.1, 0.9, 3000)
    outcomes = (rng.random(3000) < true_p).astype(int)

    # Stretch the log-odds to make the model overconfident.
    odds = np.log(true_p / (1 - true_p))
    overconfident = 1 / (1 + np.exp(-1.8 * odds))

    calibrator = calibrate.fit_calibrator(overconfident, outcomes, "platt")
    fixed = calibrate.apply_calibrator(calibrator, overconfident)

    # A slope below 1 is exactly the correction for overconfidence.
    assert calibrator["slope"] < 1.0
    assert (metrics.brier_score(fixed, outcomes)
            < metrics.brier_score(overconfident, outcomes))


def test_isotonic_corrects_a_shape_platt_cannot():
    """Isotonic's extra freedom earns its keep on a non-monotone-in-logit fault."""
    rng = np.random.default_rng(5)
    true_p = rng.uniform(0.05, 0.95, 4000)
    outcomes = (rng.random(4000) < true_p).astype(int)

    # A kinked distortion: squashed at the bottom, stretched at the top.
    distorted = np.where(true_p < 0.5, true_p * 0.5, 0.25 + (true_p - 0.5) * 1.5)
    distorted = np.clip(distorted, 0.01, 0.99)

    calibrator = calibrate.fit_calibrator(distorted, outcomes, "isotonic")
    fixed = calibrate.apply_calibrator(calibrator, distorted)

    assert (metrics.brier_score(fixed, outcomes)
            < metrics.brier_score(distorted, outcomes))


def test_calibrator_is_plain_serialisable_data():
    """The map is part of the model, so it must survive JSON — no pickles."""
    import json

    rng = np.random.default_rng(6)
    p = rng.uniform(0.1, 0.9, 500)
    outcomes = (rng.random(500) < p).astype(int)

    for method in ("platt", "isotonic"):
        calibrator = calibrate.fit_calibrator(p, outcomes, method)
        reloaded = json.loads(json.dumps(calibrator))

        np.testing.assert_allclose(
            calibrate.apply_calibrator(calibrator, p),
            calibrate.apply_calibrator(reloaded, p),
        )


def test_single_class_fitting_window_is_refused():
    with pytest.raises(calibrate.CalibrationError):
        calibrate.fit_calibrator([0.3, 0.5, 0.7], [0, 0, 0], "platt")


def test_unknown_method_is_refused():
    with pytest.raises(calibrate.CalibrationError):
        calibrate.fit_calibrator([0.3, 0.7], [0, 1], "magic")


# =========================================================================
# Three outcomes at once
# =========================================================================

def test_calibrated_outcomes_sum_to_one_after_renormalising(config):
    predictions, results = make_predictions(600, overconfidence=1.6)
    fit_frame, eval_frame = calibrate.split_by_time(
        predictions.assign(Result=results))

    calibration = calibrate.fit_multiclass(fit_frame, fit_frame["Result"])
    calibrated, _ = calibrate.apply_multiclass(eval_frame, calibration)

    totals = calibrated[calibrate.OUTCOMES].sum(axis=1)
    np.testing.assert_allclose(totals, 1.0, atol=1e-9)


def test_renormalisation_cost_is_measured_not_hidden(config):
    """Each outcome is calibrated independently, so nothing forces them to sum
    to 1. The size of that compromise must be visible."""
    predictions, results = make_predictions(600, overconfidence=1.6)
    fit_frame, eval_frame = calibrate.split_by_time(
        predictions.assign(Result=results))

    calibration = calibrate.fit_multiclass(fit_frame, fit_frame["Result"])
    _, cost = calibrate.apply_multiclass(eval_frame, calibration)

    assert "mean_absolute_deviation" in cost
    assert cost["worst_deviation"] >= cost["mean_absolute_deviation"]
    assert np.isfinite(cost["mean_total_before_renormalising"])


# =========================================================================
# The discipline that makes it mean anything
# =========================================================================

def test_split_by_time_is_chronological():
    """Not a random split: every fitting match must precede every judged one."""
    predictions, results = make_predictions(400)
    fit_frame, eval_frame = calibrate.split_by_time(
        predictions.assign(Result=results), fit_fraction=0.5)

    assert fit_frame["MatchDate"].max() <= eval_frame["MatchDate"].min()
    assert len(fit_frame) + len(eval_frame) == len(predictions)


def test_split_refuses_when_a_side_would_be_too_small():
    predictions, results = make_predictions(30)
    with pytest.raises(calibrate.CalibrationError):
        calibrate.split_by_time(predictions.assign(Result=results),
                                fit_fraction=0.95)


def test_calibration_improves_a_broken_model_out_of_sample(config):
    """The roadmap's 'done when' for this phase, as a test."""
    predictions, results = make_predictions(1200, seed=7, overconfidence=1.7)
    result = calibrate.evaluate(predictions, results, config, method="platt")

    assert result["improvement"]["log_loss"] > 0, (
        "calibration must beat the raw model out of sample on a known-broken one"
    )
    assert result["improvement"]["brier"] > 0
    assert result["ece"]["calibrated"] < result["ece"]["raw"]
    assert "PASS" in calibrate.calibration_report(result)


def test_in_sample_calibration_flatters_itself(config):
    """Why the time split exists, demonstrated rather than asserted.

    Fitting a calibration map and then scoring it on the SAME matches shows a
    larger improvement than scoring it on later ones — because the map has seen
    the answers. The in-sample number is not a forecast of anything.
    """
    predictions, results = make_predictions(1200, seed=8, overconfidence=1.7)
    frame = predictions.assign(Result=results)

    # The honest version: fit early, judge late.
    honest = calibrate.evaluate(predictions, results, config, method="isotonic")

    # The flattering version: fit and judge on everything.
    cheating_map = calibrate.fit_multiclass(frame, results, "isotonic")
    cheating, _ = calibrate.apply_multiclass(frame, cheating_map)
    raw_in_sample = metrics.multiclass_log_loss(frame, results,
                                                calibrate.RESULT_CODES)
    cheating_in_sample = metrics.multiclass_log_loss(cheating, results,
                                                     calibrate.RESULT_CODES)
    flattered_gain = raw_in_sample - cheating_in_sample

    assert flattered_gain > honest["improvement"]["log_loss"], (
        "in-sample calibration should look better than it is — if it does not, "
        "this demonstration is no longer demonstrating anything"
    )


def test_evaluate_reports_the_baseline_floor(config):
    """A model that cannot beat the base rates has nothing for calibration to fix."""
    predictions, results = make_predictions(800, seed=9)
    result = calibrate.evaluate(predictions, results, config)

    assert result["baseline"]["log_loss"] > 0
    assert result["raw"]["log_loss"] < result["baseline"]["log_loss"], (
        "the simulated model should beat base rates; if not, check the fixture"
    )


def test_compare_methods_scores_both_out_of_sample(config):
    predictions, results = make_predictions(1200, seed=10, overconfidence=1.5)
    table = calibrate.compare_methods(predictions, results, config)

    assert set(table["method"]) == {"platt", "isotonic"}
    assert "log_loss_improvement" in table.columns
    assert table["log_loss_improvement"].notna().all()


def test_calibration_survives_a_json_round_trip(config, tmp_path):
    predictions, results = make_predictions(800, seed=11, overconfidence=1.5)
    result = calibrate.evaluate(predictions, results, config)

    cfg = dict(config)
    cfg["data"] = dict(config["data"], processed_dir=str(tmp_path))
    reloaded = calibrate.load_calibration(
        calibrate.save_calibration(result["calibration"], cfg))

    _, eval_frame = calibrate.split_by_time(predictions.assign(Result=results))
    before, _ = calibrate.apply_multiclass(eval_frame, result["calibration"])
    after, _ = calibrate.apply_multiclass(eval_frame, reloaded)

    pd.testing.assert_frame_equal(before[calibrate.OUTCOMES],
                                  after[calibrate.OUTCOMES])
