"""
Phase 9 tests — the simulation study.

The study validates the backtester, so these tests validate the study. Two
kinds, split by cost:

* Fast — the market generator produces what it claims to: coherent books, a
  mispricing of the stated size, and a closing line that corrects toward the
  truth. If the generator is wrong, every conclusion drawn from it is wrong,
  and nothing downstream would notice.
* Recorded — the committed `simulation_study.json` must satisfy the two
  invariants the study exists to establish. Checking the recorded run rather
  than re-running keeps the suite fast; regenerating is a deliberate act.

Run with:  pytest tests/test_simulation.py -v
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

STUDY = REPO_ROOT / "docs" / "charts" / "simulation_study.json"


@pytest.fixture(scope="module")
def study_module():
    spec = importlib.util.spec_from_file_location(
        "simulation_study", REPO_ROOT / "scripts" / "simulation_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recorded():
    if not STUDY.exists():
        pytest.skip("docs/charts/simulation_study.json is absent — run "
                    "`python scripts/simulation_study.py` first")
    return json.loads(STUDY.read_text())


# --- The generator must generate what it says ------------------------------

def test_an_efficient_book_has_no_mispricing(study_module):
    matches, mispricing = study_module.build_market(1.00, n_seasons=1, seed=1)
    assert mispricing == pytest.approx(0.0, abs=1e-12)
    assert len(matches) > 300


def test_a_biased_book_is_wrong_by_the_amount_claimed(study_module):
    """The headline is quoted in probability points, so that number must be real."""
    _, small = study_module.build_market(0.95, n_seasons=1, seed=1)
    _, large = study_module.build_market(0.80, n_seasons=1, seed=1)

    assert 0 < small < large, "a bigger bias factor must mean a bigger mispricing"
    assert 0.005 < small < 0.05
    assert 0.04 < large < 0.20


def test_the_simulated_odds_form_a_coherent_book(study_module):
    """Three prices must imply probabilities summing to a little over one.

    The same check `verify_raw_data.py` applies to a downloaded file is applied
    to the simulator's own output. A study built on incoherent prices would
    produce a confident, meaningless number — and the de-vigging step would be
    the thing quietly producing nonsense.
    """
    matches, _ = study_module.build_market(1.00, n_seasons=1, seed=3)

    for opening, closing in (("OddsH", "CloseH"), ("OddsD", "CloseD"),
                             ("OddsA", "CloseA")):
        assert (matches[opening] > 1.0).all()
        assert (matches[closing] > 1.0).all()

    book_open = (1 / matches[["OddsH", "OddsD", "OddsA"]]).sum(axis=1)
    book_close = (1 / matches[["CloseH", "CloseD", "CloseA"]]).sum(axis=1)

    assert 1.02 < book_open.median() < 1.12
    assert 1.00 < book_close.median() < 1.06
    # The closing line is sharper than the opening — thinner margin.
    assert book_close.median() < book_open.median()


def test_the_closing_line_corrects_toward_the_truth(study_module):
    """CLV only means anything if the close moves toward the real probability.

    In a biased market the opening price understates the home team. The closing
    price must sit between the opening and the truth — otherwise the study is
    measuring drift, not correction, and a positive CLV would prove nothing.
    """
    matches, _ = study_module.build_market(0.80, n_seasons=1, seed=5)

    implied_open = 1.0 / matches["OddsH"]
    implied_close = 1.0 / matches["CloseH"]

    # Margins differ, so compare de-vigged shares rather than raw implied.
    share_open = implied_open / (1 / matches[["OddsH", "OddsD", "OddsA"]]).sum(axis=1)
    share_close = implied_close / (1 / matches[["CloseH", "CloseD", "CloseA"]]).sum(axis=1)

    assert share_close.mean() > share_open.mean(), (
        "with home underpriced at the open, the close must shorten the home "
        "price; it does not, so the simulated correction runs the wrong way"
    )


def test_significance_requires_the_whole_interval_above_zero(study_module):
    assert study_module.significance(
        {"bets": 100, "clv_low": 0.001, "clv_points": 0.01}) is True
    assert study_module.significance(
        {"bets": 100, "clv_low": -0.001, "clv_points": 0.01}) is False
    assert study_module.significance({"bets": 0}) is False


# --- The recorded run must hold the study's two invariants -----------------

def test_no_edge_is_reported_on_an_efficient_book(recorded):
    """The failure that would matter most: inventing an edge from nothing."""
    assert recorded["no_false_positive"] is True

    efficient = [lvl for lvl in recorded["levels"] if lvl["bias"] >= 1.0]
    assert efficient, "the sweep must include an efficient book as its control"
    for level in efficient:
        assert not level["detected"], (
            "the pipeline reported a significant edge on a book that is "
            "correct by construction"
        )


def test_a_model_with_no_information_finds_no_edge(recorded):
    noise = recorded["noise_control"]
    assert not noise.get("detected", False), (
        "random probabilities produced a significant CLV — the measurement is "
        "manufacturing signal"
    )


def test_measured_clv_rises_with_the_planted_edge(recorded):
    """The instrument must respond monotonically to the thing it measures.

    A flat response would mean it is blind; a non-monotonic one would mean the
    number it reports is not tracking the quantity it claims to track.
    """
    levels = sorted((lvl for lvl in recorded["levels"] if lvl.get("bets", 0)),
                    key=lambda lvl: lvl["mispricing_points"])
    clv_values = [lvl["clv_points"] for lvl in levels]

    assert len(levels) >= 3, "the sweep needs at least three points to be a sweep"
    assert clv_values == sorted(clv_values), (
        f"measured CLV is not monotonic in the planted mispricing: {clv_values}"
    )
    assert clv_values[0] < 0 < clv_values[-1], (
        "the sweep should straddle zero: no edge at the left, a clear one at "
        "the right"
    )


def test_the_detection_threshold_is_reported_in_usable_units(recorded):
    """'Bias factor 0.88' means nothing to a reader; probability points do."""
    threshold = recorded.get("detection_threshold_points")
    if threshold is None:
        pytest.skip("no level in the recorded sweep reached detection")
    assert 0 < threshold < 0.5
    assert recorded["matches_per_level"] > 0, (
        "a detection threshold without a sample size is not a result — the "
        "same edge is detectable or not depending on how many matches you have"
    )


def test_the_study_states_that_it_is_simulated(recorded):
    """A simulated result that does not say so is the project's worst bug."""
    note = recorded.get("note", "").lower()
    assert "simulated" in note
    assert "no claim" in note or "makes no claim" in note


# --- Added 26 Oct 2026: the decomposition, recorded ------------------------

def test_every_recorded_arm_is_decomposed(recorded):
    """Each arm records margin paid and market movement, and the identity holds:
    CLV points = margin paid + movement. Without both parts, the simulated and
    real runs can only be compared on CLV, which mixes in a margin that is a
    SETTING here and Bet365's choice in the real data."""
    arms = [level for level in recorded["levels"] if level.get("bets", 0)]
    arms.append(recorded["noise_control"])
    for arm in arms:
        assert "movement" in arm, "simulation_study.json predates the decomposition"
        assert arm["margin_paid"] + arm["movement"] == pytest.approx(
            arm["clv_points"], abs=1e-9)


CALIBRATION = REPO_ROOT / "docs" / "charts" / "simulation_calibration.json"


@pytest.fixture(scope="module")
def calibration():
    if not CALIBRATION.exists():
        pytest.skip("docs/charts/simulation_calibration.json is absent - run "
                    "`python scripts/simulation_study.py --calibrate` on real data")
    return json.loads(CALIBRATION.read_text())


def test_no_information_earns_no_more_movement_than_the_real_market_gives(
        recorded, calibration):
    """Regression, 26 Oct 2026: the control passed for the wrong reason.

    Decomposed, the random-probability control showed market movement of
    +0.46 points TOWARD its bets, interval clear of zero. It passed the
    no-false-positive check only because the 1.7-point margin hid it. The
    cause was the simulator, not the instrument: bets chosen because a price
    looked long were prices that were long by chance, and the simulated close
    corrected all of that error and also mis-read a proportionally loaded
    margin through Shin.

    The real market has a faint version of the same effect, so the bar is not
    zero. It is the movement real-market price selection earns with no model
    at all: an information-free bettor may not do better in the simulator than
    in the market it imitates.
    """
    ceiling = calibration["real_target"]["upper"]
    efficient = next(level for level in recorded["levels"] if level["bias"] >= 1.0)
    for name, arm in (("efficient book", efficient),
                      ("random-model control", recorded["noise_control"])):
        assert arm["movement"] <= ceiling, (
            f"{name}: movement {arm['movement']:+.2%} exceeds what price selection "
            f"earns in the real market ({ceiling:+.2%})")


def test_the_simulator_loads_margin_the_way_the_pipeline_unwinds_it(study_module):
    """Shin-loaded prices must de-vig back to exactly the probabilities that
    built them. If this fails, every simulated CLV carries a phantom movement
    on long prices (about +0.5 points when the loading was proportional)."""
    from src.staking import devig_shin
    for fair, margin in (([0.50, 0.30, 0.20], 0.055),
                         ([0.85, 0.10, 0.05], 0.022),
                         ([0.08, 0.17, 0.75], 0.055)):
        implied = study_module.load_margin_shin(fair, margin)
        assert implied.sum() == pytest.approx(1 + margin, abs=1e-10)
        recovered, note = devig_shin(1 / implied)
        assert note["method"] == "shin"
        assert recovered == pytest.approx(fair, abs=1e-10)


def test_the_simulator_uses_the_calibrated_share(study_module, calibration):
    """The constant in the file must be the one the calibration chose. A
    hand-edited share would make the study's realism an assertion again."""
    assert study_module.OPENING_ERROR_CORRECTED == pytest.approx(
        calibration["closest_share"])


def test_the_calibrated_simulator_matches_the_real_statistic(study_module,
                                                             calibration):
    """Recompute the calibration statistic live at the chosen share: it must
    still fall inside the real market's interval. Catches any later change to
    the generator that quietly breaks the calibration."""
    market, _ = study_module.build_market(
        1.00, calibration["simulated_seasons_per_point"], calibration["seed"])
    stat = study_module.selection_movement(
        market[["OddsH", "OddsD", "OddsA"]], market[["CloseH", "CloseD", "CloseA"]])
    target = calibration["real_target"]
    assert target["lower"] <= stat["estimate"] <= target["upper"]


def test_a_sensitivity_run_cannot_overwrite_the_published_study(study_module,
                                                                 monkeypatch):
    """--close-corrects changes the simulator, so its output must go elsewhere."""
    monkeypatch.setattr(sys, "argv", ["simulation_study.py",
                                      "--close-corrects", "0.2"])
    with pytest.raises(SystemExit) as stopped:
        study_module.main()
    assert stopped.value.code != 0


def test_calibration_refuses_when_the_named_seasons_are_missing(study_module,
                                                                tmp_path, capsys):
    """The calibration target comes from seasons named in config.yaml. If any
    are missing it must stop, not quietly calibrate on whatever is present -
    two seasons give a target of a different sign from thirteen."""
    from src.config import load_config
    config = load_config()
    config["simulation"] = {"calibration_seasons": ["9091"]}   # never built

    class Args:
        seed, out = 20261023, str(tmp_path)

    assert study_module.calibrate(config, Args()) == 1
    assert "E0_9091.csv" in capsys.readouterr().out
    assert not (tmp_path / study_module.CALIBRATION_FILE).exists()
