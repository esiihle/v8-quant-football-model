"""
Phase 8 tests — the README must stay true.

A README is the one file nobody re-reads after the code changes, which is how a
portfolio project ends up claiming numbers its own pipeline no longer produces.
These tests make that failure loud.

Two kinds of test here:

* Always run — the README references only charts the chart script actually
  writes, carries no leftover placeholders, and names the real repository.
* Run when ``docs/charts/numbers.json`` is present — every headline number in
  the README matches the run that produced the charts. Without the real raw
  data there is no numbers.json, so those tests skip with instructions rather
  than fail on a fresh clone.

Run with:  pytest tests/test_docs.py -v
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

README = REPO_ROOT / "README.md"
CHART_DIR = REPO_ROOT / "docs" / "charts"
NUMBERS = CHART_DIR / "numbers.json"

# Percentages are quoted to one decimal place, so anything inside half of the
# last place is the same number. Wider than float noise, far tighter than a real
# change in the model.
TOLERANCE = 0.0005


@pytest.fixture(scope="module")
def readme():
    return README.read_text(encoding="utf-8")


def collapse(text):
    """Whitespace-normalised, lowercased text, for matching a wrapped phrase.

    A phrase check against the raw file fails the moment the sentence wraps
    across a line — which says nothing about the README and everything about
    the test.
    """
    return " ".join(text.split()).lower()


@pytest.fixture(scope="module")
def simulation_script():
    """The simulation study owns one chart of its own."""
    spec = importlib.util.spec_from_file_location(
        "simulation_study", REPO_ROOT / "scripts" / "simulation_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chart_script():
    """Import scripts/make_charts.py for its CHARTS manifest.

    Imported by path rather than duplicated here, so the manifest has exactly
    one definition and this test cannot drift from it.
    """
    spec = importlib.util.spec_from_file_location(
        "make_charts", REPO_ROOT / "scripts" / "make_charts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def numbers():
    if not NUMBERS.exists():
        pytest.skip(
            "docs/charts/numbers.json is absent — run "
            "`python scripts/make_charts.py` on real data first. On a clone "
            "without data/raw this is expected."
        )
    recorded = json.loads(NUMBERS.read_text())

    # A numbers.json built from the bundled sample is itself the bug, not a
    # reason to skip: it means sample output reached the published folder. Fail
    # here with the cause, rather than letting every downstream test die on a
    # KeyError for a figure the sample run never produced.
    if recorded.get("real_data") is not True:
        pytest.fail(
            "docs/charts/numbers.json came from the bundled sample "
            f"({recorded.get('matches', recorded.get('n_train', '?'))} rows). "
            "Delete docs/charts/, put the real data in data/raw/, and run "
            "`python scripts/make_charts.py` again."
        )
    return recorded


# --- Always run ------------------------------------------------------------

def test_readme_only_links_charts_the_script_writes(readme, chart_script,
                                                    simulation_script):
    """A broken image in a README is a broken promise on the front page."""
    linked = set(re.findall(r"docs/charts/([\w.-]+\.png)", readme))
    produced = (set(chart_script.CHARTS.values())
                | set(simulation_script.CHARTS.values()))

    unknown = linked - produced
    assert not unknown, (
        f"README links charts that make_charts.py does not write: {sorted(unknown)}"
    )
    assert linked, "the README should show the charts"


def test_every_chart_the_script_writes_is_shown(chart_script, readme,
                                                simulation_script):
    """A chart nobody looks at is a chart nobody maintains."""
    produced = (set(chart_script.CHARTS.values())
                | set(simulation_script.CHARTS.values()))
    linked = set(re.findall(r"docs/charts/([\w.-]+\.png)", readme))

    unused = produced - linked
    assert not unused, (
        f"make_charts.py writes charts the README never shows: {sorted(unused)}. "
        "Either show them or stop generating them."
    )


def test_no_placeholders_survive(readme):
    """Phase 0 wrote 'in active development'. Phase 8 must have removed it."""
    forbidden = ["🚧", "TODO", "TBD", "Planned headline outputs",
                 "in active development", "<your-username>", "coming soon"]
    found = [marker for marker in forbidden if marker.lower() in readme.lower()]
    assert not found, f"placeholder text left in the README: {found}"


def test_readme_names_the_real_repository(readme):
    """The clone command has to work when a recruiter pastes it."""
    assert "github.com/esiihle/v8-quant-football-model" in readme


def test_readme_states_the_honest_result(readme):
    """The negative result is the deliverable; it may not quietly disappear.

    This test exists because the temptation it guards against is real: a README
    is the easiest place to let an uncomfortable finding soften into nothing.
    """
    lowered = readme.lower()
    assert "does not beat the closing line" in lowered
    assert "walk-forward" in lowered
    for phrase in ("what this project shows", "it does not show"):
        assert phrase in lowered


def test_chart_manifest_has_a_purpose_for_every_chart(chart_script):
    assert set(chart_script.CHARTS) == set(chart_script.CHART_PURPOSE)


# --- Run when a real chart run is committed --------------------------------

def quoted_percentages(text):
    """Every percentage in the README, as fractions, keyed by their raw text."""
    found = {}
    for raw in re.findall(r"[−+-]?\d+(?:\.\d+)?%", text):
        # The README uses a typographic minus; Python does not.
        normalised = raw.replace("−", "-").replace("+", "")
        found[raw] = float(normalised.rstrip("%")) / 100.0
    return found


def assert_quoted(readme, value, label):
    """The README must quote `value` somewhere, to its own precision."""
    quoted = quoted_percentages(readme)
    assert any(abs(candidate - value) <= TOLERANCE
               for candidate in quoted.values()), (
        f"the README does not quote {label} = {value:+.2%} from numbers.json. "
        f"Re-run scripts/make_charts.py and update the tables."
    )


def test_readme_quotes_the_walk_forward_roi(readme, numbers):
    assert_quoted(readme, numbers["walk_forward_roi"], "walk-forward ROI")


def test_readme_quotes_the_walk_forward_yield(readme, numbers):
    assert_quoted(readme, numbers["walk_forward_yield"], "walk-forward yield")


def test_readme_quotes_the_clv_verdict(readme, numbers):
    """The authoritative CLV measure and both ends of its interval."""
    for key, label in (("clv_points_mean", "mean CLV in probability points"),
                       ("clv_points_low", "the CLV interval's lower bound"),
                       ("clv_points_high", "the CLV interval's upper bound")):
        assert_quoted(readme, numbers[key], label)


def test_readme_quotes_the_yield_interval(readme, numbers):
    """Added 25 Oct 2026. The yield's interval decides whether the loss is
    established, and the README once quoted the interval of a different
    number (the unweighted mean return). Both ends must match numbers.json."""
    for key, label in (("walk_forward_yield_low", "the yield interval's lower bound"),
                       ("walk_forward_yield_high", "the yield interval's upper bound")):
        assert_quoted(readme, numbers[key], label)


def test_readme_quotes_the_clv_decomposition(readme, numbers):
    """Added 25 Oct 2026. The README's central claim - that the negative CLV
    is all margin and the market did not move - rests on these four figures."""
    for key, label in (("clv_margin_paid", "the margin paid"),
                       ("clv_movement_mean", "the market movement"),
                       ("clv_movement_low", "the movement interval's lower bound"),
                       ("clv_movement_high", "the movement interval's upper bound")):
        assert_quoted(readme, numbers[key], label)


def test_readme_claims_no_movement_only_when_the_data_says_so(readme, numbers):
    """The prose verdict must follow the generated verdict, not outlive it.
    If a rerun on more seasons finds movement, 'all of it is the margin'
    becomes false and this test says so."""
    claims_none = "every bit of the negative clv is the margin" in readme.lower()
    assert claims_none == (numbers.get("clv_movement_verdict") == "none"), (
        "the README's CLV verdict does not match numbers.json's "
        f"clv_movement_verdict ({numbers.get('clv_movement_verdict')})")


def test_readme_quotes_the_bet_count(readme, numbers):
    assert str(numbers["walk_forward_bets"]) in readme, (
        f"the README should say how many bets the walk-forward run placed "
        f"({numbers['walk_forward_bets']})"
    )


def test_readme_quotes_the_run_fingerprint(readme, numbers):
    """The fingerprint ties the README's numbers to a specific run."""
    assert numbers["run_fingerprint"] in readme, (
        "the README should print the fingerprint of the run its numbers "
        f"came from ({numbers['run_fingerprint']})"
    )


def test_committed_charts_came_from_real_data(numbers):
    """Charts built on the 12-match sample must never be committed as results.

    make_charts.py stamps them ILLUSTRATIVE, but a stamped PNG is still a PNG.
    This is the test that stops one reaching the front page.
    """
    assert numbers.get("real_data") is True, (
        "docs/charts/ was generated from the bundled sample. Download the raw "
        "data and run `python scripts/make_charts.py` again before committing."
    )


def test_every_chart_the_readme_shows_exists_on_disk(readme):
    """Charts the README shows must be in the repo — with one exemption.

    While the real-data section carries its "Awaiting regeneration" notice, the
    charts belonging to it legitimately do not exist: they are produced by a run
    against raw season files that has not happened. Requiring them would force
    the choice between a red suite and committing charts from the wrong data,
    and that second option is the mistake this project already made once.

    The exemption is tied to the notice, so it expires with it: delete the
    notice and every linked chart must be present.
    """
    linked = set(re.findall(r"docs/charts/([\w.-]+\.png)", readme))
    missing = [name for name in linked if not (CHART_DIR / name).exists()]
    if not missing:
        return

    if "awaiting regeneration" in collapse(readme):
        # Only the real-data charts may be missing. Anything the simulation
        # study produces is reproducible with no download and must be present.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "simulation_study", REPO_ROOT / "scripts" / "simulation_study.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        still_missing = [n for n in missing if n in module.CHARTS.values()]
        assert not still_missing, (
            "the simulation study's charts need no raw data, so they may not "
            f"be missing: {still_missing}. Run scripts/simulation_study.py"
        )
        return

    assert not missing, f"README shows charts that are not in the repo: {missing}"


# --- Added 22 Oct 2026: the guard that was missing -------------------------

def test_sample_charts_cannot_be_written_into_the_published_folder(chart_script):
    """docs/charts is what the README publishes, so sample data may not reach it.

    On 21 Oct 2026 the football project committed three ILLUSTRATIVE-stamped
    charts built from a 12-match sample, and shipped a README referencing two
    more that did not exist. The stamp had worked exactly as designed — and a
    stamp is a warning, which is something a person can walk past.

    `refuse_illustrative_output` turns the warning into a refusal: it returns
    True (refuse) for the published folder and False (allow) for anywhere else,
    so previewing a chart's layout on sample data stays easy and publishing one
    stops being possible by accident.
    """
    published = REPO_ROOT / "docs" / "charts"

    # The published folder, without the override: refused.
    assert chart_script.refuse_illustrative_output(published, False) is True

    # The same folder with the override asked for by name: allowed.
    assert chart_script.refuse_illustrative_output(published, True) is False

    # Anywhere else: always allowed, override or not.
    elsewhere = REPO_ROOT / "build" / "preview"
    assert chart_script.refuse_illustrative_output(elsewhere, False) is False


def test_the_refusal_is_decided_before_any_work_is_done(chart_script):
    """`will_use_sample` must answer from the filesystem alone.

    The first version of this guard ran after `gather()`, which meant a run
    doomed to be refused still paid for a full pipeline — on the football
    project, an entire walk-forward backtest — before saying so.
    """
    from src.config import load_config
    config = load_config()

    # Forced sample mode is knowable without touching the pipeline.
    assert chart_script.will_use_sample(config, forced=True) is True
    # And the unforced answer is a bool, derived from whether raw data exists.
    assert isinstance(chart_script.will_use_sample(config, forced=False), bool)


def test_unverified_numbers_carry_a_notice_and_verified_ones_do_not():
    """The README may assert figures OR flag them, never neither, never both.

    Two directions, one test:

    * No `numbers.json` means no run of this repository has produced the
      figures in the README. They stay, because deleting the results section
      would be worse, but they must be marked unverified.
    * Once `numbers.json` exists, the other tests in this module check every
      figure against it — so the notice is then false and must be removed.

    This exists because of 21 October 2026, when this README was committed
    quoting numbers no run on this machine had ever produced, with no mark
    saying so.
    """
    readme = README.read_text(encoding="utf-8")
    flagged = "awaiting regeneration" in readme.lower()

    if NUMBERS.exists():
        assert not flagged, (
            "docs/charts/numbers.json exists, so the figures are verified "
            "against a real run — remove the 'Awaiting regeneration' notice "
            "from the README."
        )
    else:
        assert flagged, (
            "no docs/charts/numbers.json, so nothing has verified the figures "
            "in this README. Either regenerate them with "
            "`python scripts/make_charts.py` or mark the section "
            "'Awaiting regeneration'."
        )


# --- The data verifier -----------------------------------------------------

def test_the_data_verifier_rejects_the_three_ways_a_mirror_fails(tmp_path):
    """Any candidate file must be checkable before the pipeline trusts it.

    football-data.co.uk is blocked on some networks, so mirrors get used. They
    fail in three characteristic ways, and all three produce a file that looks
    right from the outside:

      1. the closing-odds columns are dropped  -> CLV cannot be computed
      2. closing odds are a copy of the opening -> CLV reads zero, always
      3. an HTML error page saved with a .csv name

    The third is caught by any reader. The first two are not: the pipeline
    would run, produce charts, and report a CLV of zero as though it had
    measured something.
    """
    import importlib.util
    import numpy as np
    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "verify_raw_data", REPO_ROOT / "scripts" / "verify_raw_data.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    from src.config import load_config
    config = load_config()

    # A plausible season: coherent books, sensible goals, sharper close.
    rng = np.random.default_rng(0)
    n = 380
    home = rng.uniform(0.25, 0.60, n)
    draw = (1 - home) * 0.45
    away = 1 - home - draw
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-08-10", periods=n, freq="D").strftime("%d/%m/%Y"),
        "HomeTeam": "A", "AwayTeam": "B",
        "FTHG": rng.poisson(1.5, n), "FTAG": rng.poisson(1.2, n),
        "B365H": 1 / (home * 1.05), "B365D": 1 / (draw * 1.05),
        "B365A": 1 / (away * 1.05),
        "PSCH": 1 / (home * 1.02), "PSCD": 1 / (draw * 1.02),
        "PSCA": 1 / (away * 1.02),
    })
    frame["FTR"] = np.where(frame["FTHG"] > frame["FTAG"], "H",
                            np.where(frame["FTHG"] < frame["FTAG"], "A", "D"))

    good = tmp_path / "good.csv"
    frame.to_csv(good, index=False)
    verdict, failures, _, _ = verifier.describe(good, config)
    assert not failures, f"a sound file was rejected: {failures}"
    assert verdict.startswith("USABLE")

    # 1. closing odds dropped
    dropped = tmp_path / "no_closing.csv"
    frame.drop(columns=["PSCH", "PSCD", "PSCA"]).to_csv(dropped, index=False)
    _, failures, _, _ = verifier.describe(dropped, config)
    assert any("closing odds" in f for f in failures)

    # 2. closing odds copied from the opening price
    copied = tmp_path / "copied.csv"
    duplicate = frame.copy()
    duplicate[["PSCH", "PSCD", "PSCA"]] = duplicate[["B365H", "B365D", "B365A"]].to_numpy()
    duplicate.to_csv(copied, index=False)
    _, failures, _, _ = verifier.describe(copied, config)
    assert any("identical to" in f for f in failures), (
        "a closing column copied from the opening price must be refused — "
        "otherwise CLV is measured against itself and reads zero"
    )

    # 3. an error page wearing a .csv extension
    page = tmp_path / "error.csv"
    page.write_text("<html><body>404 Not Found</body></html>\n")
    _, failures, _, _ = verifier.describe(page, config)
    assert failures


def test_the_verifier_checks_the_odds_are_a_coherent_book(tmp_path):
    """Headers can be right while the numbers underneath are nonsense.

    Three decimal prices from a real bookmaker imply probabilities summing to a
    little over 1 — the overround. Prices that sum to 0.84, or to 1.6, did not
    come from a book, whatever the column is called.
    """
    import importlib.util
    import numpy as np
    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "verify_raw_data", REPO_ROOT / "scripts" / "verify_raw_data.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    from src.config import load_config
    config = load_config()

    n = 380
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-08-10", periods=n, freq="D").strftime("%d/%m/%Y"),
        "HomeTeam": "A", "AwayTeam": "B",
        "FTHG": 2, "FTAG": 1, "FTR": "H",
        # Prices that imply probabilities summing to ~0.6 — not a book.
        "B365H": 5.0, "B365D": 5.0, "B365A": 5.0,
        "PSCH": 2.0, "PSCD": 3.5, "PSCA": 4.0,
    })
    path = tmp_path / "nonsense_book.csv"
    frame.to_csv(path, index=False)

    _, failures, _, _ = verifier.describe(path, config)
    assert any("coherent set of prices" in f for f in failures)


def test_the_simulation_study_is_labelled_as_simulated_in_the_readme(readme):
    """The one thing a simulated headline must never do is look measured.

    The README leads with a result from generated data. That is legitimate —
    it validates the instrument — but only while every reader can see at a
    glance that no real market was involved.
    """
    lowered = collapse(readme)
    assert "simulat" in lowered
    assert ("makes no claim about real" in lowered
            or "no claim about real betting" in lowered
            or "not a claim about real" in lowered), (
        "the README must state explicitly that the simulation makes no claim "
        "about real betting markets"
    )



# --- Added 26 Oct 2026: the simulation table is held to its JSON too ---------

SIMULATION = REPO_ROOT / "docs" / "charts" / "simulation_study.json"
CALIBRATION = REPO_ROOT / "docs" / "charts" / "simulation_calibration.json"


def test_readme_quotes_the_simulation_study_as_recorded(readme):
    """The README's simulation table had no test until 26 Oct 2026, so the
    rule "no figure a command cannot regenerate" covered the real numbers and
    not the simulated ones. When the simulator was recalibrated, every figure
    in that table changed. This makes the next change impossible to miss."""
    if not SIMULATION.exists():
        pytest.skip("simulation_study.json is absent")
    study = json.loads(SIMULATION.read_text())
    arms = [level for level in study["levels"] if level.get("bets", 0)]
    arms.append(study["noise_control"])
    for arm in arms:
        name = f"arm with mispricing {arm.get('mispricing_points', 'control')}"
        for key in ("clv_points", "clv_low", "clv_high", "margin_paid", "movement"):
            assert_quoted(readme, arm[key], f"{name}: {key}")


def test_readme_quotes_the_simulator_calibration(readme):
    if not CALIBRATION.exists():
        pytest.skip("simulation_calibration.json is absent")
    calibration = json.loads(CALIBRATION.read_text())
    target = calibration["real_target"]
    for key in ("estimate", "lower", "upper"):
        assert_quoted(readme, target[key], f"the real calibration target ({key})")
    assert_quoted(readme, calibration["closest_share"], "the calibrated share")
