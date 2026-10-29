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

# Chart links may sit in ONE subfolder of docs/charts (the long run's). Until
# 27 Oct 2026 the pattern had no "/" in it, so a chart in a subfolder was
# invisible to every link test: broken, missing or stale, nothing would fail.
CHART_LINK = r"docs/charts/((?:[\w-]+/)?[\w.-]+\.png)"

# The 13-season run [added 27 Oct 2026]. It publishes a SUBSET of the charts:
# the two that say something the headline's charts do not. Generate exactly
# these, so the folder holds nothing the README does not show:
#   python scripts/make_charts.py --config configs/epl_2012_2025.yaml
#       --out docs/charts/epl_2012_2025 --only bankroll --only clv_by_edge
LONG_RUN_DIR = REPO_ROOT / "docs" / "charts" / "epl_2012_2025"
LONG_RUN_CHARTS = {"bankroll-curve.png", "clv-by-claimed-edge.png"}


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
    linked = set(re.findall(CHART_LINK, readme))
    produced = (set(chart_script.CHARTS.values())
                | set(simulation_script.CHARTS.values())
                | {f"{LONG_RUN_DIR.name}/{name}" for name in LONG_RUN_CHARTS})

    unknown = linked - produced
    assert not unknown, (
        f"README links charts that make_charts.py does not write: {sorted(unknown)}"
    )
    assert linked, "the README should show the charts"


def test_every_chart_the_script_writes_is_shown(chart_script, readme,
                                                simulation_script):
    """A chart nobody looks at is a chart nobody maintains."""
    produced = (set(chart_script.CHARTS.values())
                | set(simulation_script.CHARTS.values())
                | {f"{LONG_RUN_DIR.name}/{name}" for name in LONG_RUN_CHARTS})
    linked = set(re.findall(CHART_LINK, readme))

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
    linked = set(re.findall(CHART_LINK, readme))
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



# --- Added 27 Oct 2026: the 13-season run is held to its own numbers --------

@pytest.fixture(scope="module")
def long_numbers():
    path = LONG_RUN_DIR / "numbers.json"
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO_ROOT)} is absent - run make_charts.py "
                    "with configs/epl_2012_2025.yaml (see LONG_RUN_DIR)")
    recorded = json.loads(path.read_text())
    if recorded.get("real_data") is not True:
        pytest.fail("the long run's numbers.json came from sample data")
    return recorded


def test_the_long_run_folder_holds_only_what_the_readme_shows():
    """A subset is published, so the folder must not quietly gain the rest."""
    if not LONG_RUN_DIR.exists():
        pytest.skip("long run not generated")
    on_disk = {path.name for path in LONG_RUN_DIR.glob("*.png")}
    assert on_disk == LONG_RUN_CHARTS, (
        f"docs/charts/{LONG_RUN_DIR.name} holds {sorted(on_disk)}; expected "
        f"exactly {sorted(LONG_RUN_CHARTS)}. Regenerate with --only.")


def test_the_long_run_used_thirteen_seasons_and_its_own_fingerprint(long_numbers,
                                                                    numbers):
    """Its numbers must come from the long config, not the headline's."""
    from src.config import load_config
    from src.logging_setup import run_fingerprint
    expected = run_fingerprint(load_config(REPO_ROOT / "configs" / "epl_2012_2025.yaml"))
    assert long_numbers["run_fingerprint"] == expected
    assert long_numbers["run_fingerprint"] != numbers["run_fingerprint"]
    assert long_numbers["matches"] > 10 * 380


def test_readme_quotes_the_long_run_as_recorded(readme, long_numbers):
    for key in ("walk_forward_roi", "walk_forward_yield", "walk_forward_yield_low",
                "walk_forward_yield_high", "clv_points_mean", "clv_points_low",
                "clv_points_high", "clv_margin_paid", "clv_movement_mean",
                "clv_movement_low", "clv_movement_high"):
        assert_quoted(readme, long_numbers[key], f"long run: {key}")
    assert f"{long_numbers['walk_forward_bets']:,}" in readme
    assert long_numbers["run_fingerprint"] in readme


def test_the_long_run_verdict_follows_its_numbers(readme, long_numbers):
    """Same rule as the headline: the prose verdict may not outlive the data."""
    claims_none = "no detectable movement over 13 seasons" in collapse(readme)
    assert claims_none == (long_numbers.get("clv_movement_verdict") == "none")


# --- Added 27 Oct 2026: two chart faults visible only on the long run -------

def test_tick_labels_are_never_rounded_into_a_different_number(chart_script):
    """Ticks at -0.25% were printed "-0.3%": a label that disagrees with its
    own gridline. Every tick keeps exactly the decimals it needs."""
    label = chart_script.tick_pct
    assert label(-0.0025) == "-0.25%"
    assert label(-0.0075) == "-0.75%"
    assert label(0.01) == "+1%"
    assert label(-0.005) == "-0.5%"
    assert label(0.0) == "+0%"


def test_a_ruined_bankroll_is_drawn_on_a_log_axis(chart_script):
    """The headline bankroll ends near 15% of its start: linear axis. The long
    run's falls to about 0.02%: on a linear axis it lies flat on zero for eight
    years, so it gets a log axis. A bankroll at exactly zero cannot be logged
    and stays linear."""
    rule = chart_script.use_log_scale
    assert rule([1000, 800, 150], 1000) is False
    assert rule([1000, 300, 0.2], 1000) is True
    assert rule([1000, 0.0], 1000) is False
    assert rule([], 1000) is False


# --- Added 29 Oct 2026: the exploration, held like everything else ----------

METHODOLOGY = REPO_ROOT / "docs" / "methodology.md"
NOTEBOOK = REPO_ROOT / "notebooks" / "01_exploration.ipynb"
EXPLORATIONS = {"headline": REPO_ROOT / "docs" / "charts" / "exploration.json",
                "long run": LONG_RUN_DIR / "exploration.json"}


def load_exploration(name):
    path = EXPLORATIONS[name]
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO_ROOT)} is absent - run scripts/explore.py")
    found = json.loads(path.read_text())
    assert found.get("real_data") is True, f"{name} exploration.json came from sample data"
    return found


@pytest.mark.parametrize("name", sorted(EXPLORATIONS))
def test_methodology_quotes_the_exploration_as_measured(name):
    """Methodology §1.5 is the findings table the notebook used to ask for by
    hand. Every figure in it must match exploration.json."""
    found = load_exploration(name)
    text = METHODOLOGY.read_text(encoding="utf-8")
    for key in ("home_win", "draw", "away_win"):
        assert_quoted(text, found[key]["share"], f"{name}: {key}")
    for score, row in found["low_scores"].items():
        for key in ("departure_per_match", "departure_low", "departure_high"):
            assert_quoted(text, row[key], f"{name}: {score} {key}")
    assert_quoted(text, found["low_scores"]["1-1"]["departure_league_mean"],
                  f"{name}: 1-1 league-mean departure")
    for key in ("overround_bet_into", "overround_closing"):
        assert_quoted(text, found[key], f"{name}: {key}")
    # Plain decimals, not percentages: quoted to the precision shown.
    for value, places, label in (
            (found["market_ece_home"], 4, "market ECE"),
            (found["market_ece_home_noise_floor_p95"], 4, "market ECE floor"),
            (found["shots_on_target_r_home"]["r"], 3, "shots-on-target r"),
            (found["season_rho_sign_test_p"], 3 if found["season_rho_sign_test_p"] < 0.1 else 2,
             "rho sign test p")):
        assert f"{value:.{places}f}" in text, f"{name}: methodology does not quote {label}"
    assert (f"{found['season_rho_negative']} of {len(found['season_rho'])}" in text)


@pytest.mark.parametrize("name, config_path", [
    ("headline", REPO_ROOT / "config.yaml"),
    ("long run", REPO_ROOT / "configs" / "epl_2012_2025.yaml")])
def test_each_exploration_came_from_its_own_config(name, config_path):
    from src.config import load_config
    from src.logging_setup import run_fingerprint
    found = load_exploration(name)
    assert found["run_fingerprint"] == run_fingerprint(load_config(config_path))


def notebook_text_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    texts, errors, code_cells, executed = [], [], 0, 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        executed += bool(cell.get("outputs"))
        for output in cell.get("outputs", []):
            if output["output_type"] == "error":
                errors.append(f"{output.get('ename')}: {output.get('evalue')}")
            texts.append("".join(output.get("text", "")))
            texts.append("".join(output.get("data", {}).get("text/plain", "")))
    return "\n".join(texts), errors, code_cells, executed


def test_the_notebook_was_run_on_real_data_with_the_current_config():
    """Regression, 29 Oct 2026: the committed notebook showed the 12-match
    sample's numbers for four weeks, because nothing looked at its outputs.

    The notebook must be saved after a full run, on real data, with no errors,
    and with the fingerprint of the current config.yaml - so a config change
    (as on 27 Oct) makes a stale notebook fail here instead of misleading a
    reader on GitHub."""
    from src.config import load_config
    from src.logging_setup import run_fingerprint
    text, errors, code_cells, executed = notebook_text_outputs()
    assert executed == code_cells, (
        f"only {executed} of {code_cells} code cells have saved output: "
        "Restart & Run All, then save")
    assert not errors, f"the saved notebook contains errors: {errors}"
    assert "REAL data from data/raw" in text
    assert "BUNDLED SAMPLE" not in text
    fingerprint = run_fingerprint(load_config(REPO_ROOT / "config.yaml"))
    assert f"run fingerprint: {fingerprint}" in text, (
        "the notebook's saved outputs come from a different config - rerun it")


def test_readme_reads_the_reliability_diagram_against_its_noise_floor(readme, numbers):
    """Added 29 Oct 2026. A calibration error on a few hundred matches is
    mostly noise; the README must give the floor beside the diagram."""
    if "ece_noise_floor_p95" not in numbers:
        pytest.fail("numbers.json predates the noise floor: rerun make_charts.py")
    assert_quoted(readme, numbers["ece_noise_floor_p95"], "the ECE noise floor")
    assert f"{numbers['ece_matches']} matches" in readme
    claims_undetectable = "cannot\nshow miscalibration" in readme or \
        "cannot show miscalibration" in collapse(readme)
    assert claims_undetectable == (not numbers["ece_detectable"])
