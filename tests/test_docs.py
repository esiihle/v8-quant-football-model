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
    return json.loads(NUMBERS.read_text())


# --- Always run ------------------------------------------------------------

def test_readme_only_links_charts_the_script_writes(readme, chart_script):
    """A broken image in a README is a broken promise on the front page."""
    linked = set(re.findall(r"docs/charts/([\w.-]+\.png)", readme))
    produced = set(chart_script.CHARTS.values())

    unknown = linked - produced
    assert not unknown, (
        f"README links charts that make_charts.py does not write: {sorted(unknown)}"
    )
    assert linked, "the README should show the charts"


def test_every_chart_the_script_writes_is_shown(chart_script, readme):
    """A chart nobody looks at is a chart nobody maintains."""
    produced = set(chart_script.CHARTS.values())
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
    """Only meaningful once charts are committed; skips before that."""
    linked = set(re.findall(r"docs/charts/([\w.-]+\.png)", readme))
    if not any((CHART_DIR / name).exists() for name in linked):
        pytest.skip("no charts committed yet — run scripts/make_charts.py")

    missing = [name for name in linked if not (CHART_DIR / name).exists()]
    assert not missing, f"README shows charts that are not in the repo: {missing}"
