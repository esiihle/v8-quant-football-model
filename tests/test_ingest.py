"""
Phase 1 tests — ingestion, validation and the canonical table.

Unlike the smoke tests, these check behaviour: each one feeds in a specific
defect and asserts what the pipeline does about it. Written so a failure names
the actual problem, not just "assert False".

Run with:  pytest tests/test_ingest.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import canonical, ingest


# --------------------------------------------------------------------------
# A tiny, deliberately clean fixture. Each test corrupts a copy of it, so the
# thing being tested is always the single change that test makes.
# --------------------------------------------------------------------------
def make_raw(n=6):
    """Build a small raw frame in football-data.co.uk shape.

    Values cycle, so any n works — the tests that need a bigger file get one
    without a second fixture to keep in step with this one.
    """
    dates = pd.date_range("2024-08-17", periods=n, freq="7D")

    def cycle(values):
        """Repeat a short list until it is n long."""
        return [values[i % len(values)] for i in range(n)]

    home_goals = cycle([2, 1, 0, 3, 1, 1])
    away_goals = cycle([1, 1, 2, 0, 0, 3])
    results = ["H" if h > a else "A" if a > h else "D"
               for h, a in zip(home_goals, away_goals)]

    return pd.DataFrame({
        "Div": "E0",
        "Date": dates.strftime("%d/%m/%Y"),
        "HomeTeam": [f"Team {i}" for i in range(n)],
        "AwayTeam": [f"Team {n + i}" for i in range(n)],
        "FTHG": home_goals,
        "FTAG": away_goals,
        "FTR": results,
        "HS": cycle([12, 9, 7, 15, 11, 8]),
        "AS": cycle([8, 10, 13, 6, 9, 14]),
        "HST": cycle([5, 3, 2, 7, 4, 3]),
        "AST": cycle([3, 4, 6, 2, 3, 5]),
        "B365H": cycle([2.10, 2.50, 3.20, 1.80, 2.20, 3.00]),
        "B365D": cycle([3.40, 3.30, 3.40, 3.60, 3.30, 3.40]),
        "B365A": cycle([3.60, 2.90, 2.30, 4.50, 3.40, 2.40]),
        "PSCH": cycle([2.05, 2.45, 3.15, 1.78, 2.15, 2.95]),
        "PSCD": cycle([3.50, 3.35, 3.45, 3.70, 3.40, 3.50]),
        "PSCA": cycle([3.70, 2.95, 2.35, 4.60, 3.50, 2.45]),
        "League": "E0",
        "Season": "2425",
    })


@pytest.fixture
def config():
    return load_config()


# --- Structural validation -------------------------------------------------

def test_missing_required_column_stops_the_run(config):
    """A missing required column is fatal, and the error names the column."""
    raw = make_raw().drop(columns=["FTHG"])
    with pytest.raises(ingest.DataQualityError) as err:
        ingest.validate_raw(raw, config)
    assert "FTHG" in str(err.value)


def test_empty_input_stops_the_run(config):
    with pytest.raises(ingest.DataQualityError):
        ingest.validate_raw(make_raw().iloc[0:0], config)


# --- Cleaning --------------------------------------------------------------

def test_unparseable_date_row_is_dropped_and_counted(config):
    raw = make_raw()
    raw.loc[2, "Date"] = "not a date"

    cleaned, report = ingest.clean(raw, config)

    assert report["dropped_unparseable_date"] == 1
    assert len(cleaned) == len(raw) - 1


def test_whitespace_in_team_names_is_trimmed(config):
    """' Team 1' and 'Team 1' must not become two different teams."""
    raw = make_raw()
    raw.loc[1, "HomeTeam"] = "  Team 1  "

    cleaned, report = ingest.clean(raw, config)

    assert report["whitespace_trimmed_HomeTeam"] == 1
    assert "Team 1" in set(cleaned["HomeTeam"])
    assert "  Team 1  " not in set(cleaned["HomeTeam"])


def test_impossible_goal_values_are_dropped(config):
    raw = make_raw()
    raw.loc[0, "FTHG"] = 99        # above validation.max_goals
    raw.loc[1, "FTAG"] = -1        # negative

    cleaned, report = ingest.clean(raw, config)

    assert report["dropped_implausible_goals"] == 2
    assert len(cleaned) == len(raw) - 2


def test_result_is_recomputed_from_goals_not_trusted(config):
    """The file says 'H'; the goals say away won. Goals win, and it's counted."""
    raw = make_raw()
    raw.loc[0, "FTR"] = "A"        # contradicts FTHG=2, FTAG=1

    cleaned, report = ingest.clean(raw, config)

    assert report["result_mismatches_corrected"] == 1
    assert cleaned.loc[0, "Result"] == "H"


def test_out_of_range_odds_are_voided_not_dropped(config):
    """An impossible price makes the match unbettable, not unusable."""
    raw = make_raw()
    raw.loc[3, "B365H"] = 0.5      # below validation.min_odds

    cleaned, report = ingest.clean(raw, config)

    assert report["odds_values_voided"] == 1
    assert len(cleaned) == len(raw)             # row survives
    assert pd.isna(cleaned.loc[3, "B365H"])     # price is gone


# --- Duplicates ------------------------------------------------------------

def test_duplicate_fixture_is_dropped(config):
    """Same date, same two teams = the same match, counted once."""
    raw = make_raw()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

    cleaned, _ = ingest.clean(raw, config)
    deduped, n_dupes = ingest.drop_duplicate_fixtures(cleaned, config)

    assert n_dupes == 1
    assert len(deduped) == len(make_raw())


# --- Canonical table -------------------------------------------------------

def test_canonical_table_shape_and_order(config):
    raw = make_raw()
    cleaned, _ = ingest.clean(raw, config)
    table = ingest.to_canonical(cleaned, config)

    # Exact columns, in the documented order.
    assert list(table.columns) == canonical.CANONICAL_COLUMNS
    # Chronological — Phase 6 depends on this.
    assert table["MatchDate"].is_monotonic_increasing
    # Identifiers are unique.
    assert table["MatchID"].is_unique
    # Raw names must not survive into the canonical table.
    assert "FTHG" not in table.columns


def test_canonical_fills_missing_optional_columns(config):
    """A file with no shot columns still produces the full canonical shape."""
    raw = make_raw().drop(columns=["HS", "AS", "HST", "AST"])
    cleaned, _ = ingest.clean(raw, config)
    table = ingest.to_canonical(cleaned, config)

    assert list(table.columns) == canonical.CANONICAL_COLUMNS
    assert table["HomeShots"].isna().all()


# --- The orchestrator and its guard rail -----------------------------------

def test_run_ingest_end_to_end(config, tmp_path):
    raw = make_raw()
    table, report = ingest.run_ingest(raw, config, save=False)

    assert len(table) == len(raw)
    assert report["rows_in"] == len(raw)
    assert report["rows_out"] == len(raw)
    assert "Data-quality report" in ingest.quality_report(report, table)


def test_run_ingest_refuses_to_silently_discard_most_of_the_file(config):
    """If over 5% of rows are unusable, stop. Quiet degradation is the enemy."""
    raw = make_raw(n=20)
    raw.loc[0:9, "Date"] = "not a date"      # half the file

    with pytest.raises(ingest.DataQualityError) as err:
        ingest.run_ingest(raw, config, save=False)
    assert "%" in str(err.value)
