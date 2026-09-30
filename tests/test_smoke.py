"""
Phase 0 smoke tests.

These don't test model correctness (those tests arrive with each phase). They
check the scaffolding holds together: config loads, every module imports, the
sample data loads, and the shape report renders. A green run here means a fresh
clone is wired correctly.

Run with:  pytest
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import (
    ingest,
    features,
    dixon_coles,
    calibrate,
    staking,
    backtest,
    clv,
    metrics,
)


def test_config_loads():
    """The config file parses and has the sections we rely on."""
    config = load_config()
    assert "data" in config
    assert "schema" in config


def test_sample_data_loads_and_reports():
    """The bundled sample loads and the shape report renders."""
    config = load_config()
    sample = REPO_ROOT / config["data"]["sample_path"]
    df = ingest.load_matches(sample, config)
    assert len(df) > 0
    report = ingest.build_shape_report(df)
    assert "rows:" in report


def test_missing_required_column_fails_loudly():
    """A required column absent from the data raises a clear error."""
    import pandas as pd

    bad = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["B"]})  # missing goals etc.
    bad_path = REPO_ROOT / "data" / "sample" / "_tmp_bad.csv"
    bad.to_csv(bad_path, index=False)
    try:
        config = load_config()
        with pytest.raises(ValueError):
            ingest.load_matches(bad_path, config)
    finally:
        bad_path.unlink(missing_ok=True)


def test_downstream_stages_declared_but_pending():
    """Every downstream stage exists and cleanly signals 'not built yet'."""
    for call in (
        lambda: features.build_features(None),
        lambda: dixon_coles.fit(None),
        lambda: calibrate.calibrate(None, None),
        lambda: staking.stake(None, None),
        lambda: backtest.walk_forward(None),
        lambda: clv.compute_clv(None, None),
        lambda: metrics.summarise(None),
    ):
        with pytest.raises(NotImplementedError):
            call()


# --- Added 30 Sep 2026: real-data discovery -------------------------------


def test_expected_raw_files_are_named_and_linked_correctly():
    """Every configured league-season maps to a filename and a download URL."""
    config = load_config()
    entries = ingest.expected_raw_files(config)

    assert len(entries) == (
        len(config["data"]["leagues"]) * len(config["data"]["seasons"])
    )
    for entry in entries:
        # Filename follows the configured pattern, e.g. E0_2425.csv
        assert entry["path"].name == config["data"]["file_pattern"].format(
            league=entry["league"], season=entry["season"]
        )
        # And we can always tell the user where to get it.
        assert entry["season"] in entry["url"]
        assert entry["league"] in entry["url"]


def test_shape_report_flags_odds_coverage():
    """With a config, the report says whether closing odds (CLV) are present."""
    config = load_config()
    sample = REPO_ROOT / config["data"]["sample_path"]
    df = ingest.load_matches(sample, config)

    report = ingest.build_shape_report(df, config)
    assert "closing odds (CLV)" in report
