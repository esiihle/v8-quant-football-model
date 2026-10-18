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


def test_every_pipeline_stage_is_now_built():
    """Phase 0 kept a list of stages that still raised NotImplementedError.

    As of Phase 6 (18 Oct 2026) that list is empty: ingest, features,
    dixon_coles, calibrate, staking, backtest, clv and metrics are all
    implemented and each has its own test module. This test replaces the old
    "declared but pending" check, so the suite records that the scaffold is
    fully filled in rather than silently dropping the idea.
    """
    import inspect

    stages = {
        "ingest": ingest.run_ingest,
        "features": features.build_features,
        "dixon_coles": dixon_coles.fit,
        "calibrate": calibrate.evaluate,
        "staking": staking.find_value_bets,
        "backtest": backtest.walk_forward,
        "clv": clv.clv_summary,
        "metrics": metrics.summarise,
    }

    for name, function in stages.items():
        source = inspect.getsource(function)
        assert "NotImplementedError" not in source, (
            f"{name} still raises NotImplementedError"
        )

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
