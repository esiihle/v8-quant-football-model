"""
Phase 7 tests — hardening.

These check the things that make a result trustworthy rather than the results
themselves: dependencies pinned to exact versions, a run fingerprint that
changes when and only when it should, logging that is configurable and
idempotent, and a clean clone that works.

Run with:  pytest tests/test_hardening.py -v
"""
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import logging_setup


# =========================================================================
# Reproducibility: pinned dependencies
# =========================================================================

def test_every_dependency_is_pinned_to_an_exact_version():
    """`>=` is fine while building and wrong once results are being quoted.

    A loose pin means a reader who clones this repo next year installs a
    different pandas and may get different numbers — which makes every figure in
    the README unverifiable.
    """
    requirements = (REPO_ROOT / "requirements.txt").read_text().splitlines()

    unpinned = []
    for line in requirements:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            unpinned.append(line)

    assert not unpinned, f"these dependencies are not pinned exactly: {unpinned}"


def test_requirements_cover_everything_the_code_imports():
    r"""A dependency used but not declared works on the author's machine and
    fails on a clean clone — the classic way a finished project is not.

    Imports are found by parsing each module with `ast`, not by regex. The first
    version of this test matched `^\s*(from|import)\s+(\w+)` against raw lines
    and picked up the word "average" from the comment "# Average ranks so tied
    scores...". A regex cannot tell code from prose; a parser can.
    """
    import ast

    declared = set()
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            declared.add(re.split(r"[=<>\[]", line)[0].lower())

    # Import name -> distribution name, where they differ.
    aliases = {"sklearn": "scikit-learn", "yaml": "pyyaml"}

    third_party = set()
    for path in (REPO_ROOT / "src").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue

            for name in names:
                top = name.split(".")[0]
                if top in ("src", "__future__") or top in sys.stdlib_module_names:
                    continue
                third_party.add(aliases.get(top, top).lower())

    missing = third_party - declared
    assert not missing, f"imported but not in requirements.txt: {sorted(missing)}"


def test_config_is_valid_yaml_and_has_every_section_the_code_reads():
    config = load_config()
    for section in ("data", "schema", "validation", "features", "model",
                    "calibration", "staking", "backtest"):
        assert section in config, f"config.yaml is missing '{section}'"


# =========================================================================
# The run fingerprint
# =========================================================================

def test_the_same_config_gives_the_same_fingerprint():
    config = load_config()
    assert (logging_setup.run_fingerprint(config)
            == logging_setup.run_fingerprint(load_config()))


def test_changing_a_result_affecting_setting_changes_the_fingerprint():
    """Otherwise the fingerprint would certify runs that are not comparable."""
    config = load_config()
    original = logging_setup.run_fingerprint(config)

    for section, key, value in (("features", "shrinkage_k", 99),
                                ("model", "penalty", 0.5),
                                ("staking", "kelly_fraction", 0.9),
                                ("backtest", "refit_every_n_days", 3)):
        changed = dict(config)
        changed[section] = dict(config[section], **{key: value})
        assert logging_setup.run_fingerprint(changed) != original, (
            f"changing {section}.{key} did not change the fingerprint"
        )


def test_changing_the_seed_changes_the_fingerprint():
    config = load_config()
    changed = dict(config, seed=config["seed"] + 1)
    assert (logging_setup.run_fingerprint(changed)
            != logging_setup.run_fingerprint(config))


def test_cosmetic_changes_do_not_change_the_fingerprint():
    """Log level and key order are not results, so they must not invalidate a
    fingerprint — otherwise nobody would trust it."""
    config = load_config()
    original = logging_setup.run_fingerprint(config)

    noisier = dict(config, logging={"level": "DEBUG"})
    assert logging_setup.run_fingerprint(noisier) == original

    reordered = {key: config[key] for key in reversed(list(config))}
    assert logging_setup.run_fingerprint(reordered) == original


def test_run_header_carries_what_identifies_a_run():
    header = logging_setup.run_header(load_config())
    for field in ("started_utc", "fingerprint", "seed", "leagues", "seasons"):
        assert field in header


# =========================================================================
# Logging
# =========================================================================

def test_logging_writes_to_a_file_when_asked(tmp_path):
    log_path = tmp_path / "run.log"
    logger = logging_setup.configure("INFO", log_file=log_path)
    logger.info("a test line")

    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists()
    assert "a test line" in log_path.read_text()


def test_configuring_twice_does_not_duplicate_every_line(tmp_path):
    """The tests call configure() repeatedly; without clearing handlers each
    line would appear once per call, which is the classic logging bug."""
    log_path = tmp_path / "run.log"

    logging_setup.configure("INFO", log_file=log_path)
    logger = logging_setup.configure("INFO", log_file=log_path)
    logger.info("exactly once")

    for handler in logger.handlers:
        handler.flush()

    assert log_path.read_text().count("exactly once") == 1


def test_log_level_is_respected(tmp_path):
    log_path = tmp_path / "run.log"
    logger = logging_setup.configure("WARNING", log_file=log_path)

    logger.debug("should not appear")
    logger.warning("should appear")
    for handler in logger.handlers:
        handler.flush()

    contents = log_path.read_text()
    assert "should appear" in contents
    assert "should not appear" not in contents


def test_module_loggers_are_namespaced():
    logger = logging_setup.get_logger("ingest")
    assert logger.name == "v8.ingest"


def test_the_run_header_is_logged(tmp_path):
    log_path = tmp_path / "run.log"
    logger = logging_setup.configure("INFO", log_file=log_path)
    header = logging_setup.log_run_header(logger, load_config())

    for handler in logger.handlers:
        handler.flush()

    contents = log_path.read_text()
    assert "run starting" in contents
    assert header["fingerprint"] in contents


# =========================================================================
# A clean clone
# =========================================================================

def test_the_bundled_sample_is_committed():
    """The sample exists so a fresh clone runs with no download. If it is not
    in the repo, CI and every recruiter hit a missing-file error first."""
    config = load_config()
    sample = REPO_ROOT / config["data"]["sample_path"]
    assert sample.exists(), f"the bundled sample is missing: {sample}"
    assert sample.stat().st_size > 0


def test_raw_and_processed_data_are_git_ignored():
    """Derived and downloaded data must not be committed; the folders must be."""
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "data/raw/*" in gitignore
    assert "data/processed/*" in gitignore
    assert "!data/raw/.gitkeep" in gitignore


def test_ci_workflow_exists_and_runs_the_suite():
    """CI is the only thing that actually proves a clean clone works."""
    workflow_path = REPO_ROOT / ".github" / "workflows" / "tests.yml"
    assert workflow_path.exists(), "no CI workflow"

    workflow = yaml.safe_load(workflow_path.read_text())
    steps = workflow["jobs"]["test"]["steps"]
    commands = " ".join(step.get("run", "") for step in steps)

    assert "pytest" in commands, "CI does not run the tests"
    assert "requirements.txt" in commands, "CI does not install pinned deps"
    assert "run_pipeline.py" in commands, (
        "CI does not check the pipeline runs on the sample"
    )


def test_ci_tests_more_than_one_python_version():
    workflow_path = REPO_ROOT / ".github" / "workflows" / "tests.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    versions = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert len(versions) >= 2, f"only tests Python {versions}"
