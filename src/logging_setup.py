r"""
Structured logging.  [Built in Phase 7, 20 Oct 2026]

Why a logging module in a project this size
-------------------------------------------
Print statements are fine for a report a human reads once. They are useless for
the two things that matter when something goes wrong months later:

* **knowing which run produced a number.** A log line carries a timestamp and
  the run's configuration; a print statement carries neither.
* **separating the narrative from the diagnostics.** The pipeline's reports are
  written for a person to read. The log is written for whoever is working out
  why window 7 of a walk-forward produced no bets.

So both exist, deliberately. `scripts/run_pipeline.py` keeps printing its
reports, and the modules log their diagnostics. A `--log-file` run captures the
second without cluttering the first.

The run fingerprint
-------------------
`run_fingerprint` hashes the settings that affect results. Two runs with the same
fingerprint should produce the same numbers, which is what makes the Phase 7
reproducibility claim checkable rather than asserted: regenerate, compare the
fingerprint, compare the output.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure(level="INFO", log_file=None, quiet_console=False):
    """Set up logging for a run.

    Parameters
    ----------
    level : str
        "DEBUG", "INFO", "WARNING", "ERROR".
    log_file : str | Path | None
        Write a full log here as well as (or instead of) the console.
    quiet_console : bool
        Send only WARNING and above to the console. Useful when the reports are
        the thing being read and the log is going to a file.

    Returns
    -------
    logging.Logger
        The root project logger.
    """
    root = logging.getLogger("v8")
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    # Clear existing handlers so calling this twice in one process (which the
    # tests do) does not duplicate every line.
    root.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING if quiet_console else root.level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)      # the file always gets everything
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


def get_logger(name):
    """A child logger for one module, e.g. ``get_logger("ingest")``."""
    return logging.getLogger(f"v8.{name}")


# ===========================================================================
# Reproducibility
# ===========================================================================

# The config sections that change results. A change to anything here should
# change the fingerprint; a change to `logging` should not.
RESULT_AFFECTING = ["seed", "data", "schema", "validation", "features", "model",
                    "calibration", "staking", "backtest"]


def run_fingerprint(config):
    """A short hash of every setting that affects the numbers.

    Two runs with the same fingerprint should produce identical output. That is
    what turns "reproducible" from a claim into a check: note the fingerprint,
    rerun, compare.

    Deliberately excludes `logging`, and sorts keys, so cosmetic edits and
    key reordering in `config.yaml` do not invalidate a fingerprint.
    """
    relevant = {key: config[key] for key in RESULT_AFFECTING if key in config}
    serialised = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def run_header(config, extra=None):
    """A one-line summary of what this run is, for the top of a log.

    Everything needed to identify a run later: when, what seed, what data, and
    the fingerprint that ties it to its settings.
    """
    data = config.get("data", {})
    header = {
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "fingerprint": run_fingerprint(config),
        "seed": config.get("seed"),
        "leagues": data.get("leagues"),
        "seasons": data.get("seasons"),
    }
    if extra:
        header.update(extra)
    return header


def log_run_header(logger, config, extra=None):
    """Write the run header to the log as individual lines."""
    header = run_header(config, extra)
    logger.info("run starting")
    for key, value in header.items():
        logger.info("  %-14s %s", key, value)
    return header
