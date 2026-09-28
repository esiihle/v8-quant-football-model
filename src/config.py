"""
Configuration loading for the V8 pipeline.

Everything that controls a run lives in ``config.yaml`` at the repo root, so the
same command always produces the same behaviour and nothing important is
hard-coded. This module reads that file and hands back a plain dictionary, and
also fixes the global random seed for reproducibility.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

# config.yaml sits at the repo root, one level up from this file's folder (src/).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path=None):
    """Load the YAML config file and return it as a dictionary.

    Parameters
    ----------
    path : str | Path | None
        Path to the config file. If ``None``, the repo-root ``config.yaml`` is
        used.

    Returns
    -------
    dict
        The parsed configuration.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Pass --config or create config.yaml at the repo root."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Fix Python and NumPy random seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
