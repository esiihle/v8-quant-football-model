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
    return _load_with_extends(config_path, seen=())


def _deep_merge(base, override):
    """Merge `override` into a copy of `base`.

    Dicts merge key by key, recursively. Anything else, lists included, is
    REPLACED: an overlay that names `seasons` means exactly those seasons, not
    the base's seasons plus these.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_with_extends(config_path, seen):
    """Load one YAML file, resolving an optional `extends:` key. [27 Oct 2026]

    A config may begin with ``extends: <path>``, resolved relative to the file
    that says it. The base is loaded first and this file's keys are merged over
    it, so a variant states only what differs. That is the point: two full
    copies of a 200-line config drift, and a difference nobody meant to make
    becomes a difference in results that nobody can see.

    The run fingerprint is computed on the MERGED config, so a variant always
    fingerprints differently from its base, and a base edit changes both.
    """
    config_path = Path(config_path).resolve()
    if config_path in seen:
        chain = " -> ".join(str(p.name) for p in (*seen, config_path))
        raise ValueError(f"Config files extend each other in a loop: {chain}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    parent = config.pop("extends", None)
    if parent is None:
        return config
    base = _load_with_extends(config_path.parent / parent, (*seen, config_path))
    return _deep_merge(base, config)


def set_seed(seed):
    """Fix Python and NumPy random seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
