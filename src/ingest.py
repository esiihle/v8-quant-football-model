"""
Stage 1 — Data ingestion.

Phase 0 (now): load a CSV of historical matches + odds and produce a "shape
report" so we can eyeball what we've got. Light column-presence checks only.

Phase 1 (next): strict schema validation — types, ranges, duplicate fixtures,
date parsing, and loud failure on anything malformed. Look for the ``PHASE 1``
marker below for exactly where that logic goes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_matches(path, config=None):
    """Load a matches CSV into a DataFrame with light checks.

    Parameters
    ----------
    path : str | Path
        Path to the CSV (football-data.co.uk style columns).
    config : dict | None
        Optional config. If given, ``schema.required_columns`` is checked so a
        missing column fails here, loudly, rather than three stages downstream.

    Returns
    -------
    pandas.DataFrame
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Light column-presence check. Full validation arrives in Phase 1.
    if config is not None:
        required = config.get("schema", {}).get("required_columns", [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    # PHASE 1 goes here: parse Date -> datetime, enforce dtypes, drop/flag
    # duplicate fixtures, check goals are non-negative integers, and reconcile
    # FTR against (FTHG, FTAG).
    return df


def build_shape_report(df):
    """Return a short, human-readable summary of a matches DataFrame.

    Reports row/column counts, date coverage (if a ``Date`` column exists), the
    number of distinct teams, and per-column null counts. Written defensively so
    it works even before Phase 1 cleaning (e.g. dates may still be strings).
    """
    lines = []
    lines.append(f"rows: {len(df):,}")
    lines.append(f"columns: {len(df.columns)}")

    if "Date" in df.columns and len(df) > 0:
        lines.append(f"date range: {df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}")

    team_cols = [c for c in ("HomeTeam", "AwayTeam") if c in df.columns]
    if team_cols:
        teams = pd.unique(df[team_cols].values.ravel())
        lines.append(f"distinct teams: {len(teams)}")

    null_counts = df.isna().sum()
    nulls = null_counts[null_counts > 0]
    if len(nulls) == 0:
        lines.append("nulls: none")
    else:
        lines.append("nulls: " + ", ".join(f"{c}={n}" for c, n in nulls.items()))

    return "\n".join(lines)
