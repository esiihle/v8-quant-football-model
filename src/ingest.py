"""
Stage 1 — Data ingestion.

Phase 0 (done): load a CSV of historical matches + odds and produce a "shape
report" so we can eyeball what we've got. Light column-presence checks only.

30 Sep 2026: extended to work with REAL data from football-data.co.uk —
discovering the league/season files listed in the config, loading several of
them into one table, and reporting odds coverage (including closing odds, which
Phase 6 needs for CLV).

Phase 1 (next): strict schema validation — types, ranges, duplicate fixtures,
date parsing, and loud failure on anything malformed. Look for the ``PHASE 1``
marker below for exactly where that logic goes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo root = two levels up from this file (src/ingest.py -> src -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    """Turn a config path like 'data/raw' into an absolute path.

    Relative paths in config.yaml are written relative to the repo root, so the
    pipeline behaves the same no matter which folder you run it from.
    """
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def expected_raw_files(config):
    """List the raw files the config asks for, and whether each is present.

    Reads ``data.leagues``, ``data.seasons`` and ``data.file_pattern`` from the
    config and builds one entry per league-season combination.

    Returns
    -------
    list of dict
        Each dict has keys: ``league``, ``season``, ``path``, ``url``, ``exists``.
        Missing files are reported rather than raising, so the runner can print
        the exact download link instead of crashing.
    """
    data_cfg = config.get("data", {})
    raw_dir = _resolve(data_cfg.get("raw_dir", "data/raw"))
    pattern = data_cfg.get("file_pattern", "{league}_{season}.csv")
    url_pattern = data_cfg.get("download_url", "")

    files = []
    for league in data_cfg.get("leagues", []):
        for season in data_cfg.get("seasons", []):
            path = raw_dir / pattern.format(league=league, season=season)
            files.append(
                {
                    "league": league,
                    "season": season,
                    "path": path,
                    "url": url_pattern.format(league=league, season=season),
                    "exists": path.exists(),
                }
            )
    return files


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

    # Real football-data.co.uk files sometimes contain non-UTF-8 bytes (accented
    # team names, stray characters), so fall back to latin-1 rather than dying.
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    # Those files also carry fully-empty trailing columns and rows. Drop them so
    # the shape report reflects real content, not spreadsheet padding.
    df = df.dropna(axis="columns", how="all").dropna(axis="index", how="all")

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


def load_many(file_entries, config=None):
    """Load several raw files into one table, tagging each row with its source.

    Parameters
    ----------
    file_entries : list of dict
        Entries from :func:`expected_raw_files`; only those with ``exists``
        True are loaded.
    config : dict | None
        Passed through to :func:`load_matches` for the column check.

    Returns
    -------
    pandas.DataFrame
        All matches stacked together, with ``League`` and ``Season`` columns
        added so we can always tell where a row came from. A column present in
        one season but not another is filled with NaN — which the shape report
        then makes visible, instead of it becoming a silent surprise in Phase 2.
    """
    frames = []
    for entry in file_entries:
        if not entry["exists"]:
            continue
        df = load_matches(entry["path"], config)
        # Provenance columns: cheap now, invaluable when debugging later.
        df["League"] = entry["league"]
        df["Season"] = entry["season"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No raw data files found. Download them first — run the pipeline "
            "once and it prints the exact URLs."
        )

    return pd.concat(frames, ignore_index=True)


def build_shape_report(df, config=None):
    """Return a short, human-readable summary of a matches DataFrame.

    Reports row/column counts, date coverage, distinct teams, per-season counts,
    and (when a config is supplied) how complete the odds columns are. Written
    defensively so it works before Phase 1 cleaning — dates may still be
    strings, and some columns may be missing entirely.
    """
    lines = []
    lines.append(f"rows: {len(df):,}")
    lines.append(f"columns: {len(df.columns)}")

    if "Date" in df.columns and len(df) > 0:
        # football-data.co.uk dates are day-first (17/08/2024), but older seasons
        # use a 2-digit year (17/08/24). format="mixed" lets pandas handle both
        # in one column; errors="coerce" turns anything unparseable into NaT so
        # we can COUNT the bad rows instead of crashing on them.
        dates = pd.to_datetime(
            df["Date"], dayfirst=True, format="mixed", errors="coerce"
        )
        unparsed = int(dates.isna().sum())
        if dates.notna().any():
            lines.append(
                f"date range: {dates.min():%d %b %Y} -> {dates.max():%d %b %Y}"
                f"  (unparseable: {unparsed})"
            )
        else:
            lines.append(f"date range: none parseable ({unparsed} rows)")

    if "Season" in df.columns:
        counts = df["Season"].value_counts().sort_index()
        lines.append(
            "matches per season: "
            + ", ".join(f"{season}={n}" for season, n in counts.items())
        )

    team_cols = [c for c in ("HomeTeam", "AwayTeam") if c in df.columns]
    if team_cols:
        teams = pd.unique(df[team_cols].values.ravel())
        lines.append(f"distinct teams: {len(teams)}")

    # --- Odds coverage -------------------------------------------------------
    # Grouped, because what matters is whether a whole group is usable:
    # no closing odds means no CLV, which is the project's headline metric.
    if config is not None:
        schema = config.get("schema", {})
        groups = {
            "odds (bet into)": schema.get("odds_columns", []),
            "closing odds (CLV)": schema.get("closing_odds_columns", []),
        }
        for label, cols in groups.items():
            present = [c for c in cols if c in df.columns]
            if not present:
                lines.append(f"{label}: MISSING — columns {cols} not in the data")
            elif len(present) < len(cols):
                missing = sorted(set(cols) - set(present))
                lines.append(f"{label}: PARTIAL — missing {missing}")
            else:
                worst_null = df[present].isna().mean().max() * 100
                lines.append(f"{label}: complete (worst null rate {worst_null:.1f}%)")

    # --- Nulls ---------------------------------------------------------------
    null_counts = df.isna().sum()
    nulls = null_counts[null_counts > 0].sort_values(ascending=False)
    if len(nulls) == 0:
        lines.append("nulls: none")
    else:
        # Full files have ~100 columns, so show the worst offenders only.
        shown = ", ".join(f"{c}={n}" for c, n in nulls.head(8).items())
        extra = f" (+{len(nulls) - 8} more columns)" if len(nulls) > 8 else ""
        lines.append(f"nulls: {shown}{extra}")

    return "\n".join(lines)
