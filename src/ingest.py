"""
Stage 1 — Data ingestion.

Phase 0 (done): load a CSV of historical matches + odds and produce a "shape
report" so we can eyeball what we've got. Light column-presence checks only.

30 Sep 2026: extended to work with REAL data from football-data.co.uk —
discovering the league/season files listed in the config, loading several of
them into one table, and reporting odds coverage (including closing odds, which
Phase 6 needs for CLV).

2 Oct 2026 — Phase 1: strict validation and the canonical table. The flow is

    load_many()  ->  validate_raw()  ->  clean()  ->  to_canonical()  ->  save

and ``run_ingest()`` wires those together and returns a data-quality report.

Two principles run through it:

1. **Fail loudly on anything structural.** A missing required column or a file
   so broken that most rows would be dropped stops the run. Silent degradation
   is how a backtest ends up reporting a number nobody can explain.
2. **Record everything we silently fix.** Rows dropped, results recomputed, odds
   voided — all counted and reported, never discarded quietly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import canonical

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

    # Note: no cleaning happens here on purpose. load_* reads what is on disk;
    # clean() below decides what is trustworthy. Keeping those separate means a
    # raw file can always be inspected exactly as the source published it.
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


# ===========================================================================
# Phase 1 — validation, cleaning, canonical table
# ===========================================================================


class DataQualityError(Exception):
    """Raised when the input is too broken to continue.

    A separate exception type (rather than a bare ValueError) so a caller can
    tell "this data is unusable" apart from an ordinary programming error.
    """


def validate_raw(df, config):
    """Check the structure of a raw frame before any cleaning happens.

    Structural problems raise; everything else is returned as a list of notes to
    be included in the data-quality report.

    Raises
    ------
    DataQualityError
        If a required column is missing, or the frame is empty.
    """
    notes = []
    schema = config.get("schema", {})
    required = schema.get("required_columns", [])

    if len(df) == 0:
        raise DataQualityError("The input has no rows at all.")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataQualityError(
            f"Required columns are missing: {missing}. "
            f"Columns present: {sorted(df.columns)[:15]}..."
        )

    # Not fatal, but the pipeline is much less useful without them.
    for label, cols in [
        ("odds", schema.get("odds_columns", [])),
        ("closing odds", schema.get("closing_odds_columns", [])),
    ]:
        absent = [c for c in cols if c not in df.columns]
        if absent:
            notes.append(f"{label} columns absent: {absent}")

    return notes


def clean(df, config):
    """Turn a raw frame into trustworthy rows, counting everything changed.

    Steps, in order:
      1. parse dates (day-first, mixed 2- and 4-digit years)
      2. tidy team names (stray whitespace is a real and silent join-breaker)
      3. force goals to whole numbers and sanity-check the range
      4. recompute the result from the goals rather than trusting the file
      5. void odds that are outside the plausible range

    Returns
    -------
    (pandas.DataFrame, dict)
        The cleaned frame and a dictionary of counts for the report.
    """
    rules = config.get("validation", {})
    report = {"rows_in": len(df)}
    out = df.copy()

    # --- 1. Dates ---------------------------------------------------------
    out["MatchDate"] = pd.to_datetime(
        out["Date"], dayfirst=True, format="mixed", errors="coerce"
    )
    bad_dates = int(out["MatchDate"].isna().sum())
    report["dropped_unparseable_date"] = bad_dates
    out = out[out["MatchDate"].notna()]

    # --- 2. Team names ----------------------------------------------------
    # " Arsenal" and "Arsenal" are different strings and would become two teams.
    for col in ("HomeTeam", "AwayTeam"):
        before = out[col].copy()
        out[col] = out[col].astype(str).str.strip()
        report[f"whitespace_trimmed_{col}"] = int((before != out[col]).sum())

    # --- 3. Goals ---------------------------------------------------------
    max_goals = rules.get("max_goals", 15)
    for col in ("FTHG", "FTAG"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    implausible = (
        out["FTHG"].isna() | out["FTAG"].isna()
        | (out["FTHG"] < 0) | (out["FTAG"] < 0)
        | (out["FTHG"] > max_goals) | (out["FTAG"] > max_goals)
    )
    report["dropped_implausible_goals"] = int(implausible.sum())
    out = out[~implausible]
    out["FTHG"] = out["FTHG"].astype(int)
    out["FTAG"] = out["FTAG"].astype(int)

    # --- 4. Result --------------------------------------------------------
    # Derive H/D/A from the goals. If the file disagrees, the file is wrong:
    # goals are the primary record and the result is a convenience column.
    derived = np.where(
        out["FTHG"] > out["FTAG"], "H",
        np.where(out["FTHG"] < out["FTAG"], "A", "D"),
    )
    if "FTR" in out.columns:
        mismatches = int((out["FTR"].astype(str).str.strip() != derived).sum())
        report["result_mismatches_corrected"] = mismatches
    out["Result"] = derived

    # --- 5. Odds ----------------------------------------------------------
    # Out-of-range odds are voided (set to NaN), not dropped: the match still
    # happened and still trains the model — it simply isn't bettable.
    min_odds = rules.get("min_odds", 1.01)
    max_odds = rules.get("max_odds", 1000.0)
    schema = config.get("schema", {})
    odds_cols = [
        c for c in schema.get("odds_columns", []) + schema.get("closing_odds_columns", [])
        if c in out.columns
    ]
    voided = 0
    for col in odds_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bad = out[col].notna() & ((out[col] < min_odds) | (out[col] > max_odds))
        voided += int(bad.sum())
        out.loc[bad, col] = np.nan
    report["odds_values_voided"] = voided

    report["rows_out"] = len(out)
    return out.reset_index(drop=True), report


def drop_duplicate_fixtures(df, config):
    """Remove repeated fixtures — the same two teams on the same date.

    Duplicates arise from re-downloading a file or from an overlapping season
    range. Left in, they would double-count those matches in the likelihood and
    quietly over-weight them.
    """
    rules = config.get("validation", {})
    if not rules.get("drop_duplicate_fixtures", True):
        return df, 0

    key = ["MatchDate", "HomeTeam", "AwayTeam"]
    duplicated = df.duplicated(subset=key, keep="first")
    return df[~duplicated].reset_index(drop=True), int(duplicated.sum())


def to_canonical(df, config):
    """Map a cleaned frame onto the canonical schema and sort it by date.

    Missing optional columns (shots, odds) are created as NaN so every canonical
    frame has the same shape, whatever the source file happened to contain.
    """
    renames = dict(canonical.BASE_RENAMES)
    renames.update(canonical.odds_renames(config))

    out = df.rename(columns=renames)

    # A stable identifier, built from facts that cannot change for a given match.
    # Used for joins, for de-duplication across runs, and for tracing a bet in
    # the Phase 6 backtest back to its fixture.
    league = out["League"] if "League" in out.columns else "NA"
    season = out["Season"] if "Season" in out.columns else "NA"
    out["MatchID"] = (
        pd.Series(league, index=out.index).astype(str) + "_"
        + pd.Series(season, index=out.index).astype(str) + "_"
        + out["MatchDate"].dt.strftime("%Y%m%d") + "_"
        + out["HomeTeam"].str.replace(" ", "", regex=False) + "_"
        + out["AwayTeam"].str.replace(" ", "", regex=False)
    )

    for column in canonical.CANONICAL_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    out = out[canonical.CANONICAL_COLUMNS]

    # Chronological order matters from here on: Phase 6 walks forward through
    # time, and an out-of-order frame is the easiest way to leak the future.
    out = out.sort_values("MatchDate").reset_index(drop=True)

    missing_required = [
        c for c in canonical.REQUIRED_CANONICAL if out[c].isna().any()
    ]
    if missing_required:
        raise DataQualityError(
            f"Canonical columns contain nulls where none are allowed: {missing_required}"
        )
    return out


def quality_report(report, df=None):
    """Render the counts collected during cleaning as readable text."""
    lines = ["Data-quality report", "-" * 19]
    order = [
        ("rows_in", "rows read"),
        ("dropped_unparseable_date", "dropped: unparseable date"),
        ("dropped_implausible_goals", "dropped: impossible goal values"),
        ("duplicate_fixtures_dropped", "dropped: duplicate fixtures"),
        ("whitespace_trimmed_HomeTeam", "fixed: whitespace in HomeTeam"),
        ("whitespace_trimmed_AwayTeam", "fixed: whitespace in AwayTeam"),
        ("result_mismatches_corrected", "fixed: result disagreed with goals"),
        ("odds_values_voided", "voided: odds outside plausible range"),
        ("rows_out", "rows kept"),
    ]
    for key, label in order:
        if key in report:
            lines.append(f"  {label:<38} {report[key]:>6,}")

    if report.get("rows_in"):
        kept = report.get("rows_out", 0) / report["rows_in"]
        lines.append(f"  {'share of rows kept':<38} {kept:>6.1%}")

    for note in report.get("notes", []):
        lines.append(f"  note: {note}")

    if df is not None and len(df):
        lines.append("")
        lines.append(f"  canonical table: {len(df):,} matches x {len(df.columns)} columns")
        lines.append(
            f"  date coverage:   {df['MatchDate'].min():%d %b %Y}"
            f" -> {df['MatchDate'].max():%d %b %Y}"
        )
        teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
        lines.append(f"  teams:           {len(teams)}")
        for col in ("OddsH", "CloseH"):
            if col in df.columns:
                usable = df[col].notna().mean()
                lines.append(f"  {col} present:   {usable:.1%} of matches")
    return "\n".join(lines)


def save_canonical(df, config, filename=None):
    """Write the canonical table to data/processed/ and return the path.

    CSV rather than a binary format, deliberately: it can be opened and checked
    by anyone, including in a code review, with no extra dependency.
    """
    rules = config.get("validation", {})
    out_dir = _resolve(config["data"].get("processed_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (filename or rules.get("canonical_filename", "matches.csv"))
    df.to_csv(path, index=False)
    return path


def run_ingest(df_raw, config, save=True):
    """Run the whole Phase 1 flow and return (canonical_df, report_dict).

    Keeping the orchestration here, rather than in the pipeline script, means the
    tests exercise exactly the code the pipeline runs.
    """
    rules = config.get("validation", {})

    notes = validate_raw(df_raw, config)
    cleaned, report = clean(df_raw, config)
    deduped, n_dupes = drop_duplicate_fixtures(cleaned, config)
    report["duplicate_fixtures_dropped"] = n_dupes
    report["rows_out"] = len(deduped)
    report["notes"] = notes

    # The guard that turns a quiet disaster into a loud one: if most of the file
    # has been thrown away, something is wrong with the source, not the matches.
    dropped_fraction = 1 - (report["rows_out"] / report["rows_in"])
    limit = rules.get("max_dropped_fraction", 0.05)
    if dropped_fraction > limit:
        raise DataQualityError(
            f"Dropped {dropped_fraction:.1%} of rows, above the {limit:.0%} limit. "
            "Inspect the raw file before continuing — do not relax this silently."
        )

    table = to_canonical(deduped, config)

    if save:
        report["saved_to"] = str(save_canonical(table, config))

    return table, report
