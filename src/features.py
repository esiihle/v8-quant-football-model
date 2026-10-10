"""
Stage 2 — Feature engineering.  [Built in Phase 2, 9 Oct 2026]

Turns the canonical match table into one row of *predictive inputs* per match:
each team's attack and defence strength, shrunk toward the league average,
estimated from recent form only, using goals blended with a shots-based
expected-goals proxy.

The rule that governs this entire module
----------------------------------------
**A match's features may only use matches played strictly before it.**

Every function here takes an ``as_of`` date and filters on it. That is not
defensive habit; it is the difference between a backtest that means something
and one that doesn't. A model that has seen the second half of the season when
rating a team in October will look brilliant and lose money. The test
``test_features_do_not_use_the_future`` exists to prove it never happens.

What gets built, and why
------------------------
* **Attack / defence strength** — how many goals a team scores and concedes
  relative to the league average. 1.0 is exactly average; 1.3 attack means they
  score 30% more than a typical team.
* **Bayesian shrinkage** — a team with three matches played has a meaningless
  average, so each estimate is pulled toward the league mean by weight
  ``n / (n + k)``. This is the same machinery as a thin-file or
  low-default-portfolio adjustment in credit risk, which is why it is worth
  being able to explain in both languages.
* **Time decay** — a match from two seasons ago tells you less about this squad
  than one from last month, so matches are weighted ``0.5 ** (age / half_life)``.
* **xG proxy** — goals are the truth but are rare and noisy. Shots on target are
  plentiful. The feature blends them, controlled by ``features.xg_weight``.

See docs/methodology.md section 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns the feature builder produces, in order. Phase 3 reads these names.
FEATURE_COLUMNS = [
    "MatchID", "MatchDate", "League", "Season", "HomeTeam", "AwayTeam",
    "HomeAttack", "HomeDefence", "AwayAttack", "AwayDefence",
    "HomePriorMatches", "AwayPriorMatches", "HasSufficientHistory",
    "HomeGoals", "AwayGoals", "Result",
    "OddsH", "OddsD", "OddsA", "CloseH", "CloseD", "CloseA",
]


class FeatureError(Exception):
    """Raised when features cannot be built from the input given."""


# ===========================================================================
# Time decay
# ===========================================================================

def time_weights(dates, as_of, half_life_days=180):
    """Weight each past match by how recent it is.

    ``weight = 0.5 ** (age_in_days / half_life)`` — a match exactly one
    half-life old counts half as much as one played today.

    Matches dated on or after ``as_of`` get weight 0. That is the lookahead
    guard, implemented once, here, rather than trusted to every caller.

    Parameters
    ----------
    dates : array-like of datetime
    as_of : datetime
        The moment we are standing at. Nothing from this date onward may count.
    half_life_days : float
        Days for a match's influence to halve. Larger = longer memory.

    Returns
    -------
    numpy.ndarray of float
    """
    dates = pd.to_datetime(pd.Series(dates))
    as_of = pd.Timestamp(as_of)

    age_days = (as_of - dates).dt.total_seconds() / 86400.0
    weights = np.power(0.5, age_days / float(half_life_days))

    # age < 0 means the match is in the future relative to as_of: weight 0.
    return np.where(age_days > 0, weights, 0.0)


# ===========================================================================
# Shrinkage
# ===========================================================================

def shrink(estimate, n_matches, prior, k=8):
    """Pull a noisy estimate toward a prior, in proportion to how little data backs it.

        shrunk = w * estimate + (1 - w) * prior,   w = n / (n + k)

    With k = 8: a team with 0 matches is given the prior exactly; with 8 matches
    it sits halfway; with 38 it is 83% its own record. There is no cliff — the
    estimate earns trust gradually as evidence accumulates.

    Parameters
    ----------
    estimate : float or array
        The team's own (noisy) figure.
    n_matches : float or array
        How many matches back it. May be fractional — with time decay, the
        *effective* sample size is the sum of weights, not a count of rows.
    prior : float
        What to fall back on, i.e. the league average.
    k : float
        Shrinkage strength. Higher = more sceptical of small samples.

    Returns
    -------
    float or numpy.ndarray
    """
    n_matches = np.asarray(n_matches, dtype=float)
    weight = n_matches / (n_matches + float(k))
    return weight * np.asarray(estimate, dtype=float) + (1.0 - weight) * float(prior)


# ===========================================================================
# Expected-goals proxy
# ===========================================================================

def conversion_rates(history):
    """Goals per shot on target, measured from past matches only.

    Home and away are measured separately because they genuinely differ — home
    sides convert slightly better, and folding that into the xG proxy would
    smuggle home advantage into team strength, where it would be counted twice.

    Returns
    -------
    (float, float, bool)
        Home rate, away rate, and whether shot data was usable at all.
    """
    usable = (
        "HomeShotsOT" in history.columns and "AwayShotsOT" in history.columns
        and len(history) > 0
        and history["HomeShotsOT"].notna().any()
        and history["AwayShotsOT"].notna().any()
    )
    if not usable:
        return 0.0, 0.0, False

    home_ot = pd.to_numeric(history["HomeShotsOT"], errors="coerce")
    away_ot = pd.to_numeric(history["AwayShotsOT"], errors="coerce")

    home_rate = history["HomeGoals"].sum() / max(home_ot.sum(), 1)
    away_rate = history["AwayGoals"].sum() / max(away_ot.sum(), 1)
    return float(home_rate), float(away_rate), True


def blend_goals(matches, history, config=None):
    """Blend actual goals with a shots-based expected-goals proxy.

    Goals are the truth but are rare: a 1-0 win and a 1-0 loss look identical in
    the table and nothing like each other on the pitch. Shots on target are
    plentiful and carry more signal per match, at the cost of being a proxy. The
    blend weight is ``features.xg_weight``: 0.0 goals only, 1.0 proxy only.

    ``history`` is a required argument, not an optional one, and it is what the
    conversion rate is measured from. That is deliberate: an earlier version of
    this module measured the rate over the whole file, which meant a match in
    August was scored using a conversion rate computed partly from matches in
    May. The lookahead test caught it. Requiring the caller to say which history
    it means makes that mistake hard to repeat.

    Returns
    -------
    (pandas.DataFrame, str)
        A copy with ``HomeAttackGoals`` / ``AwayAttackGoals``, plus a note on
        what was actually used.
    """
    cfg = (config or {}).get("features", {})
    weight = float(cfg.get("xg_weight", 0.5))

    out = matches.copy()
    home_rate, away_rate, have_shots = conversion_rates(history)

    if not have_shots or weight == 0.0:
        out["HomeAttackGoals"] = out["HomeGoals"].astype(float)
        out["AwayAttackGoals"] = out["AwayGoals"].astype(float)
        note = ("xG proxy not used (no shots-on-target data)" if not have_shots
                else "xG proxy disabled (xg_weight = 0)")
        return out, note

    home_xg = pd.to_numeric(out["HomeShotsOT"], errors="coerce") * home_rate
    away_xg = pd.to_numeric(out["AwayShotsOT"], errors="coerce") * away_rate

    # Where a row has no shot data, fall back to that row's actual goals rather
    # than dropping it or inventing a figure.
    home_xg = home_xg.fillna(out["HomeGoals"])
    away_xg = away_xg.fillna(out["AwayGoals"])

    out["HomeAttackGoals"] = (1 - weight) * out["HomeGoals"] + weight * home_xg
    out["AwayAttackGoals"] = (1 - weight) * out["AwayGoals"] + weight * away_xg

    note = (f"xG proxy blended at {weight:.0%} "
            f"(home {home_rate:.3f}, away {away_rate:.3f} goals per shot on target)")
    return out, note


# ===========================================================================
# Team strengths as of a point in time
# ===========================================================================

def team_strengths(history, as_of, config=None):
    """Attack and defence strength for every team, using matches before ``as_of``.

    Strength is relative to the league: 1.0 is exactly average, 1.25 attack means
    a quarter more goals than a typical team, 0.80 defence means a fifth fewer
    conceded (lower defence is better).

    Each team's figure is a time-weighted average, then shrunk toward 1.0 by its
    effective sample size — so a team three matches into a season barely moves
    from average, however spectacular those three matches were.

    Parameters
    ----------
    history : pandas.DataFrame
        Canonical matches, plus the blended goal columns from :func:`xg_proxy`.
    as_of : datetime
        Only matches strictly before this count.
    config : dict | None

    Returns
    -------
    pandas.DataFrame
        Indexed by team, with columns attack, defence, prior_matches
        (the effective, time-weighted sample size) and raw_matches (a plain count).
    """
    cfg = (config or {}).get("features", {})
    half_life = float(cfg.get("half_life_days", 180))
    k = float(cfg.get("shrinkage_k", 8))

    weights = time_weights(history["MatchDate"], as_of, half_life)
    past = history.loc[weights > 0].copy()
    past["w"] = weights[weights > 0]

    if len(past) == 0:
        return pd.DataFrame(
            columns=["attack", "defence", "prior_matches", "raw_matches"]
        )

    # The league baseline: weighted mean goals in a team-match. Home and away
    # are pooled here on purpose — home advantage is a separate effect, fitted
    # in Phase 3, and folding it into team strength would double-count it.
    total_weight = past["w"].sum()
    league_mean = (
        (past["HomeAttackGoals"] * past["w"]).sum()
        + (past["AwayAttackGoals"] * past["w"]).sum()
    ) / (2 * total_weight)
    league_mean = max(league_mean, 1e-6)

    # One row per team-appearance: scored, conceded, and the match's weight.
    appearances = pd.concat([
        pd.DataFrame({
            "team": past["HomeTeam"],
            "scored": past["HomeAttackGoals"],
            "conceded": past["AwayAttackGoals"],
            "w": past["w"],
        }),
        pd.DataFrame({
            "team": past["AwayTeam"],
            "scored": past["AwayAttackGoals"],
            "conceded": past["HomeAttackGoals"],
            "w": past["w"],
        }),
    ], ignore_index=True)

    grouped = appearances.groupby("team")
    weight_sum = grouped["w"].sum()
    raw_attack = grouped.apply(
        lambda g: (g["scored"] * g["w"]).sum() / g["w"].sum(), include_groups=False
    ) / league_mean
    raw_defence = grouped.apply(
        lambda g: (g["conceded"] * g["w"]).sum() / g["w"].sum(), include_groups=False
    ) / league_mean

    # Effective sample size = sum of weights. A team with ten matches a year old
    # has far less than ten matches' worth of evidence, and shrinkage should know.
    return pd.DataFrame({
        "attack": shrink(raw_attack.values, weight_sum.values, prior=1.0, k=k),
        "defence": shrink(raw_defence.values, weight_sum.values, prior=1.0, k=k),
        "prior_matches": weight_sum.values,
        "raw_matches": grouped.size().values,
    }, index=weight_sum.index)


# ===========================================================================
# The feature matrix
# ===========================================================================

def build_features(matches, config=None):
    """Build one row of features per match, strictly out of sample.

    Walks the canonical table in date order. For each distinct match date, team
    strengths are recomputed from everything before that date, then applied to
    the matches played on it. Recomputing per date rather than per match is the
    only shortcut taken, and it is exact: two matches on the same day see the
    same history, which is also what a real bettor would have had.

    Parameters
    ----------
    matches : pandas.DataFrame
        The canonical table from Phase 1 (``data/processed/matches.csv``).
    config : dict | None

    Returns
    -------
    (pandas.DataFrame, dict)
        The feature matrix in :data:`FEATURE_COLUMNS` order, and a report dict.
    """
    cfg = (config or {}).get("features", {})
    min_prior = float(cfg.get("min_prior_matches", 5))

    required = ["MatchID", "MatchDate", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]
    missing = [c for c in required if c not in matches.columns]
    if missing:
        raise FeatureError(f"Feature building needs columns that are missing: {missing}")
    if len(matches) == 0:
        raise FeatureError("No matches to build features from.")

    work = matches.copy()
    work["MatchDate"] = pd.to_datetime(work["MatchDate"])
    work = work.sort_values("MatchDate").reset_index(drop=True)

    rows = []
    xg_note = "xG proxy not used (no history yet)"

    for as_of, todays_matches in work.groupby("MatchDate", sort=True):
        # Everything below this line sees history only. Both the xG conversion
        # rate AND the team strengths are measured from matches played strictly
        # before today — which is exactly what a bettor had at kick-off.
        history = work[work["MatchDate"] < as_of]
        blended_history, xg_note = blend_goals(history, history, config)
        strengths = team_strengths(blended_history, as_of, config)

        for _, match in todays_matches.iterrows():
            home = strengths.reindex([match["HomeTeam"]]).iloc[0]
            away = strengths.reindex([match["AwayTeam"]]).iloc[0]

            # A team never seen before gets the prior, not a NaN: an unknown team
            # is an average team until it shows otherwise.
            home_attack = 1.0 if pd.isna(home["attack"]) else home["attack"]
            home_defence = 1.0 if pd.isna(home["defence"]) else home["defence"]
            away_attack = 1.0 if pd.isna(away["attack"]) else away["attack"]
            away_defence = 1.0 if pd.isna(away["defence"]) else away["defence"]
            home_n = 0.0 if pd.isna(home["prior_matches"]) else home["prior_matches"]
            away_n = 0.0 if pd.isna(away["prior_matches"]) else away["prior_matches"]

            rows.append({
                "MatchID": match["MatchID"],
                "MatchDate": match["MatchDate"],
                "League": match.get("League"),
                "Season": match.get("Season"),
                "HomeTeam": match["HomeTeam"],
                "AwayTeam": match["AwayTeam"],
                "HomeAttack": home_attack,
                "HomeDefence": home_defence,
                "AwayAttack": away_attack,
                "AwayDefence": away_defence,
                "HomePriorMatches": home_n,
                "AwayPriorMatches": away_n,
                "HasSufficientHistory": bool(home_n >= min_prior and away_n >= min_prior),
                "HomeGoals": match["HomeGoals"],
                "AwayGoals": match["AwayGoals"],
                "Result": match.get("Result"),
                "OddsH": match.get("OddsH"), "OddsD": match.get("OddsD"),
                "OddsA": match.get("OddsA"), "CloseH": match.get("CloseH"),
                "CloseD": match.get("CloseD"), "CloseA": match.get("CloseA"),
            })

    features = pd.DataFrame(rows)[FEATURE_COLUMNS]

    usable = int(features["HasSufficientHistory"].sum())
    report = {
        "matches": len(features),
        "usable_for_training": usable,
        "warm_up_rows": len(features) - usable,
        "min_prior_matches": min_prior,
        "shrinkage_k": cfg.get("shrinkage_k", 8),
        "half_life_days": cfg.get("half_life_days", 180),
        "xg_note": xg_note,
    }
    return features, report


def feature_report(features, report):
    """Readable summary of the feature matrix and the settings that built it."""
    lines = ["Feature matrix", "-" * 14]
    lines.append(f"  {'matches':<34} {report['matches']:>7,}")
    lines.append(f"  {'usable for training':<34} {report['usable_for_training']:>7,}")
    lines.append(f"  {'warm-up rows (thin history)':<34} {report['warm_up_rows']:>7,}")
    lines.append(f"  {'shrinkage k':<34} {report['shrinkage_k']:>7}")
    lines.append(f"  {'time-decay half-life (days)':<34} {report['half_life_days']:>7}")
    lines.append(f"  {report['xg_note']}")

    if len(features):
        trained = features[features["HasSufficientHistory"]]
        if len(trained):
            lines.append("")
            lines.append("  Strength spread on usable rows (1.0 = league average):")
            for col in ("HomeAttack", "HomeDefence", "AwayAttack", "AwayDefence"):
                series = trained[col]
                lines.append(f"    {col:<12} min {series.min():.2f}"
                             f"  mean {series.mean():.2f}  max {series.max():.2f}")
    return "\n".join(lines)


def save_features(features, config, filename=None):
    """Write the feature matrix to data/processed/ and return the path."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / (filename or config.get("features", {}).get(
        "feature_filename", "features.csv"))
    features.to_csv(path, index=False)
    return path
