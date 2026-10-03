"""
The canonical table definition — the contract between Phase 1 and everything after it.

Why a separate module: downstream stages (features, model, backtest) must never
reach for a raw football-data.co.uk column name like ``FTHG``. They read these
names. If the source ever changes, only the mapping below changes with it.

Naming rules kept deliberately boring: readable words, no abbreviations that
need a lookup, and one obvious meaning each.
"""
from __future__ import annotations

# Canonical column order. Downstream code may rely on these names existing.
CANONICAL_COLUMNS = [
    "MatchID",       # stable identifier: league_season_date_home_away
    "MatchDate",     # datetime64 — parsed, never a string
    "League",        # e.g. E0
    "Season",        # e.g. 2425
    "HomeTeam",
    "AwayTeam",
    "HomeGoals",     # int
    "AwayGoals",     # int
    "Result",        # H / D / A  (recomputed from goals, not trusted from source)
    "HomeShots",
    "AwayShots",
    "HomeShotsOT",   # shots on target
    "AwayShotsOT",
    "OddsH",         # the odds we would bet into
    "OddsD",
    "OddsA",
    "CloseH",        # closing odds — the CLV benchmark in Phase 6
    "CloseD",
    "CloseA",
]

# Raw -> canonical, for the columns that are the same in every football-data file.
BASE_RENAMES = {
    "HomeTeam": "HomeTeam",
    "AwayTeam": "AwayTeam",
    "FTHG": "HomeGoals",
    "FTAG": "AwayGoals",
    "HS": "HomeShots",
    "AS": "AwayShots",
    "HST": "HomeShotsOT",
    "AST": "AwayShotsOT",
}

# Columns a downstream stage cannot work without. Shots and odds may legitimately
# be missing for some rows (older seasons, dead markets); these may not.
REQUIRED_CANONICAL = [
    "MatchID", "MatchDate", "HomeTeam", "AwayTeam",
    "HomeGoals", "AwayGoals", "Result",
]


def odds_renames(config):
    """Build the odds rename map from the config, in H/D/A order.

    The bookmaker is a config choice (today Bet365 for the bet-into price and
    Pinnacle for the close). Reading the names from config means switching
    bookmaker never touches this code.
    """
    schema = config.get("schema", {})
    bet = schema.get("odds_columns", [])
    close = schema.get("closing_odds_columns", [])

    mapping = {}
    for raw, canon in zip(bet, ["OddsH", "OddsD", "OddsA"]):
        mapping[raw] = canon
    for raw, canon in zip(close, ["CloseH", "CloseD", "CloseA"]):
        mapping[raw] = canon
    return mapping
