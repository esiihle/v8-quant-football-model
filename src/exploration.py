"""
Exploration: what the real data says before any modelling. [29 Oct 2026]

Until now these measurements lived only inside notebooks/01_exploration.ipynb,
which had never been run on real data: its committed outputs came from the
12-match sample (a 58% home-win rate, "0-3 is 603% more common than Poisson"),
and its last cell asked for the numbers to be copied into PROGRESS.md by hand.

Moving them here gives them the same standing as the rest of the pipeline:

* functions that return results rather than print them, so the notebook, the
  script and the tests all see the same numbers;
* `scripts/explore.py` writes them to `exploration.json` beside numbers.json,
  and the documentation quotes that file, never a hand-copied value;
* every rate comes with an interval, the project's rule for any measurement.

Each function answers a question a later phase depends on:

    outcome_split / goal_means   home advantage exists and is stable (Phase 3)
    low_score_departures         is the Dixon-Coles tau correction needed? (3)
    overround                    what margin a value bet must beat (Phase 5)
    market_calibration           the bar the model's probabilities must clear (4)
    shots_signal                 do shots carry the signal the xG proxy uses (2)
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, poisson

from src import calibrate, dixon_coles, ingest, staking

LOW_SCORES = [(0, 0), (1, 0), (0, 1), (1, 1)]


# ---------------------------------------------------------------------------
# Loading: exactly what the pipeline reads
# ---------------------------------------------------------------------------

def load_canonical(config):
    """The canonical match table, from real files if ALL are present.

    Returns (table, real) where `real` says whether it came from data/raw.
    Goes through ingest.run_ingest, so exploration sees the same cleaned,
    validated, de-duplicated rows the model does - not a second loader with
    its own quirks.
    """
    files = ingest.expected_raw_files(config)
    if files and all(entry["exists"] for entry in files):
        raw = ingest.load_many(files, config)
        real = True
    else:
        raw = ingest.load_matches(ingest._resolve(config["data"]["sample_path"]), config)
        real = False
    table, _ = ingest.run_ingest(raw, config, save=False)
    table["MatchDate"] = pd.to_datetime(table["MatchDate"])
    return table, real


# ---------------------------------------------------------------------------
# 1. Home advantage
# ---------------------------------------------------------------------------

def wilson_interval(successes, n, z=1.96):
    """95% interval for a proportion. Wilson, not normal: it stays inside
    [0, 1] and behaves on small counts, where the normal interval does not."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


def outcome_split(table):
    """Share of home wins, draws and away wins, each with a Wilson interval."""
    n = len(table)
    out = {}
    for code, name in (("H", "home"), ("D", "draw"), ("A", "away")):
        k = int((table["Result"] == code).sum())
        low, high = wilson_interval(k, n)
        out[name] = {"share": k / n if n else np.nan, "low": low, "high": high}
    out["matches"] = n
    return out


def goal_means(table):
    """Mean goals per match for each side, and their ratio."""
    home = float(table["HomeGoals"].mean())
    away = float(table["AwayGoals"].mean())
    return {"home": home, "away": away, "ratio": home / away if away else np.nan}


def per_season(table):
    """Outcome shares and goal means by season - is the effect stable?"""
    grouped = table.groupby("Season")
    frame = pd.DataFrame({
        "matches": grouped.size(),
        "home_win": grouped["Result"].apply(lambda r: (r == "H").mean()),
        "draw": grouped["Result"].apply(lambda r: (r == "D").mean()),
        "away_win": grouped["Result"].apply(lambda r: (r == "A").mean()),
        "home_goals": grouped["HomeGoals"].mean(),
        "away_goals": grouped["AwayGoals"].mean(),
    })
    return frame.reset_index()


# ---------------------------------------------------------------------------
# 2. Are goals Poisson? The tau question, with the right null
# ---------------------------------------------------------------------------

def _descriptive_config(config):
    """A config for IN-SAMPLE, descriptive fits: no time decay within the
    data, no cutoff. This is exploration, not forecasting - the question is
    what the data looks like, not what could have been known in advance."""
    features = {**config.get("features", {}),
                "half_life_days": 1e9, "min_match_weight": 0.0}
    return {**config, "features": features}


def low_score_departures(table, config):
    """Observed vs expected counts of 0-0, 1-0, 0-1, 1-1 under TWO nulls.

    Dixon and Coles (1997) found independent Poisson gets the four low scores
    wrong, and their tau correction with rho < 0 RAISES 0-0 and 1-1 and LOWERS
    1-0 and 0-1. (This notebook's text once said it raised all four. It does
    not; `tau` in dixon_coles.py has always had it right, and is tested.)

    The answer depends on what "independent Poisson" means:

    * league-mean null - every match uses the league's average goals. Crude:
      a league mixes strong and weak teams, and a MIXTURE of Poissons has more
      extreme and fewer middling scores than one Poisson, whatever tau says.
    * per-match null - each match uses its own expected goals from a
      Dixon-Coles fit with rho set to 0, so team strength is accounted for and
      only the low-score dependence is left to find.

    On EPL 2012-25 the league-mean null shows 1-1 eleven per cent RARER than
    Poisson, and the per-match null shows that was team mixing: 1-1 is +0.5%.
    The null decides the answer.

    Each season is fitted separately (in-sample, no decay), because team
    strengths change between seasons. Returns a DataFrame, one row per score,
    and the list of per-season fitted rho.
    """
    cfg = _descriptive_config(config)
    observed = dict.fromkeys(LOW_SCORES, 0)
    per_match = dict.fromkeys(LOW_SCORES, 0.0)
    league = dict.fromkeys(LOW_SCORES, 0.0)
    rhos = []

    for season, games in table.groupby("Season"):
        model = dixon_coles.fit(games, cfg)
        rhos.append({"season": str(season), "rho": float(model["rho"])})
        lam = np.exp(games["HomeTeam"].map(model["attack"])
                     + games["AwayTeam"].map(model["defence"])
                     + model["home_advantage"]).to_numpy()
        mu = np.exp(games["AwayTeam"].map(model["attack"])
                    + games["HomeTeam"].map(model["defence"])).to_numpy()
        mean_home, mean_away = games["HomeGoals"].mean(), games["AwayGoals"].mean()

        for h, a in LOW_SCORES:
            observed[(h, a)] += int(((games["HomeGoals"] == h)
                                     & (games["AwayGoals"] == a)).sum())
            per_match[(h, a)] += float((poisson.pmf(h, lam) * poisson.pmf(a, mu)).sum())
            league[(h, a)] += float(poisson.pmf(h, mean_home)
                                    * poisson.pmf(a, mean_away) * len(games))

    rows = []
    for h, a in LOW_SCORES:
        k = observed[(h, a)]
        # The observed count is the random quantity: a Poisson interval on it,
        # expressed relative to the expected count.
        k_low, k_high = poisson.ppf(0.025, k), poisson.ppf(0.975, k)
        e_match, e_league = per_match[(h, a)], league[(h, a)]
        rows.append({
            "score": f"{h}-{a}",
            "observed": k,
            "expected_per_match": e_match,
            "departure_per_match": k / e_match - 1,
            "departure_low": k_low / e_match - 1,
            "departure_high": k_high / e_match - 1,
            "expected_league_mean": e_league,
            "departure_league_mean": k / e_league - 1,
        })
    return pd.DataFrame(rows), rhos


# ---------------------------------------------------------------------------
# 3. What the market charges, and how good it is
# ---------------------------------------------------------------------------

def overround(table, columns):
    """Mean book margin (sum of implied probabilities minus one) and its range."""
    implied = (1.0 / table[list(columns)]).sum(axis=1) - 1.0
    implied = implied[np.isfinite(implied)]
    return {"mean": float(implied.mean()), "p10": float(implied.quantile(0.10)),
            "p90": float(implied.quantile(0.90)), "matches": int(len(implied))}


def overround_by_season(table):
    """Bet-into and closing overround per season - the margin is not a constant."""
    rows = []
    for season, games in table.groupby("Season"):
        rows.append({"season": str(season),
                     "bet_into": overround(games, ["OddsH", "OddsD", "OddsA"])["mean"],
                     "closing": overround(games, ["CloseH", "CloseD", "CloseA"])["mean"]})
    return pd.DataFrame(rows)


def ece_noise_floor(probabilities, n_bins=10, n_sims=500, seed=7):
    """The ECE a PERFECTLY calibrated forecaster would score on these matches.

    Added 29 Oct 2026. ECE is never zero on finite data: each bin's realised
    frequency is itself an estimate. With ~60 matches per bin (380 matches,
    6 occupied bins) a perfectly honest forecaster scores about 0.05 - the
    same size as the "miscalibration" one would be tempted to report.

    So: keep the forecaster's own probabilities, draw outcomes FROM them (the
    world in which it is perfectly calibrated), and score ECE many times. An
    observed ECE above the 95th percentile of these is miscalibration the data
    can actually see; below it, the honest reading is "consistent with perfect
    calibration".
    """
    p = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    sims = np.array([calibrate.expected_calibration_error(
        p, (rng.random(len(p)) < p).astype(float), n_bins) for _ in range(n_sims)])
    return {"median": float(np.median(sims)), "p95": float(np.quantile(sims, 0.95))}


def market_calibration(table, devig_method="shin", n_bins=10, fit_fraction=None):
    """How well the CLOSING market's home-win probability matches reality.

    De-vigged with the pipeline's own method (Shin), and scored with the same
    expected-calibration-error function used on the model in Phase 4, so the
    two numbers are directly comparable: the market's ECE is the bar the
    model's must clear.

    `fit_fraction`: when given, score only the LATER part of the data, split
    chronologically exactly as Phase 4 splits it to judge the model. ECE is
    noisier on fewer matches, so a market scored on all matches against a
    model scored on half of them would flatter the market.
    """
    if fit_fraction is not None:
        _, table = calibrate.split_by_time(table, fit_fraction)
    odds = table[["CloseH", "CloseD", "CloseA"]].to_numpy(dtype=float)
    keep = np.all(np.isfinite(odds) & (odds > 1.0), axis=1)
    fair = np.array([staking.market_probabilities(row, devig_method)[0]
                     for row in odds[keep]])
    p_home = fair[:, 0]
    home_won = (table.loc[keep, "Result"] == "H").astype(float).to_numpy()
    floor = ece_noise_floor(p_home, n_bins)
    ece = float(calibrate.expected_calibration_error(p_home, home_won, n_bins))
    return {
        "ece": ece,
        "noise_floor_median": floor["median"],
        "noise_floor_p95": floor["p95"],
        "detectably_miscalibrated": bool(ece > floor["p95"]),
        "table": calibrate.reliability_table(p_home, home_won, n_bins),
        "matches": int(keep.sum()),
    }


# ---------------------------------------------------------------------------
# 4. Shots as an expected-goals proxy
# ---------------------------------------------------------------------------

def correlation_interval(r, n, z=1.96):
    """95% interval for a Pearson correlation, via Fisher's z-transform."""
    if n <= 3 or not np.isfinite(r):
        return (np.nan, np.nan)
    centre, half = np.arctanh(r), z / np.sqrt(n - 3)
    return (float(np.tanh(centre - half)), float(np.tanh(centre + half)))


def shots_signal(table):
    """Correlation of shots and shots on target with goals, each side."""
    out = {}
    for label, shots, goals in (("home_shots_on_target", "HomeShotsOT", "HomeGoals"),
                                ("away_shots_on_target", "AwayShotsOT", "AwayGoals"),
                                ("home_shots", "HomeShots", "HomeGoals"),
                                ("away_shots", "AwayShots", "AwayGoals")):
        pair = table[[shots, goals]].dropna()
        r = float(pair[shots].corr(pair[goals])) if len(pair) > 2 else np.nan
        low, high = correlation_interval(r, len(pair))
        out[label] = {"r": r, "low": low, "high": high, "n": int(len(pair))}
    return out


# ---------------------------------------------------------------------------
# Everything, for exploration.json
# ---------------------------------------------------------------------------

def findings(table, config):
    """Every headline number, as plain values ready for JSON."""
    split = outcome_split(table)
    goals = goal_means(table)
    departures, rhos = low_score_departures(table, config)
    devig = config.get("staking", {}).get("devig_method", "shin")
    market = market_calibration(table, devig)
    market_later = market_calibration(
        table, devig, fit_fraction=config.get("calibration", {}).get("fit_fraction", 0.5))
    shots = shots_signal(table)
    rho_values = np.array([row["rho"] for row in rhos])

    result = {
        "matches": int(len(table)),
        "first_match": str(table["MatchDate"].min().date()),
        "last_match": str(table["MatchDate"].max().date()),
        "seasons": sorted(str(s) for s in table["Season"].unique()),
        "home_win": split["home"], "draw": split["draw"], "away_win": split["away"],
        "goals_home": goals["home"], "goals_away": goals["away"],
        "goals_ratio": goals["ratio"],
        "low_scores": {row["score"]: {key: row[key] for key in (
            "observed", "departure_per_match", "departure_low", "departure_high",
            "departure_league_mean")} for row in departures.to_dict("records")},
        "season_rho": rhos,
        "season_rho_median": float(np.median(rho_values)),
        "season_rho_negative": int((rho_values < 0).sum()),
        # Sign test: if rho were zero, each season's estimate would be negative
        # with probability one half. One-sided, because Dixon-Coles predicts
        # the sign.
        "season_rho_sign_test_p": float(binomtest(
            int((rho_values < 0).sum()), len(rho_values), 0.5,
            alternative="greater").pvalue) if len(rho_values) else float("nan"),
        "overround_bet_into": overround(table, ["OddsH", "OddsD", "OddsA"])["mean"],
        "overround_closing": overround(table, ["CloseH", "CloseD", "CloseA"])["mean"],
        "market_ece_home": market["ece"],
        "market_ece_home_noise_floor_p95": market["noise_floor_p95"],
        "market_ece_home_later_half": market_later["ece"],
        "market_ece_later_half_noise_floor_p95": market_later["noise_floor_p95"],
        "market_detectably_miscalibrated": market["detectably_miscalibrated"],
        "market_ece_later_half_matches": market_later["matches"],
        "shots_on_target_r_home": shots["home_shots_on_target"],
        "shots_on_target_r_away": shots["away_shots_on_target"],
    }
    return result
