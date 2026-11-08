"""
The model against the market, scored on the same matches.

Why this module exists
----------------------
The project's verdict so far ("no edge") comes from CLV: does the model find
prices the market later moves toward? This module asks the prior question
directly: are the model's PROBABILITIES better or worse than the market's?

A fair comparison has three requirements, and each one was missing before:

1. **The same matches.** Every walk-forward match the model priced, scored
   against the market's probabilities for those same matches.
2. **Out-of-sample model probabilities.** The walk-forward ones, from a model
   and calibration map that had seen only earlier matches. Phase 4's
   probabilities come from one fit at the end of the data, which has seen the
   results it is scored on (and on 13 seasons, with the minimum match weight,
   prices 2013 matches with 2024 team strengths).
3. **A paired test.** Model and market are scored match by match, and the
   bootstrap resamples MATCHES, keeping each match's two scores together. Most
   of the variation in log-loss is between matches (an upset is expensive for
   everyone), so the paired difference is far more precise than comparing two
   separate averages.

Two market benchmarks
---------------------
* **opening** (Bet365, the price the model bets INTO): the market's view with
  roughly the same information the model had. To find value, the model must
  beat this one, at least on the bets it makes.
* **closing** (Pinnacle at kick-off): the market's best estimate, with
  everything it learned by kick-off. The harder bar, and the one CLV uses.

Both are de-vigged with the pipeline's own method (Shin), so the margin is not
counted against the market.
"""
import numpy as np
import pandas as pd

from src import calibrate, staking
from src.exploration import ece_noise_floor

PROBABILITY_COLUMNS = ["home_win", "draw", "away_win"]
LABELS = ["H", "D", "A"]
MARKETS = {"opening": ["OddsH", "OddsD", "OddsA"],
           "closing": ["CloseH", "CloseD", "CloseA"]}
LOG_CLIP = 1e-12


def market_probability_matrix(priced, columns, devig_method="shin"):
    """De-vigged market probabilities, one row per match (home, draw, away)."""
    odds = priced[columns].to_numpy(dtype=float)
    return np.array([staking.market_probabilities(row, devig_method)[0]
                     for row in odds])


def per_match_scores(probabilities, results):
    """Log-loss and Brier for EACH match, so they can be compared pairwise.

    Log-loss: minus the log of the probability given to what happened.
    Brier: squared distance from the probabilities to the outcome, summed over
    the three outcomes (0 is perfect, 2 is certain and wrong).
    """
    probabilities = np.asarray(probabilities, dtype=float)
    realised = np.column_stack([(np.asarray(results) == label).astype(float)
                                for label in LABELS])
    chosen = np.clip(np.sum(probabilities * realised, axis=1), LOG_CLIP, 1.0)
    return {"log_loss": -np.log(chosen),
            "brier": np.sum((probabilities - realised) ** 2, axis=1)}


def paired_difference(model_scores, market_scores, n_resamples=2000,
                      confidence=0.95, seed=42):
    """Mean of (model - market) per match, with a bootstrap over MATCHES.

    Negative means the model scored better (both scores are losses). One index
    draw selects the same matches for both, which is what makes it paired.
    """
    difference = np.asarray(model_scores, float) - np.asarray(market_scores, float)
    difference = difference[np.isfinite(difference)]
    if len(difference) < 2:
        return {"estimate": float("nan"), "lower": float("nan"),
                "upper": float("nan"), "n": int(len(difference))}
    rng = np.random.default_rng(seed)
    draws = np.array([difference[rng.integers(0, len(difference), len(difference))].mean()
                      for _ in range(n_resamples)])
    tail = (1 - confidence) / 2
    return {"estimate": float(difference.mean()),
            "lower": float(np.quantile(draws, tail)),
            "upper": float(np.quantile(draws, 1 - tail)),
            "n": int(len(difference))}


def verdict(interval):
    """Who wins on a loss-difference interval (model minus market)."""
    if interval["upper"] < 0:
        return "model better"
    if interval["lower"] > 0:
        return "market better"
    return "no detectable difference"


def compare(priced, config, n_resamples=2000, seed=42):
    """Score the model and both markets on every match all three can price.

    Parameters
    ----------
    priced : DataFrame
        backtest.walk_forward(...)["priced"]: the model's out-of-sample
        probabilities, the odds and the result for every priced match.

    Returns
    -------
    dict: per-forecaster scores, paired differences and verdicts, and the
    home-win calibration error of each against its noise floor.
    """
    devig = config.get("staking", {}).get("devig_method", "shin")
    needed = PROBABILITY_COLUMNS + MARKETS["opening"] + MARKETS["closing"] + ["Result"]
    data = priced.dropna(subset=[c for c in needed if c in priced.columns])
    data = data[data["Result"].isin(LABELS)]
    odds_ok = np.all(data[MARKETS["opening"] + MARKETS["closing"]].to_numpy(float) > 1.0,
                     axis=1)
    data = data[odds_ok]

    forecasts = {"model": data[PROBABILITY_COLUMNS].to_numpy(dtype=float)}
    for name, columns in MARKETS.items():
        forecasts[name] = market_probability_matrix(data, columns, devig)

    results = data["Result"].to_numpy()
    scores = {name: per_match_scores(p, results) for name, p in forecasts.items()}
    home_won = (results == "H").astype(float)

    summary = {"matches": int(len(data)), "forecasters": {}, "differences": {}}
    for name, p in forecasts.items():
        floor = ece_noise_floor(p[:, 0], 10, n_sims=300, seed=seed)
        summary["forecasters"][name] = {
            "log_loss": float(scores[name]["log_loss"].mean()),
            "brier": float(scores[name]["brier"].mean()),
            "ece_home": float(calibrate.expected_calibration_error(p[:, 0], home_won, 10)),
            "ece_home_floor_p95": floor["p95"],
        }
    for market in MARKETS:
        for rule in ("log_loss", "brier"):
            interval = paired_difference(scores["model"][rule], scores[market][rule],
                                         n_resamples, seed=seed)
            interval["verdict"] = verdict(interval)
            summary["differences"][f"{rule}_vs_{market}"] = interval
    return summary


def benchmark_report(summary):
    """The comparison, readable in a terminal."""
    lines = ["Model vs market, same matches", "-" * 29,
             f"  matches scored (walk-forward, out of sample)   {summary['matches']:,}", "",
             f"  {'forecaster':<22}{'log-loss':>10}{'Brier':>10}{'ECE home':>10}{'floor p95':>11}"]
    names = {"model": "model (walk-forward)", "opening": "opening (Bet365)",
             "closing": "closing (Pinnacle)"}
    for key, label in names.items():
        f = summary["forecasters"][key]
        lines.append(f"  {label:<22}{f['log_loss']:>10.4f}{f['brier']:>10.4f}"
                     f"{f['ece_home']:>10.4f}{f['ece_home_floor_p95']:>11.4f}")
    lines += ["", "  Paired differences, model minus market (negative = model better):"]
    for key, d in summary["differences"].items():
        rule, market = key.split("_vs_")
        lines.append(f"    {rule:<9} vs {market:<8} {d['estimate']:+.4f}  "
                     f"[{d['lower']:+.4f}, {d['upper']:+.4f}]  {d['verdict']}")
    return "\n".join(lines)
