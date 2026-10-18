r"""
Stage 6 — Walk-forward backtesting.  [Built in Phase 6, 18 Oct 2026]

**The honesty layer.** Everything before this phase produced numbers that were,
in one way or another, fitted on data they were then measured against. This
module is what turns them into a forecast.

The protocol
------------
Step forward through time in refit windows. At each step:

1. **Fit the Dixon-Coles model** on matches strictly before the window opens.
2. **Fit the calibration map** on matches strictly before the window opens.
3. **Price** the matches inside the window with that frozen model.
4. **Stake** using those prices and the odds that were available.
5. **Record** the odds taken and the closing odds, for CLV.

Then move the window and refit. Nothing inside a window ever influences the model
that priced it.

Why a refit cadence rather than refitting every match
-----------------------------------------------------
Refitting before every single match is the purest version and is also what a
real bettor could not do — the model would change between two matches kicking
off at the same time. A cadence (`backtest.refit_every_n_days`) is both honest
and realistic: it is what someone running this system weekly would actually have.
The cost is that a model is up to one window stale when it prices the last match
in that window, which is a conservative error, not a flattering one.

The guard
---------
`assert_no_lookahead` is called on **every** window, every run, and raises rather
than warns. It is not a test-only helper: a leak that only appears on real data
with a particular season boundary would never be caught by a unit test, so the
check runs in production.

Three leaks this phase is built to prevent
------------------------------------------
* **Model leak** — fitting on matches after the bet. Guarded per window.
* **Calibration leak** — the subtler one. A calibration map fitted on all the
  data has seen the outcomes of the matches it is calibrating, and the effect is
  invisible in the output: probabilities simply look better than they were.
* **Odds leak** — staking at the closing price. The closing line is the
  *benchmark*; betting into it would mean betting at a price that did not exist
  when the decision was made, and would quietly inflate everything.

See docs/methodology.md section 6.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import calibrate, dixon_coles, features, staking

# Mapping from the model's probability columns to the result codes, and to the
# odds columns the bet is actually placed at.
OUTCOMES = {
    "home_win": {"code": "H", "odds": "OddsH", "closing": "CloseH"},
    "draw": {"code": "D", "odds": "OddsD", "closing": "CloseD"},
    "away_win": {"code": "A", "odds": "OddsA", "closing": "CloseA"},
}


class LookaheadError(Exception):
    """Raised when training data is found at or after the matches being priced.

    A distinct exception type because this is not an ordinary failure: it means
    a result produced by this engine would be fiction.
    """


class BacktestError(Exception):
    """Raised when the backtest cannot run on the data given."""


# ===========================================================================
# The guard
# ===========================================================================

def assert_no_lookahead(training, window, date_column="MatchDate"):
    """Raise if any training row is dated at or after the earliest priced match.

    Called on every window of every run, not just in tests. A leak that only
    shows up on real data at a particular season boundary would never be caught
    by a unit test with synthetic fixtures.

    Note the boundary is **strict**: a match on the same date as the window's
    first match is excluded from training. Two matches on one Saturday afternoon
    cannot inform each other, because at the moment the bets are placed neither
    has happened.
    """
    if len(window) == 0:
        return
    if len(training) == 0:
        return

    train_dates = pd.to_datetime(pd.Series(training[date_column]))
    window_start = pd.to_datetime(pd.Series(window[date_column])).min()

    offending = train_dates[train_dates >= window_start]
    if len(offending) > 0:
        raise LookaheadError(
            f"{len(offending)} training matches are dated on or after "
            f"{window_start:%Y-%m-%d}, the first match being priced. The latest "
            f"offender is {offending.max():%Y-%m-%d}. Any result from this run "
            "would be fiction."
        )


# ===========================================================================
# The engine
# ===========================================================================

def walk_forward(matches, config=None, progress=False):
    """Run a leakage-free walk-forward backtest over the match history.

    Parameters
    ----------
    matches : pandas.DataFrame
        The canonical table from Phase 1, carrying results, the odds bet into
        (``OddsH/D/A``) and the closing odds (``CloseH/D/A``).
    config : dict | None
        Reads the ``backtest`` block plus the model, calibration and staking
        blocks used by the stages it drives.
    progress : bool
        Print one line per window. Useful on a long run.

    Returns
    -------
    dict
        The bet history, the per-window record, and the counts needed to explain
        why the number of bets is what it is.
    """
    config = config or {}
    cfg = config.get("backtest", {})
    min_training = int(cfg.get("min_training_matches", 200))
    refit_days = int(cfg.get("refit_every_n_days", 14))
    calibrate_during = bool(cfg.get("calibrate", True))
    min_calibration = int(cfg.get("min_calibration_matches", 150))

    required = ["MatchDate", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals",
                "Result", "OddsH", "OddsD", "OddsA"]
    missing = [c for c in required if c not in matches.columns]
    if missing:
        raise BacktestError(f"The backtest needs columns {missing}")

    data = matches.copy()
    data["MatchDate"] = pd.to_datetime(data["MatchDate"])
    data = data.sort_values("MatchDate").reset_index(drop=True)

    if len(data) <= min_training:
        raise BacktestError(
            f"{len(data)} matches is not enough to back-test with a "
            f"{min_training}-match minimum training window. Use more seasons or "
            "lower backtest.min_training_matches — but lowering it means the "
            "early fits are built on very little."
        )

    # The first date we can price: the earliest with enough history behind it.
    start_date = data["MatchDate"].iloc[min_training]
    windows = pd.date_range(start_date, data["MatchDate"].max() + pd.Timedelta(days=1),
                            freq=f"{refit_days}D")

    all_bets = []
    window_log = []
    counts = {"windows": 0, "windows_skipped_no_fit": 0,
              "matches_priced": 0, "matches_unpriceable": 0}

    for index, window_start in enumerate(windows):
        window_end = (windows[index + 1] if index + 1 < len(windows)
                      else data["MatchDate"].max() + pd.Timedelta(days=1))

        window = data[(data["MatchDate"] >= window_start)
                      & (data["MatchDate"] < window_end)]
        if len(window) == 0:
            continue

        training = data[data["MatchDate"] < window_start]
        if len(training) < min_training:
            continue

        # THE GUARD. Every window, every run.
        assert_no_lookahead(training, window)

        counts["windows"] += 1

        # --- Fit the model on history only ------------------------------
        try:
            blended, _ = features.blend_goals(training, training, config)
            seeds = features.team_strengths(
                blended, as_of=window_start, config=config)
            model = dixon_coles.fit(training, config, as_of=window_start,
                                    seed_strengths=seeds)
        except (dixon_coles.ConvergenceError, features.FeatureError) as error:
            counts["windows_skipped_no_fit"] += 1
            window_log.append({
                "window_start": window_start, "matches": len(window),
                "training_matches": len(training), "bets": 0,
                "note": f"no fit: {error}",
            })
            continue

        # --- Price the window -------------------------------------------
        priced = dixon_coles.predict(model, window, config)
        priced["MatchDate"] = window["MatchDate"].to_numpy()

        # --- Calibrate, also on history only ----------------------------
        # The subtle leak. A map fitted on everything has seen the outcomes of
        # the matches it is calibrating, and the effect is invisible: the
        # probabilities simply look better than they were.
        calibration_note = "not calibrated"
        if calibrate_during and len(training) >= min_calibration:
            try:
                training_priced = dixon_coles.predict(model, training, config)
                training_priced["MatchDate"] = training["MatchDate"].to_numpy()
                calibration = calibrate.fit_multiclass(
                    training_priced, training["Result"].to_numpy(),
                    config.get("calibration", {}).get("method", "platt"))
                priced, _ = calibrate.apply_multiclass(priced, calibration)
                calibration_note = "calibrated on training history"
            except calibrate.CalibrationError as error:
                calibration_note = f"calibration skipped: {error}"

        # --- Attach the odds that existed when the bet would be placed ---
        for column in ("OddsH", "OddsD", "OddsA", "CloseH", "CloseD", "CloseA",
                       "Result"):
            if column in window.columns:
                priced[column] = window[column].to_numpy()

        # --- Stake --------------------------------------------------------
        bets = staking.find_value_bets(priced, config)
        counts["matches_priced"] += len(window)

        if not bets.empty:
            placed = bets[bets["stake_fraction"] > 0].copy()
            if not placed.empty:
                # Record the closing price for CLV. Note this is used ONLY as a
                # benchmark afterwards — the bet is struck at `odds`.
                placed["closing_odds"] = [
                    priced.set_index("MatchID").loc[
                        row["MatchID"], OUTCOMES[row["outcome"]]["closing"]]
                    if OUTCOMES[row["outcome"]]["closing"] in priced.columns
                    else np.nan
                    for _, row in placed.iterrows()
                ]
                for label, column in (("closing_home", "CloseH"),
                                      ("closing_draw", "CloseD"),
                                      ("closing_away", "CloseA")):
                    if column in priced.columns:
                        placed[label] = priced.set_index("MatchID").loc[
                            placed["MatchID"], column].to_numpy()

                placed["window_start"] = window_start
                placed["training_matches"] = len(training)
                all_bets.append(placed)

        window_log.append({
            "window_start": window_start,
            "matches": len(window),
            "training_matches": len(training),
            "bets": int((bets["stake_fraction"] > 0).sum()) if not bets.empty else 0,
            "note": calibration_note,
        })

        if progress:
            print(f"    {window_start:%Y-%m-%d}: {len(window):>3} matches, "
                  f"{len(training):>4} training, "
                  f"{window_log[-1]['bets']:>2} bets ({calibration_note})")

    history = (pd.concat(all_bets, ignore_index=True) if all_bets
               else pd.DataFrame())

    return {
        "bets": history,
        "windows": pd.DataFrame(window_log),
        "counts": counts,
        "settings": {
            "min_training_matches": min_training,
            "refit_every_n_days": refit_days,
            "calibrated": calibrate_during,
        },
        "period": {
            "first_match": str(data["MatchDate"].min().date()),
            "last_match": str(data["MatchDate"].max().date()),
            "first_bet_eligible": str(start_date.date()),
        },
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def backtest_report(result, bankroll_summary=None):
    """What the walk-forward engine did, and what it is and is not."""
    counts = result["counts"]
    settings = result["settings"]
    period = result["period"]

    lines = ["Walk-forward backtest", "-" * 21]
    lines.append(f"  {'match history':<32} {period['first_match']} to "
                 f"{period['last_match']}")
    lines.append(f"  {'first bet eligible from':<32} "
                 f"{period['first_bet_eligible']}")
    lines.append(f"  {'minimum training matches':<32} "
                 f"{settings['min_training_matches']:>10,}")
    lines.append(f"  {'refit cadence (days)':<32} "
                 f"{settings['refit_every_n_days']:>10}")
    lines.append(f"  {'calibrated per window':<32} "
                 f"{str(settings['calibrated']):>10}")
    lines.append("")
    lines.append(f"  {'windows run':<32} {counts['windows']:>10,}")
    if counts["windows_skipped_no_fit"]:
        lines.append(f"  {'windows skipped (no fit)':<32} "
                     f"{counts['windows_skipped_no_fit']:>10,}")
    lines.append(f"  {'matches priced':<32} {counts['matches_priced']:>10,}")
    lines.append(f"  {'bets placed':<32} {len(result['bets']):>10,}")

    if bankroll_summary:
        lines.append("")
        lines.append("  Bankroll (walk-forward — these are the honest numbers):")
        lines.append(f"    {'ROI (on bankroll)':<30} "
                     f"{bankroll_summary['roi']:>+10.2%}")
        lines.append(f"    {'yield (on turnover)':<30} "
                     f"{bankroll_summary['yield']:>+10.2%}")
        lines.append(f"    {'hit rate':<30} "
                     f"{bankroll_summary['hit_rate']:>10.2%}")
        lines.append(f"    {'maximum drawdown':<30} "
                     f"{bankroll_summary['max_drawdown']:>10.2%}")
        if bankroll_summary.get("ruined"):
            lines.append("    RUINED — the bankroll reached zero.")

    lines.append("")
    lines.append("  Every window was checked for lookahead before it was priced:")
    lines.append("  the model and the calibration map saw only matches dated "
                 "strictly before")
    lines.append("  the window opened, and bets were struck at the available "
                 "odds, never the")
    lines.append("  closing price. The closing line is the benchmark, not a "
                 "price we bet into.")
    return "\n".join(lines)
