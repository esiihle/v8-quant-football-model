"""
Phase 6 tests — walk-forward backtest, CLV and the headline metrics.

The test that matters most is `test_lookahead_guard_catches_a_leak`: the engine's
only real job is to never see the future, so that is what gets attacked hardest.

Run with:  pytest tests/test_backtest.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import backtest, clv, dixon_coles, metrics, staking


@pytest.fixture
def config():
    """A config tuned for fast tests: small training window, frequent refits."""
    cfg = load_config()
    cfg["backtest"] = dict(cfg.get("backtest", {}),
                           min_training_matches=120,
                           refit_every_n_days=30,
                           min_calibration_matches=100,
                           bootstrap_resamples=200)
    return cfg


def make_history(n_rounds=90, seed=0, margin=0.055, model_edge=0.0):
    """Simulate a league with a realistic bookmaker and closing line.

    ``model_edge`` shifts the closing line toward the home side relative to the
    opening price, which simulates a market that moves the way our bets point —
    the situation CLV is designed to detect.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team {c}" for c in "ABCDEFGHIJ"]
    attack = {t: rng.normal(0, 0.26) for t in teams}
    defence = {t: rng.normal(0, 0.20) for t in teams}
    gamma, rho = 0.26, -0.08

    rows = []
    start = pd.Timestamp("2023-08-05")
    for rnd in range(n_rounds):
        order = list(rng.permutation(teams))
        for i in range(0, len(order), 2):
            home, away = order[i], order[i + 1]
            lam = np.exp(attack[home] + defence[away] + gamma)
            mu = np.exp(attack[away] + defence[home])

            matrix = dixon_coles.score_matrix(lam, mu, rho, max_goals=8)
            flat = rng.choice(matrix.size, p=matrix.ravel())
            hg, ag = np.unravel_index(flat, matrix.shape)

            probabilities = dixon_coles.market_probabilities(matrix)
            fair = np.array([probabilities["home_win"], probabilities["draw"],
                             probabilities["away_win"]])

            # Opening odds carry the margin.
            opening = 1.0 / (fair * (1 + margin))

            # Closing odds: tighter margin, and shifted by model_edge toward
            # home, so the price we took on home looks good against the close.
            closing_fair = fair.copy()
            closing_fair[0] = np.clip(closing_fair[0] + model_edge, 0.02, 0.95)
            closing_fair = closing_fair / closing_fair.sum()
            closing = 1.0 / (closing_fair * 1.02)

            rows.append({
                "MatchID": f"R{rnd:03d}_{i}",
                "MatchDate": start + pd.Timedelta(days=7 * rnd),
                "League": "SIM", "Season": "2425",
                "HomeTeam": home, "AwayTeam": away,
                "HomeGoals": int(hg), "AwayGoals": int(ag),
                "Result": "H" if hg > ag else "A" if ag > hg else "D",
                "HomeShotsOT": int(rng.poisson(lam * 3)),
                "AwayShotsOT": int(rng.poisson(mu * 3)),
                "OddsH": round(opening[0], 2), "OddsD": round(opening[1], 2),
                "OddsA": round(opening[2], 2),
                "CloseH": round(closing[0], 2), "CloseD": round(closing[1], 2),
                "CloseA": round(closing[2], 2),
            })
    return pd.DataFrame(rows)


def make_bets_with_closing_market(n, taken_home_prob, closing_home_prob,
                                  margin=0.03, seed=0):
    """Build bets whose CLOSING market is a coherent book.

    A fixture mistake worth naming: an earlier version fixed the closing draw
    and away prices while letting the home price vary, so the closing book summed
    to anywhere from 0.84 to 1.03. De-vigging nonsense produces nonsense, and the
    probability-points measure was reading the broken fixture rather than the
    signal the test intended.

    Here the three closing prices are built from probabilities that sum to 1,
    then loaded with a margin — a real book.
    """
    rng = np.random.default_rng(seed)
    taken_home_prob = np.asarray(taken_home_prob, dtype=float)
    closing_home_prob = np.asarray(closing_home_prob, dtype=float)

    # Split the remaining probability between draw and away.
    draw_share = rng.uniform(0.4, 0.6, n)
    closing = np.column_stack([
        closing_home_prob,
        (1 - closing_home_prob) * draw_share,
        (1 - closing_home_prob) * (1 - draw_share),
    ])
    closing_odds = 1.0 / (closing * (1 + margin))

    return pd.DataFrame({
        "odds": 1.0 / taken_home_prob,          # the price we took on home
        "closing_odds": closing_odds[:, 0],
        "outcome": "home_win",
        "closing_home": closing_odds[:, 0],
        "closing_draw": closing_odds[:, 1],
        "closing_away": closing_odds[:, 2],
    })


# =========================================================================
# The lookahead guard — the engine's whole purpose
# =========================================================================

def test_lookahead_guard_passes_a_clean_split():
    history = make_history(30)
    cutoff = history["MatchDate"].sort_values().unique()[15]

    training = history[history["MatchDate"] < cutoff]
    window = history[history["MatchDate"] >= cutoff]

    backtest.assert_no_lookahead(training, window)      # must not raise


def test_lookahead_guard_catches_a_leak():
    """The single most important test in the project.

    If this ever passes something it should not, every performance number
    downstream becomes fiction.
    """
    history = make_history(30)
    cutoff = history["MatchDate"].sort_values().unique()[15]

    window = history[history["MatchDate"] >= cutoff]
    leaky_training = history                              # includes the window

    with pytest.raises(backtest.LookaheadError) as error:
        backtest.assert_no_lookahead(leaky_training, window)

    assert "fiction" in str(error.value)


def test_guard_treats_the_same_day_as_a_leak():
    """The boundary is STRICT. Two matches kicking off the same afternoon cannot
    inform each other: at the moment the bets are placed, neither has happened."""
    history = make_history(30)
    one_date = history["MatchDate"].sort_values().unique()[10]

    window = history[history["MatchDate"] == one_date]
    training = history[history["MatchDate"] <= one_date]   # same date included

    with pytest.raises(backtest.LookaheadError):
        backtest.assert_no_lookahead(training, window)


def test_guard_handles_empty_inputs():
    """The first window has no training data; that is not a leak."""
    history = make_history(10)
    backtest.assert_no_lookahead(history.head(0), history)
    backtest.assert_no_lookahead(history, history.head(0))


# =========================================================================
# The engine
# =========================================================================

def test_walk_forward_runs_and_places_bets(config):
    history = make_history(90, seed=1, model_edge=0.05)
    result = backtest.walk_forward(history, config)

    assert result["counts"]["windows"] > 3
    assert result["counts"]["matches_priced"] > 0
    assert not result["windows"].empty
    assert "Walk-forward backtest" in backtest.backtest_report(result)


def test_walk_forward_never_prices_a_match_without_enough_history(config):
    """No bet may be struck before the minimum training window is satisfied."""
    history = make_history(90, seed=2)
    result = backtest.walk_forward(history, config)

    if not result["bets"].empty:
        earliest_bet = pd.to_datetime(result["bets"]["MatchDate"]).min()
        eligible_from = pd.Timestamp(result["period"]["first_bet_eligible"])
        assert earliest_bet >= eligible_from


def test_every_window_has_more_training_data_than_the_last(config):
    """An expanding window: each refit sees everything the last one did, plus
    the matches that have since been played."""
    history = make_history(90, seed=3)
    result = backtest.walk_forward(history, config)

    training_counts = result["windows"]["training_matches"].to_numpy()
    assert np.all(np.diff(training_counts) >= 0), training_counts


def test_walk_forward_is_reproducible(config):
    history = make_history(60, seed=4, model_edge=0.04)

    first = backtest.walk_forward(history, config)
    second = backtest.walk_forward(history, config)

    if not first["bets"].empty:
        pd.testing.assert_frame_equal(
            first["bets"].reset_index(drop=True),
            second["bets"].reset_index(drop=True))


def test_bets_are_struck_at_the_available_odds_not_the_close(config):
    """The third leak: betting into the closing price would mean betting at a
    price that did not exist when the decision was made."""
    history = make_history(90, seed=5, model_edge=0.05)
    result = backtest.walk_forward(history, config)

    if result["bets"].empty:
        pytest.skip("no bets placed in this run")

    bets = result["bets"]
    # Every recorded `odds` must match an OPENING price from the source data,
    # never a closing one.
    source = history.set_index("MatchID")
    for _, bet in bets.head(40).iterrows():
        opening_column = backtest.OUTCOMES[bet["outcome"]]["odds"]
        assert bet["odds"] == pytest.approx(
            float(source.loc[bet["MatchID"], opening_column]))


def test_too_little_data_is_refused(config):
    tiny = make_history(5)
    with pytest.raises(backtest.BacktestError):
        backtest.walk_forward(tiny, config)


def test_missing_columns_are_refused(config):
    history = make_history(60).drop(columns=["OddsD"])
    with pytest.raises(backtest.BacktestError):
        backtest.walk_forward(history, config)


# =========================================================================
# CLV — the headline metric
# =========================================================================

def test_clv_ratio_by_hand():
    """Took 3.50 where the close was 3.00:  3.50 / 3.00 - 1 = +16.67%."""
    bets = pd.DataFrame([{"odds": 3.50, "closing_odds": 3.00,
                          "outcome": "home_win"}])
    with_clv = clv.clv_per_bet(bets)

    assert with_clv.loc[0, "clv_ratio"] == pytest.approx(1 / 6, abs=1e-4)
    assert bool(with_clv.loc[0, "beat_close"]) is True


def test_clv_is_negative_when_the_price_moves_against_you():
    """Took 2.50 and the close went out to 3.00 — the market decided the outcome
    was LESS likely than we were paid for."""
    bets = pd.DataFrame([{"odds": 2.50, "closing_odds": 3.00,
                          "outcome": "home_win"}])
    with_clv = clv.clv_per_bet(bets)

    assert with_clv.loc[0, "clv_ratio"] < 0
    assert bool(with_clv.loc[0, "beat_close"]) is False


def test_clv_probability_points_use_the_devigged_close():
    """The version worth averaging, because odds ratios are asymmetric: 2.00 to
    1.80 and 1.80 to 2.00 are the same probability move and very different
    ratios."""
    bets = pd.DataFrame([{
        "odds": 3.50, "closing_odds": 3.00, "outcome": "home_win",
        "closing_home": 3.00, "closing_draw": 3.40, "closing_away": 3.60,
    }])
    with_clv = clv.clv_per_bet(bets)

    assert "clv_probability_points" in with_clv.columns
    assert with_clv.loc[0, "clv_probability_points"] > 0


def test_clv_summary_reports_confidence_intervals(config):
    """A mean CLV of +1.5% means nothing without knowing whether the interval
    includes zero."""
    rng = np.random.default_rng(6)
    n = 300
    bets = pd.DataFrame({
        "odds": rng.uniform(2.0, 4.0, n),
        "outcome": rng.choice(["home_win", "draw", "away_win"], n),
        "edge": rng.uniform(0.02, 0.10, n),
    })
    # A genuine positive CLV: the close is consistently shorter than we took.
    bets["closing_odds"] = bets["odds"] * rng.normal(0.95, 0.04, n)

    summary = clv.clv_summary(bets, config)

    assert summary["clv_ratio"]["lower"] < summary["clv_ratio"]["estimate"]
    assert summary["clv_ratio"]["estimate"] < summary["clv_ratio"]["upper"]
    assert summary["clv_ratio"]["estimate"] > 0
    assert summary["positive_and_significant"] is True
    assert "POSITIVE CLV" in clv.clv_report(summary)


def test_clv_summary_refuses_to_claim_an_edge_on_noise(config):
    """The honesty check. Random price moves must NOT be reported as an edge."""
    rng = np.random.default_rng(7)
    n = 400

    # We take a price implying 0.35; the close lands randomly around 0.35 too,
    # so there is no information in either direction.
    taken = np.full(n, 0.35)
    closing = np.clip(0.35 + rng.normal(0, 0.02, n), 0.1, 0.7)

    bets = make_bets_with_closing_market(n, taken, closing, seed=7)
    summary = clv.clv_summary(bets, config)
    report = clv.clv_report(summary)

    assert summary["positive_and_significant"] is False
    assert ("not distinguishable from zero" in report
            or "NEGATIVE" in report
            or "DISAGREE" in report)


def test_clv_needs_usable_closing_odds():
    bets = pd.DataFrame([{"odds": 3.0, "closing_odds": np.nan,
                          "outcome": "home_win"}])
    with pytest.raises(clv.CLVError):
        clv.clv_per_bet(bets)


def test_clv_by_band_shows_whether_the_edge_estimate_carries_signal(config):
    """If the model knows WHERE its edge is, high-claimed-edge bets should show
    the most CLV. A flat profile means the estimate is noise."""
    rng = np.random.default_rng(8)
    n = 400
    edge = rng.uniform(0.0, 0.12, n)
    bets = pd.DataFrame({
        "odds": rng.uniform(2.0, 4.0, n),
        "outcome": rng.choice(["home_win", "draw", "away_win"], n),
        "edge": edge,
    })
    # Build in real signal: more claimed edge means the close moves further.
    bets["closing_odds"] = bets["odds"] * (1 - edge * 0.8)

    table = clv.clv_by_band(bets, "edge", n_bands=4, config=config)

    assert len(table) == 4
    assert table["mean_clv_ratio"].is_monotonic_increasing


# =========================================================================
# Headline metrics
# =========================================================================

def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(9)
    values = rng.normal(0.03, 0.5, 500)

    interval = metrics.bootstrap_ci(values, n_resamples=500)
    assert interval["lower"] < interval["estimate"] < interval["upper"]
    assert interval["n"] == 500


def test_bootstrap_widens_as_the_sample_shrinks():
    """The whole reason intervals are reported: +3% over 200 bets and +3% over
    20,000 are different claims."""
    rng = np.random.default_rng(10)
    population = rng.normal(0.03, 0.8, 20000)

    small = metrics.bootstrap_ci(population[:200], n_resamples=400)
    large = metrics.bootstrap_ci(population, n_resamples=400)

    small_width = small["upper"] - small["lower"]
    large_width = large["upper"] - large["lower"]
    assert small_width > large_width * 3


def test_max_drawdown_by_hand():
    """Peak 120, trough 90:  (120 - 90) / 120 = 25%."""
    curve = [100, 110, 120, 100, 90, 130]
    assert metrics.max_drawdown(curve) == pytest.approx(0.25)


def test_max_drawdown_is_zero_for_a_rising_curve():
    assert metrics.max_drawdown([100, 110, 120, 130]) == pytest.approx(0.0)


def test_sharpe_like_is_nan_without_variation():
    assert np.isnan(metrics.sharpe_like([0.01, 0.01, 0.01]))


def test_summarise_reports_intervals_and_a_verdict(config):
    history = make_history(90, seed=11, model_edge=0.06)
    result = backtest.walk_forward(history, config)

    if result["bets"].empty:
        pytest.skip("no bets placed in this run")

    ledger, bankroll = staking.simulate_bankroll(result["bets"], config)
    summary = metrics.summarise(ledger, bankroll, config)

    assert summary["bets"] == len(ledger)
    assert summary["yield"]["lower"] < summary["yield"]["upper"]
    assert 0 <= summary["max_drawdown"] <= 1

    report = metrics.metrics_report(summary)
    assert "bootstrap intervals" in report
    assert ("excludes zero" in report) or ("INCLUDES ZERO" in report)


def test_summarise_handles_no_bets(config):
    summary = metrics.summarise(pd.DataFrame(), None, config)
    assert summary["bets"] == 0
    assert "nothing to report" in metrics.metrics_report(summary)


def test_performance_by_band_needs_profit_and_edge():
    with pytest.raises(ValueError):
        metrics.performance_by_band(pd.DataFrame({"edge": [0.1, 0.2]}))


# =========================================================================
# End to end
# =========================================================================

def test_full_backtest_to_clv_pipeline(config):
    """The roadmap's 'done when': a leakage-free backtest with CLV and the
    headline metrics, all reported with confidence intervals."""
    history = make_history(120, seed=12, model_edge=0.05)
    result = backtest.walk_forward(history, config)

    if result["bets"].empty:
        pytest.skip("no bets placed in this run")

    ledger, bankroll = staking.simulate_bankroll(result["bets"], config)
    headline = metrics.summarise(ledger, bankroll, config)
    clv_result = clv.clv_summary(result["bets"], config)

    # Every headline figure carries an interval.
    for key in ("yield", "hit_rate"):
        assert set(headline[key]) >= {"estimate", "lower", "upper", "n"}
    for key in ("clv_ratio", "beat_rate"):
        assert set(clv_result[key]) >= {"estimate", "lower", "upper", "n"}

    # And the reports render.
    assert "Headline metrics" in metrics.metrics_report(headline)
    assert "Closing-line value" in clv.clv_report(clv_result)


# =========================================================================
# Regression test for the 18 Oct 2026 CLV verdict bug
# =========================================================================

def test_disagreeing_clv_measures_are_flagged_not_papered_over(config):
    """The first real walk-forward run produced this, and the report got it wrong.

        mean CLV (odds ratio)          +1.82%  [+0.81%, +3.00%]   positive
        mean CLV (probability points)  -0.67%  [-0.87%, -0.47%]   NEGATIVE
        beat the close                 47.46%                      below 50%

    Both intervals excluded zero, in opposite directions, and the report said
    "the confidence interval INCLUDES ZERO" — which was simply false.

    The cause is the asymmetry of the odds ratio: a price halving scores -50%
    while a doubling scores +100%, so the mean is pulled up by its right tail.
    One +108% outlier was carrying it while the model was losing to the close.
    """
    rng = np.random.default_rng(31)
    n = 400

    # Most bets lose a little to the close; a few win enormously. In probability
    # points the average is negative; as an odds ratio it comes out positive.
    taken = rng.uniform(2.0, 4.0, n)
    closing = taken * 1.04                      # the close is usually longer
    outliers = rng.choice(n, 12, replace=False)
    closing[outliers] = taken[outliers] * 0.45  # a few huge price collapses

    bets = pd.DataFrame({
        "odds": taken,
        "closing_odds": closing,
        "outcome": "home_win",
        "closing_home": closing,
        "closing_draw": np.full(n, 3.4),
        "closing_away": np.full(n, 3.6),
    })

    summary = clv.clv_summary(bets, config)
    report = clv.clv_report(summary)

    assert summary["headline_measure"] == "probability points"
    if summary["measures_disagree"]:
        assert "DISAGREE" in report
        assert "authoritative" in report
        # And the flattering measure must not be allowed to stand as the verdict.
        assert "POSITIVE CLV on" not in report


def test_a_clearly_negative_clv_is_called_negative(config):
    """No hedging when the answer is unambiguous.

    We take prices implying 0.40 and the close settles at 0.34 — the market
    decides the outcome is LESS likely than we were paid for. Both CLV measures
    and the beat rate should agree that this is bad.
    """
    n = 400
    taken = np.full(n, 0.40)           # we take a price implying 40%
    closing = np.full(n, 0.34)         # the close says 34%

    bets = make_bets_with_closing_market(n, taken, closing, seed=32)
    summary = clv.clv_summary(bets, config)
    report = clv.clv_report(summary)

    assert summary["positive_and_significant"] is False
    assert summary["points_verdict"] == "negative"
    assert "NEGATIVE" in report


def test_beat_rate_is_reported_with_its_own_verdict(config):
    """The tie-breaker, because it needs no averaging and no de-vigging."""
    rng = np.random.default_rng(33)
    n = 300
    taken = rng.uniform(2.0, 4.0, n)

    bets = pd.DataFrame({
        "odds": taken,
        "closing_odds": taken * 0.92,          # the close always shortens
        "outcome": "home_win",
    })

    summary = clv.clv_summary(bets, config)
    assert summary["beat_rate"]["lower"] > 0.5
    assert "above 50%" in clv.clv_report(summary)
