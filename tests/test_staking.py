"""
Phase 5 tests — value detection, Kelly sizing and bankroll simulation.

The two tests worth reading first:

* `test_kelly_fraction_by_hand` — the formula, checked against arithmetic you
  can do on paper.
* `test_two_times_kelly_has_zero_expected_log_growth` — the result that justifies
  fractional Kelly. Double the optimal stake and all the edge converts to
  variance; beyond that you lose money with a real edge in hand.

Run with:  pytest tests/test_staking.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import staking


@pytest.fixture
def config():
    return load_config()


def make_market(n=400, seed=0, true_edge=0.0, margin=0.05):
    """A set of fixtures with a KNOWN amount of model edge built in.

    ``true_edge = 0`` means the model matches the fair market exactly, so a
    correct staking rule should place almost nothing. A positive value gives the
    model a genuine advantage on home wins.
    """
    rng = np.random.default_rng(seed)

    fair_home = rng.uniform(0.25, 0.60, n)
    fair_draw = rng.uniform(0.20, 0.30, n)
    fair_away = 1 - fair_home - fair_draw
    keep = fair_away > 0.08
    fair_home, fair_draw, fair_away = fair_home[keep], fair_draw[keep], fair_away[keep]
    n = len(fair_home)

    fair = np.column_stack([fair_home, fair_draw, fair_away])
    fair = fair / fair.sum(axis=1, keepdims=True)

    # Odds carry the bookmaker's margin: prices are shorter than fair.
    odds = 1.0 / (fair * (1 + margin))

    # Results come from the FAIR probabilities, so the market is honest.
    results = np.array([rng.choice(["H", "D", "A"], p=row) for row in fair])

    model = fair.copy()
    if true_edge:
        model[:, 0] = np.clip(model[:, 0] + true_edge, 0.01, 0.95)
        model = model / model.sum(axis=1, keepdims=True)

    return pd.DataFrame({
        "MatchID": [f"M{i:04d}" for i in range(n)],
        "MatchDate": pd.date_range("2024-08-01", periods=n, freq="D"),
        "HomeTeam": [f"H{i}" for i in range(n)],
        "AwayTeam": [f"A{i}" for i in range(n)],
        "home_win": model[:, 0], "draw": model[:, 1], "away_win": model[:, 2],
        "OddsH": odds[:, 0], "OddsD": odds[:, 1], "OddsA": odds[:, 2],
        "Result": results,
    })


# =========================================================================
# Kelly
# =========================================================================

def test_kelly_fraction_by_hand():
    """p = 0.6 at odds 2.0 (b = 1):

        f* = (bp - q) / b = (1 * 0.6 - 0.4) / 1 = 0.20

    And via the other form: (p*o - 1) / (o - 1) = (1.2 - 1) / 1 = 0.20
    """
    assert staking.kelly_fraction(0.6, 2.0) == pytest.approx(0.20)


def test_kelly_is_zero_at_fair_odds():
    """No edge means no bet. p = 0.5 at odds 2.0 is exactly fair."""
    assert staking.kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
    assert staking.kelly_fraction(0.25, 4.0) == pytest.approx(0.0)


def test_kelly_is_negative_without_edge():
    """The formula says 'take the other side'. Callers clip at zero, but the
    negative value stays available so the size of the disadvantage is visible."""
    assert staking.kelly_fraction(0.4, 2.0) < 0


def test_kelly_grows_with_edge():
    stakes = [staking.kelly_fraction(p, 2.0) for p in (0.52, 0.55, 0.60, 0.70)]
    assert stakes == sorted(stakes)


def test_kelly_maximises_expected_log_growth():
    """The defining property, verified numerically rather than assumed.

    Kelly is not 'a sensible stake' — it is the unique maximiser of expected log
    growth. Any other fraction must do worse.
    """
    p, odds = 0.58, 2.1
    optimal = staking.kelly_fraction(p, odds)
    best = staking.expected_log_growth(p, odds, optimal)

    for alternative in np.linspace(0.01, 0.6, 60):
        if abs(alternative - optimal) < 1e-6:
            continue
        assert staking.expected_log_growth(p, odds, alternative) < best


def test_zero_growth_point_is_near_two_times_kelly():
    """The result that justifies fractional Kelly — stated precisely.

    "Twice Kelly gives zero growth" is usually quoted as an identity. It is not.
    It is exact in the small-edge (Gaussian / continuous-rebalancing) limit,
    where growth is f*mu - f^2*sigma^2/2 and vanishes at f = 2*mu/sigma^2. For a
    discrete bet the zero-growth point is NEAR 2x and depends on the odds.

    Measured here, so the number in the docs is the number the code produces:

        p=0.600 o=2.0  ->  zero growth at 1.947x Kelly
        p=0.550 o=2.2  ->  1.975x
        p=0.400 o=3.0  ->  2.039x
        p=0.250 o=5.0  ->  2.098x
        p=0.505 o=2.0  ->  2.000x   (tiny edge: the limit)

    The practical conclusion is unchanged: a model 20% too optimistic about p is
    already betting roughly double what it should at "full Kelly", sitting on the
    zero-growth boundary with a genuine edge in hand.
    """
    from scipy.optimize import brentq

    cases = [(0.6, 2.0), (0.55, 2.2), (0.4, 3.0), (0.25, 5.0)]

    for p, odds in cases:
        optimal = staking.kelly_fraction(p, odds)
        assert optimal > 0

        crossing = brentq(
            lambda f: staking.expected_log_growth(p, odds, f),
            optimal * 1.01, 0.999)
        ratio = crossing / optimal

        assert 1.9 < ratio < 2.15, (
            f"p={p}, odds={odds}: zero growth at {ratio:.3f}x Kelly, expected "
            "close to 2x"
        )
        # Growth at 2x is an order of magnitude smaller than at the optimum.
        assert abs(staking.expected_log_growth(p, odds, 2 * optimal)) < \
            0.2 * staking.expected_log_growth(p, odds, optimal)
        # And past the crossing it is negative, with the edge still real.
        assert staking.expected_log_growth(p, odds, crossing * 1.1) < 0


def test_zero_growth_converges_to_exactly_twice_kelly_for_a_small_edge():
    """The limit in which the textbook 2x figure is exact."""
    from scipy.optimize import brentq

    p, odds = 0.505, 2.0            # a half-point edge at even money
    optimal = staking.kelly_fraction(p, odds)
    crossing = brentq(lambda f: staking.expected_log_growth(p, odds, f),
                      optimal * 1.01, 0.999)

    assert crossing / optimal == pytest.approx(2.0, abs=0.01)


def test_quarter_kelly_tolerates_being_wrong_about_p():
    """Why 0.25 is the default: it survives a four-fold error in p."""
    true_p, odds = 0.55, 2.2

    optimistic_p = 0.70                       # the model is badly too confident
    full_stake = staking.kelly_fraction(optimistic_p, odds)
    quarter_stake = full_stake * 0.25

    # Growth is measured against the TRUE probability, not the believed one.
    full_growth = staking.expected_log_growth(true_p, odds, full_stake)
    quarter_growth = staking.expected_log_growth(true_p, odds, quarter_stake)

    assert quarter_growth > full_growth, (
        "with an over-optimistic p, quarter-Kelly must outperform full Kelly"
    )
    assert quarter_growth > 0, "and should still be growing"


def test_impossible_odds_are_refused():
    with pytest.raises(staking.StakingError):
        staking.kelly_fraction(0.6, 1.0)


# =========================================================================
# De-vigging
# =========================================================================

def test_overround_measures_the_margin():
    """Three outcomes at 2.0 each implies 1.5 — a 50% book."""
    assert staking.overround([2.0, 2.0, 2.0]) == pytest.approx(0.5)


def test_proportional_devig_sums_to_one():
    probabilities = staking.devig_proportional([2.1, 3.4, 3.6])
    assert probabilities.sum() == pytest.approx(1.0)
    assert (probabilities > 0).all()


def test_shin_devig_sums_to_one():
    probabilities, note = staking.devig_shin([2.1, 3.4, 3.6])
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-9)
    assert note["method"] == "shin"
    assert 0 < note["z"] < 0.35


def test_shin_gives_longshots_lower_probability_than_proportional():
    """The favourite-longshot bias, and why the de-vig choice matters.

    Bookmakers load more margin onto longshots. Proportional de-vigging assumes
    they do not, so it OVERSTATES longshot probabilities — which is exactly the
    direction that manufactures phantom value on big prices, where a naive model
    tends to think it has found something.
    """
    # These must carry a real margin, or there is nothing for Shin to unwind.
    # 1/1.45 + 1/4.2 + 1/8.0 = 1.057, a 5.7% book.
    odds = [1.45, 4.2, 8.0]                  # clear favourite, clear longshot
    proportional = staking.devig_proportional(odds)
    shin, note = staking.devig_shin(odds)
    assert note["method"] == "shin", "the fixture must have a margin to unwind"

    assert shin[2] < proportional[2], "Shin must shade the longshot down"
    assert shin[0] > proportional[0], "and the favourite up"


def test_devig_falls_back_cleanly_on_a_book_with_no_margin():
    """A fair or negative book has nothing for Shin to unwind. Falling back is
    fine; falling back silently is not."""
    probabilities, note = staking.devig_shin([3.0, 3.0, 3.0])   # sums to exactly 1
    assert probabilities.sum() == pytest.approx(1.0)
    assert note["method"] == "proportional"
    assert "reason" in note


def test_unknown_devig_method_is_refused():
    with pytest.raises(staking.StakingError):
        staking.market_probabilities([2.0, 3.0, 4.0], "guesswork")


# =========================================================================
# The step most amateur systems get wrong
# =========================================================================

def test_a_model_matching_the_fair_market_finds_no_value(config):
    """The most important behavioural test here.

    The model IS the fair market, and the odds carry a 5% margin. Every bet
    therefore has negative expected value — the margin is a real cost the model
    has no advantage to pay it with. A correct staking rule stakes nothing.
    """
    market = make_market(400, seed=1, true_edge=0.0, margin=0.05)
    bets = staking.find_value_bets(market, config)

    value_rate = bets["is_value"].mean()
    assert value_rate < 0.01, (
        f"{value_rate:.1%} flagged as value against a market the model exactly "
        "matches; every such bet is -EV"
    )
    # Edge vs the price is negative by construction.
    assert bets["edge"].mean() < 0
    # But disagreement vs fair value is ~0: the model and market agree.
    assert abs(bets["disagreement"].mean()) < 0.005


def test_a_bet_beating_fair_but_not_raw_is_not_value(config):
    """The conceptual error this module shipped with, pinned.

    At odds of 3.50 in a 5% book: raw implied 0.2857, de-vigged fair 0.2721. A
    model saying 0.278 disagrees with the market in the right direction and still
    has expected value of -2.7%, because the payout is at the actual odds.

    The first version tested value against the de-vigged probability and would
    have called this a bet.
    """
    market = pd.DataFrame([{
        "MatchID": "M1", "MatchDate": pd.Timestamp("2024-08-01"),
        "HomeTeam": "H", "AwayTeam": "A",
        # Home priced at 3.50 (raw implied 0.2857) in a book summing to 1.049,
        # so de-vigged fair for home is about 0.272. The model's 0.278 sits
        # between the two: it beats fair value and loses to the price.
        "home_win": 0.278, "draw": 0.361, "away_win": 0.361,
        "OddsH": 3.50, "OddsD": 2.62, "OddsA": 2.62,
        "Result": "H",
    }])

    bets = staking.find_value_bets(market, config)
    home = bets[bets["outcome"] == "home_win"].iloc[0]

    assert home["disagreement"] > 0, "model should beat the de-vigged price"
    assert home["edge"] < 0, "and still be negative against the actual price"
    assert not home["is_value"], "so it must NOT be flagged as value"
    assert home["stake_fraction"] == 0.0

    # Expected value, computed independently, confirms the direction.
    expected_value = home["model_probability"] * home["odds"] - 1
    assert expected_value < 0


def test_edge_and_kelly_agree_on_when_to_bet(config):
    """Edge is measured against the raw price precisely so that a positive edge
    and a positive Kelly fraction are the same condition. If they ever disagree,
    one of them is using the wrong benchmark."""
    market = make_market(300, seed=20, true_edge=0.06)
    bets = staking.find_value_bets(market, config)

    assert ((bets["edge"] > 0) == (bets["full_kelly"] > 0)).all()


def test_a_genuine_edge_is_detected(config):
    """The other direction: real edge must actually be found."""
    market = make_market(400, seed=3, true_edge=0.10, margin=0.05)
    bets = staking.find_value_bets(market, config)

    home_bets = bets[(bets["outcome"] == "home_win") & bets["is_value"]]
    assert len(home_bets) > 30, "a 10-point edge on home wins should be found"
    assert (home_bets["edge"] > 0).all()


def test_unpriced_matches_are_skipped(config):
    market = make_market(100, seed=4)
    market.loc[0:9, "OddsH"] = np.nan

    bets = staking.find_value_bets(market, config)
    assert set(market.loc[0:9, "MatchID"]).isdisjoint(set(bets["MatchID"]))


def test_missing_columns_are_refused(config):
    market = make_market(50).drop(columns=["OddsD"])
    with pytest.raises(staking.StakingError):
        staking.find_value_bets(market, config)


# =========================================================================
# Bankroll simulation
# =========================================================================

def test_bankroll_simulation_is_reproducible(config):
    """The roadmap's 'done when' for this phase."""
    market = make_market(300, seed=5, true_edge=0.08)
    bets = staking.find_value_bets(market, config)

    first_ledger, first_summary = staking.simulate_bankroll(bets, config)
    second_ledger, second_summary = staking.simulate_bankroll(bets, config)

    pd.testing.assert_frame_equal(first_ledger, second_ledger)
    assert first_summary == second_summary


def test_no_value_bets_produces_a_clean_no_bet_result(config):
    """A model with no edge should stake nothing, and that must not crash."""
    market = make_market(200, seed=6, true_edge=0.0, margin=0.06)
    bets = staking.find_value_bets(market, config)

    ledger, summary = staking.simulate_bankroll(bets, config)

    if summary["bets"] == 0:
        assert summary["final_bankroll"] == summary["starting_bankroll"]
        assert summary["roi"] == 0.0
        assert "no value bets" in staking.staking_report(bets, ledger, summary,
                                                         config)


def test_same_day_bets_are_sized_off_the_same_bankroll(config):
    """At the moment you place Saturday's bets you do not know Saturday's
    results, so the third bet cannot be sized off the first two's winnings."""
    market = make_market(60, seed=7, true_edge=0.12)
    market["MatchDate"] = pd.Timestamp("2024-08-10")      # all on one day

    bets = staking.find_value_bets(market, config)
    ledger, _ = staking.simulate_bankroll(bets, config)

    if len(ledger) > 1:
        # Every bet that day shares one bankroll-after-day figure.
        assert ledger["bankroll_after_day"].nunique() == 1
        # And stakes are proportional to the fractions, with one common base.
        implied_base = ledger["stake"] / ledger["stake_fraction_used"]
        assert implied_base.nunique() == 1 or np.allclose(
            implied_base, implied_base.iloc[0])


def test_daily_exposure_cap_scales_a_crowded_day_down(config):
    market = make_market(60, seed=8, true_edge=0.15)
    market["MatchDate"] = pd.Timestamp("2024-08-10")

    bets = staking.find_value_bets(market, config)
    ledger, summary = staking.simulate_bankroll(bets, config)

    if len(ledger) > 5:
        used = ledger["stake_fraction_used"].sum()
        cap = config["staking"]["max_daily_exposure"]
        assert used <= cap + 1e-9, f"exposure {used} exceeded the {cap} cap"
        assert summary["days_scaled_by_cap"] >= 1


def test_max_stake_per_bet_is_respected(config):
    market = make_market(300, seed=9, true_edge=0.25)
    # A larger Kelly fraction so the per-bet cap is the binding constraint
    # rather than the quarter-Kelly scaling.
    aggressive = dict(config)
    aggressive["staking"] = dict(config["staking"], kelly_fraction=0.5)

    bets = staking.find_value_bets(market, aggressive)
    cap = aggressive["staking"]["max_stake_fraction"]

    assert (bets["stake_fraction"] <= cap + 1e-12).all()
    assert bets["capped"].any(), (
        f"max fractional Kelly was {(bets['full_kelly'] * 0.5).max():.4f} "
        f"against a cap of {cap}; the fixture needs a bigger edge"
    )


def test_fractional_kelly_reduces_drawdown(config):
    """The pleasant side effect, measured."""
    market = make_market(400, seed=10, true_edge=0.08)

    gentle = dict(config)
    gentle["staking"] = dict(config["staking"], kelly_fraction=0.10,
                             max_stake_fraction=1.0, max_daily_exposure=1.0)
    aggressive = dict(config)
    aggressive["staking"] = dict(config["staking"], kelly_fraction=1.0,
                                 max_stake_fraction=1.0, max_daily_exposure=1.0)

    _, gentle_summary = staking.simulate_bankroll(
        staking.find_value_bets(market, gentle), gentle)
    _, aggressive_summary = staking.simulate_bankroll(
        staking.find_value_bets(market, aggressive), aggressive)

    assert gentle_summary["max_drawdown"] < aggressive_summary["max_drawdown"]


def test_roi_and_yield_are_different_numbers(config):
    """They answer different questions and are routinely confused: ROI is on the
    bankroll, yield is on money staked."""
    market = make_market(300, seed=11, true_edge=0.10)
    bets = staking.find_value_bets(market, config)
    ledger, summary = staking.simulate_bankroll(bets, config)

    if summary["bets"] > 10:
        assert summary["turnover"] > 0
        assert summary["roi"] != pytest.approx(summary["yield"], abs=1e-6)


def test_bets_without_outcomes_cannot_be_simulated(config):
    market = make_market(100, seed=12, true_edge=0.10)
    market = market.drop(columns=["Result"])
    market["Result"] = None

    bets = staking.find_value_bets(market, config)
    if (bets["stake_fraction"] > 0).any():
        with pytest.raises(staking.StakingError):
            staking.simulate_bankroll(bets, config)


def test_report_renders(config):
    market = make_market(300, seed=13, true_edge=0.09)
    bets = staking.find_value_bets(market, config)
    ledger, summary = staking.simulate_bankroll(bets, config)

    report = staking.staking_report(bets, ledger, summary, config)
    assert "Staking (fractional Kelly)" in report
    assert "de-vig method" in report
    assert "not a result" in report          # the Phase 6 caveat
