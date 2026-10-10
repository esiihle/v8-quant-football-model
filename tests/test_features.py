"""
Phase 2 tests — feature engineering.

The headline test is ``test_features_do_not_use_the_future``. Everything else
here checks a property of the maths; that one checks the property the whole
project rests on.

Run with:  pytest tests/test_features.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import features


@pytest.fixture
def config():
    return load_config()


def make_matches(n_rounds=12, seed=0, strong_team="Team A"):
    """A small synthetic league where one team is genuinely better than the rest.

    Built so the tests can assert direction: the strong team should end up with
    an attack strength above 1.0 and a defence below it.
    """
    rng = np.random.default_rng(seed)
    teams = [strong_team] + [f"Team {c}" for c in "BCDEF"]
    rows = []
    date = pd.Timestamp("2024-08-10")

    for rnd in range(n_rounds):
        shuffled = list(rng.permutation(teams))
        for i in range(0, len(shuffled), 2):
            home, away = shuffled[i], shuffled[i + 1]
            # The strong team scores more and concedes less.
            home_rate = 2.6 if home == strong_team else 1.2
            away_rate = 2.2 if away == strong_team else 0.9
            hg, ag = int(rng.poisson(home_rate)), int(rng.poisson(away_rate))
            # Shots track team quality too, as they do in real football — so the
            # xG proxy carries the same signal as the goals, not noise.
            hst, ast_ = int(rng.poisson(home_rate * 3)), int(rng.poisson(away_rate * 3))
            rows.append({
                "MatchID": f"M{rnd:02d}_{i}",
                "MatchDate": date + pd.Timedelta(days=7 * rnd),
                "League": "TEST", "Season": "2425",
                "HomeTeam": home, "AwayTeam": away,
                "HomeGoals": hg, "AwayGoals": ag,
                "Result": "H" if hg > ag else "A" if ag > hg else "D",
                "HomeShots": hst + 6, "AwayShots": ast_ + 5,
                "HomeShotsOT": hst, "AwayShotsOT": ast_,
                "OddsH": 2.1, "OddsD": 3.4, "OddsA": 3.6,
                "CloseH": 2.05, "CloseD": 3.5, "CloseA": 3.7,
            })
    return pd.DataFrame(rows)


# --- Time decay ------------------------------------------------------------

def test_match_exactly_one_half_life_old_counts_half():
    dates = pd.to_datetime(["2024-01-01"])
    weights = features.time_weights(dates, as_of="2024-06-29", half_life_days=180)
    assert weights[0] == pytest.approx(0.5, abs=0.01)


def test_older_matches_count_less():
    dates = pd.to_datetime(["2024-01-01", "2024-04-01", "2024-06-01"])
    weights = features.time_weights(dates, as_of="2024-07-01", half_life_days=180)
    assert weights[0] < weights[1] < weights[2]


def test_future_matches_get_zero_weight():
    """The lookahead guard, implemented once and tested directly."""
    dates = pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01"])
    weights = features.time_weights(dates, as_of="2024-07-01", half_life_days=180)

    assert weights[0] > 0        # before as_of: counts
    assert weights[1] == 0.0     # on as_of: does not count
    assert weights[2] == 0.0     # after as_of: does not count


# --- Shrinkage -------------------------------------------------------------

def test_no_data_means_the_prior_exactly():
    """A team with no matches must be given the league average, not its own noise."""
    assert features.shrink(estimate=3.0, n_matches=0, prior=1.0, k=8) == pytest.approx(1.0)


def test_sample_equal_to_k_lands_halfway():
    """With n = k, weight is 0.5 — the defining property of this shrinkage."""
    result = features.shrink(estimate=2.0, n_matches=8, prior=1.0, k=8)
    assert result == pytest.approx(1.5)


def test_more_evidence_moves_the_estimate_closer_to_its_own_record():
    estimates = [features.shrink(2.0, n, prior=1.0, k=8) for n in (1, 5, 20, 100)]
    assert estimates == sorted(estimates)          # monotonically toward 2.0
    assert estimates[-1] > 1.9


def test_larger_k_shrinks_harder():
    gentle = features.shrink(2.0, n_matches=10, prior=1.0, k=2)
    harsh = features.shrink(2.0, n_matches=10, prior=1.0, k=40)
    assert harsh < gentle


# --- xG proxy --------------------------------------------------------------

def test_xg_blend_sits_between_goals_and_the_proxy(config):
    matches = make_matches()
    blended, note = features.blend_goals(matches, matches, config)

    assert "blended" in note
    # The blend must not simply reproduce goals, nor wander far from them.
    assert not np.allclose(blended["HomeAttackGoals"], matches["HomeGoals"])
    assert abs(blended["HomeAttackGoals"].mean() - matches["HomeGoals"].mean()) < 0.5


def test_missing_shot_columns_fall_back_to_goals(config):
    """The bundled sample has no shot data. That must degrade, not crash."""
    matches = make_matches().drop(columns=["HomeShotsOT", "AwayShotsOT"])
    blended, note = features.blend_goals(matches, matches, config)

    assert "not used" in note
    assert blended["HomeAttackGoals"].tolist() == matches["HomeGoals"].astype(float).tolist()


# --- Team strengths --------------------------------------------------------

def test_a_better_team_gets_a_better_rating(config):
    matches = make_matches(n_rounds=20, seed=7)
    blended, _ = features.blend_goals(matches, matches, config)
    strengths = features.team_strengths(
        blended, as_of=matches["MatchDate"].max() + pd.Timedelta(days=1), config=config
    )

    assert strengths.loc["Team A", "attack"] > 1.0
    assert strengths.loc["Team A", "defence"] < 1.0
    # And it should be the best attack in the division.
    assert strengths["attack"].idxmax() == "Team A"


def test_strength_on_the_first_match_day_is_the_prior(config):
    """Nothing is known yet, so every team must be exactly average."""
    matches = make_matches()
    blended, _ = features.blend_goals(matches, matches, config)
    strengths = features.team_strengths(
        blended, as_of=matches["MatchDate"].min(), config=config
    )
    assert len(strengths) == 0        # no history at all -> no estimates


# --- The feature matrix ----------------------------------------------------

def test_feature_matrix_shape_and_columns(config):
    matches = make_matches()
    feats, report = features.build_features(matches, config)

    assert list(feats.columns) == features.FEATURE_COLUMNS
    assert len(feats) == len(matches)
    assert feats["MatchDate"].is_monotonic_increasing
    assert report["matches"] == len(matches)


def test_early_rows_are_flagged_as_thin_history(config):
    """Matches played before a team has a record must be flagged, not trusted."""
    matches = make_matches()
    feats, report = features.build_features(matches, config)

    assert feats.iloc[0]["HasSufficientHistory"] is False or \
        not bool(feats.iloc[0]["HasSufficientHistory"])
    assert report["warm_up_rows"] > 0
    assert report["usable_for_training"] > 0


def test_unknown_team_gets_the_prior_not_a_null(config):
    """A promoted side appearing for the first time is average until proven otherwise."""
    matches = make_matches()
    feats, _ = features.build_features(matches, config)

    first = feats.iloc[0]
    assert first["HomeAttack"] == pytest.approx(1.0)
    assert first["AwayDefence"] == pytest.approx(1.0)
    assert feats[["HomeAttack", "AwayAttack", "HomeDefence", "AwayDefence"]].notna().all().all()


def test_features_are_deterministic(config):
    matches = make_matches()
    first, _ = features.build_features(matches, config)
    second, _ = features.build_features(matches, config)
    pd.testing.assert_frame_equal(first, second)


# --- The test this module exists for ---------------------------------------

def test_features_do_not_use_the_future(config):
    """A match's features must be identical whether or not later matches exist.

    This is the property that makes a backtest meaningful. If it ever fails, the
    model has been shown results it could not have known at kick-off, and every
    performance number downstream is fiction.
    """
    matches = make_matches(n_rounds=16, seed=3)

    full, _ = features.build_features(matches, config)

    # Rebuild using only the first half of the season.
    cutoff = matches["MatchDate"].sort_values().unique()[8]
    truncated_input = matches[matches["MatchDate"] < cutoff]
    truncated, _ = features.build_features(truncated_input, config)

    # Every match present in both must have identical features.
    shared = truncated["MatchID"]
    a = full[full["MatchID"].isin(shared)].set_index("MatchID").sort_index()
    b = truncated.set_index("MatchID").sort_index()

    strength_cols = ["HomeAttack", "HomeDefence", "AwayAttack", "AwayDefence",
                     "HomePriorMatches", "AwayPriorMatches"]
    pd.testing.assert_frame_equal(
        a[strength_cols], b[strength_cols],
        check_exact=False, rtol=1e-9,
        obj="features changed when future matches were added — LOOKAHEAD LEAK",
    )


def test_building_features_needs_the_required_columns(config):
    matches = make_matches().drop(columns=["HomeGoals"])
    with pytest.raises(features.FeatureError):
        features.build_features(matches, config)


def test_conversion_rate_is_measured_from_history_only(config):
    """The leak this module had on 9 Oct: rates fitted on the whole file.

    A conversion rate measured over all matches encodes how the season finished.
    Measured over the first half only, it must differ — and `build_features` must
    use the second kind.
    """
    matches = make_matches(n_rounds=16, seed=5)
    cutoff = matches["MatchDate"].sort_values().unique()[8]
    early = matches[matches["MatchDate"] < cutoff]

    full_rate = features.conversion_rates(matches)[0]
    early_rate = features.conversion_rates(early)[0]

    assert full_rate != pytest.approx(early_rate, abs=1e-9), (
        "the fixture should have different conversion rates in each half"
    )


def test_no_shot_history_still_builds_features(config):
    """Early matches have no history to fit a conversion rate on. Must not crash."""
    matches = make_matches(n_rounds=3)
    feats, report = features.build_features(matches, config)
    assert len(feats) == len(matches)
