"""
Phase 3 tests — the Dixon-Coles model.

Three kinds of test here, in increasing order of what they prove:

1. **Identities.** Properties the maths must satisfy exactly: tau is 1 outside
   the four low-score cells, the correction preserves total probability, rho = 0
   reduces to independent Poisson, every market sums to 1.
2. **Recovery.** Simulate matches from known parameters, fit the model, and check
   it finds them again. This is the real test of a maximum-likelihood estimator:
   if it cannot recover parameters it generated, nothing it says about real data
   can be believed.
3. **Discipline.** The fit respects `as_of`, so the walk-forward backtest in
   Phase 6 cannot see the future.

Run with:  pytest tests/test_dixon_coles.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import dixon_coles as dc


@pytest.fixture
def config():
    return load_config()


def simulate_league(n_rounds=26, seed=0, gamma=0.26, rho=-0.08):
    """Simulate matches from KNOWN Dixon-Coles parameters.

    Returns the match frame and the true parameters, so a test can fit the model
    and check it recovers what generated the data.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team {c}" for c in "ABCDEFGH"]

    # True attack parameters, forced to sum to zero exactly as the model assumes.
    true_attack = np.linspace(0.45, -0.45, len(teams))
    true_attack = true_attack - true_attack.mean()
    true_defence = np.linspace(-0.30, 0.30, len(teams))

    attack = dict(zip(teams, true_attack))
    defence = dict(zip(teams, true_defence))

    rows = []
    start = pd.Timestamp("2024-08-10")
    for rnd in range(n_rounds):
        order = list(rng.permutation(teams))
        for i in range(0, len(order), 2):
            home, away = order[i], order[i + 1]
            lam = np.exp(attack[home] + defence[away] + gamma)
            mu = np.exp(attack[away] + defence[home])

            # Draw from the Dixon-Coles distribution itself (not independent
            # Poisson) so the simulated data really contains the rho effect.
            matrix = dc.score_matrix(lam, mu, rho, max_goals=8)
            flat = rng.choice(matrix.size, p=matrix.ravel())
            hg, ag = np.unravel_index(flat, matrix.shape)

            rows.append({
                "MatchID": f"R{rnd:02d}_{i}",
                "MatchDate": start + pd.Timedelta(days=7 * rnd),
                "League": "SIM", "Season": "2425",
                "HomeTeam": home, "AwayTeam": away,
                "HomeGoals": int(hg), "AwayGoals": int(ag),
                "Result": "H" if hg > ag else "A" if ag > hg else "D",
            })

    truth = {"attack": attack, "defence": defence, "gamma": gamma, "rho": rho}
    return pd.DataFrame(rows), truth


# =========================================================================
# 1. Identities
# =========================================================================

def test_tau_is_one_outside_the_four_low_scores():
    """The correction must touch 0-0, 0-1, 1-0 and 1-1 and nothing else."""
    lam, mu, rho = 1.5, 1.1, -0.1

    for x, y in [(2, 0), (0, 2), (2, 2), (3, 1), (1, 3), (5, 4)]:
        assert dc.tau(x, y, lam, mu, rho) == pytest.approx(1.0)

    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert dc.tau(x, y, lam, mu, rho) != pytest.approx(1.0)


def test_tau_formulas_match_the_paper():
    """Each of the four cells, checked against Dixon & Coles (1997) by hand."""
    lam, mu, rho = 1.4, 1.2, -0.11

    assert dc.tau(0, 0, lam, mu, rho) == pytest.approx(1 - lam * mu * rho)
    assert dc.tau(0, 1, lam, mu, rho) == pytest.approx(1 + lam * rho)
    assert dc.tau(1, 0, lam, mu, rho) == pytest.approx(1 + mu * rho)
    assert dc.tau(1, 1, lam, mu, rho) == pytest.approx(1 - rho)


def test_negative_rho_raises_low_score_draws_and_lowers_1_0():
    """The direction the data asks for, and the reason the parameter exists."""
    lam, mu, rho = 1.5, 1.2, -0.10

    assert dc.tau(0, 0, lam, mu, rho) > 1      # more 0-0
    assert dc.tau(1, 1, lam, mu, rho) > 1      # more 1-1
    assert dc.tau(1, 0, lam, mu, rho) < 1      # fewer 1-0
    assert dc.tau(0, 1, lam, mu, rho) < 1      # fewer 0-1


def test_tau_correction_preserves_total_probability():
    """The four adjustments cancel exactly, so no probability is created or lost.

    This is a real property of the Dixon-Coles construction, not an accident of
    our implementation, and it is why the score matrix needs only a negligible
    renormalisation for truncation.
    """
    lam, mu, rho = 1.6, 1.25, -0.12
    goals = np.arange(0, 60)        # far enough out that the tail is nothing

    home = poisson.pmf(goals, lam)
    away = poisson.pmf(goals, mu)
    independent = np.outer(home, away)

    x, y = np.meshgrid(goals, goals, indexing="ij")
    corrected = independent * dc.tau(x, y, lam, mu, rho)

    assert corrected.sum() == pytest.approx(independent.sum(), abs=1e-12)
    assert corrected.sum() == pytest.approx(1.0, abs=1e-9)


def test_rho_zero_is_exactly_independent_poisson():
    """With the correction switched off the model must reduce to the baseline."""
    lam, mu = 1.7, 1.1
    matrix = dc.score_matrix(lam, mu, rho=0.0, max_goals=25)

    expected = np.outer(poisson.pmf(np.arange(26), lam),
                        poisson.pmf(np.arange(26), mu))
    expected = expected / expected.sum()

    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_score_matrix_sums_to_one():
    for lam, mu, rho in [(1.5, 1.2, -0.1), (0.6, 2.4, 0.0), (3.0, 0.4, -0.05)]:
        matrix = dc.score_matrix(lam, mu, rho, max_goals=10)
        assert matrix.sum() == pytest.approx(1.0, abs=1e-12)
        assert (matrix >= 0).all()


def test_every_market_sums_to_one():
    """1X2, over/under and BTTS are all partitions of the same matrix."""
    probabilities = dc.market_probabilities(
        dc.score_matrix(1.6, 1.2, -0.09, max_goals=12)
    )

    assert (probabilities["home_win"] + probabilities["draw"]
            + probabilities["away_win"]) == pytest.approx(1.0, abs=1e-9)
    assert (probabilities["over_2_5"]
            + probabilities["under_2_5"]) == pytest.approx(1.0, abs=1e-9)
    assert (probabilities["btts_yes"]
            + probabilities["btts_no"]) == pytest.approx(1.0, abs=1e-9)


def test_stronger_home_team_is_more_likely_to_win():
    """A basic directional sanity check — the kind a reviewer asks for first."""
    strong = dc.market_probabilities(dc.score_matrix(2.2, 0.8, -0.1))
    even = dc.market_probabilities(dc.score_matrix(1.4, 1.4, -0.1))

    assert strong["home_win"] > even["home_win"]
    assert strong["away_win"] < even["away_win"]


def test_rho_bounds_keep_probabilities_valid():
    """Inside the bounds every cell is non-negative; outside, the model breaks."""
    lam, mu = 1.5, 1.2
    low, high = dc.rho_bounds(lam, mu)

    for rho in (low + 1e-6, 0.0, high - 1e-6):
        assert (dc.score_matrix(lam, mu, rho, max_goals=10) >= 0).all()

    # Well outside the valid range, a cell would go negative. The matrix clips
    # and renormalises rather than returning nonsense, but the clipping is itself
    # the signal that rho is invalid — so the optimiser is bounded to avoid it.
    assert low > -np.inf and high < np.inf


# =========================================================================
# 2. Parameter recovery — the real test of the estimator
# =========================================================================

def test_fit_recovers_known_parameters(config):
    """Simulate from known parameters, fit, and check we find them again.

    The sample is deliberately large. On 240 matches gamma scatters by about
    +/- 0.12 and rho by more, which would make this test flaky without proving
    anything about the estimator. Recovery is an asymptotic property, so it is
    tested at a sample size where the noise is small enough to see past.
    """
    matches, truth = simulate_league(n_rounds=200, seed=11)

    # A long half-life and a light penalty: this test is about the estimator, not
    # about recency weighting or regularisation.
    cfg = dict(config)
    cfg["features"] = dict(config["features"], half_life_days=100000)
    cfg["model"] = dict(config["model"], penalty=0.0005, n_starts=3)

    model = dc.fit(matches, cfg)

    fitted_attack = np.array([model["attack"][t] for t in model["teams"]])
    true_attack = np.array([truth["attack"][t] for t in model["teams"]])
    fitted_defence = np.array([model["defence"][t] for t in model["teams"]])
    true_defence = np.array([truth["defence"][t] for t in model["teams"]])

    # Rank order of team strength must be recovered strongly.
    assert np.corrcoef(fitted_attack, true_attack)[0, 1] > 0.9
    assert np.corrcoef(fitted_defence, true_defence)[0, 1] > 0.85

    # Home advantage and rho recovered to a sensible tolerance. rho gets the
    # looser one on purpose: it is identified only by four cells of the score
    # matrix, so it is always the hardest parameter here to pin down.
    assert model["home_advantage"] == pytest.approx(truth["gamma"], abs=0.06)
    assert model["rho"] == pytest.approx(truth["rho"], abs=0.05)


def test_attack_parameters_sum_to_zero(config):
    """The identifiability constraint. Without it the fit is not unique."""
    matches, _ = simulate_league(n_rounds=20, seed=2)
    model = dc.fit(matches, config)

    total = sum(model["attack"].values())
    assert total == pytest.approx(0.0, abs=1e-8)


def test_home_advantage_comes_out_positive(config):
    """Home sides score more in the simulated data, so gamma must be > 0."""
    matches, _ = simulate_league(n_rounds=30, seed=5)
    model = dc.fit(matches, config)

    assert model["home_advantage"] > 0
    # exp(gamma) is the home scoring multiplier; 1.1x-1.5x is the realistic band.
    assert 1.05 < np.exp(model["home_advantage"]) < 1.6


def test_multiple_starts_agree(config):
    """Convergence means several independent starts find the same optimum.

    If they scatter, the likelihood surface is awkward and the fit should not be
    trusted — which is why the report warns rather than staying silent.
    """
    matches, _ = simulate_league(n_rounds=30, seed=8)
    cfg = dict(config)
    cfg["model"] = dict(config["model"], n_starts=4)

    model = dc.fit(matches, cfg)
    diagnostics = model["diagnostics"]

    assert diagnostics["starts_tried"] >= 3
    assert diagnostics["agreed_across_starts"] is True
    assert diagnostics["objective_spread"] < 1e-3


def test_penalty_shrinks_strengths_toward_average(config):
    """The regularisation is Phase 2's shrinkage in the language of an optimiser."""
    matches, _ = simulate_league(n_rounds=12, seed=4)

    light = dict(config); light["model"] = dict(config["model"], penalty=0.0001)
    heavy = dict(config); heavy["model"] = dict(config["model"], penalty=5.0)

    spread_light = np.std(list(dc.fit(matches, light)["attack"].values()))
    spread_heavy = np.std(list(dc.fit(matches, heavy)["attack"].values()))

    assert spread_heavy < spread_light


# =========================================================================
# 3. Discipline — no lookahead
# =========================================================================

def test_fit_uses_only_matches_before_as_of(config):
    """What keeps the Phase 6 walk-forward backtest honest."""
    matches, _ = simulate_league(n_rounds=30, seed=6)
    cutoff = matches["MatchDate"].sort_values().unique()[15]

    with_cutoff = dc.fit(matches, config, as_of=cutoff)
    only_past = dc.fit(matches[matches["MatchDate"] < cutoff], config, as_of=cutoff)

    assert with_cutoff["matches_used"] == only_past["matches_used"]
    for team in with_cutoff["teams"]:
        assert with_cutoff["attack"][team] == pytest.approx(
            only_past["attack"][team], abs=1e-8
        )


def test_fit_with_no_prior_matches_fails_loudly(config):
    matches, _ = simulate_league(n_rounds=10, seed=1)
    with pytest.raises(dc.ConvergenceError):
        dc.fit(matches, config, as_of=matches["MatchDate"].min())


# =========================================================================
# Prediction
# =========================================================================

def test_predict_produces_valid_probabilities(config):
    matches, _ = simulate_league(n_rounds=30, seed=9)
    model = dc.fit(matches, config)

    predictions = dc.predict(model, matches.head(10), config)

    assert len(predictions) == 10
    assert predictions["KnownTeams"].all()
    outcome_sum = (predictions["home_win"] + predictions["draw"]
                   + predictions["away_win"])
    np.testing.assert_allclose(outcome_sum, 1.0, atol=1e-9)
    assert (predictions["ExpectedHomeGoals"] > 0).all()


def test_unknown_team_is_priced_at_league_average(config):
    """A promoted side must be priceable on debut, and flagged as unknown."""
    matches, _ = simulate_league(n_rounds=20, seed=3)
    model = dc.fit(matches, config)

    fixture = pd.DataFrame([{
        "MatchID": "NEW", "MatchDate": pd.Timestamp("2025-05-01"),
        "HomeTeam": "Newly Promoted FC", "AwayTeam": model["teams"][0],
    }])
    prediction = dc.predict(model, fixture, config)

    assert prediction.loc[0, "KnownTeams"] is False or \
        not bool(prediction.loc[0, "KnownTeams"])
    assert 0 < prediction.loc[0, "home_win"] < 1


def test_model_survives_a_json_round_trip(config, tmp_path):
    matches, _ = simulate_league(n_rounds=20, seed=7)
    model = dc.fit(matches, config)

    cfg = dict(config)
    cfg["data"] = dict(config["data"], processed_dir=str(tmp_path))
    reloaded = dc.load_model(dc.save_model(model, cfg))

    before = dc.predict(model, matches.head(5), config)
    after = dc.predict(reloaded, matches.head(5), config)
    pd.testing.assert_frame_equal(before, after)


def test_fit_report_renders(config):
    matches, _ = simulate_league(n_rounds=20, seed=10)
    report = dc.fit_report(dc.fit(matches, config))

    assert "Dixon-Coles fit" in report
    assert "home advantage" in report
    assert "rho" in report


# =========================================================================
# Regression tests for bugs found during the Phase 3 build (10 Oct 2026)
# =========================================================================

def test_rho_is_not_pinned_at_its_bound(config):
    """The second bug of 10 Oct 2026.

    The rho box was sized from the largest lambda the strength box could
    theoretically produce — about 55 goals a match — which squeezed the box to
    [-0.018, 0.0003]. rho sat on the lower edge, the fit reported success, and
    the starts agreed with each other. Nothing looked wrong.

    The tell was that the bias did not shrink as the sample grew. Noise averages
    out; a binding constraint does not.
    """
    matches, truth = simulate_league(n_rounds=120, seed=21)
    cfg = dict(config)
    cfg["features"] = dict(config["features"], half_life_days=100000)
    cfg["model"] = dict(config["model"], penalty=0.0005)

    model = dc.fit(matches, cfg)
    diagnostics = model["diagnostics"]

    assert diagnostics["rho_at_bound"] is False, (
        f"rho {model['rho']:.4f} is stuck on the edge of {diagnostics['rho_box']} "
        "— the bound is driving the estimate, not the data"
    )
    low, high = diagnostics["rho_box"]
    assert low <= truth["rho"] <= high, (
        f"the true rho {truth['rho']} is outside the box {diagnostics['rho_box']}, "
        "so it could never have been recovered"
    )


def test_rho_bias_shrinks_as_the_sample_grows(config):
    """The diagnostic that exposed the bound bug. A bias that ignores sample
    size is a constraint or a coding error, never bad luck."""
    cfg = dict(config)
    cfg["features"] = dict(config["features"], half_life_days=100000)
    cfg["model"] = dict(config["model"], penalty=0.0005, n_starts=2)

    small, truth = simulate_league(n_rounds=40, seed=22)
    large, _ = simulate_league(n_rounds=400, seed=22)

    error_small = abs(dc.fit(small, cfg)["rho"] - truth["rho"])
    error_large = abs(dc.fit(large, cfg)["rho"] - truth["rho"])

    assert error_large < 0.04, f"rho still off by {error_large:.4f} on a large sample"
    assert error_large <= error_small + 0.02


def test_likelihood_never_returns_infinity(config):
    """The first bug of 10 Oct 2026.

    Returning +inf for an invalid parameter set destroys L-BFGS-B's
    finite-difference gradient (inf - inf = nan), so the optimiser stops
    improving while still reporting success. Parameter recovery came back at
    r = 0.21 with no error raised anywhere.

    An invalid point must cost a large but FINITE amount, growing with the size
    of the violation, so there is always a way back downhill.
    """
    matches, _ = simulate_league(n_rounds=20, seed=23)
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    index = {t: i for i, t in enumerate(teams)}

    home_idx = matches["HomeTeam"].map(index).to_numpy()
    away_idx = matches["AwayTeam"].map(index).to_numpy()
    home_goals = matches["HomeGoals"].to_numpy(dtype=float)
    away_goals = matches["AwayGoals"].to_numpy(dtype=float)
    weights = np.ones(len(matches))

    # A deliberately invalid rho: large and positive makes tau(0,0) negative.
    bad = dc._pack(np.zeros(len(teams)), np.zeros(len(teams)), 0.25, 0.95)
    value = dc.log_likelihood(bad, home_idx, away_idx, home_goals, away_goals,
                              weights, len(teams), 0.01)

    assert np.isfinite(value), "an invalid parameter set must not return infinity"
    assert value > 1e6, "an invalid parameter set must still be heavily penalised"


def test_invalid_region_penalty_has_a_usable_gradient(config):
    """Deeper into the invalid region must cost strictly more, so the optimiser
    can feel which way is out."""
    matches, _ = simulate_league(n_rounds=20, seed=24)
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    index = {t: i for i, t in enumerate(teams)}
    args = (matches["HomeTeam"].map(index).to_numpy(),
            matches["AwayTeam"].map(index).to_numpy(),
            matches["HomeGoals"].to_numpy(dtype=float),
            matches["AwayGoals"].to_numpy(dtype=float),
            np.ones(len(matches)), len(teams), 0.01)

    mild = dc.log_likelihood(
        dc._pack(np.zeros(len(teams)), np.zeros(len(teams)), 0.25, 0.60), *args)
    severe = dc.log_likelihood(
        dc._pack(np.zeros(len(teams)), np.zeros(len(teams)), 0.25, 0.99), *args)

    assert severe > mild, "the penalty must increase with the size of the violation"
