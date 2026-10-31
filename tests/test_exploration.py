"""
Tests for src/exploration.py and scripts/explore.py. [added 29 Oct 2026]

The central claim of the exploration is that the answer to "are low scores
Poisson?" depends on the null: a league-mean Poisson mistakes team mixing for a
low-score effect, and a per-match Poisson (team strengths known) does not. The
first two tests establish that on data whose truth is known by construction.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import dixon_coles as dc
from src import exploration as ex


@pytest.fixture(scope="module")
def config():
    return load_config()


def simulate_seasons(n_seasons=6, rho=0.0, spread=0.45, seed=0):
    """A canonical-format league with KNOWN low-score behaviour.

    `spread` sets how different the teams are (sd of log attack/defence); a
    wide spread is what makes a league-mean Poisson misleading. Scores are
    drawn from the Dixon-Coles distribution with the given rho: rho = 0 is
    exactly independent Poisson per match.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(20)]
    rows = []
    for season in range(n_seasons):
        attack = dict(zip(teams, rng.normal(0, spread, 20)))
        defence = dict(zip(teams, rng.normal(0, spread * 0.8, 20)))
        start = pd.Timestamp(f"{2010 + season}-08-10")
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        for i, (home, away) in enumerate(fixtures):
            lam = np.exp(attack[home] + defence[away] + 0.25)
            mu = np.exp(attack[away] + defence[home])
            matrix = dc.score_matrix(lam, mu, rho, max_goals=10)
            hg, ag = np.unravel_index(rng.choice(matrix.size, p=matrix.ravel()
                                                 / matrix.sum()), matrix.shape)
            rows.append({
                "MatchID": f"S{season}_{i}", "Season": f"S{season}",
                "MatchDate": start + pd.Timedelta(days=i // 10),
                "HomeTeam": home, "AwayTeam": away,
                "HomeGoals": int(hg), "AwayGoals": int(ag),
                "Result": "H" if hg > ag else "A" if ag > hg else "D",
            })
    return pd.DataFrame(rows)


# --- The null decides the answer --------------------------------------------

def test_per_match_null_finds_nothing_where_there_is_nothing(config):
    """Independent Poisson per match (rho = 0), very unequal teams: the
    per-match null's interval must contain zero for all four low scores."""
    table = simulate_seasons(rho=0.0, seed=1)
    departures, _ = ex.low_score_departures(table, config)
    for row in departures.to_dict("records"):
        assert row["departure_low"] <= 0 <= row["departure_high"], row


def test_league_mean_null_mistakes_team_mixing_for_a_low_score_effect(config):
    """Same data, crude null. A mixture of Poissons with different means has
    more 0-0s than one Poisson at the average mean - so the league-mean null
    reports an 'effect' that is only team heterogeneity, and reports it far
    larger than the per-match null does."""
    table = simulate_seasons(rho=0.0, seed=1)
    departures, _ = ex.low_score_departures(table, config)
    nil_nil = departures.set_index("score").loc["0-0"]
    assert nil_nil["departure_league_mean"] > 0.10
    assert abs(nil_nil["departure_league_mean"]) > 3 * abs(nil_nil["departure_per_match"])


def test_a_real_dixon_coles_effect_is_found_with_the_right_signs(config):
    """With rho = -0.15 the per-match null must see 0-0 and 1-1 MORE often
    than Poisson and 1-0 and 0-1 LESS often - the direction tau actually
    moves them (not 'all four up', which the notebook used to say)."""
    table = simulate_seasons(rho=-0.15, seed=2)
    departures, rhos = ex.low_score_departures(table, config)
    d = departures.set_index("score")["departure_per_match"]
    assert d["0-0"] > 0 and d["1-1"] > 0
    assert d["1-0"] < 0 and d["0-1"] < 0
    assert all(row["rho"] < 0 for row in rhos)


def test_findings_include_a_sign_test_on_seasonal_rho(config):
    table = simulate_seasons(n_seasons=6, rho=-0.15, seed=3)
    table["OddsH"], table["OddsD"], table["OddsA"] = 2.0, 3.4, 3.8
    table["CloseH"], table["CloseD"], table["CloseA"] = 2.05, 3.4, 3.7
    for col in ("HomeShotsOT", "AwayShotsOT", "HomeShots", "AwayShots"):
        table[col] = table["HomeGoals"] + 2
    found = ex.findings(table, config)
    assert found["season_rho_negative"] == 6
    assert found["season_rho_sign_test_p"] == pytest.approx(0.5 ** 6)


# --- The small statistics ------------------------------------------------------

def test_wilson_interval_matches_a_known_value():
    """45 successes in 100: Wilson gives [0.3561, 0.5475] (textbook value)."""
    low, high = ex.wilson_interval(45, 100)
    assert low == pytest.approx(0.3561, abs=1e-4)
    assert high == pytest.approx(0.5475, abs=1e-4)


def test_correlation_interval_brackets_r_and_narrows_with_n():
    low_small, high_small = ex.correlation_interval(0.55, 100)
    low_big, high_big = ex.correlation_interval(0.55, 5000)
    assert low_small < 0.55 < high_small
    assert (high_big - low_big) < (high_small - low_small)


def test_overround_of_a_known_book():
    """Prices 2.00 / 3.40 / 3.80 imply 0.5 + 0.2941 + 0.2632 = 1.0573."""
    table = pd.DataFrame({"OddsH": [2.0], "OddsD": [3.4], "OddsA": [3.8]})
    assert ex.overround(table, ["OddsH", "OddsD", "OddsA"])["mean"] == pytest.approx(
        0.0573, abs=1e-4)


def test_a_calibrated_market_scores_a_small_ece():
    """Closing prices built from the true probabilities, results drawn from
    them: the market is calibrated by construction and its ECE is small."""
    rng = np.random.default_rng(4)
    # 60,000 matches: ECE has a noise floor (~0.011 at 6,000, ~0.003 here),
    # and a threshold set inside the noise fails on an unlucky seed. It did.
    n = 60000
    p_home = rng.uniform(0.15, 0.75, n)
    p_draw = (1 - p_home) * 0.45
    p_away = 1 - p_home - p_draw
    draws = rng.random(n)
    result = np.where(draws < p_home, "H", np.where(draws < p_home + p_draw, "D", "A"))
    table = pd.DataFrame({
        "MatchDate": pd.date_range("2015-01-01", periods=n, freq="h"),
        "CloseH": 1 / (p_home * 1.025), "CloseD": 1 / (p_draw * 1.025),
        "CloseA": 1 / (p_away * 1.025), "Result": result})
    whole = ex.market_calibration(table, "proportional")
    later = ex.market_calibration(table, "proportional", fit_fraction=0.5)
    assert whole["ece"] < 0.006
    assert not whole["detectably_miscalibrated"]
    assert later["matches"] == n - n // 2


def test_the_noise_floor_is_what_small_samples_score_when_perfect():
    """A perfectly calibrated forecaster on 380 matches scores around 0.05 -
    the trap the floor exists for. On 38,000 it scores a tenth of that."""
    rng = np.random.default_rng(5)
    small = ex.ece_noise_floor(rng.uniform(0.15, 0.75, 380), n_sims=300)
    large = ex.ece_noise_floor(rng.uniform(0.15, 0.75, 38000), n_sims=100)
    assert 0.025 < small["median"] < 0.08
    assert large["median"] < small["median"] / 5


def test_a_miscalibrated_market_is_detected_above_its_floor():
    """Prices that overstate the home win by 8 points on 5,000 matches."""
    rng = np.random.default_rng(6)
    n = 5000
    truth = rng.uniform(0.2, 0.6, n)
    stated = np.clip(truth + 0.08, 0.05, 0.9)
    draw_share = (1 - stated) * 0.45
    won = rng.random(n) < truth
    table = pd.DataFrame({
        "MatchDate": pd.date_range("2015-01-01", periods=n, freq="h"),
        "CloseH": 1 / stated, "CloseD": 1 / draw_share,
        "CloseA": 1 / (1 - stated - draw_share),
        "Result": np.where(won, "H", "A")})
    assert ex.market_calibration(table, "proportional")["detectably_miscalibrated"]


# --- The script --------------------------------------------------------------

@pytest.fixture(scope="module")
def explore_script():
    spec = importlib.util.spec_from_file_location(
        "explore", REPO_ROOT / "scripts" / "explore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_folders_are_recognised(explore_script, tmp_path):
    assert explore_script.is_published(REPO_ROOT / "docs" / "charts")
    assert explore_script.is_published(REPO_ROOT / "docs" / "charts" / "epl_2012_2025")
    assert not explore_script.is_published(tmp_path)


def test_sample_data_cannot_reach_a_published_folder(explore_script, monkeypatch,
                                                     tmp_path, capsys):
    """Seasons that were never built force the sample; publishing must refuse."""
    overlay = tmp_path / "no_data.yaml"
    overlay.write_text(f"extends: {REPO_ROOT / 'config.yaml'}\n"
                       "data:\n  seasons: ['9091']\n")
    monkeypatch.setattr(sys, "argv", ["explore.py", "--config", str(overlay),
                                      "--out", str(REPO_ROOT / "docs" / "charts")])
    assert explore_script.main() == 1
    assert "REFUSED" in capsys.readouterr().out


def test_the_notebook_runner_refuses_without_real_data(monkeypatch, capsys):
    """scripts/run_notebook.py must not execute, let alone save, a notebook on
    the bundled sample - the fault it was added to stop. [29 Oct 2026]"""
    spec = importlib.util.spec_from_file_location(
        "run_notebook", REPO_ROOT / "scripts" / "run_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    before = module.NOTEBOOK.read_bytes()
    monkeypatch.setattr(module.ingest, "expected_raw_files",
                        lambda config: [{"path": "data/raw/E0_9091.csv", "exists": False}])
    monkeypatch.setattr(sys, "argv", ["run_notebook.py"])
    assert module.main() == 1
    assert "REFUSED" in capsys.readouterr().out
    assert module.NOTEBOOK.read_bytes() == before
