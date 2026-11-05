"""
Tests for src/benchmark.py: the model against the market on the same matches.
[added 5 Nov 2026]

Every test builds forecasts whose quality is known by construction, so the
comparison can be checked against the truth rather than against itself.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src import backtest, benchmark

sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_backtest import make_history          # the simulated league


@pytest.fixture(scope="module")
def config():
    return load_config()


def simulated_matches(n=4000, model_noise=0.0, market_noise=0.0, seed=0):
    """Matches with TRUE probabilities, and two forecasters that add noise to
    them. Noise 0 is a perfect forecaster; more noise is a worse one. The
    market is priced with a proportional 2.5% margin (de-vigged by the test's
    config, so the margin is not held against it)."""
    rng = np.random.default_rng(seed)
    home = rng.uniform(0.2, 0.65, n)
    draw = (1 - home) * rng.uniform(0.38, 0.5, n)
    truth = np.column_stack([home, draw, 1 - home - draw])

    def noisy(scale):
        logits = np.log(truth) + rng.normal(0, scale, truth.shape)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    model, market = noisy(model_noise), noisy(market_noise)
    u = rng.random(n)
    result = np.where(u < truth[:, 0], "H",
                      np.where(u < truth[:, 0] + truth[:, 1], "D", "A"))
    frame = pd.DataFrame(model, columns=benchmark.PROBABILITY_COLUMNS)
    for prefix in ("Odds", "Close"):
        for i, side in enumerate("HDA"):
            frame[f"{prefix}{side}"] = 1 / (market[:, i] * 1.025)
    frame["Result"] = result
    return frame


@pytest.fixture(scope="module")
def proportional(config):
    return {**config, "staking": {**config.get("staking", {}),
                                  "devig_method": "proportional"}}


# --- The scoring itself ---------------------------------------------------------

def test_per_match_scores_known_values():
    """Probabilities 0.5 / 0.3 / 0.2, home win: log-loss -ln(0.5) = 0.6931,
    Brier (0.5)^2 + 0.3^2 + 0.2^2 = 0.38."""
    scores = benchmark.per_match_scores([[0.5, 0.3, 0.2]], ["H"])
    assert scores["log_loss"][0] == pytest.approx(0.693147, abs=1e-6)
    assert scores["brier"][0] == pytest.approx(0.38)


def test_identical_forecasts_differ_by_exactly_zero():
    same = np.random.default_rng(1).random(500)
    interval = benchmark.paired_difference(same, same, n_resamples=200)
    assert interval["estimate"] == interval["lower"] == interval["upper"] == 0.0
    assert benchmark.verdict(interval) == "no detectable difference"


# --- The verdicts follow the truth --------------------------------------------

def test_a_perfect_model_beats_a_noisy_market(proportional):
    summary = benchmark.compare(simulated_matches(model_noise=0.0, market_noise=0.4,
                                                  seed=2), proportional, n_resamples=500)
    for key, d in summary["differences"].items():
        assert d["verdict"] == "model better", key


def test_a_noisy_model_loses_to_a_perfect_market(proportional):
    summary = benchmark.compare(simulated_matches(model_noise=0.4, market_noise=0.0,
                                                  seed=3), proportional, n_resamples=500)
    for key, d in summary["differences"].items():
        assert d["verdict"] == "market better", key


def test_equal_forecasters_are_rarely_told_apart(proportional):
    """Same truth, same noise, independent errors: the null is true.

    A 95% interval excludes zero 5% of the time when the null is true. That
    is its definition, not a bug. The first version of this test used one
    seed, and that seed happened to be one of the 5%: "model better" for two
    equal forecasters. Across 20 seeds the mean difference was +0.0002 and
    the false-positive count 0, so the code was fine and the test was wrong.

    So this tests the RATE. With 20 independent nulls and a true rate of 5%,
    three or more false positives has probability about 7.5%, and the seeds
    are fixed, so the outcome is deterministic: 0 of 20.
    """
    false_positives = 0
    for seed in range(100, 120):
        summary = benchmark.compare(
            simulated_matches(model_noise=0.15, market_noise=0.15, seed=seed),
            proportional, n_resamples=300)
        verdict = summary["differences"]["log_loss_vs_closing"]["verdict"]
        false_positives += verdict != "no detectable difference"
    assert false_positives <= 2


def test_the_margin_is_not_held_against_the_market(proportional):
    """Market = truth with a 2.5% margin, model = truth: after de-vigging they
    are the same forecast, so the difference is zero to rounding."""
    summary = benchmark.compare(simulated_matches(seed=5), proportional, n_resamples=200)
    for d in summary["differences"].values():
        assert abs(d["estimate"]) < 1e-9


def test_pairing_is_what_makes_the_comparison_precise():
    """Why a PAIRED test. Log-loss varies mostly between matches (an upset is
    costly for every forecaster), so two forecasters' scores are strongly
    correlated. The interval for the paired difference is far narrower than
    an interval built from the two averages as if they were independent."""
    frame = simulated_matches(model_noise=0.1, market_noise=0.1, seed=6)
    p_model = frame[benchmark.PROBABILITY_COLUMNS].to_numpy()
    p_market = 1 / frame[["CloseH", "CloseD", "CloseA"]].to_numpy()
    p_market = p_market / p_market.sum(axis=1, keepdims=True)
    a = benchmark.per_match_scores(p_model, frame["Result"])["log_loss"]
    b = benchmark.per_match_scores(p_market, frame["Result"])["log_loss"]

    paired = benchmark.paired_difference(a, b, n_resamples=800)
    paired_width = paired["upper"] - paired["lower"]
    unpaired_se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    assert paired_width < 0.5 * (2 * 1.96 * unpaired_se)


# --- The walk-forward now keeps what the comparison needs ---------------------

def test_walk_forward_keeps_every_priced_match(config):
    """Every priced match is kept, once, with probabilities that sum to one,
    and every bet's match is among them."""
    result = backtest.walk_forward(make_history(120, seed=12, model_edge=0.05), config)
    priced = result["priced"]
    assert len(priced) == result["counts"]["matches_priced"]
    assert priced["MatchID"].is_unique
    sums = priced[benchmark.PROBABILITY_COLUMNS].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)
    if not result["bets"].empty:
        assert set(result["bets"]["MatchID"]) <= set(priced["MatchID"])


def test_compare_runs_on_a_real_walk_forward(config):
    priced = backtest.walk_forward(make_history(120, seed=13), config)["priced"]
    summary = benchmark.compare(priced, config, n_resamples=200)
    assert summary["matches"] > 0
    assert set(summary["forecasters"]) == {"model", "opening", "closing"}
    assert "Model vs market" in benchmark.benchmark_report(summary)
