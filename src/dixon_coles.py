r"""
Stage 3 — The Dixon-Coles model.  [Built in Phase 3, 10 Oct 2026]

A bivariate Poisson goal model with the low-score dependence correction, fitted
by time-weighted, penalised maximum likelihood. This is the statistical core of
the project: everything downstream — calibration, staking, CLV — consumes the
probabilities this module produces.

The model
---------
Each team *i* carries an **attack** parameter :math:`\\alpha_i` and a **defence**
parameter :math:`\\beta_i`. There is one global **home advantage**
:math:`\\gamma` and one global dependence parameter :math:`\\rho`.

For a fixture (home *i*, away *j*) the expected goals are

.. math::
    \\lambda = \\exp(\\alpha_i + \\beta_j + \\gamma), \\qquad
    \\mu     = \\exp(\\alpha_j + \\beta_i)

and the joint probability of the scoreline (x, y) is

.. math::
    P(X = x, Y = y) = \\tau_\\rho(x, y)\;
        \\frac{\\lambda^x e^{-\\lambda}}{x!}\\,\\frac{\\mu^y e^{-\\mu}}{y!}

Why the correction exists
-------------------------
Independent Poisson gets football *almost* right, and is measurably wrong in one
specific place: the four low scorelines. Real matches produce more 0-0 and 1-1
draws, and fewer 1-0 and 0-1 results, than independence predicts — the
exploration notebook measures exactly that on our own data. Dixon and Coles
(1997) fix it with a single parameter:

.. math::
    \\tau_\\rho(x,y) = \\begin{cases}
        1 - \\lambda\\mu\\rho & x = 0, y = 0 \\\\
        1 + \\lambda\\rho     & x = 0, y = 1 \\\\
        1 + \\mu\\rho         & x = 1, y = 0 \\\\
        1 - \\rho             & x = 1, y = 1 \\\\
        1                     & \\text{otherwise}
    \\end{cases}

A negative :math:`\\rho` raises 0-0 and 1-1 while lowering 1-0 and 0-1, which is
the direction the data asks for. The correction is also **mass-preserving**: the
four adjustments cancel exactly, so the full distribution still sums to 1 (the
test ``test_tau_correction_preserves_total_probability`` checks it).

Three implementation decisions worth defending
----------------------------------------------
1. **Identifiability.** :math:`\\lambda` and :math:`\\mu` depend only on sums
   like :math:`\\alpha_i + \\beta_j`, so adding a constant to every attack and
   subtracting it from every defence changes nothing. One constraint fixes it:
   the attack parameters are forced to sum to zero. The optimiser therefore
   works with *n-1* free attack parameters and the last is implied.

2. **Penalised likelihood.** A team with three matches played can push the raw
   MLE to an absurd strength. An L2 penalty on the attack and defence parameters
   pulls them toward zero (= league average). That is Phase 2's Bayesian
   shrinkage expressed as regularisation — the same argument in the language of
   an optimiser, and the direct analogue of a ridge-penalised PD model.

3. **Phase 2's strengths seed the optimiser.** The shrunk strengths are a cheap,
   stable estimate, which makes them an excellent starting point for the MLE.
   The two phases are not rivals: Phase 2 describes, Phase 3 fits.

See docs/methodology.md section 3.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.features import time_weights

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keep probabilities away from exactly zero inside logarithms.
EPSILON = 1e-10

# The likelihood never returns infinity for an invalid parameter set. See the
# comment in `log_likelihood` — an infinite objective destroys L-BFGS-B's
# finite-difference gradient and the optimiser silently stops improving. A large
# finite value that grows with the size of the violation keeps the surface
# navigable right up to the boundary.
INVALID_PENALTY = 1e8
INVALID_FLOOR = 1e-9        # tau must stay at least this far above zero


class ConvergenceError(Exception):
    """Raised when the optimiser cannot produce a usable fit."""


# ===========================================================================
# The tau correction
# ===========================================================================

def tau(x, y, lam, mu, rho):
    """The Dixon-Coles low-score correction factor.

    Returns 1 everywhere except the four cells (0,0), (0,1), (1,0), (1,1).

    All arguments broadcast, so this works on scalars or arrays of matching
    shape — which is what lets the likelihood be evaluated on every match at
    once rather than in a Python loop.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)

    out = np.ones(np.broadcast(x, y, lam, mu).shape, dtype=float)

    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


def rho_bounds(lam, mu):
    """The range of rho that keeps every corrected probability non-negative.

    From the four tau expressions, rho must satisfy

        max(-1/lam, -1/mu)  <=  rho  <=  min(1/(lam*mu), 1)

    Outside that range the model assigns a negative probability to a scoreline,
    which is not a near-miss — it is nonsense.

    **The bound is sensitive to what you feed it, and that matters.** On
    10 Oct 2026 the optimiser was bounded using the largest lambda the parameter
    box could theoretically produce, exp(1.5 + 1.5 + 1.0) = 54.6 goals a match.
    That is not a football score, and it squeezed the rho box to
    [-0.018, 0.0003] — so rho sat pinned at its lower bound while the fit
    reported success and the starts agreed with each other. The recovery test
    caught it only because the bias did not shrink as the sample grew: noise
    averages out, a binding constraint does not.

    So `fit` derives the ceiling from the goals actually in the data (see
    `goal_ceiling`) and leaves the exact, per-fixture constraint to the smooth
    penalty inside `log_likelihood`. A box bound is a blunt instrument; the
    pointwise penalty is the real constraint.
    """
    lam = float(np.max(lam))
    mu = float(np.max(mu))
    lower = max(-1.0 / lam, -1.0 / mu)
    upper = min(1.0 / (lam * mu), 1.0)
    return lower, upper


def goal_ceiling(home_goals, away_goals, headroom=2.5, minimum=3.0):
    """A realistic upper bound on expected goals, taken from the data.

    Used only to size the rho box. The mean goals per side, times `headroom`,
    comfortably covers the strongest attack against the weakest defence while
    staying in the land of actual football — which is the whole point.
    """
    observed = max(float(np.mean(home_goals)), float(np.mean(away_goals)))
    return max(observed * headroom, minimum)


# ===========================================================================
# Parameter packing
# ===========================================================================

def _unpack(params, n_teams):
    """Split the flat parameter vector the optimiser works with.

    Layout: ``[attack_0 ... attack_{n-2}, defence_0 ... defence_{n-1}, gamma, rho]``

    The final attack parameter is not free — it is minus the sum of the others,
    which imposes the sum-to-zero constraint that makes the model identifiable.
    """
    n_free = n_teams - 1
    attack_free = params[:n_free]
    attack = np.append(attack_free, -attack_free.sum())
    defence = params[n_free:n_free + n_teams]
    gamma = params[-2]
    rho = params[-1]
    return attack, defence, gamma, rho


def _pack(attack, defence, gamma, rho):
    """Flatten parameters into the optimiser's vector, dropping the implied attack."""
    return np.concatenate([np.asarray(attack)[:-1], np.asarray(defence),
                           [gamma], [rho]])


# ===========================================================================
# The likelihood
# ===========================================================================

def expected_goals(attack, defence, gamma, home_idx, away_idx):
    """Expected goals for each fixture, given parameters and team indices."""
    lam = np.exp(attack[home_idx] + defence[away_idx] + gamma)
    mu = np.exp(attack[away_idx] + defence[home_idx])
    return lam, mu


def _log_poisson(k, rate):
    """log Poisson pmf, computed with gammaln for numerical stability.

    Using ``k * log(rate) - rate - log(k!)`` directly overflows for larger k;
    ``gammaln(k+1)`` is log(k!) without ever forming the factorial.
    """
    from scipy.special import gammaln
    rate = np.maximum(rate, EPSILON)
    return k * np.log(rate) - rate - gammaln(k + 1.0)


def log_likelihood(params, home_idx, away_idx, home_goals, away_goals,
                   weights, n_teams, penalty=0.0):
    """Time-weighted, penalised log-likelihood. Returns the NEGATIVE for minimising.

    Each match contributes ``weight * log P(score)``, so a match two seasons old
    counts a quarter as much as one played last week (see
    ``features.time_weights``).

    The penalty term ``penalty * sum(attack^2 + defence^2)`` is the shrinkage:
    with no evidence a team's parameters are pushed toward 0, which is exactly
    league average on this scale.

    Returns ``+inf`` for any parameter set that produces a non-positive
    probability, which keeps the optimiser out of invalid regions rather than
    letting it chase a NaN.
    """
    attack, defence, gamma, rho = _unpack(params, n_teams)
    lam, mu = expected_goals(attack, defence, gamma, home_idx, away_idx)

    correction = tau(home_goals, away_goals, lam, mu, rho)

    # A non-positive tau means this rho assigns a negative probability to an
    # observed scoreline. The point is invalid — but returning +inf here is a
    # mistake that cost us a working fit on 10 Oct 2026: L-BFGS-B estimates its
    # gradient by finite differences, so one infinite probe gives inf - finite
    # = inf (or inf - inf = nan), the gradient becomes unusable, and the
    # optimiser stops moving while still reporting success. Parameter recovery
    # came back at r = 0.21.
    #
    # Instead, return a large but FINITE value that grows with how badly the
    # constraint is violated. That gives a real downhill direction back into the
    # valid region, so the optimiser steers around the boundary instead of
    # dying on it.
    violation = float(np.sum(np.clip(INVALID_FLOOR - correction, 0.0, None)))
    if violation > 0:
        return INVALID_PENALTY * (1.0 + violation)

    log_prob = (np.log(correction)
                + _log_poisson(home_goals, lam)
                + _log_poisson(away_goals, mu))

    if not np.all(np.isfinite(log_prob)):
        return INVALID_PENALTY

    weighted = float(np.sum(weights * log_prob))
    regularisation = penalty * float(np.sum(attack ** 2) + np.sum(defence ** 2))

    return -(weighted - regularisation)


# ===========================================================================
# Fitting
# ===========================================================================

def fit(features, config=None, as_of=None, seed_strengths=None):
    """Fit the model on matches played before ``as_of``.

    Parameters
    ----------
    features : pandas.DataFrame
        The Phase 2 feature matrix (or the canonical table — anything carrying
        MatchDate, HomeTeam, AwayTeam, HomeGoals, AwayGoals).
    config : dict | None
        Reads ``features.half_life_days``, ``model.penalty``,
        ``model.n_starts``, ``model.max_iterations``.
    as_of : datetime | None
        Only matches strictly before this date are used. Defaults to just after
        the last match, i.e. fit on everything given. **Phase 6 passes a real
        date here, which is what keeps the walk-forward backtest honest.**
    seed_strengths : pandas.DataFrame | None
        Optional Phase 2 strengths (indexed by team, columns attack/defence) used
        as the first starting point. Speeds up and stabilises convergence.

    Returns
    -------
    dict
        The fitted model: team parameters, gamma, rho, convergence diagnostics
        and the settings used.
    """
    config = config or {}
    feature_cfg = config.get("features", {})
    model_cfg = config.get("model", {})

    half_life = float(feature_cfg.get("half_life_days", 180))
    penalty = float(model_cfg.get("penalty", 0.01))
    n_starts = int(model_cfg.get("n_starts", 3))
    max_iter = int(model_cfg.get("max_iterations", 500))

    data = features.copy()
    data["MatchDate"] = pd.to_datetime(data["MatchDate"])

    if as_of is None:
        as_of = data["MatchDate"].max() + pd.Timedelta(days=1)
    as_of = pd.Timestamp(as_of)

    weights = time_weights(data["MatchDate"], as_of, half_life)
    data = data.loc[weights > 0].copy()
    data["weight"] = weights[weights > 0]

    if len(data) == 0:
        raise ConvergenceError(
            f"No matches dated before {as_of:%Y-%m-%d} — nothing to fit on."
        )

    teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
    if len(teams) < 2:
        raise ConvergenceError("Need at least two teams to fit the model.")

    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_idx = data["HomeTeam"].map(index).to_numpy()
    away_idx = data["AwayTeam"].map(index).to_numpy()
    home_goals = data["HomeGoals"].to_numpy(dtype=float)
    away_goals = data["AwayGoals"].to_numpy(dtype=float)
    match_weights = data["weight"].to_numpy(dtype=float)

    # --- Starting points -------------------------------------------------
    starts = []

    # 1. Phase 2's shrunk strengths, converted to this model's log scale.
    if seed_strengths is not None and len(seed_strengths):
        attack0 = np.zeros(n_teams)
        defence0 = np.zeros(n_teams)
        for team, i in index.items():
            if team in seed_strengths.index:
                row = seed_strengths.loc[team]
                # Phase 2 strengths are multiplicative around 1.0; this model is
                # additive in logs, so log() is the natural bridge.
                attack0[i] = np.log(max(float(row["attack"]), 0.05))
                defence0[i] = np.log(max(float(row["defence"]), 0.05))
        attack0 -= attack0.mean()        # satisfy the sum-to-zero constraint
        starts.append(_pack(attack0, defence0, 0.25, -0.05))

    # 2. A flat start: every team average, modest home advantage.
    starts.append(_pack(np.zeros(n_teams), np.zeros(n_teams), 0.25, -0.05))

    # 3. Random perturbations, so "it converged" is a claim about the surface
    #    rather than about one lucky guess.
    rng = np.random.default_rng(config.get("seed", 42))
    while len(starts) < max(n_starts, 1):
        attack_r = rng.normal(0, 0.15, n_teams)
        attack_r -= attack_r.mean()
        starts.append(_pack(attack_r, rng.normal(0, 0.15, n_teams),
                            rng.uniform(0.1, 0.4), rng.uniform(-0.15, 0.0)))

    # --- Bounds ----------------------------------------------------------
    # Strength bounds are generous but finite: exp(1.5) ~ 4.5x league average is
    # already beyond anything real, and a bound stops a thin-sample team running
    # away to infinity.
    #
    # The rho box is sized from the goals actually observed, NOT from the most
    # extreme lambda the strength box could produce. See `rho_bounds` for why:
    # the theoretical ceiling implies ~55 goals a match and pins rho at its
    # bound. The exact per-fixture constraint is enforced by the smooth penalty
    # in `log_likelihood`, so this box only has to be sane, not airtight.
    ceiling = goal_ceiling(home_goals, away_goals)
    rho_low, rho_high = rho_bounds(ceiling, ceiling)
    bounds = ([(-1.5, 1.5)] * (n_teams - 1)      # free attacks
              + [(-1.5, 1.5)] * n_teams          # defences
              + [(-1.0, 1.0)]                    # gamma (home advantage)
              + [(rho_low, rho_high)])           # rho

    # --- Optimise from every start --------------------------------------
    results = []
    for start in starts[:max(n_starts, 1)]:
        outcome = minimize(
            log_likelihood, start,
            args=(home_idx, away_idx, home_goals, away_goals,
                  match_weights, n_teams, penalty),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max_iter},
        )
        if np.isfinite(outcome.fun):
            results.append(outcome)

    if not results:
        raise ConvergenceError(
            "No starting point produced a finite likelihood. Inspect the input "
            "data before relaxing anything here."
        )

    best = min(results, key=lambda r: r.fun)
    attack, defence, gamma, rho = _unpack(best.x, n_teams)

    # Agreement between starts is the real convergence evidence: if several
    # independent starts land on the same optimum, it is very likely the global
    # one. If they scatter, the surface is awkward and the fit is not trustworthy.
    objectives = np.array([r.fun for r in results])
    spread = float(objectives.max() - objectives.min())

    return {
        "teams": teams,
        "attack": dict(zip(teams, attack)),
        "defence": dict(zip(teams, defence)),
        "home_advantage": float(gamma),
        "rho": float(rho),
        "fitted_as_of": as_of.strftime("%Y-%m-%d"),
        "matches_used": int(len(data)),
        "effective_sample_size": float(match_weights.sum()),
        "settings": {
            "half_life_days": half_life,
            "penalty": penalty,
            "n_starts": len(results),
            "max_iterations": max_iter,
        },
        "diagnostics": {
            "negative_log_likelihood": float(best.fun),
            "rho_box": [float(rho_low), float(rho_high)],
            # If rho lands on an edge of its box, the box is the constraint, not
            # the data. That is exactly the bug of 10 Oct 2026, so the fit now
            # says so out loud instead of leaving it to be discovered.
            "rho_at_bound": bool(
                abs(rho - rho_low) < 1e-6 or abs(rho - rho_high) < 1e-6
            ),
            "goal_ceiling_used": float(ceiling),
            "starts_converged": int(sum(r.success for r in results)),
            "starts_tried": len(results),
            "objective_spread": spread,
            "agreed_across_starts": bool(spread < 1e-3),
            "optimiser_message": str(best.message),
        },
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ===========================================================================
# From parameters to probabilities
# ===========================================================================

def score_matrix(lam, mu, rho, max_goals=10):
    """The full joint distribution over scorelines, as a (max_goals+1)^2 array.

    ``matrix[x, y]`` is P(home scores x, away scores y). Truncating at
    ``max_goals`` loses a negligible tail, so the matrix is renormalised to sum
    to exactly 1 — otherwise every derived market probability would be very
    slightly low, and those errors would compound through staking.
    """
    goals = np.arange(max_goals + 1)
    home_pmf = np.exp(_log_poisson(goals, lam))
    away_pmf = np.exp(_log_poisson(goals, mu))

    matrix = np.outer(home_pmf, away_pmf)

    x_grid, y_grid = np.meshgrid(goals, goals, indexing="ij")
    matrix = matrix * tau(x_grid, y_grid, lam, mu, rho)

    # Guard: a bad rho can push a cell slightly negative. Clip, then renormalise.
    matrix = np.clip(matrix, 0.0, None)
    total = matrix.sum()
    if total <= 0:
        raise ValueError("Score matrix has no probability mass — check rho.")
    return matrix / total


def market_probabilities(matrix):
    """Derive every market we trade from the score matrix.

    Everything here is a sum over the relevant cells — which is the real payoff
    of modelling the full scoreline distribution rather than the result alone:
    one fit prices 1X2, over/under and both-teams-to-score together, and they are
    automatically consistent with each other.
    """
    size = matrix.shape[0]
    x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    total_goals = x_grid + y_grid
    return {
        "home_win": float(matrix[x_grid > y_grid].sum()),
        "draw": float(matrix[x_grid == y_grid].sum()),
        "away_win": float(matrix[x_grid < y_grid].sum()),
        "over_2_5": float(matrix[total_goals >= 3].sum()),
        "under_2_5": float(matrix[total_goals <= 2].sum()),
        "btts_yes": float(matrix[(x_grid >= 1) & (y_grid >= 1)].sum()),
        "btts_no": float(matrix[(x_grid == 0) | (y_grid == 0)].sum()),
        "most_likely_score": (
            int(np.unravel_index(matrix.argmax(), matrix.shape)[0]),
            int(np.unravel_index(matrix.argmax(), matrix.shape)[1]),
        ),
    }


def predict(model, fixtures, config=None):
    """Market probabilities for a set of fixtures.

    A fixture whose team was not in the training data gets the league-average
    parameters (0 on this log scale) rather than an error — a promoted side must
    be priceable on debut. The ``KnownTeams`` column records whether both teams
    were actually seen, so Phase 5 can decline to stake on a fixture the model
    does not really know.
    """
    config = config or {}
    max_goals = int(config.get("model", {}).get("max_goals", 10))

    attack = model["attack"]
    defence = model["defence"]
    gamma = model["home_advantage"]
    rho = model["rho"]

    rows = []
    for _, fixture in fixtures.iterrows():
        home, away = fixture["HomeTeam"], fixture["AwayTeam"]
        known = home in attack and away in attack

        lam = np.exp(attack.get(home, 0.0) + defence.get(away, 0.0) + gamma)
        mu = np.exp(attack.get(away, 0.0) + defence.get(home, 0.0))

        probabilities = market_probabilities(score_matrix(lam, mu, rho, max_goals))

        row = {
            "MatchID": fixture.get("MatchID"),
            "MatchDate": fixture.get("MatchDate"),
            "HomeTeam": home, "AwayTeam": away,
            "KnownTeams": known,
            "ExpectedHomeGoals": float(lam),
            "ExpectedAwayGoals": float(mu),
        }
        score = probabilities.pop("most_likely_score")
        row.update({k: v for k, v in probabilities.items()})
        row["MostLikelyScore"] = f"{score[0]}-{score[1]}"
        rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
# Reporting and persistence
# ===========================================================================

def fit_report(model):
    """Readable summary of a fit, with the diagnostics that decide trust."""
    diagnostics = model["diagnostics"]
    lines = ["Dixon-Coles fit", "-" * 15]
    lines.append(f"  {'matches used':<32} {model['matches_used']:>9,}")
    lines.append(f"  {'effective sample size':<32} {model['effective_sample_size']:>9.1f}")
    lines.append(f"  {'fitted as of':<32} {model['fitted_as_of']:>9}")
    lines.append(f"  {'home advantage (gamma)':<32} {model['home_advantage']:>9.4f}")
    lines.append(f"  {'  -> home scoring multiplier':<32} "
                 f"{np.exp(model['home_advantage']):>9.3f}x")
    lines.append(f"  {'low-score dependence (rho)':<32} {model['rho']:>9.4f}")
    lines.append(f"  {'negative log-likelihood':<32} "
                 f"{diagnostics['negative_log_likelihood']:>9.2f}")
    lines.append(f"  {'starts converged':<32} "
                 f"{diagnostics['starts_converged']}/{diagnostics['starts_tried']}")
    lines.append(f"  {'objective spread across starts':<32} "
                 f"{diagnostics['objective_spread']:>9.2e}")

    if not diagnostics["agreed_across_starts"]:
        lines.append("  WARNING: starts disagree — the optimum may be local. "
                     "Do not trust this fit without investigating.")

    if diagnostics.get("rho_at_bound"):
        lines.append(f"  WARNING: rho sits on the edge of its allowed box "
                     f"{diagnostics['rho_box']}. The bound is driving the "
                     f"estimate, not the data — widen it and refit.")

    strengths = pd.DataFrame({
        "attack": pd.Series(model["attack"]),
        "defence": pd.Series(model["defence"]),
    })
    # exp() turns the log-scale parameters into multipliers, which is how a
    # human reads them: 1.30 attack = scores 30% more than league average.
    strengths["attack_x"] = np.exp(strengths["attack"])
    strengths["defence_x"] = np.exp(strengths["defence"])
    top = strengths.sort_values("attack", ascending=False)

    lines.append("")
    lines.append("  Strongest attacks (multiplier vs league average):")
    for team, row in top.head(5).iterrows():
        lines.append(f"    {str(team)[:22]:<24} {row['attack_x']:.3f}x")
    lines.append("  Weakest attacks:")
    for team, row in top.tail(3).iterrows():
        lines.append(f"    {str(team)[:22]:<24} {row['attack_x']:.3f}x")

    return "\n".join(lines)


def save_model(model, config, filename=None):
    """Write the fitted parameters to data/processed/ as JSON."""
    out_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / (filename or config.get("model", {}).get(
        "model_filename", "dixon_coles.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, default=float)
    return path


def load_model(path):
    """Read fitted parameters back — used by Phase 6 and by the tests."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
