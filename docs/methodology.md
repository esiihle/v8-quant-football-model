# Methodology — V8 Quantitative Football Model

> **Living document.** This is the mathematical spine of the project. It is
> scaffolded now and filled in as each phase lands (see `ROADMAP.md`). Each
> section carries a **Status** tag so the reader always knows what is settled and
> what is pending. Nothing here is reported as a *result* until it comes from a
> leakage-free walk-forward backtest.

---

## 1. Notation & data
**Status: 🟡 skeleton — finalised in Phase 1.**

- Matches indexed by $m$; home/away goals $x_m, y_m$; match date $t_m$.
- Teams indexed by $i$; each team has an **attack** parameter $\alpha_i$ and a
  **defence** parameter $\beta_i$. Global **home advantage** $\gamma$.
- Odds: decimal odds $o$ for each outcome; market-implied probability
  $\tilde{p} = 1/o$ (before de-vig); closing odds captured separately for CLV.

To be completed: exact schema, leagues/seasons covered, data sources, de-vig
method (proportional vs. Shin).

## 2. Feature engineering
**Status: 🟡 skeleton — built in Phase 2.**

### 2.1 Attack / defence strengths
Team strengths estimated relative to league average, so that expected goals for a
fixture combine home attack, away defence, and home advantage.

### 2.2 Bayesian shrinkage
Raw strengths for teams with few matches are unstable. We shrink each estimate
toward the league mean:

$$\hat{\theta}_i^{\text{shrunk}} = w_i\,\hat{\theta}_i + (1 - w_i)\,\bar{\theta},
\qquad w_i = \frac{n_i}{n_i + k}$$

where $n_i$ is the team's match count and $k$ is a shrinkage strength chosen by
cross-validation. Small $n_i$ ⇒ small $w_i$ ⇒ heavy pull toward the prior. This
is the direct analogue of thin-file / low-default-portfolio treatment in credit.

### 2.3 Expected-goals (xG) proxies
A more stable performance signal than realised goals, approximated from shot
volume and shot quality where the data allows. To be specified in Phase 2.

### 2.4 Time decay
Recent matches weigh more via exponential decay with half-life parameter $\xi$:
$\;\phi(t) = \exp(-\xi\,\Delta t)$. Enters the likelihood in §3.3.

## 3. The Dixon-Coles model
**Status: 🟡 skeleton — built and tested in Phase 3.**

### 3.1 Independent-Poisson baseline
Expected goals for a fixture (home team $i$, away team $j$):

$$\lambda = \exp(\alpha_i + \beta_j + \gamma), \qquad
  \mu = \exp(\alpha_j + \beta_i)$$

with $x \sim \text{Poisson}(\lambda)$, $y \sim \text{Poisson}(\mu)$ as the
starting point.

### 3.2 The low-score dependence correction
Independent Poisson misprices the low-score outcomes (0-0, 1-0, 0-1, 1-1).
Dixon-Coles multiply the joint mass by a correction $\tau_{\rho}(x, y)$ governed
by a single dependence parameter $\rho$:

$$P(X = x, Y = y) = \tau_{\rho}(x, y)\;
  \frac{\lambda^{x} e^{-\lambda}}{x!}\,\frac{\mu^{y} e^{-\mu}}{y!}$$

where $\tau_{\rho}$ adjusts only the four low-score cells and equals 1 elsewhere.
Exact form to be documented alongside the implementation.

### 3.3 Time-weighted maximum likelihood
Parameters $\{\alpha_i, \beta_i, \gamma, \rho\}$ are fitted by maximising the
time-weighted log-likelihood:

$$\mathcal{L} = \sum_{m} \phi(t_m)\,\log P(X = x_m, Y = y_m)$$

with a sum-to-zero (or mean-zero) identifiability constraint on the strengths.
Optimiser, constraints, and multi-start strategy documented in Phase 3 — this is
the section most likely to need care for numerical stability.

### 3.4 From parameters to market probabilities
The fitted model gives a full score matrix $P(X = x, Y = y)$, from which every
market (1X2, over/under, correct score, etc.) is derived by summing the relevant
cells. Details in Phase 3.

## 4. Calibration
**Status: 🟡 skeleton — built in Phase 4.**

Raw model probabilities are mapped to calibrated ones via isotonic regression or
Platt scaling, and assessed with proper scoring rules:

$$\text{Brier} = \frac{1}{N}\sum (p_i - o_i)^2, \qquad
  \text{LogLoss} = -\frac{1}{N}\sum \big[o_i \log p_i + (1 - o_i)\log(1 - p_i)\big]$$

plus reliability diagrams (predicted vs. realised frequency). Method choice
justified out-of-sample in Phase 4.

## 5. Staking
**Status: 🟡 skeleton — built in Phase 5.**

### 5.1 Value definition
A bet is flagged as value when model probability exceeds market-implied
probability by a margin threshold (set to control false positives).

### 5.2 Kelly criterion
For a single bet at decimal odds $o$ (net odds $b = o - 1$), win probability $p$,
$q = 1 - p$, the growth-optimal fraction is:

$$f^{\*} = \frac{bp - q}{b}$$

We stake **fractional Kelly** ($c \cdot f^{\*}$, e.g. $c = 0.25$) to blunt the cost
of estimation error and reduce variance. Multi-outcome generalisation documented
in Phase 5.

### 5.3 Bankroll simulation
The staking plan is simulated over the bet history to expose realistic drawdowns
and variance, not just terminal return.

## 6. Backtesting & evaluation
**Status: 🟡 skeleton — built in Phase 6.**

### 6.1 Walk-forward protocol
At each match, the model is fitted only on data available *before* $t_m$, on a
rolling/expanding window. Refit cadence documented in Phase 6.

### 6.2 Lookahead guards
Explicit assertions that no feature or decision uses information dated at or after
the match. This is the integrity backbone of the whole project.

### 6.3 Closing-line value (CLV) — the headline
CLV compares the odds taken against the closing odds. Consistent positive CLV is
the strongest available evidence of a real, repeatable edge, precisely because the
closing line is the market's most efficient estimate. Exact CLV metric (e.g.
mean beat vs. closing implied probability) fixed in Phase 6.

### 6.4 Supporting metrics
ROI, yield, hit rate by confidence band, a Sharpe-like ratio, and maximum
drawdown — all reported with confidence intervals, never as bare point estimates.

## References
- Dixon, M. J. & Coles, S. G. (1997). *Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market.* Applied Statistics, 46(2).
- Kelly, J. L. (1956). *A New Interpretation of Information Rate.*
- Additional references added as the methodology is written up.
