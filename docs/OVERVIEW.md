# Project Overview — V8 Quantitative Football Model

> **Read this first.** This document is the orientation for anyone picking up the
> project — a future collaborator, a recruiter, or myself in six months. It
> explains *what* we are building, *why* it is built this way, and *how* the
> pieces fit together. The code-level detail lives in `methodology.md`; the
> schedule lives in `ROADMAP.md`.

---

## 1. What this project is

V8 is a **quantitative football match-prediction and staking system**. Given
historical results and bookmaker odds, it does three things end to end:

1. **Predicts** the probability of every match outcome (home / draw / away, and
   derived markets such as over/under and correct score) using a statistical
   model of goal-scoring.
2. **Decides** whether — and how much — to stake, by comparing the model's
   probabilities against the market's implied probabilities and sizing positions
   with the Kelly criterion.
3. **Proves** whether the edge is real, using out-of-sample, out-of-time
   backtesting and, above all, **closing-line value (CLV)** tracking.

It is deliberately built as a small **production-style Python system** (~10
source modules, config-driven, tested, reproducible) rather than a single
notebook. The point is not just to predict football — it is to demonstrate the
full lifecycle of a validated predictive model under uncertainty.

## 2. Why it exists (the thesis)

The project rests on one honest premise: **the betting market is an efficient
aggregator of information, and beating it is hard.** The closing odds on a match
are the single best public estimate of the true probabilities. Any system that
claims an edge must clear two bars:

- **Better probabilities** than the market on some identifiable subset of
  matches — not on average, on a subset we can find in advance.
- **Disciplined staking** that compounds that edge into bankroll growth without
  risking ruin.

Everything in V8 is designed to *detect and honestly prove* edge, not to produce
a flattering backtest. That discipline — separating genuine signal from
overfitting and hindsight — is the whole skill, and it is exactly the skill that
transfers to credit-risk and quantitative-finance modelling.

## 3. Why a recruiter should care (the transfer)

Every component maps directly onto a core quant/risk technique:

| V8 component | The general technique | Where it shows up in finance |
|---|---|---|
| Dixon-Coles goal model | Parametric MLE modelling of count data | Frequency models, claim/loss counts |
| Bayesian shrinkage | Regularisation toward a prior for thin samples | Thin-file / low-default-portfolio scoring |
| Probability calibration | Turning raw scores into trustworthy probabilities | PD calibration in credit risk |
| Kelly staking | Position sizing under uncertainty | Portfolio allocation, risk budgeting |
| Closing-line value | Measuring realised edge vs. consensus | Alpha vs. benchmark |
| Walk-forward backtest + lookahead guards | Out-of-time validation, no leakage | The core discipline of model validation |

The football is the wrapper. The content is a validated stochastic model with
honest performance measurement — which is the job.

## 4. How it works — the pipeline

The system is a linear pipeline of well-separated stages. Data flows in one
direction, each stage has a single responsibility, and every stage is
independently testable.

### Stage 1 — Data ingestion (`ingest.py`)
Load historical match results and odds, validate them against a strict schema
(no silent type coercion, no missing-column surprises), and store a single
canonical representation. **Rule: bad data fails loudly here, not silently three
stages downstream.**

### Stage 2 — Feature engineering (`features.py`)
Turn raw results into predictive signal:
- **Team attack/defence strengths**, estimated relative to league average.
- **Bayesian shrinkage** of those strengths toward the league mean — heavier
  shrinkage for teams with few matches (newly promoted sides, early season).
  This is the single most important guard against overreacting to small samples.
- **Expected-goals (xG) proxies** built from shot volume and shot quality where
  available — a more stable signal of underlying performance than actual goals.
- **Time decay**: recent matches weigh more than old ones (exponential
  weighting), so the model tracks form without ignoring history.
- **Home advantage** as an explicit parameter.

### Stage 3 — The model (`dixon_coles.py`)
The core is the **Dixon-Coles** model. It treats home and away goals as (nearly)
Poisson-distributed given the teams' attack/defence strengths and home
advantage, then applies the **low-score dependence correction** (the `rho`
parameter) that fixes the probabilities of 0-0, 1-0, 0-1 and 1-1 — the outcomes
where a naive independent-Poisson model is measurably wrong. Parameters are
fitted by **maximum likelihood** with the time-decay weighting from Stage 2.
From the fitted score matrix we derive every market probability.

### Stage 4 — Calibration (`calibrate.py`)
Raw model probabilities are close but not perfectly calibrated. This stage maps
them to calibrated probabilities (isotonic regression or Platt scaling) and
proves the result with **reliability diagrams** and proper scoring rules
(**Brier score**, **log-loss**). A prediction of "60%" should mean it happens
~60% of the time — this stage is what earns the right to say that.

### Stage 5 — Staking (`staking.py`)
Convert `(model probability, market odds)` into decisions:
- Flag **value bets** where model probability exceeds market-implied probability
  by a meaningful margin.
- Size each bet with **fractional Kelly** (a fraction of full Kelly, to blunt the
  cost of estimation error and reduce variance).
- **Simulate the bankroll** over the bet history to expose realistic drawdowns,
  not just end-point returns.

### Stage 6 — Backtest & evaluation (`backtest.py`, `clv.py`, `metrics.py`)
The honesty layer:
- **Walk-forward backtesting**: at each point in time the model sees only data
  that existed *before* the match. Models are refit on a rolling/expanding window.
- **Lookahead guards**: explicit checks that no future information leaks into any
  feature or decision. This is where most amateur betting models quietly cheat —
  ours is built so it can't.
- **Closing-line value (CLV)** — the headline metric. If the odds we take
  consistently beat the closing odds, we are systematically finding value the
  market only prices in later. Positive CLV is the strongest available leading
  indicator of a real, repeatable edge, precisely *because* the closing line is
  so efficient. This is our answer to "how do you know it isn't luck?"
- Supporting metrics: ROI, yield, hit rate by confidence band, Sharpe-like
  ratio, maximum drawdown.

## 5. Design principles

- **Reproducible** — fixed random seeds, pinned dependencies, a single
  config-driven entry point. The same command produces the same numbers.
- **Tested** — the numerically fragile parts (Dixon-Coles likelihood, the `rho`
  correction, Kelly fractions, calibration) have unit tests with known-answer
  cases.
- **Validated** — out-of-sample and out-of-time by construction; CLV as the
  final arbiter of edge.
- **Honest** — no lookahead, no cherry-picked windows, no reporting the one seed
  that looked good. If the edge is thin, the README will say so.
- **Legible** — simple, heavily commented code that covers multiple scenarios,
  because the goal is to be understood in an interview, not just to run.

## 6. Planned repository structure

```
v8-quant-football-model/
├── README.md
├── requirements.txt            # pinned dependencies
├── config.yaml                 # single source of run configuration
├── data/
│   ├── raw/                    # immutable inputs
│   └── processed/              # canonical, validated data
├── src/
│   ├── ingest.py               # load + schema validation
│   ├── features.py             # strengths, shrinkage, xG proxies, time decay
│   ├── dixon_coles.py          # bivariate Poisson + rho correction + MLE fit
│   ├── calibrate.py            # isotonic / Platt + reliability diagrams
│   ├── staking.py              # value detection + fractional Kelly + bankroll sim
│   ├── backtest.py             # walk-forward engine + lookahead guards
│   ├── clv.py                  # closing-line-value capture + metrics
│   └── metrics.py              # log-loss, Brier, ROI, yield, drawdown, Sharpe
├── tests/
│   ├── test_dixon_coles.py
│   ├── test_staking.py
│   └── test_features.py
├── notebooks/
│   └── 01_exploration.ipynb
├── scripts/
│   └── run_pipeline.py         # config-driven end-to-end run
└── docs/
    ├── OVERVIEW.md             # this file
    ├── ROADMAP.md              # dated build plan
    └── methodology.md          # the maths, in detail
```

## 7. What "done" looks like

The project is presentation-ready when:

- A single command runs the full pipeline reproducibly from raw data to a
  backtest report.
- The test suite is green and covers the fragile numerics.
- The README leads with a **CLV chart** and states the edge honestly, with
  confidence intervals, not point estimates alone.
- A short methodology write-up and a slide deck exist, so the project can be
  *talked about* as fluently as it can be run.

See `ROADMAP.md` for the week-by-week plan to get there.
