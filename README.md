# ⚽ V8 — Quantitative Football Model

**A football match-prediction and staking system built to quant-validation standards: Dixon-Coles goal modelling, out-of-sample probability calibration, fractional-Kelly staking, walk-forward backtesting with lookahead guards, and closing-line value as the honest test of edge.**

![tests](https://github.com/esiihle/v8-quant-football-model/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![tests count](https://img.shields.io/badge/tests-178-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

---

## The headline result

**The model does not beat the closing line, and this README says so.**

| | In-sample | Walk-forward |
|---|---|---|
| ROI | +311% | **−28.5%** |
| Yield | +22.9% | **−4.2%**  [−17.9%, +10.6%] |
| Bets | — | 413 |

Same model, same settings, same data. The only difference is whether the model
was allowed to see the matches it was betting on. A 340-point swing in ROI is
what a lookahead-free test boundary costs an impressive-looking backtest — and
it is the single most useful number in this repository.

![in-sample vs walk-forward](docs/charts/in-sample-vs-walk-forward.png)

The walk-forward interval includes zero, so even the *loss* is not statistically
established on 413 bets. That is precisely the situation closing-line value
exists for: with too few bets to settle the question from profit alone, CLV asks
whether the model is finding prices the market later agrees were wrong.

### What CLV says

![CLV distribution](docs/charts/clv-distribution.png)

| Measure | Result | Verdict |
|---|---|---|
| CLV, probability points *(authoritative)* | **−0.67%**  [−0.87%, −0.47%] | negative, interval excludes zero |
| CLV, odds ratio | +1.82%  [+0.81%, +3.00%] | positive — and misleading, see below |
| Beat the close | 47.46% | below the 50% coin flip |

The two measures **disagree, and both intervals exclude zero**. The odds ratio is
asymmetric — a halved price scores −50% while a doubled one scores +100% — so
its mean was carried by a single +108% price collapse while the model was losing
to the close on most bets. Probability points is the authoritative measure, the
disagreement is reported rather than resolved in the flattering direction, and
`test_disagreeing_clv_measures_are_flagged_not_papered_over` makes sure it stays
that way.

The sharpest diagnostic is whether the model's *own* claimed edge predicts CLV.
If it had real information, the bets where it claimed most edge would show the
most CLV:

![CLV by claimed edge](docs/charts/clv-by-claimed-edge.png)

A flat, negative profile means the edge estimate carries no signal — the model
cannot tell you where it is right, which is a more damning finding than a
negative average.

### Calibration and bankroll

![reliability diagram](docs/charts/reliability-diagram.png)
![bankroll curve](docs/charts/bankroll-curve.png)

Calibration is the component that *does* work: Platt scaling, fitted on the
earlier half of the data and judged on the later, keeps predicted and realised
frequencies close across the range. A model can be well calibrated and still
unprofitable — calibration says the probabilities are honest, not that they are
better than the market's.

---

## The problem

Bookmaker odds are an efficient estimate of match probabilities, and beating
them is genuinely hard. Most "profitable" betting models are overfit backtests
that quietly use information unavailable at bet time. V8 is built to do the
opposite: model match outcomes rigorously, size bets sensibly, and **test any
claimed edge the way a model-validation function would** — out of time, with
confidence intervals, against a benchmark that cannot be gamed.

## The method

```
ingest → features → Dixon-Coles → calibration → staking → walk-forward → CLV
```

1. **Ingest & validate** — results and odds against a strict schema; every
   validation rule lives in `config.yaml`, not in code.
2. **Features** — attack/defence strengths with **Bayesian shrinkage**
   (`w = n/(n+k)`), **exponential time decay** (180-day half-life), and an
   expected-goals proxy from shots on target.
3. **Dixon-Coles** — bivariate Poisson with the τ low-score correction, fitted
   by penalised, time-weighted maximum likelihood with multi-start.
4. **Calibration** — Platt vs isotonic, both fitted on earlier data and judged
   on later, scored with Brier, log-loss and expected calibration error.
5. **Staking** — value measured against the **raw price** (not the de-vigged
   one), sized with **fractional Kelly** at 25%, capped per bet and per day.
6. **Walk-forward** — refit every 14 days, with a strict same-day boundary and a
   lookahead guard that runs on **every window of every run** and raises.
7. **CLV** — closing-line value in three expressions, each with a bootstrap
   interval, with the authoritative one named in advance.

Full derivations in [`docs/methodology.md`](docs/methodology.md).

## What this project shows — and what it doesn't

**It shows** that the pipeline is leakage-free, that the numbers are
reproducible, and that the honest out-of-sample result is a loss. Reporting that
rather than tuning until the backtest looks good **is the deliverable**.

**It does not show** a profitable betting strategy. It also does not test
multiple leagues, team news, or in-play markets, and 413 bets is a small sample
on which to settle anything — which is exactly why CLV is the headline rather
than profit.

## Bugs this project found, and what each one taught

Every one of these was invisible in the output. Each is now pinned by a
regression test whose docstring tells the story.

| Bug | Symptom | Fix |
|---|---|---|
| **Lookahead in the xG proxy** | Conversion rate fitted over the whole file, so August matches were priced using May's rate | `blend_goals` takes `history` as a **required** argument — structural, not a patch |
| **`+inf` objective** | `inf − inf = nan` in the finite-difference gradient; L-BFGS-B stopped improving **while reporting `success: True`** | Finite `INVALID_PENALTY` scaled by violation size |
| **ρ pinned at its bound** | Bias that did *not* shrink with sample size (−0.012 at 240 matches, −0.018 at 2,400) | Bounds sized from observed goals; `rho_at_bound` diagnostic |
| **Value tested against the wrong price** | EV computed vs the de-vigged probability, but the payout is at the actual odds | `edge` vs raw price decides bets; `disagreement` vs fair is diagnostic only |
| **CLV verdict inverted** | Report said the interval "includes zero" when both excluded it, in opposite directions | Probability points made authoritative; disagreement flagged explicitly |
| **A regex that read prose as code** | Dependency check failed on a package called `average`, picked out of a comment | Imports found with `ast`, not a regex |

## Reproducing it

```bash
git clone https://github.com/esiihle/v8-quant-football-model.git
cd v8-quant-football-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # 178 tests (7 skip until
                                         # the charts are generated)
python scripts/run_pipeline.py --sample  # runs on the bundled 12-match sample
```

For the real numbers, download the two season files the pipeline names on
startup into `data/raw/`, then:

```bash
python scripts/run_pipeline.py           # the full pipeline
python scripts/make_charts.py            # regenerates every chart above
pytest tests/test_docs.py                # checks this README against that run
```

Every run prints a **fingerprint** — a hash of the result-affecting config:

```
run fingerprint: c6c0bb194dcd9ba2   (same fingerprint => same numbers)
```

If a rerun's numbers differ while the fingerprint matches, the code is
non-deterministic and that is a bug. If the fingerprints differ, the settings
differed and the comparison was never valid. `docs/charts/numbers.json` records
both the figures and the fingerprint of the run that produced them, and
`tests/test_docs.py` checks the tables above against it.

## Why this project

Every component is a transferable quant/credit-risk technique:

| This repo | The real-world skill |
|---|---|
| Dixon-Coles MLE | Parametric modelling of count data |
| Bayesian shrinkage | Regularisation for thin samples / thin-file borrowers |
| Probability calibration | PD calibration in credit risk |
| Fractional Kelly | Position sizing under uncertainty |
| Closing-line value | Performance vs a benchmark that cannot be gamed |
| Walk-forward + lookahead guards | Out-of-time model validation |
| Bootstrap intervals on every metric | Knowing when a result is not established |

## Repository structure

```
src/            ingest, features, dixon_coles, calibrate, staking, backtest, clv, metrics
tests/          178 tests; the fragile numerics and every bug above
scripts/        run_pipeline.py (end to end) · make_charts.py (every chart here)
docs/           methodology.md · OVERVIEW.md · ROADMAP.md · charts/
notebooks/      exploration
.github/        CI: pytest + a full pipeline run, on Python 3.11 and 3.12
```

## Author

**Sihle Myeza** — BSc Computer Science (Wits), data analyst targeting
quantitative and credit-risk roles. Built in the open, one commit at a time.

## License

MIT — see [`LICENSE`](LICENSE).
