# Roadmap — V8 Quantitative Football Model

> **This is the project timeline: from inception to presentation.** It is the
> centrepiece of the initial commit — a public commitment to a schedule and a
> definition of done for every phase.

## Planning assumptions

- **Inception:** 1 September 2026.


## Milestone summary

| Phase | Focus | Weeks | Target window | Hours |
|---|---|---|---|---|
| 0 | Foundations & scaffolding | 1 | Sep 1 – Sep 7 | 6 |
| 1 | Data pipeline & validation | 2–3 | Sep 8 – Sep 21 | 12 |
| 2 | Feature engineering | 4–5 | Sep 22 – Oct 5 | 12 |
| 3 | Dixon-Coles model | 6–7 | Oct 6 – Oct 19 | 12 |
| 4 | Probability calibration | 8 | Oct 20 – Oct 26 | 6 |
| 5 | Staking (Kelly) | 9 | Oct 27 – Nov 2 | 6 |
| 6 | Backtest & CLV | 10 | Nov 3 – Nov 9 | 6 |
| 7 | Hardening | 11 | Nov 10 – Nov 16 | 6 |
| 8 | Docs & presentation | 12 | Nov 17 – Nov 23 | 6 |

**🎯 Target presentation date: 21 November 2026.**

---

## Phase detail

### Phase 0 — Foundations (Week 1 · Sep 1–7)
Scaffold the repo, environment, config system, and data schema.
- **Deliverables:** repo live on GitHub; `pip install -r requirements.txt` works;
  `config.yaml` drives a stub run; sample data loads and prints a shape report.
- **Done when:** a fresh clone runs `python scripts/run_pipeline.py` w

### Phase 1 — Data pipeline & validation (Weeks 2–3 · Sep 8–21)
Ingestion, cleaning, strict schema validation, canonical storage.
- **Deliverables:** `ingest.py`; a data-quality report (row counts, null map,
  date coverage, dedup); validated processed dataset.
- **Done when:** malformed input fails loudly with a clear error, and clean input
  produces a canonical table that every downstream stage can rely on.

### Phase 2 — Feature engineering (Weeks 4–5 · Sep 22–Oct 5)
Attack/defence strengths, **Bayesian shrinkage**, **xG proxies**, time decay,
home advantage.
- **Deliverables:** `features.py`; a notebook visualising the shrinkage effect
  (small-sample teams pulled toward the mean); a feature matrix.
- **Done when:** features are reproducible, documented, and the shrinkage
  behaviour is demonstrably correct on a toy example.

### Phase 3 — Dixon-Coles model (Weeks 6–7 · Oct 6–19)
Bivariate Poisson with the `rho` low-score correction, fitted by time-weighted
MLE.
- **Deliverables:** `dixon_coles.py`; fitted parameters; a full score-matrix →
  market-probability derivation; sanity checks against known match probabilities.
- **Done when:** the optimiser converges reliably from multiple starts, unit
  tests on the likelihood and `rho` correction pass, and probabilities for a
  known fixture look sensible.

### Phase 4 — Calibration (Week 8 · Oct 20–26)
Isotonic / Platt scaling and reliability analysis.
- **Deliverables:** `calibrate.py`; reliability diagrams; Brier and log-loss
  before/after.
- **Done when:** calibrated probabilities measurably beat raw ones on a proper
  scoring rule, out of sample.

### Phase 5 — Staking (Week 9 · Oct 27–Nov 2)
Value detection, **fractional Kelly**, bankroll simulation.
- **Deliverables:** `staking.py`; a bankroll curve with drawdowns; tests on the
  Kelly fraction maths.
- **Done when:** staking is fully driven by `(model prob, market odds)` and the
  bankroll simulation is reproducible.

### Phase 6 — Backtest & CLV (Week 10 · Nov 3–9)
Walk-forward engine, lookahead guards, **closing-line value**, headline metrics.
- **Deliverables:** `backtest.py`, `clv.py`, `metrics.py`; a backtest report;
  the CLV distribution chart.
- **Done when:** the backtest provably uses only pre-match information, and CLV +
  ROI/yield/drawdown are reported with confidence intervals.

### Phase 7 — Hardening (Week 11 · Nov 10–16)
Tests, input validation, logging, reproducibility, CI.
- **Deliverables:** green test suite; pinned deps; structured logging; a GitHub
  Actions workflow running tests on push.
- **Done when:** a clean clone reproduces the headline numbers exactly, and CI is
  green.

### Phase 8 — Docs & presentation (Week 12 · Nov 17–23)
Finalise README with real charts, complete `methodology.md`, build the deck.
- **Deliverables:** results-populated README; methodology write-up; slide deck
  (problem → method → results → honest limitations).
- **Done when:** the project can be *presented* as fluently as it can be run.
  **← Presentation milestone, target 21 Nov 2026.**

---

## Parallel context

This project runs alongside the **Credit Default Scorecard** at a matching ~1
hour/day. The scorecard is the smaller build and is scheduled to reach its
presentation milestone first (~31 Oct 2026); V8 follows (~21 Nov 2026). Both
begin at inception on 1 September 2026.

## Risks & adjustments

- **Dixon-Coles convergence** is the likeliest place to lose time. If the
  optimiser is stubborn, Phase 3 borrows a week from the buffer rather than
  shipping an unstable fit.
- **Data quality** on free odds/results feeds can be uneven; Phase 1 is
  deliberately generous for this reason.
- If time compresses, the honest cut is to ship a smaller, fully-validated
  system rather than a larger, half-validated one. CLV integrity is never cut.
