# Roadmap — V8 Quantitative Football Model

> **This is the project timeline: from inception to presentation.** It is the
> centrepiece of the initial commit — a public commitment to a schedule and a
> definition of done for every phase.

## Outcome: plan against actual *(added 6 Nov 2026)*

The plan below is kept as written on 30 Sep, because a schedule only means
something if it can be compared with what happened. The dates under "actual"
are the session headings in `PROGRESS.md`, where each phase's work is recorded.

| Phase | Planned window | Actual | Done-when, honestly assessed |
|---|---|---|---|
| 0 Foundations | Sep 28 – Oct 4 | 2026-09-28 | Met: a fresh clone runs the pipeline (CI does this on every push) |
| 1 Data pipeline | Oct 5 – Oct 18 | 2026-10-02 | Met: malformed input fails loudly; plus a cross-mirror audit added 23 Oct |
| 2 Features | Oct 19 – Nov 1 | 2026-10-09 | Met |
| 3 Dixon-Coles | Nov 2 – Nov 15 | 2026-10-10 | Met: multi-start convergence and parameter recovery, after two optimiser bugs |
| 4 Calibration | Nov 16 – Nov 22 | 2026-10-13 | **Partly met**: calibration improved the model out of sample on 13 seasons, not on 2. Reported, not tuned |
| 5 Staking | Nov 23 – Nov 29 | 2026-10-14 | Met |
| 6 Backtest & CLV | Nov 30 – Dec 6 | 2026-10-18 | Met: lookahead guard on every window; CLV with intervals and decomposition |
| 7 Hardening | Dec 7 – Dec 13 | 2026-10-20 | Met, with one qualification: exact reproduction holds per machine; across platforms, to the reported precision (methodology §7.3) |
| 8 Docs & presentation | Dec 14 – Dec 20 | 2026-10-21 onward | README, methodology, exploration and STUDY_GUIDE.md done; **slide deck outstanding** |

**What the plan did not foresee**, and took the time it saved: the data source
being blocked (solved 23 Oct by rebuilding from two mirrors), every published
number having to be regenerated and re-verified (22–25 Oct), validating the
validator with a calibrated simulation study (23–26 Oct), a 13-season second
result (27 Oct), and a direct model-versus-market comparison (5 Nov).

The build ran about six weeks ahead of the baseline. The 18 December
presentation date stands.

---

## Planning assumptions

- **Inception:** 28 September 2026 (documentation committed 1 Sep; the build
  itself started 28 Sep and the schedule is baselined from that date).
- **Pace:** ~1 hour/day, with **one commit per working day** — the commit
  history is itself the evidence of consistent, incremental delivery.
- **Effort:** ~1 hour per day, ~6 days per week → **~6 focused hours/week**.
- **Honesty note on small sessions:** one-hour blocks carry a context-switching
  cost (you spend a few minutes reloading state each session). Estimates below
  include a modest buffer for this. Numeric/optimisation work (Dixon-Coles fit)
  is deliberately given more time — it is the hardest part to get *stable*, not
  just *running*.
- Dates are targets, not promises. The commit history is the real record.

## Milestone summary

| Phase | Focus | Weeks | Target window | Hours |
|---|---|---|---|---|
| 0 | Foundations & scaffolding | 1 | Sep 28 – Oct 4 | 6 |
| 1 | Data pipeline & validation | 2–3 | Oct 5 – Oct 18 | 12 |
| 2 | Feature engineering | 4–5 | Oct 19 – Nov 1 | 12 |
| 3 | Dixon-Coles model | 6–7 | Nov 2 – Nov 15 | 12 |
| 4 | Probability calibration | 8 | Nov 16 – Nov 22 | 6 |
| 5 | Staking (Kelly) | 9 | Nov 23 – Nov 29 | 6 |
| 6 | Backtest & CLV | 10 | Nov 30 – Dec 6 | 6 |
| 7 | Hardening | 11 | Dec 7 – Dec 13 | 6 |
| 8 | Docs & presentation | 12 | Dec 14 – Dec 20 | 6 |

**🎯 Target presentation date: 18 December 2026.**

---

## Phase detail

### Phase 0 — Foundations (Week 1 · Sep 28–Oct 4)
Scaffold the repo, environment, config system, and data schema.
- **Deliverables:** repo live on GitHub; `pip install -r requirements.txt` works;
  `config.yaml` drives a stub run; sample data loads and prints a shape report.
- **Done when:** a fresh clone runs `python scripts/run_pipeline.py` without error
  (even if it only echoes the config).

### Phase 1 — Data pipeline & validation (Weeks 2–3 · Oct 5–18)
Ingestion, cleaning, strict schema validation, canonical storage.
- **Deliverables:** `ingest.py`; a data-quality report (row counts, null map,
  date coverage, dedup); validated processed dataset.
- **Done when:** malformed input fails loudly with a clear error, and clean input
  produces a canonical table that every downstream stage can rely on.

### Phase 2 — Feature engineering (Weeks 4–5 · Oct 19–Nov 1)
Attack/defence strengths, **Bayesian shrinkage**, **xG proxies**, time decay,
home advantage.
- **Deliverables:** `features.py`; a notebook visualising the shrinkage effect
  (small-sample teams pulled toward the mean); a feature matrix.
- **Done when:** features are reproducible, documented, and the shrinkage
  behaviour is demonstrably correct on a toy example.

### Phase 3 — Dixon-Coles model (Weeks 6–7 · Nov 2–15)
Bivariate Poisson with the `rho` low-score correction, fitted by time-weighted
MLE.
- **Deliverables:** `dixon_coles.py`; fitted parameters; a full score-matrix →
  market-probability derivation; sanity checks against known match probabilities.
- **Done when:** the optimiser converges reliably from multiple starts, unit
  tests on the likelihood and `rho` correction pass, and probabilities for a
  known fixture look sensible.

### Phase 4 — Calibration (Week 8 · Nov 16–22)
Isotonic / Platt scaling and reliability analysis.
- **Deliverables:** `calibrate.py`; reliability diagrams; Brier and log-loss
  before/after.
- **Done when:** calibrated probabilities measurably beat raw ones on a proper
  scoring rule, out of sample.

### Phase 5 — Staking (Week 9 · Nov 23–29)
Value detection, **fractional Kelly**, bankroll simulation.
- **Deliverables:** `staking.py`; a bankroll curve with drawdowns; tests on the
  Kelly fraction maths.
- **Done when:** staking is fully driven by `(model prob, market odds)` and the
  bankroll simulation is reproducible.

### Phase 6 — Backtest & CLV (Week 10 · Nov 30–Dec 6)
Walk-forward engine, lookahead guards, **closing-line value**, headline metrics.
- **Deliverables:** `backtest.py`, `clv.py`, `metrics.py`; a backtest report;
  the CLV distribution chart.
- **Done when:** the backtest provably uses only pre-match information, and CLV +
  ROI/yield/drawdown are reported with confidence intervals.

### Phase 7 — Hardening (Week 11 · Dec 7–13)
Tests, input validation, logging, reproducibility, CI.
- **Deliverables:** green test suite; pinned deps; structured logging; a GitHub
  Actions workflow running tests on push.
- **Done when:** a clean clone reproduces the headline numbers exactly, and CI is
  green.

### Phase 8 — Docs & presentation (Week 12 · Dec 14–20)
Finalise README with real charts, complete `methodology.md`, build the deck.
- **Deliverables:** results-populated README; methodology write-up; slide deck
  (problem → method → results → honest limitations).
- **Done when:** the project can be *presented* as fluently as it can be run.
  **← Presentation milestone, target 18 Dec 2026.**

---

## Parallel context

This project runs alongside the **Credit Default Scorecard** at a matching ~1
hour/day. The scorecard is the smaller build and is scheduled to reach its
presentation milestone first (~27 Nov 2026); V8 follows (~18 Dec 2026). Both
are baselined from 28 September 2026.

## Risks & adjustments

- **Dixon-Coles convergence** is the likeliest place to lose time. If the
  optimiser is stubborn, Phase 3 borrows a week from the buffer rather than
  shipping an unstable fit.
- **Data quality** on free odds/results feeds can be uneven; Phase 1 is
  deliberately generous for this reason.
- If time compresses, the honest cut is to ship a smaller, fully-validated
  system rather than a larger, half-validated one. CLV integrity is never cut.
