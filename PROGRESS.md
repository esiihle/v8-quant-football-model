# Progress Log — V8 Quantitative Football Model

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus `docs/ROADMAP.md`)
> and you know exactly where the project stands. Newest entry at the top.

**Current phase:** Phase 5 — Staking (Nov 23 – Nov 29), started early
**Overall status:** 🟢 Pipeline runs end-to-end on **real** EPL data; tests green (6);
exploration notebook written and run.
**Next action:** Run the exploration notebook and fill in its findings table
(carried over from 1 Oct), then review the Phase 1 data-quality report against
what the notebook shows.
**Schedule:** baselined from 28 Sep 2026; target presentation **18 Dec 2026**.
**Cadence:** one commit per working day (missed 29 Sep — first and last).

---

## Open decisions
- [x] **Real data source:** football-data.co.uk, EPL (`E0`), seasons 2023/24 and
      2024/25 — chosen because it carries pre-match *and* closing odds in one
      file, which is what makes CLV measurable. Settled 30 Sep 2026.
- [ ] **De-vig method** for market-implied probabilities (proportional vs. Shin) —
      settle before Phase 5 staking.

## Phase checklist
- [x] Documentation (OVERVIEW, README, ROADMAP, methodology, LICENSE, .gitignore)
- [x] Phase 0 — Foundations & scaffolding
- [ ] Phase 1 — Data pipeline & validation
- [ ] Phase 2 — Feature engineering
- [ ] Phase 3 — Dixon-Coles model
- [ ] Phase 4 — Probability calibration
- [ ] Phase 5 — Staking (Kelly)
- [ ] Phase 6 — Backtest & CLV
- [ ] Phase 7 — Hardening
- [ ] Phase 8 — Docs & presentation

## How to run (current state)
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py     # real data from data/raw if present, else the sample
python scripts/run_pipeline.py --sample   # force the bundled sample
pytest                             # 117 tests (smoke + Phases 1-5), all green
python scripts/run_pipeline.py --no-save  # run the checks without writing output
# notebooks/01_exploration.ipynb   # open in VS Code, Run All
```

---

## Session log

### 2026-10-14 — Phase 5: value detection, fractional Kelly, bankroll simulation
- `src/staking.py` built: `overround`, `devig_proportional`, `devig_shin`,
  `market_probabilities`, `kelly_fraction`, `expected_log_growth`, `stake_size`,
  `find_value_bets`, `simulate_bankroll`, `staking_report`.

#### A conceptual error this module shipped with, and the fix
The first version tested value against the **de-vigged** market probability.
That is the permissive direction and it is wrong: the payout is at the actual
odds, so expected value is positive if and only if `p > 1/o`. At odds of 3.50 in
a 4.9% book, a model saying 0.278 beats de-vigged fair value (0.272) and has
EV of **-2.7%**. The corrected module measures:
  - `edge` against the **raw price** — decides bets, and agrees with Kelly's own
    sign by construction (`test_edge_and_kelly_agree_on_when_to_bet`);
  - `disagreement` against de-vigged fair value — a diagnostic, and the basis of
    Phase 6's CLV. It never triggers a bet.
`test_a_bet_beating_fair_but_not_raw_is_not_value` pins it.

#### The "2x Kelly" claim, corrected
Written first as an exact identity ("2x Kelly gives zero growth"). It is not —
that is the small-edge / Gaussian limit. Measured for discrete bets the
zero-growth point runs 1.947x (p=0.6, odds 2.0) to 2.098x (p=0.25, odds 5.0),
converging on exactly 2.000x as the edge shrinks (p=0.505). The practical
conclusion is unchanged and is the real argument for fractional Kelly: a model
20% too optimistic is already staking about double what it should.

- **De-vig decision settled** (open since Phase 1 methodology §1): **Shin**, not
  proportional. Bookmakers load margin onto longshots, so proportional
  de-vigging overstates longshot probabilities; Shin corrects it. Matters for
  CLV rather than bet selection, but a biased fair-value estimate would bias the
  headline metric.
- Risk caps imposed on top of Kelly and labelled as risk management, not theory:
  5% per bet, 15% total exposure per day (crowded days scaled proportionally).
- Same-day bets sized against the same bankroll — at the moment Saturday's bets
  are placed, Saturday's results do not exist.
- ROI (on bankroll) and yield (on turnover) both reported, because they answer
  different questions and are routinely conflated.
- `tests/test_staking.py` (new): 30 tests — Kelly by hand, zero at fair odds,
  negative without edge, **Kelly verified as the numerical maximiser of expected
  log growth**, the zero-growth point measured, quarter-Kelly outperforming full
  Kelly under an over-optimistic p, overround, both de-vig methods summing to 1,
  **Shin shading longshots down**, clean fallback on a no-margin book, a
  fair-matching model finding no value, the fair-but-not-raw case, edge/Kelly
  agreement, caps binding, same-day sizing, reproducibility, fractional Kelly
  reducing drawdown, ROI != yield. Suite now 117 green.
- `config.yaml`: `staking` block expanded — `devig_method`,
  `max_stake_fraction`, `max_daily_exposure`, with the reasoning for
  `kelly_fraction: 0.25` written in the file.
- `scripts/run_pipeline.py`: Stage 5 stakes only the calibration **evaluation**
  period (the fitting period taught the map, so betting it would be hindsight)
  and writes `data/processed/bet_ledger.csv`.
- **Still outstanding:** `notebooks/01_exploration.ipynb` has not been run.
- **Next:** Phase 6 — walk-forward backtest and CLV, the headline metric. The
  staking numbers are not a result until that exists.

### 2026-10-13 — Phase 4: probability calibration
- `src/metrics.py` **partially built**: the proper scoring rules land now
  (Brier, log-loss, multiclass versions, base-rate baseline); the betting
  metrics still belong to Phase 6, so `metrics.summarise` remains a stub.
- `src/calibrate.py` built: `reliability_table`, `expected_calibration_error`,
  `fit_calibrator` / `apply_calibrator` (Platt and isotonic),
  `fit_multiclass` / `apply_multiclass`, `renormalisation_cost`,
  `split_by_time`, `evaluate`, `compare_methods`, `calibration_report`,
  `save_calibration` / `load_calibration`.
- **Calibration is fitted on the earlier half and judged on the later half.** A
  map scored on the predictions it was fitted to always looks good because it
  has seen the answers. `test_in_sample_calibration_flatters_itself` demonstrates
  the gap rather than asserting it.
- **An interface bug caught while wiring:** the probability columns are
  `home_win`/`draw`/`away_win` but the results are coded `H`/`D`/`A`. The first
  version of the multiclass metrics conflated the two and would have scored the
  wrong thing silently. Every multi-outcome metric now takes an explicit
  `{column: label}` mapping, and `test_mismatched_labels_are_rejected` pins it.
- **Platt vs isotonic, measured out of sample on the first real run:** Platt
  +0.0018 log-loss, isotonic **-0.0129** — isotonic made the model worse, which
  is exactly the overfitting the docs predicted for a ~336-match fitting window
  split three ways. The choice of method is now evidence, not assertion.
- Three outcomes are calibrated separately and renormalised; the size of that
  approximation is measured by `renormalisation_cost` rather than hidden.
- Reliability bands are judged against **three standard errors of their own
  count**, not a flat tolerance — a 200-match band has a standard error of 0.035,
  so a 0.06 gap there is noise while the same gap in a 2,000-match band is real.
- `tests/test_calibrate.py` (new): 26 tests — Brier and log-loss by hand,
  log-loss punishing confident misses, zero-probability survival, multiclass by
  hand, mismatched labels rejected, baseline floor, reliability and ECE on a
  known-calibrated model, ECE detecting bias, Platt correcting overconfidence,
  isotonic correcting a shape Platt cannot, JSON round trips, single-class and
  unknown-method guards, renormalisation measured, chronological split enforced,
  out-of-sample improvement, in-sample flattery demonstrated. Suite now 87 green.
- `config.yaml`: `calibration` block — `method`, `fit_fraction`,
  `reliability_bins`, `calibration_filename`.
- `scripts/run_pipeline.py`: Stage 4 prices every match with sufficient history,
  reports before/after out of sample with a PASS/FAIL verdict, compares both
  methods, and degrades cleanly on the bundled sample.
- **Still outstanding:** `notebooks/01_exploration.ipynb` has not been run.
- **Next:** Phase 5 — staking (value detection, fractional Kelly, bankroll
  simulation).

### 2026-10-10 — Phase 3: Dixon-Coles fit by penalised, time-weighted MLE
- `src/dixon_coles.py` built: `tau`, `rho_bounds`, `goal_ceiling`,
  `expected_goals`, `log_likelihood`, `fit`, `score_matrix`,
  `market_probabilities`, `predict`, `fit_report`, `save_model` / `load_model`.
- Identifiability by forcing attack parameters to sum to zero (n-1 free).
- L2 penalty on strengths = Phase 2's shrinkage as regularisation; Phase 2's
  shrunk strengths seed the optimiser as its first starting point.
- Multi-start fitting, with the objective spread across starts reported —
  agreement is the evidence the optimum is global, not lucky.
- One fit prices 1X2, over/under 2.5 and BTTS from the same score matrix, so
  the markets are automatically consistent.

#### Two bugs found by tests, both invisible in the output
1. **`+inf` for invalid parameters broke the optimiser.** L-BFGS-B uses
   finite-difference gradients, so `inf - inf = nan` and the fit stopped
   improving *while reporting success*. Parameter recovery was r = 0.21 with no
   error anywhere. Invalid points now cost a large FINITE amount that grows with
   the violation.
2. **The rho box was sized from a theoretical 55-goals-a-match ceiling**, which
   squeezed it to [-0.018, 0.0003] and left rho pinned on the lower edge. The
   tell: the bias did not shrink as the sample grew — noise averages out, a
   binding constraint does not. The box is now sized from observed goals, and
   the fit reports `rho_at_bound`.

- After both fixes, recovery on simulated data: attack r = 0.99, defence
  r = 0.98, rho -0.084 against a true -0.080 at 2,400 matches.
- `tests/test_dixon_coles.py` (new): 24 tests in three groups — identities (tau
  formulas against the paper, mass preservation to 1e-12, rho = 0 reduces to
  independent Poisson, every market sums to 1), recovery (known parameters
  refound, sum-to-zero holds, gamma positive, starts agree, penalty shrinks
  spread), and discipline (`as_of` respected, no-history fit fails loudly), plus
  four regression tests pinning the two bugs above. Suite now 61 green.
- `config.yaml`: `model.penalty`, `model.n_starts`, `model.max_iterations`,
  `model.model_filename`.
- `scripts/run_pipeline.py`: Stage 3 fits, reports diagnostics, prices the most
  recent fixture as a worked example, and writes `data/processed/dixon_coles.json`.
- **Still outstanding:** `notebooks/01_exploration.ipynb` has not been run. Its
  measured low-score departures are what the fitted rho should be consistent
  with — a free sanity check on this phase.
- **Next:** Phase 4 — probability calibration (isotonic / Platt, reliability
  diagrams, Brier and log-loss).

### 2026-10-09 — Phase 2: features (shrinkage, time decay, xG proxy)
- `src/features.py` built: `time_weights`, `shrink`, `conversion_rates`,
  `blend_goals`, `team_strengths`, `build_features`, `feature_report`,
  `save_features`.
- **A real lookahead leak was found and fixed during the build.** The first
  version fitted the xG conversion rate over the whole file, so an August match
  was scored partly on matches played in May. `test_features_do_not_use_the_future`
  caught it. `blend_goals` now requires `history` as an argument, and the
  conversion rate is recomputed per match date from past matches only.
- Shrinkage uses the **effective** sample size (sum of time-decay weights), not a
  row count — ten matches from last season are not ten matches' worth of evidence.
- Warm-up rows flagged, not deleted; unknown teams get the prior, not a null.
- `notebooks/02_features.ipynb` (new): the Phase 2 "done when" deliverable — the
  shrinkage toy example verifiable by hand (n = k lands exactly halfway), the
  decay curve with its checkable points, the real strength table, an
  attack-vs-defence chart, and a lookahead check run in the open.
- `tests/test_features.py` (new): 19 tests — half-life weighting, future weight
  zero, shrinkage identities (n=0 is the prior, n=k is halfway, monotone in n,
  harsher k shrinks more), xG blend and its fallback, better team rates better,
  matrix shape, warm-up flags, unknown team prior, determinism, history-only
  conversion rate, and the lookahead test. Suite now 37 green.
- `config.yaml`: `features` block — `shrinkage_k`, `half_life_days`, `xg_weight`,
  `min_prior_matches`, `feature_filename`.
- `scripts/run_pipeline.py`: Stage 2 runs and writes `data/processed/features.csv`.
- **Still outstanding:** `notebooks/01_exploration.ipynb` has not been run and its
  findings table is unfilled.
- **Next:** Phase 3 — Dixon-Coles fit by maximum likelihood.

### 2026-10-02 — Phase 1: validation, cleaning, canonical table
- `src/canonical.py` (new): the canonical column contract — names, order, the
  raw→canonical mapping, and which columns may never be null. Downstream stages
  read these names and never a raw source column like `FTHG`, so a change of data
  source touches one file.
- `src/ingest.py`: Phase 1 flow — `validate_raw` → `clean` → `drop_duplicate_fixtures`
  → `to_canonical` → `save_canonical`, orchestrated by `run_ingest`.
  - Structural faults (missing required column, empty input) raise `DataQualityError`.
  - Cleaning: date parsing, team-name whitespace, impossible goals dropped,
    **result recomputed from goals** rather than trusted, out-of-range odds voided.
  - Duplicate fixtures (same date + same two teams) dropped and counted.
  - `MatchID` added: league_season_date_home_away — stable, unique, traceable.
  - Guard rail: if more than `validation.max_dropped_fraction` (5%) of rows would
    be discarded, the run stops rather than quietly continuing on a broken file.
- `config.yaml`: new `validation` block — every threshold is configuration, not
  a number buried in code.
- `scripts/run_pipeline.py`: runs Phase 1 and prints the data-quality report;
  writes `data/processed/matches.csv` (`--no-save` to skip).
- `tests/test_ingest.py` (new): 11 behaviour tests, one per defect — missing
  column, unparseable date, whitespace team name, impossible score, result
  mismatch, bad price, duplicate fixture, canonical shape/order/uniqueness, and
  the drop-fraction guard. Suite now 18 green.
- **Not yet done:** the exploration notebook from 1 Oct still needs to be run and
  its findings table filled in.
- **Next:** Phase 2 — features (shrinkage, xG proxy, time decay).

### 2026-10-01 — Exploration notebook
- Added `notebooks/01_exploration.ipynb`. It imports from `src/` rather than
  duplicating loader logic, and runs on the real files when present, the sample
  otherwise.
- Sections, each tied to the phase it justifies: home advantage (Phase 3),
  observed scorelines vs independent Poisson (motivates the Dixon-Coles τ
  correction, Phase 3), bookmaker overround and market calibration (de-vig
  decision + the Phase 4 benchmark), shots vs goals (the xG proxy, Phase 2).
- Added `ipykernel` to requirements.txt (VS Code's Jupyter extension needs it).

#### Findings from the real data — FILL IN after running the notebook
| Measurement | Value | Expected / note |
|---|---|---|
| Outcome split H / D / A | | ~46 / 26 / 28 in top leagues |
| Mean goals home / away | | home should be clearly higher |
| 0-0 vs Poisson (`diff_pct`) | | Dixon-Coles predicts an excess |
| 1-0 vs Poisson | | |
| 0-1 vs Poisson | | |
| 1-1 vs Poisson | | |
| Mean overround, bet-into odds | | |
| Mean overround, closing odds | | closing is usually tighter |
| Shots on target vs goals, r | | ~0.5 is normal and useful |

- **Next:** draft the canonical schema that Phase 1's validation will enforce.

### 2026-09-30 — Real data ingestion + schedule re-baseline
- Re-baselined `docs/ROADMAP.md` from a 28 Sep inception; target presentation
  moves to **18 Dec 2026**. Phase lengths unchanged — only the start slipped.
- Settled the data source: football-data.co.uk (pre-match **and** closing odds
  in one file → CLV is measurable). Added `leagues`, `seasons`, `file_pattern`
  and `download_url` to `config.yaml`.
- `src/ingest.py`: added `expected_raw_files()` (what the config asks for, and
  whether it's on disk, with the download URL) and `load_many()` (stack several
  seasons, tagging each row with `League`/`Season`). Loader now survives
  non-UTF-8 files and strips empty padding rows/columns.
- Shape report now covers date range (mixed 2- and 4-digit years), matches per
  season, and **odds-group coverage**, so "no closing odds" can never pass
  unnoticed.
- `scripts/run_pipeline.py`: uses real data when present, falls back to the
  bundled sample and prints exact download URLs when not (`--sample` forces it).
- `docs/methodology.md` §1 filled in: source, fields used, and the source's
  known quirks.
- Tests: 6 green (added raw-file discovery + odds-coverage checks).
- **Next:** Phase 1 — strict schema validation and the data-quality report.

### 2026-09-28 — Phase 0 scaffolding
- Built the runnable skeleton: `config.py` (+ `config.yaml`), implemented
  `ingest.py` (loader + shape report), and stubs for every downstream stage.
- `scripts/run_pipeline.py` runs end-to-end and reports each stage as
  done/pending — it doubles as a live progress indicator that lights up as
  phases land.
- Added `tests/test_smoke.py` (4 tests, green) and a bundled sample dataset.
- Verified: pipeline runs clean on a fresh checkout; tests pass.
- **Next:** Phase 1 — strict schema validation + real data ingestion.

### 2026-09-01 — Project inception
- Created repo and committed all documentation (overview, README, roadmap,
  methodology skeleton, LICENSE, .gitignore).

<!--
Template for each new entry (copy this, newest at the top):

### YYYY-MM-DD — <short title>
- What I did this session:
- Decisions made:
- Anything broken / to revisit:
- Next action:
-->
