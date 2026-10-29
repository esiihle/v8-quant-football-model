# Progress Log — V8 Quantitative Football Model

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus `docs/ROADMAP.md`)
> and you know exactly where the project stands. Newest entry at the top.

**Current phase:** Complete, with two real-data results: the 2-season headline
(fingerprint `eddedac58a664dd8`) and the 13-season long run (`9ed0f574b163c749`).
Every figure in README and methodology is checked against its `numbers.json`.
**Overall status:** 🟢 All stages built and run on real data; see the latest
entry for the test count. CI runs the suite and a full pipeline run on two
Python versions.
**Next action:** A fair model-vs-market calibration comparison on the
model's walk-forward probabilities (see 29 Oct), then `docs/STUDY_GUIDE.md`,
then the slide deck.
**Schedule:** baselined from 28 Sep 2026; target presentation **18 Dec 2026**.
**Cadence:** one commit per working day (missed 29 Sep — first and last).

---

## Open decisions
- [x] **Real data source:** football-data.co.uk, EPL (`E0`), seasons 2023/24 and
      2024/25 — chosen because it carries pre-match *and* closing odds in one
      file, which is what makes CLV measurable. Settled 30 Sep 2026.
- [x] **De-vig method:** Shin — settled 14 Oct 2026 (see that entry).
- [x] **Data access:** football-data.co.uk is blocked on the author's network;
      files are rebuilt from two GitHub mirrors by
      `scripts/rebuild_raw_from_mirrors.py`, with a cross-mirror score audit.
      Settled 23 Oct 2026.

## Phase checklist
- [x] Documentation (OVERVIEW, README, ROADMAP, methodology, LICENSE, .gitignore)
- [x] Phase 0 — Foundations & scaffolding
- [x] Phase 1 — Data pipeline & validation
- [x] Phase 2 — Feature engineering
- [x] Phase 3 — Dixon-Coles model
- [x] Phase 4 — Probability calibration
- [x] Phase 5 — Staking (Kelly)
- [x] Phase 6 — Backtest & CLV
- [x] Phase 7 — Hardening
- [x] Phase 8 — Docs & presentation

## How to run (current state)
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py     # real data from data/raw if present, else the sample
python scripts/run_pipeline.py --sample   # force the bundled sample
pytest                             # 208 tests, none skip once numbers.json exists
python scripts/rebuild_raw_from_mirrors.py --odds ../mirror-odds --shots ../mirror-shots
python scripts/verify_raw_data.py data/raw/*.csv
python scripts/run_pipeline.py --no-save  # run the checks without writing output
python scripts/run_pipeline.py --log-file run.log   # full diagnostic log to a file
python scripts/make_charts.py      # regenerate every README chart + numbers.json
# notebooks/01_exploration.ipynb   # open in VS Code, Run All
```

---

## Session log

### 2026-10-29 — The exploration notebook, finally run, and what it overturned
The notebook carried since 1 Oct had in fact been run once, on the 12-match
sample, and those outputs had been on GitHub for four weeks: a 58% home-win
rate, and "0-3 is 603% more common than Poisson". It is the same fault as the
sample charts of 22 Oct, in a file no test looked at. Its text was also wrong in
two places: it called the Shin decision "open", and it said Dixon-Coles raises
all four low scores. ρ < 0 raises 0-0 and 1-1 and *lowers* 1-0 and 0-1.

#### Rebuilt as code, not a notebook
- `src/exploration.py`: every measurement is a function returning a result,
  each rate with an interval (Wilson for shares, Fisher for correlations,
  Poisson for counts).
- `scripts/explore.py` writes `exploration.json` for either config and refuses
  sample data into any published folder.
- The notebook now only calls and displays, and prints the run fingerprint and
  data source in its first cell.
- `test_the_notebook_was_run_on_real_data_with_the_current_config`: every code
  cell executed, no errors, "REAL data", and the current config's fingerprint.
  It fails on the notebook as committed until today.
- Methodology §1.5 holds the findings table this log asked for on 1 Oct,
  checked against both exploration.json files.

#### The null decides the answer
Against a league-average Poisson, 13 seasons show 1-1 **11.3% rarer** than
independence. Against each match's own expected goals (a per-season ρ = 0
fit), **+0.5%**. The deficit was team mixing, since a mixture of Poissons is
not a Poisson. `test_league_mean_null_mistakes_team_mixing_for_a_low_score_effect`
reproduces the trap on simulated data with ρ = 0. What survives: 0-1 rarer by
12.7% [−21.7%, −3.5%], and ρ negative in 10 of 13 seasons (sign test p =
0.046). That is partial support for the correction, consistent with the
walk-forward fitting ρ near zero.

#### Calibration error has a noise floor
ECE is never zero on finite data. `ece_noise_floor` draws outcomes from a
forecaster's own probabilities to get the ECE a perfectly calibrated one would
score. Two consequences:
- The closing line sits under its floor on both datasets: calibrated to within
  what the data can show.
- The model's Phase 4 ECE on 2 seasons, 0.061 on 341 matches, is under its
  floor of about 0.080: the reliability diagram cannot show miscalibration
  either way. `numbers.json` now records `ece_noise_floor_p95`, and the README
  says so beside the chart.
- An earlier draft of this session claimed "the market beats the model on
  calibration". Withdrawn before it reached a document: the Phase 4
  probabilities come from one end-of-data fit, which on 13 seasons, with the
  minimum match weight, prices 2013 matches with 2024 strengths. A fair
  comparison needs the walk-forward probabilities. That is the next item.

Suite: **251 tests, 0 skipped** with data and generated outputs present.

#### A test threshold set inside the noise
`test_a_calibrated_market_scores_a_small_ece` failed at 6,000 matches with ECE
0.0226 against a 0.02 threshold. Across seeds, a perfectly calibrated market
scores 0.007–0.023 at that size, so the threshold was inside the noise. Fixed
with 60,000 matches (floor ~0.003) and a threshold of 0.006, not by loosening
the number until it passed.


### 2026-10-27 — Thirteen seasons, committed as a second result
#### A minimum match weight: 67 minutes to 11
`features.min_match_weight: 0.01` drops matches whose time-decay weight is
below 1%: older than 180 × log2(100) ≈ 1,196 days at the 180-day half-life.
For an unbroken fixture list those matches carry exactly 1% of the total
weight (the cutoff is its own tail share), yet over 13 seasons they were most
of the rows, and they kept long-relegated teams in the parameter vector. Each
parameter costs a likelihood evaluation per finite-difference gradient.
- Implemented once, in `time_weights`; both callers already drop zero-weight
  rows.
- **2-season headline: all 37 figures in numbers.json bit-identical** with and
  without the cutoff. Only the fingerprint moved: `c6c0bb194dcd9ba2` became
  `eddedac58a664dd8`.
- 13-season chart run: **11 minutes**, against 67 for the uncut pipeline run.
- Tests: the cutoff boundary at 1,196 days, the default unchanged, a fit on all
  history equalling a fit on recent history alone (1e-8), a team seen only long
  ago leaving the parameter vector, and inertness on short history. All but
  the last fail on the old code; the last is an invariance test and passes on
  both, as it should.

#### Variant configs: `extends:`
`configs/epl_2012_2025.yaml` is 11 lines that extend `config.yaml` and change
only the seasons. Dicts merge and lists replace, and the fingerprint is taken
on the merged config. `test_the_long_run_differs_from_the_headline_only_in_its_seasons`
makes the attribution claim checkable: strip the seasons from both configs and
they must be identical. Loops are refused, and paths resolve relative to the
file that names them.

#### The long run (fingerprint `9ed0f574b163c749`)
4,940 matches, 4,309 walk-forward bets. CLV −1.22% [−1.29%, −1.15%] = margin
−1.24% + movement **+0.02% [−0.05%, +0.09%]**. Same verdict as two seasons,
with the movement interval far narrower. Yield −6.05% [−15.20%, +3.44%] includes
zero while ROI is −99.98%: quarter-Kelly on no edge compounds to ruin even when
the average bet is not significantly negative.
- Why the margin is smaller than the headline's: Bet365's bet-into overround
  was about 2.7–3.0% from 2014/15 to 2018/19 and about 5.4% from 2020/21
  (`verify_raw_data.py`, file by file).
- Published subset: the bankroll curve and CLV-by-claimed-edge, generated with
  `--only` so the folder holds nothing the README does not show. A docs test
  enforces that.
- **Not published: the in-sample comparison.** With the cutoff, the "fitted on
  everything" model only knows the last ~3.3 seasons, so for most of 13 seasons
  it is not in-sample at all (in-sample ROI came out −98.94%). The comparison
  is only meaningful on the headline.
- On 13 seasons, Platt calibration improved the model out of sample, which it
  did not on 2.

Suite: **233 tests, 0 skipped** with both runs' outputs present.

#### A borderline bet that flipped between machines
The author's Windows run placed **4,309** long-run bets; the build machine's
Linux run placed 4,310, with the same fingerprint and pinned libraries. One
bet's edge sat on the 2% minimum, and platform floating point put it on
opposite sides. The yield estimate was identical (−6.05%), but the bootstrap
resampled a different number of bets, so both ends of the interval moved
(−15.20% / +3.44% against −15.30% / +3.57%) by more than the docs tests'
0.05-point tolerance. The tests caught it before commit, and the published
figures are the author's machine's, as §7.3 requires. The 2-season run has no
bet that close to the line and matched to every reported digit.

#### Three faults found by looking at the charts
- **Chart links in subfolders were invisible to the docs tests.** The link
  pattern had no "/", so the long run's charts could have been missing or
  stale with nothing failing. The pattern now allows one subfolder.
- **Tick labels rounded into different numbers.** Ticks at −0.25% printed as
  "−0.3%". `tick_pct` uses exactly the decimals a tick needs, on every
  percentage axis.
- **A ruin curve on a linear axis** lay flat on zero for eight years. The
  bankroll chart now uses a log axis when the bankroll falls below 5% of its
  start (`use_log_scale`). The headline's curve stays linear.


### 2026-10-26 — The simulator was too easy; now it is calibrated
Decomposing the simulation study's CLV, the like-for-like comparison promised
yesterday, showed the control had been passing for the wrong reason.

| original simulator | CLV | margin | movement |
|---|---|---|---|
| efficient book | −1.59% | −1.89% | **+0.30%** [+0.11, +0.48] |
| random-model control | −1.20% | −1.66% | **+0.46%** [+0.35, +0.56] |

A model betting random probabilities got the market moving toward its bets.
It passed the no-false-positive check only because the margin was bigger.

#### Diagnosis by elimination
- Every outcome of the simulated efficient book, no selection: movement
  −0.00%. **The instrument is sound**, and the movement came from selection.
- Longest 25% of opening prices, no model at all: simulator **+0.89%**, real
  EPL 2012–25 (Bet365 open, Pinnacle close) **+0.12%** [+0.06, +0.17], about
  seven times too generous.
- Two causes, separated by switching the de-vig and the error structure:
  margin loaded proportionally but read with Shin (**+0.50**), and a close
  that corrected 100% of the opening error (**+0.32**).

#### Fixes
- `load_margin_shin`: the exact inverse of `staking.devig_shin`, so the
  simulator loads margin the way the pipeline unwinds it. Round trip to 1e-10.
- The close keeps a share of the opening error. The share corrected is chosen
  by `simulation_study.py --calibrate`, which measures the model-free statistic
  on the seasons named in `config.yaml` `simulation.calibration_seasons` and
  across a grid on the simulator, then writes `simulation_calibration.json`.
  **Result: 0.4** (simulated +0.13% vs real +0.12%); 0.2–0.5 all fit.
- Sensitivity: detection threshold **4.9 points at shares 0.2, 0.4 and 0.5**.
  The headline survives, and is now earned.
- Calibrated control movement: +0.06% [+0.01, +0.12], within what real-market
  price selection gives, which is now the test's bar. The test I wrote first
  ("no movement toward the bets") was wrong: the real market has the faint
  effect too.
- `--close-corrects` for sensitivity runs refuses to write without `--out`.
- The README's simulation table was never held to its JSON. It is now, by two
  docs tests, which is how all of its figures were caught changing.

#### 13 seasons: the fit does not scale
The walk-forward on EPL 2012–25 (4,940 matches) was only about halfway (151
of roughly 300 windows) by the end of the session, and each window was slower
than the last. The cause is structural: every fit uses
all history and every team ever seen, and L-BFGS-B's finite-difference
gradient costs (parameters × matches) per step. At a 180-day half-life a
match three years old carries weight 2^-6 ≈ 1.6%, and a match from 2013 carries
about 10^-7 by 2024, so almost all of that cost buys nothing.
On 13 seasons, calibration DID improve the model out of sample on both
scoring rules (it did not on 2), consistent with the 2-season failure being a
short fitting window.

#### Next
1. A history cap (`model.max_history_days`), justified by the weight it
   discards, with a test that the 2-season headline is unchanged. Two seasons
   fit inside any sensible cap, so only the fingerprint moves.
2. Then the 13-season walk-forward as a second, larger result.
3. Exploration notebook, STUDY_GUIDE.md.


### 2026-10-25 — The real numbers, two reporting bugs, and where the CLV went
football-data.co.uk is still blocked, but two GitHub mirrors of it together
carry everything: `AnishKhetani/premier-league-data` (every bookmaker's odds,
including Pinnacle's close) and `datasets/football-datasets` (shots, original
column names). `scripts/rebuild_raw_from_mirrors.py` (committed 23 Oct) joins
them and refuses to write if the two copies disagree on a single score. All 13
seasons 2012/13–2024/25 pass that audit. The verifier passes 2023/24 and
2024/25 with 100% odds coverage and 0.0% closing-copied-from-opening.

#### The regenerated result (fingerprint `c6c0bb194dcd9ba2`)
| | in-sample | walk-forward |
|---|---|---|
| bets | 260 | 433 |
| ROI | +603.26% | −82.60% |
| yield (profit ÷ turnover) | +34.07% | −15.84% [−28.57%, −3.47%] |
| CLV, probability points | — | −1.87% [−2.11%, −1.63%] |
| of which margin paid | — | −1.89% |
| of which market movement | — | +0.02% [−0.22%, +0.27%] |

The old README figures (ROI −28.5%, 413 bets, CLV −0.67%) did not reproduce,
and neither did the "measures disagree" example in methodology §6.3. Both are
withdrawn rather than edited. The notice and `numbers.json` rule did exactly
what they were built for.

#### Bug 10: two yields under one name
The same 433 bets printed **−15.84%** in the backtest report (profit ÷ turnover)
and **−10.08% [−22.90%, +3.26%]** in the headline metrics (unweighted mean of
profit/stake). The interval, and the verdict "includes zero, not distinguishable
from no edge", belonged to the second number. On the right number the interval
is [−28.57%, −3.47%], so **the loss is established**: the conclusion flipped.
- `metrics.bootstrap_ratio_ci`: resamples whole bets and recomputes
  Σprofit ÷ Σstake, keeping each bet's profit and stake together.
- Mean return per bet stays in the report under its own name, with a note on
  what the gap means: Kelly's bigger stakes did worse than its smaller ones.
- Six regression tests, all failing on the old code. The existing summarise
  test passed throughout because it only checked `lower < upper`, which is
  true of either yield. A test that cannot tell two candidate answers apart
  does not pin either.

#### Bug 11: negative CLV blamed on the market
The report said the model takes "prices the market later moves AWAY from".
CLV in probability points charges the bookmaker's margin, so it splits exactly
into margin paid + market movement (`clv.decompose_clv`). Decomposed, the
movement is +0.02% with an interval straddling zero: the market never moved,
and **the whole −1.87% is margin**. That is what a model with no information
looks like.
- The verdict is now chosen from the movement: none, toward, or away.
- Found while writing the "toward" test: if the market moves toward the bet
  by less than the margin, the odds ratio is *positive* and points *negative*.
  The report used to blame every disagreement on the ratio's asymmetry, but
  here the cause is the margin, and the report now says so.
- The beat rate (34.4%) compares Bet365's price (5.4% margin) with Pinnacle's
  close (2.9%). Below 50% partly measures that gap between the books, and the
  report says so when movement is zero.
- Five regression tests, all failing on the old code; `numbers.json` carries
  the decomposition; three docs tests hold the README to it, including one
  that fails if "every bit of the negative CLV is the margin" survives a rerun
  that finds movement.

#### Reproducibility, stated precisely
Same fingerprint and pinned libraries, Windows vs Linux: identical bet count,
ROI and CLV to two decimals, but turnover 5,214.99 vs 5,215.11. That comes from
platform floating point inside L-BFGS-B, not non-determinism. `numbers.json` is
generated on the author's machine; methodology §7.3 now states the claim
precisely.

#### Also
- `make_charts.py`: yield interval, mean return per bet and the CLV
  decomposition written to `numbers.json`; the chart description no longer
  hard-codes "the 340-point swing" from the unverified run.
- Calibration did not improve the model out of sample on real data (Platt
  ECE 0.0578 → 0.0613; isotonic worse). The README says so.
- A correction to 23 Oct's handoff: all six former skips were documentation
  tests waiting on `numbers.json`.

#### Next
1. More seasons. Rebuild 2012/13–2024/25, run, and see how far the movement
   interval tightens.
2. Decompose the simulation study's CLV too, so the real and simulated
   *movement* components can be compared like for like.
3. Exploration notebook, then STUDY_GUIDE.md.


### 2026-10-23 — V8 gets a headline that needs no download
football-data.co.uk is unreachable from this network — browser, PowerShell and
curl all time out at TCP connect, while the site itself is up and was fetched
successfully from elsewhere. The mirrors that exist (footballcsv, openfootball)
reformat to a handful of columns and **strip every odds field**, which is the
one thing V8 needs.

Rather than keep hunting, the project now answers a better question.

#### scripts/simulation_study.py — can this backtester tell signal from noise?
Markets generated with properties known by construction, swept over the size of
the mispricing planted in the book:

| mispricing | measured CLV | detected? |
|---|---|---|
| none (efficient) | −1.59% [−1.77%, −1.38%] | no |
| 2.0 points | −0.92% [−1.13%, −0.71%] | no |
| 4.9 points | +0.74% [+0.48%, +0.97%] | **yes** |
| 8.3 points | +3.34% [+3.03%, +3.64%] | **yes** |
| control: random model | −1.20% [−1.31%, −1.09%] | no |

- **The instrument responds** — CLV monotonic in the planted edge, crossing zero
  where it should.
- **The instrument does not hallucinate** — no edge reported on an efficient
  book, none from a model betting random probabilities.
- **Detection threshold: ~5 probability points on 760 matches.** A power
  statement, and the thing that makes any real-data CLV interpretable: near zero
  means *no edge of detectable size*, not *no edge*.

This is a complete, reproducible headline result requiring no data download, and
it is arguably stronger than the real-data number would have been on its own —
it is what a model-validation function would demand before trusting a backtest.

#### The study found a bug in itself
`price()` could emit decimal odds **below 1.00** — impossible, and rejected by
`validation.min_odds` had it come from a real file. The simulator was producing
data its own pipeline would refuse. Caught by
`test_the_simulated_odds_form_a_coherent_book`, which holds the simulator to the
same checks `verify_raw_data.py` applies to a download. Fixed by capping any
single outcome at 0.92 before the margin is loaded.

#### Also
- README now leads with the simulation study; the real-data section keeps its
  **Awaiting regeneration** notice.
- `test_every_chart_the_readme_shows_exists_on_disk` exempts the real-data
  charts **while that notice stands**, and the exemption expires with it — so
  the choice is never between a red suite and committing charts from the wrong
  data. The simulation charts get no exemption: they need no download.
- `tests/test_simulation.py` — 10 tests. Suite now **194** (188 passed,
  6 skipped without raw data).

#### Still outstanding
Raw season files, if a reachable source turns up. `scripts/verify_raw_data.py`
checks any candidate in seconds. `notebooks/01_exploration.ipynb` has still
never been run.


### 2026-10-22 — The stamp lost in production; the numbers are unverified
Pulled yesterday's Phase 8 commit and found two faults, one of them mine from
an earlier session.

**1. Sample charts were published.** `make_charts.py` had fallen back to the
bundled 12-match sample, so `docs/charts/` held three ILLUSTRATIVE-stamped
charts and `numbers.json` said `"real_data": false` — while the README
referenced five charts, two of which the sample run had skipped. The front page
showed three meaningless charts and two broken images.

Nothing malfunctioned. The stamp appeared, the flag was recorded, and
`test_committed_charts_came_from_real_data` would have failed. **Every control
worked and the wrong thing happened anyway**, because all of them were
advisory.

- `make_charts.py` now **refuses** to write into `docs/charts/` from sample
  data, exits non-zero, and names the two legitimate alternatives (`--out`
  elsewhere, or `--allow-sample` by name).
- The check runs **before any computation**, from the filesystem alone — the
  first version ran after `gather()`, so a doomed run paid for a full
  walk-forward backtest before being refused.
- `test_sample_charts_cannot_be_written_into_the_published_folder` and
  `test_the_refusal_is_decided_before_any_work_is_done` pin both properties.
- The committed sample charts and `numbers.json` are deleted.

**2. This README's figures have never been reproduced here.** The companion
scorecard project was found today to have documented numbers taken from a run
made on *synthetic* data — wrong by a factor of three, with two conclusions
inverted. V8's headline figures (ROI −28.5%, 413 bets, CLV −0.67%) come from the
same kind of session and `data/raw/` has never held the season files on this
machine, so they are unverified by the same standard.

They stay in the README for now — deleting the results section would be worse —
but under an **Awaiting regeneration** notice, and
`test_unverified_numbers_carry_a_notice_and_verified_ones_do_not` enforces the
rule in both directions: no `numbers.json` means the notice is required; once
`numbers.json` exists the notice is false and must go, at which point the other
tests check every figure against it.

#### The rule this leaves behind
A control that *reports* a problem is not the same as a control that *prevents*
one, and the difference only matters when nobody is reading — which is exactly
when it fires. The required `history` argument in `blend_goals` and the finite
`INVALID_PENALTY` are the same move: make the wrong outcome unrepresentable
rather than merely visible.

#### Added: scripts/verify_raw_data.py
football-data.co.uk is unreachable from this machine's network (three clients
time out at TCP connect; the site itself is up and was fetched successfully
elsewhere). Mirrors exist, and they fail in three characteristic ways — two of
them silently:

1. the closing-odds columns are dropped → CLV cannot be computed
2. closing odds are a **copy of the opening price** → CLV reads zero forever,
   and the pipeline runs, charts and all, as though it had measured something
3. an HTML error page saved with a `.csv` name

The verifier checks columns, row count, goal and home-win rates, and — the
strongest check — whether the three prices form a **coherent book**: a real
bookmaker's odds imply probabilities summing to between 1.00 and 1.12. Prices
summing to 0.6 did not come from a book, whatever the column is called.

Two tests pin it, including the copied-closing case, which is the one that would
otherwise have produced a confident zero.

#### Next action
Download `E0_2324.csv` and `E0_2425.csv` into `data/raw/`, run
`python scripts/make_charts.py`, delete the notice, and commit the real charts.


### 2026-10-21 — Phase 8: the README, the charts, and a measure reported wrong
The last phase. Nothing new about football; everything about the gap between a
result and a reader.

- **`scripts/make_charts.py`** — regenerates all five README charts from one
  pipeline run and writes `docs/charts/numbers.json` (headline figures + the
  Phase 7 run fingerprint). `--sample`, `--only`, `--list` and `--out` for
  working on one chart at a time.
- **`README.md` rewritten** — leads with the honest result (in-sample +311% ROI
  vs walk-forward −28.5%), the CLV verdict with both measures and the reason
  they disagree, what the project shows and explicitly what it does not, and a
  table of all six bugs found with the lesson from each.
- **`tests/test_docs.py`** — 13 tests. The README may only link charts the
  script writes, must show every chart it writes, may carry no placeholders, and
  must quote the walk-forward ROI, yield, bet count, CLV interval and run
  fingerprint **as they appear in numbers.json**. Suite now **178 tests**
  (7 skip until charts are generated from real data).

#### A measure that was fixed in one place and not the other
`clv_by_band` aggregated `mean_clv_ratio` alone — the asymmetric measure the
18 Oct bug showed can read positive while the model is losing to the close. So
the sharpest diagnostic in the project ("does my claimed edge predict CLV?") was
only readable in the units already declared misleading. Now reports
`mean_clv_points` first, pinned by
`test_clv_bands_report_the_authoritative_measure`.

**The lesson, which generalises:** deciding which measure is authoritative is not
finished until every place that reports it has been changed. A correct verdict in
one function and the old measure in the one beside it is not a fixed bug — it is
a bug with a narrower blast radius.

#### Two charting faults that no test could have caught
ROI (+409%) and yield (+2.2%) shared one axis, flattening yield to an invisible
sliver — now two panels with independent scales. And `fig.autofmt_xdate()` calls
`subplots_adjust` internally, which switches constrained layout off: one chart's
subtitle printed through its title while four others were fine. Neither raises,
logs, or fails a unit test. Both were found by opening the PNG and looking.

#### The guard against the obvious mistake
Charts built on the 12-match sample are stamped **ILLUSTRATIVE** across their
face, `numbers.json` records `"real_data": false`, and a test refuses the commit.
The stamp lives in `finish()`, which every chart is saved through — the same
structural argument as the required `history` argument in `blend_goals`: make
the safe path the only path.

#### Still outstanding
`notebooks/01_exploration.ipynb` has still never been run.


### 2026-10-20 — Phase 7: hardening, CI and the run fingerprint
Phases 1–6 built a model. This session makes the claim *"these are the numbers
this code produces"* checkable by somebody who is not me.

- **`.github/workflows/tests.yml`** — CI on every push and pull request, Python
  3.11 **and** 3.12 with `fail-fast: false`. Each job installs the pinned
  requirements, runs `pytest -v --durations=10`, then runs the pipeline
  end-to-end on the bundled sample. The pipeline run is the part unit tests
  cannot replace: every test can pass while the stages no longer fit together.
- **`src/logging_setup.py`** — reports to stdout for a person, diagnostics to
  stderr (or `--log-file`) for debugging; loggers namespaced `v8.<module>`;
  `configure()` clears handlers so a second call does not double every line.
- **`run_fingerprint(config)`** — SHA-256 over the result-affecting config
  blocks, first 16 characters, printed by every run:
  `run fingerprint: c6c0bb194dcd9ba2`. Stable across processes, sensitive to
  `kelly_fraction` 0.25 → 0.20, invariant to key order and comments.
- **`tests/test_hardening.py`** — 17 tests. Total suite now **164 green**.

#### Why the fingerprint is the real deliverable
Before it, "it reproduces" was an assertion. Now, if a reported ROI and a rerun
disagree *while the fingerprints match*, the code is non-deterministic and that
is a bug. If the fingerprints differ, the settings differed and the comparison
was never valid. The question became answerable.

#### A bug in a test, which is still a bug
`test_requirements_cover_everything_the_code_imports` failed with a missing
dependency called **`average`**. The first version found imports by matching
`^\s*(from|import)\s+(\w+)` against raw lines, and picked the word up from the
comment `# Average ranks so tied scores...` in `src/metrics.py`. A regex cannot
tell code from prose; a parser can. It now walks `ast.Import` / `ast.ImportFrom`
nodes, which by construction only see real imports. The lesson is in the test's
docstring, because the next person to reach for a regex over source will be me.

#### Deliberately not hardened, and why
No network code in the pipeline (a run must not depend on football-data.co.uk
being up) and no cached fitted models (a cache is a place for stale parameters
to hide — precisely the bug class Phase 6 exists to catch).


### 2026-10-18 — Phase 6: walk-forward backtest, CLV and headline metrics
- `src/backtest.py` built: `assert_no_lookahead`, `walk_forward`,
  `backtest_report`. `src/clv.py` built: `clv_per_bet`, `bootstrap_interval`,
  `clv_summary`, `clv_by_band`, `clv_report`. `src/metrics.py` **completed**:
  `bootstrap_ci`, `max_drawdown`, `sharpe_like`, `performance_by_band`,
  `summarise`, `metrics_report`.
- **Every pipeline stage is now implemented.** The Phase 0 "declared but
  pending" smoke test is replaced by one asserting no stage still raises
  `NotImplementedError`.
- The lookahead guard runs on **every window of every run**, not just in tests,
  and raises rather than warns. Boundary is strict: a match on the same date as
  the window's first match is excluded, because two matches kicking off together
  cannot inform each other.

#### The honest numbers, and why they matter
| | Phase 5 (in-sample) | Phase 6 (walk-forward) |
|---|---|---|
| ROI | +311% | **-28.5%** |
| yield | +22.9% | **-4.2%** [-17.9%, +10.6%] |

The in-sample figure was a model that had seen the matches it was betting on.
The walk-forward figure is the forecast, and its interval includes zero — so
even the loss is not established on 413 bets. That is exactly the situation CLV
exists for.

#### A reporting bug the first run exposed
The two CLV measures **disagreed and both intervals excluded zero**:
```
odds ratio          +1.82%  [+0.81%, +3.00%]   positive
probability points  -0.67%  [-0.87%, -0.47%]   NEGATIVE
beat the close      47.46%                      below 50%
distribution: worst -23.17% | median -0.33% | best +108.37%
```
The report said "the confidence interval INCLUDES ZERO", which was false. Cause:
the odds ratio is asymmetric — a halving scores -50%, a doubling +100% — so its
mean was carried by one +108% price collapse while the model was losing to the
close. Probability points is now authoritative, disagreement is flagged
explicitly, and the flattering measure can never stand as the verdict.
`test_disagreeing_clv_measures_are_flagged_not_papered_over` pins it.

#### A test-fixture mistake worth recording
Two CLV tests failed because the fixture fixed the closing draw and away prices
while letting the home price vary — the closing book summed to between 0.84 and
1.03, so de-vigging it produced nonsense and the probability-points measure was
reading the broken fixture. `make_bets_with_closing_market` now builds coherent
books from probabilities that sum to 1.

- `config.yaml`: `backtest` block — `min_training_matches`,
  `refit_every_n_days`, `calibrate`, `min_calibration_matches`,
  `bootstrap_resamples`, `confidence_level`.
- `tests/test_backtest.py` (new): 30 tests — the guard passing a clean split,
  **catching a leak**, treating the same day as a leak, empty inputs; the engine
  running, never pricing without enough history, expanding windows,
  reproducibility, **bets struck at available odds not the close**; CLV ratio by
  hand, negative when the price moves against, probability points via the
  de-vigged close, intervals reported, refusing to claim an edge on noise,
  banding by claimed edge; bootstrap bracketing and widening with smaller
  samples, max drawdown by hand, Sharpe NaN without variation, the full
  pipeline end to end. Suite now 147 green.
- `scripts/run_pipeline.py`: Stage 6 runs the walk-forward, prints the backtest
  report, headline metrics with intervals and the CLV report, and writes
  `walkforward_bets.csv` and `clv_per_bet.csv`. `--verbose` prints one line per
  window. A `LookaheadError` is never swallowed.
- **Still outstanding:** `notebooks/01_exploration.ipynb` has not been run.
- **Next:** Phase 7 — hardening (pinned dependencies, CI, structured logging).
  Then Phase 8: the README with the real CLV chart and an honest statement of
  what the model does and does not show.

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
*(Superseded 29 Oct 2026: measured by `scripts/explore.py`, quoted in
methodology §1.5 and checked by tests. The table below stays as it was.)*
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
