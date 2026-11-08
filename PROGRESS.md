# V8 Build Log

This is my record of how V8 was built: what I did on each working day, the
decisions I made, and the bugs I found. Newest entries are at the top. For the
results themselves, read the README; for plan against actual, read
`docs/ROADMAP.md`.

**Status:** complete, with two real-data results: the 2-season headline
(fingerprint `eddedac58a664dd8`) and the 13-season long run
(`9ed0f574b163c749`). 269 tests pass, and CI runs them plus a full pipeline run
on Python 3.11 and 3.12.
**Remaining:** the presentation on 18 December 2026.

---

## Decisions

- **Data source:** football-data.co.uk, English Premier League. It is the only
  free source I found that carries pre-match and closing odds in one file, which
  is what makes closing-line value measurable. (30 Sep)
- **De-vig method:** Shin rather than proportional, because bookmakers load
  their margin onto longshots. (14 Oct)
- **Data access:** football-data.co.uk is blocked on my network, so I rebuild its
  files from two GitHub mirrors with `scripts/rebuild_raw_from_mirrors.py`,
  which audits every score across the two. (23 Oct)
- **Headline vs long run:** two seasons for the headline, 13 seasons as a second
  result through a config that changes only the seasons. (27 Oct)

## How to run

```bash
pip install -r requirements.txt
pytest                                    # 269 tests
python scripts/run_pipeline.py --sample   # the bundled 12-match sample
python scripts/simulation_study.py        # the simulation study, no data needed

# real data
python scripts/rebuild_raw_from_mirrors.py --odds ../mirror-odds --shots ../mirror-shots
python scripts/verify_raw_data.py data/raw/*.csv
python scripts/run_pipeline.py            # full pipeline
python scripts/make_charts.py             # README charts and numbers.json
python scripts/explore.py                 # exploration.json
python scripts/run_notebook.py            # re-run the exploration notebook
```

---

## Log

### 2026-11-06: Study guide, slide deck, and closing out the plan
- Wrote `docs/STUDY_GUIDE.md`: every stage with the real code quoted, the maths
  derived, the trade-offs, the bug from that stage, and interview questions. A
  test fails if any quoted code block stops matching its source file, or any
  figure stops matching `numbers.json`. I checked the maths numerically as well
  as on paper, which corrected one claim: double Kelly gives roughly zero
  growth, not exactly zero.
- Writing the guide caught a stale docstring: `log_likelihood` still said it
  returned `+inf` for invalid parameters, which bug 2 had removed on 10 Oct.
- Built the slide deck.
- Added a plan-against-actual table to `docs/ROADMAP.md` and assessed the
  definition of done in `docs/OVERVIEW.md`. A test checks that every "actual"
  date in the roadmap is a heading in this log.

### 2026-11-05: The model against the market, on the same matches
- The walk-forward now keeps the model's out-of-sample probabilities for every
  match it prices, not only the bets. No existing figure changed.
- `src/benchmark.py` scores the model against the opening (Bet365) and closing
  (Pinnacle) markets, de-vigged with Shin, match by match on log-loss and Brier,
  with a paired bootstrap over matches.
- Result: the market is better on all four comparisons, on both datasets, with
  every interval clear of zero. On 13 seasons, log-loss against the opening line
  is +0.0248 [+0.0188, +0.0308]. The model's home-win calibration error is also
  above its noise floor there, while both markets are below theirs.
- A test I wrote was wrong, not the code. "Two equally good forecasters can't be
  told apart" failed on its one seed. A 95% interval rejects a true null 5% of
  the time by definition; across 20 seeds there were no false positives. The
  test now checks the false-positive rate.
- CI then failed on one figure: my Windows run gave an ECE noise floor of 0.0213
  and the README quoted 0.0212 from a Linux run. Percentages had always been
  checked within a tolerance, but my new check for plain decimals matched the
  exact text. It now uses a tolerance of 0.0002, with a test that a genuinely
  different figure still fails.

### 2026-10-29: The exploration, rebuilt as code
- The exploration notebook from 1 Oct had only ever been run on the 12-match
  sample, and those outputs had been on GitHub for four weeks ("58% home wins",
  "0-3 is 603% more common than Poisson"). Its text also had the Dixon-Coles
  correction backwards: a negative ρ raises 0-0 and 1-1 and lowers 1-0 and 0-1.
- Moved every measurement into `src/exploration.py`, with intervals (Wilson for
  shares, Fisher for correlations, Poisson for counts). `scripts/explore.py`
  writes `exploration.json` for either config, and methodology §1.5 quotes it.
- Finding: the answer to "are low scores Poisson?" depends on the null. Against
  a league-average Poisson, 13 seasons show 1-1 eleven per cent rarer than
  independence. Against each match's own expected goals it is +0.5%. The
  deficit came from mixing strong and weak teams. What survives is weak support
  for the correction (0-1 rarer, ρ negative in 10 of 13 seasons).
- Added a noise floor for calibration error: the ECE a perfectly calibrated
  forecaster scores on the same probabilities and number of matches. The
  2-season reliability diagram (0.061 on 341 matches) sits under its floor of
  about 0.080, so it cannot show miscalibration either way.
- CI failed on the first push: the notebook was committed without its outputs.
  I replaced the manual "Run All" with `scripts/run_notebook.py`, which runs a
  fresh kernel, saves only if every cell succeeds, and refuses to run without
  the real data. A test checks the saved notebook came from real data and the
  current config.

### 2026-10-27: Thirteen seasons as a second result
- Added `features.min_match_weight: 0.01`. Matches older than about 3.3 seasons
  carry 1% of the total weight but were most of the fitting time on 13 seasons,
  and they kept long-relegated teams in the model. The 13-season run went from
  67 minutes to 11, and all 37 headline figures were unchanged by it; only the
  fingerprint moved (`c6c0bb194dcd9ba2` to `eddedac58a664dd8`).
- Added `extends:` to configs, so `configs/epl_2012_2025.yaml` is 11 lines that
  change only the seasons. A test strips the seasons from both configs and
  requires the rest to be identical.
- Long run: 4,940 matches, 4,309 bets. CLV −1.22% is margin −1.24% plus market
  movement +0.02% [−0.05%, +0.09%]: the same verdict, much more tightly. ROI is
  −99.98% while the yield interval includes zero, because quarter Kelly on no
  edge compounds losses.
- One long-run bet sat exactly on the 2% minimum edge, and Windows and Linux put
  it on opposite sides (4,309 bets vs 4,310). The docs tests caught the
  difference before commit. Published figures come from my machine.
- Fixed three chart faults found by looking at the charts: links in subfolders
  were invisible to the docs tests, tick labels were rounded into different
  numbers (−0.25% printed as "−0.3%"), and the ruin curve needed a log axis.

### 2026-10-26: Calibrating the simulator
- Splitting the simulation study's CLV into margin and movement showed the
  random-model control getting +0.46 points of movement toward its bets. It had
  passed only because the margin hid it.
- Diagnosis: with no bet selection the movement was zero, so the measurement was
  sound. Two causes in the simulator: margin loaded proportionally but read with
  Shin (+0.50), and a closing line that corrected all of the opening error
  (+0.32).
- Fixes: `load_margin_shin`, the exact inverse of the pipeline's Shin de-vig;
  and a closing line that keeps part of the opening error. `--calibrate`
  chooses that share by matching a model-free statistic to 13 real seasons
  (real +0.12%, simulated +0.13% at 40%). The detection threshold is 4.9 points
  at every share the data allows.

### 2026-10-25: The real numbers, and two reporting bugs
- First run on the rebuilt real data. The old README figures (ROI −28.5%, 413
  bets, CLV −0.67%) did not reproduce, so I withdrew them. The real run: 433
  bets, ROI −82.60%, yield −15.84% [−28.57%, −3.47%].
- **Bug 10, two yields under one name.** The backtest printed profit ÷ turnover
  (−15.84%) and the headline printed the plain mean of per-bet returns
  (−10.08%), with the confidence interval attached to the second. On the right
  number the interval excludes zero, so the conclusion flipped. The old test had
  only checked that the low end was below the high end, which is true of either.
- **Bug 11, negative CLV blamed on the market.** The report said the market
  "moves AWAY from" the model's prices. CLV splits exactly into margin paid and
  market movement; the movement was +0.02% and the whole −1.87% was margin.
  `decompose_clv` now chooses the verdict from the movement.

### 2026-10-23: A headline that needs no download, and a data source
- `scripts/simulation_study.py`: simulated markets with known mispricings, run
  through the real pipeline. It detects about 5 probability points on 760
  matches and reports nothing on an efficient book. It also found a bug in
  itself: the simulator could emit decimal odds below 1.00.
- `scripts/rebuild_raw_from_mirrors.py`: rebuilds football-data.co.uk's files
  from two GitHub mirrors and refuses to write if they disagree on any score.
  All 13 seasons from 2012/13 passed.

### 2026-10-22: Sample charts published, and unverified numbers
- `make_charts.py` had fallen back to the 12-match sample, and the commit
  published three ILLUSTRATIVE-stamped charts and two broken images. Every
  control had fired and the wrong thing happened anyway, because they were all
  advisory. The script now refuses to write into `docs/charts/` from sample
  data, and decides that before doing any work.
- A companion project turned out to have published figures from a synthetic
  run. V8's README figures came from the same kind of session and had never been
  reproduced, so I put them under an "Awaiting regeneration" notice enforced by a
  test in both directions.
- `scripts/verify_raw_data.py` catches the two silent failures of a mirror:
  missing closing odds, and closing odds that copy the opening price (which
  would make CLV read zero).

### 2026-10-21: Phase 8, the README and charts
- `scripts/make_charts.py` builds every README chart from one pipeline run and
  writes `numbers.json`. `tests/test_docs.py` checks that the README quotes those
  numbers and links only the charts the script writes.
- `clv_by_band` reported only the odds ratio, the measure already shown to be
  misleading. It now reports probability points first. Deciding which measure is
  authoritative is not finished until every place that reports it has changed.
- Two chart faults no test could catch, found by opening the images: ROI and
  yield on one axis, and a title overlapped by its subtitle.

### 2026-10-20: Phase 7, hardening
- CI on every push, Python 3.11 and 3.12, running the suite and the full
  pipeline on the sample.
- `run_fingerprint`: a hash of every result-affecting setting, printed by every
  run, so "same fingerprint, same numbers" can be checked.
- A test bug: the dependency check reported a missing package called `average`,
  read from a comment by a regex. It now parses imports with `ast`.

### 2026-10-18: Phase 6, walk-forward backtest and CLV
- Walk-forward refit every 14 days, with a lookahead guard on every window that
  raises rather than warns, and a strict same-day boundary.
- First run: in-sample ROI +311% against walk-forward −28.5%. (These figures
  did not reproduce on the real data later; see 25 Oct.)
- **Bug 5:** the two CLV measures disagreed, with both intervals excluding zero,
  and the report said the interval "includes zero". The odds ratio's mean was
  carried by one +108% outlier. Probability points is now authoritative and
  disagreement is flagged.

### 2026-10-14: Phase 5, staking
- **Bug 4:** value was first tested against the de-vigged probability. The
  payout is at the actual odds, so expected value is positive only when the
  model's probability beats 1/odds. Now `edge` against the raw price decides
  bets and `disagreement` against the fair price is a diagnostic.
- Settled on Shin de-vigging, quarter Kelly, and caps of 5% per bet and 15% per
  day. Measured the zero-growth point of over-betting: near 2x Kelly, but not
  exactly 2x for discrete bets.

### 2026-10-13: Phase 4, calibration
- Platt and isotonic, fitted on the earlier half and judged on the later half.
  A test shows an in-sample fit flatters itself.
- Caught an interface bug while wiring: probability columns and result labels
  use different names, and the first version conflated them. Every metric now
  takes an explicit mapping.

### 2026-10-10: Phase 3, Dixon-Coles
- Penalised, time-weighted maximum likelihood from several starting points, with
  attack parameters constrained to sum to zero.
- **Bug 2:** returning `+inf` for invalid parameters broke L-BFGS-B's
  finite-difference gradient, and the optimiser stopped improving while still
  reporting success (parameter recovery r = 0.21). Invalid points now cost a
  large finite penalty.
- **Bug 3:** the ρ bound was sized from a theoretical 55-goal match, which
  pinned ρ at its lower edge. The bias did not shrink with more data, which is
  what gave it away. Bounds are now sized from observed goals.
- After both fixes, recovery on simulated data: attack r = 0.99, defence
  r = 0.98, ρ −0.084 against a true −0.080.

### 2026-10-09: Phase 2, features
- Time decay (180-day half-life), Bayesian shrinkage on the effective sample
  size, and an expected-goals proxy from shots on target.
- **Bug 1:** the xG conversion rate was fitted over the whole file, so August
  matches used May's data. `blend_goals` now takes `history` as a required
  argument.

### 2026-10-02: Phase 1, validation
- A canonical table with a fixed column contract, strict validation, results
  recomputed from goals, and a guard that stops the run if more than 5% of rows
  would be dropped.

### 2026-10-01: Exploration notebook
- First version of `notebooks/01_exploration.ipynb`. (It was not run on real
  data until 29 Oct.)

### 2026-09-30: Real data and schedule
- Chose football-data.co.uk and re-baselined the roadmap from 28 Sep, with the
  presentation on 18 December 2026.

### 2026-09-28: Phase 0, scaffolding
- A runnable skeleton: config, loader, stubs for every stage, a pipeline runner
  that reports each stage as done or pending, and the first smoke tests.

### 2026-09-01: Inception
- Created the repository with the overview, roadmap, methodology outline and
  licence.
