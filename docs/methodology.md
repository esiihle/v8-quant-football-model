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

### 1.1 Data source (settled 30 Sep 2026)

**football-data.co.uk**, one CSV per league-season, downloaded from
`https://www.football-data.co.uk/mmz4281/{season}/{league}.csv`. It was chosen
over alternatives for one decisive reason: it publishes **both pre-match and
closing odds** in the same file. Closing odds are what make closing-line value
measurable, and CLV is the headline claim of this project — a source without
them would quietly remove the project's main result.

**Coverage at present:** English Premier League (`E0`), seasons 2023/24 and
2024/25. One league first, to prove the pipeline end to end before scaling out.

### 1.2 Fields we rely on

| Group | Columns | Used for |
|---|---|---|
| Identity | `Div`, `Date`, `HomeTeam`, `AwayTeam` | joining, ordering, time-decay weights |
| Result | `FTHG`, `FTAG`, `FTR` | the Dixon-Coles likelihood (§3) |
| Shots | `HS`, `AS`, `HST`, `AST` | xG proxy (§2.3) |
| Odds (bet into) | `B365H`, `B365D`, `B365A` | value detection and staking (§5) |
| Closing odds | `PSCH`, `PSCD`, `PSCA` | closing-line value (§6.3) |

Everything else in the file (half-time scores, cards, corners, referee, the
other bookmakers, Asian handicap lines) is carried through untouched for now and
either used or dropped explicitly in Phase 1 — never dropped by accident.

### 1.3 Known quirks of the source

Observed when loading the real files, and handled in `src/ingest.py`:

- **Dates are day-first**, and the year is 2-digit in some seasons and 4-digit
  in others, so dates are parsed with `dayfirst=True, format="mixed"`.
- **Trailing empty columns and rows** appear in several files (spreadsheet
  padding) and are dropped on load.
- **Column sets differ between seasons** — a column present in one season may be
  absent in another, which shows up as a block of nulls exactly the size of one
  season. The shape report surfaces this rather than hiding it.
- **Occasional missing odds** on individual matches; these rows are simply not
  bettable and will be excluded at the staking stage, not silently imputed.
- Encoding is not always UTF-8, so the loader falls back to latin-1.

### 1.4 The canonical table (Phase 1, 2 Oct 2026)

Raw files are never read by anything downstream. Phase 1 validates and cleans
them into one canonical table (`data/processed/matches.csv`), defined in
`src/canonical.py`:

`MatchID, MatchDate, League, Season, HomeTeam, AwayTeam, HomeGoals, AwayGoals,
Result, HomeShots, AwayShots, HomeShotsOT, AwayShotsOT, OddsH, OddsD, OddsA,
CloseH, CloseD, CloseA`

Rules applied, each counted in the data-quality report:

| Problem | What happens | Why |
|---|---|---|
| Required column missing | run stops | nothing downstream can be trusted |
| Date won't parse | row dropped | a match with no date can't be ordered in time |
| Goals impossible (<0 or >15) | row dropped | a data error, not a result |
| `FTR` disagrees with the goals | **result recomputed**, mismatch counted | goals are the primary record |
| Team name has stray whitespace | trimmed | otherwise one team becomes two |
| Odds outside 1.01–1000 | price voided (NaN), row kept | the match still trains the model; it just isn't bettable |
| Same date + same two teams | duplicate dropped | otherwise that match is double-counted in the likelihood |
| More than 5% of rows unusable | run stops | quiet degradation is worse than a loud failure |

The table is sorted by date, because Phase 6 walks forward through time and an
out-of-order frame is the easiest possible way to leak the future into a
backtest. `MatchID` is built from league, season, date and both teams, so it is
stable across re-downloads and lets a bet in the backtest be traced to a fixture.

Still to be settled: the exact de-vig method (proportional vs. Shin), before
Phase 5 staking.

## 2. Feature engineering  *(built 9 Oct 2026)*

One row of model inputs per match: each team's attack and defence strength,
estimated from recent form, shrunk toward the league average, using goals
blended with a shots-based expected-goals proxy.

### 2.0 The rule that governs the whole stage

**A match's features may use only matches played strictly before it.**

Every estimate takes an `as_of` date and filters on it; the filter lives in one
place (`time_weights`) rather than being trusted to each caller. A model that
has seen the second half of the season while rating a team in October will look
excellent in a backtest and lose money in practice.

This is not theoretical. The first version of this module fitted the xG
conversion rate over the whole file, so an August match was scored partly on
matches played in May. `test_features_do_not_use_the_future` caught it, and
`blend_goals` now takes `history` as a required argument so the mistake is hard
to repeat.

### 2.1 Attack / defence strengths

Relative to the league, where 1.0 is exactly average: attack 1.25 means a quarter
more goals than a typical team; defence 0.80 means a fifth fewer conceded (lower
is better). Home and away appearances are pooled, because home advantage is a
separate parameter fitted in Phase 3 — folding it in here would count it twice.

### 2.2 Bayesian shrinkage

$$\hat{\theta}_i^{\text{shrunk}} = w_i\,\hat{\theta}_i + (1 - w_i)\,\bar{\theta},
\qquad w_i = \frac{n_i}{n_i + k}$$

with $k$ = `features.shrinkage_k` (currently 8). A team with 0 matches is given
the prior exactly; with 8 it sits halfway; with 38 it is 83% its own record.
Trust is earned gradually, with no threshold to argue about.

$n_i$ is the **effective** sample size — the sum of time-decay weights, not a row
count. Ten matches from last season carry far less than ten matches' worth of
evidence, and the shrinkage should know that.

This is the direct analogue of a thin-file adjustment or a low-default-portfolio
treatment in credit risk: same formula, same argument, different quantity.

### 2.3 Expected-goals proxy

Goals are the truth but are rare and noisy — a 1-0 win and a 1-0 loss look
identical in the table and nothing alike on the pitch. Shots on target are far
more plentiful. The feature blends them:

    attack_goals = (1 - w) * goals + w * (shots_on_target * conversion_rate)

with $w$ = `features.xg_weight` (currently 0.5). The conversion rate is measured
**from past matches only**, separately for home and away sides, so it reflects
this league and era rather than a constant from a paper — and so that home
advantage is not smuggled in a second time.

Where shot data is absent (the bundled sample, some older seasons), the feature
falls back to actual goals and says so in the run report rather than inventing a
figure.

### 2.4 Time decay

$$w(\Delta t) = 0.5^{\,\Delta t / H}$$

with $H$ = `features.half_life_days` (currently 180, about half a season). A
match played today counts 1.000, one a season old 0.500, two seasons old 0.250.
Anything dated on or after `as_of` gets weight 0 — that is where the lookahead
guard lives.

### 2.5 Warm-up rows

A match played before either team has `features.min_prior_matches` of history is
flagged `HasSufficientHistory = False`, not deleted. Phase 3 decides whether to
train on them. One place decides what enters the model.

A team never seen before (a promoted side) is given the prior, not a null: an
unknown team is an average team until it shows otherwise.

### 2.6 Output

`data/processed/features.csv`, one row per match, in `FEATURE_COLUMNS` order,
carrying the strengths, the history counts, the outcome, and both odds sets for
Phases 5 and 6.

## 3. The Dixon-Coles model  *(built 10 Oct 2026)*

### 3.1 The model

Each team carries an attack parameter $\alpha_i$ and a defence parameter
$\beta_i$; there is one global home advantage $\gamma$ and one dependence
parameter $\rho$. For a fixture (home $i$, away $j$):

$$\lambda = \exp(\alpha_i + \beta_j + \gamma), \qquad
  \mu     = \exp(\alpha_j + \beta_i)$$

$$P(X = x, Y = y) = \tau_{\rho}(x, y)\;
  \frac{\lambda^{x} e^{-\lambda}}{x!}\,\frac{\mu^{y} e^{-\mu}}{y!}$$

Parameters live on a log scale, so $e^{\alpha}$ reads as a multiplier: attack
1.30 means a team scores 30% more than league average.

### 3.2 The low-score correction

Independent Poisson is almost right, and measurably wrong in one place — the
four lowest scorelines. Real football produces more 0-0 and 1-1 draws and fewer
1-0 and 0-1 results than independence predicts, which §2 of
`notebooks/01_exploration.ipynb` measures on our own data before any model is
fitted. Dixon and Coles (1997) correct it with one parameter:

$$\tau_{\rho}(x,y) = \begin{cases}
  1 - \lambda\mu\rho & x = 0, y = 0 \\
  1 + \lambda\rho    & x = 0, y = 1 \\
  1 + \mu\rho        & x = 1, y = 0 \\
  1 - \rho           & x = 1, y = 1 \\
  1                  & \text{otherwise}
\end{cases}$$

A negative $\rho$ raises 0-0 and 1-1 while lowering 1-0 and 0-1 — the direction
the data asks for. The correction is **mass-preserving**: the four adjustments
cancel exactly, so the distribution still sums to 1. That is a property of the
construction, not of our code, and
`test_tau_correction_preserves_total_probability` checks it to 1e-12.

Validity requires
$\max(-1/\lambda, -1/\mu) \le \rho \le \min(1/(\lambda\mu), 1)$; outside that
range the model assigns a negative probability to a scoreline.

### 3.3 Fitting: penalised, time-weighted maximum likelihood

$$\mathcal{L} = \sum_{m} \phi(t_m)\,\log P(X = x_m, Y = y_m)
  \; - \; \kappa \sum_i (\alpha_i^2 + \beta_i^2)$$

with $\phi$ the exponential time-decay weight from §2.4 and $\kappa$ =
`model.penalty`.

**Identifiability.** $\lambda$ and $\mu$ depend only on sums such as
$\alpha_i + \beta_j$, so adding a constant to every attack and subtracting it
from every defence changes nothing. One constraint fixes it: the attack
parameters are forced to sum to zero, and the optimiser works with $n-1$ free
attacks.

**The penalty is Phase 2's shrinkage, restated.** An L2 term pulls a team's
parameters toward 0, which on this scale is exactly league average, and it binds
hardest where there is least evidence. Same argument as the Bayesian shrinkage in
§2.2, expressed in the language of an optimiser, and the direct analogue of a
ridge-penalised PD model in credit risk.

**Phase 2 seeds the optimiser.** The shrunk strengths are converted to this log
scale and used as the first starting point. A cheap, stable estimate is a far
better start than a flat guess; the phases are complementary, not competing.

**Multi-start.** The fit runs from `model.n_starts` starting points and reports
the spread of the objective across them. Agreement is the evidence that the
optimum is global rather than lucky; disagreement raises a warning in the run
report, because a fit nobody can reproduce is not a fit.

### 3.4 Two bugs worth recording

Both were invisible in the output and both were caught by tests, not by reading
the numbers. The roadmap predicted this phase would be where numerical care was
needed, and it was right.

**An infinite objective breaks the optimiser.** The first version returned
`+inf` for an invalid parameter set. L-BFGS-B estimates its gradient by finite
differences, so a single infinite probe gives `inf - inf = nan`, the gradient
becomes unusable, and the optimiser stops improving *while still reporting
success*. Parameter recovery came back at $r = 0.21$ and nothing raised an
error. An invalid point now costs a large but finite amount that grows with the
size of the violation, so there is always a route back downhill.

**A "safe" bound became a binding constraint.** The $\rho$ box was sized from the
largest $\lambda$ the strength box could theoretically produce —
$e^{1.5+1.5+1.0} \approx 55$ goals a match. That squeezed the box to
$[-0.018,\ 0.0003]$, and $\rho$ sat pinned on the lower edge while the fit
reported success and the starts agreed with each other. The tell was that **the
bias did not shrink as the sample grew**: noise averages out, a binding
constraint does not. The box is now sized from the goals actually observed, the
exact per-fixture constraint is left to the smooth penalty, and the fit reports
`rho_at_bound` so the same failure would announce itself.

### 3.5 From parameters to market probabilities

The fit gives a full score matrix $P(X=x, Y=y)$ truncated at `model.max_goals`
and renormalised for the lost tail. Every market is then a sum over cells:

| Market | Cells |
|---|---|
| home win / draw / away win | $x>y$ / $x=y$ / $x<y$ |
| over / under 2.5 goals | $x+y\ge3$ / $x+y\le2$ |
| both teams to score | $x\ge1 \wedge y\ge1$ |
| correct score | the cell itself |

This is the real payoff of modelling the whole scoreline distribution rather than
the result alone: one fit prices every market, and they are automatically
consistent with one another.

A fixture involving a team absent from the training data is priced at
league-average parameters rather than rejected — a promoted side must be
priceable on debut — and flagged `KnownTeams = False`, so Phase 5 can decline to
stake on a fixture the model does not really know.

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
