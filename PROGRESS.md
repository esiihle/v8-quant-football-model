# Progress Log — V8 Quantitative Football Model

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus `docs/ROADMAP.md`)
> and you know exactly where the project stands. Newest entry at the top.

**Current phase:** Phase 1 — Data pipeline & validation (Oct 5 – Oct 18), started early
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
pytest                             # 18 tests (smoke + Phase 1 ingest), all green
python scripts/run_pipeline.py --no-save  # run the checks without writing output
# notebooks/01_exploration.ipynb   # open in VS Code, Run All
```

---

## Session log

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
