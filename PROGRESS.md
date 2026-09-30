# Progress Log — V8 Quantitative Football Model

> **Purpose.** A living status tracker, updated at the end of each working
> session. This is the fastest way to resume: read this file (plus `docs/ROADMAP.md`)
> and you know exactly where the project stands. Newest entry at the top.

**Current phase:** Phase 0 complete (Sep 28 – Oct 4); Phase 1 opens 5 Oct
**Overall status:** 🟢 Pipeline runs end-to-end on **real** EPL data; tests green (6).
**Next action:** Implement strict schema validation in `src/ingest.py` (the
`PHASE 1` marker) — types, date parsing, duplicate fixtures, goal/result checks —
and a data-quality report.
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
pytest                             # 6 smoke tests, all green
```

---

## Session log

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
