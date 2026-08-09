# RSNA Knee Abnormality Detection

Solo entry for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
predict per-study probabilities for 12 binary knee-MRI findings from multi-series DICOM studies,
scored by macro-averaged ROC AUC. Targets both the main leaderboard and the Efficiency Track.

Full strategy lives in the plan document (not tracked in this repo); empirical findings from actually
running things (timing numbers, data quirks, model behavior) go in `NOTES.md` as they're discovered,
kept separate from the plan so the plan stays a stable strategy reference. Status: **Phase 1 baseline
complete — trained, submitted, and scored on the real leaderboard; Phase 2 (preprocessing) started.**

Phase 1: `efficientnet_b0`, single sagittal fluid-sensitive series, 16 slices, trained on lexical
(keyword-derived) labels for the 4 labels with any coverage (ACL, Medial Meniscus, Effusion,
Baker's — the other 8 have no lexical rule yet). 5-fold CV, pooled OOF macro AUC 0.7985 on those 4
labels; real competition leaderboard score **0.558** (the other 8 labels score ~0.5 each since they
were never trained, diluting the 12-label macro average — see `NOTES.md` for the full breakdown,
including the ~0.674 real-world AUC backed out for the 4 trained labels vs. their 0.80 lexical-label
validation average). Submission notebooks live in `notebooks/phase1-train/` (training) and
`notebooks/phase1-submit/` (the actual scored submission).

Phase 2: a corpus-wide laterality/slice-count census (`notebooks/phase2-laterality-census/`, results
in `results/laterality_census.csv`) found laterality resolves for only **51.1%** of the real ~4,407
studies via DICOM tags (`ImageLaterality`/`Laterality`/description string-match) — well below the
~75% a small local sample suggested, and confirmed structural (not a sampling gap) via a
multi-instance follow-up check (`notebooks/phase2-laterality-verify/`,
`results/laterality_verify_sample.csv`). `src/knee/prep.py` has the per-series normalization,
laterality-mirroring, and JPEG/`.npz` storage building blocks with local tests; the full-corpus prep
run and the orchestrating `prep_study` function are the next step — see `NOTES.md` (2026-08-09) for
the complete numbers and reasoning.

Core library (`src/knee/`) covered by tests (71 passing): DICOM series/slice selection, laterality
resolution (per-header and per-study, with real-world tag-value normalization), pixel decode/
normalize, per-series normalization and laterality mirroring, JPEG-in-`.npz` study storage, dataset
classes (including an in-memory decode cache), model, NaN-masked BCE loss, fold assignment,
training/eval loops, experiment logging, macro AUC, and a submission writer with a 0.5 fallback.

Only 58 of 4,407 training studies carry gold rubric labels (verified directly, not the "a few
hundred" first assumed) — this is effectively a weak-supervision competition, not a conventional
supervised one. See the plan document's "Label hierarchy" section and `NOTES.md` for details.

## Compute

Free tier only — Kaggle CPU/GPU notebooks for all preprocessing and training, Colab free for
ad-hoc experiments. Local machine is code-authoring and unit-testing only (tests run on a small
sample of studies pulled via the Kaggle API), never full training.

## Layout

Target layout (full plan); items marked `[done]` exist today, everything else is Phase 2+:

```
src/knee/            package pushed to Kaggle as a private dataset, imported by thin notebooks
  dicom.py           [done] header scan, series/slice selection, per-header and per-study
                      laterality resolution + census, decode/normalize
  prep.py            [in progress] normalize_series, mirror_to_canonical, JPEG-in-.npz study
                      storage done + tested; prep_study orchestrator and the full-corpus
                      run are next (see NOTES.md 2026-08-09)
  dataset.py         [done, Phase 1 shape] torch Dataset directly over raw DICOMs;
                      will read prepped artifacts once prep_study exists
  model.py           [done, Phase 1 shape] single backbone + mean-pool over slices;
                      slice/series attention is a later Phase 5 upgrade
  train.py           [done] masked BCE, fold assignment, one epoch, evaluate, experiment logging --
                      used for the real Phase 1 5-fold training run (notebooks/phase1-train/)
  infer.py           [done] submission writer, byte-for-byte header, 0.5 fallback
  reports.py         [done, EN/ES only] lexical label rules; LLM calibration is Phase 3
  metrics.py         [done] macro AUC, per-label AUC (NaN-safe); bootstrap CI not yet added
notebooks/           thin Kaggle notebooks: import knee, call one function
  knee-phase1-smoke-test.ipynb  validated end-to-end round trip on real Kaggle infra (internet off)
  phase1-train/        real 5-fold training run -- lexical labels, efficientnet_b0, OOF macro AUC 0.7985
  phase1-submit/        the actual scored submission notebook (offline, loads the trained checkpoint)
  phase2-laterality-census/  corpus-wide laterality/slice-count census, CPU-only
  phase2-laterality-verify/  multi-instance follow-up confirming the census's unknown rate is real
tests/               pytest, runs locally on a small sample, no GPU needed
data/sample/         studies pulled via Kaggle API for local dev (gitignored)
checkpoints/         trained model weights + OOF arrays (gitignored, large binaries)
results/             experiment tracking + Phase 2 census outputs, see below
```

## Experiment tracking

Three CSVs under `results/`, appended by `train.py`, never hand-edited:

- `baseline.csv` — the first working (lexical-label) submission. Written once; every later number
  is a delta against this row.
- `experiments.csv` — every training run: config, fold set, seed, per-label AUC, macro AUC,
  paired delta vs. current best, train/inference time, whether it was promoted.
- `hall_of_fame.csv` — only runs that beat the previous best on a paired per-study delta across
  frozen folds and held up over 2-3 seeds. The final ensemble is built exclusively from these rows.

Plus two one-off Phase 2 census outputs (not part of the experiment-tracking discipline above, not
appended to): `laterality_census.csv` (per-study route/side/slice-counts across the real corpus) and
`laterality_verify_sample.csv` (the multi-instance follow-up check).

## Running tests

```
pip install -r requirements-dev.txt
pytest tests/
```

Some tests read real competition data from `data/sample/` (gitignored) and are skipped
automatically if that data hasn't been pulled yet.

## Kaggle setup

```
pip install --user kaggle
```

Create an API token at kaggle.com/settings. The Kaggle CLI (2.2.4) uses a single bare token, not
the older `username`+`key` JSON — save it as `~/.kaggle/access_token` (mode 600), not
`~/.kaggle/kaggle.json`:

```
mkdir -p ~/.kaggle && mv <downloaded-token-file> ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
kaggle competitions list -s knee   # sanity check
```

Competition rules must be accepted on kaggle.com before any data download works.
