# RSNA Knee Abnormality Detection

Solo entry for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
predict per-study probabilities for 12 binary knee-MRI findings from multi-series DICOM studies,
scored by macro-averaged ROC AUC. Targets both the main leaderboard and the Efficiency Track.

Full strategy lives in the plan document (not tracked in this repo); empirical findings from actually
running things (timing numbers, data quirks, model behavior) go in `NOTES.md` as they're discovered,
kept separate from the plan so the plan stays a stable strategy reference. Status: **Phase 1
(pipeline-validation baseline) infrastructure built and tested — DICOM series/slice selection,
laterality resolution, pixel decode/normalize, a `torch.utils.data.Dataset` over raw DICOMs, a
single-backbone mean-pool model, NaN-masked BCE loss, macro AUC, and a submission writer with a 0.5
fallback, all covered by tests (46 passing). The full offline import → decode → inference →
`submission.csv` round trip has been validated on real Kaggle infrastructure (internet off) — see
`notebooks/knee-phase1-smoke-test.ipynb`. Not yet built: the actual training loop/CV harness
(`train.py` currently only has the loss function), and Phase 2's `prep.py` preprocessing pipeline.**

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
  dicom.py           [done] header scan, series/slice selection, laterality, decode/normalize
  prep.py            study -> compact volume artifact (writer) -- Phase 2, not started
  dataset.py         [done, Phase 1 shape] torch Dataset directly over raw DICOMs;
                      will read prepped artifacts once prep.py exists
  model.py           [done, Phase 1 shape] single backbone + mean-pool over slices;
                      slice/series attention is a later Phase 5 upgrade
  train.py           [loss function done] masked BCE; the fold/CV/checkpointing loop is next
  infer.py           [done] submission writer, byte-for-byte header, 0.5 fallback
  reports.py         [done, EN/ES only] lexical label rules; LLM calibration is Phase 3
  metrics.py         [done] macro AUC, per-label AUC (NaN-safe); bootstrap CI not yet added
notebooks/           thin Kaggle notebooks: import knee, call one function
  knee-phase1-smoke-test.ipynb  validated end-to-end round trip on real Kaggle infra (internet off)
tests/               pytest, runs locally on a small sample, no GPU needed
data/sample/         studies pulled via Kaggle API for local dev (gitignored)
results/             experiment tracking, see below
```

## Experiment tracking

Three CSVs under `results/`, appended by `train.py`, never hand-edited:

- `baseline.csv` — the first working (lexical-label) submission. Written once; every later number
  is a delta against this row.
- `experiments.csv` — every training run: config, fold set, seed, per-label AUC, macro AUC,
  paired delta vs. current best, train/inference time, whether it was promoted.
- `hall_of_fame.csv` — only runs that beat the previous best on a paired per-study delta across
  frozen folds and held up over 2-3 seeds. The final ensemble is built exclusively from these rows.

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
