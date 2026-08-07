# RSNA Knee Abnormality Detection

Solo entry for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
predict per-study probabilities for 12 binary knee-MRI findings from multi-series DICOM studies,
scored by macro-averaged ROC AUC. Targets both the main leaderboard and the Efficiency Track.

Full strategy lives in the plan document (not tracked in this repo); empirical findings from actually
running things (timing numbers, data quirks, model behavior) go in `NOTES.md` as they're discovered,
kept separate from the plan so the plan stays a stable strategy reference. Status: **data
infrastructure under test — `dicom.py` (series/slice selection, laterality resolution),
`metrics.py` (macro AUC), `infer.py` (submission writer with 0.5 fallback), and `reports.py`
(lexical label rules for English/Spanish) are implemented and covered by tests. No model or training
code yet.**

Only 58 of 4,407 training studies carry gold rubric labels (verified directly, not the "a few
hundred" first assumed) — this is effectively a weak-supervision competition, not a conventional
supervised one. See the plan document's "Label hierarchy" section and `NOTES.md` for details.

## Compute

Free tier only — Kaggle CPU/GPU notebooks for all preprocessing and training, Colab free for
ad-hoc experiments. Local machine is code-authoring and unit-testing only (tests run on a small
sample of studies pulled via the Kaggle API), never full training.

## Layout

```
src/knee/            package pushed to Kaggle as a private dataset, imported by thin notebooks
  dicom.py           header scan, series/slice selection, decode, normalize
  prep.py            study -> compact volume artifact (writer)
  dataset.py         torch Dataset over prepped artifacts
  model.py           backbone + slice attention + series attention + 12 heads
  train.py           folds, loss, AMP, checkpointing
  infer.py           single entry point used by both submission notebooks
  reports.py         LLM prompting, JSON parse, calibration to soft targets
  metrics.py         macro AUC, per-label AUC, bootstrap CI
notebooks/           thin Kaggle notebooks: import knee, call one function
tests/               pytest, runs locally on a small sample, no GPU needed
data/sample/         ~20 studies pulled via Kaggle API for local dev (gitignored)
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
