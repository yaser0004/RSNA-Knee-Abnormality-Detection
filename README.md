# RSNA Knee Abnormality Detection

Solo entry for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
predict per-study probabilities for 12 binary knee-MRI findings from multi-series DICOM studies,
scored by macro-averaged ROC AUC. Targets both the main leaderboard and the Efficiency Track.

Full strategy lives in the plan document (not tracked in this repo); empirical findings from actually
running things (timing numbers, data quirks, model behavior) go in `NOTES.md` as they're discovered,
kept separate from the plan so the plan stays a stable strategy reference. Status: **Phase 1 baseline
complete — trained, submitted, and scored on the real leaderboard; Phase 2 (preprocessing) gate
closed — all 4,407 studies prepped and verified.**

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
`results/laterality_verify_sample.csv`).

The prep pipeline is written, tested, and validated on Kaggle: `prep_study` turns one study into up
to 4 series × 24 slices at 256px (per-series normalized, padded to square, JPEG q=92 in a single
`.npz`, ~1.4 MB/study), and `PreppedStudyDataset` reads those artifacts back into the same tensor
contract `KneeStudyDataset` produces — **51 ms/study against 1,912 ms for the raw-DICOM path it
replaces, a 37× speedup**. `prep_study` is pure (returns data, writes nothing) so Phase 6's
submission notebook can run the identical path over the test set instead of growing a second copy
inside `infer.py`, and artifacts store *unmirrored* pixels plus the resolved `side`/`route`, with
mirroring applied at load time so they stay neutral to a laterality resolution Phase 3 may still
improve.

**The Phase 2 gate is closed: all 4,407 studies are prepped** (four parallel shard kernels,
`notebooks/phase2-prep-shard0..3/`, 6.2 GB), verified by a consumer kernel — 0 missing, 0 duplicated,
0 failed, 0 unreadable, all 58 gold studies present, and a laterality route mix matching the
published census to the tenth of a percent. A 108-study pilot (stride-sampled across the corpus plus
every gold study) cleared the four blocking gates first: the artifact hop works via `kernel_sources`
with no files lost, decode coverage is clean (432/432 series one transfer syntax, all MONOCHROME2,
zero decode failures), and size held to budget. The pilot also found **7.9% of series are non-square,
worst ratio 2:1**, so slices are letterboxed to square before the resize rather than squashed. Loader
latency misses the plan's <20 ms/study target at the full training configuration (39–51 ms for 96
slices, ~85% irreducible JPEG decode) but clears it at the efficiency-track shape; loading is no
longer the bottleneck either way.

The visual check (`notebooks/phase2-visual-check/`) caught a real bug worth knowing about: laterality
mirroring was being applied to *every* series, but a sagittal image's horizontal axis is
anterior–posterior, not medial–lateral, so flipping one mirrors the knee front-to-back. Only
Axial/Coronal series are mirrored now (`mirrors_in_plane`). Because artifacts store unmirrored pixels
by design, this was a loader-only fix — no re-prep. Sagittal laterality is still not canonicalized
(it would require reversing slice order, a Phase 4/5 modelling decision). Full measured numbers and
the reasoning behind each decision are in `NOTES.md` (2026-08-09 and 2026-08-10).

Core library (`src/knee/`) covered by tests (96; 94 run without a GPU-capable box — the two
`test_model.py` cases instantiate a backbone): DICOM series/slice selection, laterality
resolution (per-header and per-study, with real-world tag-value normalization), pixel decode/
normalize, per-series normalization, pad-to-square and laterality mirroring, JPEG-in-`.npz` study
storage with selective decode, the `prep_study` orchestrator (including its skip-unusable-series and
decode-failure recording), dataset classes for both raw and prepped studies (plus an in-memory
decode cache), model, NaN-masked BCE loss, fold assignment, training/eval loops, experiment logging,
macro AUC, and a submission writer with a 0.5 fallback.

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
                      laterality resolution + census, decode/normalize; also owns
                      StudyDecodeError and the slice-header read both dataset.py and
                      prep.py need (kept here to avoid a circular import)
  prep.py            [done] normalize_series, pad-to-square, mirror_to_canonical +
                      mirrors_in_plane (only Axial/Coronal flip), JPEG-in-.npz storage with
                      selective decode, and the pure prep_study orchestrator
  dataset.py         [done] KneeStudyDataset over raw DICOMs (Phase 1 shape) and
                      PreppedStudyDataset over the Phase 2 .npz artifacts, same contract
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
  phase2-prep/         the pilot: PILOT_N studies stride-sampled across the corpus plus every
                        gold study; carries the four blocking gates
  phase2-prep-shard0..3/  the full-corpus run, one kernel per shard index range -- identical
                        code, differing only in SHARD_INDEX, so a lost shard re-runs alone
  phase2-prep-consume/  consumer kernel that proves the artifact hop and re-counts the corpus
  phase2-visual-check/  8 studies x 4 series grid, mirrored to canonical -- the human check
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
