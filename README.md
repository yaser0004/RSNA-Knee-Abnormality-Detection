# RSNA Knee Abnormality Detection

Solo entry for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
predict per-study probabilities for 12 binary knee-MRI findings from multi-series DICOM studies,
scored by macro-averaged ROC AUC. Targets both the main leaderboard and the Efficiency Track.

Full strategy lives in the plan document (not tracked in this repo); empirical findings from actually
running things (timing numbers, data quirks, model behavior) go in `NOTES.md` as they're discovered,
kept separate from the plan so the plan stays a stable strategy reference. Status: **Phase 1 baseline
complete — trained, submitted, and scored on the real leaderboard; Phase 2 (preprocessing) gate
closed — all 4,407 studies prepped and verified; Phase 3 gate closed — Qwen3-4B-Instruct won the
bake-off (gold macro AUC 0.8613 vs 0.625 lexical anchor) and labeled the full corpus
(`results/pseudo_labels_qwen3_4b.csv`, 52,884/52,884 answered, all consumer gates passed);
**Phase 4 run 1 complete** — all 12 labels trained on those pseudo-labels, pooled OOF 0.7779 and
gold transfer 0.7252 with the 58 gold studies excluded from both sides of every split, and
**submitted** (`notebooks/phase4-submit/`, version 1) — awaiting the rerun score, to be read against
the bands pre-registered in `NOTES.md`. Next: Phase 5 experiment B.**

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

Phase 3 attacks the actual bottleneck: the LB score decomposes as `0.558 = (4 × 0.674 + 8 × 0.500) / 12`,
because 8 of the 12 label columns have no training target at all (`reports.py` had lexical rules for
only 4). No image-side work can move a column with no label, so an open-weight LLM reads all 4,407
radiology reports against the host's own rubric to produce soft pseudo-labels for all 12 findings.
Competition rules §2.4.b.1 forbids sending report text to hosted APIs, so the model runs *inside* a
Kaggle GPU notebook; the rubric is transcribed verbatim in `notebooks/phase3-labels/rubric_733343.md`.

The prompt asks one single-token yes/no question per label, carrying **only that label's rubric
criterion** (not the whole 12-definition rubric), and reads the score from the Yes/No logits at the
single answer position — no JSON parsing, and `null`/silent stays NaN rather than becoming a negative.
An earlier version prepended the entire rubric to all 12 questions, which cost ~1,495 tokens/prompt
and appeared to require vLLM's prefix caching; that dependency then failed twice on a CUDA-variant
mismatch. Per-criterion prompts measure **389 tokens** (median report), need no prefix cache, and run
on the `transformers` already in the Kaggle image — no install step to fail. See `NOTES.md`
(2026-08-17) for the full post-mortem and the measured token table.

Scoring lives behind a `generate_fn` seam (`score_from_top_logprobs`, `score_report_with_llm`) and is
tested against a fake, which is why swapping the entire inference engine changed neither those
functions nor their tests. `notebooks/phase3-probe/` is a CPU-only kernel (no GPU quota) that
establishes environment facts before any GPU spend; `notebooks/phase3-labels/` runs the model
bake-off, selecting a generator by measured gold AUC rather than by name. Run 2 (both candidates,
100% answer rate, zero OOM retries after the `logits_to_keep=1` fix) put Qwen3-8B at 0.8759 and
Qwen3-4B-Instruct at 0.8613 — a within-noise gap, so the pre-registered near-tie rule took the
faster 4B, which labels the full corpus in one ~4h session instead of two. Both clear the lexical
anchor (+0.24), and both score matrices are kept in `results/bakeoff_scores/` for Phase 4
calibration analysis without another GPU session.

Phase 4 trains the Phase 1 architecture on those pseudo-labels — all 12 findings, soft targets,
5 folds over the frozen `primary_v2` assignment, with the **58 gold studies excluded from both
sides of every split** (Phase 1 had silently trained on 36 of them, which is why its gold gate had
to be substituted with an LB back-out). Run 1: pooled OOF macro AUC **0.7779** against the
pseudo-labels, and gold transfer **0.7252** — the direct LLM-label → rubric number Phase 1 could
only infer. On the four labels Phase 1 also trained, gold transfer averages **0.777 vs its ~0.674**,
and the other eight labels now carry real signal instead of a flat 0.500 each.
`notebooks/phase4-submit/` puts that model on the leaderboard: each test study goes through the same
`prep_study` → `PreppedStudyDataset` path the training artifacts went through (Phase 2's prep
parameters, run 1's read parameters), prep inside `predict_fn` so a decode failure falls back to 0.5
instead of killing the submission, and the mean of all five fold models.

Phase 5 starts from run 1's most informative failure. Gold transfer put **MCL at 0.5034 — random —
and Lateral OA at 0.6132**, while Effusion reached 0.9391. The Qwen3-4B *label* for MCL scored 0.887
gold AUC in the bake-off, so the label is fine; run 1 feeds the model a single **sagittal** series
and MCL is a coronal structure. Next in `_SERIES_PRIORITY` is coronal-fluid-sensitive, so
`notebooks/phase5-train/` changes exactly one thing — `max_series` 1 → 2 — and predicts, in writing
before the run, that the coronal labels move most while the sagittal ones barely do. It also carries
the first real **paired delta**: run 1's OOF arrays ride in as a kernel input, so the two runs are
compared study-by-study on identical folds with a bootstrap CI (`paired_macro_auc_delta`), logged as
one pooled row rather than faked per fold.

Core library (`src/knee/`) covered by tests (118, all of which run on a laptop in ~25 s): DICOM series/slice selection, laterality
resolution (per-header and per-study, with real-world tag-value normalization), pixel decode/
normalize, per-series normalization, pad-to-square and laterality mirroring, JPEG-in-`.npz` study
storage with selective decode, the `prep_study` orchestrator (including its skip-unusable-series and
decode-failure recording), dataset classes for both raw and prepped studies (plus an in-memory
decode cache), model, NaN-masked BCE loss, fold assignment, training/eval loops, experiment logging,
macro AUC, the paired bootstrap delta, and a submission writer with a 0.5 fallback.

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
                       used for the real Phase 1 5-fold training run (notebooks/phase1-train/);
                       also Timer (accumulating wall-clock stopwatch feeding train_minutes),
                       load_gold_holdout + train_val_split (Phase 4's explicit gold exclusion --
                       Phase 1 trained on 36 of the 58 gold studies by accident)
  infer.py           [done] submission writer, byte-for-byte header, 0.5 fallback
  reports.py         [done] EN/ES lexical rules (kept as a per-label fallback candidate,
                      not replaced) + the Phase 3 LLM generator: verbatim rubric,
                      per-label criterion prompts, yes/no-logit soft scoring
  metrics.py         [done] macro AUC, per-label AUC (NaN-safe), and paired_macro_auc_delta:
                      the bootstrap 95% CI the promotion rule ("the paired delta clears
                      noise") is evaluated against, resampling studies for both arms
                      together
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
  phase3-probe/        CPU-only environment probe (no GPU quota): installed versions, disk,
                        measured prompt token counts, per-candidate tokenizer reachability
  phase3-labels/       rubric_733343.md (the host's rubric, verbatim) + the model bake-off:
                        candidates scored on the 58 gold reports, winner picked on gold AUC
  phase3-label-shard0..1/  full-corpus labeling with the bake-off winner: contiguous UID ranges,
                        incremental saves, per-shard completion sentinel
  phase3-label-consume/  CPU stitch+verify kernel: exact partition, value sanity, gold cross-check
                        vs the attached bake-off output, macro AUC recheck -> pseudo_labels CSV
  phase4-train/        run 1: all 12 labels on the LLM pseudo-labels, gold excluded from both
                        sides of every split, plus the post-training gold transfer tier
  phase4-submit/       the scored submission for run 1: prep_study -> PreppedStudyDataset per
                        study (same path the training artifacts took), 5-fold mean, 0.5 fallback
  phase5-train/        experiment B: max_series 1 -> 2 and nothing else, with the paired
                        bootstrap delta against run 1's OOF arrays
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
  paired delta vs. current best, train/inference time, whether it was promoted. One row per fold,
  plus (from Phase 5 on) a `primary_v2_pooled` row carrying the paired delta — that number is a
  property of the whole OOF, not of any single fold. Phase 4 run 1's rows carry `paired_delta=0.0`
  and must keep it: Phase 1 is a different fold set, study count and label set, so no paired
  comparison between them exists.
- `hall_of_fame.csv` — only runs that beat the previous best on a paired per-study delta across
  frozen folds and held up over 2-3 seeds. The final ensemble is built exclusively from these rows.

Plus two one-off Phase 2 census outputs (not part of the experiment-tracking discipline above, not
appended to): `laterality_census.csv` (per-study route/side/slice-counts across the real corpus) and
`laterality_verify_sample.csv` (the multi-instance follow-up check). Plus the Phase 3 bake-off
record: `bakeoff.csv` (one row per candidate, full precision), `bakeoff_scores/` (each candidate's
58×12 soft-label matrix and confidence weights over the gold reports), and the Phase 3 output
`pseudo_labels_qwen3_4b.csv` + `verification_manifest.json` (all 4,407 studies' soft labels — the
Phase 4 training target).

Two frozen reference files, written once and then read-only — both must stay stable, since a silent
change to either invalidates every comparison made against them:

- `folds_primary_v2.csv` — `make_folds(all 4,407 train.csv UIDs, n_folds=5, seed=0)`. The earlier
  `primary_v1` fold set covered only the 2,151 lexically-labeled studies; once training runs on the
  ~4,349 studies with any label, re-deriving folds would reshuffle nearly every study and make every
  paired delta meaningless. `baseline.csv` is deliberately **not** re-anchored to `primary_v2` — it
  records the one actually-submitted Phase 1 run (`primary_v1`, LB 0.558), and a `primary_v2` rerun
  of that config is a new experiment, not a new baseline.
- `gold_study_uids.csv` — the 58 studies carrying gold rubric labels (all 12 label columns non-null;
  all 58 also have report text). Makes the holdout greppable without opening 4,407 archives.

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

**Set `machine_shape` in every new `kernel-metadata.json`.** `kaggle kernels push` forwards that
field verbatim as the requested accelerator, and the empty string every older notebook here carries
means "server picks" — which is why CLI pushes kept landing on a Tesla P100 the container's torch
cannot use. `"machine_shape": "NvidiaTeslaT4"` selects GPU T4 x2 (or
`--accelerator NvidiaTeslaT4`, which overrides the metadata). The pre-Phase-4 notebooks still carry
the empty value; they are records of runs already made, so they are left as they are rather than
rewritten — don't copy one as a template without fixing this field. Note also that a push **always**
starts a run: there is no upload-without-running flag, and importing through the web editor is the
only way to get a notebook onto Kaggle without executing it.
