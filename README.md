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
**submitted** (`notebooks/phase4-submit/`, version 1) and scored **0.763 on the leaderboard**, up
from Phase 1's 0.558 — clearing the pre-registered 0.70 band, so the inference path is verified.
That is rank 2187/2936: the field's median is 0.899 and 10th place is 0.947, so the pseudo-label
thesis is confirmed while the model itself — 1 epoch, one series, `efficientnet_b0` — is barely
trained; **Phase 5 complete and promoted** — pooled OOF **0.8523** against run 1's 0.7779, paired
delta **+0.0744 (95% CI [+0.0669, +0.0820])**, gold transfer **0.8189** against 0.7252, all twelve
labels improved, and submitted from `notebooks/phase5-submit/` — scored **0.835**, up from 0.763;
**Phase 6 complete and scored** — prep v2 (130mm physical-scale crop, six plane/weighting slots with
a presence mask, 3-slice groups), report-hash folds v3, a public label target, and a per-diagnosis
slot-attention head, confirmed on five folds (pooled OOF 0.8485, gold transfer **0.8794** against
Phase 5's 0.8189) and submitted from `notebooks/phase6-submit/` — scored **0.905**, up from 0.835,
inside its pre-registered band; **Phase 7 in progress** — the recovered slot table (prep v3) is
built and the corpus re-prepped, c3 is confirmed on five folds, and the two-config ensemble measures
**+0.0156 [+0.0136, +0.0175]** pooled OOF over the submitted model — see below.**

**Where we actually stand, and what Phase 7 aims at.** As of 2026-09-07 the field is 3,275 teams:
1st 0.954, 10th **0.950**, median **0.902**, and we are **0.905, rank 1602** — three thousandths
above the median. The shape that matters is a shelf: **~940 teams are packed into 0.930–0.940**, so
in that band **+0.01 AUC is worth ~940 places**. Phase 7 therefore targets **0.930–0.940**, not the
podium. That shelf is teams running `pilkwang/rsna-knee-baseline-v1`'s *fitted weights*, not 940
independent reimplementations — which matters, because Phase 6 implemented most of that recipe and
landed at 0.905.

**Phase 6 existed because of an earlier leaderboard pull, and the framing was uncomfortable.** As of
2026-09-06 the field was 3,221 teams: 1st 0.954, 10th **0.950**, median **0.899**, and roughly a
thousand teams at >=0.93 — the score the most-forked public notebook produces. At 0.835 we were
rank 2163, i.e. **below what forking a public inference notebook costs nothing to get**. Phase 5's
+0.072 was real, but it was 60% of the distance to the *median*, not to the top ten. The Efficiency
Track does not rescue this: its formula prices **0.01 AUC at 12 minutes** of runtime, so cutting a
68-minute run to 10 buys the equivalent of 0.048 AUC against a gap of 0.115, and matching a 0.95
rival that runs in 30 minutes would need 0.925 *at zero runtime*. (It is not hopeless there — a 0.95
model taking four hours scores worse than our submitted 0.835 at 68 minutes, because the track really
does reward accuracy per second — but accuracy is the lever in both tracks.)

**Phase 7 so far.** Three findings, each measured rather than assumed:

| finding | evidence |
|---|---|
| **An acquisition axis was invisible to the model.** `train_series.csv`'s one column encodes *fat suppression*; TR/TE encode *weighting*, and they are orthogonal. `SAG_STRUCT` alone carried 1,645 T1, 1,702 PD, 1,224 T2 and 388 GRE into a single attention position | corpus header census, 24,371 series in 2.5 min |
| **DINOv2 was overfitting, not underfitting.** Phase 6 read "training loss still falling at epoch 8" as unfinished learning. At 16 epochs the loss falls *below* efficientnet's while both validation tiers drop | val −0.0075 [−0.0156, −0.0006], gold −0.0391 [−0.0701, −0.0045], both CIs excluding zero |
| **Config diversity is fully additive to fold averaging.** c2+c3 over five folds each beats either member by the same margin the fold-0 screen measured over single-fold members | OOF +0.0156 [+0.0136, +0.0175]; gold 0.8919 vs 0.8794 |

**Prep v3** re-prepped all 4,407 studies into a nineteen-slot table (plane × fat-suppression ×
{T1, PD, T2, UNK}) with 0 failures, mean 5.04 filled slots per study against a census prediction of
5.04, and **every slot holding exactly one weighting class**. The `_UNK` tier is load-bearing: 238
studies (5.4%) have no recoverable TR/TE on any series and would otherwise get zero slots. Whether
nineteen sparse positions beat six dense ones is the next screen, not an assumption — the median
study fills 5 of 19.

Phase 6 takes the half of the 2026-09-05 recon that Phase 5 skipped — Phase 5 banked the cheap levers
(schedule, AMP, augmentation, a second series) and left the structural ones. Landed so far:

| step | what | result |
|---|---|---|
| prep v2 | 130mm physical-scale crop at 336px, six plane/weighting slots + presence mask, contiguous 3-slice groups | corpus prepped: **21,334 series carrying 1,028 distinct `PixelSpacing` values all came out at one mm/px** |
| folds v3 | re-frozen on normalized report text | report-group leakage **4.31% -> 0** |
| label A/B | our Qwen3-4B against four public label sets | **ours placed 3rd of 5**; the winner is now the training target |
| loader/model | slot axis with a presence mask, masked pooling, per-diagnosis attention | replaces a flat volume that could not express a missing plane at all |
| ensembling | probability mean over configs and folds | +0.0159 [+0.0114, +0.0209] over the best single model |

Fold-0 screens, all on the promoted `steven_v4` target, scored on 863 held-out studies and on the 58
rubric-graded gold studies that no model trains on:

| arm | val | gold | min |
|---|---|---|---|
| c0 — 2 slots, efficientnet_b0 | 0.8464 | 0.8657 | 18.5 |
| c1 — 6 slots + presence mask | 0.8580 | 0.8579 | 47.1 |
| **c2 — + per-diagnosis attention** | **0.8607** | 0.8759 | 51.0 |
| c3 — DINOv2-small, 2 slots | 0.8512 | 0.8686 | 23.3 |
| c5 — DINOv2, full stack | 0.8549 | 0.8724 | 60.1 |
| **c1+c2+c3+c5, probability mean** | **0.8767** | **0.8897** | — |

Three findings worth carrying out of that table. **The label source mattered more than any
architecture** — swapping our Qwen3-4B labels for a public set moved gold transfer +0.0339
[+0.0067, +0.0625], and on the *circular* metric (each model scored against its own labels) the
swap looked like a regression, which is why the 58 gold studies are used as the arbiter whenever the
target itself is the variable. **No single backbone or head wins; the ensemble does** — c3 is null as
a standalone model and the best ensemble partner in the set, because its per-label strengths (MCL,
Contusion, Fracture) are where c2 is weakest. And **the attention head learned the anatomy unprompted**:
averaged over gold studies it sends ACL to the sagittal slot (0.54), MCL to coronal (0.55), Baker's to
axial (0.56), with nothing in the loss describing knee anatomy.

One public label set (`yunusgmsoy/...-4-source-merged`) is excluded from both scoring and training
because it scores a perfect 1.0000 on all twelve gold labels — its 58 gold rows are copied from
`train.csv`. A candidate that scores near-perfectly against a small gold set is contaminated until
proven otherwise.

Phase 5 came out of reading the top public notebook for intel (not code): it runs DINOv2 at 10
epochs with physical-scale sampling and rank-mean ensembling. Two of its claims were tested against
our own data, and the phase was re-ranked around what they showed. What actually paid:

| lever | effect | where |
|---|---|---|
| 1 -> 8 epochs + OneCycle | **+0.059** fold-0 macro | `notebooks/phase5-screen/` |
| augmentation | +0.006, train-val gap 0.143 -> 0.090 | `notebooks/phase5-screen-b/` |
| 2nd (coronal) series **with** augmentation | +0.005, concentrated in coronal labels | `notebooks/phase5-screen-b/` |
| geometry laterality | side coverage 51.1% -> 98.5%, but **null on accuracy** | `notebooks/phase5-laterality/` |

The laterality result is worth stating plainly: the Phase 0 `ImagePositionPatient` rule was rejected
for reading the image *corner* rather than its *centre*, and the centre form agrees with the real
tag on **99.08%** of the 2,182 studies that carry one, recovering a side for 47.4% of the corpus
that had none. It cost no re-prep, because Phase 2 stored pixels unmirrored. It also bought no
measurable accuracy at this model's capacity — it is kept because it is free and because the
submission path resolves test-study sides through the same function, so training and inference
agree. See `NOTES.md` 2026-09-05/06.

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
  prep.py            [done] normalize_series, mirror_to_canonical + mirrors_in_plane (only
                      Axial/Coronal flip), JPEG-in-.npz storage with selective decode, the
                      Phase 2 prep_study orchestrator (pad-to-square letterbox), and the
                      Phase 6 prep_slots orchestrator: crop_to_mm gives every study the same
                      physical scale, which the letterbox never did -- studies differ in
                      mm/px by a factor of several and no downstream capacity recovers that
  dataset.py         [done] KneeStudyDataset over raw DICOMs (Phase 1 shape),
                      PreppedStudyDataset over the Phase 2 .npz artifacts, and
                      PreppedSlotDataset over the Phase 6 slot artifacts -- the last
                      returns (image [slot, group, 3, H, W], presence mask, labels, uid)
                      rather than one flat slice axis, so "this study has no coronal
                      acquisition" is expressible instead of being padded over
  model.py           [done] single backbone, pooled over images; accepts either the flat
                      volume or the slot-structured one and pools only over present slots
                      (an unmasked mean averages anatomy with the black squares that
                      absent slots arrive as). Per-diagnosis attention is the next step
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
  phase5-laterality/   corpus-scale test of the image-centre laterality route, header-only:
                        4,407 studies in 2.1 min, coverage 51.1% -> 98.5%
  phase5-screen/       screen A -- the training schedule (epochs, OneCycle, AMP) on one fold
  phase5-screen-b/     screen B -- augmentation, a second series, and the laterality override
  phase5-confirm/      the 5-fold confirm behind the 0.835 submission (pooled OOF 0.8523)
  phase5-submit/       that submission's inference kernel: 5-fold mean, geometry laterality
  phase6-prep-shard0..3/  prep v2 over the corpus: physical-scale crop, slot table, slice
                        groups. Gates on slot fill, constant mm/px, and an 8x6 visual grid
  phase6-screen/       Step 3's re-baseline on the new artifact, then the Phase 6 screen cells.
                        Regenerates folds v3 in-kernel and pins them by SHA rather than
                        mounting a CSV, and asserts the corpus prep census from the shard
                        manifests -- 9.5GB of artifacts need not leave Kaggle to be checked
  phase6-confirm/      5-fold confirm of the six-slot attention config: pooled OOF 0.8485,
                        gold transfer 0.8794, and the finding that fold 0 is an easy fold
  phase6-submit/       the scored 0.905 submission: prep v2 on test DICOMs, the presence mask
                        passed to the model, probability mean over the five folds
  phase7-census/       CPU header walk over all 24,371 series recovering TR/TE weighting --
                        the run that showed every _STRUCT slot mixes T1, PD, T2 and GRE
  phase7-prep-shard0..3/  prep v3: the nineteen-slot recovered table (plane x fat-suppression
                        x weighting) with an explicit UNK tier for the 238 studies whose
                        headers carry no recoverable TR/TE
  phase7-screen/       the s1 schedule gate -- DINOv2 at 8 vs 16 epochs, both arms in one
                        session so the pairing does not rest on an unrecoverable batch size
  phase7-confirm-c3/   c3's 5-fold confirm, the second ensemble member: OOF 0.8491, gold
                        0.8805, at 43% of c2's GPU cost
  phase7-submit/       the two-config ensemble kernel: one prep pass, two loader shapes and
                        two architectures, probability mean over ten models
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

Three frozen reference files, written once and then read-only — a silent change to any of them
invalidates every comparison made against it:

- `folds_primary_v3.csv` — Phase 6's fold set:
  `make_folds(uids, n_folds=5, seed=0, groups=report_group_key(report))`, grouped so studies sharing
  a report never straddle a fold. `primary_v2` leaked 47 report groups over 190 studies (4.31%) —
  that much of its OOF was scored against text the model had already trained on. Re-freezing was
  deferred while it would have broken Phase 5's paired comparisons; prep v2 breaks them anyway, which
  made this the free moment. The key normalizes case and whitespace before hashing: byte-identical
  finds 46 duplicate groups, `.strip()` 49, and this key 54 — two reports differing only in spacing
  are the same report and leak identically. Screen kernels regenerate it in-kernel and assert
  SHA256 `f1d6ba7c341f8d2e82241cb4ddd5ca5131b0c6d2033eb6d2e43bcb597cf12868`.
- `label_source_gold.csv` — every candidate report-label source scored against the 58 gold studies.

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
