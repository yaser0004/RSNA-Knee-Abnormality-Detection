# RSNA Knee Abnormality Detection: the full record

This is the complete write-up of a solo entry in the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
(August 8 to October 22, 2026). `README.md` is the short version. This file has everything else: results,
decisions and why they were made, every phase, the code layout, data rules, and where each artifact lives now
that the local working copy is gone. `NOTES.md` is the dated lab notebook behind it (every measurement, every
pre-registered band and the score it was read against). `RESOURCES.md` lists the third-party weights and datasets in
the final submissions.

The project is finished and archived. Nothing here is maintained.

Contents:

1. Final results
2. The task and the shape of the problem
3. How the project was run (rules it kept)
4. Phase summary and the decision behind each phase
5. Phase 11 in detail (the last phase)
6. What worked, what did not
7. Where everything lives
8. Data handling and what is deliberately not in this repository
9. `tools/`
10. The competition-time write-up (phase-by-phase findings, code layout, experiment tracking, Kaggle setup), kept as
    it was written, with only paths corrected

---

## 1. Final results

**Final private leaderboard: rank 2,717, score 0.928** (public: rank 2,900 of 6,299 at 0.943; 1st place on the public
board scored 0.970).

Every submission with a score, read from the Kaggle submissions API after the close (two blend submissions on
2026-10-01/02 timed out on the hidden set and have no score):

| date | submission | what it was | public | private |
|---|---|---|---|---|
| 08-08 | 55355919 | Phase 1 baseline: efficientnet_b0, one sagittal series, keyword labels for 4 findings | 0.558 | 0.542 |
| 09-02 | 55968638 | Phase 4: 12 labels from Qwen3-4B report labels, 5-fold mean | 0.763 | 0.736 |
| 09-06 | 56047536 | Phase 5: 2 series, 8 epochs OneCycle + AMP + augmentation | 0.835 | 0.827 |
| 09-07 | 56076866 | Phase 6: prep v2 (130 mm crop, 6 slots + mask), slot attention, steven_v4 target, folds v3 | 0.905 | 0.893 |
| 09-12 | 56193964 | Phase 7 A: c2 + c3, 10 fold models | 0.916 | 0.903 |
| 09-14 | 56227030 | Phase 7 B: + ConvNeXt-tiny, 9 slice groups, 15 models | 0.921 | 0.908 |
| 09-19 | 56360159 | Phase 8 C: + c2 at 336 px, 20 models | 0.925 | 0.911 |
| 10-01 | 56754187 | Phase 8 D: six members, 30 models | 0.926 | 0.912 |
| 10-01 | 56754867 | Phase 9 P: licence-compliant public stack | 0.938 | 0.924 |
| 10-02 | 56771279 | Phase 9 G: d3 + d3@336, 10 models (own weights) | 0.928 | 0.917 |
| 10-03 | 56785577 | blend-g v2: P + G, deadline watchdog | 0.938 | 0.925 |
| 10-04 | 56809803 | blend-h: P + d3, equal weight | 0.936 | 0.924 |
| 10-04 | 56819611 | **blend-i: P + d3 at 0.2 (planned final pick 1)** | 0.939 | 0.925 |
| 10-04 | 56829311 | full public stack, unmodified (includes licence-restricted members) | 0.943 | 0.927 |
| 10-05 | 56838003 | full public stack + d3 at 0.15 | 0.943 | **0.928** |
| 10-07 | 56899076 | Phase 11 anchor: d3 fold 0 alone | 0.916 | 0.907 |
| 10-07 | 56912716 | student arm b, seed 0, fold 0 | 0.922 | 0.911 |
| 10-10 | 57053697 | student arm b, seed 1, fold 0 | 0.923 | 0.909 |
| 10-10 | 57053875 | student arm c (Claude/muse labels in the blend), fold 0 | 0.923 | 0.913 |
| 10-15 | 57217419 | round-2 student c2, fold 0 | 0.920 | 0.911 |
| 10-15 | 57221367 | S15: 15 round-1 students | 0.925 | 0.914 |
| 10-15 | 57233603 | **GS25: G + S15, 25 own-weights models (planned final pick 2)** | 0.928 | 0.917 |

**The final score and the planned picks do not match.** The planned finals (decided 2026-10-16, `RESOURCES.md`) were
blend-i (private 0.925) and GS25 (private 0.917). The leaderboard's 0.928 is the private score of 56838003. A team's
leaderboard score is the better of its two selected submissions, and when none is selected Kaggle uses the two best
public scores, which here are exactly the two 0.943 entries. The API does not say which submissions were selected, so
this record does not claim either way.

What public-versus-private shows:

- Private was 0.008-0.027 below public for every entry (0.009-0.016 for everything from Phase 6 on), and the ranking of
  our entries barely moved. Nothing gained on the reshuffle.
- The gap between our own models and other teams' public weights held on private: own best 0.917, compliant public
  stack 0.924, full public stack 0.927-0.928.
- Rank moved from 2,900 (public) to 2,717 (private): a net gain of 183 places against the rest of the field.

---

## 2. The task and the shape of the problem

Predict, per knee MRI study, the probability of 12 findings (ACL tear, MCL tear, medial and lateral meniscus tear,
medial / lateral / patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, acute fracture). Scored by
macro-averaged ROC AUC. Each study is several DICOM series (different planes and weightings). Training data: 4,407
studies, each with the original radiology report. Hidden test set: about 1,300 studies (public LB about 390, private
about 910), scored in a Kaggle notebook with a 9 h limit.

The defining fact, found on day one: **only 58 of the 4,407 training studies carry expert (gold) labels.** The rest
have only a free-text report in several languages. So this is a weak-supervision problem. The whole project is
about turning reports into training targets, then training image models on them, with the 58 gold studies kept out of
training as the only honest judge of label quality.

---

## 3. How the project was run (rules it kept)

- **Free compute only.** Kaggle notebooks (GPU T4 x2, later also an 8-chip TPU), no paid compute, solo. The local machine
  (GTX 1650 4 GB, 7 GB RAM) only wrote code and ran unit tests. The real Kaggle GPU allowance turned out to be 6 h per
  weekly window, not the 30 h earlier plans assumed (found 2026-10-07 via `api.quota_view()`); TPU was 20 h per window.
- **Measure, don't assume.** A change was promoted only on a paired, per-study bootstrap delta over frozen folds
  (`paired_macro_auc_delta`), with the 95% interval above zero. Gold-transfer (scoring on the 58 gold studies) was the
  arbiter whenever the training target itself was the variable, because scoring a model against its own labels is
  circular.
- **Pre-registered leaderboard reads.** From Phase 6 on, every submission had an expected band written in `NOTES.md`
  before it was pushed, and was read against that band. Phase 11 capped itself at 8 decision reads. No weights or
  hyperparameters were tuned on the public LB.
- **Gold stays out.** The 58 gold studies were excluded from both sides of every training split from Phase 4 on. (Phase 1
  trained on 36 of them by accident; that is why its gold check had to be backed out from the LB.)
- **Frozen reference files.** Fold assignments and the gold list were written once and pinned by hash
  (`results/folds_primary_v3.csv`, SHA256 in section 10).
- **Test-first library code.** `src/knee` was written test-first; the final suite is 334 tests, all CPU-only (about 90 s).
- **Competition data rules.** Report-derived labels were treated as private: kept on disk and in private Kaggle datasets,
  never in this repository (section 8). Until host thread 733965 was read (2026-10-07), report text was only sent to
  open-weight models running inside Kaggle; after it, hosted LLMs (Claude, and a free cloud model via opencode) were used
  for relabelling, as the host allowed.
- **Licences.** Public weights were audited for licence compatibility with the winner licence (CC BY-NC 4.0 at the time),
  which is why stack P excludes the DINOv3-derived and RadImageNet members.

---

## 4. Phase summary and the decision behind each phase

The plan documents themselves were working files and are not kept. This is what they decided and why.

| phase | dates | bet | outcome |
|---|---|---|---|
| 0-1 | Aug 8 | end-to-end baseline: one sagittal series, efficientnet_b0, keyword labels | LB 0.558. Only 4 of 12 labels had any keyword rule, so 8 columns scored 0.5: `0.558 = (4 x 0.674 + 8 x 0.5) / 12` |
| 2 | Aug 9-10 | preprocess every study once into compact `.npz` artifacts | 4,407 studies, 37x faster loading; found laterality from tags resolves only 51% of studies and a sagittal-mirroring bug |
| 3 | Aug 17-Sep 1 | an open-weight LLM reads every report against the host rubric | Qwen3-4B won a gold bake-off (0.861 macro AUC vs 0.625 keyword anchor) and labelled the corpus |
| 4 | Sep 1-2 | train on those labels, all 12 findings | LB 0.763, pseudo-label thesis confirmed |
| 5 | Sep 5-6 | cheap levers from reading the top public notebook: schedule, AMP, augmentation, a second series | +0.074 OOF, LB 0.835 |
| 6 | Sep 6-7 | structural: physical-scale crop, plane/weighting slots with a presence mask, per-diagnosis attention, better public label target, leak-free folds | LB 0.905; the label source mattered more than any architecture |
| 7 | Sep 8-18 | ensemble diversity: DINOv2, ConvNeXt-tiny, 336 px, slice-window averaging | LB 0.916 -> 0.921 |
| 8 | Sep 19-Oct 1 | close the gap to the 0.95 cluster with our own models: 336 px member, random training windows, a three-source label blend, group-token attention | C 0.925, D 0.926, own best G = 0.928 (scored Oct 2); pooled OOF rose to 0.887 but the LB stopped responding (+/-0.002) |
| 9 | Oct 1-5 | user decision: use public weights, licence-filtered, and blend ours in as the decorrelated half | P 0.938, blend-i 0.939; licence route reopened Oct 4: full public stack 0.943 |
| 10 | Oct 6 | frozen large-backbone features (DINOv3 ViT-L) cached once, cheap heads on top, to make experiments fast | Gate G1 failed: 0.808 OOF vs 0.878 for fine-tuned d3, +0.0004 as a member; stopped after ~2.5 GPU-h |
| 11 | Oct 7-16 | own weights only (host had not ruled on public weights); a Claude relabel and the teacher-student pseudo-label loop the forum credited for 0.95 single models | single students +0.006-0.007 on LB, but the gain vanished in ensembles: S15 0.925, GS25 0.928 = G |
| finals | Oct 16 | host ruled public weights prize-eligible (thread 744056); finals set to blend-i and GS25 | see section 1 |

Decisions worth knowing, with the reasoning:

- **Efficiency Track dropped (2026-09-19).** Its formula priced 0.01 AUC at about 12 minutes of runtime; the accuracy gap
  was too large for runtime to make up. Both final slots went to accuracy.
- **Leaderboard ties.** The public board was spikes of identical scores (hundreds of teams at exactly 0.936, 0.941,
  0.943), each a forked public notebook. Ties go to the earlier submission, so matching a spike's score lands behind all
  of it. This is why "0.943" meant rank about 1,100 to 2,900 depending on the date.
- **Phase 8's thesis:** enter the 0.95 cluster with the least LB-overfit model and hope the private reshuffle favours it.
  The measured levers summed to far less than needed. A team merge before the Oct 15 deadline was considered and declined.
- **Phase 9's licence filter (Oct 1):** use everything public minus anything whose licence conflicts with the winner
  licence. Reopened on Oct 4 for the full stack, with the explicit consequence that entries including the restricted
  members could not ship in a winner release.
- **Phase 10's reasoning:** a member only helps a 0.94 stack if it is itself near 0.94; ours were 0.015 below. T4s cannot
  fine-tune CoAtNets at 384 px, but they can extract frozen features once. The registered fallback (partial unfreeze via
  cached tokens) was infeasible by storage arithmetic (~170 GB), so the track stopped as its rule said.
- **Phase 11's reasoning (Oct 7):** forum thread 735304 reported single models at 0.950-0.955 at 224 px from teachers
  trained on several label sources, then soft pseudo-labels (50/50 between labels and teacher OOF). The bet was the target,
  not the network. Odds were written down beforehand: own-weights >= 0.945 about 35%, first place 1-2%.
- **Gold-in-training was not run.** 58 studies against 4,349 is about 4% effective weight, no read could validate it, and
  retraining 25 models would cost a multi-hour TPU queue for an unmeasurable effect.
- **Finals (Oct 16):** pick 1 = blend-i (best licence-compatible public-weights entry, 0.939), pick 2 = GS25 (best own-weights
  entry, largest ensemble). The full 0.943 stack was left out because the winner-licence question (thread 739905) was
  unanswered.

---

## 5. Phase 11 in detail (the last phase)

The competition-time write-up in section 10 stops at Phase 10. Full entries are in `NOTES.md` from 2026-10-07.

**Claude relabel (Gate C1).** A zero-shot prompt (`tools/phase11-claude-labels/prompt.md`, sha256 `44f2ceea...`, frozen
before any gold output) asks for the probability that the expert panel marked each finding positive, on a 9-step scale,
plus 18 auxiliary "atoms". On the 58 gold reports, measured by cell agreement with gold:

| source | cell agreement | gold macro AUC |
|---|---|---|
| blend3 (steven_v4 + ours + lixin73, the Phase 8 target) | 0.8333 | 0.8852 |
| Claude, mean of two passes | 0.8448 | 0.8953 |
| **blend3 + Claude as a fourth source** | **0.8592** | 0.8888 |
| blend3 + muse-spark (opencode free model) | 0.8563 | 0.8863 |

Blend3 + Claude beat blend3 by +0.0259 [+0.0086, +0.0431], a pass. The corpus was labelled single-pass (Sonnet) for
2,280 studies and by `opencode/muse-spark-1.3-contributor-free` for the other 2,069 after the account's usage limit was
hit; the two labellers agreed on 98.5% of cells on a shared chunk. Synovitis stayed the weakest label for every source.

**Teacher-student loop.** Teacher 1 = the mean OOF of the eight confirmed members. Students use d3's recipe with target
0.5 x label blend + 0.5 x teacher OOF. Arms: b (blend3), c (blend3 + Claude/muse labels), d (c + atoms as auxiliary
heads at weight 0.3).

| read | LB (single fold-0 model unless stated) |
|---|---|
| anchor, d3 fold 0 | 0.916 |
| arm b seed 0 / seed 1 | 0.922 / 0.923 |
| arm c | 0.923 |
| round-2 c2 (teacher 2 = mean of b, c, d) | 0.920, below the 0.926 stop line, loop stopped |
| S15 (b, c, d x 5 folds) | 0.925 |
| GS25 (G + S15, 25 models) | 0.928 |

The single-model gain was real (+0.006-0.007 over the anchor on two seeds). It did not survive ensembling: pseudo-label
students and fold/config averaging remove the same label noise, so S15 gained only +0.002 over its own single fold, where
G gained +0.012. Gold could not separate G, S15 and GS25 (0.9139 / 0.9127 / 0.9140, interval about +/-0.01).

**TPU farm.** With the GPU allowance spent, the d3 recipe was ported to Kaggle's TPU v3-8: one independent job per chip
(no gradient sharing, so the GPU recipe's batch and BatchNorm semantics hold), static shapes via presence-weighted BatchNorm
(`knee/masked_bn.py`), tensor-valued Adam and OneCycle so the schedule does not recompile (`knee/xla_train.py`), and a
per-chip launcher (`knee/tpu_farm.py`) because `torch_xla.launch` fails on Kaggle's TPU VM. A TPU-trained fold reproduced
its GPU twin (val -0.002, gold -0.005, prediction correlation 0.984). Eight fold-trainings took about 0.4 h of compute,
but queue waits ran 7-20 h and Kaggle shows a finished TPU kernel as RUNNING for hours during output handling: do not
cancel on elapsed time.

**Host ruling (thread 744056, 2026-10-14):** submissions using publicly shared weights are award-eligible if the weights
and their source datasets are listed. That list is `RESOURCES.md`.

---

## 6. What worked, what did not

Worked (measured):

- Reading reports with an LLM to label all 12 findings: the single largest gain (0.558 -> 0.763 -> onward).
- The label source over architecture: swapping our labels for a better public set moved gold +0.034.
- Physical-scale cropping and a plane/weighting slot table with a presence mask; per-diagnosis attention, which learned
  knee anatomy unprompted (ACL to sagittal, MCL to coronal, Baker's to axial).
- Ensembling diverse configs, slice-window averaging at inference (+0.0045 OOF for free), 336 px as an extra member,
  random training windows.

Did not work, or stopped mattering:

- Geometry laterality (coverage 51% -> 98.5%, no accuracy change at this capacity; kept because it is free).
- Confidence weights on the loss (-0.009).
- Own-model variants beyond G: the public LB stopped responding at +/-0.002 from Phase 8 on.
- A fourth public label source (Screen F), blend weights against the public stack, frozen DINOv3-L features (Phase 10),
  round-2 teachers, and students inside an ensemble.

The distance to the top (0.964-0.970) was not closed with this hardware and these levers. The forum credited its 0.95
single models to label work and pseudo-labelling; this project reproduced that recipe's single-model gain, not its level.

---

## 7. Where everything lives

The local working copy (with `data/`, `checkpoints/` and the AI workspace) was deleted after the competition. What remains:

| artifact | where |
|---|---|
| code, tests, notebooks, docs | this repository (GitHub, GitLab, Codeberg, Bitbucket mirrors) |
| trained checkpoints and OOF arrays | outputs of the training kernels on Kaggle, user `mohammadyaserhussain` (e.g. `knee-phase8-confirm-d3`, `knee-phase9-confirm-d3-336`, `knee-phase11-tpu-farm1`, `-farm2`, `-farm3`). Kernel names match the folders in `notebooks/` with a `knee-` prefix |
| our code as imported by the notebooks | private Kaggle dataset `mohammadyaserhussain/rsna-knee-src` |
| preprocessed slice caches (prep v1-v3) | outputs of the `knee-phase2-prep-shard*`, `knee-phase6-prep-shard*`, `knee-phase7-prep*-shard*` kernels |
| Qwen3-4B report labels and confidence weights | outputs of `knee-phase3-label-consume` (private kernel); never in this repo |
| Claude / muse-spark corpus labels | private dataset `mohammadyaserhussain/rsna-knee-phase11-claude-labels` |
| teacher OOF arrays (teacher 1 and 2) | private dataset `mohammadyaserhussain/rsna-knee-phase11-teacher` |
| submissions and scores | the competition's submissions page; table in section 1 |
| third-party resources in the finals | `RESOURCES.md` |

Not kept anywhere: the plan documents (summarised in section 4), the local scratch area beyond the scripts in `tools/`,
and the modified copy of the third-party full public-stack notebook (its source is the public notebook
`yamadan96/rsna-knee-d4-public0946`).

---

## 8. Data handling and what is deliberately not in this repository

- `data/` (competition sample) and `checkpoints/` are gitignored: competition data must not be redistributed, and weights
  are large and live on Kaggle.
- Report-derived labels are never committed: `results/pseudo_labels_qwen3_4b.csv`, `results/label_confidence.csv` and the
  Phase 3 bake-off score matrices `results/bakeoff_scores/` (LLM scores over the 58 gold reports). The first two were once
  committed by mistake and the bake-off matrices were tracked; all three were removed from the whole history before the
  final push. `results/bakeoff.csv` (one summary row per candidate) and `results/label_source_gold.csv` (AUCs per
  source) are aggregate numbers and stay.
- The files still in `results/` are study-level metadata derived from DICOM headers (laterality routes, weighting census),
  fold assignments, the gold study UID list and experiment logs. They carry no labels or images.
- No credentials are stored here. The Kaggle token lives in `~/.kaggle/access_token` on the user's machine only.

---

## 9. `tools/`

Offline helper scripts that produced numbers or notebooks cited in `NOTES.md`. They are kept as records of the runs:
most read inputs from the deleted local workspace (paths like `ai/scratch/...`, run from the workspace root) and will
not run as-is.

| script | what it did |
|---|---|
| `phase8-submit/score_members.py` | scored candidate ensemble members on pooled OOF and gold from saved arrays |
| `phase9-public/build_p.py` | built `notebooks/phase9-public-stack/` from the public notebook by asserted text swaps (a changed parent fails the build) |
| `phase9-public/build_blend.py` | built the `phase9-submit-blend*` notebooks: public stack plus our members, per-label rank mean, deadline watchdog |
| `phase9-offline/score8.py`, `subsets.py` | eight-member pooled OOF at nine groups; exhaustive subset search |
| `phase9-labels/score_gold.py` | scored public label datasets on the 58 gold studies with a contamination check |
| `phase10-g1/g1.py` | local harness for Gate G1 (heads on cached frozen features) |
| `phase11-claude-labels/prompt.md` | the frozen relabelling prompt (host rubric verbatim; no report text) |
| `phase11-claude-labels/score_c1.py` | Gate C1 scorer: cell agreement with gold, paired study bootstrap |
| `phase11-claude-labels/assemble.py` | assembled the two labellers' outputs into one corpus label file |
| `phase11-tpu/read_farm3.py` | readout of the round-2 student farm |

---

## 10. The competition-time write-up

What follows is the README as it stood during the competition (last updated after Phase 10), kept intact so no detail is
lost. It is ordered by when things were written, not strictly by phase. Paths were corrected; nothing else changed except
the notes on removed files.

### Status as written during the competition

Empirical findings from actually
running things (timing numbers, data quirks, model behavior) go in `NOTES.md` as they're discovered,
kept separate from the plan documents (summarised in section 4). Status: **Phase 1 baseline
complete — trained, submitted, and scored on the real leaderboard; Phase 2 (preprocessing) gate
closed — all 4,407 studies prepped and verified; Phase 3 gate closed — Qwen3-4B-Instruct won the
bake-off (gold macro AUC 0.8613 vs 0.625 lexical anchor) and labeled the full corpus
(`results/pseudo_labels_qwen3_4b.csv`, never committed, 52,884/52,884 answered, all consumer gates passed);
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
inside its pre-registered band; **Phase 7 in progress** — c3 is confirmed on five folds, and the
two-config ensemble (pooled OOF **+0.0156 [+0.0136, +0.0175]** over c2 alone) was submitted from
`notebooks/phase7-submit/` and scored **0.916**, up from 0.905 and inside its pre-registered band;
prep v3's nineteen-slot table was screened and tied the six-slot one, while averaging slice windows
at inference and ConvNeXt-tiny as a third member each measured as a gain; Submission B combined
them and scored **0.921** (rank 1,789 of 3,959 on 2026-09-18), at the floor of its pre-registered band;
c2's config read at 336px is confirmed as a fourth member (pooled OOF +0.0042 [+0.0033, +0.0051])
and read at nine groups joined the others in Submission C, which scored **0.925** (rank 1,712 of
3,994 on 2026-09-19), the top of its pre-registered expected row;
**Phase 8** (random training windows, a three-source blend target, group-token attention) took our own
ensemble from 0.925 to **0.928** (two models, d3 and d3@336, pooled OOF 0.8874) while pooled OOF rose
from 0.8754 to 0.8874: the public leaderboard stopped rewarding our own variants beyond +/-0.002.
**Phase 9** (2026-10-01 on) added the licence-compliant part of the public stack — DINOv2 x20, Raptor
CoAtNet x4 views and four CoAtNet readers; RadImageNet heads and the DINOv3-derived members are excluded
because their licences conflict with the CC-BY-NC 4.0 winner release, audit in `NOTES.md` — which scores
**0.938 alone** (`notebooks/phase9-public-stack/`). Blended with our d3 at 0.2 weight it scored **0.939**
(`notebooks/phase9-submit-blend-i/`; at 0.5 weight it scored 0.936). The public stack's hidden-set run takes
6.3-7.3 h of the 9 h limit, so the blend kernel carries a deadline that falls back to the public stack
alone. On 2026-10-04 the licence route was reopened by decision: the full public stack, unmodified
(the excluded members included, so it cannot ship in a CC-BY-NC 4.0 winner release), scored **0.943** (rank
about 1,100 behind its 717 ties), and the same stack with our d3 at 0.15 weight also scored **0.943**.
**Phase 10** (2026-10-06) tested the one remaining new lever, frozen large-backbone features cached once and read by cheap
heads (`src/knee/features.py`, `feature_head.py`, `notebooks/phase10-*`): DINOv3 ViT-L at 336px scored **0.8076** pooled OOF
against 0.8781 for our fine-tuned d3 and added +0.0004 as a ninth member, so Gate G1 failed and the track stopped after ~2.5
GPU-hours. Findings and every band read against its score are in `NOTES.md`; plans summarised in section 4.**

**Where we actually stand.** On 2026-09-18 the field was 3,959 teams: 1st **0.957**, median **0.913**,
our 0.921 at rank **1,789**. The spikes grew — 538 teams at exactly 0.941, 321 at 0.936 — so 0.942 ranks
175, 0.945 ranks 100, 0.950 ranks 39 and 0.955 ranks 5: the 0.95 cluster is Phase 8's target, and the
honest arithmetic (gold→LB offset now +0.015 and shrinking) says it needs gold near 0.945. Earlier
readings, kept for the record: as of 2026-09-12 the field was 3,620 teams:
1st 0.955, median **0.909**, and our 0.905 sat **below** the median at rank ~1855; on 2026-09-13
Submission A's **0.916** ranks **1,671 of 3,638**, above it. The top is
not a smooth shelf but **spikes of identical scores** — 379 teams at exactly 0.936, 140 at 0.940,
264 at 0.941 — each a public fitted-weights notebook forked hundreds of times. Ties go to the earlier
submission, so reaching a spike's score lands behind all of it: 0.936 is rank 1,139, **0.937 is rank
760**, 0.941 is rank 392 and **0.942 is rank 128**. Phase 7 targets those two lines. The notebooks at
0.94 stack DINOv2, DINOv3 and RadImageNet heads fitted by other teams; none of that can be used here
(other teams' fitted heads, and weights whose licences conflict with the CC-BY-NC 4.0 winner
licence), which is why the measure of progress stays our own gold transfer rather than rank.

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

**Phase 7 so far.** Ten findings, each measured rather than assumed:

| finding | evidence |
|---|---|
| **An acquisition axis was invisible to the model.** `train_series.csv`'s one column encodes *fat suppression*; TR/TE encode *weighting*, and they are orthogonal. `SAG_STRUCT` alone carried 1,645 T1, 1,702 PD, 1,224 T2 and 388 GRE into a single attention position | corpus header census, 24,371 series in 2.5 min |
| **DINOv2 was overfitting, not underfitting.** Phase 6 read "training loss still falling at epoch 8" as unfinished learning. At 16 epochs the loss falls *below* efficientnet's while both validation tiers drop | val −0.0075 [−0.0156, −0.0006], gold −0.0391 [−0.0701, −0.0045], both CIs excluding zero |
| **Config diversity is fully additive to fold averaging.** c2+c3 over five folds each beats either member by the same margin the fold-0 screen measured over single-fold members | OOF +0.0156 [+0.0136, +0.0175]; gold 0.8919 vs 0.8794 |
| **Absent slots were still run through the backbone.** The mask kept them out of pooling, but each fed a black image into BatchNorm's training statistics, and at 19 slots they would be 73% of backbone compute. Now packed out before the backbone; eval logits are unchanged for any study with a present slot, so the scored checkpoints are unaffected | equivalence tests on both heads; 19-slot load 1,243 → 120 ms/study |
| **The recovered nineteen-slot table is a different model, not a better one.** Under c2's config it ties the six-slot table on fold 0, and averaging the two beats either | val 19 − 6: −0.0015 [−0.0080, +0.0044]; both averaged − 6: +0.0085 [+0.0049, +0.0118] |
| **A third backbone family adds to the ensemble when it matches c2 alone, not when it trails it.** ConvNeXt-tiny equals c2 and adds to c2+c3; EVA02-small trails c2 by 0.045 and subtracts | f1 added to c2+c3: +0.0062 [+0.0040, +0.0086]; f2: −0.0023 [−0.0047, +0.0001] |
| **Averaging slice windows at inference is worth a third model.** Each confirm checkpoint re-read its own fold from 9 stored groups instead of 5, with nothing retrained; the 5-group read reproduced the confirm exactly | c2+c3 pooled OOF 0.8641 → 0.8686, +0.0045 [+0.0034, +0.0055]; gold +0.0109 [+0.0032, +0.0203] |
| **ConvNeXt-tiny is confirmed as the third member.** Slightly below c2 alone on OOF, the best single config on gold, and additive to c2+c3 over five folds | added to c2+c3: +0.0038 [+0.0028, +0.0049] pooled OOF; gold 0.8945 alone |
| **Confidence weights on the loss hurt.** Discounting studies where the label sources disagree lowered the model, as the weak prior recorded beforehand suggested | s3 minus control: −0.0089 [−0.0153, −0.0014] |
| **336px pays as an extra member.** At matched batch it beats 224px, and next to c2+c3+f1 it adds more than one more 224px model does | 336 − 224: +0.0074 [+0.0007, +0.0146]; as a fourth member +0.0021 [+0.0006, +0.0037] against +0.0006 |

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
anchor (+0.24), and both score matrices were kept in `results/bakeoff_scores/` for Phase 4
calibration analysis without another GPU session (local only; not in this repository, see section 8).

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
supervised one. See `NOTES.md` for details.

### Compute

Free tier only — Kaggle CPU/GPU notebooks for all preprocessing and training, Colab free for
ad-hoc experiments. Local machine is code-authoring and unit-testing only (tests run on a small
sample of studies pulled via the Kaggle API), never full training.

### Layout

Layout; every module marked `[done]` exists today:

```
src/knee/            package pushed to Kaggle as a private dataset, imported by thin notebooks
  dicom.py           [done] header scan, series/slice selection, per-header and per-study
                      laterality resolution + census, decode/normalize; also owns
                      StudyDecodeError and the slice-header read both dataset.py and
                      prep.py need (kept here to avoid a circular import), the six- and
                      nineteen-slot tables, and TR/TE weighting recovery
  prep.py            [done] normalize_series, mirror_to_canonical + mirrors_in_plane (only
                      Axial/Coronal flip), JPEG-in-.npz storage with selective decode, the
                      Phase 2 prep_study orchestrator (pad-to-square letterbox), and the
                      Phase 6 prep_slots orchestrator: crop_to_mm gives every study the same
                      physical scale, which the letterbox never did -- studies differ in
                      mm/px by a factor of several and no downstream capacity recovers that.
                      prep_slots files series under the six- or nineteen-slot table
                      (slot_scheme), and window_groups lets a 9-group prep reproduce the
                      5-group artifact's pixels exactly
  dataset.py         [done] KneeStudyDataset over raw DICOMs (Phase 1 shape),
                      PreppedStudyDataset over the Phase 2 .npz artifacts, and
                      PreppedSlotDataset over the Phase 6 slot artifacts -- the last
                      returns (image [slot, group, 3, H, W], presence mask, labels, uid)
                      rather than one flat slice axis, so "this study has no coronal
                      acquisition" is expressible instead of being padded over. It reads
                      either slot table, can yield per-label sample weights, reads a label
                      frame wider than the twelve columns (label_columns, for the
                      multi-source target), and with random_groups draws a random sorted
                      subset of the stored slice groups at train time so the model sees
                      the windows a fixed read never shows it (eval keeps the fixed spread)
  model.py           [done] single backbone, pooled over images; accepts either the flat
                      volume or the slot-structured one and pools only over present slots
                      (an unmasked mean averages anatomy with the black squares that
                      absent slots arrive as), attends over slots per diagnosis
                      (slot_attention), or attends over every (slot, group) token per
                      diagnosis (group_attention) so a label can weigh the one window that
                      shows its finding instead of the mean over a series. Absent slots
                      never reach the backbone, which keeps them out of BatchNorm's batch
                      statistics
  train.py           [done] masked BCE, fold assignment, one epoch, evaluate, experiment logging --
                       used for the real Phase 1 5-fold training run (notebooks/phase1-train/);
                       also Timer (accumulating wall-clock stopwatch feeding train_minutes),
                       load_gold_holdout + train_val_split (Phase 4's explicit gold exclusion --
                       Phase 1 trained on 36 of the 58 gold studies by accident). An epoch runs
                       AMP, a per-step scheduler, optional gradient accumulation and an
                       optional ModelEma updated once per optimizer step; the loss takes
                       optional per-label weights, and folds can be grouped by report text
  infer.py           [done] submission writer, byte-for-byte header, 0.5 fallback, rank mean,
                      and average_source_heads, which folds a multi-source model's 12 x K
                      outputs back to one probability per label
  reports.py         [done] EN/ES lexical rules (kept as a per-label fallback candidate,
                      not replaced) + the Phase 3 LLM generator: verbatim rubric,
                      per-label criterion prompts, yes/no-logit soft scoring, per-label
                      confidence weights from agreement among label sources, and the Phase 8
                      target builders: fold_confidence (a hard label plus a confidence column
                      becomes a graded score), multi_source_labels (one wide frame, a block
                      of twelve per source, for a model with one head per source) and
                      blend_sources (a per-label probability mean that keeps prevalence,
                      unlike the rank blend rejected in Phase 6)
  metrics.py         [done] macro AUC, per-label AUC (NaN-safe), and paired_macro_auc_delta:
                      the bootstrap 95% CI the promotion rule ("the paired delta clears
                      noise") is evaluated against, resampling studies for both arms
                      together
  features.py        Phase 10: frozen-backbone feature extraction and its on-disk cache
  feature_head.py    Phase 10: the per-diagnosis attention readout trained on cached features
  claude_labels.py   Phase 11: report chunks out (uid + report only, never a label column) and
                      strict parsing of the hosted LLM's output back in
  masked_bn.py       Phase 11: BatchNorm weighted by presence, so the TPU path keeps static shapes
                      and still matches the packed GPU model's statistics
  xla_train.py       Phase 11: loss, Adam and OneCycle in forms that do not recompile under XLA
  tpu_farm.py        Phase 11: one independent training job per TPU chip, per-chip launcher
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
  phase7-screen-v3/    nineteen slots against six under c2's config, both arms with absent
                        slots packed out of the backbone, paired in one session
  phase7-screen-family/  ConvNeXt-tiny and EVA02-small on c2's six-slot config, each judged by
                        what it adds to the c2+c3 ensemble rather than by its own macro
  phase7-prep9-shard0..3/  prep v2 re-run at 9 slice groups, whose groups 0,2,4,6,8 are the
                        five the confirm checkpoints trained on -- checked pixel for pixel
  phase7-tta/          the c2 and c3 confirm checkpoints re-reading their own folds at 5 and
                        9 groups: slice-window averaging measured without training anything
  phase7-confirm-f1/   5-fold confirm of ConvNeXt-tiny, the third-member candidate, scored alone
                        and added to c2's and c3's own pooled OOF
  phase7-screen-res/   336px (batch 4 x 2 accumulated) and confidence weights on c2's config,
                        each arm paired with a control that differs in one thing and scored as a
                        fourth member next to c2+c3+f1
  phase7-tta-f1/       the same 5-vs-9 group read for f1's five confirm checkpoints
  phase7-submit-b/     the three-member submission kernel: one 9-group prep pass windowed on the
                        five training groups, and all three members reading all nine
  phase7-confirm-s2/   5-fold confirm of c2's config read at 336px (batch 4 x 2 accumulated), scored
                        as a fourth member next to c2's, c3's and f1's own pooled OOF
  phase8-tta-s2/       the same 5-vs-9 group read for s2's five confirm checkpoints, at 336px --
                        the read that decides whether the fourth member enters Submission C at nine
                        groups like the other three
  phase8-submit-c/     the four-member submission kernel: Submission B plus s2 reading the same
                        9-group windowed artifact at its stored 336px, probability mean over twenty
                        models
  phase8-screen-arch/  Screen A, fold 0: attention over every (slot, group) token, random training
                        windows and EMA weights on c2's config, each pair one variable apart, every
                        arm read at five and nine groups and scored as a fifth member next to
                        c2+c3+f1+s2 (bar +0.0015, CI above zero)
  phase8-screen-target/  Screen B, fold 0: the label target -- steven_v4, one head per source, and a
                        probability blend of the three best sources -- scored against one fixed truth
                        and on gold, never each arm against its own target
  phase8-screen-scale/ Screen C, fold 0: DINOv2-small at 336, ConvNeXt-tiny at 336 and DINOv2-base
                        at 224, each on its family's recipe, scored as a fifth member and against the
                        member it would replace
  phase8-confirm-t2/   5-fold confirm of c2's recipe trained on the probability blend of steven_v4,
                        ours and lixin73 -- the member config Screen B promoted -- scored against the
                        fixed steven_v4 truth and as a fifth member next to c2, c3, f1 and s2
  phase8-screen-windows/  Screen D, fold 0: random training windows on c3's and f1's configs, and
                        a2's config trained on the blend against steven_v4 -- the target question
                        gold could not resolve, asked on 863 studies
  phase8-tta-t2/       the 5-vs-9 group read for t2's five confirm checkpoints, so the blend-target
                        member reads nine windows like the other four in Submission D
  phase8-confirm-a2/   5-fold confirm of the config Screen A promoted: per-diagnosis attention over
                        every (slot, group) token, trained on five of the nine stored windows drawn
                        at random each epoch. Reads five and nine groups in the same session, so it
                        needs no separate TTA kernel, and saves its arrays under both conventions
  phase8-confirm-d3/   5-fold confirm of Phase 8's two findings together: a2's config (group-token
                        attention, random windows) trained on the blend target Screen D promoted,
                        read at five and nine groups, scored as a seventh member
  phase8-submit-d/     Submission D: six members (c2, c3, f1, s2, t2, a2), thirty models, scored 0.926
  phase9-confirm-d3-336/  d3's config at 336px (batch 4 x 2 accumulated), the eighth member (pooled OOF
                        0.8823 at nine groups, the best single model)
  phase9-screen-coatnet/  Screen E, fold 0: our own CoAtNet-1 on d3's recipe; passed the member bar by
                        a hair, not confirmed
  phase9-screen-labels/   Screen F, fold 0: a fourth label source (dreaddevelopment's soft labels); null
  phase9-submit-e/ -f/ -g/ -h/  Submissions E (seven members), F (eight), G (d3 + d3@336, scored 0.928)
                        and H (d3 alone, the member of the blends below)
  phase9-public-stack/ the licence-compliant part of a public notebook (DINOv2 x20, Raptor CoAtNet x4
                        views, four CoAtNet readers; RadImageNet and DINOv3-derived members removed),
                        scored 0.938. Third-party code by the authors credited in its first cell; the
                        edits are asserted text swaps in `tools/phase9-public/build_p.py`
  phase9-submit-blend/ -e/ -f/ -g/ -h/ -i/  the public stack plus our members at a stated weight, with a
                        deadline watchdog from -g's second version on (the public stack's hidden-set run
                        takes 6.3-7.3 h of the 9 h limit; -e and -g's first version timed out). -h at
                        weight 0.5 scored 0.936, -i at 0.2 scored 0.939
  phase10-features-pilot/ -dinov3l/  frozen DINOv3 ViT-L feature cache over the corpus (pilot, then full)
  phase10-g1/          Gate G1: linear and adapter heads on the cache (failed, 0.8076 pooled OOF)
  phase11-screen-b-s0/ -b-s1/ -c-s0/ -d-s0/  GPU student screens on fold 0 (arms b, c, d)
  phase11-submit-d3f0/ -b-s0/ -b-s1/ -c-s0/ -c2-s0/  single fold-0 checkpoints on the LB (decision reads 1-5)
  phase11-tpu-probe/ -probe2/  TPU environment probes
  phase11-tpu-farm1/ -farm2/ -farm3/  eight fold-trainings per TPU session (arms b, c, d; round-2 c2)
  phase11-submit-s15/ -gs25/  the student ensemble (0.925) and the final own-weights entry GS25 (0.928)
tests/               pytest, runs locally on a small sample, no GPU needed
data/sample/         studies pulled via Kaggle API for local dev (gitignored)
checkpoints/         trained model weights + OOF arrays (gitignored, large binaries)
results/             experiment tracking + Phase 2 census outputs, see below. The report-derived label
                     files (`pseudo_labels_qwen3_4b.csv`, `label_confidence.csv`, `bakeoff_scores/`) are
                     gitignored: the competition rules keep report-derived labels private, so they were
                     kept on disk and in private Kaggle datasets only
tools/               offline scripts cited in NOTES.md, kept as records (section 9)
```

### Experiment tracking

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
Phase 4 training target). Only `bakeoff.csv` and `verification_manifest.json` are in this repository;
the score matrices and the label file are report-derived and were kept out (section 8).

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

### Running tests

```
pip install -r requirements-dev.txt
pytest tests/
```

(334 tests at the close.)

Some tests read real competition data from `data/sample/` (gitignored) and are skipped
automatically if that data hasn't been pulled yet.

### Kaggle setup

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
