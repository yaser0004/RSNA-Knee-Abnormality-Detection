# Empirical notes

Raw observations from actually running things — timing numbers, model quirks, data quality
issues, language patterns. Not plans, not intentions. Dated, short, falsifiable. The plan document
stays the stable strategy reference; this is the fast-moving log of what was actually learned.

## 2026-08-08

- Gold-labeled set is **58 studies**, not "a few hundred" as first assumed — verified by reading
  `train.csv` directly. Per-label gold positive counts: ACL 24, MCL 9, Medial Meniscus 26, Lateral
  Meniscus 23, Medial OA 15, Lateral OA 11, PF OA 21, Effusion 35, Synovitis 27, Baker's 12,
  Contusion 19, Fracture 18 (all out of n=58).
- All 4,407 studies have a report; the gold∩report overlap is exactly the 58 gold studies (every
  gold study has a report, none of the report-only studies have gold labels).
- `test.csv` (public example) currently lists only 3 studies — the real hidden test set is ~1,300
  per the competition page, this is just the visible example.
- Kaggle CLI 2.2.4 uses a new single-token auth format at `~/.kaggle/access_token`, not the older
  `username`+`key` JSON at `~/.kaggle/kaggle.json`. A token downloaded from kaggle.com/settings as
  `kaggle.json` in the new format is actually a bare string, not JSON — move it to
  `~/.kaggle/access_token` instead of trying to use it as `kaggle.json`.
- Actual DICOM header check (one test-set slice, Siemens MAGNETOM Avanto fit, Explicit VR LE):
  **`Laterality` (0020,0060) survived the 86-tag allowlist** with a direct value (`'L'`) — laterality
  resolution may mostly be a lookup, not inference from `ImagePositionPatient` sign, on studies where
  this tag is populated. `PatientSex` also survived (confirms host's claim it's in the DICOM header
  even though absent from `train.csv`). `ImageLaterality` (0020,0062) was *not* present in this file.
  Still need to check how consistently `Laterality` is populated across studies before leaning on it
  as the primary route in Phase 2 — one file isn't enough to call this settled.
- **Correction, checked against 20 real studies (one file per study, sampled from `train_series/`):**
  the plan's `ImagePositionPatient`-sign laterality fallback ("+x = patient-left") is **wrong far more
  often than assumed** — `sign(IPP[0])` disagreed with the real `Laterality` tag on **6 of 15
  comparable cases (40%)**, nowhere near the "2-3% silently poisons four labels" risk the plan
  budgeted a validation day for. Root cause: `ImagePositionPatient` is the corner of the first pixel
  relative to a knee-centered coil FOV, not a stable proxy for body-relative left/right — it depends
  on how the coil/FOV was placed for that particular acquisition, not just which knee is being
  scanned. **Removed this fallback route entirely from `resolve_laterality`** (see
  `src/knee/dicom.py`) rather than accept a coin-flip-level heuristic; studies with no
  `ImageLaterality`/`Laterality` tag and no left/right string in `SeriesDescription`/
  `BodyPartExamined` now resolve to `unknown` instead of a guess.
  Also from the same 20-study sample: `Laterality` tag was present on 15/20 (75%), missing on 5/20
  (25%) — so roughly a quarter of studies will need a different resolution route entirely (string
  match, or possibly none available at all). Phase 2's "budget a full day to validate this" note in
  the plan undersold the actual risk here; worth revisiting whether a same-study series majority vote
  or another derivation is needed for that 25% before Phase 2 starts for real. All 20 files checked
  were Explicit VR Little Endian — still zero JPEG-compressed slices seen in any sample so far (60
  total files checked across three separate pulls), so the runtime probe's decode-time extrapolation
  still only covers one transfer syntax.
- Public notebook `nekkon/58-studies-cannot-see-a-0-01-gain` claims the 58 gold studies are
  **"enriched 2x"** relative to the general population (title: "Your CV cannot see a 0.01 gain, and
  the 58 gold studies are enriched 2x") — i.e. the gold subset may not be a random sample, possibly
  oversampled for positive findings. Couldn't get the full notebook body via the fetch proxy (renders
  shell/TOC only), so this is unverified secondhand — worth opening directly on Kaggle to check the
  actual prevalence-shift argument before trusting Phase 3 calibration curves at face value.
- Public notebook `navazshfathi/rsna-knee-abnormality-detection` (0.832 public LB) uses a DINOv2
  backbone, runs on T4x2, 57s execution time — very fast, notably lighter than the plan's Phase 6
  efficiency target of 10-25 min, though unclear if that 57s covers full DICOM decode or a cached/
  preprocessed input. Worth a closer look later for efficiency-track ideas.
- Kaggle's discussion-thread comment count is unreliable through the r.jina.ai fetch proxy — thread
  733592 shows "0 Comments" via direct fetch even though the discussion index lists 1 comment from
  the host (Po-Hao "Howard" Chen). Comments are likely lazy-loaded via a separate XHR call the proxy
  doesn't execute. Need to check this thread directly in a browser if the answer becomes load-bearing.
- Public notebook `gengsr/rsna-knee-roman-v1` (0.807 public LB, DINOv2 backbone) explicitly does
  "normalising left and right" as a preprocessing step — independent confirmation from a top public
  solution that laterality canonicalization matters, same conclusion the plan reached from first
  principles. Runtime ~1h43m on T4x2 — presumably their accuracy submission, nowhere near the
  10-25min efficiency sweet spot, consistent with most public notebooks not targeting Efficiency.
- Local runtime probe (GTX 1650, NVMe SSD, single-threaded, warm cache — **not Kaggle hardware, a
  local lower bound**), on the official 3-study test example set (557 slices, ~600MB total):
  - Header-only read (`stop_before_pixels=True`): 0.58 ms/file → extrapolates to ~1.9 min for
    ~195k slices across 1,300 studies. Confirms CSV+header-based series/slice selection before pixel
    decode is cheap, as the plan assumed.
  - Full pixel decode, Explicit VR Little Endian only (no JPEG-compressed slices seen yet in the
    ~40 files checked so far): 2.54 ms/file → extrapolates to ~8.3 min for 195k slices *if* all
    slices were this transfer syntax, single-threaded. Real number will be higher once JPEG
    Lossless/JPEG2000 slices are included (not yet seen in the sample) and lower in practice from
    multiprocessing across vCPUs. Numbers to be finalized once the full 557-file download completes
    and transfer-syntax diversity can be checked.
  - GPU forward pass (efficientnet_b0, 16 slices @ 224x224/study, GTX 1650 — Kaggle's T4 should be
    faster): 28.1 ms/study → ~37s for 1,300 studies. **Not the bottleneck at all** — decode and I/O
    dominate by an order of magnitude. Confirms the plan's assumption that slice/series selection
    before decode is where the runtime budget actually goes, not the model forward pass.
- **Report language distribution (Phase 0 item 3, 500-report random sample via langdetect):** English
  39.4%, Spanish 16.4%, **Turkish 10.8%, Croatian 10.8%**, Greek 7.2%, German 7.0%, Dutch only 3.6%,
  Bulgarian 2.6%, French 2.2%. This overturns the initial guess (the plan's first draft assumed
  English/Spanish/Dutch based on skimming the first few `train.csv` rows) — **Turkish and Croatian
  are each 3x bigger than Dutch.** Phase 1's lexical rules need to prioritize en/es/tr/hr coverage,
  not en/es/nl. Exactly the kind of premature-freeze mistake the plan text was corrected to avoid.
- Verified directly (not just via the public notebook's vague hint): in `train_series.csv` /
  `test_series.csv`, **`Fluid_Sensitive` and `Fat_Suppression` are identical for all 24,371 rows**
  (`ts['Fluid_Sensitive'].equals(ts['Fat_Suppression'])` → `True`). These two columns carry exactly
  one bit of information between them in this dataset, not two independent signals — likely the
  "column that is lying to you" flagged by `stevenleehans/rsna-knee-101`. Relevant to the model's
  series-descriptor conditioning (Architecture section) — no need to concat both as if independent.
- Lexical rule (`knee.reports.lexical_label`, English/Spanish only) checked against all 58 gold
  studies, per label — coverage (non-None match rate) and agreement (accuracy on the matched subset):
  - Effusion: 62% coverage, 66.7% agreement (36 matched)
  - Baker's: 19% coverage, 81.8% agreement (11 matched)
  - ACL: 12% coverage, 85.7% agreement (7 matched)
  - Medial Meniscus: 16% coverage, **55.6% agreement (9 matched) — barely above chance, n too small
    to trust either way, and the pattern itself may be too loose (currently requires "tear/torn/
    rupture" within 40 chars of "medial meniscus", which real report phrasing may not satisfy)**
  Coverage is low everywhere (12-62%) since these are English/Spanish-only patterns over a corpus
  that's only 56% English+Spanish — expected, per the plan's Phase 1 framing this is a pipeline
  validation baseline, not a coverage-complete labeling system. ACL and Baker's have decent agreement
  where they do match; Medial Meniscus needs its pattern revisited before being trusted in training,
  or should train on a small enough loss weight that a coin-flip-level signal can't hurt much.
- Phase 1's single-series pick (`select_series(..., max_series=1)` → only tries Sagittal +
  Fluid_Sensitive) covers **94.2% of studies (4,150/4,407)** with an exact match; the rest fall
  through to the top-up branch and get an arbitrary series. Close enough to ignore for a pipeline-
  validation baseline — 100% of studies have *some* Sagittal series, just not always fluid-sensitive.
- Caught two real bugs via advisor review before they'd have silently broken training: (1)
  `per_label_auc` crashed on NaN targets (`np.unique` doesn't treat NaN as "one class", so the
  single-class guard let NaN through to `roc_auc_score`, which raises) — lexical labels produce NaN
  for every unmatched study, so this would have broken on the very first real run. Fixed by masking
  NaN rows out per-column before scoring. (2) `masked_bce_loss` didn't exist yet — an unmasked BCE
  loss on NaN targets produces NaN loss/grads with no exception, silently killing the model. Both
  are covered by tests now (`test_metrics.py`, `test_train.py`) using real NaN-shaped data, not just
  clean synthetic data — a reminder that "all green" tests only prove what they actually exercise.
- Corrected the `select_series` fallback: the original implementation only topped up toward
  `max_series` when *zero* priority slots matched — a study with 1 matched + 3 unmatched slots got
  only 1 series back even with room for more. Now tops up whenever `len(selected) < max_series`.
- **Kaggle's dataset zip upload auto-extracts and flattens.** `kaggle datasets create -p . -r zip`
  on a folder containing a `knee/` subfolder does not preserve `knee/` as a folder in the resulting
  dataset — the `.py` files land at the dataset root. Any package uploaded this way needs to be
  reconstructed into a real package folder inside the notebook before `import knee.x` will work
  (copy the flat files into `/kaggle/working/knee/` and add `/kaggle/working` to `sys.path`).
- **The actual `/kaggle/input` layout on this account is namespaced, not the classic flat
  `/kaggle/input/<slug>/` documented in most older Kaggle tutorials/notebooks.** Real structure
  observed: `/kaggle/input/datasets/<owner>/<dataset-slug>/...` and
  `/kaggle/input/competitions/<competition-slug>/...`. A notebook written against the flat-path
  assumption fails immediately with `FileNotFoundError`. Fix: glob for the target folder name under
  `/kaggle/input/**` rather than hardcoding the classic path — this is what
  `notebooks/knee-phase1-smoke-test.ipynb` does now. Worth rechecking whether this is
  account/environment-specific or a genuine platform-wide change before assuming it's universal.
- **Full offline round-trip validated on real Kaggle infrastructure (2026-08-08), 3rd kernel push:**
  private dataset (`rsna-knee-src`) + competition data attached, `enable_internet: false`,
  `enable_gpu: false`. Package import, DICOM decode, model forward pass (untrained, random-init
  efficientnet_b0), and `submission.csv` writing all worked end to end against the real 3-study test
  set. Submission header, shape, and value range all correct (verified by eye in the kernel output).
  **Real single-threaded CPU-only decode+forward timing: 1.4s/study → 30.3 min extrapolated to 1,300
  studies.** This is with sequential decode (no multiprocessing across vCPUs) and no GPU — the
  Phase 6 efficiency submission (parallel decode, fp16, GPU) should land well inside the 10-25 min
  sweet spot; this number is the unoptimized upper bound, not the target. This also finally answers
  the JPEG-transfer-syntax question locally unresolved after 60 sampled files: the real Kaggle test
  set apparently decoded without incident at this speed, though the log didn't break down time by
  transfer syntax — worth adding that breakdown before trusting the number precisely.
  Kernel saved at `notebooks/knee-phase1-smoke-test.ipynb` + `kernel-metadata.json`. **This was a
  smoke test only (untrained model, 8/12 labels clamped to 0.5) — it validates the pipeline, it is
  not the official Phase 1 baseline submission.** The real Phase 1 gate (a scored public-LB entry
  from an actually-trained model) still needs a training run.
- `/code-review` on the working tree found 3 real bugs, 2 confirmed by direct repro: (1)
  `KneeStudyDataset.__getitem__` crashed with a bare `IndexError` on `slices[-1]` if a series
  directory had zero `.dcm` files (e.g. a partial download) — the padding loop indexed an empty
  list. (2) Same method crashed with `IndexError` from `select_series(...)[0]` if a study had no
  rows in `series_df` at all. Both fixed by consolidating series/slice resolution into one
  `_load_image` helper that raises a new named `StudyDecodeError` instead — this doesn't change
  `build_submission`'s existing safety net (it already caught bare `Exception`, so `IndexError` was
  being handled at the inference/submission layer already) but gives training code a specific,
  catchable exception to skip a bad study by instead of crashing the whole DataLoader worker.
  (3) `build_lexical_labels` only caught `ValueError` from `lexical_label`, not the `TypeError` a
  NaN `Report` value would raise from `re.search`. Not observed in the real data (0/4407 rows have
  a NaN `Report` today) but no guard existed — fixed with an `isinstance(report, str)` check per row
  before the label loop, consistent with how every other "no evidence" case in this codebase is
  handled (NaN, not a crash). All three now covered by tests using the real `KneeStudyDataset` /
  `build_submission` composition, not just isolated unit calls.
- `train.py` training/CV building blocks done (54 tests total): `make_folds` (sorts before
  shuffling so the assignment never depends on input order -- important since the plan requires
  the fold assignment frozen across every experiment), `log_experiment` (reads the target CSV's own
  header to build row order, so it can never silently drift out of sync with
  `results/experiments.csv`'s actual columns), `train_one_epoch` (skips a batch if
  `masked_bce_loss` raises on an all-NaN batch rather than crashing the epoch -- possible with a
  small batch size and a low-coverage label like Medial Meniscus), `evaluate` (returns full
  `[n, 12]` y_true/y_pred arrays with NaN preserved, for `knee.metrics.per_label_auc` to mask).
  Not yet built: the actual Kaggle GPU training run that ties these together into the real Phase 1
  baseline (lexical labels from all 4,407 reports, k-fold across them, a genuinely trained
  checkpoint) -- everything so far has been validated on tiny local/synthetic data only, per the
  "local GPU is code-authoring and unit-tests only" constraint.
- **First real Phase 1 training attempt on Kaggle, three separate failures before a clean run:**
  (1) Kaggle assigned a **P100 GPU**, which is Pascal architecture (compute capability sm_60) --
  the PyTorch build in Kaggle's current container image only supports sm_70+ (Turing and newer),
  so the very first real CUDA kernel launch (a batch-norm op inside efficientnet_b0's stem) failed
  with `AcceleratorError: no kernel image is available for execution on the device`. **Kaggle's
  `kernel-metadata.json` has no field to request a specific GPU model** -- confirmed by pulling the
  kernel's actual live metadata after a UI-based accelerator change; there's no `accelerator`/
  `gpu_type` field, only `enable_gpu: true/false`. A T4 selected once in the interactive editor did
  not carry over to a later `kaggle kernels push` -- that run landed back on P100. Fix: probe with
  a real CUDA op (`torch.zeros(1, device='cuda') + 1`, not just allocation) before committing to
  `cuda`, and fall back to `cpu` if it fails, rather than crash. (2) Interactive "Run All" in the
  Kaggle notebook editor is **not reliable for a ~50+ minute job** -- the browser session
  disconnected/reset multiple times, and the editor kept displaying the *last successful run's
  cached cell output* even though the live kernel behind it had restarted and none of those
  variables actually existed any more. This produced a confusing loop of identical-looking pasted
  output followed by `NameError`s for different variables each time, since each attempt actually
  died at a different, arbitrary point after reconnecting. Fix: use `kaggle kernels push` (or the
  editor's "Save Version -> Save & Run All (Commit)") instead of interactive Run All -- these run
  detached on Kaggle's infrastructure, independent of any browser session. (3) Once running on a
  real (P100-forced-CPU-fallback) session, the kernel died with **no Python traceback at all**
  (`Kernel died while waiting for execute reply`) immediately after printing `device: cpu` --
  classic OOM-kill signature. Root cause: `CachedDataset` (as originally written in
  `src/knee/dataset.py`) caches each study's **already-expanded float32, 3-channel tensor**
  (16 slices * 3ch * 224 * 224 * 4 bytes ~= 9.2 MB/study). Across the 2,151 lexically-labeled
  studies this notebook trains on, that's **~19.3 GB held in RAM simultaneously** -- almost
  certainly the actual OOM trigger, on top of whatever the (CPU-fallback) training itself needed.
  The 3 "channels" are identical copies (grayscale MRI repeated only to satisfy an
  ImageNet-pretrained backbone's input shape) and the float32 upcast happens right after a uint8
  decode -- neither is worth persisting per-study. Fix, implemented as a **notebook-local**
  `CompactCachedDataset` (deliberately not a change to the tested library `CachedDataset`, to avoid
  touching working code under time pressure): cache one uint8 channel per slice (~0.8 MB/study,
  ~1.6 GB total, a 12x reduction), reconstruct the float32/3-channel form fresh on every read --
  cheap to compute, expensive to store repeatedly. Verified locally before pushing: round-trip
  reconstruction error is bounded by uint8 quantization (<=1/255) and the shape/dtype contract
  KneeModel expects is preserved exactly. **This validates Phase 2's plan design retroactively** --
  the plan's `prep.py` spec already called for storing prepped slices as uint8 (JPEG-encoded, even
  more compact), not float32; Phase 1's in-memory shortcut hit exactly the problem that design was
  built to avoid. Worth considering whether `src/knee/dataset.py`'s `CachedDataset` should get this
  same fix for anyone else who reaches for it at this scale, rather than leaving the compact version
  notebook-only.
- **Phase 1 baseline completed end to end: trained, submitted, and scored on the real leaderboard.**
  Full sequence of what actually happened, in order:
  - After the OOM fix (previous note), the fixed notebook (kernel push v6) landed on **P100 again**
    -- 3rd P100 out of 3 CLI-triggered pushes so far, 1-in-4 T4 hit rate overall including the one
    interactive session that got T4. The CPU fallback worked this time (no OOM), but CPU training
    of `efficientnet_b0` at this scale is genuinely slow: **~84 minutes per fold for 1 epoch**
    (2151 studies, batch_size=8, ~1720 train studies/fold). At that pace the full 5-fold job would
    have taken ~7.5 hours total. User stopped it after fold 0 finished via the web UI's Stop
    button (there is no CLI command for this). Cancelling still allowed pulling output afterward
    (`kaggle kernels output` works once status is `CANCEL_ACKNOWLEDGED`, not just `COMPLETE`/
    `ERROR`) -- both `experiments.csv` (1 real row, fold 0) and the fold-0 checkpoint survived,
    since disk writes made before a manual stop aren't lost the way in-memory state is.
  - Retried by setting the accelerator to **T4 x2 in the interactive editor's settings**, then
    triggering the run via **"Save Version -> Save & Run All (Commit)"** from that same editor
    session (not a plain `kaggle kernels push`). This combination -- UI accelerator selection +
    UI-triggered detached commit run -- is the one that actually worked reliably: landed on T4,
    ran fully detached (survived fine), completed all 5 folds in ~2.4 min/fold (vs. ~84 min/fold
    on CPU -- roughly a 35x speedup from the GPU alone). **This is the combination to use for every
    future Kaggle GPU training job on this project** -- a plain CLI push has gone to P100 100% of
    the time (3/3) so far, while UI-accelerator-pick + Save&RunAll has gone to T4 100% of the time
    (2/2, counting the very first successful-but-lost interactive run). Small sample, but consistent.
  - **Real Phase 1 baseline result: pooled OOF macro AUC = 0.7985** across all 5 folds (per-label:
    ACL 0.695, Medial Meniscus 0.837, Effusion 0.867, Baker's 0.795; the other 8 labels NaN, no
    lexical rule exists for them). Verified independently by loading the saved `oof_true.npy`/
    `oof_pred.npy` through the project's own `knee.metrics.macro_auc` -- matched the notebook's
    printed number exactly (0.79846...), a useful consistency check that the metrics code and the
    notebook's copy of it (reconstructed from the same `rsna-knee-src` dataset) agree.
  - Also confirmed empirically: **the single fold-0 result (0.86-ish, seen twice: once from the
    CPU partial run, once as the first fold of the full T4 run) was not representative** -- fold 0
    happens to be the easiest of the 5 splits (fold assignment is deterministic, same seed=0 every
    run, so it's the same ~430 studies every time). Per-fold macro AUCs on the full T4 run: fold 0
    0.8614, fold 1 0.8023, fold 2 0.7649, fold 3 0.8123, fold 4 0.8402 -- real spread, and the
    pooled/OOF number (0.7985) sits below every fold except the worst one, which is expected: it's
    not an average of the 5 fold numbers, it's computed by pooling all out-of-fold predictions and
    scoring once, which is more sensitive to the harder studies. This is exactly why the plan
    insists on full k-fold CV rather than trusting a single split.
  - Results integration hiccup: appending the 5 real rows into `results/experiments.csv` via
    `tail -n +2 ... >> results/experiments.csv` got run twice, duplicating all 5 rows (11 lines
    instead of 6). Fixed with `head -n 6` to truncate back to header + 5 unique rows. Worth
    double-checking row counts after any manual append, not just trusting the command ran once.
  - `results/baseline.csv` deliberately NOT written until a real `public_lb` score existed --
    that file is "write once, never changes" per the plan's experiment discipline, and writing it
    with a blank `public_lb` field and coming back to fill it in later would violate that.
  - **Built the real submission notebook** (`notebooks/phase1-submit/`, separate from the
    smoke-test one): loads the trained fold-0 checkpoint via `torch.load(..., weights_only=True)`
    (flagged by the security-guidance hook as good practice even for our own checkpoint -- default
    `weights_only=False` unpickles arbitrary objects), runs offline inference (internet off, GPU
    off -- inference doesn't need GPU per the earlier smoke-test finding, and going GPU-off
    sidesteps the whole P100/T4 lottery for this one), clamps the 8 untrained label columns to 0.5
    (their shared `nn.Linear` head rows never received a gradient during training since
    `masked_bce_loss` excluded them from every batch -- they're still at random init, worse than
    useless to expose), and asserts row count / no-NaN / [0,1]-range before writing `submission.csv`.
    Checkpoint had to be uploaded as its own private Kaggle dataset (`rsna-knee-checkpoints`) since
    a submission notebook can't reach out to load it live (internet off).
  - **This is a Code Competition -- `kaggle competitions submit -f <file>` is rejected outright**
    (`400 Bad Request`, no useful message from the CLI). Confirmed by trying it directly: a raw
    local-file upload doesn't work here regardless of how the file was produced. The actual
    mechanism is the web UI's **"Submit to Competition"** panel, reached from the notebook's
    Output tab -- it links a specific notebook *version*'s *output file* to the competition, which
    gets privately re-run against the real hidden test set (~1,300 studies, vs. the 3 visible in
    the public example) to produce the score. Worth remembering for every future submission on
    this project: build/verify the submission notebook, run it, then submit through that panel,
    not the CLI.
  - **Real leaderboard score: 0.558** (rank ~496 at submission time; leaderboard tightly packed
    0.548-0.559 in that neighborhood, top 10 at 0.90-0.94). Backed out what this implies about the
    4 trained labels' real-world performance: since the other 8 columns score exactly 0.5 (constant
    prediction, zero ranking power), `0.558 = (4*X + 8*0.5)/12` gives `X ≈ 0.674` -- the true
    average AUC of the 4 lexically-trained labels against the real doctor-graded rubric, down from
    their ~0.80 average against the lexical-label OOF validation. **That ~0.13 absolute gap is the
    first real, quantified measurement of how much accuracy is lost between "keyword-matched proxy
    label" and "actual gold-standard rubric label"** -- exactly the risk the plan flagged as its
    central assumption from the start, now measured for real rather than assumed. It's a genuine
    data point for deciding how much Phase 3's calibrated-LLM labeling is worth building: even a
    fairly crude proxy label transferred *some* real signal (0.674 >> 0.5), which is the core
    justification for the whole weak-supervision strategy, but the gap also shows plenty of room
    for a better label source to close.
- **Closed the Phase 1 gate, with one deliberate substitution.** The gate calls for three things: a
  scored public-LB entry, a recorded lexical-label OOF macro-AUC, and a gold LOO recalibration
  number. The first two are done (0.558 LB, 0.7985 pooled OOF). The third was skipped on purpose:
  checked overlap between the 58 gold studies and the 2,151-study lexically-labeled pool Phase 1
  actually trained on (`df['StudyInstanceUID']` intersection, keyed off which of the 4 trained
  labels' lexical rule fired per study) and **36 of the 58 gold studies (62%) are inside the
  training set**. A frozen-feature LOO fit from `checkpoints/knee_phase1_fold0.pt` would be
  evaluating the model partly on data it was trained on -- not a valid transfer measurement, and
  not worth a Kaggle session to compute. The plan's own Phase 3 recipe requires holding gold out of
  pretraining before a LOO number means anything; Phase 1 never did that because it wasn't gated on
  gold at all. **Substituted a better measurement that already exists:** the 0.558 LB back-out above
  (X ≈ 0.674 report-target-vs-rubric-target transfer, on ~390 LB studies -- 6.7x the local gold
  set) answers the same underlying question ("does report-derived performance track rubric
  performance") with more statistical power, not less. Recorded as the gate-closing evidence instead.
  **Forward-looking constraint this creates:** if a clean 58-study gold LOO is ever wanted later
  (e.g. for the Phase 4 gold transfer-check tier), the 58 UIDs need an explicit holdout flag applied
  *before* any pretraining -- tag them during Phase 2 prep so Phase 3/4 training can't silently
  include them the way Phase 1 did.
- **`results/baseline.csv` written** (was empty pending a real `public_lb`, which now exists):
  pooled OOF numbers (macro 0.7985; ACL 0.695, Medial Meniscus 0.837, Effusion 0.867, Baker's 0.795,
  the other 8 labels blank -- untrained), not fold 0's easier 0.8614, since fold 0 is not
  representative (see above) and every future delta should compare against the honest pooled
  number. `fold_set=primary_v1` (pooled across all 5 folds, not `primary_v1_fold0`). `train_minutes`
  (12.0) and `inference_seconds` (11.18) are approximate: 12.0 is 5 folds x ~2.4 min/fold on T4 from
  the session narrative, not from `experiments.csv` (whose `train_minutes` column is 0.0 for every
  row -- never actually populated by `train.py`, worth fixing before Phase 4's experiment logging is
  trusted). 11.18 is the mean of the 5 folds' *validation*-inference seconds from `experiments.csv`
  (~430 studies each), **not** the true 1,300-study submission-notebook wall time the plan wants for
  the efficiency track -- that number was never captured because Kaggle's "Submit to Competition"
  panel doesn't surface per-run timing outside the notebook UI itself. Flagging this so a future
  session doesn't compare the recorded `inference_seconds` against the efficiency formula's
  `RuntimeSeconds` as if they were the same thing; getting the real number means checking the
  submitted notebook version's run log on Kaggle directly.

## 2026-08-09

- **Started Phase 2.** Added `resolve_study_laterality` (`src/knee/dicom.py`) on top of the existing
  per-header `resolve_laterality`: resolves a study from one representative header per series, and
  returns `('conflict')` rather than a majority vote when resolved series disagree -- the plan
  flagged same-study majority-vote as untested against real data, and a knee study shouldn't have
  two different sides across series in the first place, so a disagreement is more likely a bad tag
  on one series than a real majority worth trusting blindly.
- Added `census_study_laterality` (walks `dcm_root/<StudyUID>/*/`, one header read per series) to
  answer both open Phase 2 gate questions in a single pass: laterality-route coverage and the
  per-series slice-count distribution. Tested it against the local 20-study sample first --
  **discovered the local sample only has 1 downloaded file per study** (not the full series), so
  every local `n_series`/`slice_counts` reads as `1`/`[1]`. This is a real limit of the local dev
  sample, not a bug: the local box is a code-correctness harness only, per the plan's compute
  section, and the real distribution requires the actual multi-series Kaggle data. Local route
  counts on this 1-file sample: 11/20 via `Laterality`, 9/20 `unknown` -- lower than the ~75%
  `Laterality`-tag coverage an earlier session logged, plausibly because a single file per study
  sometimes isn't the file that happened to carry the tag; not treated as a real coverage number,
  just a shape/round-trip check that the function doesn't crash on real files.
- **Pushed `knee-phase2-laterality-census`** (CPU, internet off, `kaggle kernels push` -- per the
  earlier finding that CLI push only mattered for the GPU accelerator lottery, which a CPU-only
  metadata pass doesn't hit) to run the census across the real ~4,407-study corpus: study-level
  route counts overall and restricted to the 58 gold studies, a route-vs-route agreement check
  (does `ImageLaterality` agree with `Laterality`/string-match when more than one resolves, sampled
  over the first 3,000 studies) via a new `_all_routes` helper kept notebook-local (not worth adding
  to `dicom.py` for a one-off diagnostic), and series-level + per-study slice-count percentiles.
  Bumped `rsna-knee-src` to a new dataset version first so the notebook picks up the new functions.
  Results pending -- see below once the run completes.
- Refactored `decode_and_normalize` to extract `read_rescaled_pixels` (raw float32, slope/intercept
  applied, no clip) and `percentile_clip_to_uint8` (renamed from a module-private helper so
  `prep.py` can share it without reaching into another module's underscore-prefixed internals) --
  pure extract-method, no behavior change; all pre-existing decode tests still pass unchanged.
- **`src/knee/prep.py` skeleton, TDD, local tests only (no full-corpus run yet -- that's gated on
  the census results above):**
  - `normalize_series`: the per-series percentile clip `decode_and_normalize`'s docstring deferred
    to this module. Computed once across every slice in a series (not per slice), so a uniformly
    bright slice lands near the bright end of the series' own range instead of being independently
    flattened.
  - `mirror_to_canonical`: flips a slice left-right when its resolved side isn't the canonical
    handedness; a no-op when `side is None` (unresolved) rather than guessing -- unresolved studies
    stay the caller's responsibility to exclude from laterality-dependent training, per the plan.
  - `save_study_npz`/`load_study_npz`: one `.npz` per study, JPEG-encoded (q=92) slices per series,
    round-tripped through a single pickled payload dict (metadata + encoded series) rather than one
    npz array key per series, since `SeriesInstanceUID` strings aren't guaranteed clean array keys.
    Uses `allow_pickle=True` -- flagged by the security-guidance hook; documented inline in
    `prep.py` why it's safe (every `.npz` this pipeline reads was written by this same pipeline onto
    a private Kaggle dataset, never third-party or competition-supplied data). Round-trip test
    bounds JPEG's lossy error (mean abs diff < 8/255) rather than requiring exact equality.
  - Not yet built: the orchestrating `prep_study` that ties `select_series` + `order_slices` +
    `select_k_evenly_spaced` + `normalize_series` + `mirror_to_canonical` + `save_study_npz`
    together into the actual per-study prep step, and the sharded Kaggle CPU notebook that runs it
    across all 4,407 studies. Deliberately deferred until the census above answers how common
    `unknown`/`conflict` laterality is at full scale -- that number decides whether `prep_study`
    needs a majority-vote fallback route or can ship with the current three-tag resolution order.
- **Laterality census results, real ~4,407-study corpus (`results/laterality_census.csv`,
  `results/series_laterality_routes.csv`), kernel completed in 200s (45.3 ms/study), CPU-only:**
  study-level route counts: `Laterality` 2179 (49.4%), `unknown` 2129 (48.3%), `SeriesDescription`
  74 (1.7%), `conflict` 25 (0.6%). **`ImageLaterality` matched zero studies** -- despite being
  first in the resolution priority order, the tag appears to be entirely absent or empty across
  this dataset (worth a direct spot-check later, but 0/4,407 with a working `Laterality` fallback
  strongly suggests it just isn't populated here, not a bug in the reader). **The gold-labeled set
  is worse than the corpus average: 31/58 (53.4%) unknown, only 27/58 (46.6%) resolve at all** --
  meaningfully more of the calibration set is laterality-blind than the general population.
  **This overturns the plan's working assumption.** The 20-study local sample (Phase 0/2 planning)
  suggested ~25% unresolved; the real number is **48.3%, nearly double**. `SeriesDescription`
  string-match, hoped to be a meaningful fallback, only rescues 1.7% of studies -- not the "roughly
  a quarter recovered by string-match" the plan's Phase 2 section speculated about. Conflict rate
  (series within a study disagreeing) is reassuringly low at 0.6%, and among the 24 studies (of the
  first 3,000 sampled) where two independent routes both resolved, they agreed 100% of the time --
  so where a route *does* fire, it's trustworthy; the problem is coverage, not accuracy.
  Series-level slice counts: p50=30, p95=45, p100=320 (matches the plan's "median 30, long tail to
  a few hundred" expectation). Per-study total (all series summed): p50=162, p95=369, p100=632. Note
  for `prep_study`, not yet built: **p0=11**, so some real series have fewer than the planned K=24
  slices. `select_k_evenly_spaced` already handles n<=k by returning all n without raising, but
  nothing pads it back up to K yet -- `KneeStudyDataset._load_image` (Phase 1) pads by repeating the
  last slice; `prep_study` needs the equivalent decision made deliberately, not discovered later at
  training time.
- **Bug found and fixed via this census, before it reached prep.py:** 20 studies carry `Laterality`
  values as spelled-out `"RIGHT"`/`"LEFT"` (1 study has `"B"`, DICOM's bilateral code) instead of
  the standard single-letter `R`/`L` code string -- real-world tag messiness the plan's "verify
  before trusting" lesson exists for. `resolve_laterality` returned these raw and unnormalized;
  none of the 20 happened to land in the `conflict` bucket (each study's series were internally
  consistent), but downstream `prep.py.mirror_to_canonical` compares `side != canonical` directly,
  so `side="RIGHT"` would have been treated as a different, non-canonical side from `"R"` and
  flipped -- silently mirroring an already-correctly-sided image. Fixed: added `_normalize_side`
  in `dicom.py` (`RIGHT`->`R`, `LEFT`->`L`, case-insensitive) applied at both the `ImageLaterality`
  and `Laterality` resolution points, and tightened `mirror_to_canonical` to only flip on a
  recognized `{"L", "R"}` side -- `"B"` (bilateral) and any other non-standard code now pass through
  unmirrored, same as `side=None`, rather than being guessed at. This fix doesn't change the
  published route/coverage counts above (normalization only affects which literal string `side`
  holds for those 20 studies, not which route resolved them or whether they conflicted), so the
  Kaggle census didn't need re-running.
- **Consequence for Phase 2 design.** At 48.3% unknown (53.4% on gold), the plan's Phase 2 gate
  ("laterality resolved for 100% of studies by an explicit route") is unreachable with tag-based
  resolution alone -- amended the gate line in the plan doc rather than leaving it looking unmet
  (see below). Same-study majority-vote is already ruled out, not an open option: conflict is only
  0.6%, so almost every unknown study has *zero* series resolving, not disagreeing series to vote
  between. **This costs almost nothing today:** of the four side-dependent labels
  (Medial/Lateral Meniscus, Medial/Lateral OA), only Medial Meniscus currently has any lexical
  training signal at all (Phase 1), and its lexical-rule gold agreement is already only 55.6% on 9
  matched studies -- barely above chance. The coverage gap becomes load-bearing in **Phase 3**, once
  the calibrated LLM starts producing labels for all four side-dependent labels across the full
  report set -- that is where the exclusion-vs-recovery decision actually needs to be made, not now.
- **Multi-instance verification, closes the recovery question above.** The census read only the
  first sorted instance per series; `Laterality` is documented as a series-level DICOM attribute so
  that should be equivalent to reading every instance, but `ImageLaterality` matching 0/4,407 studies
  was suspicious enough to check directly rather than assume. Sampled 50 studies from the `unknown`
  bucket, read **every** instance of **every** series for each (`results/laterality_verify_sample.csv`,
  8,654 instances total) via `read_laterality_header` instead of `[0]`. **Zero instances, anywhere in
  the sample, carried `ImageLaterality` or `Laterality`.** The 48.3% unknown figure is confirmed real
  and structural -- not an artifact of only sampling one file per series -- so there's no cheap
  per-instance recovery path left to try; the only remaining options are the ones already named above
  (accept the exclusion, or find a genuinely different signal) and neither is due until Phase 3.
