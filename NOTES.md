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
- Caught two real bugs in review before they'd have silently broken training: (1)
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

### Phase 2 prep pipeline — local dry run before the Kaggle pilot (2026-08-09)

`prep_study` (orchestrator) and `PreppedStudyDataset` (prepped loader) written and run end to end
against `data/sample/test_series` — one real study, 5 series, 30/30/34/108 slices, prepped at
K=24 / 256px / max_series=4. Numbers below are local hardware on one study; the Kaggle pilot
(`notebooks/phase2-prep/`, 50 studies) is the authoritative measurement. Recorded here because two
of them already change decisions.

- **Non-square matrices are real.** The four stored series report `(Rows, Columns)` of
  `(640, 640)`, `(640, 1280)`, `(800, 800)`, `(960, 960)`. The 640×1280 axial is 1:2 — resizing it
  to a square 256×256 squashes it horizontally by 2×, and does so unevenly across the corpus since
  other series are square. This is the pad-to-square question the pilot's shape counter exists to
  answer, and it fired on the very first real study rather than staying hypothetical. Not fixed
  yet: the corpus-wide frequency decides whether it's a handful of series or a systematic bias, and
  guessing at 4,407 studies is exactly what the census discipline exists to prevent.
- **Transfer syntax / photometric:** all four series Explicit VR Little Endian
  (`1.2.840.10008.1.2.1`), all MONOCHROME2. No MONOCHROME1 (which would be tone-inverted and which
  `percentile_clip_to_uint8` does not correct for) on this study — the pilot checks the corpus.
- **Size: 1.59 MB/study** at 4 series × 24 slices × 256px, JPEG q=92 — above the plan's 0.7–1.5
  MB/study estimate, though this study stores the full 4-series budget and the census median is
  fewer series. Extrapolates to ~7 GB for 4,407 studies, inside the plan's 4–8 GB total.
- **Prep throughput: ~1.9 s/study** → ~2.3 h for the full corpus on local hardware, comfortably
  inside one 12 h Kaggle session even with CPU-speed margin. Sharding stays the resume mechanism
  regardless (`/kaggle/working` starts empty each run).
- **Loader latency: 32 ms/study, against the plan's <20 ms gate — and the cause was not what the
  plan assumed.** Breaking down the 96-slice load:

  | step | ms |
  |---|---|
  | `.repeat(1, 3, 1, 1)` to 3 channels | 21.9 |
  | 96 PIL JPEG decodes | 20.6 |
  | mirror + stack + `/255` | 10.7 |
  | npz open + unpickle | 1.4 |

  The single largest cost was the grayscale→3-channel expansion, not the JPEG decode the plan
  flagged: `repeat` materializes a `96 × 3 × 256 × 256` float32 copy — **75 MB per study** — and
  then collate copies it again into the batch. Switched `PreppedStudyDataset` to
  `.expand(-1, 3, -1, -1)`, a view: byte-identical values (asserted in the tests), 56 ms → 32 ms,
  and one materialization instead of two. Verified through a real `DataLoader` at
  `num_workers` 0 and 2 — the batch collates contiguous as normal.

  32 ms still misses the <20 ms gate, and the prepared fix in the plan (`slice_indices` on
  `load_study_npz`, decode a subset) only addresses the 20.6 ms decode — training needs all 96
  slices, so it would buy nothing there. It remains the right fix for the efficiency-track
  submission (2 series × 12–16 slices). For training throughput the honest answer is the existing
  `CachedDataset` plus DataLoader workers: 32 ms/study is ~60× faster than the 1.9 s raw-DICOM path
  it replaces, and the first epoch is the only one that pays it. **Gate 1's real number comes from
  the Kaggle pilot; this is the local prior, not the verdict.** Cutting K to make the gate pass is
  still off the table — that trades training signal for a loader problem.
- **75 MB/study of float32 is a Phase 4/5 memory constraint**, not a Phase 2 one: a batch of 8 at
  4 series × 24 slices × 3 channels × 256px is ~600 MB before the model allocates anything. Named
  here so it lands as a known budget rather than a surprise OOM at the first training batch.
- **One truncated file exists in the local sample** (`data/sample/test_series/.../44334485…/…198.dcm`,
  0 bytes — a partially completed API download, not a corpus defect). It surfaced a real hole:
  header reads can fail as well as pixel decodes, and one unreadable file was killing the whole
  study. `prep_study` now records header failures and pixel-decode failures separately in
  `meta["decode_failures"]` and skips an unusable series rather than the study, raising
  `StudyDecodeError` only when no series survives.

**Layering change made while implementing this:** `StudyDecodeError` and `_read_slice_header` moved
from `dataset.py` to `dicom.py`. `prep.py` needs both, and `dataset.py` needs `prep.py`'s
`load_study_npz`/`mirror_to_canonical` — importing across would have been circular, and would also
have dragged `torch` (via `dataset.py` → `infer.py`) into the CPU-only prep notebooks. `dicom.py` is
the layer both already depend on. Both names are re-exported from `knee.dataset` so existing callers
and tests are unaffected. `_read_slice_header` also now returns the acquisition tags the pilot's
decode census counts (`TransferSyntaxUID`, `PhotometricInterpretation`, `Rows`, `Columns`,
`PixelSpacing`, `PatientSex`) — it is already called on every slice for ordering, so this costs
nothing and describes the slices prep actually read rather than a separate sweep over files.

**Deviation from the plan doc's Phase 2 wording, recorded so it isn't re-litigated:** artifacts store
*unmirrored* pixels plus `side`/`route`; `mirror_to_canonical` runs at load time in
`PreppedStudyDataset`. The plan's "canonicalize laterality — required, not optional" is about what
the model sees, and that still holds. Doing the flip at load keeps ~7 GB of artifacts neutral to a
laterality resolution Phase 3 may still improve (48.3% unknown today), so better coverage later
costs no re-prep.

### Phase 2 Kaggle pilot — 50 studies, the four blocking gates (2026-08-10)

`notebooks/phase2-prep/` at `PILOT_N=50`, K=24 / 256px / max_series=4, CPU kernel, internet off.
Consumer kernel `notebooks/phase2-prep-consume/` run against its output for gate 2.

**Gate 2 — aggregation hop: PASS.** The prep kernel's output attached to a separate consumer kernel
via `kernel_sources` and landed at `/kaggle/input/notebooks/<user>/knee-phase2-prep/prepped`.
**50/50 `.npz` files present, 0 missing, 0 unexpected, 0 unreadable**, all 50 read back through
`PreppedStudyDataset`. The file-count question the plan called the main operational unknown is
answered: individual files survive the hop, no Kaggle-API dataset-creation fallback needed.

**Gate 3 — decode coverage: PASS.** 200 series across 50 studies, every one stored (4/4 series for
all 50 studies), **zero decode failures, zero header failures, zero skipped series**.
- `TransferSyntaxUID`: 200/200 `1.2.840.10008.1.2.1` (Explicit VR Little Endian, uncompressed). No
  JPEG Lossless, no JPEG 2000 — the pydicom-plugin risk the plan flagged does not exist in this
  corpus. Phase 1 seeing only one syntax was representative, not lucky.
- `PhotometricInterpretation`: 200/200 MONOCHROME2. **No MONOCHROME1**, so the tone-inversion fix
  `percentile_clip_to_uint8` would have needed is not required. Counted rather than assumed, which
  was the point.
- Matrix size: **10/200 series (5%) are non-square**, worst ~1.23:1. **Superseded — see the spread
  re-run below; the contiguous sample understated this.**
- `PixelSpacing`: **72 distinct values** across 200 series (0.234–0.5625 mm). Resizing every series
  to a fixed 256px therefore hands the model a different physical field of view per study — a Phase
  4 modelling consideration, not a prep bug, but it is now a measured number rather than a guess.
- Slice counts: **40/200 series (20%) store fewer than 24 slices** (min 15 originals). Storing the
  true count and padding at load is load-bearing for a fifth of the corpus, not an edge case.

**Gate 4 — size budget: PASS on the total.** 1.46 MB/study mean, p50 1.47, p95 1.80, max 1.99 —
slightly over the plan's 0.7–1.5 MB/study band, but every study here stored the full 4-series
budget. Extrapolates to **6.4 GB for 4,407 studies**, inside the plan's 4–8 GB target.
Prep throughput **1.95 s/study → ~2.4 h for the full corpus**, comfortably inside one 12 h session,
so `N_SHARDS = 1` is sufficient (sharding stays available as the resume mechanism regardless).

**Gate 1 — loader latency: MISSES the <20 ms target at the training configuration.** Measured on
Kaggle CPU over the 50 pilot artifacts:

| configuration | volume | ms/study (mean) | p95 |
|---|---|---|---|
| 4 series × 24 slices (training) | 96×3×256×256 | **58.9** | 66.6 |
| 2 series × 16 slices (efficiency track) | 32×3×256×256 | 22.4 | 25.5 |
| 2 series × 12 slices | 24×3×256×256 | **17.6 — under 20** | 20.0 |
| raw DICOM path this replaces | — | **1945** | — |

Two fixes were applied before this measurement, both real:
- `.repeat(1,3,1,1)` → `.expand(-1,3,-1,-1)` for the grayscale→3-channel step. Byte-identical values
  (asserted in tests), but a view instead of a 75 MB float32 copy per study. The copy had been the
  single largest cost in the loader — larger than all 96 JPEG decodes — and collate materializes the
  batch anyway, so this removed one of two materializations.
- `load_study_npz` gained `max_series` / `max_slices`, and `PreppedStudyDataset` now decodes only
  the blobs it returns instead of decoding 4×24 and discarding most of it. Cost is close to linear
  in blobs decoded, which is what makes the efficiency-track row above viable.

Cutting K to make the number pass remains off the table — that trades training signal for a loader
problem. What is left at 4×24 is ~85% irreducible PIL JPEG decode of 96 slices.

**The gate's threshold is missed; its purpose is met.** 58.9 ms/study × 4,407 = **~4.3 min of
loading per epoch**, against ~2.4 h for the raw-DICOM path — a **33× speedup**, which is what Phase 2
existed to buy. Loading is no longer the bottleneck: 4,407 × 96 = 423k slice forward/backward passes
per epoch is the dominant cost, and DataLoader prefetch overlaps loading with it. (That comparison is
an estimate — GPU epoch time has not been measured at this configuration, and should be before the
plan's ~42-experiment budget is committed.) One measured caveat: DataLoader `num_workers` > 0 did
**not** help locally (0/2/4 workers all ~65 ms/study) because each sample is large enough that IPC
serialization offsets the parallel decode — prefetch overlap, not worker count, is the mechanism
that hides this. `CachedDataset` drops epoch 2+ to ~0 ms/study but cannot hold the full corpus in
RAM (4,407 × 25 MB), so it stays a small-subset tool.

**Pilot re-run on a corpus-spread sample (108 studies) — gate 3 confirmed, and one earlier
conclusion overturned.** The first pilot took `all_study_uids[:50]`, a contiguous prefix, and drew
corpus-wide conclusions from it — the opposite of how the laterality census was done, and it caught
only **1 of the 58 gold studies**, the only ones with real labels. Re-ran over a stride sample
unioned with every gold study: **108 studies, 432 series, all 58 gold included**.

Confirmed and strengthened: **432/432 series** Explicit VR Little Endian, **432/432 MONOCHROME2**,
**zero decode failures, zero header failures, zero skipped series**, 4/4 series stored for all 108
studies. Route mix (`Laterality` 53 / `unknown` 52 / `conflict` 2 / `SeriesDescription` 1) now
reproduces the shape of the full census, which the prefix sample did not. Size 1.43 MB/study mean
(**6.3 GB** extrapolated), prep 2.14 s/study (**~2.6 h** full corpus). Loader: **53.4 ms/study** at
4×24, 20.1 at 2×16, 15.6 at 2×12.

**Overturned: non-square series are more common and more extreme than the prefix suggested, and
2:1 series exist in the training set after all.** 34/432 series (**7.9%**) are non-square, worst
ratio **2.0:1** — including four 640×1280 series, the exact shape the local dry run found in the
*test* sample. The earlier note here claimed that aspect ratio was test-set-only and framed
pad-to-square as a Phase 6 inference-consistency question; that was an artifact of sampling one
contiguous block, and is wrong. Resizing straight to 256×256 squashes those 34 series by up to 2×
horizontally, unevenly across the corpus. Full distribution: 320×300 (9), 384×348 (6), 640×1280 (4),
348×384 (3), 640×540 (2), 640×580 (2), 512×480 (2), and one each of 384×332, 512×384, 512×544,
544×512, 640×680, 808×800.

Also confirmed: **81/432 series (18.8%) store fewer than 24 slices**, so pad-at-load is load-bearing
for nearly a fifth of the corpus, not an edge case.

**Hardening applied before the re-run**, all of it aimed at the full run surviving:
- The shard loop now catches `Exception`, not just `StudyDecodeError`. `select_k_evenly_spaced`
  raises `AssertionError` by design, a malformed `series_df` row raises `KeyError`, and a 320-slice
  series can raise `MemoryError` — any of which would otherwise abort a multi-hour shard at study
  4,000 and lose every artifact with it. Failures are recorded per study in
  `prep_failed_shard*.csv` rather than being silently absent from the count.
- `prep_failed_shard*.csv` is written with explicit columns, so the expected zero-failure case
  produces a readable file rather than a headerless one `read_csv` rejects.

### Phase 2 full corpus run + visual check (2026-08-10)

Four shard kernels (`notebooks/phase2-prep-shard0..3/`), identical code differing only in
`SHARD_INDEX`, run in parallel on Kaggle CPU. Consumer kernel re-verified the union.

**Full corpus prepped and verified: 4,407/4,407.** Shard sizes 1102/1102/1102/1101, **0 duplicated,
0 missing, 0 unexpected, 0 studies failed prep, 0 artifacts unreadable** (all 4,407 read back).
**All 58 gold studies present.**

Verified as a real partition, not just a matching total: each shard's UID set is *exactly* its own
contiguous index range over the sorted corpus — `[0,1102) [1102,2204) [2204,3306) [3306,4407)`, all
four reported "exact". A count of 4,407 with 0 duplicates could in principle hide two shards
swapping studies; this checks the ranges themselves, which is what the resume model depends on (a
re-run of shard *i* has to reproduce shard *i*'s studies and nothing else).
`gold_study_uids.csv` (written by shard 0 only) is present, holds **58 UIDs, all of them prepped,
and exactly matches the `is_gold` flag across the four manifests** — the gold holdout Phase 3/4
requires is on disk and consistent, not merely assumed from a count. Route mix over the full corpus reproduces the published census
exactly: `Laterality` 2179 (49.4%), `unknown` 2129 (48.3%), `SeriesDescription` 74 (1.7%),
`conflict` 25 (0.6%) — matching `results/laterality_census.csv` to the tenth of a percent, which is
the independent confirmation that `prep_study` resolves laterality the same way the census did.
Loader over the attached 4-shard dataset: **39–66 ms/study** across two consumer-kernel runs (39.2
p95 44.8, then 65.7 p95 73.9 on a later run reading the same data) — the spread is Kaggle CPU
contention, not a code change, and is worth remembering before treating any single loader
measurement as precise. Gate 2 re-confirmed at full scale, not just on the 50-file pilot.

**The visual check earned its place — it caught a real bug.** `mirror_to_canonical` was being
applied to every series regardless of acquisition plane. Verified against real
`ImageOrientationPatient` direction cosines:

| plane | row direction (image horizontal) | in-plane flip valid? |
|---|---|---|
| Axial | dominantly patient **+x** | yes — horizontal axis is medial-lateral |
| Coronal | dominantly patient **+x** | yes |
| Sagittal | dominantly patient **+y** | **no** — horizontal axis is anterior-posterior |

Flipping a sagittal slice mirrors the knee **front-to-back**, not side-to-side. It was doing this to
L-side studies only, leaving R-side ones untouched — introducing exactly the systematic L-vs-R
difference canonicalization is supposed to remove. Scope: `_SERIES_PRIORITY` puts sagittal first and
fourth, so **2 of the 4 stored series per study** were affected, on ~49% of the corpus.

Fixed with `mirrors_in_plane()` in `prep.py`; `PreppedStudyDataset` now reads each series'
`Anatomical_Plane` from the artifact's meta and only flips Axial/Coronal. An unknown plane is not
flipped, consistent with the rest of the pipeline's refusal to guess.

**The store-unmirrored decision paid for itself here.** Because artifacts hold raw orientation plus
`side`/`route`, this was a loader-only fix — no re-prep of 6.2 GB, no re-run of four shard kernels.
Had mirroring been baked in at prep time (the plan's literal Phase 2 wording), the corpus would have
needed regenerating.

**Still open: sagittal laterality is not canonicalized at all.** For a sagittal series the
medial-lateral axis runs along the *slice normal*, i.e. the slice ordering — a left and a right knee
traverse medial→lateral in opposite directions through the stack. Canonicalizing would mean
reversing slice order for one side, which is a different mechanism from an in-plane flip and changes
what the depth axis means to the model. Not implemented: it is a Phase 4/5 modelling decision
(attention pooling over slices may be order-invariant anyway), and the artifacts stay neutral to it.
Not doing it is strictly better than the flip that was there, which was actively wrong.

**Visual check result (`notebooks/phase2-visual-check/`):** 8 studies × 4 series, mid-slice,
including 2 `route == "unknown"` studies. All recognizably knees, in the labelled plane, correct
slice counts. Captions confirm the fix reads correctly: `L` + Axial/Coronal → flipped, `L` +
Sagittal → as-stored, `R` and `unknown` → as-stored. (One iteration was needed on the caption logic
itself: it initially labelled a series "flipped" whenever a side was resolved, but
`mirror_to_canonical` no-ops when the side is already canonical, so R-side studies were mislabelled.
The pipeline was correct; a gate artifact that misreports is worse than none, so it was fixed and
re-run.)

**Phase 2 gate: closed.** 4,407 artifacts, 6.2 GB, all 58 gold studies, explicit laterality route on
every study, decode coverage counted with every distinct value accounted for, artifact hop proven at
full scale, and the visual check reviewed.

## 2026-08-17

### Phase 3 Step 0 — a bug found while re-verifying the Phase 3 plan against current code

Before spending any Kaggle GPU time on Phase 3, re-checked the two prior Phase 3 plan drafts against
the code as it actually stands post-Phase-2. Found one real bug: `load_study_npz` was limiting
series with `sorted(stored)[:max_series]` — alphabetical by `SeriesInstanceUID`. `prep_study` picks
series by `_SERIES_PRIORITY` (sagittal-fluid-sensitive first) before storing them, but once stored
they live in a plain dict keyed by UID, so that ordering was lost. Harmless at `max_series=4`
(everything loads regardless of order), but Phase 1's config trains on a *single* series — at
`max_series=1`, `PreppedStudyDataset` would have silently handed back whichever series sorts first
alphabetically, not the sagittal-fluid-sensitive one Phase 1 actually trained on. Same failure shape
as the sagittal-mirroring bug from Phase 2 (a function's real precondition not checked before wiring
it into a new call path) — never triggered before because nothing had called `max_series=1` against
real multi-series artifacts until now.

Fixed with a `_priority_rank` sort key in `prep.py` that reconstructs `_SERIES_PRIORITY` order from
each series' stored `Anatomical_Plane`/`Fluid_Sensitive` meta, falling back to alphabetical-by-UID
for anything unmatched. `load_study_npz` now returns series in that order, and
`PreppedStudyDataset._load_image` iterates the dict as returned instead of re-sorting it — no
re-prep needed, loader-only fix. New test:
`test_load_study_npz_orders_series_by_priority_not_alphabetically`. 97 tests pass (was 96).

Also written locally (no Kaggle needed): `results/folds_primary_v2.csv` (frozen `make_folds` over
all 4,407 `train.csv` study UIDs, `n_folds=5, seed=0` — near-even 882/882/881/881/881) and
`results/gold_study_uids.csv` (the 58 studies where all 12 label columns are non-null; all 58 also
have non-null `Report` text, so Step 2's gate has no report-availability gap to work around).
`results/baseline.csv` is deliberately left untouched — it holds the one row that matters, the
actual scored Phase 1 submission (`primary_v1`, LB 0.558); a `primary_v2` rerun of the same frozen
config is a new experiment to log later, not a new baseline, since it won't be separately submitted.

**Lexical-vs-gold anchor, and it's noisier than expected.** Ran the existing lexical rules
(`build_lexical_labels`) against the 58 gold labels directly, masking on both sides (a study where
the rule found no match has `NaN` in the *prediction*, not just possibly in the target — plain
`per_label_auc` only masks target NaN, so this needed its own masking):

| label | AUC vs gold | matched | positives |
|---|---|---|---|
| ACL | 0.500 | 7 | 6 |
| Medial Meniscus | 0.500 | 9 | 5 |
| Effusion | 0.644 | 36 | 20 |
| Baker's | 0.857 | 11 | 4 |

The two labels with the fewest matches (ACL, Medial Meniscus) land at exactly 0.5 — no ranking
power — but at n=7 and n=9 matched studies (6 and 5 positives respectively) that is close to the
noise floor, not necessarily a broken rule; a single flipped pair moves AUC by a lot at this size.
Macro over the 4 covered labels is 0.625, below the ~0.674 backed out of the LB for the trained
image model on these same 4 labels — but that 0.674 is a single pooled number over ~390 public-LB
gold studies, while this 0.625 is an unweighted average of four numbers computed on wildly different
and much smaller match counts (7 to 36). Treat this as a first noisy read, not yet a usable
attenuation ratio — Step 2 will have a much larger, less sparse comparison (every gold study gets an
LLM score for every label, no "rule didn't match" gaps), and that number is the one worth trusting.

**Rubric fetched.** `kaggle competitions topic-messages` (not just the web UI, which is a JS SPA
WebFetch can't read) pulled thread 733343's host post directly — saved verbatim at
`notebooks/phase3-labels/rubric_733343.md`, including a mapping table to `LABEL_COLUMNS`. Confirms
the negative criteria the plan paraphrased from memory were right: mild ACL signal change without
discontinuity → negative, low-grade MCL sprain → negative, meniscal intrasubstance degeneration not
reaching the surface → negative, OA needs >50% cartilage-thickness loss over ~1cm, meniscus needs
surface contact on ≥2 images, and any "on the fence" finding → negative across the board. This is
public forum content (the host's own rubric post, no patient data), so it's fine to commit.
Blocking prerequisite for Step 1 is resolved.

### Phase 3 Step 1 — two GPU sessions lost to vLLM, and the prompt bug underneath it

**Session 1: P100.** Kaggle assigned a Tesla P100 (sm_60) despite the notebook requesting GPU. The
container's torch supports sm_70+, so this was unusable. The explicit capability assertion in cell 1
caught it in ~8s rather than letting it fail deep inside a CUDA kernel launch. Confirms the
NOTES.md 2026-08-09 finding: `kaggle kernels push` does not reliably honor an accelerator choice —
**the T4×2 has to be selected in the notebook editor's own settings, then triggered via "Save
Version → Save & Run All (Commit)"**. That ritual worked; the second session got T4×2 correctly.

**Session 2: vLLM will not install on the current Kaggle image.** `import vllm` fails with
`ImportError: libcudart.so.13`. This is an open upstream bug (vllm-project/vllm#43435, #44335):
`--torch-backend=cu129` selects the CUDA build of *PyTorch* only — vLLM's own precompiled wheel is
fetched as a cu13 build regardless, and Kaggle's image provides CUDA 12. Retried once via
`uv pip install vllm --torch-backend=auto` (vLLM's own documented recommendation) with the identical
result. The remaining workaround is `VLLM_PRECOMPILED_WHEEL_VARIANT` against a nightly index —
unpinned and unverifiable from here. **Do not spend a third session on vLLM.**

**The real cause was a prompt bug, not a packaging bug.** vLLM was in the design purely for prefix
caching, and prefix caching was only needed because `build_label_prompt` prepended the *entire*
`RUBRIC_TEXT` — all 12 label definitions, ~1,400 tokens — to each of a report's 12 questions. So
one report cost ~12 × 1,800 tokens of prefill, and an inference engine was being recruited to hide
that. Measured on the real corpus:

| prompt design | chars/prompt (median report) | ≈tokens/prompt | ≈tokens/report (×12) |
|---|---|---|---|
| whole rubric per question | 5,433 | ~1,495 | ~17,900 |
| **per-label criterion only** | 1,785 | ~510 | ~6,100 |

Sending only the criterion for the label being asked about is ~3× cheaper, removes the need for
prefix caching, and therefore removes vLLM entirely in favour of the `transformers` already in the
Kaggle image — no install step, so no install failure. It is also the better prompt: the model gets
the one definition it must apply instead of eleven competing ones as distractors. The rubric's three
back-referencing criteria ("the same criteria applied to the lateral meniscus") are resolved inline
so each stands alone; substitution is limited to the structure/compartment name, thresholds
unchanged. `RUBRIC_TEXT` is retained verbatim as the source record.

**`score_from_top_logprobs`/`score_report_with_llm` did not change at all.** They were written
against a `generate_fn` seam and tested with a fake, so swapping the entire inference engine
underneath them touched neither the functions nor their tests — the clearest payoff yet from that
Phase 2 habit of keeping the pure logic separate from the runtime.

**Verified locally before spending any further quota** (transformers 5.8.0, real Qwen3 tokenizers):

- `enable_thinking=False` is accepted by both Qwen3 tokenizers — and it is **not optional for
  Qwen3-8B**. Its default template leaves the assistant turn open (`...assistant\n`), so the first
  generated token would open a `<think>` block and *every* answer would score NaN. With the flag the
  template pre-fills `<think>\n\n</think>\n\n`, putting the real answer at the first position. The
  4B-Instruct-2507 variant ignores the flag (already non-thinking). Gemma's template rejects the
  kwarg entirely, so it is passed inside a `try/except TypeError` rather than assumed either way.
- "Yes"/"No" encode to exactly one token in every casing tested for both Qwen3 tokenizers. This is a
  precondition, not a detail: the score is read at a single answer position, so a form spanning two
  tokens has no logit there and would silently produce no score.
- `torch_dtype=` still works in transformers 5.8.0 (deprecated in favour of `dtype`, not removed).
  Kept over `dtype` because Kaggle's image may still be on 4.x, where `dtype` does not exist —
  `torch_dtype` is the form valid in both.

**Process changes, both aimed at the same failure mode.** A CPU-only probe kernel
(`notebooks/phase3-probe/`) now runs first: Kaggle CPU notebooks cost no GPU quota, so environment
facts (installed versions, free disk, measured prompt token counts, whether each candidate's
tokenizer downloads at all) get established for free instead of being guessed at inside a GPU
session. And the bake-off writes `bakeoff_results.csv` **after each candidate** rather than once at
the end, so a failure on candidate 3 can no longer discard candidates 1 and 2.

Candidate list trimmed to Qwen3-4B-Instruct-2507, Qwen3-8B, Gemma-3-12B-IT. **Qwen3-14B-AWQ was
dropped**: AWQ needs `autoawq` kernels, i.e. exactly the kind of extra install that just cost two
sessions. It can come back as its own deliberate session if the two Qwen candidates suggest scale
matters.

**CPU probe results (`notebooks/phase3-probe/`, 2026-08-17).** Cost nothing, and returned three
things worth having had before either GPU session:

| fact | value |
|---|---|
| `transformers` on the Kaggle image | **5.0.0** (had been assumed 4.x) |
| `accelerate` | 1.13.0 present — `device_map='auto'` across both T4s is available |
| `/kaggle/working` free | 20.9 GB; `~/.cache` (where HF downloads land) 1.1 TB — weights are not a constraint |
| both Qwen3 tokenizers | load fine, `enable_thinking` accepted, **all six** Yes/No forms single-token |
| `google/gemma-3-12b-it` | **401 gated** — no `HF_TOKEN` secret on this account |
| prompt size, median report | **389 tokens** → 4,668/report → **20.6M tokens** for the full corpus |
| prompt size, longest report | 1,287 tokens → 15,444/report |

Two changes fall straight out of that. **Gemma is now skipped unless `HF_TOKEN` is actually set** —
the probe proved it cannot download, so including it would only burn session time on a guaranteed
401. (To enable it later: accept the license on the model page, then add the token as a Kaggle
Secret named `HF_TOKEN`.) And **prompts are sorted by token length before batching**: at 389 median
against 1,287 max, batching in corpus order would pad most batches to roughly 3x the tokens they
need. Results are scattered back by original index, so ordering never reaches the output.

**Adapter validated locally before the push, not on Kaggle.** A real Qwen3-0.6B download stalled on
HF rate limiting, so the tensor path was validated against a 2 MB random model instead — legitimate
here because the tokenizer/template risks were already settled separately against the *real* Qwen3
tokenizers, leaving only model-independent tensor behaviour to check. Asserted: batched scores equal
one-at-a-time scores to **0.0 exactly** across prompts spanning 162–370 tokens (a left-padding or
logit-indexing bug is exactly what breaks that), a 12-label sweep produces 12 distinct in-range
scores, and every prompt yields a usable Yes/No. Also fixed while there: tokenize with
`add_special_tokens=False`, since `apply_chat_template` already emits the model's special tokens and
letting the tokenizer add BOS again would silently double it on Llama-family models.

**The reusable lesson from the whole detour:** the failure was not the CUDA packaging bug, it was
adopting a heavyweight dependency without asking what in the design was creating the need for it.
The need was self-inflicted, the prompt fix took minutes, and the dependency disappeared entirely.
Ask what a dependency is compensating for before installing it.

### Bake-off run 1 (2026-08-17) — the weak-supervision thesis holds

Qwen3-4B-Instruct-2507 over the 58 gold reports, 696 prompts, **100% answer rate** (696/696 got a
usable Yes/No — the `enable_thinking` handling is correct):

| label | gold AUC | positives | | label | gold AUC | positives |
|---|---|---|---|---|---|---|
| ACL | 0.9571 | 24/58 | | PF OA | 0.8282 | 21/58 |
| MCL | 0.8866 | 9/58 | | Effusion | 0.8174 | 35/58 |
| Medial Meniscus | 0.9147 | 26/58 | | Synovitis | 0.7372 | 27/58 |
| Lateral Meniscus | 0.8671 | 23/58 | | Baker's | 0.9529 | 12/58 |
| Medial OA | 0.9271 | 15/58 | | Contusion | 0.8124 | 19/58 |
| Lateral OA | 0.8124 | 11/58 | | Fracture | 0.8292 | 18/58 |

**Macro 0.8618 against a lexical anchor of 0.6252 — margin +0.2366.** Every label clears 0.73, and
critically the **eight labels that had no lexical rule at all** — the ones contributing a flat 0.500
each to the 0.558 LB score — now carry real signal (MCL 0.887, Lateral Meniscus 0.867, Medial OA
0.927, Lateral OA 0.812, PF OA 0.828, Synovitis 0.737, Contusion 0.812, Fracture 0.829). Synovitis
is weakest, which matches the plan's prediction that the rubric's thresholds are least reflected in
report prose there. Caveat that does not go away: n=58, and MCL rests on 9 positives, so treat these
as directional, not precise.

**Qwen3-8B OOMed, and it was my bug, not the model's size.** `model(**enc).logits` materialises
logits at *every* position: batch 8 × seq 1849 × 151,936 vocab in fp16 is **4.49 GB**, which is
exactly the "Tried to allocate 4.03 GiB" in the traceback. Only the last position is ever read.
`logits_to_keep=1` reduces that tensor to **2.4 MB** — verified locally that the answer-position
logits are bit-identical either way (max delta 6e-08). The 8B was never too big for two T4s; it was
being asked to build a 4.5 GB tensor and discard 99.95% of it.

**Two further corrections to the run-1 harness:**

- **The "NO CANDIDATE FITS" verdict was wrong, not just unlucky.** It fired because the 4B's
  projected full-corpus time (4.79h) exceeded a 4h *session* budget — but this project already
  shards long jobs by index range (Phase 2 prepped 4,407 studies across 4 kernels). Runtime beyond
  one session is a scheduling detail, not a disqualification. Eligibility is now "it ran and
  answered ≥90% of questions"; shard count is reported instead of used as a veto.
- **The projection was also inflated.** It extrapolated per-report time from the gold subset, but
  gold reports are longer than corpus average — measured locally with the real tokenizer, 6,619 vs
  6,003 tokens per report across all 12 prompts (n=400 sample). Scaling by tokens rather than report
  count gives 4.34h, not 4.79h.

Fixed-size batching also replaced with **token-budget batching** (16,384 padded tokens/batch, cap
64): prompts span 157–1,845 tokens, so a fixed count of 8 idled the GPU on the short end and OOMed
on the long end. An OOM now halves the batch and retries rather than killing the candidate outright.
Validated locally over 200 random length distributions (no prompt lost or duplicated, budget and cap
respected, an oversized single prompt still forms its own batch rather than looping).

### Bake-off run 2 (2026-08-25) — harness fix confirmed; 4B wins on the pre-registered tie-break

Both candidates re-scored under the corrected harness, T4x2, 696 prompts each, **100% answer rate
for both**, zero OOM retries (the `logits_to_keep=1` fix resolved the 8B's run-1 death — it was
never too big, it was building a 4.49 GB tensor and discarding it):

| candidate | gold macro AUC | answer rate | elapsed | projected corpus | shards |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 0.8613 | 100% | 204s | 3.91h | 1 |
| Qwen3-8B | **0.8759** | 100% | 406s | 7.78h | 2 |

Per-label (4B / 8B): ACL 0.958/0.973, MCL 0.887/**0.964**, Medial Meniscus **0.915**/0.898,
Lateral Meniscus 0.866/0.871, Medial OA 0.926/0.938, Lateral OA 0.811/**0.872**, PF OA
**0.832**/0.826, Effusion **0.817**/0.801, Synovitis 0.737/**0.757**, Baker's 0.949/0.954,
Contusion 0.810/0.808, Fracture 0.826/**0.849**.

- **The gap (+0.0146) sits inside the pre-registered NEAR_TIE=0.02 band, so the harness took the
  faster model (4B)** — that rule was set before seeing any numbers precisely so it couldn't be
  re-litigated after. At n=58 a 0.015 macro gap is not distinguishable from noise.
- **But the 8B's improvement is not uniform noise either**: 8 of 12 labels improved, and the four
  biggest gains (MCL +0.077 on 9 positives, Lateral OA +0.061 on 11, Fracture +0.022 on 18,
  ACL +0.015 on 24) concentrate exactly where positives are scarcest — consistent with more
  capacity helping most on the hardest-to-call criteria, though with n this small each individual
  gain is also the easiest to flip by chance.
- **The two models genuinely disagree per-report while agreeing on ranking**: score matrices
  differ by mean abs delta 0.069, p95 0.84, max 1.00 — yet macro AUCs land 0.015 apart. The soft
  scores are far less stable across model scale than their downstream AUCs suggest.
- **Reproducibility across harness changes is good at the macro level**: run-1 4B (fixed batch 8)
  vs run-2 4B (token-budget batching) moved macro 0.8618→0.8613 (delta 0.0005); worst single label
  PF OA moved 0.8282→0.8320 (delta 0.0038). fp16 batching numerics shift individual scores but not
  conclusions.
- Raw artifacts kept for Phase 4 calibration analysis without another GPU session:
  `results/bakeoff.csv` (full metrics row per candidate) and `results/bakeoff_scores/`
  ({scores,weights}_{candidate}.npy, 58x12 soft-label matrices from each model).
- Cost framing for the Step 1b decision the run leaves open: 4B labels the corpus in one <=4h
  session; 8B needs two (~7.8h total GPU). This is a one-time spend either way — the labels then
  feed every later experiment for free.

### Step 1b — full corpus labeled and verified (2026-08-26)

Qwen3-4B-Instruct-2507 scored all 4,407 studies x 12 labels (52,884 prompts) across two parallel
T4x2 shard kernels, ~115 min each, **100% answer rate (52,884/52,884), zero OOM retries**. The
shard design copied Phase 2's exactly: contiguous UID ranges over the sorted corpus
([0,2203)/[2203,4407)), incremental CSV saves every 25 batches, a completion sentinel per shard
that the consumer kernel requires before stitching.

- **The consumer's five gates all passed**: sentinels present; exact partition (no duplicated,
  missing, or unexpected UIDs across shards); no NaN anywhere, scores and weights inside [0,1];
  gold cross-check vs the bake-off output mean|delta|=0.00059 (drift gate: 0.02); recomputed gold
  macro AUC from the stitched file 0.8616 vs recorded 0.8613.
- **The two-gate cross-check design earned its keep immediately**: max|delta| hit 0.163 on a
  handful of prompts — fp16 batching numerics flipping knife-edge (score~0.5) cases, precisely as
  predicted when the gate was split into a strict mean test and a report-only max. A single hard
  max threshold would have either false-alarmed here or been too loose to catch real prompt drift.
- **Corpus prevalence corroborates the "gold is enriched 2x" claim independently**: corpus mean
  score is 0.110 vs the gold set's 0.217 — almost exactly half, matching the public notebook's
  prevalence-shift warning (NOTES.md 2026-08-08). Any calibration fitted on the gold set must not
  be assumed to transfer to corpus-wide thresholds unchanged.
- Per-label corpus means: Medial Meniscus 0.318 and Effusion 0.262 dominate; Fracture 0.014,
  MCL 0.021, Lateral OA 0.025 are rare. Class imbalance this extreme matters for Phase 4 loss
  weighting and for how much a rare-label AUC can move at all.
- Artifacts: `results/pseudo_labels_qwen3_4b.csv` (the Phase 4 label source: StudyInstanceUID +
  12 score cols + 12 weight cols) and `results/verification_manifest.json`. Shard outputs remain
  on Kaggle (scores_shard{0,1}.csv + manifests) attached to the consumer kernel via kernel_sources.
- Process note: CLI `kaggle kernels push` always triggers an auto-run, which has landed on P100
  essentially every time — GPU kernels now get built and validated locally, then handed to the
  human for the T4x2 editor ritual; only CPU-only kernels get pushed directly.

### Phase 4 run 1 — pseudo-labels train all 12 labels (2026-08-26, ingested 2026-09-02)

`efficientnet_b0` + slice mean-pool, 1 sagittal series x 16 slices, Adam 1e-4, batch 8, **1 epoch
per fold**, 5 folds over frozen `primary_v2`, trained on Qwen3-4B soft pseudo-labels with all 58
gold studies excluded from **both** sides of every split. T4x2, ~35 min total training.

- **Pooled OOF macro AUC vs the pseudo-labels (4,349 studies): 0.7779.** Per fold 0.7642-0.8000.
- **Gold transfer (58 held-out studies, their true rubric labels): 0.7252** for the mean of the 5
  fold models; per-fold 0.6908-0.7264. The ensemble is not better than the best single fold
  (fold 1, 0.7264) — at n=58 that 0.0012 is noise, and picking the best fold on 58 studies would
  be selection bias, so the ensemble ships.
- **The like-for-like number: 0.777 vs Phase 1's ~0.674.** On the four labels Phase 1 also trained
  (ACL 0.805, Medial Meniscus 0.630, Effusion 0.939, Baker's 0.736) gold transfer averages 0.777
  against the ~0.674 backed out of Phase 1's LB score — **+0.10**, and the other eight labels now
  carry real signal instead of contributing a flat 0.500 each.
- **`paired_delta` is 0.0 in all five rows and must stay that way.** Phase 1 is primary_v1 / 2,151
  studies / 4 labels; Phase 4 is primary_v2 / 4,349 / 12. There is no study-paired comparison
  between them — the gold-transfer number above is the only valid one. The five appended rows are
  folds 0-4 in order (the run logged `fold_set=primary_v2` without a fold suffix).
- No `hall_of_fame.csv` row: run 1 beats nothing on a paired delta because there is nothing to pair
  it against. It is the **reference run** every Phase 5 delta is measured from.

**MCL is the most informative number in the run, and it points straight at the input shape.**
Gold transfer per label against the finding's plane and its corpus prevalence:

| label | gold AUC | prevalence | plane |
|---|---|---|---|
| MCL | **0.5034** (random) | 0.020 | coronal |
| Lateral OA | 0.6132 | 0.025 | coronal |
| Medial Meniscus | 0.6298 | 0.319 | coronal + sagittal |
| PF OA | 0.6821 | 0.058 | axial / sagittal |
| Fracture | 0.7056 | 0.014 | any |
| Lateral Meniscus | 0.7081 | 0.104 | coronal + sagittal |
| Baker's | 0.7355 | 0.147 | sagittal / axial |
| Synovitis | 0.7802 | 0.113 | sagittal |
| Medial OA | 0.7922 | 0.074 | coronal |
| ACL | 0.8051 | 0.082 | sagittal |
| Contusion | 0.8084 | 0.103 | any |
| Effusion | **0.9391** | 0.262 | sagittal |

The Qwen3-4B *label* for MCL scored 0.887 gold AUC in the bake-off, so the label is fine — the model
learned nothing from it. Run 1 feeds one **sagittal** series and MCL is a coronal structure; the
model cannot see it. The two worst labels are both coronal, the best is the one most visible on
sagittal. Next in `_SERIES_PRIORITY` after sagittal-fluid-sensitive is coronal-fluid-sensitive, so
`max_series=2` is the one-variable change with a mechanism behind it rather than a guess.

Caveats that don't go away: n=58 for every gold number, and MCL rests on 9 positives. Directional.

**Process failure worth naming:** the kernel finished 2026-08-26 and the result sat on Kaggle for a
week — unread, unlogged, and unsubmitted — while the leaderboard still showed the Phase 1 score. A
run that isn't pulled back into `results/` didn't happen, in exactly the sense the plan's experiment
discipline means it. Pull the output in the same sitting the kernel completes.

### Phase 4 submission path built and dry-run locally (2026-09-02)

The Phase 4 model trained on Phase 2 artifacts (256px, letterboxed, per-series normalized,
priority-ordered, mirrored at load time); `notebooks/phase1-submit/` reads raw DICOMs at 224px with
none of that. Reusing it would have fed the model something it never saw, so `phase4-submit` sends
each test study through **the same two functions the training data went through** — `prep_study`
with Phase 2's parameters (`max_series=4, k_slices=24, size=256`), then `PreppedStudyDataset` with
run 1's (`max_series=1, n_slices=16`). No third copy of the pipeline to drift.

- **Prepping fewer series would not have been equivalent.** `select_series` exhausts
  `_SERIES_PRIORITY[:max_series]` and then falls back to arbitrary dataframe order, so for a study
  with no sagittal fluid-sensitive series a `max_series=1` prep picks a different series than a
  `max_series=4` prep does. Prep parameters have to match Phase 2's, not the read config's.
- **Local dry-run against the real fold checkpoints, 3 sample test studies:** prep **2.66 s/study**
  (consistent with Phase 2's 1.9-2.1 s on Kaggle hardware), forward **1.40 s/study for all 5 models
  on CPU** — the forward collapses on a GPU, so the hidden-test budget is prep-bound at roughly
  0.75 h per 1,000 studies. Predictions for the one complete study look sane (all low, Medial
  Meniscus highest at 0.16, matching it being the most prevalent label at 0.319).
- **`data/sample/` only ships DICOMs for 1 of its 3 test studies**, so 2 hit the 0.5 fallback. Not a
  bug — it accidentally became the first end-to-end proof that the fallback works: a study whose
  pixels cannot be read yields a 0.5 row instead of killing the submission.
- Artifacts are written to a temp dir and deleted as they are consumed, so peak disk is one study.
  The hidden test set's size is not knowable in advance and `/kaggle/working` has ~21 GB.

**Pre-registered reading of the resulting LB score** (recorded before submitting so it cannot be
rationalized afterwards). Phase 1's backed-out 0.674-on-4-labels reproduces its 0.558 exactly under
`(4 x 0.674 + 8 x 0.5) / 12`, which makes gold-transfer the working predictor of LB — imperfect,
since that 0.674 was itself derived from the LB:

- **>= 0.70** — inference path verified, proceed to Phase 5.
- **0.60-0.70** — prevalence shift or a mild preprocessing mismatch; investigate before building on it.
- **< 0.60** — treat as a **broken inference path, not a weak model**; debug preprocessing parity first.

**`make_folds` reproduces `results/folds_primary_v2.csv` exactly** (verified locally, all 4,407
rows), so the frozen file and the function cannot have drifted apart. Its digest —
`cace8c45733ee7920a442fa4ae3f1db4a0d76401504e4729b49562d5eab52047` over sorted `uid,fold` lines — is
now asserted in the Phase 5 notebook. Run 1 asserted fold *sizes* (`[881,881,881,882,882]`), which
hold for any seed and would not have caught a reshuffle.

### The P100 problem was a metadata bug, not a Kaggle policy (2026-09-02)

Every previous `kaggle kernels push` landed on a Tesla P100 (sm_60, unusable — the container's torch
is sm_70+), which is why the project adopted a rule of never CLI-pushing a GPU kernel and instead
hand-triggering every run from the editor. **The cause was in our own metadata.** Kaggle's kernel
metadata carries a `machine_shape` field that names the accelerator, the CLI passes it straight
through (`request.machine_shape = acc if acc else meta_data['machine_shape']`), and every one of our
`kernel-metadata.json` files had it as the empty string — which means "server picks". The editor
selection was never being sent at all.

Pulling the metadata of the kernel that *did* get T4 x2 gives the value:

```
"machine_shape": "NvidiaTeslaT4"     # this is the T4 x2 option, not a single T4
```

Set on `phase4-submit` and pushed via CLI, the run reported `device: cuda` and completed — a P100
would have failed the probe and fallen back to CPU. CLI 2.2.4 also exposes it as
`kaggle kernels push --accelerator NvidiaTeslaT4`, which overrides the metadata.

Two things that came free with it:
- Kaggle **auto-pinned the new kernel's `docker_image` to the same sha as `phase4-train`**
  (`gcr.io/kaggle-private-byod/python@sha256:37c64f7d...`), so the checkpoints run in the environment
  they were trained in without asking. `docker_image` and `docker_image_pinning_type`
  (`original`/`latest`) are both honoured from metadata if it ever needs forcing.
- Kaggle accepts **at least 5 notebook inputs** (`phase4-train` has 5). The Phase 5 kernel asks for
  6; still unverified.

**What stays true:** `kaggle kernels push` always starts a run — the CLI's own help says "Push new
code to a kernel *and run the kernel*", and there is no draft/no-run flag. Importing through the web
editor is the only way to get a notebook onto Kaggle without executing it. So the rule changes from
"never CLI-push a GPU kernel" to "a CLI push always runs, so only push what you mean to run — and
set `machine_shape` when you do."

**Phase 4 submission run (version 1, T4, 3 visible studies, 0 fallbacks):** prep **2.66 s/study**
(p95 4.05), forward **0.428 s/study for all five fold models** on GPU — 3.09 s/study end to end, so
roughly **67 min per 1,000 hidden studies**, prep-bound, well inside the 9 h cap. Mean predictions
track the training positive rates across all 12 labels and preserve their ordering (Medial Meniscus
0.268 vs 0.319, Effusion 0.144 vs 0.262, Fracture 0.006 vs 0.014). At n=3 that is a smoke signal,
not a measurement, but a broken preprocessing path would not produce that shape.

### Phase 4 scored: LB 0.763, and the leaderboard says the model is undertrained (2026-09-02)

**0.558 -> 0.763 (+0.205)**, rank 2556 -> **2187 of 2936**. Clears the pre-registered >= 0.70 band,
so the inference path is verified and preprocessing parity holds on the hidden set. It also beats
what gold transfer predicted (0.7252), which is the pleasant direction to be wrong in — the 58-study
gold set is enriched roughly 2x over corpus prevalence, so it was always a pessimistic proxy.

**The gate is passed and the strategy is confirmed. The standings are the sobering part:**

| percentile | score | | |
|---|---|---|---|
| 1st | 0.952 | teams >= 0.94 | 95 |
| top 1% | 0.945 | teams >= 0.90 | **1,386** |
| top 25% | 0.934 | teams >= 0.80 | 2,076 |
| **median** | **0.899** | teams in [0.760, 0.766] | 16 |
| ours | **0.763** | teams <= 0.558 (Phase 1) | 353 |

We are in the **bottom quartile**, 0.184 below 10th place. Note how flat the top is: 5th percentile
0.937 to 25th percentile 0.934 is a 0.003 spread across ~700 teams, which is the signature of a
widely-copied public notebook rather than 700 independent solutions.

**Checked the obvious explanation and it is not the one: `test.csv` carries only
`StudyInstanceUID` — there is no `Report` column at test time.** Reports exist for training studies
only. So the teams at 0.93 are predicting from images alone, exactly the frame this project is
built on. The most-voted public notebooks are titled after their backbones — "Bend the Knee to
DinoV3 (ensembled)", "DINOsaur V4", "DINO-RadImageNet Rank Ensemble" — alongside
"read the report, then the knee", which is the pseudo-label thesis under another name.

**So the weak-supervision thesis is vindicated and the model is the problem.** Run 1 is
`efficientnet_b0` (ImageNet), **1 epoch**, 1 series x 16 slices, no augmentation, no AMP — 435
optimizer steps in total. That is a smoke test that happens to score, not a trained model. The field
is running domain-pretrained or self-supervised backbones (RadImageNet, DINOv3) with ensembling.

**Consequence for the Phase 5 payoff order, recorded now rather than after another experiment:**
the plan ranks series routing above backbone choice, and that ordering was set before there was any
evidence of where we stand. Experiment B (2 series) is still worth its 70 minutes — it is cheap and
MCL 0.503 is a real mechanistic failure with a named cause — but **no single one-variable experiment
in the current list closes 0.184.** Epoch count and backbone are the two variables that have never
been moved at all, and both are cheaper to test than they look. Re-rank before spending the
experiment budget top-down on the old order.

### Phase 5 Step 1 — public-notebook recon, and two of its claims tested against our data (2026-09-05)

Read the five most-voted public notebooks (intel only; we write our own code). The 0.934 cluster is **`pilkwang/rsna-knee-baseline-v1`, 512 votes** — nearly 3x the next
notebook and comfortably the thing ~700 teams forked. It is an *inference* notebook that loads
pre-fitted weights, but it carries the full recipe in its source.

**Its recipe, against ours:**

| | ours (run 1) | pilkwang |
|---|---|---|
| input | 1 series x 16 slices | **6 slots** (plane x weighting x fat-sat) + presence mask |
| scale | 256px letterbox, fixed *pixels* | crop to **130 mm**, resize 224/336 -> fixed **mm/pixel** |
| channels | slice replicated to 3 | **3 adjacent slices** as the 3 channels |
| backbone | `efficientnet_b0`, ImageNet, all trainable | **DINOv2-small**, last **6** blocks only |
| lr | Adam 1e-4, flat | AdamW **8e-6 backbone / 1e-3 head**, OneCycle, wd 0.02 |
| epochs | **1** | **10** |
| aug | none | rot 8deg, scale 0.08, shift 0.05, intensity 0.10, **no flips** |
| head | mean-pool -> linear | per-diagnosis **attention over slots**, masked softmax |
| ensembling | probability mean of 5 folds | **rank mean** (AUC reads only order) |
| label weights | `weight_*` computed, unused | confidence as **sample weight**; gold at elevated weight |
| split | shuffled by uid | grouped by **report-text hash** |
| AMP | none | autocast + GradScaler |

The six slots are `SAG_FLUID_FS`, `COR_FLUID_FS`, `AX_FLUID_FS`, `SAG_FLUID_NOFS`, `COR_T1`,
`SAG_T1`. It also reports that `train_series.csv`'s `Fluid_Sensitive` and `Fat_Suppression` columns
**agree on every row**, so as delivered they carry one axis rather than two; it recovers both from
TR/TE and `ScanningSequence` in the headers.

Two arguments worth keeping even where we don't copy the implementation:

- **Nyquist bounds the resize, and no capacity downstream recovers it.** A meniscal tear is 1-3 mm,
  so the pitch must be <= d/2. At a 130 mm crop, 224px gives 0.580 mm/px and 336px gives 0.387.
  Our 256px *letterbox* normalises no physical scale at all — studies differ in mm/px by a factor
  of several, and the model sees them as if they didn't.
- **Rank-mean, not probability-mean, for ensembles.** AUC is invariant under any increasing map, so
  averaging probabilities lets the most confident member dominate while averaging ranks combines
  exactly the information the metric reads. Free change to `phase4-submit`.

**Two of its claims tested directly against our own data — both land, one of them big:**

**1. The laterality heuristic we rejected was rejected for the wrong reason.** NOTES 2026-08-08
killed the `ImagePositionPatient`-sign rule at 40% disagreement with the real `Laterality` tag, and
the plan has carried "laterality unresolved for 48.9% of studies" as a known ceiling ever since.
pilkwang uses the **image centre**, not `p` itself — `p` is a corner half a field of view away,
which flips sign on a knee scanned near the midline:

```
c = p + r * col_spacing * (Columns/2) + d * row_spacing * (Rows/2)
side = R if c_x < 0 else L
```

Re-run on the same 20 local sample studies (11 carry a real tag):

| form | agreement |
|---|---|
| corner `p_x` (the rejected one) | 9/11 = **81.8%** |
| **centre `c_x`** | **11/11 = 100%** |

**Caveat on the comparison:** only 11 of the 20 local sample studies carry a tag, so the corner
form measures 18.2% disagreement here, not the 40% the 2026-08-08 entry recorded over 20 studies.
That entry's denominator and sample are not reproducible from what is in `results/` (the census CSVs
carry no geometry), so treat 81.8% as "the corner form on the 11 studies we can check today", not as
a restatement of the original number. What is not in doubt is the *paired* comparison: on identical
series, centre beats corner, and the two corner failures are exactly the predicted case — `p_x` of -17.1 and -3.2 on knees whose
centres sit at +69.5 and +78.9. Median `|c_x|` is 83 mm, well outside pilkwang's 20 mm unresolved
band, so the sign is not marginal here. n=11 is small and this must be re-verified at corpus scale
(every study has the geometry, ~51% carry a tag to check against) — but the original 40% figure
measured the corner form, and the centre form is a different rule. **This plausibly unlocks the half
of the corpus that currently enters the model unmirrored, which is the named cap on medial/lateral
discrimination and on MCL.**

**2. Shared reports leak across our folds, but only mildly.** Byte-identical reports appear in
**49 groups covering 183 studies (4.2%)** — one group has 37 members. Against
`folds_primary_v2.csv`, **45 groups straddle more than one fold, affecting 175 studies (3.97%)**.
So our OOF is optimistic by whatever those 4% contribute — real, worth fixing when the folds are
next re-frozen, not worth invalidating run 1 over. Re-freezing folds now would break the paired
comparison Phase 5 is built on, so this is logged and deferred, not acted on.

**What this does to the Step 3 sweep order.** The recon was the gate on the grid, and it says the
grid was aimed too low: epochs 1 -> 10, a domain-appropriate self-supervised backbone, physical-scale
sampling and multi-slot input are not one-variable refinements of run 1's configuration, they are
most of the distance to 0.93. `efficientnet_b0` at 1 series x 16 slices on a fixed-pixel letterbox
cannot reach it at any schedule.

### The geometry laterality route holds at corpus scale — coverage 51.1% -> 98.5% (2026-09-05)

`notebooks/phase5-laterality/`, CPU-only, 4,407 studies walked in **2.1 min**. Scored against the
2,182 studies where a real tag and the geometry route both resolve:

| | |
|---|---|
| **agreement** | **2162/2182 = 99.08%** |
| tag route resolves | 2,252 / 4,407 = 51.1% |
| geometry route resolves | 4,271 / 4,407 = 96.9% |
| **either resolves** | **4,341 / 4,407 = 98.5%** |
| newly resolved by geometry | **2,089 (47.4% of the corpus)** |
| still unresolved | 66 (all `geometry_midline`) |

Both pre-registered gate conditions cleared (>= 99% agreement, >= 30% newly resolved). **Verdict:
adopt.**

**The 20 disagreements are not the midline band.** Their |centre_x| runs 60-113 mm (median 87), so
the band is not too narrow — these are genuine disagreements, 0.92% of scored studies, all against
the `Laterality` tag (0/74 against `SeriesDescription`), in both directions (13 L->R, 7 R->L) and
with no concentration by series count. Whether the tag or the geometry is wrong on those 20 is not
determinable from here, and **it does not matter for the wiring**: geometry is strictly a fallback
*after* every tag route, so those 20 studies keep their tag value and nothing changes for them. The
99.08% is evidence that the rule is sound where it actually gets used — the 2,089 studies that had
nothing at all.

**The correction costs no re-prep.** Phase 2 deliberately stored pixels **unmirrored** with the
resolved side in `meta`, and `PreppedStudyDataset` mirrors at load time — a decision recorded in
`prep.py` as keeping "artifacts neutral to a laterality decision Phase 3 may still improve." That
decision pays off exactly here: the 4,407 artifacts stay valid and only the side each one is read
with changes.

Wiring, both halves needed:

- `resolve_study_laterality` falls back to `side_from_geometry` when no tag resolves. **This is the
  half that matters for the leaderboard**: `phase4-submit` preps each hidden-test study fresh
  through this function, so without it training would use corrected sides and inference would not —
  a train/test mismatch on the four medial/lateral labels, which is worse than leaving both wrong.
- `PreppedStudyDataset(sides=...)` overrides the stored side per study, so the existing artifacts
  can be read with corrected sides. `results/laterality_sides_v2.csv` is that map, and it reproduces
  what the patched resolver now returns (tag wins; geometry only where no tag resolved):
  `Laterality` 2178, `geometry` 2089, `SeriesDescription` 74, `geometry_midline` 66.
  Side balance R 2298 / L 2043.

One diagnostic was nearly lost in the wiring and put back: falling back unconditionally would have
replaced the `conflict` and `unknown` routes with `geometry_unavailable`, collapsing "the tags
contradicted each other" and "there were no tags" into one label the Phase 2 census counts
separately. Geometry now overrides only when it is actually readable.

**What this does not yet do.** Nothing has been retrained on it. The screen running now
(`phase5-screen`) deliberately uses the stored sides, so it stays a clean one-variable read of the
training schedule; the laterality fix is its own screen row afterwards. The prediction on record,
before that run: MCL (0.5034, random) and Lateral OA (0.6132) move most, because they are coronal
findings whose plane is exactly what an unmirrored study scrambles, and Effusion (0.9391, sagittal)
barely moves.

### Two process notes from launching the Phase 5 screen (2026-09-05)

**The submission half of the laterality fix is verified locally, not just the training half.**
`census_study_laterality` is what `prep_study` calls, and it is the function the hidden-test path
runs per study. On the 20 local sample studies it now resolves **20/20 (11 `Laterality`, 9
`geometry`)** where it previously resolved 11 and left 9 unknown. So training and inference will
canonicalize the same way, which was the actual risk: correcting only the training side would have
introduced a train/test mismatch on the four medial/lateral labels, and that is worse than leaving
both uncorrected.

**A stale `rsna-knee-src` dataset version killed the first screen run 7.5 min in.** The kernel was
pushed after `train.py` grew its AMP/scheduler parameters but *before* the src dataset was
re-versioned, so the kernel mounted a snapshot without them. The cheap guard at the top —

```python
assert {'scaler', 'scheduler'} <= set(inspect.signature(train_one_epoch).parameters)
```

— turned a confusing mid-run `TypeError` two hours deep into an explicit failure at cell 2. **Rule
worth keeping: a kernel that depends on a code change must re-push the src dataset first, and
should assert the specific attribute it depends on rather than trusting the mount.** The guard now
covers `PreppedStudyDataset(sides=...)` as well.

The relaunched screen also folds in the laterality comparison rather than paying for a second
kernel: `s6` is run 1's schedule with corrected sides (isolating the mirroring against `s1`), and
`s7` is the best schedule off the epoch ladder plus corrected sides — built after the ladder runs,
since which schedule wins is not knowable in advance. It asks Kaggle for **7 notebook inputs**;
NOTES previously recorded 5 as verified and 6 as untested, and 7 mounted fine.

### Phase 5 screen A: the schedule was worth +0.061, and the model now overfits (2026-09-05)

Fold 0 only, 867 val studies, `efficientnet_b0` 1 series x 16 slices throughout. Decisions on the
six stable labels, full 12-label macro reported alongside. Paired bootstrap against run 1's own
fold-0 OOF predictions.

| config | full macro | stable-6 val | stable-6 **train** | gap | delta vs run 1 (95% CI) | min |
|---|---|---|---|---|---|---|
| run 1 (e1, flat, fp32) | 0.7805 | 0.7769 | — | — | — | 6.5 |
| s1 e1 flat **+AMP** | 0.7962 | 0.7788 | 0.8322 | +0.053 | +0.0019 [-0.0083, +0.0111] | 4.8 |
| s2 e4 OneCycle | 0.8250 | 0.8212 | 0.8995 | +0.078 | +0.0443 [+0.0344, +0.0548] | 16.0 |
| **s3 e8 OneCycle** | **0.8413** | **0.8362** | 0.9788 | +0.143 | **+0.0592 [+0.0458, +0.0726]** | 31.8 |
| s4 e8 **flat** | 0.8184 | 0.8135 | 0.9973 | +0.184 | +0.0366 [+0.0211, +0.0512] | 31.6 |
| s5 e12 OneCycle | 0.8311 | 0.8276 | 0.9975 | +0.170 | +0.0506 [+0.0357, +0.0657] | 48.2 |
| s6 e1 flat + laterality | 0.7963 | 0.7785 | 0.8321 | +0.054 | +0.0016 [-0.0085, +0.0108] | 4.4 |
| s7 e8 OneCycle + laterality | 0.8417 | 0.8365 | 0.9788 | +0.142 | +0.0595 [+0.0460, +0.0728] | 35.7 |

**Four findings, in order of how much they change what happens next.**

**1. Epoch count was the single largest lever in the project so far: +0.061 full macro on fold 0**
(0.7805 -> 0.8413) for 32 minutes of T4. Per label, e8 beats e1 on nine of twelve — Baker's +0.102,
ACL +0.080, Contusion +0.075, Medial Meniscus +0.070, Lateral Meniscus +0.061, MCL +0.056.

**2. The schedule is separable from the epoch count and worth +0.023 on its own.** s4 (8 epochs,
flat LR) reaches 0.8135 against s3's 0.8362 at the same 8 epochs. Including that control is the only
reason "more epochs" and "OneCycle" are not confounded here.

**3. The curve turns over between 8 and 12 epochs**, and the train column says why: s3 reaches
train 0.979 and s5 train 0.9975 — **the model has memorised its training set** while val went
*down* 0.836 -> 0.828. The two labels that lose ground under longer training are the two rarest and
the one already saturated: Fracture 0.866 -> 0.829 (prevalence 0.014) and Effusion 0.849 -> 0.838.
**This settles the question Step 0 could not:** at 1 epoch the model was underfit, by 8 it is
overfit, so the next lever is augmentation and regularisation, not more epochs. That was
pre-registered as the reading of this column before the run.

**4. AMP is free** — +0.0019 with a CI spanning zero, at 4.8 min against run 1's 6.5 (~1.35x). Take
it as throughput, never as accuracy. s1 also validates the harness: it lands on run 1's own fold-0
number (stable-6 0.7788 vs 0.7769), so the screening rig reproduces the baseline it pairs against.

**The laterality comparison measured nothing, and that is a design error on my side, not a null
result.** s6-s1 and s7-s3 are +0.0016 and +0.0003, and the per-label deltas are uniformly under
0.0015 — the loss curves are identical to four decimals. The cause: `_SERIES_PRIORITY[0]` is
`("Sagittal", 1)` and `mirrors_in_plane` is true only for Axial/Coronal, **so at `max_series=1` the
one series a study contributes is sagittal and is never mirrored.** Only the **257 studies (5.83%)
that have no sagittal fluid-sensitive series** get a Coronal/Axial series in slot 0 and could be
affected at all — about 50 of fold 0's 867. The screen was ~94% inert by construction, and roughly
40 minutes of T4 went on confirming that a no-op is a no-op.

**The laterality fix is therefore still untested, not disproven**, and the experiment that tests it
is `max_series>=2` — which is the old experiment B, now with a much better rationale than it had:
coronal series are where MCL and the OA compartments live, and they are only worth adding if they
are correctly mirrored, which is exactly what was fixed today (51.1% -> 98.5% side coverage).

### Phase 5 screen B: augmentation works, coronal only pays with it, laterality is null (2026-09-06)

Fold 0, schedule frozen at screen A's winner (8 epochs, OneCycle, AMP) so every row differs from
`s3` in input or augmentation only.

| config | series | lat | aug | full macro | stable-6 | train | gap | delta vs run 1 | min |
|---|---|---|---|---|---|---|---|---|---|
| s3 (screen A) | 1 | – | – | 0.8413 | 0.8362 | 0.9788 | +0.143 | +0.0592 | 31.8 |
| b3 | 1 | – | **yes** | 0.8432 | 0.8420 | 0.9314 | **+0.090** | +0.0650 | 36.3 |
| b1 | 2 | no | – | 0.8432 | 0.8344 | 0.9668 | +0.132 | +0.0575 | 66.3 |
| b2 | 2 | yes | – | 0.8417 | 0.8352 | 0.9671 | +0.132 | +0.0583 | 65.7 |
| **b4** | **2** | **yes** | **yes** | **0.8484** | **0.8442** | 0.9247 | **+0.080** | **+0.0672 [+0.053, +0.081]** | 68.6 |

**1. Augmentation did what the overfit diagnosis predicted.** b3 vs s3 narrows the train-val gap
0.143 -> 0.090 (train 0.979 -> 0.931) and val rises. The label it helps most is **Fracture
+0.021** — the rarest label (prevalence 0.014) and the one longer training was actively damaging in
screen A. That is the predicted mechanism, not a coincidence: memorisation hurts the rare labels
first, so regularisation helps them first.

**2. A second (coronal) series is worth nothing on its own.** b1 vs s3 is +0.0019 full macro, and
per label it is a wash traded around — Medial Meniscus +0.032 and Medial OA +0.020 against Baker's
**-0.050** and Lateral Meniscus -0.018. It costs **2x the compute** (66 min vs 32) for that. Taken
alone this kills the old experiment B.

**3. But coronal composes with augmentation, and that is the finding.** b4 vs b3 adds the second
series *given* augmentation and gains +0.0052 full macro, concentrated exactly where the anatomy
says it should be: **Lateral Meniscus +0.037, Medial Meniscus +0.029, MCL +0.027, Lateral OA
+0.019**. Without augmentation the extra plane just gave the model more to memorise (train 0.967);
with it, the plane's information is actually used. **The interaction is why b1 read as a failure and
b4 does not** — and it is a warning about one-variable screening, which would have discarded the
coronal series on b1 alone.

**4. The laterality fix is null.** b2 - b1 is +0.0008 stable-6 and **-0.0014 full macro**. The
medial/lateral pairs move the right way but trivially (Medial Meniscus +0.011, Lateral Meniscus
+0.006, Lateral OA +0.004) and MCL moves the wrong way (-0.027). At one fold these are noise.

**So the day's biggest structural fix bought no measurable accuracy.** The honest reading: 99.08%
agreement and 51.1% -> 98.5% coverage are real and verified, and the pixels genuinely change for
~47% of studies — but the model apparently was not relying on canonical medial/lateral orientation
much in the first place, at least not at `efficientnet_b0` capacity with mean-pooling over slices.
It is kept because it costs nothing at training time and because **the submission path already
resolves test-study sides through the same function**, so training and inference stay consistent.
It may matter more once the head can actually attend per-plane rather than mean-pool everything.

**Confirming b4 on 5 folds** (~5.75h): it wins the competition's own metric, the per-label pattern
is mechanistically coherent rather than diffuse, and Efficiency Track is on "track, don't constrain"
so the 2x inference is acceptable for now. Caveat recorded before the run: b4 - b3 is +0.0052 full
macro on **one fold**, and the two paired CIs ([+0.052,+0.078] vs [+0.053,+0.081]) overlap almost
entirely, so this is a judgement on mechanism, not a measured separation.

### Phase 5 confirmed on 5 folds: pooled OOF 0.8523, gold transfer 0.8189 (2026-09-06)

Config `phase5_confirm_efficientnet_b0_s2x16_e8_onecycle_amp_lat1_aug1` — b4 — over all five
`primary_v2` folds, gold excluded from both sides of every split. ~67 min/fold, 5.6 h total.
Per-fold macro 0.8484 / 0.8406 / 0.8439 / 0.8761 / 0.8623.

**Tier 1, the promotion decision (pooled OOF vs pseudo-labels, full 12-label macro, n=4,349):**

| | run 1 | Phase 5 | delta |
|---|---|---|---|
| pooled OOF macro | 0.7779 | **0.8523** | **+0.0744**, 95% CI **[+0.0669, +0.0820]** |

CI excludes zero: **promoted**. All twelve labels improve — ACL +0.114, Medial Meniscus +0.114,
Lateral Meniscus +0.096, Lateral OA +0.089, Contusion +0.086, Baker's +0.076, Medial OA +0.071,
Fracture +0.066, MCL +0.062, PF OA +0.049, Synovitis +0.049, Effusion +0.020.

**Tier 2, gold transfer (58 studies, real rubric labels, the LB predictor): 0.7252 -> 0.8189.**

| label | run 1 gold | Phase 5 gold | delta |
|---|---|---|---|
| Lateral OA | 0.6132 | 0.8472 | **+0.234** |
| Medial Meniscus | 0.6298 | 0.8486 | **+0.219** |
| Medial OA | 0.7922 | 0.9504 | +0.158 |
| **MCL** | **0.5034** | **0.6259** | **+0.123** |
| ACL | 0.8051 | 0.9118 | +0.107 |
| Contusion | 0.8084 | 0.9055 | +0.097 |
| Baker's | 0.7355 | 0.8279 | +0.092 |
| PF OA | 0.6821 | 0.7722 | +0.090 |
| Fracture | 0.7056 | 0.7764 | +0.071 |
| Lateral Meniscus | 0.7081 | 0.7280 | +0.020 |
| Effusion | 0.9391 | 0.9193 | **-0.020** |
| Synovitis | 0.7802 | 0.7133 | **-0.067** |

**MCL is no longer random** (0.5034 -> 0.6259) and the three worst labels of run 1 — MCL, Lateral OA,
Medial Meniscus — are the three that gained most. That is the coronal hypothesis behaving as
predicted **on the tier that matters**, the one scored against labels read from images rather than
text.

**What it does not license is attributing that to the laterality fix.** This run differs from run 1
in five variables at once (epochs, schedule, AMP, augmentation, second series). Screen B measured
laterality *in isolation* at -0.0014 full macro, and nothing here contradicts that; the coronal
*plane* is doing the work, not the mirroring. Two labels went backwards — Synovitis -0.067 and
Effusion -0.020, both sagittal findings that run 1 read well — which is consistent with a fixed
16-slice budget now split across two planes rather than spent on one.

**Pre-registered LB reading (NOTES 2026-09-05): >= 0.82 means the recipe was the gap.** Run 1's
gold 0.7252 preceded LB 0.763, an offset of +0.038; the same offset on 0.8189 would put this near
0.857. That extrapolation rests on one point at n=58 and should not be trusted beyond "expect a
large jump" — the pre-registered bands, not the point estimate, are what this gets read against.

### Phase 5 submission notebook run: 0 fallbacks, and the laterality guard fired on the test set (2026-09-06)

`notebooks/phase5-submit/`, version 1, T4, 3 visible studies:

- **prep 2.605 s/study** (p95 3.981), **forward 0.559 s/study for all five fold models** on GPU —
  **3.16 s/study end to end**, so roughly **68 min per 1,300 hidden studies**, still prep-bound and
  far inside the 9 h cap.
- **Doubling the input series cost almost nothing at inference.** Phase 4 measured 3.09 s/study at
  `max_series=1`; this is 3.16 s at `max_series=2`. The forward went 0.428 -> 0.559 s and prep did
  not move at all, because prep already decoded four series in both cases — the read config changes
  what the loader *returns*, not what prep *does*. Worth knowing for the Efficiency Track: the
  second plane is nearly free at inference and only expensive at training time.
- **0 of 3 studies hit the 0.5 fallback** (the local sample only ships DICOMs for 1 of 3, so the
  local dry run showed 2 fallbacks; the competition mount has all three).
- **`laterality routes on the test set: {'Laterality': 1, 'geometry': 2}`** — the geometry fallback
  is firing on hidden-test studies, two of three. This is the check that matters most in this
  notebook: it is direct evidence that training and inference resolve sides by the same rule, which
  is the mismatch that would otherwise have been invisible.
- Mean predictions track the training positive rates and preserve their ordering (Medial Meniscus
  0.141 vs 0.319, Effusion 0.120 vs 0.262, Fracture 0.005 vs 0.014). n=3, so a smoke signal.

**Submission mechanism, re-confirmed rather than re-learned:** `kaggle competitions submit -f` is
rejected with a bare 400 on this Code Competition (NOTES 2026-08-08). The submission has to be made
from the notebook's **Output tab -> "Submit to Competition"**, which links that notebook version's
output file and privately re-runs it against the real hidden test set.

### Correction: the CLI *can* submit to this Code Competition (2026-09-06)

The 2026-08-08 entry concluded that "`kaggle competitions submit -f <file>` is rejected outright
(400) ... the actual mechanism is the web UI's Submit to Competition panel, not the CLI." **The
first half is right and the conclusion is too broad.** A raw *local file* upload is rejected, which
is what was tested. But `kaggle competitions submit` also takes `-k` and `-v`, and that form works:

```
kaggle competitions submit -c rsna-knee-abnormality-detection \
  -k mohammadyaserhussain/knee-phase5-submit -v 1 -f submission.csv -m "<message>"
```

`-f` here is not a path — for a code competition it names *the output file the kernel produced*, and
`-k`/`-v` name the notebook version to link. Phase 5 went in this way (submission 56047536) and
registered as PENDING with the daily quota decremented, exactly as the web panel does.

The generalisation error is the lesson worth keeping: one failing invocation was recorded as "the
CLI cannot do this" rather than "this *form* of the CLI call cannot do this", and that stood
unchallenged for a month, adding a manual step to every submission. `--help` on the subcommand
would have shown `-k/-v` at any point.

### Leaderboard census: the field moved further than we did (2026-09-06)

Phase 5 scored **LB 0.835** (submission 56047536), up from Phase 4's 0.763. Then the leaderboard was
pulled in full for the first time since the plan was written — 3,221 teams, snapshot kept outside the
repo. It reframes the whole project.

| | plan doc (2026-08-08) | today |
|---|---|---|
| 1st | 0.934 | **0.954** |
| 10th | ~0.872 | **0.950** |
| 100th | — | 0.940 |
| **median team** | — | **0.899** |
| us | 0.558 (rank ~2440 equivalent) | **0.835, rank 2163 / 3221** |

Teams at or above a score: 0.95 -> 11, 0.94 -> 119, **0.93 -> 1,000**, 0.92 -> 1,299, 0.90 -> 1,606,
0.87 -> 2,008, 0.835 -> 2,169.

**The 0.93 shelf is the finding.** A thousand teams sit at >=0.93, and `pilkwang/rsna-knee-baseline-v1`
(512 votes) is an *inference* notebook shipping pre-fitted weights. That shape — a thousand teams
clustered on one number, at the score its most-forked public notebook produces — is a fork cluster,
not a thousand independent solutions. **Our own pipeline scores below what forking costs nothing to
get.** Two submissions of real work (0.763, 0.835) moved us from roughly rank 2440 to 2163; the
distance to 10th is 0.115 and the distance to the median team is 0.064.

**Three consequences, all load-bearing.**

1. **Correction to this entry, made the same week (see 2026-09-07 below): the field's standing was
   already known.** README.md has recorded "rank 2187/2936: the field's median is 0.899 and 10th place
   is 0.947" since commit 433328e on **2026-09-03** — before Phase 5's screens were run. So the failure
   was not a missing measurement. **The plan document was never updated from it**, and the plan document
   is what drives phase prioritization, so Phase 5 proceeded down its pre-registered payoff order as
   though 10th place were still 0.872. Phase 5's +0.072 was read as most of the distance to the top ten;
   against the number already in our own README it was 60% of the distance to the *median*. The lesson is
   not "re-pull the leaderboard" — that had been done. It is that a number recorded in a status document
   and never propagated into the document that governs decisions changes nothing.
2. **The Efficiency Track is not an escape hatch.** `AUC/(0.5 - maxAUC) + s/32400` prices 0.01 AUC at
   720 s of runtime. Against a field at 0.95, a fast 0.87 does not place — the AUC term swamps any
   runtime saving a free-tier notebook can produce. Accuracy gates both prizes, so there is no version
   of this project where the accuracy work gets skipped.
3. **The gold-transfer -> LB offset is shrinking and the extrapolation rule is retired.** Run 1: gold
   0.7252 -> LB 0.763, offset **+0.038**. Phase 5: gold 0.8189 -> LB 0.835, offset **+0.016**. The
   pre-registered reading extrapolated the first offset onto the second gold number and predicted
   ~0.857; the actual was 0.835. Two points do not fit a curve, but they do rule out a constant offset,
   which is what the rule assumed. From here, pre-register a *band* before each submission, never a
   point estimate off an offset.

**What it does not change:** Phase 5's internal findings all held (schedule +0.061, augmentation, the
coronal-plus-augmentation interaction, laterality null in isolation). The recipe improvements were
real. They were aimed at a target that had moved.

**What it does change:** the 2026-09-05 recon listed physical-scale sampling, multi-slot input, a
per-diagnosis attention head and a domain-appropriate backbone as "most of the distance to 0.93", and
Phase 5 took only the cheap half of that list — schedule, AMP, augmentation, a second series. Phase 6
takes the expensive half, plus the deferred report-hash fold leak and the label-source A/B that has
been item #1 in the payoff order since Phase 3 and has never been run.

### Step 4 gold arm: the field's report labels beat ours, and one public set is contaminated (2026-09-06)

The label-source A/B has been item #1 in the plan's payoff order since Phase 3 and had never been
run. It is CPU-only and took an afternoon. Four public label sets pulled (all CC0), scored against
the 58 gold studies alongside our Qwen3-4B pass. No fitting happens — each source is already a
per-study score — so plain AUC on the 58 is unbiased and LOO would measure the same thing slower.
Full per-label table in `results/label_source_gold.csv`.

| source | gold macro | paired delta vs ours, 95% CI |
|---|---|---|
| **rank-blend of all 5** | **0.8976** | **+0.0360 [+0.0105, +0.0619]** |
| stevenleehans `llm_labels_v4_blend` | 0.8927 | +0.0311 [+0.0040, +0.0582] |
| stevenleehans `llm_labels_v2` | 0.8873 | +0.0257 [-0.0010, +0.0523] |
| **ours (Qwen3-4B)** | **0.8616** | — |
| lixin73 `gpt56sol` | 0.8488 | -0.0128 [-0.0440, +0.0227] |
| pilkwang `report_labels_v2` | 0.8142 | -0.0474 [-0.0940, -0.0009] |

**Our labels are third of five, and the gap to the best single public set clears its CI even at
n=58.** Per label, stevenleehans v4 beats ours on nine of twelve, ties Medial OA, and loses only
Baker's (-0.01) and Fracture (-0.03); the biggest losses are MCL 0.89 -> 0.97, PF OA 0.83 -> 0.90 and
Contusion 0.81 -> 0.86. Blending all five by **rank** (the sources are on incomparable scales — hard
0/1 versus confidence-weighted) beats every single source and is the widest CI-excluding-zero margin
on the table. Both go into Step 5's c6 cell for the OOF arm; n=58 ranks candidates, it does not
settle per-label questions.

Worth noting what this costs to know: two months of Phase 3 work — a vLLM dead end, two bake-off
runs, a two-shard corpus labeling job — produced a label set the field beats with a public CSV. The
Phase 3 machinery was not wasted (it is what makes the comparison measurable, and ours still beats
two of the four public sets), but "report quality is the edge" was held as a thesis for two months
without once checking what everyone else's reports scored. **The comparison was one afternoon of CPU
at any point after 2026-08-09.**

**`yunusgmsoy/rsna-knee-llm-labels-4-source-merged` is excluded: it scores 1.0000 macro on all
twelve labels.** Its 58 gold rows match `train.csv` exactly — every label, every study. The other
four agree with gold on 78-82% of cells, which is a normal LLM-vs-rubric rate. Substituting gold on
those 58 rows is legitimate for *training* (the labels are public), but it makes the set unusable as
an evaluation candidate and would silently poison any gold comparison it entered. Caught only
because a perfect score is implausible enough to look at twice. **Rule from this: any candidate
scoring near-perfect on the gold set is contaminated until proven otherwise — check row-exact
agreement with `train.csv` before believing a number.**

### Phase 6 Step 1 — prep v2 built and dry-run; the slot census came free (2026-09-06)

`crop_to_mm`, `select_slots`, `select_slice_groups`, `prep_slots` implemented TDD-first;
`_read_slice_header` gained four weighting tags; `load_study_npz` gained an explicit key order.
**160 tests pass (142 before, +18).** Four `phase6-prep-shard{0..3}` notebooks generated and
dry-run end to end against the local 20-study sample — every gate cell executes, and the physical
scale assertion passes at exactly 130/336 = 0.3869 mm/px.

**The slot design changed after one crosstab.** The plan said to recover the weighting axis from
TR/TE/`ScanningSequence`. `Fluid_Sensitive` and `Fat_Suppression` are **identical on all 24,371
rows** of `train_series.csv` — one column under two names, exactly as pilkwang reports. So the six
slots are 3 planes x fluid 0/1, straight from the delivered CSV: no header parsing, no hidden-test
risk, and it is pilkwang's own `SLOTS_PUBLIC` switch. The T1-vs-non-suppressed-PD/T2 split is a later
screen cell, and the header tags are recorded into per-series meta now so it can be measured without
re-prepping 4,407 studies to obtain them.

**Slot fill, computed from the CSV alone before prepping anything:**

| slot | studies | fill |
|---|---|---|
| AX_FLUID | 4,407 | 100.0% |
| SAG_STRUCT | 4,266 | 96.8% |
| COR_FLUID | 4,248 | 96.4% |
| SAG_FLUID | 4,150 | 94.2% |
| COR_STRUCT | 3,406 | 77.3% |
| **AX_STRUCT** | **857** | **19.4%** |

Mean **4.84 of 6**; no study has zero slots; 566 studies fill all six. **AX_STRUCT is the one to
remember**: it is masked off for four studies in five, so if the slot or attention screens read null,
check whether that slot is carrying the null before concluding the head cannot use the input. It is
kept rather than dropped because an empty slot stores nothing and the presence mask is exactly the
mechanism for this — but it is one attention position that is absent 80% of the time.

**Sizing, measured on a real study before launching:** 15 slices/slot at 336px is 1.86 MB for a
4-slot study, i.e. ~465 KB per filled slot. At 4.84 slots/study over 4,407 studies that is **9.5 GB
total, 2.4 GB/shard** — inside the output cap with room, so `N_GROUPS=5` at 336 stands. Prep runs
~1.6 s/study locally, so a 1,102-study shard is well inside a 12 h CPU session. If it ever has to
come down the knob is `N_GROUPS`; `OUT_SIZE` is the axis under test and 224px misses a 1 mm tear at
this crop.

Two edge routes decided rather than defaulted, per the gate: a series with **no `PixelSpacing`** is
skipped and its slot left empty (a fixed-pixel fallback would silently reintroduce the exact defect
this rewrite removes, on only the rows nobody can check), and a **field of view under 130 mm** is
zero-padded after centring, not upscaled. Both counted in the shard census. The local sample shows
0 of 20 under the crop, min FOV 150 mm.

Prep v1 stays alongside v2 until Step 3's re-baseline confirms the artifact; the two share
`_series_headers`/`_decode_selected` so their failure census cannot drift. Deleting v1 — the
letterbox path, `_pad_to_square`, `_priority_rank`, `load_study_npz`'s `max_series` ordering, and
their tests — is part of Step 3, not a someday item.

**Crop centring checked before launching, because it is the one defect that would make all 9.5 GB
wrong silently.** `crop_to_mm` centres on the array, and a 130 mm crop of a 320 mm acquisition takes
the middle of the *image*, which is only the middle of the *knee* if the knee is centred. Note the
distinction that nearly sent this the wrong way: `image_centre_x`'s median |c_x| of 83 mm is the
image centre in **patient** coordinates — it says a knee is off the body midline, which is expected,
and says nothing about where the joint sits inside the array. The testable form is the signal
centroid against the array centre. On 25 real series (both sample sets, middle slice, >60th
percentile intensity as the anatomy mask): median offset **10.6 mm**, max **26.5 mm**, **0 of 25
past 30 mm** (a quarter of the crop), and the one 320 mm-wide-FOV series is 10.6 mm off. So a centred
crop lands on the joint. n=25 and a centroid is a proxy, so the shard-0 visual grid is still the real
gate — it now prefers studies filling all six slots, since AX_STRUCT at 19.4% would otherwise never
appear in an arbitrary first-8 grid and would ship unlooked-at.

### Phase 6 Step 2 — folds re-frozen on report text, leak eliminated (2026-09-06)

`make_folds` gained a `groups` argument: groups are assigned largest-first to whichever fold is
currently smallest, rather than round-robin, because one real report group has 37 members and
round-robin over groups would drop that whole block into one fold. `groups=None` keeps the original
path byte-for-byte — `folds_primary_v2.csv` is frozen and every Phase 4/5 number is paired against
it, so a silent change there would invalidate the Phase 5 promotion decision retroactively. There is
a regression test asserting exactly that.

`results/folds_primary_v3.csv`: 5 folds of 882/882/881/881/881 (identical balance to v2), gold still
all 58 present for the holdout filter, and **0 report groups straddling folds, against 47 groups /
190 studies (4.31%) in v2.**

**The grouping key is normalized, and that revises the 2026-09-05 count.** That entry recorded "49
groups covering 183 studies". Re-measured today, the count depends entirely on how report text is
compared:

| key | duplicate groups | studies |
|---|---|---|
| raw bytes | 46 | 177 |
| `.strip()` (what the 2026-09-05 entry used) | 49 | 183 |
| **strip + lowercase + collapse whitespace** | **54** | **204** |

v3 uses the last one. Two reports differing only in case or run-length of whitespace are the same
report and leak across a fold boundary in exactly the same way, so the conservative key is the
correct one and it catches 27 studies the byte-identical count misses. Nothing about the earlier
entry was wrong — it measured what it said it measured — but "byte-identical" was the wrong
definition of "same report" for this purpose.

### Phase 6 Step 1 pilot: every gate passes on real multi-series studies (2026-09-06)

`knee-phase6-prep-shard0` v1, `PILOT_N=50` (65 studies after unioning in shard 0's 15 gold),
**0 failed, 0 decode failures**, 74.3 slices stored per study.

**The physical-scale gate is the one that matters, and it is unambiguous: 322 stored series carrying
90 distinct `PixelSpacing` values came out at exactly one mm/px — 0.38690 = 130/336.** That is the
defect prep v2 exists to remove, demonstrated rather than asserted. Under the old letterbox those 90
scales reached the model as 90 different physical magnifications of the same anatomy.

**Slot fill matched the CSV prediction**, which also confirms series on disk and series in
`train_series.csv` agree — a mismatch there would have been invisible downstream:

| slot | pilot | predicted from CSV |
|---|---|---|
| AX_FLUID | 100.0% | 100.0% |
| SAG_FLUID | 98.5% | 94.2% |
| SAG_STRUCT | 96.9% | 96.8% |
| COR_FLUID | 95.4% | 96.4% |
| COR_STRUCT | 86.2% | 77.3% |
| AX_STRUCT | 18.5% | 19.4% |

Mean 4.95 of 6 against 4.84 predicted. Laterality routes on the pilot: `Laterality` 34, `geometry`
27, `SeriesDescription` 2, unresolved (`geometry_midline`) 2 — the geometry route is carrying 42% of
these studies, as at corpus scale. FOV under the crop: **1 of 322 series** (min 81 mm), so the
zero-pad path fires about as rarely as predicted.

**Visual gate, 8 studies x 6 slots, rendered and inspected.** Planes are correct per column
(sagittal profile, coronal condyles, axial cross-section with anterior patella); the `_FLUID` columns
are visibly fat-suppressed against bright marrow in `_STRUCT`; **the joint sits centred in every
panel**, which retires the crop-centring risk on real data rather than on the 25-series centroid
proxy; and `AX_STRUCT` is populated in the five six-slot rows, so the 19.4% slot was actually looked
at instead of shipping unseen.

### Phase 6 Steps 5-6 groundwork: slot loader and rank-mean (2026-09-06)

Two pieces built while the prep shards run, both local and independent of the artifacts.

**`PreppedSlotDataset`** returns `(image [n_slots, n_groups, 3, H, W], mask [n_slots] bool, labels,
uid)`. This is the replacement `PreppedStudyDataset`'s own docstring asked for: v1 flattened every
series into one slice axis and padded by *repeating the last series*, so a study with three
acquisitions and a study with two look identical to the model — a repeated series and a real second
series are the same tensor. An absent slot here is zeros with its mask entry false, which is what a
per-diagnosis attention head needs to decline to attend. The channel axis holds the three adjacent
slices of a stored group, which is the reason prep stores groups at all. Mirroring stays a load-time
decision and still respects `mirrors_in_plane` — the slot's plane comes from the slot table, so
sagittal slots are never flipped.

**`infer.rank_mean`** combines fold predictions by averaging per-label ranks across studies rather
than averaging probabilities. Macro AUC is invariant under any increasing map, so it reads order and
nothing else, and a probability mean lets the member with the widest spread dominate. The test that
pins the claim: three members where one is confident and wrong about two studies and two are
diffident and right — the probability mean inherits the confident member's error, the rank mean
follows the majority. `scipy` moved from a transitive scikit-learn dependency to an explicit one in
`requirements-dev.txt`, since `rankdata` is now imported directly (and it handles ties as ties,
where an argsort-based rank would invent an ordering the models never expressed).

**175 tests pass** (165 before). Note on verification: this box has 7 GB and a browser open, and the
torch-importing test files get OOM-killed mid-run when free memory drops under ~600 MB — an `exit
143` from `pytest` here is the OOM killer, not a failing test. Run them as two groups if that
happens: the seven torch-free files (119) and the three torch files (56).

### Phase 6 prep v2 ran the corpus; screen notebook built and dry-run (2026-09-07)

All four `phase6-prep-shard` kernels completed. The corpus census is asserted inside the screen
kernel rather than downloaded — 9.5 GB of artifacts do not need to leave Kaggle to be checked, and
the training kernel should assert its own inputs anyway. (Worth recording why: the parallel download
was started and killed — the box has 8.5 GB free on `/`, so pulling three shards would have
near-filled the disk for information the next kernel reads for free.)

**Model and training loop now carry the slot structure.** `KneeModel.forward` accepts both
`[batch, n_images, 3, H, W]` and `[batch, n_slots, n_groups, 3, H, W]`, folding the slot and group
axes itself so `mask` stays addressable per slot, and pools only over present slots. That masking is
not cosmetic: an absent slot arrives as zeros, and an unmasked mean averages real anatomy with a
black square, so the bias would scale with how many slots a study is missing — and slot fill runs
from 100% (AX_FLUID) to 19% (AX_STRUCT), which means the bias would track missingness, a property of
the scanner protocol rather than of the knee. `train_one_epoch`/`evaluate` read either loader shape
through `_unpack_batch`, so one loop serves v1 and v2 while v1 is still the artifact behind the last
scored submission.

**A note on testing the mask.** The first version of that test used a real `efficientnet_b0` and
asserted that unmasked pooling differs from masked. It does not — an untrained backbone emits nearly
identical features for black as for noise, so the test passed whether the mask was honoured or
silently dropped, which is the one thing it exists to catch. It now stubs the backbone with a module
returning each image's own mean, where the arithmetic is checkable exactly (4.0 masked vs 2.0
unmasked).

**`report_group_key` moved into `knee/reports.py`** because the screen kernel regenerates folds
in-kernel and pins them by SHA (`FOLDS_PRIMARY_V3_SHA256 =
f1d6ba7c341f8d2e82241cb4ddd5ca5131b0c6d2033eb6d2e43bcb597cf12868`) rather than mounting a CSV — the
Phase 5 pattern. A normalization living only in a scratch script could not be reproduced there.

**The screen dry run caught a bug that would have cost a GPU session.** Running the real training
path over the 64 pilot artifacts locally failed at scoring: `roc_auc_score` rejects a continuous
`y_true`, and the notebook was passing the soft pseudo-label scores to both loaders. Phase 5 used
*two* frames — soft scores as the training target, `(scores >= 0.5)` for evaluation — and the new
notebook had collapsed them into one. Fixed, and the dry run now completes the full
loader -> model -> train -> evaluate path.

**One simplification the new artifact allows:** the screen drops Phase 5's `sides` override. That
existed because v1 artifacts were prepped before the geometry laterality route and carried
`side=None` for ~49% of studies; `prep_slots` calls `census_study_laterality`, so a v2 artifact
already carries the geometry-resolved side. Re-applying an external map would be a second source of
truth for the same fact.

180 tests pass.

### Step 3 re-baseline: prep v2 verified, and it is faster than what it replaces (2026-09-07)

`knee-phase6-screen` v1, fold 0, 2 of 6 slots at 224px, 8 epochs OneCycle + AMP + augmentation,
3,486 train / 863 val, gold excluded.

| | Phase 5 b4 (prep v1) | Step 3 re-baseline (prep v2) |
|---|---|---|
| fold-0 macro | 0.8484 | **0.8574** |
| minutes | 68.6 | **18.6** |
| encoder inputs per study | 32 (1 slice each) | 10 (3 adjacent slices each) |

**+0.0090 macro at 3.7x less compute.** Read the sign, not just the band: the gate was "within ~0.01
with the per-label profile intact, a large *drop* means the prep is wrong", and this is inside the
band on the good side. The two are not paired — different pixels, different folds — so this is not a
promotion, it is the row every Phase 6 screen delta gets measured against. The compute number is the
more useful half: a 5-fold confirm is now ~1.5 h rather than 5.6 h, which buys back most of the
budget the phase was worried about.

**The physical-scale fix at corpus scale: 21,334 stored series carrying 1,028 distinct
`PixelSpacing` values, every one of them at 0.38690 mm/px.** The pilot showed 90 distinct spacings
collapsing to one; the corpus shows 1,028. Under the old letterbox those were 1,028 different
physical magnifications of the same anatomy, and no downstream capacity recovers that. 4,407
artifacts, 0 failed studies, 0 decode failures; slot fill matched the CSV prediction exactly (mean
4.84 of 6); folds v3 verified against its SHA with 0 straddling report groups; laterality routes
Laterality 2,179 / geometry 2,088 / SeriesDescription 74 / unresolved 66.

**The kernel still exited ERROR, from my own bug:** a `del model, optimizer, scaler, scheduler` was
inserted ahead of the `torch.save(model.state_dict(), ...)` that follows it, so the checkpoint save
raised `NameError`. Everything that matters survived — the OOF arrays and the `experiments.csv` row
are written before the del — so the run is not repeated; only the fold-0 weights are lost, and a
screen row does not need them. Ordering fixed, with the reason recorded next to it.

### Why the rank blend cannot be the training target, even though it wins on gold (2026-09-07)

The Step 4 gold arm ranked a 5-source rank blend first (0.8976 vs steven_v4's 0.8927). Carrying that
into c0 as a *training* target was about to be a mistake, and the check that caught it was looking at
the target's own distribution rather than its AUC:

| label | ours (Phase 5 target) | probability mean x5 | rank blend x5 |
|---|---|---|---|
| Fracture | 0.014 | 0.220 | **0.500** |
| MCL | 0.021 | 0.182 | **0.500** |
| Lateral OA | 0.025 | 0.303 | **0.500** |
| Effusion | 0.262 | 0.518 | **0.500** |

Percentile ranks are uniform by construction, so **every label's target mean is exactly 0.500** and
all prevalence information is gone. AUC never noticed because AUC reads order only — which is
precisely why the rank blend is the right way to *combine model predictions at submission time*
(`infer.rank_mean`) and the wrong way to build a target to train against.

The probability mean is usable but not innocent either: the public sources are far more liberal than
ours (overall mean 0.45 against 0.22), so swapping the target also shifts every label's positive
rate. **c0 therefore tests `steven_v4` alone against ours** — the cleanest source-vs-source
comparison, holding the target's construction as close to Phase 5's as the sources allow, and
steven_v4 is the best *single* source on gold anyway. Rank-blending the sources with a per-label
recalibration back to a sane prevalence is a follow-up worth trying only if the source swap pays.

### c0 promoted: the public label set trains a better model, and the circular metric said otherwise (2026-09-07)

`knee-phase6-screen` v2 ran both arms in one session, identical in everything but the training
target, with the 58 gold studies scored for each.

| arm | val macro (own target) | **gold macro** | min |
|---|---|---|---|
| re-baseline — our Qwen3-4B | **0.8509** | 0.8279 | 18.5 |
| c0 — stevenleehans `llm_labels_v4_blend` | 0.8483 | **0.8617** | 17.7 |

**Gold transfer, c0 minus re-baseline: +0.0339 [+0.0067, +0.0625]. CI excludes zero — promoted.**

**The reason this run was designed with a gold arbiter is visible in the first column: on
val-against-own-target, c0 looks *worse* (0.8483 vs 0.8509), and a screen that compared label
sources that way would have rejected it.** Each arm graded by its own marker measures how learnable
a label set is, not which one trains a better model.

The saved OOF arrays give a free home-and-away check at n=863 — both arms scored the same val
studies in the same order, so either model can be re-scored against either label set:

| scored against | base (ours) | c0 (steven) | paired delta (c0 - base) |
|---|---|---|---|
| **our labels** (incumbent's home turf) | 0.8509 | 0.8533 | +0.0024 [-0.0142, +0.0165] |
| steven_v4 labels (challenger's home turf) | 0.7953 | 0.8483 | +0.0530 [+0.0423, +0.0652] |
| **gold, n=58** (neutral, rubric-graded) | 0.8279 | 0.8617 | **+0.0339 [+0.0067, +0.0625]** |

Read together these are coherent: c0 **ties on the incumbent's own marker**, wins on its own, and
wins on the only marker neither model was trained toward. A promotion resting on the challenger's
home turf alone would be worthless; this one does not. The two label sets agree on only **83.3%** of
label cells, so these are genuinely different labelings, not a re-scaling.

**Deviation from the plan, stated rather than slipped in:** the plan said c0 "ships on the OOF paired
delta over the 4,349, not on 58 studies." That instruction cannot be followed as written when the
*target* is the variable — every OOF is scored against one of the two competing label sets, so it is
circular by construction. The gold set is the only non-circular instrument available, and the n=863
cross-scoring above is the power the plan wanted, recovered in the one form that stays neutral.

**Free measurement worth keeping: fold-0 run-to-run noise is about 0.0065.** The same re-baseline
config ran in v1 and v2 of this kernel and scored 0.8574 and 0.8509 — same code, same folds, same
data, different seed state. **Any single-fold screen delta smaller than ~0.007 is noise**, which
retroactively puts Phase 5 screen B's b4-vs-b3 (+0.0052) below the resolution of the instrument that
measured it — consistent with what that entry already said ("a judgement on mechanism, not a measured
separation"), but now with a number attached.

Consequence: **steven_v4 is the training target for every subsequent Phase 6 cell.** Our Qwen3-4B
labels are retired as the primary target after two months as the plan's central edge — kept in
`results/pseudo_labels_qwen3_4b.csv`, and still the better source on Fracture and Baker's, so a
per-label blend is a live follow-up.

### c2's attention head built and dry-run while c1 trains (2026-09-07)

`KneeModel(head='slot_attention')` replaces the mean pool with per-diagnosis masked attention over
slots: one attention score per (slot, diagnosis), softmax over slots with absent slots masked out,
and a per-diagnosis readout so each label reads its own pooled feature rather than a shared one.
Groups within a slot are averaged first — they are the same acquisition at different depths, so the
axis worth attending over is the slot, which is the plane and the weighting.

`attention_weights()` is exposed deliberately: a macro delta says whether c2 helped, and the weights
say whether it helped *for the reason claimed*. The screen prints them per diagnosis, averaged over
the gold studies, so "MCL reads the coronal plane" is checkable against anatomy rather than assumed.

**Two of the four new tests would have passed on a broken implementation, which is the interesting
part.** With a real untrained `efficientnet_b0`, every slot produces nearly the same features, so the
attention softmax comes out uniform (0.25 across four slots) no matter what the head does — the
per-diagnosis test passed trivially and proved nothing. Same failure mode as the masked-pooling test
earlier in the phase. Both now stub the backbone with a module returning each image's own mean, where
slot features genuinely differ and the arithmetic is checkable; the stub is lifted to module level
with the reason documented, since it is now needed twice and will be needed again.

Dry-run on the pilot artifacts: 6 slots, attention head, 1 epoch, attention weights `(24, 6, 12)`
summing to 1 over slots per study, `AX_STRUCT` receiving 0.028 average weight — the mask doing its
job on the slot that is absent for four studies in five.

### c3 groundwork: differential learning rates and a DINOv2 that builds (2026-09-07)

`differential_param_groups(model, backbone_lr, head_lr)` splits parameters on the `backbone.` prefix.
A pretrained trunk already represents images well and the head is random, so a single learning rate
is either low enough not to wreck the trunk or high enough to train the head and cannot be both —
that split is what makes a 21.6M-parameter backbone usable on 3,486 studies. The slot-attention
parameters go with the head, since they are as freshly initialised as the linear one.

No explicit block freezing (marked `ponytail:` in the source): a backbone at 8e-6 against a head at
1e-3 is already a near-freeze, and "last N blocks" is spelled differently on every architecture. Add
it only if a screen shows the trunk drifting.

**Checked before committing a GPU session to it:** `vit_small_patch14_dinov2.lvd142m` builds and runs
at **both 224 and 336** (21.6M/21.8M params, 384-dim features). Patch-14 needs an input divisible by
14, which 224 and 336 both are — worth confirming rather than discovering on Kaggle. It also has to
be told its input size at construction, because timm's default for that checkpoint is 518 and the
position embedding is built for it; `KneeModel` now forwards `**backbone_kwargs` to `timm.create_model`
rather than growing a branch per architecture.

**Known follow-up, not a blocker:** the screen kernel has internet on, so timm downloads the DINOv2
weights. The *submission* kernel must run offline, so if c3 wins, the weights have to come from an
attached Kaggle model (`metaresearch/dinov2/PyTorch/small/1` is the public one pilkwang uses) rather
than a download. That is a Step 7 problem and it is recorded here so it is not discovered at
submission time.

### c1: six slots pay on the high-power tier and not on the low-power one (2026-09-07)

`knee-phase6-screen` v3, both arms on the steven_v4 target so val-against-own-target is a *legitimate*
paired comparison this time — same labels, same studies, same order. That was not true for c0, where
the target itself was the variable.

| arm | val macro (n=863) | gold macro (n=58) | min |
|---|---|---|---|
| c0 — 2 slots | 0.8464 | **0.8657** | 18.5 |
| c1 — 6 slots + presence mask | **0.8580** | 0.8579 | 47.1 |

| tier | c1 - c0 | |
|---|---|---|
| **val, n=863** | **+0.0116 [+0.0044, +0.0185]** | CI excludes zero |
| gold, n=58 | -0.0078 [-0.0338, +0.0181] | spans zero |

**Eleven of twelve labels improve on val** — Synovitis +0.026, PF OA +0.018, Contusion +0.018,
MCL +0.016 — and only Lateral Meniscus goes backwards (-0.007). The per-label pattern is diffuse
rather than concentrated in the planes the new slots add, which is what a mean-pool head over more
input should look like: more evidence, no ability to route it.

**The two tiers do not contradict each other, and it matters not to over-read either.** The gold CI
[-0.034, +0.018] comfortably contains the val estimate of +0.0116; n=58 simply cannot resolve a
difference that size. But the direction is worth holding onto, because gold is the tier that
resembles the test set — rubric-graded from images — while val is scored against report-derived
labels. A gain that shows up only against report labels would be "better at fitting the LLM's
reading" rather than better at the task, which is exactly the failure the plan's three-tier structure
exists to catch. **Unresolved, not resolved in c1's favour.**

**Runtime: 2.5x for +0.0116.** The promotion rule prices 0.01 AUC at 720s, so +0.0116 buys 837s and
c1 spends 1,716s. It passes for the accuracy submission, where runtime is irrelevant under a 9h cap,
and **fails for the Efficiency Track** — the first Phase 6 change where the two final submissions
want different answers. Recorded now so Sub B is not assembled later on the assumption that every
promoted change belongs in it.

Also: c0 re-ran identically in v2 and v3 and scored gold 0.8617 then 0.8657, val 0.8483 then 0.8464.
**Gold run-to-run noise is ~0.004 at n=58, val ~0.002** — consistent with the ~0.0065 measured earlier
on a different config, and a reminder that the gold tier is noisier per point than its CI width alone
suggests.

Next is c2: the same six slots with per-diagnosis attention instead of the mean pool. Screen B's
lesson applies directly — a second plane read as a failure alone (b1, +0.0019) and as the phase's
best finding once the model could use it. c1 is this phase's b1.

### The stale-src failure, third occurrence — and why the existing guard was useless (2026-09-07)

c2 died in the first config of the training loop: `KneeModel.__init__() got an unexpected keyword
argument 'head'`. `head=` was implemented locally and tested, and `rsna-knee-src` was never
re-versioned before the kernel was pushed, so the kernel imported a snapshot from before the c2 head
existed. Same failure as NOTES 2026-09-05, third time on this project.

**The guard that was supposed to prevent this passed.** It read:

```python
assert 'mask' in inspect.signature(KneeModel.forward).parameters
assert 'groups' in inspect.signature(make_folds).parameters
```

Both true — those were the *previous* change. A hand-maintained list of signatures only ever catches
the mistake that has already been made once, and it has to be extended by exactly the person who is
already forgetting to sync the dataset. It gave the appearance of protection while checking nothing
relevant to the run being launched.

Two fixes, because the guard and the habit are different problems:

1. The guard now lists every capability the configs actually use — the `head` and `**backbone_kwargs`
   arguments, `mask`, `groups`, and `differential_param_groups`'s two rates — with an error naming the
   real cause ("rsna-knee-src is stale"). **It was tested against a stand-in for the stale signature
   and seen to fire**, which the previous one never was; a guard nobody has watched fail is an
   assumption, not a check.
2. The push step now re-versions the `rsna-knee-src` dataset, waits for it to report ready, and only
   then pushes the kernel. Versioning costs ~60s against a kernel that costs an hour, so it is
   unconditional rather than a judgement call about whether this particular push needs it.

Cost this time: one queue slot and ~20 minutes, caught in the first cell of the loop rather than
after training. The v1 checkpoint bug and this one share a shape — both were mistakes in the
*scaffolding* around a correct experiment, not in the experiment.

### Correction: the Efficiency Track is AUC-dominated, but "places in neither track" was too strong (2026-09-07)

The 2026-09-06 census entry, the plan and the README all say a fast 0.87 "does not place in either
track." Worked through properly, that overstates it. Score = `AUC/(0.5 - 0.95) + s/32400`, minimized:

| entry | AUC | min | score |
|---|---|---|---|
| a 0.95 rival, brisk | 0.950 | 30 | **-2.0556** |
| a fast 0.87 | 0.870 | 20 | -1.8963 |
| us, hypothetical 0.90 | 0.900 | 68 | -1.8741 |
| us, Phase 5 as submitted | 0.835 | 68 | -1.7296 |
| **a 0.95 rival, leisurely** | 0.950 | 240 | **-1.6667** |

**A 0.95 model that takes four hours scores worse than our submitted 0.835 at 68 minutes.** So the
track does what it advertises — it rewards accuracy per second, and a lean weak model genuinely beats
a heavy strong one. "Places in neither track" was wrong.

What survives from that entry is the part that drives decisions: **the exchange rate is 0.010 AUC per
12 minutes**, so AUC dominates anything runtime can recover. Cutting 68 minutes to 10 buys the
equivalent of 0.048 AUC; the gap to a 0.95 rival is 0.115. To match one running at 30 minutes we
would need **0.925 at zero runtime**. Accuracy is still the priority for both tracks — the conclusion
was right and the argument for it was not.

The practical consequence is narrower than "ignore runtime": our efficiency pick should be the most
accurate configuration that stays under roughly half an hour, and c1's 2.5x runtime for +0.0116 is
exactly the trade to refuse there (+0.0116 is worth 14 minutes; c1 spends 29). It remains the right
trade for the accuracy submission under a 9h cap.

### c2: the attention head learned the anatomy, and did not measurably beat the mean pool (2026-09-07)

`knee-phase6-screen` v5. Six slots, per-diagnosis masked attention, steven_v4 target, 51.0 min.

| arm | val (n=863) | gold (n=58) | min |
|---|---|---|---|
| c0 — 2 slots, mean pool | 0.8464 | 0.8657 | 18.5 |
| c1 — 6 slots, mean pool | 0.8580 | 0.8579 | 47.1 |
| **c2 — 6 slots, attention** | **0.8607** | **0.8759** | 51.0 |

| comparison | val (n=863) | gold (n=58) |
|---|---|---|
| c2 - c1 | +0.0027 [-0.0035, +0.0092] | +0.0179 [-0.0056, +0.0401] |
| c2 - c0 | **+0.0143 [+0.0062, +0.0216]** | +0.0101 [-0.0166, +0.0405] |

**c2 is not a measured separation from c1.** The val delta is +0.0027 against a run-to-run noise floor
of ~0.0065, and both CIs span zero. c2's gold macro is the best of the phase (0.8759) and the gold
delta leans positive, but at n=58 that is a lean, not a result. The honest statement is: **six slots
pay (c1 vs c0, CI excludes zero); routing them with attention rather than averaging them does not
demonstrably pay yet.** This is where Phase 5's b4 decision went wrong in the other direction, and the
note there — "a judgement on mechanism, not a measured separation" — applies verbatim here.

**But the mechanism is verified, and that is the part worth keeping.** The attention weights, averaged
over the gold studies, put each diagnosis on the plane the anatomy says it belongs to — with nothing
in the loss telling it so:

| diagnosis | highest-weight slot | weight | anatomically right? |
|---|---|---|---|
| ACL | SAG_FLUID | 0.314 | yes — the ACL is read sagittally |
| **MCL** | **COR_FLUID** | **0.440** | yes — the MCL is a coronal structure |
| **Baker's** | **AX_FLUID** | **0.700** | yes — a popliteal cyst is an axial finding |
| Effusion | AX_FLUID | 0.394 | yes — suprapatellar pouch |
| Medial Meniscus | COR_FLUID 0.426 / SAG_FLUID 0.335 | | yes — menisci read on both |
| Contusion | SAG_FLUID 0.377 / COR_FLUID 0.341 | | yes — fluid-sensitive, marrow oedema |

`AX_STRUCT` receives 0.008-0.056 everywhere, which is the mask behaving on the slot that is absent for
four studies in five. And the val labels that gain most over c1 are exactly the plane-specific ones —
**Lateral Meniscus +0.032, MCL +0.018** — with Fracture the notable loss (-0.021).

So the head is doing what it was built to do; the macro just cannot yet tell the difference. **Keep
c2**: it costs 4 minutes over c1, it carries the phase's best gold number, its per-label gains land
where the mechanism predicts, and a 5-fold confirm scores val on 4,349 studies rather than 863, which
is roughly 2.3x the resolution and may separate what this screen cannot. Recorded as "kept on
mechanism at equal cost", not as promoted on a delta.

Process note: the band check printed "OUTSIDE the band -- check the prep" for c2's **+0.0123**. That
message was written for a large *drop*; a positive delta trips it too. Read the sign, not the verdict.

### c3 launched: DINOv2-small, sized to what the session actually allows (2026-09-07)

A FLOP estimate before booking the session, rather than after losing one to a 12h kill:

| backbone | GFLOPs @224 |
|---|---|
| efficientnet_b0 | 0.4 |
| vit_small_patch14_dinov2 | 6.0 |
| vit_small_patch14_dinov2 @336 | 15.3 |

**~15x.** Against c2's 51 min for 30 images per study, that puts the configurations at roughly:
6 slots x 5 groups **13.2 h** (past the cap), 2 x 5 **4.4 h**, 2 x 3 **2.6 h**.

So c3 runs at **2 slots x 5 groups, 224, mean pool** — c0's exact shape. That is not a compromise
chosen to save time so much as the one that pairs cleanly: c0's arrays are already saved with the same
input, target and head, so the backbone is the only thing that moves and no control needs re-running.
The six-slot version is a later question, and it needs either a cheaper backbone or fewer groups.

**The differential learning rate is not a separable second variable.** A pretrained ViT trained at
OneCycle 3e-4 would be destroyed; "use this backbone" and "run its trunk at 8e-6 against a 1e-3 head"
are one change, and pretending otherwise would mean screening a configuration nobody would ship.

Constructed and run locally before pushing — 174 backbone parameter tensors against 2 head tensors,
per-group `max_lr` accepted by OneCycleLR, forward/backward/step clean. `OneCycleLR` takes a list of
max_lr in param-group order and raises on a length mismatch only at construction, which on Kaggle is
three minutes into a four-hour booking.

### c3: DINOv2 is not better at this shape, and my runtime model was wrong by 12x (2026-09-07)

| arm | val (n=863) | gold (n=58) | min |
|---|---|---|---|
| c0 — 2 slots, efficientnet_b0 | 0.8464 | 0.8657 | 18.5 |
| c3 — 2 slots, DINOv2-small | 0.8512 | 0.8686 | **23.3** |

| comparison | val | gold |
|---|---|---|
| c3 - c0 (backbone isolated) | +0.0048 [-0.0032, +0.0129] | +0.0028 [-0.0223, +0.0285] |
| c3 - c2 (best so far) | -0.0095 [-0.0183, -0.0007] | -0.0073 [-0.0379, +0.0228] |

**The backbone swap is not a measured improvement.** Both CIs against c0 span zero, and c3's val gain
is below the ~0.0065 noise floor.

**But the runtime estimate that sized this run was wrong by a factor of twelve.** The FLOP model
predicted DINOv2 would be ~15x efficientnet_b0; wall clock came in at **1.26x** (23.3 min vs 18.5).
The reason is architectural: `efficientnet_b0` is FLOP-cheap and *memory-bandwidth-bound* — depthwise
convolutions leave a T4 badly under-utilised — while a ViT is dense matmul that saturates the tensor
cores. **FLOPs do not predict wall clock across architecture families**, and I sized c3 down to two
slots to fit a session that was never at risk. Six slots x five groups with DINOv2 is roughly 70
minutes, not 13 hours, and was affordable all along.

**Two reasons not to write DINOv2 off on this row.**

1. **It is underfit.** Final training loss 0.4459 against c0's 0.3966, still falling steeply at epoch
   8 (0.4671 -> 0.4459). A trunk at 8e-6 barely moves in eight epochs; the schedule that suits a
   from-scratch efficientnet does not suit a frozen-ish pretrained trunk.
2. **It is a genuinely different model, not a worse one.** Per label against c0: **MCL +0.039,
   Contusion +0.036, Fracture +0.020, Effusion +0.020** against **ACL -0.032, PF OA -0.022, Lateral
   Meniscus -0.017**. Complementary strengths that trade off almost evenly in the macro are exactly
   what makes an ensemble member valuable, and Phase 6 has an ensembling step (`infer.rank_mean`)
   waiting for members.

Next is the stack: DINOv2 with six slots and the attention head. That pairs against c2 with only the
backbone moving, is the plan's c5, and — now that the runtime is known rather than modelled — costs
about 70 minutes. Screen B's lesson is the reason not to stop at c3's null: the coronal series read as
a failure alone and as the phase's best finding in combination.

### Ensembling the screen arms: free, large, and it rescues c3 (2026-09-07)

Every arm saved its val and gold predictions, so the ensemble question costs no GPU at all. All
fold-0 models, same 863 val studies and same 58 gold.

| members | val | gold |
|---|---|---|
| c0 alone | 0.8464 | 0.8657 |
| c1 alone | 0.8580 | 0.8579 |
| c2 alone | 0.8607 | 0.8759 |
| c3 alone | 0.8512 | 0.8686 |
| c2+c3 | 0.8715 | 0.8885 |
| **c1+c2+c3** | **0.8742** | 0.8860 |
| c0+c2+c3 | 0.8714 | **0.8918** |

**Every pair beats both its members.** c1+c2+c3 against c2 alone is **+0.0116 [+0.0073, +0.0163]** on
val — the same size as the entire six-slot gain, for no training at all.

**c3 earns its place here, having been null as a single model.** c2+c3 (0.8715) beats c2 (0.8607) by
more than c2 beat c1, and c3's contribution is exactly the complementarity its per-label profile
advertised — it is strong on MCL, Contusion, Fracture and Effusion where c2 is weaker. A screen that
had discarded c3 on its own macro would have thrown away the best ensemble partner in the set.

### Correction: rank-mean lost to probability-mean, 11 combinations out of 11

Step 6 of the plan called rank-mean "one line, unconditional... it cannot lose on the metric", and
`infer.rank_mean` carries the argument in its docstring: AUC reads order only, so a probability mean
lets the widest-spread member dominate. **Measured, that is wrong here.**

| | val | gold |
|---|---|---|
| prob-mean beats rank-mean | **11 / 11** combinations | 6 / 11 |
| mean difference | **+0.0016** | +0.0002 |
| paired, c1+c2+c3 | **+0.0018 [+0.0009, +0.0027]** | — |

The theory is not wrong, its precondition is absent. Rank-averaging protects against members on
*incomparable scales*; these arms are sigmoid outputs of similarly-trained models on one target, so
they are already comparable, and ranking throws away agreement strength — how confidently the members
concur — which is real signal. The pathology it guards against never arises, so only its information
loss shows up.

`rank_mean` stays in the codebase with the measurement recorded in its docstring: it is the right
tool for blending members that genuinely differ in calibration (a public model's outputs against
ours), and the wrong default for a homogeneous fold/config ensemble. Plan Step 6 revised.

This is the second time this phase that a theoretically-motivated choice failed a cheap empirical
check — the other being the rank blend as a *training* target. Both were caught by looking at the
thing itself rather than at the argument for it.

### c5 and the ensemble picture; Step 7 begins (2026-09-07)

**c5 — DINOv2 on the full stack — does not beat the efficientnet stack.**

| arm | val (n=863) | gold (n=58) | min |
|---|---|---|---|
| c2 — 6 slots + attention, efficientnet_b0 | **0.8607** | **0.8759** | 51.0 |
| c5 — 6 slots + attention, DINOv2 | 0.8549 | 0.8724 | 60.1 |

c5 - c2: val -0.0059 [-0.0130, +0.0018], gold -0.0035 [-0.0285, +0.0179]. No separation either way,
but nothing to promote. DINOv2 is underfit again — final loss 0.4441 against c2's 0.3869 — the same
pattern as c3, and the honest reading is that eight epochs with a trunk at 8e-6 does not finish
training this backbone rather than that the backbone is worse.

Its attention is markedly **sharper** than c2's while staying anatomically correct: Medial Meniscus ->
COR_FLUID **0.699** (c2: 0.426), MCL -> COR_FLUID 0.547, ACL -> SAG_FLUID 0.540, Effusion -> AX_FLUID
0.555, Fracture -> COR_FLUID 0.508. The self-supervised features separate the planes better; that is
simply not converting into macro AUC at this schedule.

**The phase's gain is in the ensemble, not in any architecture** (probability mean, fold-0 models):

| members | val | gold |
|---|---|---|
| c2 alone, best single | 0.8607 | 0.8759 |
| c2+c3 | 0.8715 | 0.8885 |
| **c1+c2+c3+c5** | **0.8767** | 0.8897 |
| c0+c2+c3+c5 | 0.8752 | **0.8962** |

Best ensemble against the best single member: **+0.0159 [+0.0114, +0.0209]**. The top five
combinations sit inside 0.0025 of one another, which is not a separable difference, so the member set
should be chosen on cost rather than by taking the maximum — picking the argmax of an unresolvable
ranking is how a screen overfits its own validation set.

**Step 7 structure.** One kernel per config over five folds, rather than one kernel for the ensemble:
c2 is ~4.3h, c3 ~1.9h, c5 ~5h, and each fits a session on its own while a combined run would not.
Members are ensembled afterwards, outside the kernel, which also means adding or dropping one later
costs nothing. `phase6-confirm` reuses the screen notebook's first three cells verbatim (setup,
corpus gate, folds and labels) rather than carrying a second copy that can drift.

Verified before pushing: every one of the 4,349 labelled studies lands in exactly one validation fold,
so the kernel's `assert not np.isnan(oof_pred).any()` will pass for the right reason rather than
failing four hours in.

### phase6-submit built and dry-run before the confirm finished (2026-09-07)

The submission path is the one place where a mistake costs the whole competition rather than one
experiment, so it was built and exercised locally while the confirm kernel was still training, using
five stand-in checkpoints from an untrained model. That tests the plumbing, which is what actually
fails on a hidden test set; accuracy is the confirm kernel's job.

Verified on the local sample: `prep_slots` runs on **test** DICOMs (1.254 s/study, 4 of 6 slots
filled on the one study that ships pixels), the artifact round-trips through `save_study_npz` into
`PreppedSlotDataset`, the **presence mask reaches the model**, five checkpoints load and combine by
probability mean, `build_submission`'s 0.5 fallback fires for the 2 of 3 studies with no DICOMs
locally, and the header comes out byte-for-byte including the apostrophe in `Baker's`.

Three deliberate differences from `phase5-submit`, each a consequence of the artifact change rather
than a preference:

- **Prep at 336 and read at 224, not prep at 224.** Prepping directly at the read size would be
  faster and wrong — training saw 336 stored and then resized, and that resampling chain is part of
  what the model learned.
- **The mask is passed to the model.** Dropping it would make every absent slot read as real black
  anatomy, reintroducing precisely the bias masked pooling exists to remove, and the only symptom
  would be a slightly worse score.
- **Probability mean, not rank mean** — following the 11-of-11 measurement rather than the plan's
  original instruction.

The src guard here checks `KneeModel`'s `head` argument and the six-slot table, because a stale src
in *this* kernel does not crash: it silently prepares different pixels from the ones the checkpoints
were trained on.

### Pre-registered band for the Phase 6 submission, written before the confirm finished (2026-09-07)

Registered now, while the confirm kernel is still on fold 3 and the number is unknown. The Phase 5
entry's point-estimate extrapolation was retired for over-predicting (0.857 predicted, 0.835 actual),
so this is a band with a stated reasoning and stated failure conditions, not a forecast.

**What is known.** Two gold-to-LB pairs exist, and their offsets disagree: run 1 gold 0.7252 -> LB
0.763 (**+0.038**), Phase 5 gold 0.8189 -> LB 0.835 (**+0.016**). Two points rule out a constant
offset without establishing what replaces it. Separately, c2's *single fold-0 model* scored gold
0.8759, and a five-fold mean should sit above a single fold — Phase 5's five-fold gold was 0.8189
against a single-fold configuration screened at 0.8484 val, but the single-fold gold for that config
was never recorded, so the size of the fold-ensemble gain on gold is genuinely unknown here.

**The band, for the c2-only five-fold submission:**

| LB | reading |
|---|---|
| **below 0.83** | something is broken. Phase 5 scored 0.835 from a *worse* gold number, so a drop means the submission path, not the model — suspect prep v2 on test studies or the mask |
| 0.83 - 0.85 | the gold gain did not transfer. Would say the screens have been optimising against report-derived labels rather than the task |
| **0.85 - 0.89** | **expected.** The gold improvement (0.8189 -> ~0.88) transfers at something like the observed offsets |
| 0.89 - 0.91 | better than expected; the fold ensemble is worth more on the hidden set than on 58 gold studies |
| above 0.91 | implausible from this configuration — check for a leak before celebrating |

**What it is not.** Even the top of that band leaves us short of the 0.93 public-fork shelf and far
from the 0.950 needed for tenth. A good result here means Phase 6's structural work transferred, not
that the standing problem is solved; the ensemble submission and whatever follows still have roughly
0.06 to find.

**One thing this submission specifically tests**, beyond the score: prep v2 has never run against the
hidden test set. The fold-0 screens all read artifacts built from *training* DICOMs. If the 130mm
crop, the slot table or the geometry laterality route behave differently on test studies, the
fallback count and the slot-fill histogram printed by the notebook will say so before the score does.

### c2 confirmed on 5 folds — and fold 0 was not representative (2026-09-07)

`knee-phase6-confirm` v1, 4.02 h, 5 folds of `primary_v3`, target steven_v4, gold excluded from every
split.

| fold | val macro | gold |
|---|---|---|
| 0 | 0.8571 | 0.8699 |
| 1 | 0.8446 | 0.8634 |
| 2 | **0.8332** | 0.8616 |
| 3 | **0.8652** | 0.8597 |
| 4 | 0.8449 | 0.8576 |
| **pooled OOF (n=4,349)** | **0.8485** | — |
| **5-fold mean ensemble** | — | **0.8794** |

**Gold transfer 0.8189 -> 0.8794, +0.0605**, and that *is* a comparable number: the same 58 studies,
the same rubric labels, both a five-fold ensemble. It is the tier that resembles the hidden test set.

**Pooled OOF 0.8485 is not comparable to Phase 5's 0.8523** and must not be read as a regression —
different folds (v3 vs v2), a different target (steven_v4 vs our Qwen3-4B) and different pixels. The
notebook prints that caveat next to the number for the same reason.

**The finding that matters for how the rest of this phase is run: fold 0 is an easy fold.** Its val
macro is 0.8571 against a pooled 0.8485, and the folds span **0.8332 to 0.8652 — a range of 0.032**,
five times the ~0.0065 run-to-run noise measured on repeated fold-0 runs. Every Phase 6 screen was
fold 0 only, so **every screen number in this phase is optimistic by roughly 0.01 in absolute terms.**
Paired comparisons within fold 0 are unaffected — both arms saw the same easy studies — so the
promotion decisions stand. But two consequences follow: absolute screen numbers should never be quoted
as if they were corpus performance, and a single-fold screen cannot resolve anything smaller than the
fold effect, which makes the earlier "+0.0027 for c2 over c1 is below the noise floor" reading more
firmly correct, not less.

**The fold ensemble is worth +0.0095 on gold** over the best single fold (0.8699 -> 0.8794), which is
consistent with the +0.0159 the config ensemble bought on fold-0 val, and confirms that averaging is
doing real work rather than smoothing noise.

Gold per label at n=58 is wildly dispersed — Baker's 0.9909 and MCL 0.9569 against Synovitis 0.7479
and Lateral Meniscus 0.7665 — and with 9 to 35 positives per label these are not to be read as a
per-label profile. The macro is the number.

### Phase 6 scored: LB 0.905, and the shelf is the thing to aim at (2026-09-07)

Submission `56076866`, the c2-only five-fold ensemble, scored **0.905** — up from 0.835, and inside
the pre-registered band's "0.89-0.91, better than expected" row. **The band worked.** It was written
before the confirm finished, it named its failure conditions, and the outcome landed where the
reasoning put it rather than where a point estimate would have.

**The gold->LB offset now has three points and is tightening:**

| run | gold transfer | LB | offset |
|---|---|---|---|
| Phase 4 run 1 | 0.7252 | 0.763 | +0.038 |
| Phase 5 | 0.8189 | 0.835 | +0.016 |
| **Phase 6** | **0.8794** | **0.905** | **+0.026** |

Three points do not establish a law, but they bracket a range: the next band can be narrower than
this one, and gold transfer remains the tier worth protecting.

**The leaderboard, pulled today (3,275 teams), and this is the number that should drive the next
phase:**

| score | rank | teams at or above |
|---|---|---|
| 0.954 | 1 | 1 |
| 0.950 | 8 | 11 |
| **0.940** | **104** | **130** |
| **0.930** | **1042** | **1052** |
| 0.920 | 1294 | 1351 |
| **0.905 (us)** | **1602** | 1607 |
| 0.902 | median | — |

**~940 teams are packed into 0.930-0.940.** In that band **+0.01 AUC is worth roughly 940 places** —
the steepest gradient anywhere on this leaderboard. Crossing 0.93 moves us ~560 places; reaching 0.94
moves us ~1,500.

Three things follow, and they are why this is written into the plan document and not only here:

1. **The phase target is the shelf, not the podium.** 0.930-0.940 is +0.025 to +0.035 from here.
   Top-10 needs 0.950 and is a later question.
2. **We are 0.003 above the median.** 0.905 is not a good standing; it is an average one that took
   four submissions to reach.
3. **The shelf is ~940 teams running `pilkwang/rsna-knee-baseline-v1`'s fitted weights**, not 940
   independent reimplementations. It is not evidence that the recipe-as-trained reaches 0.93 — which
   matters, because Phase 6 implemented most of that recipe and landed at 0.905.

**What Phase 6 left unspent**, each measured or documented rather than guessed:

| unspent | evidence | cost |
|---|---|---|
| config ensemble over 5 folds | +0.0159 [+0.0114, +0.0209] on fold-0 val; the phase's largest CI-excluding-zero result, never confirmed or submitted | ~2 h GPU |
| DINOv2 trained to convergence | c5 final loss 0.4441 vs c2's 0.3869; c3 still falling steeply at epoch 8; pilkwang runs 10 epochs at these same LRs with DINOv2 as their production backbone | ~47 min |
| reading at 336 | prep v2 stores 336px (0.387 mm/px) *for the Nyquist argument*, and every screen then downsampled to 224 (0.580). c4 was dropped on a runtime estimate later found wrong by 12x | ~52 min |
| confidence sample weights | `weight_*` computed and unused since Phase 1; a recipe item | ~51 min |
| `SLOTS_RECOVERED` | pilkwang's default is `recovered` and `SLOTS_PUBLIC` is their fallback — **Phase 6 shipped the fallback** | free CPU census, then re-prep only if it pays |

**A record-keeping failure worth fixing before the next screen appends a row.** `results/experiments.csv`
in the repo has **10 rows** — Phase 1 and Phase 4. Every Phase 5 and Phase 6 screen row was written
inside a Kaggle kernel and never pulled down, and Kaggle keeps only the latest version's output, so
the earlier arms' rows and prediction arrays are gone. The numbers survive in NOTES prose and nowhere
queryable. Two consequences already bit: c3's batch size is not recoverable from any local record
(so Phase 7's s1 re-runs its own control rather than pairing against a config it cannot reproduce),
and the schema has no `gold_macro` column even though every screen this project has run reports gold
as its second tier and promotion turns on it.

### Phase 7 Step 1: confidence weights built, and one check ran the wrong way (2026-09-07)

`confidence_from_agreement` in `reports.py`, 7 tests. Per study, per label, the weight falls with
the **spread across independent label sources** — agreement, not decisiveness, so a study every
source calls a confident negative keeps full weight.

**Why a weight at all, when the target is already soft.** `steven_v4` carries 130 distinct values,
so per-label uncertainty is expressed. But soft BCE cannot *discount* a row: a study the sources
split on carries an irreducible loss floor and actively teaches the model to answer 0.5 there. The
sample weight is the only place that doubt can be spent. It also uses the multi-source information
without making a blend the training target, which was rejected on 2026-09-07 for a different reason.

**Four independent labellings vote**, not five files: ours (Qwen3-4B), `steven_v4`, `pilkwang_v2`,
`lixin73`. Steven ships full/v2/v4_blend but those are one author's successive versions and would
trip-count that author's reading; `yunusgmsoy` stays excluded for carrying gold values on the 58
held-out rows.

**The weights are not a no-op** — per-label std 0.27 to 0.56 around a mean of 1:

| label | rows weighted < 0.5 | label | rows weighted < 0.5 |
|---|---|---|---|
| **PF OA** | **35.3%** | Baker's | 9.1% |
| **Effusion** | **30.4%** | Lateral Meniscus | 6.1% |
| Medial OA | 23.9% | Contusion | 5.1% |
| Lateral OA | 18.9% | Fracture | 4.1% |
| MCL | 12.1% | **Synovitis** | **1.2%** |
| ACL | 10.6% | Medial Meniscus | 9.5% |

**The check that ran the wrong way, recorded because it nearly went unmade.** If source
disagreement tracked label *quality*, the labels sources fight over should be the ones that score
worst against gold. Spearman over the 12 labels between disagreement share and `steven_v4`'s gold
per-label AUC is **+0.40 (n=12, p=0.20)** — the wrong sign, though not separable from zero at this
n. Synovitis is the extreme case: the four sources agree on it almost perfectly (1.2% disputed) and
it is the *worst-labelled* of the twelve (gold 0.790). **Agreement is not accuracy** — four LLMs
reading the same reports share the same systematic blind spots, and unanimity measures the shared
prior as readily as the truth.

**This does not sink the scheme, and the reason is the normalisation.** Weights are normalised to
mean 1 **per label**, which was done so a down-weighted run would not also be a lower-effective-
learning-rate run — and which, as a side effect, removes the entire between-label axis the
correlation above lives on. The hypothesis under test is strictly *within* label: given two studies
whose reports the sources read differently, is the disputed one worth less as a training row. That
is untouched by the per-label finding. But the prior for s3 should be weaker than it was before
this check, and if s3 reads null this is the reason to reach for first.

`results/label_confidence.csv` (4,407 x 12). The screen computes it in-kernel from the mounted label
datasets rather than reading this file, so the derivation is reproducible where it is used.

### The census: `_STRUCT` mixes three contrasts, and the CSV axis is not the one I assumed (2026-09-07)

`knee-phase7-census`, CPU, 24,371 series walked in **2.5 min**. TR/TE/`ScanningSequence` recovered
per series; 1,206 (4.9%) lack TR or TE and are counted as unknown rather than guessed.

| slot | n | T1 | PD | T2 | GRE | unknown | dominant |
|---|---|---|---|---|---|---|---|
| SAG_FLUID | 4,667 | 0 | 2,837 | 1,356 | 0 | 474 | PD 67.7% |
| COR_FLUID | 4,624 | 0 | 3,235 | 1,148 | 1 | 240 | PD 73.8% |
| AX_FLUID | 4,719 | 2 | 3,039 | 1,165 | 269 | 244 | PD 67.9% |
| **SAG_STRUCT** | 5,197 | **1,645** | **1,702** | **1,224** | 388 | 238 | **PD 34.3%** |
| **COR_STRUCT** | 3,985 | 2,854 | 182 | 948 | 0 | 1 | T1 71.6% |
| **AX_STRUCT** | 1,179 | 570 | 11 | 482 | 107 | 9 | **T1 48.7%** |

**VERDICT against the rule fixed before the run (85% pure): the re-prep earns its day.** Every
`_STRUCT` slot fails, and SAG_STRUCT fails badly — 1,645 T1, 1,702 PD, 1,224 T2 and 388 GRE, four
contrasts in near-equal parts arriving at **one attention position**. c2's attention head learned to
route each diagnosis to a plane; on the `_STRUCT` slots there is no consistent tissue contrast at that
position to route *to*, and the head cannot tell a T1 from a T2 because nothing in the input says
which it got.

**The control failed, and the failure is the more useful half of this run.** The `_FLUID` slots were
predicted to come out overwhelmingly **T2**; they came out overwhelmingly **PD** (68-74%), with zero
T1 in the sagittal and coronal ones. That is not a bug — **fat-suppressed proton-density is the
workhorse knee sequence** — and it means the prediction was wrong about what the CSV column encodes.

**So the two axes are genuinely orthogonal, and we have been using one of them:**

- `Fluid_Sensitive` / `Fat_Suppression` (identical columns) encode **fat suppression**, not weighting.
- TR/TE encode **weighting** (T1 / PD / T2), which `sequence_weighting` recovers and no delivered
  column carries.

The zero-T1 count in SAG_FLUID/COR_FLUID confirms it: fat-suppressed T1 is rare in knee protocols, so
the fat-suppression axis and the weighting axis separate cleanly and independently. Phase 6's slot
table crosses plane × fat-suppression and leaves weighting entirely unexpressed. **That is a stronger
case for the re-prep than the one the plan was written on** — it is not "the CSV cannot separate T1
from non-suppressed PD/T2", it is "an entire acquisition axis is invisible to the model", and the
sagittal slot is where it costs most.

Also worth carrying: **765 gradient-echo series** exist. GRE contrast follows neither the T1 nor the
T2 rule, so under any TR/TE-only scheme they need their own bucket rather than a silent assignment —
which is why `sequence_weighting` reports GRE separately rather than folding it in.

`results/weighting_census.csv` and `results/slot_weighting_table.csv` are the kernel's outputs.

### The v3 slot table, designed from the census rather than copied (2026-09-07)

The census says the delivered CSV encodes **fat suppression** and TR/TE encode **weighting**, and that
those are two orthogonal axes of which Phase 6 used one. The full cross over 4,407 studies:

| combination | study fill | | combination | study fill |
|---|---|---|---|---|
| COR_FS_PD | 71.1% | | SAG_FS_T2 | 29.5% |
| AXI_FS_PD | 65.6% | | SAG_NOFS_T2 | 27.3% |
| COR_NOFS_T1 | 63.6% | | COR_FS_T2 | 25.2% |
| SAG_FS_PD | 63.0% | | AXI_FS_T2 | 25.1% |
| SAG_NOFS_PD | 36.3% | | COR_NOFS_T2 | 16.4% |
| SAG_NOFS_T1 | 35.9% | | AXI_NOFS_T1 / AXI_NOFS_T2 | 12.4% / 10.8% |

**The defect that a naive recovered scheme would have shipped, caught before prepping anything.**
1,206 series (4.9%) carry no recoverable TR/TE, and **238 studies (5.4%) consist *entirely* of such
series** — every one of them a complete 5-to-7-series study covering all three planes, so they are
not broken studies but one export batch with stripped headers. Under any weighting-only slot table
those 238 studies get **zero slots**: dropped from training and answered with the 0.5 fallback at
inference. The threshold makes no difference — at a 0% cut they are still lost, because the problem
is the unknown bucket, not the rarity of a combination.

**So the table keeps an explicit fallback tier.** The rule: every combination at **>=10% study fill**,
plus **every `_UNK` combination regardless of fill**, because an unrecoverable weighting is a real
category and not a leftover. 19 slots:

| | v2 (shipped, LB 0.905) | v3 (proposed) |
|---|---|---|
| slots | 6 (plane x fat-sup) | **19** (plane x fat-sup x {T1, PD, T2, UNK}) |
| mean filled per study | 4.84 | **5.04** |
| studies with zero slots | 0 | **0** |
| series represented | — | **23,411 / 24,371 (96.1%)** |

**Storage is roughly unchanged**, which is the part that makes this affordable: the same series are
sorted into finer buckets rather than more series being stored. Mean fill goes *up* slightly because
a study with two differently-weighted sagittal series now keeps both instead of one winning the slot.

Two things this does not settle, both for the screen rather than the plan:

- **19 attention positions is not obviously better than 6.** Several slots fill below 30%, and
  AX_STRUCT at 19.4% was already the one Phase 6 flagged as mostly-masked. More positions means fewer
  studies teaching each one its routing. This gets screened against the 6-slot table, not assumed.
- **GRE (765 series) is excluded** by the 10% rule and lands nowhere. It is a fifth contrast whose
  TR/TE follow neither the T1 nor the T2 rule; folding it into either would be the silent-fallback
  mistake. Losing it costs at most the studies where it is the *only* series in a plane — worth
  counting before the prep runs, not after.

### A2: what the 10% rule drops, counted before prepping on it (2026-09-07)

The v3 slot rule discards **960 series (3.9%)**: 764 GRE across three planes, 182 COR_NOFS_PD, and
14 stragglers. The question that matters is not how many series that is but whether it costs a study
a whole acquisition plane — the 238-study `_UNK` finding is the precedent for checking.

**19 studies (0.4%) lose a plane** — 9 axial, 7 coronal, 3 sagittal. Two orders of magnitude smaller
than the `_UNK` problem, and none of them lose *all* their slots. **So GRE does not get its own tier.**

That is a decision about the *first* v3 table, not a permanent one. SAG_NOFS_GRE fills 8.7% of
studies and AXI_FS_GRE 6.0%, so a GRE tier is three more sparse attention positions — worth adding
only if the 19-vs-6 slot screen (A5) shows that more positions help at all. Adding them first would
be expanding the table on the assumption the screen exists to test.

### Prep v3 built and launched: the recovered table, verified against the census before it ran (2026-09-07)

`sequence_weighting`, `SLOTS_V3`, `select_slots_v3`, `study_weightings` and
`prep_slots(slot_scheme=...)` implemented TDD-first. **238 tests pass** (191 at the phase start).
Four `phase7-prep-shard{0..3}` kernels pushed to CPU (free quota) while the s1 GPU screen runs — the
two tracks do not compete for the same budget, which is the whole reason the phase is not serial.

**`select_slots_v3` reproduces the census prediction exactly** over all 4,407 studies: mean **5.04**
filled slots per study, **0** studies with no slot, every per-slot fill matching to the study. That
check is worth more than the unit tests: it runs the real function over the real corpus and compares
against a number derived independently, before 9.7 GB of artifacts get built on it.

**Sized before launching**, per the Phase 6 rule: 456 KB per filled slot on a real study, so
**9.7 GB total / 2.4 GB per shard** — indistinguishable from v2's 9.5 GB and 2.4 GB. Nineteen slots
cost nothing extra because the same series are sorted into finer buckets rather than more series
being stored.

**What the split actually does, on the first real study:**

| | v2 | v3 |
|---|---|---|
| slots filled | COR_FLUID, AX_FLUID, **SAG_STRUCT**, AX_STRUCT | COR_FS_T2, AXI_NOFS_T2, **SAG_NOFS_PD**, **SAG_NOFS_T2** |

That study has **two** sagittal non-fat-suppressed series, a PD and a T2. v2 has one `SAG_STRUCT`
slot, so it kept whichever won the UID tie-break and **discarded the other series entirely**. v3
keeps both, in slots that say which is which. That is the defect in one line, and it is why mean fill
goes *up* rather than staying flat.

**Three decisions recorded next to the code rather than left implicit:**

1. **GRE lands nowhere, not in `_UNK`.** `_UNK` means "the header could not answer". GRE is a class we
   positively identified and chose not to model; filing it under unknown would put a known class in
   the residual bucket and rebuild the conflation v3 exists to remove. Measured cost: 764 series,
   19 studies (0.4%) losing a plane, none losing all their slots.
2. **v2 records `weighting: None`, not `"unknown"`.** v2 never asks; "unknown" means it asked and the
   header could not answer. Writing "unknown" for both would make them indistinguishable in any later
   audit — and being unable to audit the slot table after the fact is precisely how Phase 6 shipped
   this defect.
3. **v2 stays the default and stays in the codebase** until v3's re-baseline lands, exactly as prep v1
   outlived prep v2's arrival through Phase 6 Step 3. v2 is the artifact behind the scored 0.905.

**One regression against v2, stated plainly:** v2 needed no header parsing, so it could not be
surprised by the hidden test set. v3 depends on TR/TE. The `_UNK` tier is the mitigation — an
unreadable test series lands in a slot the model has actually been trained on, rather than in a guess
or nowhere — and `study_weightings` derives the axis at prep time rather than looking it up in
`weighting_census.csv`, because a lookup table would work in every screen and then fail at submission.

### s1: DINOv2 was not underfit — it was memorising. The hypothesis is dead (2026-09-07)

`knee-phase7-screen` v1, two arms in one session, c3's exact shape (2 slots x 5 groups, 224px, mean
pool, DINOv2-small, AdamW 8e-6 / 1e-3), schedule the only variable.

| arm | val (n=863) | gold (n=58) | min |
|---|---|---|---|
| **8 epochs** | **0.8548** | **0.8683** | 20.7 |
| 16 epochs | 0.8473 | 0.8292 | 41.4 |

| comparison | delta | |
|---|---|---|
| **val (n=863), 16ep minus 8ep** | **-0.0075 [-0.0156, -0.0006]** | **CI excludes zero** |
| **gold (n=58), 16ep minus 8ep** | **-0.0391 [-0.0701, -0.0045]** | **CI excludes zero** |

**Training longer makes it measurably worse on both tiers, both CIs excluding zero.** These are the
phase's first CI-excluding-zero results and they point the opposite way to the hypothesis that
motivated the run.

**The training loss is the mechanism, and it settles the argument.** At 16 epochs DINOv2 reached a
final training loss of **0.3476** -- *below* c2's efficientnet_b0 at 0.3869 -- while both validation
tiers fell. It is not that the backbone cannot fit this data; it fits it better than the incumbent
and generalises worse. The Phase 6 inference from "loss 0.4459, still falling" was measuring
capacity-to-memorise and reading it as unfinished learning.

**The Phase 6 reading was wrong, and worth stating plainly.** The c3 and c5 entries both argued "it is
underfit -- final training loss 0.4459 against c0's 0.3966, still falling steeply at epoch 8 -- so the
honest reading is that eight epochs with a trunk at 8e-6 does not finish training this backbone rather
than that the backbone is worse." **That is falsified.** A training loss still falling at epoch 8 under
a *fully annealed* OneCycle was memorisation, not unfinished learning; doubling the schedule bought
more of it. DINOv2 at this shape is not better than efficientnet_b0, and the reason is not the
schedule.

**The control reproduced c3 almost exactly**, which is why running it was worth 21 minutes:

| | c3 (Phase 6, batch unknown) | s1 8ep control |
|---|---|---|
| val | 0.8512 | 0.8548 (+0.0036, inside the ~0.0065 noise floor) |
| gold | 0.8686 | 0.8683 (**-0.0003**) |

So the configuration matches, the comparison is clean, and the noise floor is re-confirmed at roughly
where Phase 6 measured it. Had the 16-epoch arm been paired against the recorded row instead, the same
conclusion would have rested on an unverifiable batch size.

**What this settles, and what it does not.**

- **Branch B applies: c3's 8-epoch configuration is the one to confirm.** No re-screen is needed
  before its 5-fold confirm, and the confirm is no longer at risk of banking an undertrained member.
- **The DINOv2 ensemble members stand as they are.** c3 and c5 remain valuable for complementarity
  (c3 vs c0 per label: MCL +0.039, Contusion +0.036, Fracture +0.020 against ACL -0.032, PF OA -0.022),
  and the Phase 6 ensemble measurement is untouched by this. What is gone is the *upside* case that a
  longer schedule would make DINOv2 a better single model.
- **s4's prior weakens too.** If DINOv2 overfits between 8 and 16, efficientnet at 12 is a smaller
  version of the same test, and Phase 5's screen A already found that curve turning over between 8
  and 12. s4 is now the least promising of the remaining v2 screens and should go last.
- **s2 (336px) is unaffected** and now runs on efficientnet_b0 by default, since s1 leaves DINOv2 no
  better than it was. That reintroduces the pretraining-scale confound noted when s2 was ordered --
  read a null there as "336 did not pay *for this backbone*", not as a verdict on resolution.

### Prep v3 ran the corpus: every slot is pure, and the census predicted it exactly (2026-09-07)

All four `phase7-prep-shard` kernels complete. **4,407 studies, 0 failed, 22,208 stored series**,
every one at **0.38690476 mm/px** — the physical-scale property v2 established, preserved.

**The prediction held to two decimals.** Mean filled slots per study was predicted at **5.04** from
`weighting_census.csv` before any DICOM was prepped; the real prep on real DICOMs gives **5.04**
(shard 1: 5.05). **0 studies with no slot**, which is the `_UNK` tier doing exactly the job it was
added for.

**The defect is fixed, and this is the check that proves it** — slot x recovered weighting over all
22,208 stored series:

| | PD | T1 | T2 | unknown |
|---|---|---|---|---|
| every `*_PD` slot | its own count | 0 | 0 | 0 |
| every `*_T1` slot | 0 | its own count | 0 | 0 |
| every `*_T2` slot | 0 | 0 | its own count | 0 |
| every `*_UNK` slot | 0 | 0 | 0 | its own count |

**A perfect diagonal: no slot holds more than one weighting class.** Against v2, where `SAG_STRUCT`
alone held 1,645 T1, 1,702 PD, 1,224 T2 and 388 GRE. Corpus weighting mix of what got stored:
PD 10,403 / T2 5,919 / T1 4,926 / unknown 960.

**Storage came in where it was sized:** 2.17 MB/study mean, **2.39 GB/shard**, against the 2.4 GB
predicted from one study and v2's actual 2.4 GB. Nineteen slots cost nothing over six.

**Two costs to record, neither fatal:**

1. **Prep is now 3.0 s/study, up from v2's ~1.6.** That is `study_weightings` reading one header per
   series for *every* series in the study rather than only the selected ones. 55 min per shard on
   free CPU, run in parallel with a GPU screen — the price is wall-clock nobody was waiting on.
2. **The table is sparse per study: ~5 filled of 19 (26%), against v2's 4.84 of 6 (81%).** This is
   the risk flagged when the table was designed, now measured. The distribution is
   `{2:1, 3:7, 4:364, 5:551, 6:47, 7:57, 8:37, 9:37, 10:1}` for shard 0 — most studies fill 4 or 5.
   **Fourteen attention positions are empty for the median study**, so each one is taught by a
   minority of the corpus. Whether that expressiveness beats v2's density is exactly what the
   19-vs-6 screen exists to answer, and the sparsity is the reason it could plausibly lose.

**One gate is not verifiable from the logs: the visual grid.** It renders as matplotlib images in the
notebook, not stdout, so "correct plane, weighting and handedness" still needs eyes on
`knee-phase7-prep-shard0` on Kaggle. Everything countable passed; the pixels themselves have not been
looked at, and Phase 6's rectangle-squash and sagittal-mirror bugs were both found this way rather
than by any assertion.

### The two-config submit kernel, built and dry-run before the confirm finished (2026-09-07)

Same discipline as `phase6-submit`: the submission path is the one place a mistake costs the whole
competition rather than one experiment, so it was built and exercised locally against `data/sample/`
with **ten stand-in checkpoints from untrained models** while `phase7-confirm-c3` was still training.
That tests the plumbing, which is what actually fails on a hidden test set; accuracy is the confirm's
job.

**What the dry run showed, and the middle two rows are the whole point:**

| check | result |
|---|---|
| both members load 5 folds each | 10 models total |
| **c2 reads** | `image(6, 5, 3, 224, 224)`, `mask(6,)`, 4 slots present |
| **c3 reads** | `image(2, 5, 3, 224, 224)`, `mask(2,)`, 1 slot present |
| presence mask reaches both models | yes, in both shapes |
| `build_submission` 0.5 fallback | fired for 2 of 3 (the studies with no local DICOMs) |
| header byte-exact incl. the apostrophe in `Baker's` | yes |
| every probability in [0, 1] | yes |

**Two loader shapes, two architectures, one prep pass.** The artifact is written once per study and
each member reads it with its own slot subset and read size. Sharing one tensor would hand c3 six
slots it never trained on -- which would not crash, and would only appear as a lower score.

**Three guards this kernel needs that `phase6-submit` did not:**

1. `assert len(SLOTS) == 6` now means something new. `SLOTS_V3` exists in src, and prepping test
   studies under the nineteen-slot table would file every series differently from what these
   checkpoints trained on. Silent, and worth exactly the whole submission.
2. `PREP_SCHEME = 'v2'` is passed explicitly rather than defaulted, so the prep version is a stated
   fact in the kernel rather than an inherited default that a later change to `prep_slots` could flip.
3. `backbone_kwargs` is asserted present, because the DINOv2 member needs `img_size` passed through
   and a src predating that would fail at the first forward, an hour in.

**What this submission does and does not test.** The single variable against 0.905 is a **second
config** -- 0.905 was already a five-fold average, so this is not a test of "ensembling". Phase 6
measured c2+c3 at +0.0108 over c2 on fold-0 val, but over *single-fold* members, while the confirm
separately showed fold-averaging buying +0.0095 on gold. The two effects may substantially overlap,
so the band's lower edge has to be wider than the fold-0 delta suggests.

### c3 confirmed, and the config ensemble is fully additive to fold averaging (2026-09-08)

`knee-phase7-confirm-c3`, 1.75 h, 5 folds of `primary_v3`, target `steven_v4`, gold excluded from
every split.

| fold | val macro | gold |
|---|---|---|
| 0 | 0.8552 | 0.8600 |
| 1 | 0.8432 | 0.8723 |
| 2 | 0.8449 | 0.8677 |
| 3 | 0.8544 | 0.8612 |
| 4 | 0.8498 | 0.8740 |
| **pooled OOF (n=4,349)** | **0.8491** | — |
| **5-fold mean** | — | **0.8805** |

**c3 matches c2 on both tiers at 43% of the training cost.**

| | pooled OOF | gold | GPU |
|---|---|---|---|
| c2 — 6 slots, attention, efficientnet_b0 | 0.8485 | 0.8794 | 4.02 h |
| **c3 — 2 slots, mean pool, DINOv2-small** | **0.8491** | **0.8805** | **1.75 h** |

Two slots and a mean pool, on a backbone that lost every single-model screen it entered, equal the
six-slot attention stack. Worth remembering the next time a screen's argument is about mechanism.
Also note c3's fold spread (0.8432-0.8552, range 0.012) is a *third* of c2's (0.8332-0.8652, 0.032)
on the identical folds -- the same data is much easier for one architecture to fit consistently.

**The ensemble, and this is the number Submission A rests on:**

| tier | c2 | c3 | **c2+c3** | ensemble vs c2 |
|---|---|---|---|---|
| **OOF (n=4,349)** | 0.8485 | 0.8491 | **0.8641** | **+0.0156 [+0.0136, +0.0175]** |
| gold (n=58) | 0.8794 | 0.8805 | **0.8919** | +0.0125 [+0.0006, +0.0250] |

Both CIs exclude zero on both tiers against c2. Against c3 the OOF delta also excludes zero
(+0.0150 [+0.0129, +0.0170]); the gold one spans it (+0.0113 [-0.0023, +0.0247]), which is n=58
failing to resolve a difference the 4,349 sees clearly, not a disagreement.

**Correction to the framing written when the submit kernel was built.** That entry said the fold-0
+0.0108 was measured over *single-fold* members while fold-averaging separately bought +0.0095 on
gold, so "the two effects may substantially overlap" and the band's lower edge should widen.
**Measured, they do not overlap at all.** The fold-0 screen saw +0.0159 for the config ensemble over
single-fold members; here the config ensemble buys **+0.0156 over five-fold members**. Config
diversity is fully additive to fold averaging, and the caution was unnecessary. The band can be
written on the +0.0156 rather than discounted from it.

**Why that is the more interesting half:** averaging five folds of one config and averaging two
different configs are removing *different* errors. Fold averaging cancels what varies with the
training split; config averaging cancels what varies with architecture and input. Nothing in the
first exhausts the second, which is also the argument for eventually adding c5 rather than more
folds of c2.

### Pre-registered band for Submission A, written before it is pushed (2026-09-08)

Registered now, with the kernel built and dry-run but **not** submitted -- next week's GPU. Written
against measurements, with the failure conditions stated, in the form that worked for Phase 6.

**What is known.** Three gold->LB pairs: 0.7252 -> 0.763 (+0.038), 0.8189 -> 0.835 (+0.016),
0.8794 -> 0.905 (**+0.026**). The submitted artifact would be the c2+c3 probability mean, whose gold
is **0.8919** (+0.0125 over the c2-only submission that scored 0.905) and whose pooled OOF delta is
**+0.0156 [+0.0136, +0.0175]**. Two tiers agreeing on a gain of roughly +0.013 to +0.016.

| LB | reading |
|---|---|
| **below 0.905** | **something is broken.** A second config cannot score below a strict subset of itself unless the submission path is wrong -- suspect the c3 loader shape (2 slots vs 6) or a checkpoint/architecture mismatch, not the model |
| 0.905 - 0.912 | the gain partly transferred. Would say the two members are more correlated on the hidden set than on our folds |
| **0.913 - 0.925** | **expected.** +0.0125 gold / +0.0156 OOF transferring at something like the observed offsets |
| 0.925 - 0.935 | better than expected; config diversity is worth more on the hidden set than on 4,349 studies |
| above 0.935 | implausible from two members. Check for a leak before celebrating |

**What it specifically tests, beyond the score:** the c3 checkpoints have never run against test
DICOMs, and c3 is the first member whose loader reads a *subset* of the slots. The kernel prints
slot-fill and fallback counts, which will say whether the 2-slot read behaves on test studies before
the score does.

**What it is not.** Even the top of that band leaves us below the 0.930 shelf where ~940 teams sit,
and the shelf is the phase's target. Submission A banks a measured, already-paid-for gain; the
structural work (prep v3's nineteen slots) is what is supposed to cross the line, and it has not been
screened yet.

### The field moved again, and the top of it is spikes, not a shelf (2026-09-13)

Leaderboard pulled 2026-09-12, before Submission A. **3,620 teams** (345 more than on 2026-09-07),
1st **0.955**, median **0.909** (was 0.902). **Our 0.905 is now below the median**, rank ~1855.

The 2026-09-07 entry described 0.930-0.940 as a shelf of ~940 teams. At one-thousandth resolution it
is not a shelf. It is **spikes of identical scores**, each a public fitted-weights notebook forked
hundreds of times:

| score | teams at exactly | teams at or above | our rank if we score it |
|---|---|---|---|
| 0.925 | 13 | 1,400 | 1,401 |
| 0.930 | 18 | 1,308 | 1,309 |
| 0.935 | 63 | 1,201 | 1,202 |
| **0.936** | **379** | 1,138 | 1,139 |
| 0.937 | 89 | 759 | **760** |
| 0.940 | 140 | 531 | 532 |
| **0.941** | **264** | 391 | 392 |
| 0.942 | 22 | 127 | **128** |

Ranks are counted **behind** every team already at that score, because ties go to the earlier
submission. That convention matters more than usual here: reaching a spike's score exactly places us
behind the whole spike.

**So the targets are lines, not a band: 0.937, then 0.942.** +0.001 past 0.936 is worth 379 places;
+0.001 past 0.941 is worth 264. For scale, 0.905 -> 0.930 (+0.025) buys 546 places and
0.930 -> 0.937 (+0.007) buys another 549 -- the steep part of the curve moved up.

**What the 0.94 tier is made of**, read from the public notebooks rather than guessed.
`tonylica/rsna-knee-dino-radimagenet-rank-ensemble` documents its own blend: DINOv2-small x20,
DINOv3-small x5 (fold ranks), and ten RadImageNet attention heads sharing one ResNet-50 encoder,
combined by per-member, per-target percentile rank. The other top notebooks mount the same stack
(`antoinegg1/rsna-knee-e11-diverse-heads-v20`, `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`,
`mattiaangeli/rsna-knee-coat-resgated-ep10-top3`, `pilkwang/rsna-knee-weights`). **None of it is
usable here, for three separate reasons:**

- the heads were fitted on this competition by other teams and could not be documented as ours in a
  winner's deliverable;
- RadImageNet weights are CC BY-NC-SA, and share-alike conflicts with releasing a winning solution
  under CC-BY-NC 4.0;
- **DINOv3** ships under Meta's DINOv3 licence, whose §1.b.i requires derivative works to be
  distributed under that same agreement -- the same conflict. Recorded so it is not re-litigated.

Permissive alternatives, checked on their model cards: `convnext_tiny.in12k_ft_in1k` (apache-2.0,
27.8M, no BatchNorm), `eva02_small_patch14_224.mim_in22k` (MIT, no BatchNorm),
`tf_efficientnetv2_s.in21k_ft_in1k` (apache-2.0).

**What *is* usable from the same notebooks:** averaging over overlapping slice windows at inference
(`TTA_OVERLAP`), 224px and 336px runs kept as *separate ensemble members* rather than a choice between
them, backbone-family diversity, and a rank blend across families.

**Two consequences for reading Submission A.** The band's middle row got less diagnostic: with the
median up 0.007 since it was written, 0.905-0.912 no longer separates "the gain did not transfer"
from "the public LB drifted". Its lower edge still does -- a strict superset cannot score below its
subset. And whatever it scores, it is a from-scratch two-member ensemble next to 35-member fork
stacks; whether the method is improving is still answered by gold transfer, not rank.

**Time left:** entry deadline Oct 15, final submission Oct 22 -- about five weekly GPU quotas.

### Two defects on the slot path, fixed before any v3 screen could run (2026-09-13)

Neither crashed a v2 run. Both would have shaped the v3 screen.

**`_SLOT_PLANES` knew only the six-slot table.** `PreppedSlotDataset` looks up each slot's plane to
decide mirroring, and the lookup was built from `SLOTS` alone, so the first v3 slot name raised
`KeyError` before a pixel loaded. It is now built from both tables; the default read stays v2.

**The backbone ran on absent slots.** An absent slot arrives as zeros and the mask kept it out of
pooling, but every zero image still went through the backbone. That cost twice: efficientnet_b0's 49
BatchNorm layers averaged one black image per missing slot into their training statistics, and at
19 slots with 5.04 filled on average, **73% of backbone compute would have been zeros**. Present
images are now packed before the backbone and absent positions get zero features.

What that changes, and what it does not:

- **Eval-mode logits are unchanged for every study with at least one present slot.** Tested against
  the unpacked path on both heads, with a stub that makes slots distinguishable (rows carry different
  present patterns, so a mis-scatter cannot pass) and with a real efficientnet whose BatchNorm offsets
  are non-zero. The c2 and c3 confirm checkpoints -- and Submission A -- are the same models under
  the new code.
- **A study with no present slot now reads only the label bias.** Before, it pooled
  `backbone(zeros)`, which is non-zero once BatchNorm has trained offsets; the attention docstring
  already claimed bias-only. Such studies get 0.5 from `build_submission` and never used the output.
- **Training dynamics do change** for BatchNorm backbones: batch statistics now come from anatomy
  only, and the backbone's effective batch varies with fill. That is why the v3 screen runs *both*
  arms packed, rather than pairing a packed 19-slot arm against the unpacked c2 rows.

**A third cost, found by timing rather than reading.** The loader built absent slots as zeros at the
stored 336px and resized them along with everything else. Synthetic studies, 5 filled slots, one
torch thread as inside a DataLoader worker:

| read | before | after |
|---|---|---|
| v2, 6 slots | 320 ms/study | 83 ms/study |
| v3, 19 slots | **1,243 ms/study** | **120 ms/study** |

Before the change the 19-slot read cost 3.9x the 6-slot one *for the same five series* -- nearly all
of it resizing zeros, which would have made the v3 arm loader-bound and confounded the screen's
runtime. Output is identical: antialiased resize is per image, and zeros resize to zeros. The v2 read
also got ~4x faster, which its single absent slot does not explain; the cause was not investigated.
Synthetic pixels decode unlike real JPEGs, so the absolute numbers are indicative only.

**A test that passed for the wrong reason.** The bias-only test and the equivalence tests passed
*before* the fix. An untrained BatchNorm maps black to exactly zero features, so skipping an absent
slot and running it were indistinguishable. They became meaningful only once the test backbone was
given non-zero BatchNorm offsets, and then the bias-only test failed as it should.

**246 tests pass** (238 before).

### Submission A pushed; three more kernels built and dry-run while it scores (2026-09-13)

**Submission A** went in as submission 56193964 from `knee-phase7-submit` version 1, against the
unchanged `rsna-knee-src` (byte-for-byte the files the kernel was dry-run on). Its log before
submitting: the src guard passed, 10 models loaded (c2 at 6 slots, c3 at 2), 4-5 slots filled per
test study, **0 of 3 on the 0.5 fallback**. The new src is held back from `rsna-knee-src` until the
score lands, so Kaggle's rerun on the hidden set sees the same code.

**`phase7-prep9-shard0..3`** (CPU, launched): prep v2 re-run at 9 slice groups for averaging over
slice windows at inference. Sized first: **1.80x prep v2 per study** on the local sample, so ~17 GB
over the corpus and ~4.3 GB per shard. Nine centres over the band contain the five training centres,
and locally that holds exactly: groups 0,2,4,6,8 of every 9-group artifact equal the 5-group artifact
(max pixel difference 0), and the loader's 5-group read of a 9-group artifact equals its read of the
phase6 artifact tensor for tensor. Each shard repeats the pixel comparison on 200 real studies against
the mounted phase6 shard, in a last cell that runs after the manifests are written, so a failure there
costs the check and not the shard.

**`phase7-screen-v3`** (GPU, waiting on the src upload) ran both arms end to end locally, and prints
the backbone's images per batch over the first epoch: 5.0 against 95 unpacked at nineteen slots on
the local sample, which is the property that makes the 19-slot arm cost about what the 6-slot arm
does.

**`phase7-screen-family`** (GPU, waiting on the src upload) is the backbone-family screen, **moved
ahead of the v3 verdict** rather than run on the winning table afterwards. A member's value is
diversity against the members that already exist, and both confirmed members read v2, so it runs on
v2 either way. Arms: `convnext_tiny.in12k_ft_in1k` (apache-2.0) on c2's Adam/OneCycle recipe and
`eva02_small_patch14_224.mim_in22k` (MIT) on c3's differential AdamW recipe, both on c2's six-slot
attention config, packed. Both use gradient checkpointing so batch 8 fits a T4 without becoming a
second variable; that is exact here only because neither backbone has BatchNorm -- checkpointing
re-runs the forward pass during backward, and a BatchNorm backbone would update its running
statistics twice per step. **Judged by the gain each adds to c2+c3**, scored against the confirm
runs' own fold-0 predictions, as a probability mean and a rank mean. The local dry run executed both
arms and the full report.
**`phase7-tta`** (GPU, waiting on prep9) ran end to end on mock checkpoints: every labelled study
predicted at both reads, the reproduction gate and the report executing.

**What those dry runs cannot show, so that nobody reads them as passed:**

- **Every local series is a single DICOM file.** All 20. So every stored group is the same slice, the
  9-group and 5-group reads are identical locally by construction, and whether the four in-between
  groups add anatomy is only testable on the corpus. The TTA kernel therefore prints each member's
  largest prediction change between reads and **asserts it is non-zero** -- a read that silently
  returned the 5-group tensors would otherwise report "no gain" instead of failing.
- A random-init backbone emits near-constant predictions (per-label spread ~4e-7), so equal AUCs
  across reads in a local run carry no information either way.

**A local failure that was not a code failure.** The 19-slot arm first looked hung, then exited 143
with no traceback. The kernel log has the NVIDIA driver failing host allocations (`NV_ERR_NO_MEMORY`)
and `earlyoom` sending SIGTERM to python once memory and swap were both under 10%: the dev box has
7.1 GB RAM with ~500 MB available. At 112px, batch 1 and no loader workers the arm completes. Kaggle's
T4 sessions are not bound by this, but a silent exit here should be checked against `earlyoom` before
it is debugged as a hang.

### Submission A scored 0.916, inside its band -- and the 9-group prep failed its own gate (2026-09-13)

**0.916** (submission 56193964), up from 0.905. **Rank 1,671 of 3,638**; 0.905 would rank ~1,882
today. Median 0.909.

It landed in the pre-registered **expected** row (0.913-0.925). The config ensemble's gain
transferred at roughly the measured size: **+0.011 on the LB** against +0.0125 gold and +0.0156
pooled OOF. The gold -> LB offset is **+0.024** (0.8919 -> 0.916), the fourth pair after +0.038,
+0.016, +0.026. The drift caveat recorded before scoring did not bite -- the score is above the
0.905-0.912 row where the gain and LB drift could not have been told apart.

What it is not: still 0.020 below the 0.936 spike, which ranks 1,156 behind its ties today.

**All four `phase7-prep9` shards failed their last cell -- the gate doing its job.** Every shard
prepped every study (4,407, 0 failures), filed every study into the same slots as prep v2
(0 differences), and landed at the predicted size (4.16-4.19 GB per shard against ~4.3 estimated).
Then groups 0,2,4,6,8 were compared against the phase6 artifacts over 200 studies per shard:

| shard | mean \|diff\| | max \|diff\| |
|---|---|---|
| 0 | 1.17 | 67 |
| 1 | 1.18 | 40 |
| 2 | 1.12 | 48 |
| 3 | 1.24 | 56 |

against a threshold of 0.5 grey levels.

**Cause:** `normalize_series` clips each series to the 1st/99th percentile of *the slices prep
selected* (prep.py:33). Nine groups select 27 slices instead of 15, so the window moves and every
stored pixel moves with it, including those in the five shared groups. The superset property holds
for *which* slices are stored; it does not hold for *how they are scaled*.

**Why the local dry run showed a max difference of exactly 0:** every local series is one slice, so
15 and 27 selected slices are the same image repeated and the percentile window cannot move. The
property that made the in-between groups untestable locally made this untestable too. The gate ran
on real stacks and caught it before any GPU time was spent on TTA predictions that could not have
reproduced the confirm.

**Fix:** take the window from the five training groups' slices and apply it to all nine, so the
shared groups are prep v2's pixels exactly and the in-between groups share their scale. The shards
re-run on CPU.

### v3 is not a better table but a different one; ConvNeXt earns a confirm, EVA02 does not (2026-09-13)

Both screens ran fold 0 of `primary_v3`, target `steven_v4`, gold held out, on the packed model.

**v3 screen** (`knee-phase7-screen-v3`, c2's config on each table, one session):

| arm | val macro | gold | backbone images per batch | peak GPU | train min |
|---|---|---|---|---|---|
| 6 slots, prep v2 | 0.8586 | 0.8682 | 193.5 of 240 | 9.9 GB | 45.0 |
| 19 slots, prep v3 | 0.8571 | 0.8767 | 200.9 of 760 | 13.1 GB | 91.9 |

- **19 minus 6, val: -0.0015 [-0.0080, +0.0044]**, CI spanning zero. By the rule registered before the
  run, **v3 is not worth confirming as a replacement.** Gold, +0.0085 [-0.0093, +0.0287], spans zero
  too.
- **Both tables averaged, minus 6: +0.0085 [+0.0049, +0.0118] val and +0.0148 [+0.0031, +0.0282]
  gold**, both CIs excluding zero. Two c2 models that differ only in how series are filed decorrelate
  enough to ensemble. The census finding was real; it pays as diversity rather than as a better
  single model. Whether it still pays next to c3 and f1 is the next question, measured offline.
- **Packing alone**, unpaired: 0.8586 packed against 0.8571 for the unpacked c2 confirm's fold 0,
  +0.0015 inside ~0.0065 run-to-run noise. The eval-equivalence tests covered inference; this is the
  first evidence that packing did not hurt training.
- **Cost.** Packing held backbone work level (200.9 against 193.5 images per batch), yet the 19-slot
  arm took **2.0x the wall time** and 3.2 GB more GPU memory. Not measured, but the likely cause is the
  loader still building and shipping dense 19-slot batches -- 760 images, 14 of 19 slots zeros -- that
  the model then discards. At this speed a 19-slot five-fold confirm costs ~7.7 h.

**Family screen** (`knee-phase7-screen-family`, c2's six-slot config, fold 0):

| arm | val macro | gold | peak GPU | train min |
|---|---|---|---|---|
| f1 ConvNeXt-tiny, c2's recipe | 0.8588 | 0.8735 | 4.2 GB | 70.4 |
| f2 EVA02-small, c3's recipe | 0.8122 | 0.8228 | 2.8 GB | 80.7 |

Judged against the confirm runs' own fold-0 predictions (val: c2 0.8571, c3 0.8552, c2+c3 0.8705;
gold: 0.8699, 0.8600, 0.8835):

- **f1 added to c2+c3: +0.0062 [+0.0040, +0.0086] val**, probability mean, CI excluding zero, so
  **worth confirming.** Rank mean +0.0046 [+0.0023, +0.0072]; gold +0.0073 [-0.0004, +0.0154]. Alone
  it equals c2 (+0.0017 [-0.0059, +0.0096]) -- a new family adding as much as c3 did, not a better model.
- **f2 is 0.045 worse than c2 alone** (val -0.0449 [-0.0543, -0.0357], gold -0.0471 [-0.0848, -0.0139])
  and subtracts from the ensemble (-0.0023 probability mean, -0.0060 rank mean). On top of f1 it costs
  -0.0033 [-0.0050, -0.0017]. **Not confirmed.** `mim_in22k` is a masked-image-modelling checkpoint
  with no supervised fine-tune, and c3's learning rates were chosen for DINOv2; nothing here separates
  "EVA02 fails at this task" from "EVA02 fails at this recipe". Not pursued, because the diversity it
  was meant to add is already bought by f1.
- Probability mean beat rank mean for f1 as well (+0.0062 against +0.0046), across families -- the
  Phase 6 finding held where it was least expected to.

**The 336px arm does not fit c2's model at batch 8.** Peak memory was 9.9 GB at 224; activations scale
with pixel count, 2.25x at 336, which estimates ~22 GB against a T4's 15. EfficientNet has BatchNorm,
so gradient checkpointing is not exact there. Step 5 needs a batch-size or accumulation decision
before it is built.

**prep9, second run: all four shards pass.** Groups 0,2,4,6,8 against prep v2 over 200 studies per
shard: **mean |diff| 0.0000, max 0**, on every shard; 0 slot differences; 4.16-4.19 GB per shard.
`knee-phase7-tta` pushed.

### The 19-slot model's ensemble value is model count, not table diversity (2026-09-13)

Fold-0 marginal gains, computed offline from the confirm runs' own fold-0 predictions and both screens'
arrays, with rows and labels checked identical across all five sources (863 val, 58 gold):

| ensemble | val macro | gold macro |
|---|---|---|
| c2+c3 | 0.8705 | 0.8835 |
| c2+c3+f1 | 0.8767 | 0.8908 |
| c2+c3+v19 | 0.8744 | 0.8879 |
| c2+c3+f1+v19 | 0.8781 | 0.8928 |

| added | val | gold |
|---|---|---|
| v19 to c2+c3 | +0.0038 [+0.0017, +0.0060] | +0.0044 [-0.0046, +0.0142] |
| v19 to c2+c3+f1 | +0.0014 [-0.0003, +0.0030] | +0.0020 [-0.0055, +0.0101] |
| v19 to c2+c3+f1, rank mean | -0.0001 [-0.0020, +0.0017] | +0.0027 [-0.0054, +0.0116] |
| **v6, the packed six-slot arm, to c2+c3+f1** | **+0.0015 [-0.0001, +0.0030]** | +0.0017 [-0.0060, +0.0098] |

**The control row decides it.** Next to c2+c3+f1, the 19-slot model adds what a second six-slot c2
model from a different session adds. Whatever the recovered table contributes as diversity, f1
already covers; what remains is the generic gain from averaging one more model. **So v3 gets no
five-fold confirm** -- ~7.7 h at its current loader speed, for a gain a model of a known config
matches. The census finding stands as a fact about the data; it did not become a better model, and
next to f1 it did not become a more diverse one either.

**What the control row also prices:** a fourth model of any decent config is worth about +0.0015 on
this fold. That is the bar a later member has to clear to justify its own confirm, and it is also a
cheap option in its own right -- more seeds of confirmed configs buy it without a new screen.

### Slice-window averaging is worth a third model, and f1 is confirmed as one (2026-09-14)

**`knee-phase7-tta`** (67.6 min, nothing trained): every c2 and c3 confirm checkpoint re-read its
own held-out fold and the 58 gold studies from the 9-group prep, once at 5 groups and once at 9.

**The comparison is valid.** Reading 5 of the 9 stored groups reproduced the confirm runs exactly --
c2 macro 0.84850 against 0.84850 (largest prediction difference 8.3e-07), c3 0.84912 against 0.84912
(3.1e-06) -- and reading all 9 moved predictions by up to 0.54, so the in-between groups carry
anatomy the models respond to.

| model | pooled OOF, 5 -> 9 groups | gold, 5 -> 9 groups |
|---|---|---|
| c2 | 0.8485 -> 0.8571, +0.0086 [+0.0072, +0.0100] | 0.8794 -> 0.8942, +0.0147 [+0.0033, +0.0273] |
| c3 | 0.8491 -> 0.8543, +0.0052 [+0.0038, +0.0065] | 0.8805 -> 0.8834, +0.0029 [-0.0057, +0.0122] |
| c2+c3 | 0.8641 -> 0.8686, +0.0045 [+0.0034, +0.0055] | 0.8919 -> 0.9028, +0.0109 [+0.0032, +0.0203] |

Averaging four more slice windows at inference is worth about what a confirmed third model is, and
costs only inference time. By the rule registered before the run, **9-group reads go into
Submission B.** One prep pass serves every member: a 9-group artifact windowed on its five training
groups reads back as the 5-group artifact exactly, so a member staying at 5 groups needs no second
prep.

**`knee-phase7-confirm-f1`** (6.33 h), ConvNeXt-tiny over five folds:

| fold | val macro | gold |
|---|---|---|
| 0 | 0.8516 | 0.8887 |
| 1 | 0.8479 | 0.8713 |
| 2 | 0.8438 | 0.8776 |
| 3 | 0.8216 | 0.8293 |
| 4 | 0.8547 | 0.8833 |
| **pooled OOF (n=4,349) / gold 5-fold mean** | **0.8442** | **0.8945** |

- Alone it sits slightly below c2 on OOF, -0.0043 [-0.0080, -0.0006], and is the best single config
  yet on gold (+0.0151 [-0.0036, +0.0332] over c2, spanning zero). Fold 3 is its weak fold.
- **Added to c2+c3: +0.0038 [+0.0028, +0.0049] pooled OOF**, probability mean, CI excluding zero;
  gold +0.0062 [-0.0006, +0.0127]. Rank mean again gains less (+0.0020). The fold-0 screen said
  +0.0062, so the confirm keeps about 60% of it -- fold 0 flattered f1 as it flattered c2.
- **Promoted as the third member:** it clears the +0.0015 bar with a CI that excludes zero.

**The Submission B candidate, scored offline** from the saved arrays, rows and labels checked
identical (4,349 OOF, 58 gold):

| ensemble | pooled OOF | gold |
|---|---|---|
| Submission A: c2+c3, 5 groups | 0.8641 | 0.8919 |
| c2+c3, 9 groups | 0.8686 | 0.9028 |
| c2+c3+f1, 5 groups | 0.8679 | 0.8980 |
| **c2+c3 at 9 groups, f1 at 5** | **0.8705** | **0.9051** |

- Against Submission A: **+0.0064 [+0.0051, +0.0078] OOF and +0.0132 [+0.0053, +0.0223] gold**,
  both CIs excluding zero.
- The two gains partly overlap: +0.0045 from the windows and +0.0038 from f1 separately, +0.0064
  together. f1 on top of the 9-group pair still adds +0.0020 [+0.0010, +0.0030] OOF.
- f1 at 9 groups is not measured yet; `knee-phase7-tta-f1` runs the same read on its checkpoints
  before Submission B is built.

### f1 gains from nine windows too; confidence weights hurt; 336px earns a confirm (2026-09-14)

**`knee-phase7-tta-f1`** (42.5 min): the same read on f1's five confirm checkpoints. The 5-group read
reproduced f1's confirm exactly (macro 0.84420 against 0.84420, largest prediction difference 0) and
nine groups moved predictions by up to 0.52. **f1: 0.8442 -> 0.8495, +0.0053 [+0.0041, +0.0065]
pooled OOF**; gold 0.8945 -> 0.8964, +0.0020 [-0.0049, +0.0088]. By the same rule, f1 reads nine
groups in Submission B.

**`knee-phase7-screen-res`** (fold 0, c2's six-slot config, one session):

| arm | val macro | gold | backbone images per batch | peak GPU | train min |
|---|---|---|---|---|---|
| 224, batch 8 (control) | 0.8565 | 0.8626 | 193.5 | 9.9 GB | 52.2 |
| 224, batch 8, confidence weights (s3) | 0.8476 | 0.8556 | 193.5 | 10.1 GB | 50.5 |
| 224, batch 4 x 2 accumulated | 0.8515 | 0.8536 | 96.8 | 5.3 GB | 48.7 |
| 336, batch 4 x 2 accumulated (s2) | 0.8588 | 0.8693 | 96.8 | 12.0 GB | 89.9 |

- **Confidence weights hurt: -0.0089 [-0.0153, -0.0014]** val against the control; gold -0.0070,
  spanning zero. The prior recorded before it ran was weak -- source disagreement correlated with gold
  AUC with the wrong sign -- and the screen sided with the prior. Its training loss ended lowest of the
  four (0.3263), so the weighted loss was fit, not ignored. **s3 is closed.**
- **Accumulation alone: -0.0050 [-0.0119, +0.0016]**, spanning zero. BatchNorm over 4 studies instead
  of 8 may cost a little; n=863 does not resolve it.
- **336 against 224 at the same batch: +0.0074 [+0.0007, +0.0146]**, CI excluding zero -- the first
  resolution effect this project has measured, and it shows through the accumulation penalty both
  arms share.
- **As a fourth member next to c2+c3+f1** (fold 0, 5-group reads): **s2 adds +0.0021 [+0.0006,
  +0.0037]**, CI excluding zero, against **+0.0006 [-0.0010, +0.0023]** for one more 224px c2 model
  trained in the same session (and +0.0015 in the earlier session). s3 adds +0.0005. Gold spans zero
  for all three. **s2 is worth a five-fold confirm** (`knee-phase7-confirm-s2`, ~7.7 h at 90 min a
  fold).
- **One caution for that confirm:** the member comparison used 5-group reads, and Submission B reads
  nine. Whether 336px adds to nine-window averaging or overlaps with it is not measured; the confirm's
  checkpoints can take the same 5-vs-9 read afterwards.

**Submission B, final configuration.** c2, c3 and f1, five folds each, all reading nine groups from
one windowed prep pass, probability mean over fifteen models (`knee-phase7-submit-b`; the local dry run
loaded all fifteen checkpoints and ran every cell on the multi-slice test study). Offline, from the
saved arrays with rows and labels checked identical:

| ensemble | pooled OOF | gold |
|---|---|---|
| Submission A (c2+c3, 5 groups), scored 0.916 | 0.8641 | 0.8919 |
| c2+c3 at 9 groups, f1 at 5 | 0.8705 | 0.9051 |
| **Submission B: c2+c3+f1, all at 9 groups** | **0.8713** | **0.9059** |

Against Submission A: **+0.0072 [+0.0057, +0.0087] OOF and +0.0140 [+0.0055, +0.0243] gold.** Reading
f1 at nine rather than five adds +0.0007 [+0.0003, +0.0011] OOF.

### Pre-registered band for Submission B, written before it is pushed (2026-09-14)

**What is known.** Four gold -> LB pairs: +0.038, +0.016, +0.026 and +0.024 (0.8919 -> 0.916). Two
ways to read B:

- *offset method*: B's gold 0.9059 plus the last three offsets (+0.016 to +0.026) gives 0.922-0.932;
- *delta method*: Submission A gained +0.011 on the LB over c2 alone, against a measured +0.0125 gold
  and +0.0156 OOF -- roughly 70-90% of the measurement. Applied to B's +0.0072 OOF and +0.0140 gold
  over A, that is +0.005 to +0.013: 0.921-0.929.

| LB | reading |
|---|---|
| **below 0.916** | **something is broken on the test path.** B beats A on both tiers with CIs excluding zero; a score under A means the nine-group windowed prep or the f1 load behaves differently on hidden DICOMs -- check the log's fallback count and slot fill first |
| 0.916 - 0.920 | the gains partly transferred: the windows and f1 correct errors the hidden set shares with A less than our folds do |
| **0.921 - 0.930** | **expected** |
| 0.931 - 0.936 | better than both methods predict |
| **0.937 or above** | past the 0.936 spike (rank ~790 on the 2026-09-13 board); within the offset method's upper reach, but beyond the delta method's |

The public LB is ~390 studies, so one score is a noisy read of which row B landed in.

**Submitted** as 56227030 from `knee-phase7-submit-b` version 1, after the band above. Its log before
submitting: src guards passed, 15 checkpoints across the three members all reading nine groups,
**0 of 3 on the 0.5 fallback**, slot fill (4, 4, 5) and laterality routes identical to Submission A's
on the same three studies. Inference is 6.0 s/study against A's 3.3 -- the nine-group prep and five
more models -- which at ~1,300 hidden studies is ~2.2 h, within the runtime limit.

### Submission B scored 0.921; the 336px member is confirmed; the field re-read (2026-09-19)

**0.921** (submission 56227030), up from 0.916. On the 2026-09-18 board (3,959 teams) that is rank
**1,789**; median 0.913. It landed at the **floor of the pre-registered band** (0.921-0.930): exactly
the delta method's floor and 0.001 under the offset method's. Against Submission A the LB moved
**+0.005** for a measured +0.0072 pooled OOF and +0.0140 gold, so the gains transferred at roughly 70%
of the OOF delta and a third of the gold one. The gold -> LB offset is now **+0.015** (0.9059 -> 0.921),
the fifth pair after +0.038, +0.016, +0.026, +0.024 -- it has shrunk every time gold rose. Nothing
points at breakage: the log showed 0 fallbacks and the same slot fill and laterality routes as A.

**s2 is confirmed** (`knee-phase7-confirm-s2`, 8.61 h: c2's six-slot attention config read at 336px,
batch 4 x 2 accumulated, 5 folds of `primary_v3`, target `steven_v4`, gold held out of every split):

| fold | val macro | gold |
|---|---|---|
| 0 | 0.8628 | 0.8592 |
| 1 | 0.8482 | 0.8661 |
| 2 | 0.8457 | 0.8650 |
| 3 | 0.8683 | 0.8692 |
| 4 | 0.8472 | 0.8598 |
| **pooled OOF (n=4,349)** | **0.8538** | -- |
| **gold 5-fold mean** | -- | **0.8833** |

- **s2 minus c2 (336 against 224): +0.0053 [+0.0020, +0.0085]** pooled OOF, CI excluding zero; gold
  +0.0039 [-0.0100, +0.0193], spanning zero. The fold-0 screen said +0.0074; the confirm keeps ~70%.
- **s2 added to c2+c3+f1 (all at 5 groups): +0.0042 [+0.0033, +0.0051]** pooled OOF, 0.8679 -> 0.8721,
  CI excluding zero -- nearly three times the +0.0015 one more known model prices at. Gold -0.0010
  [-0.0062, +0.0044], spanning zero: n=58 does not resolve a fourth member either way.
- By the rule registered before the run, **s2 is the fourth member.** Per label on gold it is strongest
  on Baker's (0.978), Medial OA (0.963), ACL (0.953) and MCL (0.952), weakest on Synovitis (0.736) and
  Lateral Meniscus (0.793) -- the same shape as c2.
- Not yet read at 9 groups; that read (Phase 8 Step 1) precedes Submission C. The six rows are in
  `results/experiments.csv`; checkpoints and OOF/gold arrays under `checkpoints/knee-phase7-confirm-s2/`.

**The field, re-pulled 2026-09-18.** 3,959 teams (339 more than on 09-12). 1st **0.957** (135
submissions), then 0.955 x3, 0.954 x8, 0.953 x6, 0.952 x7, 0.951 x6, 0.950 x7. The spikes grew: **538
teams at exactly 0.941** (was 264), 321 at 0.936, 179 at 0.940. Ranks behind ties:

| score | teams at it | rank if we score it |
|---|---|---|
| 0.925 | 12 | 1,681 |
| 0.936 | 321 | 1,433 |
| 0.940 | 179 | 892 |
| 0.941 | 538 | 713 |
| 0.942 | 28 | 175 |
| 0.945 | 24 | 100 |
| 0.950 | 7 | 39 |
| 0.953 | 6 | 19 |
| 0.955 | 3 | 5 |
| 0.957 | 1 | 2 (sole first needs 0.958) |

**The arithmetic for first place.** 0.921 -> 0.957 is +0.036 on the public LB. With the offset at
+0.015 and shrinking, LB 0.955 needs gold about 0.945 and pooled OOF against steven_v4 near 0.90 (now
0.8713, ~0.876 expected with s2 at 9 groups). The top 30 sit inside 0.006 and the private set (~910 of
~1,300 studies) will reshuffle them by about that much; several leaders have 100+ public submissions.
The measured levers, summed at the gold tier, are +0.02 to +0.04 -- perhaps +0.015 to +0.03 on the LB at
B's transfer rate. That is the gap with no slack, which is why Phase 8 pre-registers every screen and
promotes only on a paired pooled-OOF CI.

**Three decisions taken 2026-09-19:** main leaderboard only (the Efficiency Track entry is dropped;
runtime stays logged for the record); the 58 gold studies may enter training for the final aggressive
pick only, after every screening decision is closed; free compute only, with Colab free as a third lane
for fold-0 screens. **Phase 8 plan:** `ai/plans/2026-09-19-phase8-win-plan.md` (supersedes Phase 7's).

### Phase 8 opens: the s2 read and Submission C are built, the quota is not there yet (2026-09-19)

**GPU lane A is blocked until the weekly reset.** `kaggle kernels push` for `knee-phase8-tta-s2`
was refused with "Maximum weekly GPU quota of 30.00 hours reached" -- the 8.6 h s2 confirm took the
rest of this week's hours. The last reset was observed on 2026-09-13, so the read runs on the reset
and Submission C follows it; sequencing only, nothing in the plan changes.

**Both Step 1 kernels are built and dry-run.**

- `knee-phase8-tta-s2` is the Phase 7 TTA notebook with the member swapped for s2 and the read size
  raised to 336 (the stored size, so no resize). Dry run on the 20-study sample with the real five
  checkpoints and mock OOF files, at 112px / batch 1: every cell ran, both reads wrote their arrays.
  As with every TTA dry run, the reproduction and change gates were patched out locally -- single-slice
  series make the two reads identical by construction -- and are checked on the corpus.
- `knee-phase8-submit-c` is Submission B's kernel plus a fourth `MEMBERS` entry (s2, six slots,
  read=336, groups=9). Dry run with all twenty real checkpoints on the sample's test set: 20 models
  loaded, one multi-slice study prepped and predicted, the two studies without local DICOMs fell back
  to 0.5, header and probability checks passed. It had to run on CPU -- another process holds 2.6 GB
  of the local 4 GB GPU -- where the forward took 50 s/study for twenty models; Kaggle's T4 ran B's
  fifteen at 6.0 s/study all-in, so C should land near 10 s/study, ~3.7 h over the hidden set,
  inside the 6 h budget the plan sets.

**The offline scorer reproduces Submission B exactly** from the saved 9-group arrays: pooled OOF
0.8713, gold 0.9059 (`ai/scratch/phase8-submit/score_members.py`). It takes s2's arrays the moment
the read lands.

**Step 2 landed, test-first, 280 tests (254 before):**

- `head="group_attention"`: per-diagnosis attention over every (slot, group) token instead of the
  mean over a slot's groups. The 9-group reads gained +0.005 to +0.009 with nothing retrained, so the
  windows carry signal the mean discards; this head lets each label weigh the window that shows its
  finding. No positional embedding yet -- a token does not know its depth -- by choice, so the screen
  measures the MIL change alone. slot_attention's eval logits are untouched (the equivalence tests
  that protect the scored checkpoints still pass).
- Random group subsampling (`random_groups`, its own switch so a screen arm changes one thing): a
  5-group training read of a 9-group artifact draws a random sorted subset of the windows; eval keeps
  the fixed spread, so 5-of-9 still reproduces the confirm (pinned by a test).
- `ModelEma` (decay 0.999), updated once per optimizer step, never per batch, so accumulation does
  not change its window.
- The multi-source target needs no model change: `multi_source_labels` builds a wide frame,
  "{source}:{label}", the loader takes `label_columns`, `KneeModel(num_labels=12 * K)` and masked
  BCE skip the NaN where a source never rated a study, and `average_source_heads` folds the K
  readings back to twelve. `blend_sources` is the probability-mean target and `fold_confidence` the
  shared hard-label-plus-confidence mapping that the kernels and the gold arm each had a private copy of.

**Per-label blend weights buy nothing at three members (bet 5, measured).** `fit_weights.py`
fits non-negative per-label member weights by coordinate ascent on four folds and scores the fifth,
rotated: c2+c3+f1 at 9 groups, CV-weighted blend against the plain mean, **+0.0000 [-0.0007,
+0.0007]**. The full-fit weights move (Contusion would go 0.92 to c3, Synovitis 0.55 to c2) but the
gain does not survive the fold rotation. The plain probability mean stays; the script re-runs each
time a member is added, and only a CV delta whose CI excludes zero would change the blend.

### The three Phase 8 screens are built and dry-run; a two-slot member is now insurance-wrapped (2026-09-19)

**Screens A, B and C exist as kernels** (`phase8-screen-arch`, `-target`, `-scale`), each derived
from the Phase 7 resolution screen's setup, gate and folds cells by asserted text swaps, each with its
arms, pairings and member-gain readouts as pre-registered in the Phase 8 plan. All three ran end to end
on the 20-study sample through `nbrun.py` (64-70px, one epoch, batch 1, `pretrained=False`, mock
fold-0 arrays for c2, c3, f1 and s2 sized to the sample, four of its studies standing in as gold).
The local numbers mean nothing -- a random-init backbone emits near-constant predictions, so every
"as a fifth member" delta is exactly zero and every nine-versus-five read is identical -- but every
cell, every branch of every report, and Screen C's out-of-memory fallback (which the shared local GPU
triggered for real) executed.

**Screen C's dry run found a loader property worth knowing.** A member reading two slots
(`slot_names[:2]`: SAG_FLUID, COR_FLUID) raises `StudyDecodeError` on a study that has neither, because
`load_study_npz(order=...)` filters to the requested slots. The competition's full `train_series.csv`
says **every one of the 4,407 training studies has a sagittal or coronal fluid-sensitive series**
(SAG_FLUID fills 94.2%, COR_FLUID 96.4%, their union 100%), which is why the c3 confirm and both
submissions never hit it. The hidden set has no such guarantee, and in the submission kernel one
member's raise inside `predict_one` would have sent the *whole* study -- every member -- to the 0.5
fallback. `phase8-submit-c` now catches a member that finds none of its slots, averages the members
that could read the study, counts the skips in the log, and raises (hence 0.5) only if no member read
anything. Zero effect on any training study; insurance on the hidden ones.

**Screen B's blend arm is steven_v4 + ours + lixin73**, the three best independent sources on gold
(0.8927 / 0.8616 / 0.8488); pilkwang_v2 (0.8142) stays out of the blend but is one of the four heads
in the multi-source arm, whose report also prints each head alone and the average without it.

**Final-selection rule verified by the user on the competition's My Submissions page (2026-09-19):
two final submissions can be selected** (a third attempt is refused with "you have selected the
maximum number of submissions"). Under Kaggle's rule the better private score of the two counts, so
the Phase 8 plan's second slot -- the gold-fine-tuned aggressive pick -- is a free hedge, not a risk.

### s2 gains at nine groups too; Submission C scored offline, band registered before the push (2026-09-19)

**`knee-phase8-tta-s2`** (32.9 min, pushed at the quota reset, nothing trained): s2's five confirm
checkpoints re-read their own folds and the 58 gold studies at 336px, at 5 and at 9 groups.

- **The comparison is valid.** The 5-group read reproduced the confirm exactly: macro 0.85378 against
  0.85378, largest prediction difference 5.4e-07. Nine groups moved predictions by up to 0.60.
- **s2, 5 -> 9 groups: +0.0084 [+0.0069, +0.0099] pooled OOF** (0.8538 -> 0.8622), CI excluding zero;
  gold +0.0082 [-0.0016, +0.0194], spanning zero. The largest window gain of the four members (c2
  +0.0086, f1 +0.0053, c3 +0.0052). By the rule every member has been held to, **s2 reads nine groups.**

**Submission C offline** (c2+c3+f1+s2, all at nine groups, probability mean over twenty models), rows and
labels checked identical across the four sources (`ai/scratch/phase8-submit/score_members.py s2`):

| ensemble | pooled OOF | gold |
|---|---|---|
| Submission B (c2+c3+f1), scored 0.921 | 0.8713 | 0.9059 |
| **Submission C (c2+c3+f1+s2)** | **0.8754** | **0.9046** |

- **Against B: +0.0041 [+0.0033, +0.0049] pooled OOF**, CI excluding zero, and higher on all twelve
  labels (largest: Lateral Meniscus +0.0118, PF OA +0.0096, ACL +0.0092). **Gold -0.0013 [-0.0056,
  +0.0031]**, spanning zero: seven labels up, five down, none by more than 0.011.
- **Per-label blend weights at four members: +0.0002 [-0.0005, +0.0010]** under the fold-rotated CV,
  spanning zero. The plain probability mean stays, as at three members.

**Pre-registered band for Submission C, written before it is pushed.** Five gold -> LB pairs so far
(+0.038, +0.016, +0.026, +0.024, +0.015). B moved the LB +0.005 for +0.0072 pooled OOF, about 70% of the
OOF delta. Applied to C's +0.0041 that is **+0.003**. Scaled from n=4,349 to the ~390 public studies the
OOF CI widens about 3.3x, to roughly +/-0.003. The gold tier says -0.001 with its own CI spanning zero.

| LB | reading |
|---|---|
| **0.917 or below** | **something is broken on the test path.** C is B plus one member of c2's family; four points under B is outside both tiers. Check the log first: the s2 read at 336, members skipped, fallbacks, slot fill |
| **0.918 - 0.921** | **flat. s2's gold delta spans zero, so a flat LB is consistent with the measurement** -- it says the OOF gain did not reach the public 390, not that anything broke. s2 stays a member; members are decided on pooled OOF, never on one LB read |
| **0.922 - 0.925** | **expected**: the OOF delta transferring at the rate B's did |
| 0.926 - 0.929 | better than the delta method predicts; within the offset method's reach (gold 0.9046 + 0.015 to 0.024) |
| **0.930 or above** | implausible from one more c2-family member; check for a leak or a mis-scored file before believing it |

C runs on the published `rsna-knee-src` (commit c5bfa38, byte-identical to what Submission B scored on
and to what the tta kernels used). The Phase 8 src -- group attention, EMA, random windows, the target
builders -- stays unpublished until C's score lands, so Kaggle's rerun on the hidden set sees the code C
was built for. Expected inference on T4: ~10 s/study for twenty models, ~3.7 h over ~1,300 hidden
studies, inside the 6 h budget; the kernel log states the measured rate on the visible studies.

**Submitted** as 56360159 from `knee-phase8-submit-c` version 1 (2026-09-19 13:23 UTC), after the band
above. Its log before submitting: the src guard passed on the published c5bfa38 src, **20 checkpoints
across four members** (s2 at 336px, every member at nine groups), **0 of 3 on the 0.5 fallback, 0
members skipped**, slot fill (4, 4, 5) as in A and B. **8.06 s/study on T4** (prep 3.5, forward 4.5),
so ~2.9 h over ~1,300 hidden studies. Kaggle spent 39 minutes starting the session before the first
cell ran; that wall time counts toward the limit too, and it is why this kernel looked stuck.

**C is B plus s2, exactly.** On the three visible studies C differs from B by a mean 0.016 (max 0.069),
and backing out 4C - 3B gives a valid probability for s2 on every entry (0.005 to 0.973): B's fifteen
models ran unchanged, s2 added a quarter-weight vote.

### Submission C scored 0.925, the top of its expected row (2026-09-19)

**0.925** (submission 56360159), up from 0.921. On the 16:34 UTC board (3,994 teams, median 0.914, 1st
0.957) that is **rank 1,712**. It landed in the registered **expected** row (0.922-0.925), at its top.

- **The OOF delta transferred in full this time.** C over B: +0.004 on the LB for +0.0041 pooled OOF --
  about 100%, against B's ~70%. The gold tier's -0.0013 [-0.0056, +0.0031] was, as its CI said, noise:
  a fourth member that pooled OOF (n=4,349) favoured and 58 studies could not resolve did move the
  public 390. That is the case for deciding members on pooled OOF, now observed on the LB once.
- **Gold -> LB offset +0.020** (0.9046 -> 0.925), the sixth pair after +0.038, +0.016, +0.026, +0.024,
  +0.015 -- it did not keep shrinking, which puts the offset method's reading of "LB 0.955 needs gold
  ~0.945" at about gold 0.935 to 0.940 on the last three offsets.
- The run held: 0 fallbacks and 0 skipped members on the visible studies; the hidden-set rerun
  completed in its window.

Next, per the plan: the Phase 8 src (group attention, EMA, random windows, the target builders) goes
to `rsna-knee-src` now that C's score has landed, then Screens A and B take the two GPU lanes and
Screen C follows B.

### Pre-registered rules for Screens A, B and C, written before they are pushed (2026-09-19)

All fold 0 of `primary_v3`, gold held out, c2's recipe unless stated. "Member gain" means the arm added
as a fifth member to the confirm runs' own fold-0 predictions of c2+c3+f1+s2, probability mean; **the
bar is +0.0015 with the paired CI above zero**, the price of one more model of a known config. Nothing
is promoted on fold 0 alone: passing a rule buys a five-fold confirm, not a place in a submission.

**Screen A (`knee-phase8-screen-arch`, prep9 artifacts, fixed five groups unless stated):**
- **a1 group attention is worth confirming** if a1 minus a0 on val has a CI excluding zero *or* a1
  clears the member bar. a0 is the in-session control, since run-to-run noise here is ~0.0065.
- **a2 random training windows / a3 EMA join the recipe** only if each adds to a1: a2 minus a1, and a3's
  EMA weights minus its own raw weights (same run), each with a CI above zero. a3's raw weights minus a1
  is reported as the measured noise floor between two runs of one config.
- Every arm is also read at nine groups and reported; the nine-group member comparison uses all four
  confirmed members' nine-group arrays.

**Screen B (`knee-phase8-screen-target`, phase6 artifacts):** arms t0 steven_v4, t1 one head per
source (K=4), t2 probability blend of steven_v4 + ours + lixin73.
- **Every arm is scored against one fixed truth, steven_v4 binarised, and on gold** -- never against
  its own target. A negative fixed-truth delta for t1/t2 is expected of a better target and is not a
  veto.
- **Replacement (the target itself): gold decides.** A target whose gold delta over t0 has a CI lower
  bound above zero -- the standard c0's promotion met, +0.0339 [+0.0067, +0.0625] -- becomes the
  target for later confirms.
- **Member:** a target that clears the member bar on the fixed truth is confirmed as a member config
  even when its gold delta spans zero; a model trained on a different label source is the most
  decorrelated member available.
- **If a target is promoted, the fixed truth switches once**, to that target binarised, and every
  member's saved OOF arrays are re-scored against it before any later member decision.

**Screen C (`knee-phase8-screen-scale`, phase6 artifacts):** s4 DINOv2-small at 336 and s6 DINOv2-base
at 224 on c3's two-slot recipe; s5 ConvNeXt-tiny at 336 on f1's.
- Each arm is judged by the member bar, and against the member it would replace (s4, s6 vs c3; s5 vs
  f1) by the paired val CI.
- **s5 gets its ~13 h confirm only if its member gain is at least +0.0030**, twice the bar. If s5 falls
  back to batch 4 x 2 on memory, that is recorded as a second variable against f1.

### Screen B: no new target, but the three-source blend earns a confirm as a member (2026-09-20)

`knee-phase8-screen-target` (3.45 h, fold 0, c2's recipe on the phase6 artifacts, gold held out). Every
arm scored against the fixed truth (steven_v4 binarised) and on gold:

| arm | trains on | val vs steven_v4 | gold | min |
|---|---|---|---|---|
| t0 | steven_v4 (control) | 0.8614 | 0.8640 | 48.9 |
| t1 | one head per source, K=4, averaged | 0.8563 | 0.8754 | 46.8 |
| t2 | probability blend: steven_v4 + ours + lixin73 | **0.8650** | **0.8833** | 47.2 |

Read by the rules registered before the push:

- **Replacement -- no target is promoted.** Gold over t0: t1 +0.0114 [-0.0104, +0.0345], t2 +0.0193
  [-0.0036, +0.0433]; both CIs span zero, so steven_v4 stays the target and **the fixed truth does not
  switch.** t2 is the closest any target has come since c0 (+0.0339 [+0.0067, +0.0625]), and it is also
  ahead on the fixed truth (+0.0036 [-0.0025, +0.0100]) where a better target was expected to *lose*.
- **Member -- t2 clears the bar and is confirmed as a member config.** Added to c2+c3+f1+s2 on the fixed
  truth: **t2 +0.0027 [+0.0016, +0.0038]**, against **+0.0017 [+0.0003, +0.0030] for t0**, one more
  steven_v4 model trained in the same session. t1 adds +0.0010 [+0.0001, +0.0019]: CI above zero, point
  under the bar, **not confirmed.** On gold the member gains are t2 +0.0056 [-0.0003, +0.0115], t1
  +0.0023, t0 -0.0007: the same order, and t2's is the largest fifth-member gold gain measured.
- **The per-source heads say something about the sources.** Inside t1, the lixin73 head alone scores
  0.8610 against steven_v4's own truth, above the steven_v4 head (0.8563), and 0.8750 on gold; dropping
  the pilkwang head (0.8017 / 0.8118 alone) lifts the average to 0.8607 / 0.8792. Averaging four heads
  inside one model bought less than averaging three sources into one target.
- **The blend has lower prevalence than steven_v4** (positive rate 0.092-0.481 against 0.181-0.587),
  because ours and lixin73 call fewer positives; AUC does not see it, but it is why t2 is a genuinely
  different model and not a recalibrated t0.

**Next for t2:** a five-fold confirm of c2's recipe on the blend target (`knee-phase8-confirm-t2`), scored
against the fixed truth and as a fifth member of the confirmed four, then the 5-vs-9 group read.

### Screen A: random training windows are worth more than every member added since Phase 7 (2026-09-20)

`knee-phase8-screen-arch` (4.39 h, fold 0, c2's recipe on the prep9 artifacts, gold held out). All arms
train on the fixed five of nine stored groups -- prep v2's pixels exactly -- except a2, which draws five
of the nine at random each epoch; every arm is read at five and at nine groups.

| arm | head | train windows | EMA | val (5 / 9 groups) | gold (5 / 9) | loss | min |
|---|---|---|---|---|---|---|---|
| a0 | slot attention | fixed | no | 0.8569 / 0.8642 | 0.8605 / 0.8745 | 0.3814 | 44.5 |
| a1 | group attention | fixed | no | 0.8583 / 0.8658 | 0.8602 / 0.8762 | 0.3767 | 42.5 |
| **a2** | group attention | **random** | no | **0.8738 / 0.8778** | **0.8865 / 0.8907** | 0.4229 | 43.7 |
| a3 raw | group attention | fixed | no | 0.8619 / 0.8690 | 0.8675 / 0.8849 | 0.3748 | 45.2 |
| a3 EMA | group attention | fixed | yes | 0.8614 / 0.8664 | 0.8721 / 0.8810 | -- | -- |

Read by the rules registered before the push:

- **Random training windows are the finding: a2 minus a1 is +0.0155 [+0.0091, +0.0219] val and +0.0263
  [+0.0055, +0.0464] gold**, both CIs excluding zero. That is the largest single-change fold-0 gain since
  Phase 5's schedule (+0.059), and it costs nothing: the artifacts already store nine windows, the loader
  picks five of them, and the epoch takes the same 44 minutes. **Adopted into the recipe.**
- **Its training loss is the highest of the five (0.4229 against 0.3748-0.3814), which is what an
  augmentation looks like.** The model can no longer memorise a fixed set of slices, so it fits its
  training set less and its held-out set more.
- **Group attention alone is null**: a1 minus a0 is +0.0014 [-0.0051, +0.0078] val and -0.0003 on gold.
  It clears the member bar by a hair (+0.0016 [+0.0003, +0.0029]) but a3's raw weights minus a1 -- two
  runs of the same config -- is +0.0036, so the noise floor here is larger than the effect. It rides
  along in a2 rather than on its own merits.
- **EMA does not pay**: EMA minus its own raw weights is -0.0005 [-0.0038, +0.0027] val, +0.0046 on gold,
  both spanning zero. **Closed.**
- **Window averaging still pays, less so for a2**: nine groups minus five is +0.0073 for a0, +0.0074 for
  a1 and only +0.0040 for a2 -- training on random windows already teaches most of what reading nine
  windows buys at inference, but not all of it.
- **As a fifth member next to c2+c3+f1+s2, a2 is the strongest yet measured**: +0.0038 [+0.0025, +0.0049]
  at five groups and +0.0029 [+0.0018, +0.0040] at nine, against +0.0007 for a0 (one more known model)
  and +0.0027 for Screen B's t2. Gold member gains all span zero at n=58, ordered the same way.
- **The head's windows are not uniform**: averaged over the gold studies its attention runs
  [0.15, 0.19, 0.26, 0.24, 0.15] over the five windows -- the middle of the joint carries more than the
  ends, which is what the 0.20-0.80 band was sampling blind.

**Confirmed next:** a2's config (group attention + random windows) on five folds, trained on the prep9
artifacts, with the five- and nine-group reads in the same session. After that, the obvious question
Screen A raises: random windows are a training-time augmentation that nothing about the head or the
backbone makes specific, so **re-confirming c3 and f1 with random windows** is the next candidate queue.

### t2 confirmed: the blend target trains the best single member yet, and beats c2 on c2's own truth (2026-09-21)

`knee-phase8-confirm-t2` (4.44 h, five folds of `primary_v3`, c2's recipe trained on the probability
blend of steven_v4 + ours + lixin73, scored against the fixed truth steven_v4 binarised, gold excluded
from every split).

| fold | val macro | gold |
|---|---|---|
| 0 | 0.8633 | 0.8830 |
| 1 | 0.8491 | 0.8633 |
| 2 | 0.8552 | 0.8656 |
| 3 | 0.8652 | 0.8693 |
| 4 | 0.8515 | 0.8630 |
| **pooled OOF (n=4,349) / gold 5-fold mean** | **0.8563** | **0.8872** |

**The best single member measured on pooled OOF** -- c2 0.8485, c3 0.8491, f1 0.8442, s2 0.8538, t2
**0.8563** -- and second on gold behind f1's 0.8945.

- **t2 minus c2: +0.0078 [+0.0044, +0.0113] pooled OOF**, CI excluding zero. Read carefully: both are
  scored against **steven_v4 binarised**, which is the *control's* training target and the challenger's
  home disadvantage. The blend-trained model wins on the control's own turf, over 4,349 studies. Gold
  agrees in direction (+0.0078) and cannot resolve it (CI spanning zero, as at n=58 it never can).
- **As a fifth member: +0.0029 [+0.0023, +0.0034] pooled OOF**, CI excluding zero, against the screen's
  +0.0027 -- a fold-0 screen predicting a five-fold confirm almost exactly, which has not happened
  before on this project. Gold +0.0027 [-0.0022, +0.0080], spanning zero. **Promoted as a member.**
- The rule registered before Screen B stands: the *target* did not switch, because the switch was
  pre-registered on gold and gold cannot resolve it. But this confirm is the strongest evidence yet
  that the blend is a better training target and not merely a decorrelated one, and the cheap
  experiment that would settle it -- a2's config trained on the blend against a2's config trained on
  steven_v4, one fold -- is now the first candidate for next week's quota.
- **Still to do for Submission D:** t2 trained and was scored at five groups, while every other member
  reads nine. Its 5-vs-9 read (`knee-phase8-tta-t2`) is built and waits for a lane.

### Screen C: scaling the other families up pays nothing, and the 336 gain was efficientnet's alone (2026-09-21)

`knee-phase8-screen-scale` (6.81 h, fold 0, phase6 artifacts, each arm on its family's recipe, gold held
out). All three arms fit batch 8 with gradient checkpointing, so none needed the accumulation fallback.

| arm | backbone | val | gold | peak GPU | min |
|---|---|---|---|---|---|
| s4 | DINOv2-small at 336, c3's 2-slot recipe | 0.8541 | 0.8529 | 2.2 GB | 67.6 |
| s5 | ConvNeXt-tiny at 336, f1's 6-slot recipe | 0.8439 | 0.8553 | 8.8 GB | 166.2 |
| s6 | DINOv2-base at 224, c3's 2-slot recipe | 0.6352 | 0.6216 | 2.7 GB | 71.2 |

**None clears the member bar; none is confirmed.**

- **s4: 336 does nothing for DINOv2.** Against c3's own fold-0 predictions it is -0.0011 [-0.0074,
  +0.0050] and as a fifth member +0.0000 [-0.0011, +0.0012]. The +0.0053 that 336 bought efficientnet
  (s2, confirmed) **did not generalise to the other family**: a patch-14 ViT at 336 sees 576 patches
  instead of 256, but its position embedding is interpolated away from what it was pretrained at, and
  the two effects appear to cancel.
- **s5: ConvNeXt at 336 is worse than at 224**, -0.0077 [-0.0144, -0.0010] val and -0.0334 [-0.0595,
  -0.0106] gold against f1, both CIs excluding zero, for 2.4x the training time (166 min against f1's
  70). The most expensive arm this project has run and the clearest loss.
- **s6: DINOv2-base did not train.** val 0.6352, -0.2200 against c3. Its 86M parameters on c3's
  differential rates (8e-6 backbone / 1e-3 head) over 3,486 studies is the EVA02 situation again:
  nothing here separates "the model is wrong for the task" from "the recipe is wrong for the model".
  **Not pursued** -- the diversity a third ViT would add is already covered, and the two prior attempts
  at a bigger or differently-pretrained transformer (EVA02-small, now DINOv2-base) both failed the same
  way, which is itself the finding: on 3,486 studies the small backbones are not the binding constraint.

**What the phase's three screens say together.** Capacity and resolution are exhausted for every family
but efficientnet; the gains left are in **what the model is trained on** (Screen B's blend target,
+0.0029 as a member) and **how the data is presented** (Screen A's random windows, +0.0038). Both are
free. That is where the remaining quota goes.

### Pre-registered rules for Screen D, written before it is pushed (2026-09-21)

`knee-phase8-screen-windows`, fold 0, prep9 artifacts, gold held out. Four arms, each trained on five
of the nine stored windows drawn at random: **d1** c3's two-slot DINOv2 config, **d2** f1's six-slot
ConvNeXt config, **d3** a2's config on the three-source blend, **d4** a2's config on steven_v4 (d3's
in-session control and a second read of a2's fold 0). "Member gain" is now the arm added as a **sixth**
member to c2+c3+f1+s2+t2; the bar stays **+0.0015 with the paired CI above zero**.

- **d1 / d2 are worth confirming** if the arm minus its own confirm's fold-0 predictions has a val CI
  excluding zero, *or* if it clears the member bar. Pairing against the confirms rather than a fresh
  control is licensed by Screen A: its a0 arm (c2, fixed windows, prep9 artifacts) scored 0.8569 against
  c2's confirm fold-0 0.8571, so training on the fixed five of nine reproduces the phase6 training.
- **d3 minus d4 settles the target** on 863 studies where gold's 58 could not. The blend becomes the
  training target for later confirms if that val CI excludes zero. Both arms are scored against the
  fixed truth, steven_v4 binarised, which is d4's own training target -- so a d3 win is measured against
  the control's home advantage, as t2's +0.0078 already was.
- **Amendment to the rule registered before Screen B, stated now and before the result:** the fixed
  truth **stays steven_v4 binarised for the rest of Phase 8 even if the blend is promoted as the
  training target.** Two reasons. Every member's saved OOF arrays are scored against it, so switching
  would mean rescoring all of them and would break comparability with Submissions A-C; and scoring
  blend-trained models against steven_v4 is *conservative* -- it biases against the newer target rather
  than for it, and t2 cleared that bias anyway. The earlier rule's concern (that a promoted target's
  models would be judged on the old target's home turf) is real but points the safe way.

### t2 gains most of all from nine windows; five members reach 0.8781 (2026-09-21)

`knee-phase8-tta-t2` (28.7 min). Both gates passed: the five-group read reproduced the confirm
**exactly** (macro 0.85627 against 0.85627, max prediction difference 0.00e+00 -- the first member to
match to the bit, because t2 trains and evaluates on the same phase6 pixels the prep9 fixed five
reproduce), and nine groups moved predictions by up to 0.57.

| member | pooled OOF, 5 -> 9 | gold, 5 -> 9 |
|---|---|---|
| **t2** | 0.8563 -> **0.8654**, **+0.0091 [+0.0075, +0.0108]** | 0.8872 -> 0.8991, +0.0120 [+0.0040, +0.0212] |
| c2 | +0.0086 | +0.0147 |
| s2 | +0.0084 | +0.0082 |
| f1 | +0.0053 | +0.0020 |
| c3 | +0.0052 | +0.0029 |

**t2 gains the most of any member from window averaging, and is the only one whose gold CI also
excludes zero.** At nine groups it is the best single member by a wide margin: 0.8654 against s2's
0.8622, c2's 0.8571, c3's 0.8543 and f1's 0.8495.

**Five members at nine groups, scored offline** (`ai/scratch/phase8-submit/score_members.py +t2`):

| set | pooled OOF | gold |
|---|---|---|
| Submission C: c2+c3+f1+s2 | 0.8754 | 0.9046 |
| **+ t2** | **0.8781** | **0.9061** |

+0.0027 [+0.0023, +0.0032] OOF, CI excluding zero; gold +0.0015 [-0.0029, +0.0056]. Eleven of twelve
labels improve on OOF (Effusion unchanged); on gold, Synovitis (-0.013) and Lateral OA (-0.006) give
back a little. **a2 joins as the sixth member when its confirm lands**, and Submission D follows.

### a2 confirmed: random windows hold over five folds, and the six-member set reaches 0.8799 (2026-09-21)

`knee-phase8-confirm-a2` (5.67 h, five folds of `primary_v3`, group attention + five of nine windows
drawn at random each epoch, read at five and nine groups in the same session, gold excluded everywhere).

| fold | val (5 / 9 groups) | gold (5 / 9) | loss | min |
|---|---|---|---|---|
| 0 | 0.8683 / 0.8750 | 0.8802 / 0.8859 | 0.4239 | 50.8 |
| 1 | 0.8569 / 0.8669 | 0.8895 / 0.8979 | 0.4171 | 48.4 |
| 2 | 0.8591 / 0.8652 | 0.8744 / 0.8968 | 0.4163 | 48.0 |
| 3 | 0.8716 / 0.8802 | 0.8871 / 0.8945 | 0.4186 | 48.3 |
| 4 | 0.8599 / 0.8678 | 0.8962 / 0.9076 | 0.4207 | 47.5 |
| **pooled / gold mean** | **0.8625 / 0.8704** | **0.8983 / 0.9067** | | |

- **a2 minus c2: +0.0140 [+0.0110, +0.0171] pooled OOF and +0.0189 [+0.0027, +0.0360] gold** at five
  groups, both CIs excluding zero. Screen A predicted +0.0155 on one fold; five folds say +0.0140.
  **The single largest per-model gain this project has measured since Phase 5's schedule**, from an
  augmentation that costs nothing.
- **The best member on both tiers**: at nine groups 0.8704 OOF (t2 0.8654, s2 0.8622, c2 0.8571) and
  0.9067 gold (t2 0.8991, f1 0.8964).
- **Added to c2+c3+f1+s2: +0.0028 [+0.0023, +0.0033]** at five groups and the same at nine, CI
  excluding zero; gold +0.0031 [-0.0004, +0.0074].
- Its nine-group gain is +0.0079, still large despite training on random windows, and its largest
  per-study prediction change between reads was 0.42-0.78 on every fold.

**Submission D, scored offline** (rows and labels checked identical across all six members):

| set, all at nine groups | pooled OOF | gold |
|---|---|---|
| Submission C: c2+c3+f1+s2 (scored 0.925) | 0.8754 | 0.9046 |
| + t2 | 0.8781 | 0.9061 |
| **+ a2 = Submission D** | **0.8799** | **0.9078** |

Against Submission C: **+0.0045 [+0.0039, +0.0051] pooled OOF**, CI excluding zero; gold +0.0033
[-0.0007, +0.0077], spanning zero as every four-to-six-member gold delta has.

### Pre-registered band for Submission D, written before it is pushed (2026-09-21)

Six members, thirty models, all at nine groups. **What is known.** Two OOF -> LB pairs at this
resolution: B 0.8713 -> 0.921 and C 0.8754 -> 0.925, i.e. **the last +0.0041 of pooled OOF moved the
LB +0.004, a full-rate transfer.** D is +0.0045 over C on OOF [+0.0039, +0.0051] and +0.0033 on gold.
Offset method: gold 0.9078 plus the last two offsets (+0.015, +0.020) gives 0.923-0.928; delta method
at C's rate gives 0.929-0.930.

| LB | reading |
|---|---|
| **0.921 or below** | **broken.** D is C plus two members each measured at +0.0028 with CIs excluding zero; four points under C is outside both tiers. Check the log's member-skip and fallback counts, and that a2 loaded with the group-token head |
| 0.922 - 0.924 | the gains did not transfer: the two new members correlate with the old four more on the hidden set than on our folds |
| 0.925 - 0.928 | partial transfer, the offset method's range |
| **0.929 - 0.931** | **expected**, the delta method at the rate C transferred |
| 0.932 or above | better than both methods. Plausible rather than suspicious here -- random windows are an augmentation, so it is the one change that could help *more* on a hidden set than on our own folds -- but check the fallback count before believing it |

**What it is not.** Even the top of that band is below the 0.936 fork spike (315 teams) and well below
0.942. The levers left in the phase -- random windows on c3 and f1 (Screen D, running), the blend
target on more members, seeds of the confirmed six -- are worth perhaps +0.005 to +0.010 more on OOF.

Dry run: all six members load with their own head, slots, read size and group count (the four already
scored in C, plus t2 and a2 checked end to end on the local sample -- 7 GB of RAM cannot hold thirty
models, so the four were dropped for that pass rather than re-validated). 0 members skipped, header
and range checks pass. Expected inference: C ran 20 models at 8.06 s/study, so thirty at ~10.5 s/study
is ~3.8 h over ~1,300 hidden studies, inside the 6 h budget.

### Screen D: the blend target wins on 863 studies, random windows do not carry, and a better model is not a better member (2026-09-21)

`knee-phase8-screen-windows` (5.68 h, fold 0, prep9 artifacts, every arm trained on five of nine
windows drawn at random, gold held out).

| arm | val (5 / 9 groups) | gold (5 / 9) | loss | min |
|---|---|---|---|---|
| d1 c3 + random windows | 0.8536 / 0.8582 | 0.8768 / 0.8846 | 0.4703 | 20.3 |
| d2 f1 + random windows | 0.8589 / 0.8628 | 0.8802 / 0.8811 | 0.4620 | 57.4 |
| **d3 a2's config on the blend** | **0.8829 / 0.8884** | **0.8946 / 0.9012** | 0.3532 | 46.8 |
| d4 a2's config on steven_v4 | 0.8718 / 0.8779 | 0.8898 / 0.8985 | 0.4206 | 45.0 |

Read by the rules registered before the push:

- **The target question is settled where gold could not settle it: d3 minus d4 is +0.0111 [+0.0059,
  +0.0165] val**, same config, same session, CI excluding zero -- and measured against **steven_v4's own
  binarised labels**, the control's home turf. Gold agrees in direction (+0.0049) and spans zero at n=58,
  as it has every time. **The three-source blend is the training target for every later confirm.** The
  fixed truth stays steven_v4 binarised, per the amendment registered before this ran.
- **Random windows do not carry to the other families.** d1 minus c3's own confirm fold-0 is -0.0016
  [-0.0073, +0.0046] -- nothing for the two-slot mean-pool DINOv2. d2 minus f1's is +0.0073 [+0.0019,
  +0.0123], a real gain for ConvNeXt, and its training loss rose (0.4620 against f1's ~0.39), so the
  augmentation bit.
- **But a better model is not a better member, measured offline on the same fold:** d2 **replacing** f1
  inside the six gives **+0.0002 [-0.0007, +0.0010]** and added as a seventh **+0.0000**. Its +0.0073
  of extra accuracy is entirely in errors the other five members already cover. **d2 does not get its
  ~7 h confirm.** This is the cleanest example yet of why members are judged by marginal gain and never
  by their own macro -- c3 taught the same lesson from the other direction in Phase 6 (worst single
  model, best partner).
- **d3 is the new best config anywhere on this project**: fold-0 val 0.8829 at five groups and 0.8884 at
  nine, against a2's 0.8718 / 0.8779 and c2's 0.8571. As a **seventh member it adds +0.0025 [+0.0019,
  +0.0032]**, and replacing a2 it adds +0.0020 -- so it joins rather than replaces. **Confirmed next.**

**Queue for the next quota window (Sat 00:00 UTC):** Submission D (built, banded, awaiting the user's
confirmation), then d3's five-fold confirm (~5.7 h, both reads in one session), then seeds of the best
configs, then Submission E.

**d3's confirm kernel is built but not yet dry-run (2026-09-21).** The dev box is out of memory: swap
is 6.3 of 7.3 GB used and `earlyoom` kills python at ~1.4 GB RSS, so even the 64px batch-1 patch dies
at exit 143 (twice, also with `OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2`). The kernel is two swaps away
from two kernels that have both run on Kaggle -- the a2 confirm (same gate, folds, random windows,
dual read, array naming) and the t2 confirm (same blend-target training frame) -- and it cannot be
pushed before the quota reset anyway, so the dry run is deferred rather than skipped. **It runs before
the push.**

### Phase 9 opens: route changed to "everything public that is licence-compliant" (2026-10-01)

**LB pulled 2026-10-01, 4,736 teams.** Submission C 0.925 is rank ~2,360 (median 0.924). 1st 0.961, 2nd
0.960, six at 0.958, eleven at 0.957. **717 teams sit at exactly 0.943** (the fork stack), 300 at 0.941,
277 at 0.936. Our clean ensemble extrapolates to ~0.935-0.940, so it cannot reach the 0.943 spike alone.

**Decision (user, 2026-10-01):** use public weights, filtered for licence compliance, and blend our
ensemble into them. This supersedes the "none of it is usable" line of 2026-09-13, which was written
under a top-10/Efficiency mandate. Plan: `ai/plans/2026-10-01-phase9-plan.md`; queue state:
`ai/plans/phase9-queue.md`.

**Licence audit of the 0.943 public pipeline** (`jiweiliu/rsna-knee-fast-2xt4-inference`, same code as
`yamadan96/rsna-knee-d4-public0946`; dataset licences from `kaggle datasets metadata`):

| component | dataset licence | upstream | verdict |
|---|---|---|---|
| Raptor CoAtNet x4 views (`dreaddevelopment/raptor-knee-*`) | CC0 | timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, Apache-2.0 (model card) | **in** |
| CoAt readers: resgated, D4, global96 (`mattiaangeli/rsna-knee-coat*`) | CC0 | same timm CoAtNet family | **in** (confirm repair-v1 source) |
| DINOv2-small x20 (`pilkwang/rsna-knee-weights`) | CC0 | Meta DINOv2, Apache-2.0 | **in** |
| A5 "DINOv3 member" (`mattiaangeli/knee-mri-fold-weights`) | CC0 label, **DINOv3 upstream** | Meta DINOv3 licence (derivatives must carry it) | **out** |
| RadImageNet heads (`antoinegg1/*`, `marwanmath/resnet-50-radimagenet-marwan`) | CC-BY-NC-SA-4.0 | RadImageNet CC-BY-NC-SA | **out** |
| `prvsiyan/rsna-knee-v52-radimagenet-heads-*` | "other" | RadImageNet | **out** |

A dataset's own CC0 label does not clear a derivative of a restrictively licensed backbone, which is why
A5 is excluded despite its label (identified from the notebook author's own text: "the public DINOv3
member from `knee-mri-fold-weights`"). The checkpoint was not downloaded or unpickled: it is
third-party pickled data, and nothing in the audit needs it opened.

**Blending rule:** equal weight between the public compliant stack and our ensemble, per-label rank
mean. Other teams' weights saw our validation studies, so no OOF exists for them on our folds; any
fitted weight would be tuned on 390 public studies (+/-0.003 noise).

### Phase 9: the compliant public stack (P) is built; band registered before it is pushed (2026-10-01)

`notebooks/phase9-public-stack/` is the 0.943 public pipeline (`jiweiliu/rsna-knee-fast-2xt4-inference`)
with Stage 2 (A5, DINOv3-derived) and Stage 3 (RadImageNet heads) cut out, built by
`ai/scratch/phase9-public/build_p.py` as asserted text swaps (a changed parent fails the build). What
is left: DINOv2-small x20, Raptor CoAtNet x4 views, four CoAtNet readers, the parent's own 60/40
CoAt/Raptor-vs-base outer blend. Static check: pyflakes finds no undefined name the parent did not
already have, and the stage-file hand-off (DINO writes `_pipeline_stage.csv`, the Raptor/CoAt stage
reads it) is intact. It cannot be run locally (CoAtNet-384 x several, 7 GB RAM), so its first run is
the Kaggle GPU kernel itself.

`notebooks/phase9-submit-blend/` runs P, then our Submission D code unchanged as a subprocess, then a
per-label rank mean at equal weight (`ai/scratch/phase9-public/build_blend.py`; blend cell tested on
synthetic files, including the "ours failed, submit P alone" path and a study-set-mismatch raise).

**Pre-registered band for P alone.** The full stack scores 0.943; P removes the DINOv3 member and the
RadImageNet heads, which fed the base half of the parent's outer blend, and leaves the CoAt half at its
fixed 0.60 weight, tuned by others for the full stack.

| LB | reading |
|---|---|
| 0.925 or below | broken or the cut members carried the stack. Check the log first: DINO 20/20 gate, CoAt readers that ran, fallback count |
| 0.926 - 0.932 | the compliant core is much weaker than the full stack; the blend with ours matters more |
| **0.933 - 0.941** | **expected** |
| 0.942 - 0.943 | the cut members carried nothing |
| 0.944 or above | implausible for a strict subset; check for a leak or a mis-scored file |

### Submission D scored 0.926, the "partial transfer" row; our own ensemble has stopped moving the LB (2026-10-01)

**0.926** (submission 56754187, six members, thirty models, 10.2 s/study on the visible run, 0 fallbacks,
0 members skipped). Registered band: 0.925-0.928 "partial transfer, the offset method's range"; the
expected row was 0.929-0.931.

- **+0.001 on the LB for +0.0045 pooled OOF** [+0.0039, +0.0051]: about a fifth of the OOF delta. C
  transferred in full (+0.004 for +0.0041) and B at ~70%, so the rate is falling as members are added to
  an ensemble that already carries the same signal. Gold -> LB offset is +0.018 (0.9078 -> 0.926).
- **Reading:** the two new members (t2, a2) correlate with the old four more on the public ~390 than on
  our folds, which is the band's own 0.922-0.924 explanation applied weakly. The pooled OOF is still the
  right metric for membership; it is not a forecast of the LB at this density.
- **Consequence for the plan:** more members and seeds of our own ensemble are worth perhaps +0.001-0.002
  on the LB each. The gap to 0.943 cannot come from there. Phase 9's weight shifts to the blend with
  the compliant public stack, where the models are different in kind (CoAtNet, different labels,
  different pixel pipeline), and to whatever P itself scores.

### d3 confirmed: the best single member yet, and a seventh member above the bar (2026-10-01)

`knee-phase8-confirm-d3` (4.5 h training, five folds of `primary_v3`, a2's config -- group attention, five of
nine windows at random -- trained on the three-source blend, scored against the fixed truth steven_v4
binarised, gold excluded from every split). Both reads in one session.

| | pooled OOF (n=4,349) | gold (n=58) |
|---|---|---|
| d3, 5 groups | 0.8710 | 0.9078 |
| **d3, 9 groups** | **0.8781** | **0.9106** |

- **Best single member on pooled OOF** (a2 0.8704, t2 0.8654, s2 0.8622 at nine groups).
- **d3 minus a2 (the blend target alone): +0.0077 [+0.0051, +0.0102]** at nine groups, CI excluding zero,
  measured on steven_v4's own binarised labels -- a2's training target. Screen D's single-fold +0.0111
  held over five folds.
- **Added to the six members: +0.0021 [+0.0019, +0.0024] at nine groups**, over the +0.0015 bar with the CI
  above zero; gold +0.0014 [-0.0013, +0.0048], spanning zero as every member's gold delta has.
  **Promoted as the seventh member.** Nine groups minus five: +0.0071 [+0.0059, +0.0083].
- The gain per added member is shrinking (+0.0045 for t2+a2 together, +0.0021 for d3) and Submission D
  showed what it is worth on the public LB (+0.001). d3 enters the next submission as a member, not as a
  forecast of a jump.

Checkpoints and arrays in `checkpoints/knee-phase8-confirm-d3/`; six rows appended to
`results/experiments.csv`.

### P scored 0.938, inside its band; blend-e's band registered before it is submitted (2026-10-01)

**P, the licence-compliant public stack alone: 0.938** (submission 56754867; DINOv2 x20 + Raptor x4 views +
four CoAtNet readers, RadImageNet and the DINOv3-derived A5 removed). Registered band 0.933-0.941
"expected"; the full 0.943 stack loses about 0.005 for the cut members. P is **0.012 above Submission D
(0.926)**, our best own-ensemble score, and costs 275 s on the visible run. Rank on the 2026-10-01
board: ~1,760 (0.938), behind 277 teams at 0.936... i.e. past the 0.936 spike, below the 0.943 one.

**blend-e** (`phase9-submit-blend-e`) = P and Submission E (D + d3, seven members, 35 models) at equal weight,
per-label rank mean. Log verified: 35 models, 0 fallbacks, 0 skipped, public half COMPLETE. The two stacks
are different in kind (CoAtNet at 384 on public labels versus six-slot EfficientNet/DINOv2/ConvNeXt on a
three-source blend), but equal weight gives the weaker stack (0.926) half the vote.

| LB | reading |
|---|---|
| **0.937 or below** | the blend does not beat P: our stack costs more than it adds at 0.5. Next probe is a 0.25 weight, not a retreat to P alone without one more look |
| 0.938 - 0.940 | the stacks are largely redundant; ours adds at most +0.002 |
| **0.941 - 0.945** | **expected**: two decorrelated stacks, rank mean, one of them 0.012 weaker, typically +0.003 to +0.007 over the better one |
| 0.946 - 0.949 | better than the full public stack (0.943-0.946); check the blend is not leaking public-LB structure |
| 0.950 or above | implausible from one equal-weight blend; check the file before believing it |

### Pre-registered rules for Screen E (our own CoAtNet), written before it is pushed (2026-10-01)

**Why this screen.** The compliant public stack scores 0.938 on CoAtNet (384px, public labels) and our own
seven-member ensemble 0.926-0.927. Capacity and resolution screens closed for our other families, but
CoAtNet was never tried here, and it is the family the public stack's lead rests on. `knee-phase9-screen-coatnet`:
fold 0 of `primary_v3`, gold held out, prep9 artifacts, five of nine windows at random, read at five and
nine, target the three-source blend, scored against the fixed truth steven_v4 binarised. Backbone
`coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k` (timm, Apache-2.0; 41.0 M parameters; forward and backward
checked locally under `group_attention`), batch 4 x 2 accumulated (the memory-safe choice; it is a
confound the readout states). Arms: **e1** Adam 1e-4 / OneCycle 3e-4 (c2's recipe), **e2** AdamW
differential 5e-5 backbone / 1e-3 head.

- **Member rule (as in every earlier screen):** an arm added as an eighth member to the seven confirmed
  ones (c2, c3, f1, s2, t2, a2, d3 fold-0 predictions) clears **+0.0015 with the paired CI above zero**
  on the fixed truth. Passing buys a five-fold confirm, not a place in a submission.
- **Replacement read:** each arm minus d3's own confirm fold-0 predictions, paired val CI.
- **Read of the result:** CoAtNet-1 at 224 is smaller than the public stack's CoAtNet-2 at 384, so a null
  here does not say CoAtNet cannot help; it says this size at this resolution on this label target does
  not. A null closes only this arm; the next step then is the larger variant, not abandoning the family.
- Time budget 9 h; peak memory and minutes per epoch printed per arm.

**Screen E, first pushes (2026-10-01).** Version 1 died in its first cell (my header edit dropped the
imports; no GPU spent). Version 2 ran the full setup, then **ran out of memory in the first forward at
batch 4** (CoAtNet-1 over ~100 images per step with no gradient checkpointing, 14.5 of 14.56 GiB in use).
Version 3 trains at **batch 1 x 8 accumulated** (the optimizer still steps on 8 studies); BatchNorm in the
conv stages sees the ~30 images of one study per call, as it does at batch 4. Gradient checkpointing was
not used because the conv stages carry BatchNorm and a recomputed forward updates its running statistics
twice. Rules unchanged.

### d3 at 336px confirmed: the best model yet, and an eighth member above the bar (2026-10-02)

`knee-phase9-confirm-d3-336` (7.9 h, five folds of `primary_v3`, d3's config at 336px, batch 4 x 2
accumulated as s2 was; target the three-source blend, scored against steven_v4 binarised, gold excluded).

| | pooled OOF (n=4,349) | gold (n=58) |
|---|---|---|
| d3@336, 5 groups | 0.8755 | 0.9074 |
| **d3@336, 9 groups** | **0.8823** | **0.9120** |

The best single model on both tiers (d3 at 224: 0.8781 / 0.9106; a2 0.8704 / 0.9067). Scored offline
(`ai/scratch/phase9-offline/score8.py`, all nine-group arrays, rows verified identical):

- **d3@336 added to the seven: +0.0022 [+0.0020, +0.0025]** pooled OOF (0.8820 -> 0.8842), CI excluding
  zero, over the +0.0015 bar; gold +0.0022 [-0.0012, +0.0060], spanning zero. **Promoted as the eighth
  member.**
- d3@336 *replacing* d3: +0.0008 [+0.0004, +0.0011], so both stay.
- 336px on the blend recipe: d3@336 minus d3 is about +0.004 at nine groups on pooled OOF, against
  +0.0053 for 336 on c2's recipe (s2): the resolution gain survives the better target and recipe, smaller.
- Cost: 90 min per fold against 51 at 224. Eight members is ~40 models; inference ~13 s/study.

Six rows appended to `results/experiments.csv`.

### Screen E: our own CoAtNet-1 clears the member bar by a hair, and costs 2.7x an EfficientNet epoch (2026-10-02)

`knee-phase9-screen-coatnet` (v3, 6.1 h, fold 0, batch 1 x 8; peak GPU memory 5.5 GB; 138.5 min per arm
against ~47 for d3). Backbone `coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k` on d3's recipe.

| arm | recipe | val (5 / 9 groups) | gold (5 / 9) | loss |
|---|---|---|---|---|
| e1 | Adam 1e-4 / OneCycle 3e-4 | 0.8730 / 0.8752 | 0.8941 / 0.9002 | 0.3701 |
| e2 | AdamW 5e-5 backbone / 1e-3 head | 0.8710 / 0.8751 | 0.8901 / 0.8996 | 0.3619 |

Read by the rules registered before it ran:

- **Replacement: no.** e1 minus d3's own fold-0 confirm: -0.0080 [-0.0140, -0.0016]; e2: -0.0100
  [-0.0166, -0.0038]. A 41 M-parameter CoAtNet at 224 is a weaker model than a 5 M EfficientNet-B0 at 224 on
  this data and target, with both recipes.
- **Eighth member (five groups, the pre-registered read): e1 +0.0016 [+0.0010, +0.0024], e2 +0.0018
  [+0.0010, +0.0025]** -- over the +0.0015 bar with the CI above zero, by 0.0001 and 0.0003. At nine groups
  +0.0014 and +0.0018. Gold spans zero for both.
- **Not confirmed (a judgement, stated so it is not mistaken for the rule):** the rule permits a five-fold
  confirm and these arms qualify, but the bar is passed by less than the +0.0036 two runs of one config
  differed by in Screen A, the confirm would cost ~11.5 h of GPU, and Submission D just showed what
  +0.0045 of pooled OOF is worth on the LB (+0.001). The public stack's CoAtNet-2 at 384 is a different
  object. The recipe sensitivity is small (e1 and e2 within 0.002), so the backbone, not the optimizer, is
  the limit.
- Two rows appended to `results/experiments.csv`.

### blend-e failed on the hidden set: "Notebook Timeout"; the runtime budget is now a design constraint (2026-10-02)

Submission 56759988 (P + seven-member ensemble, 35 models, equal-weight rank mean) came back with no
score. The submission's `error_description`, readable through the Python API (`competition_submissions`)
and on the My Submissions page, says "Your submission notebook exceeded the allowed runtime ... the hidden
dataset can be larger/smaller/different than the public". The CLI table shows only an empty score next to
`COMPLETE`, which is why it looked like a stuck scoring job for hours; the Python record is the way to read
a failed submission. It costs a Kaggle submission slot, not a quota.

**What the timings say.** The visible run was clean (35 models, 6.31 s/study, 0 fallbacks) and said nothing
about the hidden run. Wall times from submission to score on the hidden set: **D (30 models) ~2.6-3.3 h;
P alone ~5.5-6.2 h** (pending at 21:49 UTC, scored by 22:29 UTC after a 16:18 submission, with no other
hidden run in the queue ahead of it). P is therefore the expensive half -- its "under 30 minutes" is a
claim for a different test size -- and P plus ours at 35 models is ~9 h or more, over the 9 h limit.
Submission F (40 models, two of them 336px) would time out the same way; **blend-f is withheld.**

**Rule from here:** a blend's budget is P's ~6 h plus ours, with at least 1 h of slack under 9 h, so ours
may cost at most ~2 h, which is roughly 15 models at 224px (a visible-run forward of ~0.14 s/model/study
plus ~1.5 s/study of shared prep, scaled to ~1,300 studies). Ours is therefore chosen by pooled OOF at a
fixed model budget, not by membership of the full ensemble.

### Fewer members is better: d3 + d3@336 beats all eight; G and blend-g, with bands registered before they are submitted (2026-10-02)

Pooled OOF at nine groups (`ai/scratch/phase9-offline/subsets.py`, exhaustive over subsets of the eight
confirmed members, rows verified identical):

| set | models | pooled OOF | gold |
|---|---|---|---|
| all eight (equal mean) | 40 | 0.8842 | 0.9114 |
| **d3 + d3@336** | **10** | **0.8874** | **0.9134** |
| t2 + d3 + d3@336 | 15 | 0.8874 | 0.9106 |
| t2 + a2 + d3 + d3@336 | 20 | 0.8867 | 0.9118 |
| Submission D's six | 30 | 0.8799 | 0.9078 |

**The two blend-target group-attention models alone are better than the full equal-weight mean**: the older
members (c2, c3, f1, s2, which trained on one label source and a fixed or mean-pooled head) dilute them.
The selection is among 247 subsets, so its optimism is small at n=4,349 but not zero; the two-member set
is also the one both members' own confirms point to (+0.0077 and +0.0119 over a2 on the blend target). It
fits the runtime budget set by blend-e's timeout: ~13 model-units against ~46.

**Submission G = d3 + d3@336, 10 models** (`phase9-submit-g`), and **blend-g = P + G**, equal weight
(`phase9-submit-blend-g`). Visible runs pushed 2026-10-02 08:49 UTC.

**Band for G alone.** D (six members, pooled OOF 0.8799) scored 0.926 and moved the LB a fifth of its OOF
delta. G is +0.0075 over D on pooled OOF and +0.0056 on gold, with a different composition (two models, one
label source family), so the transfer rate is not the last one's by assumption.

| LB | reading |
|---|---|
| 0.925 or below | the dilution result does not transfer: the older members were helping on the hidden set |
| 0.926 - 0.928 | flat against D; our own ensemble is saturated on the public LB whatever its OOF |
| **0.929 - 0.933** | **expected**: a fifth to a half of +0.0075, on top of D's 0.926 |
| 0.934 - 0.937 | better than any plausible transfer; check the file |
| 0.938 or above | implausible for ten EfficientNet-B0 models; check for a leak |

**Band for blend-g (P 0.938 + G, equal weight).** Expected runtime: P ~5.5-6.2 h + G ~1.3-1.5 h, inside 9 h
with ~1.3 h of slack; the failure mode is the same timeout again.

| LB | reading |
|---|---|
| timeout / no score | P is slower than its 6 h estimate or ours more than 2 h; cut ours to d3@336 alone or drop a P reader |
| 0.937 or below | the blend does not beat P: the two stacks are not decorrelated enough on the hidden set |
| 0.938 - 0.940 | at most +0.002 from ours |
| **0.941 - 0.945** | **expected**: two decorrelated stacks, +0.003 to +0.007 over the better |
| 0.946 or above | better than the full public stack; check the blend for leakage of public-LB structure |

### Submission G scored 0.928: +0.002 for +0.0075 of pooled OOF, the "saturated" row (2026-10-02)

**0.928** (submission 56771279; d3 + d3@336, ten models, 4.36 s/study visible, 0 fallbacks). Band: 0.926-0.928
"flat against D; our own ensemble is saturated on the public LB whatever its OOF"; the expected row was
0.929-0.933.

| submission | members | pooled OOF (9 groups) | LB |
|---|---|---|---|
| B | 3 | 0.8713 | 0.921 |
| C | 4 | 0.8754 | 0.925 |
| D | 6 | 0.8799 | 0.926 |
| **G** | **2** | **0.8874** | **0.928** |

- **The transfer rate from pooled OOF to the LB is falling steadily**: +0.004 for +0.0041 (C over B), +0.001
  for +0.0045 (D over C), +0.002 for +0.0075 (G over D) -- about 100%, 20%, 27%. The dilution result was
  real on OOF and gave the LB a little (the ten-model set beats the thirty-model one), but the order of
  magnitude says: **the public LB no longer separates our own variants by more than +/-0.002**, which is
  also the noise from the ~390-study public subset.
- **Gold -> LB offset +0.017** (0.9134 -> 0.928), inside the 0.015-0.020 range of the last five pairs.
- **Consequence:** no further own-model work is likely to move the LB by more than a few thousandths on
  its own. What can move it is the blend with a different kind of model: P at 0.938 is 0.010 above our best,
  and blend-g (P + G) is the test of whether our half adds anything to it. If blend-g reads at or below
  0.938, ours adds nothing on the hidden set at equal weight and the remaining effort goes to the public
  side (more compliant members, weights probed on at most two further reads), not to our own members.

### blend-g timed out too: P's hidden-set runtime is the unknown, so the blend now carries a deadline (2026-10-02)

Submission 56771289 (P + G, ten of our models) also failed with "Notebook Timeout", where my estimate said
~7.5-8 h of 9. **My runtime model was wrong, and the error is in P's term.** The visible run (3 studies)
takes the public pipeline's small-cohort path (readers overlapped in parallel pairs below 48 studies) and
says nothing about the hidden cohort, which "runs the readers one at a time"; "pending" time from
submission to score also includes the rerun queue, so it was never a measure of P's runtime. P alone
fits in 9 h (it scored 0.938) and P + ten models of ours does not.

**Change:** the blend kernel (`build_blend.py`, now with a deadline argument, default 7.9 h) records
`BLEND_T0` in its first cell and, after the public stack finishes, gives our ensemble only the time left
before the deadline; `subprocess.run(timeout=...)` kills it at the deadline and the blend cell then
submits the public stack's own `submission.csv` unchanged. So a slow hidden run now costs the blend, not
the submission: the worst case is P alone (0.938), not a timeout. The outcome of the next blend submission
therefore reads as: **0.938 = our half did not fit** (P's hidden run is over ~6-6.5 h); **above 0.938 =
it fit and the blend is worth the reading**. Kill path tested locally with a sleeping child.

Banded as before, plus that first row: blend-g v2: 0.938 means the deadline fired (no information about
our half, one information about P's runtime); 0.941-0.945 expected if it fit.

### Survey of 15 newer public label sets: five are contaminated, none beats the target we have (2026-10-03)

Scored on the 58 gold studies with a contamination check (`ai/scratch/phase9-labels/score_gold.py`: macro AUC,
exact agreement of binarised labels with gold, and AUC against the current three-source blend on the studies
both cover). Reference: steven_v4 gold 0.8927, ours 0.8616, lixin73 0.8488, the old five-way rank blend 0.8976.

- **Contaminated (gold macro 1.0000, agreement 1.0000 on the 58): `rayanbabur/...calibrated-soft-targets`
  (hard and soft), `tasmeemreza/...refined-llm-labels`, `yehezkielhaganta/psuedo-labeling-knee` (data, train).**
  Their gold rows are copied from `train.csv`, as `yunusgmsoy/...4-source-merged` was. Rejected: a score of
  1.0000 on gold says nothing about the other 4,349 rows.
- **Clean and scored:** `riadmohamed42/jev-knee-labels` HYBRID_scores **0.9025** (agreement 0.81, like
  steven's 0.82, so not copied; AUC against our target 0.997, i.e. it is largely made of the sources we
  already blend); `ctogaurav/...v4-blend` 0.8927 (the same file as steven_v4, AUC against the target 0.9996);
  JEV FINAL2 scores 0.8745; `laymond` Qwen weak labels 0.8295 (the `_v2` files 0.42: unusable).
- **Not scoreable on gold (the file omits the 58 gold rows):** `dreaddevelopment/rsna-knee-labels` (the labels the
  public Raptor CoAtNets train on; AUC against our target 0.93, so it is a genuinely different reading),
  `nartaa` v0/v1 (0.9996 / 0.997 against our target, i.e. the same sources), `gperdrizet` (CC-BY-NC-SA,
  excluded by licence), `vmohitrao` (CC BY-NC, excluded).
- **Licences of the clean candidates:** HYBRID, dreaddevelopment, ctogaurav, nartaa CC0; saisahanas Apache-2.0.
- **Reading:** the one clean set above steven_v4 on gold, HYBRID (+0.0098, n=58, a CI of about +/-0.04), is
  indistinguishable from it at this n and is 0.997 correlated with the target we already train on, so
  swapping it in would not change the target. Nothing here justifies a retrain. The only different
  reading is dreaddevelopment's, which cannot be validated on gold; it is the one remaining label
  experiment with a plausible edge (a fourth source in the blend), and it is a screen, not an assumption.

### Pre-registered rules for Screen F (a fourth label source), written before it is pushed (2026-10-03)

`knee-phase9-screen-labels`: fold 0 of `primary_v3`, gold held out, prep9 artifacts, d3's config (group
attention, five of nine random windows, EfficientNet-B0, Adam 1e-4 / OneCycle 3e-4, batch 8), read at five
and nine groups, scored against the fixed truth steven_v4 binarised. **f1** trains on the probability mean of
steven_v4, ours, lixin73 and `dreaddevelopment/rsna-knee-labels` (CC0, soft labels, 4,349 studies); **f2**
on `dreaddevelopment` alone. The control is d3's own fold-0 confirm predictions (Screen D licensed pairing
against a confirm: the same config on the same artifacts reproduces its fold 0).

- **The target question is read as Screen D read it: f1 minus d3's fold-0 val delta with a CI above zero
  promotes the four-source blend as the target for later confirms**, measured on steven_v4's own
  binarised labels (the control's home turf, as every earlier target comparison was). Gold (n=58) cannot
  resolve it and is reported only.
- **If the delta's CI spans zero:** the arm is judged as a member only: added as a ninth member to the eight
  confirmed (c2, c3, f1, s2, t2, a2, d3, d3@336), **+0.0015 with the paired CI above zero** -> a five-fold
  confirm as a member config, not a target change.
- **f2 is information, not a candidate:** if f2 is far below d3 the source is weak alone and its value in
  f1 is a blend effect; if it matches d3 the public Raptor labels are as good as ours.
- A null closes the label lever for Phase 9: three independent source sets (and now a fourth) have been
  tried, and the survey found no set that beats steven_v4 on gold.

### Screen F: a fourth label source changes nothing; the label lever is closed (2026-10-03)

`knee-phase9-screen-labels` (3.7 h wall, fold 0, d3's config, prep9 artifacts, gold held out; 53 and 50 min of
training per arm; the rest was session start-up and the data mount).

| arm | target | val (5 / 9 groups) | gold (5 / 9) | loss |
|---|---|---|---|---|
| f1 | mean of steven_v4, ours, lixin73 and dreaddevelopment | 0.8794 / 0.8854 | 0.9045 / 0.9169 | 0.3790 |
| f2 | dreaddevelopment alone | 0.8590 / 0.8657 | 0.8996 / 0.9074 | 0.4137 |

(d3's own fold-0 confirm: 0.8810 / 0.8852 val, 0.9008 / 0.9105 gold.)

Read by the rules registered before it ran:

- **Target: not promoted.** f1 minus d3 on val: **-0.0015 [-0.0064, +0.0032]**, CI spanning zero (gold +0.0037
  [-0.0119, +0.0205], also zero). Adding the fourth source neither helps nor hurts.
- **Member: not confirmed.** f1 as a ninth member +0.0009 [+0.0004, +0.0014], under the +0.0015 bar; f2 +0.0006
  [+0.0001, +0.0011] on val (gold +0.0064 [+0.0023, +0.0115], which at n=58 is not a basis for a ~5 h confirm).
- **f2: the source is weak alone**: -0.0220 [-0.0297, -0.0153] against d3 on val, so the public Raptor labels are
  worth less as a sole target than the three-source blend, and add nothing to it.
- **The label lever is closed for Phase 9.** The survey found no set above steven_v4 on gold (five were
  contaminated), the one different reading (dreaddevelopment's) adds nothing as a fourth source, and the
  three-source blend stays the target. Two rows appended to `results/experiments.csv`.

### blend-g v2 scored 0.938: the deadline fired, P's hidden run leaves our half less than ~1.6 h (2026-10-03)

**0.938** (submission 56785577; P + G with the 7.9 h deadline watchdog), identical to P alone (0.938). Registered
reading: "0.938 = the deadline fired". A real blend of two stacks 0.010 apart would not land on P's score to
three decimals, and the submission completed, so the watchdog worked: it killed (or skipped) our half and
shipped P's own file. This is also the first observation of a failure mode turning into a result instead of a
timeout, which is what it was built for.

**What it says about P.** Our ten-model G needs ~1.6 h on the hidden set (4.4 s/study visible; D's thirty models
scored from a 2.6-3.3 h wall time). It did not fit in the time P left, so **P's hidden run is over ~6.3 h
(7.9 h less G's 1.6 h) and at most ~8.1 h** (it finished, with its own start-up, inside the 9 h limit). P is
the expensive half by a wide margin and P alone has little slack against the limit.

**Next probe.** Our half shrunk to d3 alone (five models at 224px, best single model by pooled OOF
0.8781 at nine groups; ~0.7 h on the hidden set: ~0.5 h of shared prep, ~0.2 h of forwards), with the
deadline moved to 8.0 h. If P finishes by ~7.3 h the blend runs. Reading, registered now:
**0.938 again = P needs more than ~7.3 h, the blend is out of reach, and P itself is the final candidate with
a note that its margin under 9 h is thin.** A score above 0.938 = the blend fit; expected 0.940-0.944 (one
EfficientNet member, the best, at half weight next to a 0.938 stack).

### blend-h scored 0.936, below P: the blend ran, and our half at equal weight costs 0.002 (2026-10-04)

**0.936** (submission 56809803; P + d3 alone, five models, equal-weight per-label rank mean, 8.0 h deadline).
It is not P alone (0.938), so this time the blend ran: **P's hidden run finished with at least the ~0.7 h
that d3 needs left before the 8.0 h deadline, i.e. P takes under ~7.3 h**, and the earlier 0.938 for P + G
was the deadline firing on G's 1.6 h. P's hidden-set runtime is therefore between ~6.3 h and ~7.3 h.

Band row hit: "0.937 or below = the blend does not beat P". **A rank mean at half weight of a 0.938 stack and
a ~0.926-0.928 stack scores below the better one**, by 0.002: our models are not a decorrelated strengthener
here, they are a weaker voice. That matches the arithmetic of the last week (own variants move the LB by
+/-0.002; P is 0.010 ahead of the best of them), and it means the blend as designed cannot be the way up.

**What remains of the blend idea** is the weight: at 0.2 or less the weaker stack's votes mostly break ties,
which is the only regime in which two stacks 0.010 apart can both help. One probe at 0.2 (submission
blend-i) is worth its slot; nothing is expected beyond +0.001-0.002. If it reads at or below 0.938, P is the
final candidate and the blend is closed.
