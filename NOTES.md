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
