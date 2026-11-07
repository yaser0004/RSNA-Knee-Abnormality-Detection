# RSNA Knee Abnormality Detection

My solo entry for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
(August to October 2026). The competition is over and this repository is archived.

## The task

Given a knee MRI study (several DICOM series), predict how likely each of 12 findings is: ACL tear, MCL tear, medial and
lateral meniscus tear, osteoarthritis in each of three compartments, joint effusion, synovitis, Baker's cyst, bone
contusion and acute fracture. Submissions are scored by the average ROC AUC over the 12 findings.

The catch: of 4,407 training studies, only **58** have expert labels. The rest come with just the radiologist's written
report, in several languages.

## Result

**Final private leaderboard: rank 2,717, AUC 0.928.** (Public leaderboard: rank 2,900 of 6,299 at 0.943.)

| entry | what it is | public | private |
|---|---|---|---|
| Phase 1 baseline | one model, keyword labels for 4 of the 12 findings | 0.558 | 0.542 |
| Own best ensemble (G, GS25) | models trained only by me, on report-derived labels | 0.928 | 0.917 |
| Public stack + my models (blend-i) | other participants' public weights, with my ensemble blended in | 0.939 | 0.925 |
| Full public stack + my models | best private score among my submissions | 0.943 | 0.928 |

The full list of submissions is in [`DETAILS.md`](DETAILS.md).

## How it works, in short

1. **Turn reports into labels.** A language model (Qwen3-4B, run inside Kaggle; later also Claude) reads every report
   against the hosts' grading rubric and gives a probability for each finding. Several public label sets are blended in.
   The 58 expert-labelled studies are never trained on; they are used only to check how good the labels are.
2. **Preprocess every study once.** Pick the useful series, crop to the same physical size, sort them into slots by
   plane and MRI weighting, and store slice groups in compact files.
3. **Train image models on those labels.** EfficientNet, DINOv2 and ConvNeXt backbones with an attention head that
   learns which plane and slices matter for each finding, five folds each.
4. **Ensemble.** Average many models; in the last phase, blend in publicly shared weights from other participants
   (listed with their licences in [`RESOURCES.md`](RESOURCES.md)).

Labels mattered more than architecture: better report labels moved the score more than any model change.

## Repository layout

```
src/knee/     the library: DICOM reading, preprocessing, datasets, model, training, metrics, report labelling
notebooks/    Kaggle notebooks, one folder per run (training, preprocessing, submissions)
tests/        unit tests (CPU only)
results/      experiment logs, fold assignments, metadata censuses
tools/        offline helper scripts kept as a record of how some numbers were produced
```

Not included: competition data, trained weights (they live in the Kaggle notebooks' outputs) and any labels derived from
the reports (the competition rules keep those private).

## Running the tests

```
pip install -r requirements-dev.txt
pytest tests/
```

A few tests need a sample of competition data in `data/sample/`; they are skipped when it is missing.

## More detail

- [`DETAILS.md`](DETAILS.md): the full record. Every phase, every decision and why, what worked and what did not,
  code layout, and where each artifact lives.
- [`NOTES.md`](NOTES.md): the dated lab notebook with every measurement.
- [`RESOURCES.md`](RESOURCES.md): third-party weights and datasets used in the final submissions.
