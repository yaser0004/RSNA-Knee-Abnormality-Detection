# Resources included with the final submissions

The hosts ruled in forum thread 744056 (RSNA, 2026-10-14) that submissions incorporating publicly shared weights are eligible
for awards, and asked participants to list the shared weights and their source datasets with each submission. This file is that
list. All dataset metadata was read from Kaggle on 2026-10-16.

The two final picks:

| pick | kernel | public LB | submission |
|---|---|---|---|
| 1 | `mohammadyaserhussain/knee-phase9-submit-blend-i` (public stack P plus our d3 at weight 0.2) | 0.939 | 56819611 |
| 2 | `mohammadyaserhussain/knee-phase11-submit-gs25` (25 models trained by us) | 0.928 | 57233603 |

## Pick 1: P + d3 (blend-i)

Per-label rank mean, weight 0.8 for the public stack P and 0.2 for our d3 ensemble. A deadline watchdog at 8.0 h keeps the
blend inside the 9 h limit (P alone runs 6.3-7.3 h on the hidden set).

### A. Weights and datasets by other participants (the shared resources)

All carry the CC0-1.0 licence label on Kaggle. Where a row names an upstream model, that is the pretrained backbone the
participant fine-tuned.

| Kaggle resource | author | what it is | upstream |
|---|---|---|---|
| `dreaddevelopment/raptor-knee-maxspan` | Dread Development (Johnathan Wagner) | CoAtNet-384 fine-tuned on this competition, max-span sampling | timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, Apache-2.0 |
| `dreaddevelopment/raptor-knee-native384` | Dread Development | CoAtNet-384, native-384 corpus | same |
| `dreaddevelopment/raptor-knee-native384dense` | Dread Development | CoAtNet-384, native-384 dense corpus | same |
| `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` | Mattia Angeli | residual-gated CoAtNet reader, top-3 checkpoints | same CoAtNet family |
| `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` | Mattia Angeli | D4 depth-zone CoAtNet reader, SWA of 3 | same |
| `mattiaangeli/rsna-knee-coatnet-global96-top3` | Mattia Angeli | Global96 CoAtNet reader, top-3 checkpoints | same |
| `pilkwang/rsna-knee-weights` | Pilkwang Kim | 20 DINOv2-small slot-attention members trained on this competition | Meta DINOv2-small, Apache-2.0 |
| `pilkwang/rsna-knee-llm-labels` | Pilkwang Kim | LLM-read report labels, used by the pipeline | n/a (labels) |
| `mattiaangeli/opencv-python-headless-4120088-x86` | Mattia Angeli | pinned OpenCV 4.12.0.88 wheel (a library, no weights) | OpenCV, Apache-2.0 |
| Kaggle model `metaresearch/dinov2/PyTorch/small/1` | Meta | DINOv2-small base weights | Apache-2.0 |

Pipeline code: the inference notebook is derived from `jiweiliu/rsna-knee-fast-2xt4-inference`, itself built on maverickss26's
public solution and Mattia Angeli's "Speedy Raptors v34" recipe (credits in its header; public notebooks are deemed licensed
under an OSI-approved licence by rule 3.6.b).

Deliberately not included: the DINOv3-derived A5 weights (`mattiaangeli/knee-mri-fold-weights`) and the RadImageNet heads
(`antoinegg1/*`, `marwanmath/resnet-50-radimagenet-marwan`, `prvsiyan/rsna-knee-v52-radimagenet-heads-*`), because their
upstream licences (Meta DINOv3 licence, CC-BY-NC-SA-4.0) conflict with a winner licence. This is the whole difference between P
(0.938) and the 0.943 public stack.

### B. Our own contribution (d3, weight 0.2)

Five efficientnet_b0 group-attention models (`knee-phase8-confirm-d3`, folds `primary_v3`), trained by us with our own code
(`source/src/knee`) from timm `efficientnet_b0.ra_in1k` (Apache-2.0) ImageNet weights. Training labels: report-derived label
blend of `stevenleehans/rsna-knee-llm-report-labels`, `pilkwang/rsna-knee-llm-labels`,
`lixin73/rsna-knee-llm-report-labels-sol56` (all CC0-1.0) and our own Qwen3-4B (Apache-2.0) report labels. Inputs: our
preprocessed slice cache built from the competition data, and `mohammadyaserhussain/rsna-knee-src` (our code, private).

## Pick 2: GS25 (own weights only)

25 efficientnet_b0 group-attention models, five folds each, equal weight, all trained by us from timm `efficientnet_b0.ra_in1k`
with our code: d3 (224 px), d3@336 (`knee-phase9-confirm-d3-336`), and student arms b, c and d (`knee-phase11-tpu-farm1`,
`knee-phase11-tpu-farm2`, trained on a Kaggle TPU).

No other participant's weights are in this entry. Shared data used for training:

- the same three public label datasets as above (`stevenleehans/...`, `pilkwang/rsna-knee-llm-labels`, `lixin73/...`, CC0-1.0);
- arms c and d add labels our own Claude (hosted LLM, permitted by host thread 733965) derived from the training reports, held
  in the private dataset `mohammadyaserhussain/rsna-knee-phase11-claude-labels`;
- the student arms use a teacher built from our own models' out-of-fold predictions
  (`mohammadyaserhussain/rsna-knee-phase11-teacher`, private).

## Open point

The winner-licence question (thread 739905: CC BY-NC 4.0 against an OSI-approved licence) is unanswered as of 2026-10-14. Every
shared weight in pick 1 is CC0 over an Apache-2.0 upstream, so it fits either answer.
