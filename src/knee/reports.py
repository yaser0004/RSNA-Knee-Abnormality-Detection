import math
import re

import numpy as np
import pandas as pd

from knee.infer import LABEL_COLUMNS

# Phase 1 baseline: multilingual keyword/regex rules over report text (Tier 3 in
# the plan's Label hierarchy). Only English and Spanish patterns are seeded here
# with confidence. The real report-language distribution (measured on a 500-report
# sample, see NOTES.md) is English 39%, Spanish 16%, Turkish 11%, Croatian 11%,
# Greek 7%, German 7%, Dutch 4% -- Turkish and Croatian coverage would meaningfully
# grow Phase 1's labeled set, but their medical negation grammar needs verification
# (native speaker or a medical dictionary) before being hard-coded into training
# labels, so those patterns are deliberately left out rather than guessed.
_POSITIVE_PATTERNS = {
    "effusion": [r"\beffusion\b", r"\bderrame\b"],
    "baker's": [r"\bbaker'?s?\s+cyst\b", r"\bquiste\s+de\s+baker\b", r"\bpopliteal\s+cyst\b"],
    "acl": [r"\bacl\b.{0,30}\b(tear|rupture|torn)\b", r"\bligamento\s+cruzado\s+anterior\b.{0,30}\b(rotura|desgarro)\b"],
    "medial meniscus": [
        r"\bmedial\s+meniscus\b.{0,40}\b(tear|torn|rupture)\b",
        r"\bmenisco\s+(medial|interno)\b.{0,40}\b(rotura|desgarro)\b",
    ],
}

_NEGATION_WORDS = [r"\bno\b", r"\bsin\b", r"\bwithout\b", r"\bausencia\s+de\b"]

_NEGATION_WINDOW_CHARS = 25


def lexical_label(report_text: str, label: str) -> int | None:
    """Return 1 if a positive pattern for `label` is found without an
    immediately preceding negation, 0 if the pattern is found but negated,
    or None if no pattern matches at all (excluded from that label's loss,
    never treated as an implicit negative -- see the plan's Label hierarchy)."""
    patterns = _POSITIVE_PATTERNS.get(label.lower())
    if not patterns:
        raise ValueError(f"no lexical rule defined for label {label!r}")

    for pattern in patterns:
        match = re.search(pattern, report_text, re.IGNORECASE)
        if match:
            window_start = max(0, match.start() - _NEGATION_WINDOW_CHARS)
            preceding_text = report_text[window_start : match.start()]
            if any(re.search(neg, preceding_text, re.IGNORECASE) for neg in _NEGATION_WORDS):
                return 0
            return 1

    return None


def build_lexical_labels(reports_df: pd.DataFrame) -> pd.DataFrame:
    """Apply lexical_label across all 12 label columns for every study. Labels
    with no rule defined (e.g. Synovitis) are NaN for every row, same as an
    unmatched rule -- both mean "no evidence", never "negative"."""
    rows = []
    for _, row in reports_df.iterrows():
        report = row["Report"]
        if not isinstance(report, str):
            # missing/NaN report -- no evidence for any label, not a crash
            rows.append([row["StudyInstanceUID"], *([np.nan] * len(LABEL_COLUMNS))])
            continue
        label_values = []
        for label in LABEL_COLUMNS:
            try:
                value = lexical_label(report, label)
            except ValueError:
                value = None
            label_values.append(np.nan if value is None else float(value))
        rows.append([row["StudyInstanceUID"], *label_values])

    return pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABEL_COLUMNS)


# Phase 3: Tier-2 LLM generator (see the plan's Label hierarchy). Runs inside a
# Kaggle GPU notebook via transformers -- no inference engine is a dependency of
# this module, so everything below is plain Python/string logic, testable
# without a GPU. The notebook wires a real model to score_report_with_llm's
# generate_fn; keeping that seam is what let the engine change from vLLM to
# transformers without touching a single one of these functions or their tests.

# Verbatim from notebooks/phase3-labels/rubric_733343.md (Kaggle competition
# discussion thread 733343, the host's own rubric post -- public forum content,
# not competition data). Embedded here rather than read from that file at
# runtime because src/knee/ ships to Kaggle as its own private dataset,
# independent of notebooks/ -- keep the two in sync by hand if the rubric
# thread is ever edited.
RUBRIC_TEXT = """For the purposes of this challenge, the knee is considered in three compartments. The medial compartment lies between the medial femoral condyle and the medial tibial plateau, on the inner side of the knee. The lateral compartment lies between the lateral femoral condyle and the lateral tibial plateau, on the outer side. The patellofemoral compartment lies between the patella and the femoral trochlea, the groove on the front of the femur. Osteoarthritis, or cartilage loss, is assessed separately in each of these three compartments.

Four ligaments stabilize the knee. The anterior cruciate ligament (ACL) and posterior cruciate ligament (PCL) lie within the joint, in the intercondylar notch, and control front-to-back stability and rotation. The medial collateral ligament (MCL) and lateral collateral ligament (LCL) run along the inner and outer sides of the knee and resist side-to-side stress. This challenge focuses on tears of the ACL and the MCL.

Between the femur and tibia sit two menisci, the medial meniscus and the lateral meniscus, C-shaped wedges of fibrocartilage that cushion the joint, absorb shock, and improve the fit between the rounded femur and the relatively flat tibia. Tears of the medial and lateral meniscus are evaluated separately.

Other relevant structures include the bone marrow within the femur, tibia, and patella, where a bruise from impact is called a bone contusion (or bone marrow edema) and a break in the bone is a fracture; and the soft tissues behind the knee, where a fluid-filled outpouching of the joint lining is called a Baker, or popliteal, cyst.

For the purpose of this challenge, a "fluid sensitive" sequence refers to one in which edema, hemorrhage, and other types of fluid appear bright and fat is suppressed in some way.

Models are evaluated on twelve binary labels, each indicating the presence or absence of a specific finding in the imaged knee. The labels, and the criteria used by the annotating radiologists, are summarized below. In each case, ambiguous or borderline findings ("on the fence") were graded as negative to favor specificity.

ACL tear: A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative.

MCL tear: A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.

Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.

Lateral meniscus tear: The same criteria applied to the lateral meniscus.

Medial compartment osteoarthritis: A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial compartment, with or without underlying subchondral marrow changes.

Lateral compartment osteoarthritis: The same criteria applied to the lateral compartment.

Patellofemoral compartment osteoarthritis: The same criteria applied to the patellofemoral compartment.

Joint effusion: A moderate or large amount of fluid distending the joint.

Synovitis: Inflammation and thickening of the synovial lining of the joint.

Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee.

Contusion: A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line.

Acute fracture: An acute cortical break or fracture line.

Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee."""

# Per-label prompting, NOT the whole rubric per question. An earlier version
# prepended all of RUBRIC_TEXT (~1,400 tokens, every one of the 12 definitions)
# to each of a report's 12 questions. That made a single report cost ~12x1,800
# tokens of prefill, which is the only reason the design needed vLLM's prefix
# caching -- and chasing a vLLM install on Kaggle burned two GPU sessions to a
# CUDA-variant mismatch (vllm-project/vllm#43435). Sending only the criterion
# being asked about drops the prompt to ~500 tokens, needs no prefix cache, and
# runs on the transformers already in the Kaggle image. It is also the better
# prompt: the model sees the one definition it is being asked to apply instead
# of eleven competing ones as distractors.
#
# RUBRIC_TEXT above is kept as the verbatim record of the source; the criteria
# below are its per-finding paragraphs, split out.

# The rubric states three of its criteria as back-references ("The same
# criteria applied to the lateral meniscus"). Those are resolved inline here so
# each criterion stands alone in its own prompt -- a criterion reading only
# "the same criteria applied to X" would carry no diagnostic content at all
# once separated from the paragraph it refers back to. Substituted wording is
# limited to the structure/compartment name; the thresholds are unchanged.
_LABEL_CRITERIA = {
    "ACL": (
        "ACL tear",
        "A high-grade partial or full-thickness tear of the anterior cruciate ligament, "
        "meaning complete discontinuity of the ligament, or more than 50 percent of fibers "
        "disrupted, with or without secondary signs such as characteristic pivot-shift bone "
        "contusions. Mild signal change, degeneration, or thickening without discontinuity is "
        "graded negative.",
    ),
    "MCL": (
        "MCL tear",
        "A high-grade partial or complete acute tear of the medial collateral ligament, with "
        "disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and "
        "chronic or remote stress changes are graded negative.",
    ),
    "Medial Meniscus": (
        "medial meniscus tear",
        "Abnormal signal that definitely contacts the meniscal surface on at least two images, "
        "or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, "
        "involving the medial meniscus. Intrasubstance degeneration that does not reach the "
        "surface is negative.",
    ),
    "Lateral Meniscus": (
        "lateral meniscus tear",
        "Abnormal signal that definitely contacts the meniscal surface on at least two images, "
        "or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, "
        "involving the lateral meniscus. Intrasubstance degeneration that does not reach the "
        "surface is negative.",
    ),
    "Medial OA": (
        "medial compartment osteoarthritis",
        "A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, "
        "defined as greater than 50 percent of cartilage thickness, in the medial compartment "
        "(between the medial femoral condyle and the medial tibial plateau, on the inner side "
        "of the knee), with or without underlying subchondral marrow changes.",
    ),
    "Lateral OA": (
        "lateral compartment osteoarthritis",
        "A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, "
        "defined as greater than 50 percent of cartilage thickness, in the lateral compartment "
        "(between the lateral femoral condyle and the lateral tibial plateau, on the outer side "
        "of the knee), with or without underlying subchondral marrow changes.",
    ),
    "PF OA": (
        "patellofemoral compartment osteoarthritis",
        "A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, "
        "defined as greater than 50 percent of cartilage thickness, in the patellofemoral "
        "compartment (between the patella and the femoral trochlea, the groove on the front of "
        "the femur), with or without underlying subchondral marrow changes.",
    ),
    "Effusion": (
        "joint effusion",
        "A moderate or large amount of fluid distending the joint.",
    ),
    "Synovitis": (
        "synovitis",
        "Inflammation and thickening of the synovial lining of the joint.",
    ),
    "Baker's": (
        "Baker (popliteal) cyst",
        "A moderate or large fluid collection in the characteristic location behind the knee.",
    ),
    "Contusion": (
        "contusion",
        "A bone contusion, seen as bone marrow edema-like signal from impact, without a "
        "discrete fracture line.",
    ),
    "Fracture": (
        "acute fracture",
        "An acute cortical break or fracture line.",
    ),
}

# The rubric's one global instruction, load-bearing enough to repeat in every
# prompt: it is what separates this rubric from a naive reading of the report,
# and it is the difference the plan expects most competitors to miss.
_SHARED_FRAMING = (
    "You are grading a knee MRI radiology report against one specific criterion used by "
    "musculoskeletal radiologists. Ambiguous or borderline findings (\"on the fence\") are "
    "graded negative to favor specificity."
)


def build_label_prompt(report_text: str, label: str) -> str:
    """One (report, label) prompt: the shared framing, the single rubric
    criterion for this label, the report, and a one-word yes/no question. The
    score is read from the Yes/No logits at the one answer-token position, so
    nothing here needs to be parsed out of generated prose."""
    if label not in _LABEL_CRITERIA:
        raise ValueError(f"no rubric criterion mapped for label {label!r}")
    finding, criterion = _LABEL_CRITERIA[label]
    return (
        f"{_SHARED_FRAMING}\n\n"
        f'Criterion for "{finding}":\n{criterion}\n\n'
        f"---\nRadiology report (verbatim, may be in any language):\n{report_text}\n---\n\n"
        f'Question: applying only the criterion above, does this report describe "{finding}"?\n'
        f"Answer with exactly one word, Yes or No.\nAnswer:"
    )


_YES_TOKENS = {"Yes", "yes", " Yes", " yes", "YES"}
_NO_TOKENS = {"No", "no", " No", " no", "NO"}
# Floor for a token that isn't present in the scores handed in at all, used to
# treat "not offered a slot" as stronger evidence than merely low-probability.
# With the transformers path this is effectively unreachable -- the full logit
# vector is available, so Yes and No are always both present. It stays for
# callers that hand in a truncated top-k mapping (what a vLLM-style engine
# returns), so the scoring contract doesn't depend on which engine produced it.
_MISSING_LOGPROB = -9999.0


def score_from_top_logprobs(top_logprobs) -> tuple[float | None, float]:
    """Turn the top-k logprobs at one generated token position into a soft
    yes/no score in [0, 1] and a reliability weight in [0, 1]. top_logprobs is
    anything mapping to objects exposing .decoded_token and .logprob --
    vLLM's dict[int, Logprob] in production, a plain stand-in in tests -- so
    this file has no vllm import and needs no GPU to test.

    Returns (None, 0.0) when neither a yes nor a no token appears among the
    top-k candidates at all: the model didn't answer the question as asked,
    which is Tier-4 "no evidence" (see the plan's Label hierarchy), mapped to
    NaN by the caller, never treated as an implicit negative."""
    yes_lp = max(
        (lp.logprob for lp in top_logprobs.values() if lp.decoded_token in _YES_TOKENS),
        default=None,
    )
    no_lp = max(
        (lp.logprob for lp in top_logprobs.values() if lp.decoded_token in _NO_TOKENS),
        default=None,
    )
    if yes_lp is None and no_lp is None:
        return None, 0.0

    yes_lp = _MISSING_LOGPROB if yes_lp is None else yes_lp
    no_lp = _MISSING_LOGPROB if no_lp is None else no_lp
    # renormalize over just the two candidates -- max-subtraction keeps exp()
    # in range regardless of how negative vLLM's raw logprobs run
    peak = max(yes_lp, no_lp)
    yes_p = math.exp(yes_lp - peak)
    no_p = math.exp(no_lp - peak)
    score = yes_p / (yes_p + no_p)
    weight = abs(score - 0.5) * 2  # 0 = coin flip between yes/no, 1 = fully confident
    return score, weight


def score_report_with_llm(report_text: str, generate_fn) -> dict[str, tuple[float, float]]:
    """Score all 12 labels for one report. generate_fn(prompt) -> the top-k
    logprobs dict for the single generated token at that prompt (see
    score_from_top_logprobs); callers wire this to a real vLLM engine
    (notebooks/phase3-labels/), tests wire it to a fake -- same function
    either way, so the bake-off (Step 1a) and the full corpus run (Step 1b)
    can never drift apart. None scores map to NaN here, matching
    lexical_label's "no evidence, never an implicit negative" convention."""
    scores: dict[str, tuple[float, float]] = {}
    for label in LABEL_COLUMNS:
        prompt = build_label_prompt(report_text, label)
        score, weight = score_from_top_logprobs(generate_fn(prompt))
        scores[label] = (float("nan") if score is None else score, weight)
    return scores
