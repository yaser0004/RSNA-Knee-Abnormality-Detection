from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from knee.infer import LABEL_COLUMNS
from knee.reports import (
    build_label_prompt,
    build_lexical_labels,
    lexical_label,
    score_from_top_logprobs,
    score_report_with_llm,
)


def _logprob(decoded_token, logprob):
    # stands in for vllm.logprobs.Logprob -- score_from_top_logprobs only
    # touches .decoded_token/.logprob, so no vllm import is needed to test it
    return SimpleNamespace(decoded_token=decoded_token, logprob=logprob)


def test_positive_match_with_no_preceding_negation_is_labeled_1():
    text = "There is a large effusion in the suprapatellar recess."

    assert lexical_label(text, "effusion") == 1


def test_negation_immediately_before_match_is_labeled_0():
    text = "No effusion is seen."

    assert lexical_label(text, "effusion") == 0


def test_no_match_at_all_returns_none_not_negative():
    text = "Normal knee MRI, no acute findings."

    assert lexical_label(text, "effusion") is None


def test_spanish_positive_match():
    text = "Derrame articular moderado en el compartimento medial."

    assert lexical_label(text, "effusion") == 1


def test_spanish_negation():
    text = "No hay derrame articular."

    assert lexical_label(text, "effusion") == 0


def test_build_lexical_labels_produces_all_12_columns_nan_for_unsupported():
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1"],
            "Report": ["There is a large effusion. No acute fracture."],
        }
    )

    labels = build_lexical_labels(df)

    assert list(labels.columns) == ["StudyInstanceUID"] + LABEL_COLUMNS
    assert labels.loc[0, "Effusion"] == 1
    # Synovitis has no lexical rule defined at all -- must be NaN, not crash
    assert np.isnan(labels.loc[0, "Synovitis"])


def test_build_lexical_labels_is_nan_for_studies_with_no_rule_match():
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1"],
            "Report": ["Normal knee MRI, no acute findings."],
        }
    )

    labels = build_lexical_labels(df)

    assert np.isnan(labels.loc[0, "Effusion"])


def test_build_lexical_labels_handles_nan_report_without_crashing():
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1", "s2"],
            "Report": [np.nan, "There is a large effusion."],
        }
    )

    labels = build_lexical_labels(df)

    assert np.isnan(labels.loc[0, "Effusion"])
    assert labels.loc[1, "Effusion"] == 1


def test_build_label_prompt_raises_for_a_label_with_no_rubric_mapping():
    with pytest.raises(ValueError):
        build_label_prompt("some report", "Not A Real Label")


def test_build_label_prompt_includes_the_report_text_and_the_rubric_finding_name():
    prompt = build_label_prompt("There is a large joint effusion.", "Effusion")

    assert "There is a large joint effusion." in prompt
    assert "joint effusion" in prompt.lower()


def test_build_label_prompt_sends_only_the_asked_labels_criterion():
    # An earlier version prepended the whole 12-definition rubric to every
    # question, which is what forced the (since-abandoned) vLLM prefix-cache
    # dependency. Each prompt must now carry its own criterion and no others,
    # both to stay short and so the model isn't handed eleven competing
    # definitions as distractors.
    prompt = build_label_prompt("Normal knee MRI.", "ACL")

    assert "more than 50 percent of fibers disrupted" in prompt  # the ACL criterion
    assert "meniscal surface" not in prompt  # a different label's criterion
    assert "synovial lining" not in prompt


def test_build_label_prompt_keeps_the_borderline_is_negative_framing():
    # the rubric's one global instruction, and the main thing separating it
    # from a naive read of the report -- it has to survive in every prompt
    for label in ("ACL", "Synovitis", "Fracture"):
        assert "borderline" in build_label_prompt("Normal knee MRI.", label).lower()


def test_every_label_column_has_its_own_distinct_criterion():
    # the rubric states three criteria as back-references ("the same criteria
    # applied to the lateral meniscus"); those are resolved inline, so no two
    # labels may end up sharing identical criterion text
    report = "Normal knee MRI."
    prompts = {label: build_label_prompt(report, label) for label in LABEL_COLUMNS}

    assert len(set(prompts.values())) == len(LABEL_COLUMNS)


def test_score_from_top_logprobs_favors_yes_when_its_logprob_is_higher():
    top_logprobs = {1: _logprob("Yes", -0.1), 2: _logprob("No", -4.0)}

    score, weight = score_from_top_logprobs(top_logprobs)

    assert score > 0.9
    assert weight > 0.9


def test_score_from_top_logprobs_favors_no_when_its_logprob_is_higher():
    top_logprobs = {1: _logprob("Yes", -4.0), 2: _logprob("No", -0.1)}

    score, weight = score_from_top_logprobs(top_logprobs)

    assert score < 0.1
    assert weight > 0.9


def test_score_from_top_logprobs_is_near_half_when_yes_and_no_are_equally_likely():
    top_logprobs = {1: _logprob("Yes", -1.0), 2: _logprob("No", -1.0)}

    score, weight = score_from_top_logprobs(top_logprobs)

    assert score == pytest.approx(0.5, abs=1e-6)
    assert weight == pytest.approx(0.0, abs=1e-6)


def test_score_from_top_logprobs_returns_none_and_zero_weight_when_neither_token_present():
    # the model answered with something other than yes/no -- Tier-4 "no
    # evidence", mapped to NaN downstream, never treated as a negative
    top_logprobs = {1: _logprob("Maybe", -0.5), 2: _logprob("Unclear", -2.0)}

    score, weight = score_from_top_logprobs(top_logprobs)

    assert score is None
    assert weight == 0.0


def test_score_from_top_logprobs_treats_a_missing_no_as_high_confidence_yes():
    # only "Yes" made it into the top-k candidates at all -- "No" wasn't
    # merely low-probability, it wasn't offered a slot, so this should read
    # as strong evidence rather than a coin flip
    top_logprobs = {1: _logprob("Yes", -0.05)}

    score, weight = score_from_top_logprobs(top_logprobs)

    assert score > 0.99
    assert weight > 0.99


def test_score_report_with_llm_covers_all_12_labels_and_maps_none_to_nan():
    def fake_generate(prompt):
        # the rubric text itself mentions "fracture" repeatedly, so match the
        # quoted finding name in the trailing question, not a bare substring
        if '"acute fracture"' in prompt:
            return {}  # simulate a question the model didn't answer yes/no to
        return {1: _logprob("No", -0.2)}

    scores = score_report_with_llm("Normal knee MRI.", fake_generate)

    assert set(scores) == set(LABEL_COLUMNS)
    fracture_score, fracture_weight = scores["Fracture"]
    assert np.isnan(fracture_score)
    assert fracture_weight == 0.0
    acl_score, acl_weight = scores["ACL"]
    assert acl_score < 0.5
    assert acl_weight > 0.0
