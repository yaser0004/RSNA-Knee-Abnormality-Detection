import re

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
