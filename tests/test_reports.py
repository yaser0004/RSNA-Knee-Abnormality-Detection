from knee.reports import lexical_label


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
