"""Stage 2: validator rejects non-human labels, bad shapes, leakage."""
import pytest

from data.validate import ValidationError, assert_no_conversation_leakage, validate_example

HUMAN = ["human", "internal_support_v3"]


def base(**kw):
    ex = {"id": "ex_1", "type": "choice", "instruction": "Which?",
          "options": ["billing", "technical", "none"], "state": "charged twice",
          "label": 0, "meta": {"source": "human"}}
    ex.update(kw)
    return ex


def test_reject_llm_sources():
    for bad in ("llm", "synthetic", "distilled", "LLM", "Synthetic"):
        with pytest.raises(ValidationError):
            validate_example(base(meta={"source": bad}), HUMAN)


def test_reject_unknown_source():
    with pytest.raises(ValidationError):
        validate_example(base(meta={"source": "scraped_random"}), HUMAN)


def test_option_count_and_duplicates_and_label():
    with pytest.raises(ValidationError):
        validate_example(base(options=["only"]), HUMAN)
    with pytest.raises(ValidationError):
        validate_example(base(options=[f"o{i}" for i in range(21)]), HUMAN)
    with pytest.raises(ValidationError):
        validate_example(base(options=["a", "a", "b"]), HUMAN)
    with pytest.raises(ValidationError):
        validate_example(base(label=9), HUMAN)


def test_bool_must_be_false_true():
    with pytest.raises(ValidationError):
        validate_example(base(type="bool", options=["yes", "no"], label=0), HUMAN)
    validate_example(base(type="bool", options=["false", "true"], label=1), HUMAN)


def test_no_conversation_leakage():
    ok = {"train": [base(id="a", meta={"source": "human", "conversation_id": "c1"})],
          "val": [base(id="b", meta={"source": "human", "conversation_id": "c2"})]}
    assert_no_conversation_leakage(ok)
    bad = {"train": [base(id="a", meta={"source": "human", "conversation_id": "c1"})],
           "val": [base(id="b", meta={"source": "human", "conversation_id": "c1"})]}
    with pytest.raises(ValidationError):
        assert_no_conversation_leakage(bad)
