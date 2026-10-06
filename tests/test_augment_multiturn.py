"""Stage 2.3/2.4: augmentation + multi-turn slicing."""
import random

from data.augment import augment_example, shuffle_options
from data.multiturn import slice_conversation


def test_shuffle_remaps_label():
    ex = {"id": "a", "type": "choice", "instruction": "Q", "options": ["a", "b", "c", "none"],
          "state": "s", "label": 2}
    rng = random.Random(0)
    out = shuffle_options(ex, rng)
    assert sorted(out["options"]) == ["a", "b", "c", "none"]
    assert out["options"][out["label"]] == "c"  # same answer, new position
    assert ex["label"] == 2  # input untouched


def test_score_never_shuffled():
    import pytest

    ex = {"id": "s", "type": "score", "instruction": "Rate", "options": ["bad", "ok", "great"],
          "state": "x", "label": 2}
    with pytest.raises(AssertionError):
        shuffle_options(ex, random.Random(0))
    out = augment_example(ex, pool=None, rng=random.Random(0))
    assert out["options"] == ["bad", "ok", "great"] and out["label"] == 2


def test_multiturn_prefix_no_future_leakage():
    conv = {"id": "c9", "turns": ["hi", "order?", "done", "thanks"], "outcome": 1,
            "type": "choice", "instruction": "Q?", "options": ["a", "b"], "meta": {"source": "human"}}
    slices = list(slice_conversation(conv))
    assert len(slices) == 4
    assert slices[0]["state"].count("turn") == 1 and "thanks" not in slices[0]["state"]
    assert "thanks" in slices[3]["state"]
    assert all(s["label"] == 1 for s in slices)  # TD(1): terminal outcome everywhere
    assert all(s["meta"]["conversation_id"] == "c9" for s in slices)
