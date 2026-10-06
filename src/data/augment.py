"""Dynamic per-epoch augmentation, train split only (Stage 2.3)."""
from __future__ import annotations

import copy
import json
import random

# Hand-written instruction templates per primitive. Sampling from this pool is
# allowed (labels stay human-made); LLM paraphrasing is NOT.
INSTRUCTION_TEMPLATES: dict[str, list[str]] = {
    "choice": [
        "{inst}",
        "Task: {inst}",
        "Choose the best option. {inst}",
        "Question: {inst} Pick one of the listed options.",
    ],
    "score": [
        "{inst}",
        "Rate on the rubric (low to high). {inst}",
        "Task: {inst} Reply with a rubric level.",
    ],
    "bool": [
        "{inst}",
        "True or false: {inst}",
        "Task: {inst} Answer true/false.",
    ],
}

EMAIL_FMT = "From: user@example.com\nSubject: support request\nBody:\n{state}"

NONE_TOKENS = ("none", "irrelevant", "n/a")


def paraphrase_instruction(instruction: str, task_type: str, rng: random.Random) -> str:
    pool = INSTRUCTION_TEMPLATES.get(task_type, ["{inst}"])
    return rng.choice(pool).format(inst=instruction)


def rotate_state(state: str, rng: random.Random) -> str:
    mode = rng.choice(["raw", "email", "json"])
    if mode == "raw":
        return state
    if mode == "email":
        return EMAIL_FMT.format(state=state)
    return '{"messages": [{"turn": 1, "text": %s}]}' % json.dumps(state)


def shuffle_options(ex: dict, rng: random.Random) -> dict:
    """Shuffle option order and remap label. NEVER call for score (ordinal)."""
    ex = copy.deepcopy(ex)
    assert ex["type"] != "score", "do not shuffle ordinal score options"
    idx = list(range(len(ex["options"])))
    rng.shuffle(idx)
    ex["options"] = [ex["options"][i] for i in idx]
    ex["label"] = idx.index(ex["label"])
    return ex


def none_index(options: list[str]) -> int | None:
    for i, o in enumerate(options):
        if o.strip().lower() in NONE_TOKENS:
            return i
    return None


def augment_example(
    ex: dict,
    pool: list[dict] | None = None,
    rng: random.Random | None = None,
    p_distract: float = 0.1,
) -> dict:
    """Apply one epoch of augmentation to a single train example."""
    rng = rng or random.Random()
    orig_state = ex["state"]

    # 4. Distractor: pair this state with an unrelated question whose gold
    # label is its none/irrelevant option.
    if pool and rng.random() < p_distract:
        cands = [c for c in pool if c["id"] != ex["id"] and none_index(c["options"]) is not None]
        if cands:
            other = copy.deepcopy(rng.choice(cands))
            other["state"] = orig_state
            other["label"] = none_index(other["options"])
            return other

    out = copy.deepcopy(ex)
    # 1. Shuffle (not for score).
    if out["type"] == "choice":
        out = shuffle_options(out, rng)
    # 2. Paraphrase instruction from template pool.
    out["instruction"] = paraphrase_instruction(out["instruction"], out["type"], rng)
    # 3. Rotate state format.
    out["state"] = rotate_state(out["state"], rng)
    return out


def augment_split(
    examples: list[dict], rng: random.Random | None = None, p_distract: float = 0.1
) -> list[dict]:
    rng = rng or random.Random()
    return [augment_example(ex, pool=examples, rng=rng, p_distract=p_distract) for ex in examples]
