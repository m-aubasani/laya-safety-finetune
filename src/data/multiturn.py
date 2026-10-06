"""Multi-turn prefix slicing with TD(λ=1) MC targets (Stage 2.4)."""
from __future__ import annotations

from collections.abc import Iterable, Iterator


def render(turns: list[str]) -> str:
    return "\n".join(f"turn {i + 1}: {t}" for i, t in enumerate(turns))


def slice_conversation(conv: dict) -> Iterator[dict]:
    """Yield one example per prefix; every prefix gets the terminal outcome.

    conv = {id, turns:[...], outcome:int, type, instruction, options}
    Never includes future turns (no leakage).
    """
    turns = conv["turns"]
    for t in range(1, len(turns) + 1):
        yield {
            "id": f'{conv["id"]}_t{t}',
            "type": conv["type"],
            "instruction": conv["instruction"],
            "options": list(conv["options"]),
            "state": render(turns[:t]),
            "label": conv["outcome"],
            "meta": {
                "source": conv.get("meta", {}).get("source", "human"),
                "conversation_id": conv["id"],
                "turn": t,
                "n_turns": len(turns),
            },
        }


def slice_all(convs: Iterable[dict]) -> list[dict]:
    out: list[dict] = []
    for c in convs:
        out.extend(slice_conversation(c))
    return out
