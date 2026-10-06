"""Validator enforcing hard rules (Stage 2.2).

Rejects / raises on:
- meta.source not in human-source whitelist (incl. llm/synthetic/distilled)
- len(options) outside [2, 20]
- encoded length exceeding max_len
- duplicate options, label out of range
- conversation-ID leakage across splits
- bool options not exactly ["false","true"]
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from .schema import BANNED_SOURCES, BOOL_OPTIONS, PRIMITIVES, REQUIRED_FIELDS


class ValidationError(ValueError):
    pass


def _source_ok(source: str, whitelist: Iterable[str]) -> bool:
    s = str(source).lower()
    if s in BANNED_SOURCES:
        return False
    allowed = {str(w).lower() for w in whitelist}
    return s in allowed


def validate_example(
    ex: Mapping,
    human_sources: Iterable[str],
    tok=None,
    max_len: int = 512,
    head_max_len: int = 192,
    instr_max_len: int = 64,
) -> None:
    for f in REQUIRED_FIELDS:
        if f not in ex:
            raise ValidationError(f"missing field {f!r} in {ex.get('id')!r}")
    if ex["type"] not in PRIMITIVES:
        raise ValidationError(f"unknown type {ex['type']!r}")
    meta = ex.get("meta") or {}
    src = str(meta.get("source", ""))
    if not _source_ok(src, human_sources):
        raise ValidationError(f"non-human source rejected: {src!r} (id={ex['id']!r})")

    opts = ex["options"]
    if not isinstance(opts, list) or not (2 <= len(opts) <= 20):
        raise ValidationError(f"len(options) must be in [2,20], got {len(opts) if isinstance(opts, list) else type(opts)}")
    if len(set(opts)) != len(opts):
        raise ValidationError(f"duplicate options in {ex['id']!r}")
    label = ex["label"]
    if not isinstance(label, int) or not (0 <= label < len(opts)):
        raise ValidationError(f"label out of range in {ex['id']!r}: {label!r}")

    if ex["type"] == "bool" and list(opts) != BOOL_OPTIONS:
        raise ValidationError(f"bool options must be {BOOL_OPTIONS}, got {opts!r}")

    if tok is not None:
        from .encode import encode

        ids, _ = encode(ex, tok, max_len=max_len + 1, head_max_len=head_max_len, instr_max_len=instr_max_len)
        # encode with max_len+1 then check overflow: if it exceeds max_len, reject
        if len(ids) > max_len:
            raise ValidationError(f"encoded length {len(ids)} exceeds max_len={max_len} (id={ex['id']!r})")


def validate_split(examples: Iterable[Mapping], human_sources: Iterable[str], **kw) -> int:
    n = 0
    for ex in examples:
        validate_example(ex, human_sources, **kw)
        n += 1
    return n


def conversation_ids(examples: Iterable[Mapping]) -> set:
    out = set()
    for ex in examples:
        cid = (ex.get("meta") or {}).get("conversation_id")
        if cid is not None:
            out.add(cid)
    return out


def assert_no_conversation_leakage(splits: Mapping[str, Iterable[Mapping]]) -> None:
    """Fail if any conversation_id appears in more than one split (Hard rule 4)."""
    seen: dict[str, str] = {}
    for name, examples in splits.items():
        for ex in examples:
            cid = (ex.get("meta") or {}).get("conversation_id")
            if cid is None:
                continue
            if cid in seen and seen[cid] != name:
                raise ValidationError(f"conversation {cid!r} leaks across splits {seen[cid]!r} and {name!r}")
            seen[cid] = name
