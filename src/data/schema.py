"""Example schema constants and helpers (Stage 2.1)."""
from __future__ import annotations

PRIMITIVES = ("choice", "score", "bool")
BOOL_OPTIONS = ["false", "true"]

REQUIRED_FIELDS = ("id", "type", "instruction", "options", "state", "label", "meta")

# Anything in this set (as meta.source, case-insensitive) is rejected outright.
BANNED_SOURCES = {"llm", "synthetic", "distilled", "generated", "ai", "machine"}


def bool_options() -> list[str]:
    return list(BOOL_OPTIONS)


def is_ordinal(ex_type: str) -> bool:
    return ex_type == "score"
