"""Input-topology encoder (Stage 3.1).

Layout: [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
Truncation: state first, never instruction/options beyond their caps.
"""
from __future__ import annotations

from collections.abc import Mapping


def encode(ex: Mapping, tok, max_len: int = 512, head_max_len: int = 192, instr_max_len: int = 64):
    K = len(ex["options"])
    per_opt = max(2, head_max_len // K)  # tokens per option incl. its [MASK]
    ids = [tok.cls_token_id]
    ids += tok(f'{ex["type"]} question: {ex["instruction"]}', add_special_tokens=False).input_ids[:instr_max_len]
    ids.append(tok.sep_token_id)
    mask_pos: list[int] = []
    for o in ex["options"]:
        mask_pos.append(len(ids))
        ids.append(tok.mask_token_id)
        ids += tok(o, add_special_tokens=False).input_ids[: per_opt - 1]
    ids.append(tok.sep_token_id)
    room = max_len - len(ids) - 1
    ids += tok(ex["state"], add_special_tokens=False).input_ids[:room]  # truncate STATE only
    ids.append(tok.sep_token_id)
    return ids, mask_pos
