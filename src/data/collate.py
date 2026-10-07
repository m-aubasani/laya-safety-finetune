"""Batch collator (Stage 3.2)."""
from __future__ import annotations

import torch


def collate_batch(items: list[dict], pad_id: int = 0) -> dict:
    """items: [{input_ids, mask_pos, label, type}]. Returns padded tensors.

    - input_ids (B,L), attention_mask (B,L)
    - mask_pos (B,Kmax) padded with 0, opt_mask (B,Kmax) bool
    - y one-hot (B,Kmax), y_idx (B,), is_ordinal (B,) bool
    """
    B = len(items)
    Kmax = max(len(it["mask_pos"]) for it in items)
    Lmax = max(len(it["input_ids"]) for it in items)
    input_ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, Lmax), dtype=torch.long)
    mask_pos = torch.zeros((B, Kmax), dtype=torch.long)
    opt_mask = torch.zeros((B, Kmax), dtype=torch.bool)
    y = torch.zeros((B, Kmax), dtype=torch.float)
    y_idx = torch.zeros((B,), dtype=torch.long)
    is_ordinal = torch.zeros((B,), dtype=torch.bool)
    for b, it in enumerate(items):
        L = len(it["input_ids"])
        input_ids[b, :L] = torch.tensor(it["input_ids"], dtype=torch.long)
        attention_mask[b, :L] = 1
        K = len(it["mask_pos"])
        mask_pos[b, :K] = torch.tensor(it["mask_pos"], dtype=torch.long)
        opt_mask[b, :K] = True
        y[b, it["label"]] = 1.0
        y_idx[b] = it["label"]
        is_ordinal[b] = it["type"] == "score"
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask_pos": mask_pos,
        "opt_mask": opt_mask,
        "y": y,
        "y_idx": y_idx,
        "is_ordinal": is_ordinal,
    }
