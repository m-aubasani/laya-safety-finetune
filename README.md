# laya-safety-finetune

Quick, reliable benchmark for iterating on steering configs from my other project, [memory_block](https://github.com/m-aubasani/memory_block), to test whether LLM steering improves model safety.

Full roadmap: `plan.md` (11 stages). **Status: Stage 3 done, Stage 4 not started.**

## Stage 1 — Environment + skeleton done

- Repo layout (`configs/`, `src/utils/`, `src/data/`, `tests/`, `scripts/`, `data/`)
- Deps: `torch`, `transformers`, `accelerate`, `datasets`, `numpy`, `scikit-learn`, `pyyaml`, `pytest` (see `pyproject.toml`)
- `configs/default.yaml` — ModernBERT-large, max_len 512, head_max_len 192, instr_max_len 64, RLCD + gate defaults
- `src/utils/`: `config.py` (YAML loader), `seed.py`, `logging_utils.py` (stdout + JSONL metrics)

Done-when: `pytest` runs, backbone loads, one dummy fp16 forward pass runs.

## Stage 2 — Data schema / validation / augmentation / multi-turn done

- Schema (`src/data/schema.py`): JSONL with `choice` / `score` / `bool`, `bool` fixed to `["false","true"]`
- Validator (`src/data/validate.py`): rejects non-human sources (`llm`/`synthetic`/`distilled`), bad option counts (must be 2–20), duplicates, label out of range, over-length encodings, conversation-ID leakage across splits
- Augmentation (`src/data/augment.py`, train-only, per-epoch): option shuffle (never for `score`), template instruction paraphrases, state rotation (raw/email/JSON), 10% distractor questions
- Multi-turn (`src/data/multiturn.py`): prefix slicing, every prefix gets terminal outcome (TD lambda=1), no future leakage
- `scripts/build_dataset.py`: writes `train/val/test.jsonl` + `stats.json` (per-task counts, K histogram, label balance, length percentiles), splits by conversation ID

Done-when: stats report written, validator + leakage tests pass (`test_validate.py`, `test_augment_multiturn.py`).

## Stage 3 — Encoding + collation done

Input topology (`src/data/encode.py`):

```text
[CLS] <type> question: <instruction> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

- `per_opt = max(2, head_max_len // K)` tokens per option including its `[MASK]`; instruction capped at `instr_max_len` (64)
- Truncation policy: truncate the **state** only, never instruction/options beyond their caps; total `len <= max_len` (512)
- Returns `ids, mask_pos` where each `mask_pos[k]` points at a `[MASK]` token

Batch collation (`src/data/collate.py`):

- Pads `input_ids` / `attention_mask` to batch max length
- Pads `mask_pos` to batch `Kmax` with `0`; `opt_mask` (B, Kmax) bool marks valid options
- Returns `y` one-hot (B, Kmax) plus `y_idx`, and `is_ordinal` (B,) bool (`True` for `score`)

Done-when: `tests/test_encode_collate.py` passes (3 tests): `[MASK]` positions verified, `len <= max_len`, state-truncated-first, collate masks/one-hot/ordinal flags correct.

## Setup

```powershell
uv sync
uv run pytest tests/test_encode_collate.py -v
uv run pytest
# heavy check (downloads ~1.5GB + needs GPU for fp16 path):
$env:LAYA_TEST_BACKBONE=1; uv run pytest -k backbone
# build dataset:
uv run python scripts/build_dataset.py --augment-train
```

## Next

Stage 4 (model `src/model/laya.py`) — not started.
