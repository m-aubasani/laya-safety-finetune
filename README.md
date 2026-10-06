# laya-safety-finetune

Quick, reliable benchmark for iterating on steering configs from my other project, [memory_block](https://github.com/m-aubasani/memory_block), to test whether LLM steering improves model safety.

Full roadmap: `plan.md` (11 stages). **Status: Stage 2 done, Stage 3 not started.**

## Stage 1 — Environment + skeleton done

- Repo layout (`configs/`, `src/utils/`, `tests/`, `scripts/`, `data/`)
- Deps: `torch`, `transformers`, `accelerate`, `datasets`, `numpy`, `scikit-learn`, `pyyaml`, `pytest` (see `pyproject.toml`)
- `configs/default.yaml` — ModernBERT-large, max_len 512, RLCD + gate defaults
- `src/utils/`: `config.py` (YAML loader), `seed.py`, `logging_utils.py` (stdout + JSONL metrics)

## Stage 2 — Data schema / validation / augmentation / multi-turn done

- Schema (`src/data/schema.py`): JSONL with `choice` / `score` / `bool`, `bool` fixed to `["false","true"]`
- Validator (`src/data/validate.py`): rejects non-human sources (`llm`/`synthetic`/`distilled`), bad option counts (must be 2–20), duplicates, label out of range, over-length encodings, conversation-ID leakage across splits
- Augmentation (`src/data/augment.py`, train-only, per-epoch): option shuffle (never for `score`), template instruction paraphrases, state rotation (raw/email/JSON), 10% distractor questions
- Multi-turn (`src/data/multiturn.py`): prefix slicing, every prefix gets terminal outcome (TD lambda=1), no future leakage
- `scripts/build_dataset.py`: writes `train/val/test.jsonl` + `stats.json` (per-task counts, K histogram, label balance, length percentiles), splits by conversation ID

## Setup

```powershell
uv sync
uv run pytest
# heavy check (downloads ~1.5GB + needs GPU for fp16 path):
$env:LAYA_TEST_BACKBONE=1; uv run pytest -k backbone
# build dataset:
uv run python scripts/build_dataset.py --augment-train
```

Stage 2 Done-when: stats report written, validator + leakage tests pass (`test_validate.py`, `test_augment_multiturn.py`).

## Next

Stage 3 (encoding + collation) — not started.
