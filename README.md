# laya-safety-finetune

Quick, reliable benchmark for iterating on steering configs from my other project, [memory_block](https://github.com/m-aubasani/memory_block), to test whether LLM steering improves model safety.


## Stage 1 — Environment + skeleton ✅

- Repo layout (`configs/`, `src/utils/`, `tests/`, `scripts/`, `data/`)
- Deps: `torch`, `transformers`, `accelerate`, `datasets`, `numpy`, `scikit-learn`, `pyyaml`, `pytest` (see `pyproject.toml`)
- `configs/default.yaml` — ModernBERT-large, max_len 512, RLCD + gate defaults
- `src/utils/`: `config.py` (YAML loader), `seed.py`, `logging_utils.py` (stdout + JSONL metrics)

## Setup

```powershell
uv sync
pytest
# heavy check (downloads ~1.5GB + needs GPU for fp16 path):
$env:LAYA_TEST_BACKBONE=1; pytest -k backbone
```

Stage 1 Done-when: `pytest` passes, `AutoModel.from_pretrained("answerdotai/ModernBERT-large")` loads, one dummy fp16 forward runs on GPU.

## Next

Stage 2 (data schema / validation / augmentation / multi-turn).
