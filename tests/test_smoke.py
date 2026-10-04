"""Stage 1: env skeleton, config, seed, logging."""
import json
import os

import pytest

from utils.config import default_config_path, load_config
from utils.logging_utils import JsonlLogger, get_logger
from utils.seed import seed_everything


def test_config_loads():
    cfg = load_config(default_config_path())
    for k in ("backbone", "max_len", "group_size_G", "lr_backbone", "human_sources"):
        assert k in cfg
    assert cfg["max_len"] == 512
    assert "answerdotai/ModernBERT-large" in cfg["backbone"]


def test_seed_deterministic():
    import random

    seed_everything(7)
    a = [random.random() for _ in range(5)]
    seed_everything(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_jsonl_logger(tmp_path):
    jl = JsonlLogger(tmp_path / "m.jsonl")
    jl.log({"step": 0, "loss": 1.5})
    rows = open(tmp_path / "m.jsonl").read().strip().splitlines()
    assert json.loads(rows[0])["loss"] == 1.5


def test_logger_stdout():
    assert get_logger("laya-test") is not None


@pytest.mark.skipif(os.environ.get("LAYA_TEST_BACKBONE") != "1",
                    reason="set LAYA_TEST_BACKBONE=1 to download ModernBERT-large (~1.5GB)")
def test_backbone_loads_and_dummy_forward():
    """Stage 1 Done-when: backbone loads + one dummy fp16 forward on GPU."""
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained("answerdotai/ModernBERT-large", attn_implementation="sdpa")
    m.eval()
    ids = torch.ones(1, 16, dtype=torch.long) * 100
    mask = torch.ones(1, 16, dtype=torch.long)
    with torch.no_grad():
        if torch.cuda.is_available():
            m = m.cuda().half()
            ids, mask = ids.cuda(), mask.cuda()
            with torch.cuda.amp.autocast():
                out = m(input_ids=ids, attention_mask=mask).last_hidden_state
        else:
            out = m(input_ids=ids, attention_mask=mask).last_hidden_state
    assert out.shape[-1] == 1024
