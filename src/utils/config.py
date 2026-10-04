"""Config loader for configs/*.yaml."""
from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a mapping, got {type(cfg)}")
    return cfg


def default_config_path() -> Path:
    # <repo>/configs/default.yaml regardless of CWD (src/utils/config.py -> parents[2])
    here = Path(__file__).resolve()
    repo = here.parents[2]
    return repo / "configs" / "default.yaml"
