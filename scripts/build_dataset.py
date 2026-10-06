"""Build train/val/test.jsonl from raw sources with stats + leakage check (Stage 2)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from data.augment import augment_split  # noqa: E402
from data.multiturn import slice_all  # noqa: E402
from data.validate import assert_no_conversation_leakage, validate_split  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging_utils import get_logger  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

log = get_logger("build_dataset")


def split_by_conversation(examples: list[dict], val_ratio=0.1, test_ratio=0.1, seed=42):
    cids = sorted({(e.get("meta") or {}).get("conversation_id") or e["id"] for e in examples})
    rng = random.Random(seed)
    rng.shuffle(cids)
    n = len(cids)
    n_test = max(1 if n >= 3 else 0, int(n * test_ratio))
    n_val = max(1 if n >= 3 else 0, int(n * val_ratio))
    test_ids = set(cids[:n_test])
    val_ids = set(cids[n_test : n_test + n_val])

    def key(e):
        return (e.get("meta") or {}).get("conversation_id") or e["id"]

    train = [e for e in examples if key(e) not in val_ids | test_ids]
    val = [e for e in examples if key(e) in val_ids]
    test = [e for e in examples if key(e) in test_ids]
    return train, val, test


def stats_report(examples: list[dict]) -> dict:
    tok_lens = sorted(len((e["instruction"] + " " + e["state"]).split()) for e in examples)
    pct = lambda q: tok_lens[min(len(tok_lens) - 1, int(q * len(tok_lens)))] if tok_lens else 0
    return {
        "n": len(examples),
        "per_type": dict(Counter(e["type"] for e in examples)),
        "K_hist": dict(Counter(len(e["options"]) for e in examples)),
        "label_balance": {f"{t}/{lab}": c for (t, lab), c in Counter((e["type"], e["label"]) for e in examples).items()},
        "len_p50": pct(0.5),
        "len_p90": pct(0.9),
        "len_p99": pct(0.99),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    ap.add_argument("--raw", default=str(REPO / "data" / "raw" / "examples.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data" / "processed"))
    ap.add_argument("--augment-train", action="store_true", help="apply one epoch of train augmentation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        log.warning("no raw file at %s; writing tiny demo dataset", raw)
        demo = [
            {"id": f"demo_{i}", "type": "choice", "instruction": "Which department?",
             "options": ["billing", "technical", "account", "none"], "state": f"ticket text {i}",
             "label": i % 4, "meta": {"source": "human", "conversation_id": f"c{i // 2}"}}
            for i in range(12)
        ]
        raw.parent.mkdir(parents=True, exist_ok=True)
        with open(raw, "w", encoding="utf-8") as f:
            for e in demo:
                f.write(json.dumps(e) + "\n")

    examples = [json.loads(l) for l in open(raw, encoding="utf-8") if l.strip()]
    # Optional multiturn expansion file alongside raw.
    conv_path = raw.parent / "conversations.jsonl"
    if conv_path.exists():
        convs = [json.loads(l) for l in open(conv_path, encoding="utf-8") if l.strip()]
        examples += slice_all(convs)

    validate_split(examples, cfg["human_sources"])
    train, val, test = split_by_conversation(examples, seed=cfg.get("seed", 42))
    assert_no_conversation_leakage({"train": train, "val": val, "test": test})
    if args.augment_train:
        train = augment_split(train, rng=random.Random(cfg.get("seed", 42)))

    for name, split in (("train", train), ("val", val), ("test", test)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in split:
                f.write(json.dumps(e) + "\n")
        log.info("%s: %s", name, stats_report(split))
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump({n: stats_report(s) for n, s in (("train", train), ("val", val), ("test", test))}, f, indent=2)
    log.info("wrote %s (n=%d/%d/%d)", out, len(train), len(val), len(test))


if __name__ == "__main__":
    main()

