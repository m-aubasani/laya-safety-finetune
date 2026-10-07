"""Stage 3: encoding topology + collation."""
from data.collate import collate_batch
from data.encode import encode


class FakeTok:
    cls_token_id = 101
    sep_token_id = 102
    mask_token_id = 103

    def __call__(self, text, add_special_tokens=False):
        # wordpiece-ish stub: one id per whitespace token
        toks = text.split()
        return type("E", (), {"input_ids": [10 + (abs(hash(w)) % 500) for w in toks]})()


def ex(**kw):
    d = {"id": "e", "type": "choice", "instruction": "Which department?",
         "options": ["billing", "technical", "none"], "state": "charged twice this month"}
    d.update(kw)
    return d


def test_mask_positions_and_len_and_topology():
    tok = FakeTok()
    ids, mp = encode(ex(), tok, max_len=512, head_max_len=192, instr_max_len=64)
    assert len(ids) <= 512
    for p in mp:
        assert ids[p] == tok.mask_token_id
    # topology: [CLS] ... [SEP] ([MASK] opt)* [SEP] state [SEP]
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    assert ids.count(tok.sep_token_id) == 3


def test_state_truncated_first():
    tok = FakeTok()
    long_state = "word " * 2000
    ids, mp = encode(ex(state=long_state), tok, max_len=64)
    assert len(ids) <= 64
    for p in mp:  # options survive even under truncation pressure
        assert ids[p] == tok.mask_token_id


def test_collate_masks_and_onehot():
    tok = FakeTok()
    items = []
    for k, lab in ((2, 0), (4, 3)):
        e = ex(options=[f"o{i}" for i in range(k)], label=lab,
               type="score" if k == 4 else "choice")
        ids, mp = encode(e, tok)
        items.append({"input_ids": ids, "mask_pos": mp, "label": lab, "type": e["type"]})
    b = collate_batch(items, pad_id=0)
    assert b["opt_mask"].tolist() == [[True, True, False, False], [True] * 4]
    assert b["y"][1].tolist() == [0, 0, 0, 1.0]
    assert b["is_ordinal"].tolist() == [False, True]
    B = 2
    for i in range(B):
        for j, p in enumerate(b["mask_pos"][i].tolist()):
            if b["opt_mask"][i, j]:
                assert b["input_ids"][i, p].item() == tok.mask_token_id
