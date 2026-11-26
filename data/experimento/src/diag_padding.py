"""Diagnostic: does left-padded batching change the LLM label probabilities?

Usage: python diag_padding.py <items.jsonl> <model> [n_items]
Scores the same prompts (MUStARD++ and MT-CSD, both conditions, variant v1) twice:
in length-bucketed batches of 8, as run_hf.py does, and one prompt at a time
(no padding). Prints the largest probability difference and prediction agreement
per condition. Sample drawn with config.SEED.
"""
import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LABELS, MODELS, SEED
from prompts import label_words, llm_messages, render_state, to_canonical
from run_hf import first_token_ids


def main(items_path, name, n=48):
    cfg = MODELS[name]
    items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
    pool = [i for i in items if i["dataset"] in ("mustardpp", "mtcsd") and i["context"] and not i["sarcastic"]]
    sample = random.Random(SEED).sample(pool, n)
    jobs = [(it, c) for it in sample for c in ("no_context", "with_context")]

    tok = AutoTokenizer.from_pretrained(cfg["hf"], revision=cfg["revision"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], revision=cfg["revision"],
                                                 torch_dtype=torch.bfloat16).to("cuda").eval()
    prompts = [tok.apply_chat_template(llm_messages(render_state(it, c), it, "v1"), add_generation_prompt=True,
                                       tokenize=False, **cfg["chat_kwargs"]) for it, c in jobs]
    word_ids = {w: first_token_ids(tok, w) for it, _ in jobs for w in label_words(it).values()}

    def score(idx):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            last = model(**enc, logits_to_keep=1).logits[:, -1, :].float().softmax(-1)
        out = {}
        for row, i in zip(last, idx):
            it = jobs[i][0]
            mass = {w: float(row[word_ids[w]].sum()) for w in label_words(it).values()}
            out[i] = to_canonical(it, mass)
        return out

    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(jobs)), key=lens.__getitem__)
    batched = {}
    for k in range(0, len(order), 8):
        batched.update(score(order[k:k + 8]))
    single = {}
    for i in range(len(jobs)):
        single.update(score([i]))

    for cond in ("no_context", "with_context"):
        idx = [i for i, (_, c) in enumerate(jobs) if c == cond]
        diff = max(abs(batched[i][l] - single[i][l]) for i in idx for l in LABELS)
        agree = sum(max(batched[i], key=batched[i].get) == max(single[i], key=single[i].get) for i in idx)
        acc_b = sum(max(batched[i], key=batched[i].get) == jobs[i][0]["gold"] for i in idx) / len(idx)
        acc_s = sum(max(single[i], key=single[i].get) == jobs[i][0]["gold"] for i in idx) / len(idx)
        print(f"{name} {cond}: n={len(idx)} max|dp|={diff:.4f} same_pred={agree}/{len(idx)} "
              f"acc_batched={acc_b:.3f} acc_single={acc_s:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:4]))
