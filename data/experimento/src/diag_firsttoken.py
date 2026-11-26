"""Diagnostic: is first-token scoring the same as scoring the whole label word?

Usage: python diag_firsttoken.py <items.jsonl> <model> <out.jsonl>
run_hf.py scores each label by the probability of the FIRST token of its spellings
(word, Word, " word", " Word"). Some spellings split, e.g. "Favor" -> "F" + "avor" and
"Neutral" -> "Ne" + "utral" (Mistral), so p("F") also counts any other answer starting
with "F". Here each spelling is scored as a full token sequence,
P(word) = sum over distinct spellings of prod_t p(token_t | prompt, tokens_<t),
and compared with the first-token score on the same prompt (one prompt at a time, no
padding of the prompt; continuations are right-padded, which cannot change earlier
positions in a causal model).
Sample (config.SEED): every MT-CSD item, both conditions, v1-v3; every MUStARD++ item,
both conditions, v1; 400 Sarcasm_NLP and 200 Filatova items, no_context, v1.
"""
import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LABELS, MODELS, SEED
from prompts import VARIANTS, label_words, llm_messages, render_state, to_canonical
from run_hf import first_token_ids


def spellings(tok, word):
    forms = {word, word.capitalize(), " " + word, " " + word.capitalize()}
    return sorted({tuple(tok.encode(f, add_special_tokens=False)) for f in forms})


def main(items_path, name, out_path):
    cfg = MODELS[name]
    items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
    items = [i for i in items if not i.get("ablation_k") and not i.get("extra_condition")]
    rng = random.Random(SEED)
    jobs = [(it, c, v) for it in items if it["dataset"] == "mtcsd"
            for c in ("no_context", "with_context") for v in VARIANTS]
    jobs += [(it, c, "v1") for it in items if it["dataset"] == "mustardpp" for c in ("no_context", "with_context")]
    for ds, k in (("sarcasm_nlp", 400), ("filatova", 200)):
        jobs += [(it, "no_context", "v1") for it in rng.sample([i for i in items if i["dataset"] == ds], k)]

    tok = AutoTokenizer.from_pretrained(cfg["hf"], revision=cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], revision=cfg["revision"],
                                                 torch_dtype=torch.bfloat16).to("cuda").eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    words = sorted({w for it, _, _ in jobs for w in label_words(it).values()})
    first = {w: first_token_ids(tok, w) for w in words}
    spell = {w: spellings(tok, w) for w in words}
    print(name, "spellings:", {w: [[tok.decode([t]) for t in s] for s in spell[w]] for w in words}, flush=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        for n, (it, c, v) in enumerate(jobs):
            text = tok.apply_chat_template(llm_messages(render_state(it, c), it, v), add_generation_prompt=True,
                                           tokenize=False, **cfg["chat_kwargs"])
            p_ids = tok(text, add_special_tokens=False)["input_ids"]
            conts = [(w, s) for w in label_words(it).values() for s in spell[w]]
            L = max(len(s) for _, s in conts)
            seqs = [p_ids + list(s) + [pad] * (L - len(s)) for _, s in conts]
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor(seqs, device="cuda"), use_cache=False,
                               logits_to_keep=L + 1).logits.float().log_softmax(-1)
            # logits[:, j] predicts position len(p_ids) - 1 + j
            full = {w: 0.0 for w in label_words(it).values()}
            for row, (w, s) in enumerate(conts):
                lp = sum(float(logits[row, j, t]) for j, t in enumerate(s))
                full[w] += float(torch.tensor(lp).exp())
            last = logits[0, 0].exp()
            ft = {w: float(last[first[w]].sum()) for w in label_words(it).values()}
            pf, pw = to_canonical(it, ft), to_canonical(it, full)
            fh.write(json.dumps({"id": it["id"], "dataset": it["dataset"], "condition": c, "variant": v,
                                 "gold": it["gold"], "first": pf, "full": pw,
                                 "mass_first": sum(ft.values()), "mass_full": sum(full.values())}) + "\n")
            if n % 500 == 0:
                print(name, n, "/", len(jobs), flush=True)
    print(name, "done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
