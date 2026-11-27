"""Diagnostic (2026-11-26): is Mistral-7B's option-order sensitivity a position bias?

Same prompt and scoring as run_hf.py (first token of each label spelling, renormalised over the
three labels), Mistral-7B-Instruct-v0.3 at the pinned revision, but with all 6 option orders.
A. Content-free prior: one real item per dataset with its text replaced by "", "N/A" or
   "[MASK]" (no context). Any preference here can only come from position or label word.
B. Balanced orders on real items: 100 ironic and 100 non-ironic MUStARD++ items (seed 42), no
   context, all 6 orders, so each label appears twice in each position.
Usage: python -I diag_posicao.py <src dir with prompts.py/config.py> <items.jsonl> <out.jsonl> [A|B]
CPU, bfloat16.
"""
import itertools, json, random, sys
src, items_path, out_path = sys.argv[1:4]
parts = sys.argv[4] if len(sys.argv) > 4 else "AB"
sys.path.insert(0, src)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import prompts
from config import MODELS

LABELS = ["negative", "neutral", "positive"]
PERMS = {"".join(l[:3] for l in p): list(p) for p in itertools.permutations(LABELS)}
prompts.ORDER.update(PERMS)
cfg = MODELS["mistral-7b"]
torch.set_num_threads(16)
tok = AutoTokenizer.from_pretrained(cfg["hf"], revision=cfg["revision"])
model = AutoModelForCausalLM.from_pretrained(cfg["hf"], revision=cfg["revision"], dtype=torch.bfloat16).eval()

def first_ids(word):
    forms = {word, word.capitalize(), " " + word, " " + word.capitalize()}
    return sorted({tok.encode(f, add_special_tokens=False)[0] for f in forms})

def score(it, perm):
    msg = prompts.llm_messages(prompts.render_state(it, "no_context"), it, perm)
    p = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, **cfg["chat_kwargs"])
    enc = tok(p, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        row = model(**enc, logits_to_keep=1).logits[0, -1, :].float().softmax(-1)
    words = prompts.label_words(it)
    mass = {w: float(row[first_ids(w)].sum()) for w in words.values()}
    return prompts.to_canonical(it, mass), sum(mass.values())

items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
out = open(out_path, "a", encoding="utf-8")
if "A" in parts:
    for ds in ["mustardpp", "mtcsd", "sarcasm_nlp", "filatova"]:
        base = next(i for i in items if i["dataset"] == ds)
        for null in ["", "N/A", "[MASK]"]:
            it = {**base, "text": null, "context": []}
            for name, order in PERMS.items():
                probs, m = score(it, name)
                out.write(json.dumps({"part": "A", "dataset": ds, "null": null, "order": order,
                                      "probs": probs, "label_mass": m}) + "\n")
                out.flush()
if "B" in parts:
    rng = random.Random(42)
    mpp = [i for i in items if i["dataset"] == "mustardpp"]
    sample = rng.sample([i for i in mpp if i["sarcastic"]], 100) + rng.sample([i for i in mpp if not i["sarcastic"]], 100)
    for it in sample:
        for name, order in PERMS.items():
            probs, m = score(it, name)
            out.write(json.dumps({"part": "B", "id": it["id"], "sarcastic": it["sarcastic"], "gold": it["gold"],
                                  "order": order, "probs": probs, "label_mass": m}) + "\n")
            out.flush()
