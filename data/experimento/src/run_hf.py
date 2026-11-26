"""Stage 2a: inference with Hugging Face models (fine-tuned classifier, NLI zero-shot, LLMs).

Usage: python run_hf.py --model <name in config.MODELS> --items items.jsonl --out preds.jsonl
A failing batch is retried item by item; items that still fail are logged to
<out>.failures.jsonl and skipped (a later run with the same --out retries them).
"""
import json
import time
import traceback

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import Writer, fit_state, jobs, load_items, parse_args
from config import LABELS, MODELS
from prompts import VARIANTS, hypothesis, label_words, llm_messages, render_state, to_canonical

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available: refusing to run on CPU")
DEV = "cuda"


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def run_safely(fn, batch, out_path):
    """Run fn(batch); on error retry one by one and log the items that still fail."""
    try:
        fn(batch)
        return
    except Exception:  # noqa: BLE001 - one bad batch must not stop the model
        torch.cuda.empty_cache()
    for job in batch:
        try:
            fn([job])
        except Exception as e:  # noqa: BLE001
            torch.cuda.empty_cache()
            with open(f"{out_path}.failures.jsonl", "a") as f:
                f.write(json.dumps({"id": job[0]["id"], "condition": job[1], "variant": job[2],
                                    "error": repr(e), "trace": traceback.format_exc()[-2000:]}) + "\n")


def twitter_preprocess(text):
    # as in the model card: user mentions and links are replaced
    return " ".join("@user" if w.startswith("@") and len(w) > 1 else "http" if w.startswith("http") else w
                    for w in text.split(" "))


def run_classifier(cfg, todo, w, bs, out):
    tok = AutoTokenizer.from_pretrained(cfg["hf"], revision=cfg["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf"], revision=cfg["revision"], torch_dtype=torch.float32).to(DEV).eval()
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    assert sorted(id2label.values()) == LABELS, id2label
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    def go(batch):
        states, infos = zip(*[fit_state(it, c, count, cfg["max_tokens"] - 2) for it, c, _ in batch])
        enc = tok([twitter_preprocess(s) for s in states], return_tensors="pt", padding=True,
                  truncation=True, max_length=cfg["max_tokens"]).to(DEV)
        with torch.inference_mode():
            p = model(**enc).logits.float().softmax(-1).cpu()
        for (it, c, v), info, row in zip(batch, infos, p):
            w.write(it, c, v, {id2label[i]: float(x) for i, x in enumerate(row)}, **info)

    for batch in batched(todo, bs):
        run_safely(go, batch, out)


def run_nli(cfg, todo, w, bs, out):
    tok = AutoTokenizer.from_pretrained(cfg["hf"], revision=cfg["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf"], revision=cfg["revision"], torch_dtype=torch.float32).to(DEV).eval()
    ent = {l.lower(): i for i, l in model.config.id2label.items()}["entailment"]
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    def go(batch):
        states, infos = zip(*[fit_state(it, c, count, cfg["max_tokens"] - 40) for it, c, _ in batch])
        prem = [s for s in states for _ in LABELS]
        hyp = [hypothesis(it, lab) for it, _, _ in batch for lab in LABELS]
        enc = tok(prem, hyp, return_tensors="pt", padding=True, truncation="only_first",
                  max_length=cfg["max_tokens"]).to(DEV)
        with torch.inference_mode():
            logits = model(**enc).logits.float()[:, ent].view(len(batch), len(LABELS))
        p = logits.softmax(-1).cpu()
        for (it, c, v), info, row in zip(batch, infos, p):
            w.write(it, c, v, dict(zip(LABELS, map(float, row))), **info)

    for batch in batched(todo, max(1, bs // 3)):
        run_safely(go, batch, out)


def first_token_ids(tok, word):
    """First token of each spelling of a label word (lower/capitalised, with/without space)."""
    forms = {word, word.capitalize(), " " + word, " " + word.capitalize()}
    return sorted({tok.encode(f, add_special_tokens=False)[0] for f in forms})


def run_llm(cfg, todo, w, bs, out):
    from transformers import AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(cfg["hf"], revision=cfg["revision"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], revision=cfg["revision"],
                                                 torch_dtype=torch.bfloat16).to(DEV).eval()
    word_ids = {}
    for it, _, _ in todo:
        for word in label_words(it).values():
            if word not in word_ids:
                word_ids[word] = first_token_ids(tok, word)
    print("label first tokens:", {k: [tok.decode([i]) for i in v] for k, v in word_ids.items()}, flush=True)

    prompts = [tok.apply_chat_template(llm_messages(render_state(it, c), it, v), add_generation_prompt=True,
                                       tokenize=False, **cfg["chat_kwargs"]) for it, c, v in todo]
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(todo)), key=lens.__getitem__)  # length bucketing
    groups, cur = [], []
    for i in order:  # cap each batch by tokens (padded length x batch size)
        if cur and (len(cur) >= bs or lens[i] * (len(cur) + 1) > cfg.get("max_batch_tokens", 16000)):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)

    def go(idx_jobs):
        idx = [i for i, _ in idx_jobs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEV)
        with torch.inference_mode():
            last = model(**enc, logits_to_keep=1).logits[:, -1, :].float().softmax(-1)
        for row, i in zip(last, idx):
            it, c, v = todo[i]
            words = label_words(it)
            ids = {wd: [x for x in word_ids[wd]] for wd in words.values()}
            flat = [x for v_ in ids.values() for x in v_]
            assert len(flat) == len(set(flat)), f"label first tokens overlap: {ids}"
            mass = {wd: float(row[x].sum()) for wd, x in ids.items()}
            total = sum(mass.values())
            w.write(it, c, v, to_canonical(it, mass), label_mass=total,
                    top_token=tok.decode([int(row.argmax())]), tokens=lens[i])

    for g in groups:
        # run_safely expects (item, condition, variant)-like tuples for logging
        batch = [(i, todo[i]) for i in g]
        try:
            go(batch)
        except Exception:  # noqa: BLE001
            torch.cuda.empty_cache()
            for one in batch:
                try:
                    go([one])
                except Exception as e:  # noqa: BLE001
                    torch.cuda.empty_cache()
                    it, c, v = one[1]
                    with open(f"{out}.failures.jsonl", "a") as f:
                        f.write(json.dumps({"id": it["id"], "condition": c, "variant": v,
                                            "error": repr(e), "trace": traceback.format_exc()[-2000:]}) + "\n")


def main():
    a = parse_args()
    cfg = MODELS[a.model]
    items = load_items(a.items, a.limit)
    # classifier: no question; NLI: one hypothesis per label, so the option order does not apply
    variants = {"classifier": ["none"], "nli": ["v1"]}.get(cfg["kind"], VARIANTS)
    w = Writer(a.out)
    todo = w.todo(jobs(items, variants))
    print(f"{a.model}: {len(todo)} predictions to run", flush=True)
    t0 = time.time()
    {"classifier": run_classifier, "nli": run_nli, "llm": run_llm}[cfg["kind"]](cfg, todo, w, a.batch, a.out)
    w.close()
    print(f"{a.model}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
