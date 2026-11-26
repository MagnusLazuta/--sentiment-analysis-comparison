"""Input tokens each model actually processes, counted the same way for every model (2026-11-23).

The `tokens` column of predictions.csv is not comparable across models: for twitter-roberta,
DeBERTa-NLI, Laya and NanoJev it is only the rendered state (text + context, used to fit the
context budget); for Qwen3.5-4B and Mistral-7B it is the full chat prompt; for Kev-4B it is the
server's usage.input_tokens of one request that answers the three option orders at once.

This script rebuilds the model input of every prediction of a run and counts it with the model's
own tokenizer, including special tokens, the question and the options:
  twitter-roberta  1 pass per item: [CLS] state [SEP] (state as fed, after fit_state)
  deberta-nli      3 passes per item: premise = state, hypothesis = one per label
  qwen3.5-4b,      1 pass per prediction: the chat prompt (= the recorded `tokens`, re-counted here)
  mistral-7b
  laya             1 sequence per prediction: laya.common.build_sequence (head + state), as Agent does
  nanojev          per prediction: shared prefix (state + question) + one candidate leaf per option;
                   counted as prefix + candidates (tokens that must be read), and also as the sum of
                   the three full leaves (upper bound if the prefix is not shared)
Kev-4B is not recounted: its server already reports the whole request (state once + 3 questions).

No GPU and no model weights: only tokenizers are downloaded.
Usage: python tokens_entrada.py <items.jsonl> <out.csv> [models...]
Output: one row per (model, id, condition, variant): passes, tokens (and tokens_max for nanojev).
"""
import csv
import json
import os
import sys
import traceback

from transformers import AutoTokenizer

from common import fit_state, jobs, load_items
from config import LABELS, MODELS
from prompts import VARIANTS, choice_question, hypothesis, llm_messages, render_state


def twitter_preprocess(text):  # copy of run_hf.twitter_preprocess (run_hf refuses to import without CUDA)
    return " ".join("@user" if w.startswith("@") and len(w) > 1 else "http" if w.startswith("http") else w
                    for w in text.split(" "))


def tok_of(cfg):
    return AutoTokenizer.from_pretrained(cfg["hf"], revision=cfg["revision"])


def count_classifier(cfg, items):
    tok = tok_of(cfg)
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    for it, c, v in jobs(items, ["none"]):
        state, _ = fit_state(it, c, n, cfg["max_tokens"] - 2)
        ids = tok(twitter_preprocess(state), truncation=True, max_length=cfg["max_tokens"])["input_ids"]
        yield it, c, v, 1, len(ids), ""


def count_nli(cfg, items):
    tok = tok_of(cfg)
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    for it, c, v in jobs(items, ["v1"]):
        state, _ = fit_state(it, c, n, cfg["max_tokens"] - 40)
        total = sum(len(tok(state, hypothesis(it, lab), truncation="only_first",
                            max_length=cfg["max_tokens"])["input_ids"]) for lab in LABELS)
        yield it, c, v, len(LABELS), total, ""


def count_llm(cfg, items):
    tok = tok_of(cfg)
    for it, c, v in jobs(items, VARIANTS):
        p = tok.apply_chat_template(llm_messages(render_state(it, c), it, v), add_generation_prompt=True,
                                    tokenize=False, **cfg["chat_kwargs"])
        yield it, c, v, 1, len(tok(p, add_special_tokens=False)["input_ids"]), ""


def count_laya(cfg, items):
    from huggingface_hub import snapshot_download
    from laya.agent import Agent, _fix_tokenizer_config
    from laya.common import build_sequence
    d = snapshot_download(cfg["hf"], revision=cfg["revision"],
                          allow_patterns=["tokenizer/*", "rl_agent_config.json"])
    _fix_tokenizer_config(d)
    acfg = json.load(open(os.path.join(d, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    stok = AutoTokenizer.from_pretrained(cfg["state_tokenizer"])
    n = lambda s: len(stok(s, add_special_tokens=False)["input_ids"])
    max_len, head = acfg.get("max_len", 512), acfg.get("head_max_len", 192)
    for it, c, v in jobs(items, VARIANTS):
        state, _ = fit_state(it, c, n, cfg["state_max_tokens"])
        q = Agent._to_internal(choice_question(it, v))
        seq, _ = build_sequence(tok, state, q, max_len, head, option_order=q.get("option_order"))
        yield it, c, v, 1, len(seq), ""


def count_nanojev(cfg, items):
    from huggingface_hub import snapshot_download
    d = snapshot_download(cfg["hf"], revision=cfg["revision"], allow_patterns=["tokenizer/*"])
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    enc = lambda s: len(tok.encode(s, add_special_tokens=False))
    for it, c, v in jobs(items, VARIANTS):
        state, _ = fit_state(it, c, n, cfg["state_max_tokens"])
        q = choice_question(it, v)  # same text as predict_toy_decisions.prepare_examples
        prefix = enc(f"State:\n{state}\n") + enc(f"Question type: {q['type']}\nQuestion:\n{q['instructions']}\n")
        cands = [enc(f"Candidate:\n{k}: {d_}\nDecision:") + 1 for k, d_ in q["criteria"].items()]  # +1 EOS
        yield it, c, v, len(cands), prefix + sum(cands), sum(prefix + x for x in cands)


COUNTERS = {"classifier": count_classifier, "nli": count_nli, "llm": count_llm,
            "laya": count_laya, "nanojev": count_nanojev}


def main():
    items_path, out = sys.argv[1], sys.argv[2]
    models = sys.argv[3:] or [m for m in MODELS if MODELS[m]["kind"] in COUNTERS]
    items = load_items(items_path)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "id", "condition", "variant", "passes", "tokens", "tokens_max"])
        for m in models:
            cfg = MODELS[m]
            k = 0
            try:  # one model failing must not stop the others
                for it, c, v, passes, toks, tmax in COUNTERS[cfg["kind"]](cfg, items):
                    w.writerow([m, it["dataset"], it["id"], c, v, passes, toks, tmax])
                    k += 1
                print(f"{m}: {k} rows", flush=True)
            except Exception:  # noqa: BLE001
                print(f"{m}: FAILED after {k} rows\n{traceback.format_exc()}", flush=True)
            f.flush()


if __name__ == "__main__":
    main()
