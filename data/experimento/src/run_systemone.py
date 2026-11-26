"""Stage 2b: inference with System One decision models (Kev-4B, Laya, NanoJev).

Each (item, condition) is one request with three Choice questions (v1..v3, same
wording, different option order); the models answer each question independently.
A failing request is logged to <out>.failures.jsonl and skipped (retried on re-run).

Usage:
  python run_systemone.py --model kev-4b  --items ... --out ... [--url http://127.0.0.1:8008]
  python run_systemone.py --model laya    --items ... --out ...
  python run_systemone.py --model nanojev --items ... --out ... --code-dir <NanoJev clone> --ckpt-dir <dir>
"""
import argparse
import json
import sys
import time
import traceback
import urllib.request

from common import Writer, fit_state, load_items
from config import MODELS
from prompts import VARIANTS, choice_question, conditions, render_state, to_canonical


def kev_backend(cfg, a):
    def ask(it, c, questions):
        body = json.dumps({"state": render_state(it, c), "model": "kev-latest",
                           "questions": questions}).encode()
        req = urllib.request.Request(f"{a.url}/v1/systemone", data=body,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        return resp["answers"], {"tokens": resp.get("usage", {}).get("input_tokens")}
    return ask


def laya_backend(cfg, a):
    from laya import Router
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["state_tokenizer"])
    # force the English ModernBERT checkpoint (the default routing sends short or
    # mixed texts to the multilingual model) and pin its revision
    router = Router(device="cuda", default="english", vega=False, revision=cfg["revision"])
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    def ask(it, c, questions):
        state, info = fit_state(it, c, count, cfg["state_max_tokens"])
        res = router.predict(state, questions, model="english")
        routed = res.get("routing", {}).get("model")
        if routed != "english":
            raise RuntimeError(f"Laya routed to {routed!r}, expected 'english'")
        return res["answers"], {**info, "routed_to": routed}
    return ask


def nanojev_backend(cfg, a):
    sys.path.insert(0, f"{a.code_dir}/scripts")
    from predict_toy_decisions import DecisionPredictor
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{a.ckpt_dir}/tokenizer")
    engine = DecisionPredictor(a.ckpt_dir, max_length=cfg["max_length"], precision="bf16",
                               disable_native_triton=True)
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    def ask(it, c, questions):
        # NanoJev refuses (does not truncate) inputs over max_length: cut the context first
        state, info = fit_state(it, c, count, cfg["state_max_tokens"])
        res = engine.predict({"states": [{"id": "0", "state": state, "questions": questions}]})
        return res["states"][0]["answers"], info
    return ask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--url", default="http://127.0.0.1:8008")
    p.add_argument("--code-dir")
    p.add_argument("--ckpt-dir")
    a = p.parse_args()
    cfg = MODELS[a.model]
    ask = {"kev": kev_backend, "laya": laya_backend, "nanojev": nanojev_backend}[cfg["kind"]](cfg, a)

    w = Writer(a.out)
    todo = []
    for it in load_items(a.items, a.limit):
        for c in conditions(it):
            vs = [v for v in VARIANTS if (it["id"], c, v) not in w.done]
            if vs:
                todo.append((it, c, vs))
    print(f"{a.model}: {len(todo)} requests to run", flush=True)
    t0, failed = time.time(), 0
    for k, (it, c, vs) in enumerate(todo):
        try:
            answers, info = ask(it, c, {v: choice_question(it, v) for v in vs})
            for v in vs:
                w.write(it, c, v, to_canonical(it, answers[v]["probabilities"]), **info)
        except Exception as e:  # noqa: BLE001 - one bad request must not stop the model
            failed += 1
            with open(f"{a.out}.failures.jsonl", "a") as f:
                f.write(json.dumps({"id": it["id"], "condition": c, "variants": vs, "error": repr(e),
                                    "trace": traceback.format_exc()[-2000:]}) + "\n")
            if failed >= 50 and failed > 0.5 * (k + 1):
                raise SystemExit(f"{a.model}: {failed} of {k + 1} requests failed; stopping this model")
        if k % 200 == 0:
            print(f"  {k + 1}/{len(todo)} {time.time() - t0:.0f}s failed={failed}", flush=True)
    w.close()
    print(f"{a.model}: done in {time.time() - t0:.0f}s, failed={failed}", flush=True)


if __name__ == "__main__":
    main()
