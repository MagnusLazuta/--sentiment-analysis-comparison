"""One table for the follow-up experiments of 2026-11-21 (make_extra.py, make_swap.py).

Usage: python followup_dados.py <main RUN_ID> <ablation RUN_ID> <extra RUN_ID> <swapctx RUN_ID> <swapabl RUN_ID> <out.csv>
  main     : original run (no_context = k 0, with_context = k K), prompt "orig"
  ablation : make_ablation.py copies "<id>#k<k>", prompt "orig"
  extra    : make_extra.py copies "<id>#oldest", "<id>#noparent" (prompt "orig") and
             "<id>#swap" (no_context, prompt "swap")
  swapctx  : make_swap.py on the main run, "<id>#swapctx" (with_context = k K, prompt "swap")
  swapabl  : make_swap.py on the ablation run, "<id>#k<k>#swap" (prompt "swap")
Writes one row per (model, item, variant, prompt, cond), where cond is "k<k>" (k = 0..K),
"oldest" or "noparent". prompt "swap" = the LLMs see the option descriptions and the System
One models see only the label words (prompts.py). NanoJev is not part of the follow-ups.
"""
import csv
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
MODELS = ["twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya"]


def read_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")] if p.exists() else []


def parse(rid, condition, items):
    """(source id, prompt, cond) for one prediction id."""
    base, *tags = rid.split("#")
    K = len(items[base]["context"])
    prompt = "swap" if tags and tags[-1] in ("swap", "swapctx") else "orig"
    tags = [t for t in tags if t not in ("swap", "swapctx")]
    if tags and tags[0].startswith("k"):
        return base, prompt, f"k{int(tags[0][1:])}"
    if tags:
        return base, prompt, tags[0]
    return base, prompt, "k0" if condition == "no_context" else f"k{K}"


def main(main_run, abl, extra, swapctx, swapabl, out):
    items = {i["id"]: i for i in read_jsonl(R / main_run / "data" / "items.jsonl")}
    seen, n = set(), 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "dataset", "id", "sarcastic", "gold", "literal", "K", "prompt", "cond", "variant",
                    "pred", "correct", "confidence", "p_negative", "p_neutral", "p_positive"])
        for run in (main_run, abl, extra, swapctx, swapabl):
            for m in MODELS:
                for r in read_jsonl(R / run / "preds" / f"{m}.jsonl"):
                    sid, prompt, cond = parse(r["id"], r["condition"], items)
                    key = (m, sid, r["variant"], prompt, cond)
                    if key in seen:  # the classifier / NLI "swap" copies repeat the main run
                        continue
                    seen.add(key)
                    it, p = items[sid], r["probs"]
                    w.writerow([m, it["dataset"], sid, int(it["sarcastic"]), it["gold"], it.get("literal") or "",
                                len(it["context"] or []), prompt, cond, r["variant"], r["pred"],
                                int(r["pred"] == it["gold"]), max(p.values()),
                                p["negative"], p["neutral"], p["positive"]])
                    n += 1
    print(f"{n} rows -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:7])
