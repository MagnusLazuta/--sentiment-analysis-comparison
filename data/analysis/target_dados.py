"""Sarcasm_NLP with and without the annotated target (follow-up C, 2026-11-21).

Usage: python target_dados.py <main RUN_ID> <extra RUN_ID> <target RUN_ID> <out.csv>
  main   : original run, target "the topic", prompt "orig"
  extra  : make_extra.py, "<id>#swap" = target "the topic", prompt "swap"
  target : make_target.py, "<id>#nuc" (prompt "orig") and "<id>#nucswap" (prompt "swap"),
           target "nuclear power"
One row per (model, item, variant, prompt, target). Sarcasm_NLP only, no_context only.
"""
import csv
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
MODELS = ["twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya"]
TAGS = {"": ("orig", "topic"), "swap": ("swap", "topic"), "nuc": ("orig", "nuclear"), "nucswap": ("swap", "nuclear")}


def read_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")] if p.exists() else []


def main(main_run, extra, target, out):
    items = {i["id"]: i for i in read_jsonl(R / main_run / "data" / "items.jsonl") if i["dataset"] == "sarcasm_nlp"}
    seen, n = set(), 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "id", "sarcastic", "gold", "topic", "variant", "prompt", "target",
                    "pred", "correct", "confidence", "p_negative", "p_neutral", "p_positive"])
        for run in (main_run, extra, target):
            for m in MODELS:
                for r in read_jsonl(R / run / "preds" / f"{m}.jsonl"):
                    base, _, tag = r["id"].partition("#")
                    if base not in items or tag not in TAGS or r["condition"] != "no_context":
                        continue
                    prompt, tgt = TAGS[tag]
                    key = (m, base, r["variant"], prompt, tgt)
                    if key in seen:
                        continue
                    seen.add(key)
                    it, p = items[base], r["probs"]
                    w.writerow([m, base, int(it["sarcastic"]), it["gold"], it["subgroup"], r["variant"], prompt, tgt,
                                r["pred"], int(r["pred"] == it["gold"]), max(p.values()),
                                p["negative"], p["neutral"], p["positive"]])
                    n += 1
    print(f"{n} rows -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
