"""Build one table for the context-length ablation.

Usage: python ablacao_dados.py <source RUN_ID> <ablation RUN_ID>
Joins, for every item with context, the predictions with k = 0 (no_context, source
run), k = 1..K-1 (ablation run) and k = K (with_context, source run). Writes
results/<ablation RUN_ID>/eval/ablation.csv with one row per
(model, item, variant, k).
"""
import csv
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def read_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def main(src, abl):
    items = {i["id"]: i for i in read_jsonl(R / src / "data" / "items.jsonl") if i["context"]}
    rows = []
    for f in sorted((R / abl / "preds").glob("*.jsonl")):
        model = f.stem
        for r in read_jsonl(R / src / "preds" / f.name):
            if r["id"] in items:
                K = len(items[r["id"]]["context"])
                rows.append((model, r["id"], r, 0 if r["condition"] == "no_context" else K))
        for r in read_jsonl(f):
            sid, k = r["id"].split("#k")
            rows.append((model, sid, r, int(k)))
    out = R / abl / "eval" / "ablation.csv"
    out.parent.mkdir(exist_ok=True)
    seen = set()
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "dataset", "id", "sarcastic", "gold", "K", "k", "variant", "pred", "correct",
                    "p_negative", "p_neutral", "p_positive"])
        for model, sid, r, k in rows:
            it = items[sid]
            key = (model, sid, r["variant"], k)
            assert key not in seen, key
            seen.add(key)
            p = r["probs"]
            w.writerow([model, it["dataset"], sid, int(it["sarcastic"]), it["gold"], len(it["context"]), k,
                        r["variant"], r["pred"], int(r["pred"] == it["gold"]),
                        p["negative"], p["neutral"], p["positive"]])
    # completeness: every (model, item, variant) must have all k = 0..K
    by = {}
    for model, sid, r, k in rows:
        by.setdefault((model, sid, r["variant"]), set()).add(k)
    bad = [key for key, ks in by.items() if ks != set(range(len(items[key[1]]["context"]) + 1))]
    print(f"{len(rows)} rows, {len(by)} (model, item, variant) series, incomplete: {len(bad)} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
