"""Context-length ablation items.

Usage: python make_ablation.py <items.jsonl of a finished run> <out.jsonl>
For every item with K context turns, writes copies keeping only the last k turns,
k = 1..K-1, with id "<id>#k<k>" and field "ablation_k". These copies run only in the
with_context condition (see prompts.conditions). k = 0 (no_context) and k = K (whole
context) are the two conditions of the source run, so they are not repeated here.
"""
import json
import sys


def main(src, out):
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            it = json.loads(line)
            K = len(it["context"])
            for k in range(1, K):
                f.write(json.dumps({**it, "id": f"{it['id']}#k{k}", "source_id": it["id"],
                                    "ablation_k": k, "context_len": K,
                                    "context": it["context"][-k:]}) + "\n")
                n += 1
    print(f"{n} ablation items written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
