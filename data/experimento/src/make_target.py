"""Sarcasm_NLP with the annotated target (follow-up C, 2026-11-21).

Usage: python make_target.py <items.jsonl of the main run> <out.jsonl>

The main run asked Sarcasm_NLP "How does the author of the tweet feel about the topic?"
without naming the topic. The dataset paper (Bhargava et al., arXiv 2504.05603, sec. 3.2)
says annotators judged sentiment "in relation to nuclear power". This script writes, for
every Sarcasm_NLP item:
  - "<id>#nuc"     : target = "nuclear power", original prompt;
  - "<id>#nucswap" : target = "nuclear power", option descriptions swapped (as make_extra.py).
Both run no_context only (the dataset has no context). Nothing else changes.
"""
import json
import sys

TARGET = "nuclear power"


def main(src, out):
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            it = json.loads(line)
            if it["dataset"] != "sarcasm_nlp" or it.get("ablation_k") or it.get("extra_condition"):
                continue
            base = {**it, "target": TARGET, "source_id": it["id"]}
            f.write(json.dumps({**base, "id": f"{it['id']}#nuc", "extra": "nuc"}) + "\n")
            f.write(json.dumps({**base, "id": f"{it['id']}#nucswap", "extra": "nucswap",
                                "swap_criteria": True}) + "\n")
            n += 2
    print(f"{n} target items written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
