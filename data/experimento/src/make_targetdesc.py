"""MT-CSD with a short description of the target in the question (follow-up E, Q14, 2026-11-22).

Usage: python make_targetdesc.py <items.jsonl of the main run> <out.jsonl>

The main run only names the target ("Tesla", "SpaceX"); many comments talk about Elon Musk,
and linking him to the company needs prior knowledge. This script writes, for every MT-CSD
item, "<id>#desc" with target = "<name> (<description>)". The target string is used by the
question, the System One option descriptions and the NLI hypotheses, so all models get it.
Conditions: no_context and with_context (whole thread), as in the main run.
"""
import json
import sys

DESC = {
    "Tesla": "Tesla (the electric car company led by Elon Musk)",
    "SpaceX": "SpaceX (the rocket and spacecraft company led by Elon Musk)",
}


def main(src, out):
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            it = json.loads(line)
            if it["dataset"] != "mtcsd" or it.get("ablation_k") or it.get("extra_condition"):
                continue
            f.write(json.dumps({**it, "id": f"{it['id']}#desc", "source_id": it["id"],
                                "target": DESC[it["target"]], "extra": "desc"}) + "\n")
            n += 1
    print(f"{n} target-description items written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
