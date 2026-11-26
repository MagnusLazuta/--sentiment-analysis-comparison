"""Option-description swap for the context conditions (follow-up B, 2026-11-21).

Usage: python make_swap.py <items.jsonl of a finished run> <out.jsonl>

make_extra.py already swaps the descriptions for no_context. This script completes the
design for every with_context input:
  - source = main run items: "<id>#swapctx" for every item with context, with_context only;
  - source = ablation run items (field "ablation_k"): "<id>#swap" for every copy (the
    ablation copies already run with_context only, see prompts.conditions).
With "swap_criteria" the LLMs see the option descriptions and the System One models see
only the label words (prompts.llm_messages / prompts.choice_question).
"""
import json
import sys


def main(src, out):
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            it = json.loads(line)
            if it.get("ablation_k"):
                new = {**it, "id": f"{it['id']}#swap", "swap_criteria": True}
            elif it["context"]:
                new = {**it, "id": f"{it['id']}#swapctx", "source_id": it["id"], "extra": "swapctx",
                       "extra_condition": "with_context", "swap_criteria": True}
            else:
                continue
            f.write(json.dumps(new) + "\n")
            n += 1
    print(f"{n} swap items written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
