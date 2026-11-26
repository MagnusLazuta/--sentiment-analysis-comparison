"""Follow-up experiment items (2026-11-21), built from the items of a finished run.

Usage: python make_extra.py <items.jsonl of a finished run> <out.jsonl>

A. Context composition (MT-CSD items with K >= 2 context turns), with_context only:
   "<id>#oldest"   context = [oldest turn]            (is the thread head what moves the answer?)
   "<id>#noparent" context = all turns but the last    (is the parent comment what moves it?)
   Field "extra" names the copy; "source_id" points to the original item.
B. Option descriptions swapped (every item), no_context only:
   "<id>#swap" with "swap_criteria": True. The LLMs then see the option descriptions that
   the System One models get in the main run, and the System One models get only the
   label words (see prompts.llm_messages / prompts.choice_question). The classifier and
   the NLI model ignore the flag, so their copies repeat the main run (determinism check).
"""
import json
import sys


def main(src, out):
    n = {"oldest": 0, "noparent": 0, "swap": 0}
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            it = json.loads(line)
            ctx = it["context"] or []
            if it["dataset"] == "mtcsd" and len(ctx) >= 2:
                for name, turns in (("oldest", ctx[:1]), ("noparent", ctx[:-1])):
                    f.write(json.dumps({**it, "id": f"{it['id']}#{name}", "source_id": it["id"],
                                        "extra": name, "extra_condition": "with_context",
                                        "context_len": len(ctx), "context": turns}) + "\n")
                    n[name] += 1
            f.write(json.dumps({**it, "id": f"{it['id']}#swap", "source_id": it["id"], "extra": "swap",
                                "extra_condition": "no_context", "swap_criteria": True}) + "\n")
            n["swap"] += 1
    print(f"extra items written to {out}: {n}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
