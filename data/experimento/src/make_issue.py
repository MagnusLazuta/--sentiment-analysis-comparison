"""Sarcasm_NLP with a broader target, as the gold seems to be annotated (follow-up D, 2026-11-22).

Usage: python make_issue.py <items.jsonl of the main run> <out.jsonl>

Follow-up C (make_target.py) asked about "nuclear power". Reading samples (agent "snlp",
2026-11-22) suggests the gold is the author's sentiment about the nuclear issue the tweet
discusses (power, weapons or policy), with news headlines labelled neutral (Bhargava et al.,
arXiv 2504.05603, sec. 3.2). For every Sarcasm_NLP item this script writes:
  - "<id>#issue"     : target = ISSUE, original prompt;
  - "<id>#issuenote" : target = ISSUE and question_note = NOTE (appended to the question for the
                       LLMs and the System One instructions; the NLI hypotheses do not use it).
No context; option descriptions as in the main run.
"""
import json
import sys

ISSUE = "the nuclear issue it discusses (nuclear power, nuclear weapons or nuclear policy)"
NOTE = "A news headline that only reports facts is neutral."


def main(src, out):
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            it = json.loads(line)
            if it["dataset"] != "sarcasm_nlp" or it.get("ablation_k") or it.get("extra_condition"):
                continue
            base = {**it, "target": ISSUE, "source_id": it["id"]}
            f.write(json.dumps({**base, "id": f"{it['id']}#issue", "extra": "issue"}) + "\n")
            f.write(json.dumps({**base, "id": f"{it['id']}#issuenote", "extra": "issuenote",
                                "question_note": NOTE}) + "\n")
            n += 2
    print(f"{n} issue items written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
