"""Sarcasm_NLP follow-up D (make_issue.py): "#issue" and "#issuenote" predictions as rows in
the target.csv format (prompt "orig"; target "issue" or "issuenote").

Usage: python issue_dados.py <main RUN_ID> <issue RUN_ID> <target.csv> <out.csv>
Writes the rows of target.csv with prompt "orig" plus the new rows, so one table holds
"topic", "nuclear", "issue" and "issuenote". twitter-roberta was not rerun (its input does
not contain the question).
"""
import csv
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
MODELS = ["deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya"]


def main(main_run, issue_run, target_csv, out):
    items = {}
    for l in open(R / main_run / "data" / "items.jsonl", encoding="utf-8"):
        i = json.loads(l)
        if i["dataset"] == "sarcasm_nlp":
            items[i["id"]] = i
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        rd = csv.DictReader(open(target_csv, encoding="utf-8"))
        w.writerow(rd.fieldnames)
        for r in rd:
            if r["prompt"] == "orig":
                w.writerow([r[k] for k in rd.fieldnames]); n += 1
        for m in MODELS:
            p = R / issue_run / "preds" / f"{m}.jsonl"
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                base, _, tag = r["id"].partition("#")
                if tag not in ("issue", "issuenote"):
                    continue
                it, pr = items[base], r["probs"]
                w.writerow([m, base, int(it["sarcastic"]), it["gold"], it["subgroup"], r["variant"], "orig", tag,
                            r["pred"], int(r["pred"] == it["gold"]), max(pr.values()),
                            pr["negative"], pr["neutral"], pr["positive"]])
                n += 1
    print(n, "rows ->", out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
