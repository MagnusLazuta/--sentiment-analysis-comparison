"""Main run with Sarcasm_NLP replaced by the annotated-target run (decision of 2026-11-21).

Usage: python merge_target.py <main RUN_ID> <target RUN_ID>
Writes results/<target RUN_ID>/merged/{data/items.jsonl, preds/<model>.jsonl}:
  - every non-Sarcasm_NLP item and prediction of the main run, unchanged;
  - Sarcasm_NLP items with target "nuclear power" and the "#nuc" predictions of the target
    run (id suffix removed). NanoJev was not rerun: its Sarcasm_NLP rows are dropped.
Then: python ../src/evaluate.py merged/data/items.jsonl merged/preds merged/eval
Also writes results/<target RUN_ID>/eval/followup_nuc.csv: followup.csv of run
20261121-172919-839457 with its Sarcasm_NLP rows replaced by "#nuc" (prompt orig) and
"#nucswap" (prompt swap) rows of target.csv.
"""
import csv
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def main(main_run, target_run):
    out = R / target_run / "merged"
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    n_it = 0
    with open(out / "data" / "items.jsonl", "w", encoding="utf-8") as f:
        for line in open(R / main_run / "data" / "items.jsonl", encoding="utf-8"):
            it = json.loads(line)
            if it["dataset"] == "sarcasm_nlp":
                it["target"] = "nuclear power"
            f.write(json.dumps(it) + "\n")
            n_it += 1
    for p in sorted((R / main_run / "preds").glob("*.jsonl")):
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        keep = [r for r in rows if not r["id"].startswith("snlp:")]
        new = []
        tp = R / target_run / "preds" / p.name
        if tp.exists():
            for l in open(tp, encoding="utf-8"):
                r = json.loads(l)
                if r["id"].endswith("#nuc"):
                    new.append({**r, "id": r["id"][:-4]})
        with open(out / "preds" / p.name, "w", encoding="utf-8") as f:
            for r in keep + new:
                f.write(json.dumps(r) + "\n")
        print(p.stem, "kept", len(keep), "snlp replaced", len(new), "dropped", len(rows) - len(keep))
    # follow-up table
    fu = R / "20261121-172919-839457" / "eval" / "followup.csv"
    tg = R / target_run / "eval" / "target.csv"
    with open(R / target_run / "eval" / "followup_nuc.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        rd = csv.reader(open(fu, encoding="utf-8"))
        head = next(rd)
        w.writerow(head)
        n_old = n_new = 0
        for row in rd:
            if row[1] != "sarcasm_nlp":
                w.writerow(row)
            else:
                n_old += 1
        for r in csv.DictReader(open(tg, encoding="utf-8")):
            if r["target"] != "nuclear":
                continue
            w.writerow([r["model"], "sarcasm_nlp", r["id"], r["sarcastic"], r["gold"], "", 0, r["prompt"], "k0",
                        r["variant"], r["pred"], r["correct"], r["confidence"], r["p_negative"], r["p_neutral"],
                        r["p_positive"]])
            n_new += 1
    print("followup_nuc: snlp rows", n_old, "->", n_new, "; items", n_it)


if __name__ == "__main__":
    main(*sys.argv[1:3])
