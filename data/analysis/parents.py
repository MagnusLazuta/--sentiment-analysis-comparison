"""MT-CSD parent comment of each item, when the parent is itself an item of the dataset.

Usage: python parents.py <items.jsonl of the main run> <out.csv>
Parent = the item of the same thread whose text equals the item's last context turn.
Used to test whether, with context, the models copy the stance of the comment above.
"""
import csv
import json
import sys


def main(src, out):
    its = [json.loads(line) for line in open(src, encoding="utf-8")]
    its = [i for i in its if i["dataset"] == "mtcsd"]
    by_text = {}
    for i in its:
        by_text.setdefault((i["context"][0]["text"], i["text"]), i)
    rows = []
    for i in its:
        p = by_text.get((i["context"][0]["text"], i["context"][-1]["text"]))
        if p and p["id"] != i["id"]:
            rows.append({"id": i["id"], "parent_id": p["id"], "parent_gold": p["gold"]})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "parent_id", "parent_gold"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} of {len(its)} items have their parent in the dataset -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
