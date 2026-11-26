"""MT-CSD thread of each item (the opening post), for cluster-aware statistics.

Usage: python threads.py <items.jsonl of the main run> <out.csv>
Thread = text of the first context turn (the post that opens the Reddit thread). Items
of the same thread share context, so they are not independent: MT-CSD has 13 threads.
"""
import csv
import json
import sys


def main(src, out):
    rows, ids = [], {}
    for line in open(src, encoding="utf-8"):
        it = json.loads(line)
        if it["dataset"] != "mtcsd":
            continue
        post = it["context"][0]["text"]
        ids.setdefault(post, f"T{len(ids) + 1:02d}")
        rows.append({"id": it["id"], "thread": ids[post], "post": post[:80]})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "thread", "post"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} items, {len(ids)} threads -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
