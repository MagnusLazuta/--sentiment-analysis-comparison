"""Summarise diag_firsttoken.py outputs: first-token vs full-word label scores.

Usage: python diag_firsttoken_sum.py <dir with qwen3.5-4b.jsonl, mistral-7b.jsonl>
Per model x dataset x condition x variant: N, same prediction (%), max and mean |dp|,
accuracy with each score, and for MT-CSD the k0 -> kK accuracy change with each score.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

L = ["negative", "neutral", "positive"]
arg = lambda p: max(L, key=p.get)

for f in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["condition"], r["variant"])].append(r)
    print(f"\n== {f.stem}  (N rows {len(rows)})")
    print(f"{'dataset':12} {'cond':13} {'var':3} {'N':>5} {'same%':>6} {'max|dp|':>8} {'mean|dp|':>9} {'acc_first':>9} {'acc_full':>8} {'massF':>6} {'massW':>6}")
    for k, rs in sorted(g.items()):
        dp = [abs(r["first"][l] - r["full"][l]) for r in rs for l in L]
        same = sum(arg(r["first"]) == arg(r["full"]) for r in rs) / len(rs)
        af = sum(arg(r["first"]) == r["gold"] for r in rs) / len(rs)
        aw = sum(arg(r["full"]) == r["gold"] for r in rs) / len(rs)
        mf = sum(r["mass_first"] for r in rs) / len(rs)
        mw = sum(r["mass_full"] for r in rs) / len(rs)
        print(f"{k[0]:12} {k[1]:13} {k[2]:3} {len(rs):5} {100*same:6.1f} {max(dp):8.4f} {sum(dp)/len(dp):9.4f} {100*af:9.1f} {100*aw:8.1f} {mf:6.3f} {mw:6.3f}")
    for v in ("v1", "v2", "v3"):
        a = {r["id"]: r for r in g.get(("mtcsd", "no_context", v), [])}
        b = {r["id"]: r for r in g.get(("mtcsd", "with_context", v), [])}
        ids = [i for i in a if i in b]
        if not ids:
            continue
        for s in ("first", "full"):
            d = sum((arg(b[i][s]) == b[i]["gold"]) - (arg(a[i][s]) == a[i]["gold"]) for i in ids) / len(ids)
            print(f"MT-CSD {v} k0->kK ({s}): {100*d:+.1f} (N {len(ids)})")
