"""Target description vs bare target name on MT-CSD (2026-11-22).

New run 20261122-204843-839743 (job 839743): same 848 MT-CSD test items (Tesla + SpaceX) with the
target replaced by a short description (src/make_targetdesc.py; ids "<id>#desc"). Baseline: main
run 20261117-181106-838446, same items, same models, same option orders.
Accuracy per model x condition (no_context / with_context), all option orders pooled.
Delta = desc - base, paired on (id, condition, variant); 95% CI by thread bootstrap
(mtcsd_threads-20261117-181106-838446.csv, 13 threads), 2,000 resamples, seed 5.
Subsets: by target; comments that mention Elon/Musk.

Run: python3 alvo_desc.py   Output: alvo_desc-20261122.txt
"""
import collections, csv, json, random, re

NEW = "../results/20261122-204843-839743"
BASE = "../results/20261117-181106-838446"
MODELS = ["twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya"]
random.seed(5)
B = 2000

items = {}
for l in open(f"{BASE}/data/items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    if it["dataset"] == "mtcsd":
        items[it["id"]] = it
new_ids = {json.loads(l)["id"].removesuffix("#desc") for l in open(f"{NEW}/data/items.jsonl", encoding="utf-8")}
thr = {r["id"]: r["thread"] for r in csv.DictReader(open("mtcsd_threads-20261117-181106-838446.csv"))}


def load(path, strip):
    out = {}
    for l in open(path, encoding="utf-8"):
        p = json.loads(l)
        i = p["id"].removesuffix("#desc") if strip else p["id"]
        if i in new_ids:
            out[(i, p["condition"], p["variant"])] = p["pred"]
    return out


def boot(rows):  # rows: (thread, diff)
    g = collections.defaultdict(list)
    for t, d in rows:
        g[t].append(d)
    keys = list(g)
    ms = []
    for _ in range(B):
        s = [x for k in random.choices(keys, k=len(keys)) for x in g[k]]
        ms.append(sum(s) / len(s))
    ms.sort()
    return 100 * ms[int(0.025 * B)], 100 * ms[int(0.975 * B) - 1]


elon = {i for i in new_ids if re.search(r"\belon\b|\bmusk", items[i]["text"], re.I)}
subsets = {"todos": new_ids,
           "Tesla": {i for i in new_ids if items[i]["target"] == "Tesla"},
           "SpaceX": {i for i in new_ids if items[i]["target"] == "SpaceX"},
           "menciona Elon/Musk": elon}
out = open("alvo_desc-20261122.txt", "w", encoding="utf-8")
out.write(f"MT-CSD, {len(new_ids)} items. Accuracy (%) base -> desc, delta [95% CI thread bootstrap].\n")
for sname, ids in subsets.items():
    out.write(f"\n== {sname} (n = {len(ids)})\n")
    for m in MODELS:
        nw, bs = load(f"{NEW}/preds/{m}.jsonl", True), load(f"{BASE}/preds/{m}.jsonl", False)
        for cond in ("no_context", "with_context"):
            keys = [k for k in nw if k[1] == cond and k[0] in ids and k in bs]
            a = [bs[k] == items[k[0]]["gold"] for k in keys]
            b = [nw[k] == items[k[0]]["gold"] for k in keys]
            lo, hi = boot([(thr[k[0]], y - x) for k, x, y in zip(keys, a, b)])
            out.write(f"{m:16s} {cond:13s} N={len(keys):5d}  {100*sum(a)/len(a):5.1f} -> {100*sum(b)/len(b):5.1f}  "
                      f"d={100*(sum(b)-sum(a))/len(a):+5.1f} [{lo:+.1f}; {hi:+.1f}]\n")
    # answer distribution shift (all subsets, with_context only for brevity)
out.write("\n== Answer shares (%) all items, base -> desc\n")
for m in MODELS:
    nw, bs = load(f"{NEW}/preds/{m}.jsonl", True), load(f"{BASE}/preds/{m}.jsonl", False)
    for cond in ("no_context", "with_context"):
        keys = [k for k in nw if k[1] == cond and k in bs]
        cb, cn = collections.Counter(bs[k] for k in keys), collections.Counter(nw[k] for k in keys)
        s = "  ".join(f"{c[:3]} {100*cb[c]/len(keys):4.1f}->{100*cn[c]/len(keys):4.1f}" for c in ("negative", "neutral", "positive"))
        out.write(f"{m:16s} {cond:13s} {s}\n")
gc = collections.Counter(items[i]["gold"] for i in new_ids)
out.write("gold: " + "  ".join(f"{c[:3]} {100*gc[c]/len(new_ids):4.1f}" for c in ("negative", "neutral", "positive")) + "\n")
out.close()
print(open("alvo_desc-20261122.txt").read())
