"""Why are some models pulled to the conversation tone? (2026-11-22)

MT-CSD, fixed cohort of items with >= 4 context turns (ablation run 20261118-094702-838561,
k = 1..4; k = 0 from main run 20261117-181106-838446, no_context). All option orders pooled.
Tone of the conversation and groups as in tom_discordante.R (conversa_tom-20261122.csv).

A. Size of the pull per model on comments that disagree with the conversation:
   delta (k4 - k0) of "answer = tone" and of accuracy, paired on (id, variant).
B. What does the k4 answer copy? Agreement of the k4 answer with
   (i) the model's own k0 answer for the comment, (ii) the model's own k0 answer for the
   parent reply (last context turn, when it is an MT-CSD item).
C. Dose: discordant items split by the share of the k4 input that is the comment
   (twitter-roberta token counts k0 / k4, same measure for every model; below / above median).
95% CI: thread bootstrap (13 threads), 2,000 resamples, seed 5.
Run: python3 tom_mecanismo.py   Output: tom_mecanismo-20261122.txt
"""
import collections, csv, json, random, statistics

ABL = "../results/20261118-094702-838561"
MAIN = "../results/20261117-181106-838446"
MODELS = ["twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya"]
EN = {"negativo": "negative", "neutro": "neutral", "positivo": "positive"}
random.seed(5)
B = 2000

items = {}
for l in open(f"{MAIN}/data/items.jsonl", encoding="utf-8"):
    it = json.loads(l)
    if it["dataset"] == "mtcsd":
        items[it["id"]] = it
chain = {(it["target"],) + tuple(t["text"] for t in it["context"]) + (it["text"],): i for i, it in items.items()}
parent = {}
for i, it in items.items():
    if len(it["context"]) >= 2:
        parent[i] = chain.get((it["target"],) + tuple(t["text"] for t in it["context"]))
tone = {r["id"]: EN[r["maioria"]] for r in csv.DictReader(open("conversa_tom-20261122.csv", encoding="utf-8"))
        if r["dataset"] == "mtcsd" and r["maioria"] in EN}
thr = {r["id"]: r["thread"] for r in csv.DictReader(open("mtcsd_threads-20261117-181106-838446.csv"))}

K = collections.defaultdict(int)  # max k per (model, variant, id)
pred = {}  # (model, variant, id, k) -> pred
for r in csv.DictReader(open(f"{ABL}/eval/ablation.csv", encoding="utf-8")):
    if r["dataset"] == "mtcsd" and r["model"] in MODELS:
        pred[(r["model"], r["variant"], r["id"], int(r["k"]))] = r["pred"]
        K[(r["model"], r["variant"], r["id"])] = max(K[(r["model"], r["variant"], r["id"])], int(r["k"]))
k0 = {}  # (model, variant, id) -> no_context pred, main run
for m in MODELS:
    for l in open(f"{MAIN}/preds/{m}.jsonl", encoding="utf-8"):
        p = json.loads(l)
        if p["id"] in items and p["condition"] == "no_context":
            k0[(m, p["variant"], p["id"])] = p["pred"]
tok = {}  # twitter-roberta token counts
for l in open(f"{ABL}/preds/twitter-roberta.jsonl", encoding="utf-8"):
    p = json.loads(l)
    if p["id"].endswith("#k4") and p["id"].startswith("mtcsd"):
        tok[p["id"][:-3]] = p["tokens"]
tok0 = {json.loads(l)["id"]: json.loads(l)["tokens"] for l in open(f"{MAIN}/preds/twitter-roberta.jsonl", encoding="utf-8")
        if '"no_context"' in l and l.startswith('{"id": "mtcsd')}
for l in open(f"{MAIN}/preds/twitter-roberta.jsonl", encoding="utf-8"):  # k = K = 4 comes from the main run
    p = json.loads(l)
    if p["id"] in items and p["condition"] == "with_context" and len(items[p["id"]]["context"]) == 4:
        tok[p["id"]] = p["tokens"]
share = {i: tok0[i] / tok[i] for i in tok if i in tok0}


def boot(rows):  # rows: (thread, value) -> mean, lo, hi (in %)
    g = collections.defaultdict(list)
    for t, v in rows:
        g[t].append(v)
    ks = list(g)
    ms = sorted(statistics.fmean([x for k in random.choices(ks, k=len(ks)) for x in g[k]]) for _ in range(B))
    return 100 * statistics.fmean(v for _, v in rows), 100 * ms[int(0.025 * B)], 100 * ms[int(0.975 * B) - 1]


fmt = lambda t: f"{t[0]:+6.1f} [{t[1]:+6.1f}; {t[2]:+6.1f}]"
fm = lambda t: f"{t[0]:5.1f} [{t[1]:5.1f}; {t[2]:5.1f}]"
rows = collections.defaultdict(list)
for (m, v, i), kmax in K.items():
    if kmax < 4 or i not in tone or (m, v, i) not in k0:
        continue
    g = items[i]["gold"]
    grp = "concorda" if g == tone[i] else "discorda"
    rows[(m, grp)].append(dict(i=i, t=thr[i], a0=k0[(m, v, i)], a4=pred[(m, v, i, 4)], tone=tone[i], gold=g,
                               par=k0.get((m, v, parent.get(i))), sh=share.get(i)))

o = open("tom_mecanismo-20261122.txt", "w", encoding="utf-8")
o.write("A. Discordant comments: delta k4 - k0 (pp) [95% CI]\n")
o.write(f"{'model':16s} {'N':>4s}  {'answer = tone':24s} {'accuracy':24s}  concordant: answer = tone\n")
for m in MODELS:
    d, c = rows[(m, "discorda")], rows[(m, "concorda")]
    f = boot([(r["t"], (r["a4"] == r["tone"]) - (r["a0"] == r["tone"])) for r in d])
    a = boot([(r["t"], (r["a4"] == r["gold"]) - (r["a0"] == r["gold"])) for r in d])
    cc = boot([(r["t"], (r["a4"] == r["tone"]) - (r["a0"] == r["tone"])) for r in c])
    o.write(f"{m:16s} {len(d):4d}  {fmt(f)} {fmt(a)}  {fmt(cc)}\n")

o.write("\nB. All cohort items with a clear tone and an annotated parent: % of k4 answers equal to ...\n")
o.write(f"{'model':16s} {'N':>4s}  {'own k0 answer (comment)':24s} {'own k0 answer (parent)':24s}  "
        "among items where the two differ: % parent\n")
for m in MODELS:
    rr = [r for g in ("concorda", "discorda") for r in rows[(m, g)] if r["par"]]
    x = boot([(r["t"], r["a4"] == r["a0"]) for r in rr])
    y = boot([(r["t"], r["a4"] == r["par"]) for r in rr])
    dd = [r for r in rr if r["a0"] != r["par"] and r["a4"] in (r["a0"], r["par"])]
    z = boot([(r["t"], r["a4"] == r["par"]) for r in dd])
    o.write(f"{m:16s} {len(rr):4d}  {fm(x)} {fm(y)}  {fm(z)} (n = {len(dd)})\n")

med = statistics.median(share[r["i"]] for r in rows[("twitter-roberta", "discorda")])
o.write(f"\nC. Discordant comments, answer = tone at k4 (%), by share of the k4 input that is the comment "
        f"(median {100*med:.1f}%)\n")
for m in MODELS:
    lo = [r for r in rows[(m, "discorda")] if r["sh"] <= med]
    hi = [r for r in rows[(m, "discorda")] if r["sh"] > med]
    o.write(f"{m:16s} small share (n={len(lo):3d}) {fm(boot([(r['t'], r['a4'] == r['tone']) for r in lo]))}   "
            f"large share (n={len(hi):3d}) {fm(boot([(r['t'], r['a4'] == r['tone']) for r in hi]))}\n")
o.close()
print(open("tom_mecanismo-20261122.txt").read())
