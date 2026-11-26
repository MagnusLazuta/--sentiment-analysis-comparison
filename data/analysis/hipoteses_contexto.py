"""Two hypotheses about context, on all data (2026-11-22).

H1. More context makes the model answer "neutral" more often.
H2. More context makes the model answer "negative" more often, depending on the tone of the
    conversation (negative or not).

Data: context ablation run 20261118-094702-838561 (eval/ablation.csv): every MT-CSD item
(848) and every MUStARD++ item (570 non-ironic, 601 ironic), all six models (NanoJev out),
all option orders, with k = 0 (no context) .. K (whole conversation). Items and context
texts: main run 20261117-181106-838446 (data/items.jsonl).

Comparisons are paired on the same items: share of answers of each class at k = 0, k = 1 and
k = K (whole conversation), all items (every item has k = 0, 1, K). 95% CI by bootstrap,
2,000 resamples, seed 5: by thread for MT-CSD (13 threads), by item for MUStARD++.

Tone of the conversation (H2), two measures:
  - MT-CSD, human label: gold stance of the comment directly above (parent), when the parent is
    itself an annotated reply; "post" when the parent is the thread head (not annotated).
  - Both datasets, lexicon: VADER compound of the whole context text; negative < -0.05,
    positive > 0.05, else neutral (standard VADER cut-offs). VADER is a proxy: it reads word
    valence, not stance.

Run: uv run --with pandas --with vaderSentiment python hipoteses_contexto.py
Outputs (this folder): hipoteses_contexto-20261122.txt, hip_contexto_{h1,h2}-20261122.csv
"""
import json
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

R = "../results"
SEED, B = 5, 2000
rng = np.random.default_rng(SEED)
MODELS = ["twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya"]

abl = pd.read_csv(f"{R}/20261118-094702-838561/eval/ablation.csv")
abl = abl[abl.model.isin(MODELS)].copy()
abl["grp"] = np.where(abl.dataset == "mtcsd", "MT-CSD",
                      np.where(abl.sarcastic == 1, "MUStARD++ irônico", "MUStARD++ não irônico"))
thr = pd.read_csv("mtcsd_threads-20261117-181106-838446.csv")[["id", "thread"]]
abl = abl.merge(thr, on="id", how="left")
abl["cl"] = np.where(abl.dataset == "mtcsd", abl.thread.astype(str), abl.id)

items = {}
for line in open(f"{R}/20261117-181106-838446/data/items.jsonl", encoding="utf-8"):
    it = json.loads(line)
    if it["dataset"] in ("mtcsd", "mustardpp"):
        items[it["id"]] = it

# --- tone of the conversation -------------------------------------------------------------
sia = SentimentIntensityAnalyzer()
tone = []
key = {}  # MT-CSD: (target, context texts..., text) -> gold, to find the parent comment
for i, it in items.items():
    if it["dataset"] == "mtcsd":
        key[(it["target"],) + tuple(t["text"] for t in it["context"]) + (it["text"],)] = it["gold"]
for i, it in items.items():
    ctx = it["context"] or []
    c = sia.polarity_scores(" ".join(t["text"] for t in ctx))["compound"] if ctx else 0.0
    vt = "negativo" if c < -0.05 else ("positivo" if c > 0.05 else "neutro")
    parent = None
    if it["dataset"] == "mtcsd":
        if len(ctx) == 1:
            parent = "post"
        else:
            g = key.get((it["target"],) + tuple(t["text"] for t in ctx))
            parent = {"negative": "contra", "neutral": "nenhuma", "positive": "a favor"}.get(g, "desconhecido")
    tone.append({"id": i, "vader": c, "tom_vader": vt, "pai": parent})
tone = pd.DataFrame(tone)

# --- paired shares k = 0, 1, K ----------------------------------------------------------------
abl = abl[(abl.k == 0) | (abl.k == 1) | (abl.k == abl.K)].copy()
abl["cond"] = np.where(abl.k == 0, "k0", np.where(abl.k == abl.K, "kK", "k1"))
# items with K = 1 have k1 == kK: keep both labels
one = abl[(abl.K == 1) & (abl.k == 1)].copy()
one["cond"] = "k1"
abl = pd.concat([abl, one])
abl = abl.drop_duplicates(["model", "variant", "id", "cond"])
w = abl.pivot_table(index=["model", "variant", "grp", "id", "cl", "gold"], columns="cond",
                    values="pred", aggfunc="first").reset_index()
w = w.merge(tone, on="id", how="left")


def boot(df, col_a, col_b, cls):
    """paired cluster bootstrap of share(b == cls) - share(a == cls), in points"""
    d = (df[col_b] == cls).astype(float).values - (df[col_a] == cls).astype(float).values
    cl = df["cl"].values
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=d)
    n = np.bincount(inv)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return 100 * d.mean(), 100 * np.quantile(m, 0.025), 100 * np.quantile(m, 0.975)


def table(df, by, conds=(("k0", "k1"), ("k0", "kK")), classes=("neutral", "negative", "positive")):
    rows = []
    for keys, g in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        r = dict(zip(by, keys), n=len(g))
        for cls in classes:
            r[f"{cls}_k0"] = 100 * (g.k0 == cls).mean()
            for a, b in conds:
                dlt, lo, hi = boot(g, a, b, cls)
                r[f"{cls}_{a}{b}"] = dlt
                r[f"{cls}_{a}{b}_lo"] = lo
                r[f"{cls}_{a}{b}_hi"] = hi
        rows.append(r)
    return pd.DataFrame(rows)


h1 = table(w, ["grp", "model", "variant"])
h1.to_csv("hip_contexto_h1-20261122.csv", index=False)
h2v = table(w, ["grp", "model", "tom_vader"], conds=(("k0", "kK"),))
h2v["medida"] = "vader"
h2p = table(w[w.grp == "MT-CSD"], ["grp", "model", "pai"], conds=(("k0", "kK"),))
h2p["medida"] = "pai"
h2 = pd.concat([h2v.rename(columns={"tom_vader": "tom"}), h2p.rename(columns={"pai": "tom"})])
h2.to_csv("hip_contexto_h2-20261122.csv", index=False)


def f(x, lo, hi):
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{x:+6.1f} [{lo:+5.1f};{hi:+5.1f}]{star}"


with open("hipoteses_contexto-20261122.txt", "w", encoding="utf-8") as out:
    p = lambda *a: print(*a, file=out)
    p("Hypotheses about context, all items, all option orders (twitter-roberta and DeBERTa-NLI")
    p("have one variant each). Paired changes in the share of answers, in points, from k0 (no")
    p("context) to k1 (previous turn) and kK (whole conversation). [95% CI]; * = CI excludes 0.")
    p("MT-CSD: thread bootstrap (13 threads); MUStARD++: item bootstrap. 2,000 resamples, seed 5.")
    p("N items: " + ", ".join(f"{g} {n}" for g, n in w.groupby("grp").id.nunique().items()))
    p("\n=== H1: more context -> more 'neutral'? ===")
    for g in h1.grp.unique():
        p(f"\n-- {g}")
        p(f"{'model':16s} {'var':4s} {'neu k0':>7s}  {'k0->k1':>24s}  {'k0->kK':>24s}")
        for _, r in h1[h1.grp == g].iterrows():
            p(f"{r.model:16s} {r.variant:4s} {r.neutral_k0:7.1f}  "
              f"{f(r.neutral_k0k1, r.neutral_k0k1_lo, r.neutral_k0k1_hi)}  "
              f"{f(r.neutral_k0kK, r.neutral_k0kK_lo, r.neutral_k0kK_hi)}")
    p("\nCount of model x order cells (k0 -> kK) with neutral share significantly up / down:")
    for g in h1.grp.unique():
        x = h1[h1.grp == g]
        up = ((x.neutral_k0kK_lo > 0)).sum(); dn = ((x.neutral_k0kK_hi < 0)).sum()
        p(f"  {g}: up {up}, down {dn}, n.s. {len(x) - up - dn} (of {len(x)})")

    p("\n=== H2: more context -> more 'negative'? (k0 -> kK) ===")
    for g in h1.grp.unique():
        x = h1[h1.grp == g]
        up = (x.negative_k0kK_lo > 0).sum(); dn = (x.negative_k0kK_hi < 0).sum()
        p(f"  {g}: negative share up {up}, down {dn}, n.s. {len(x) - up - dn} (of {len(x)})")
    for g in h1.grp.unique():
        p(f"\n-- {g}, all orders")
        p(f"{'model':16s} {'var':4s} {'neg k0':>7s}  {'neg k0->kK':>24s}  {'pos k0->kK':>24s}")
        for _, r in h1[h1.grp == g].iterrows():
            p(f"{r.model:16s} {r.variant:4s} {r.negative_k0:7.1f}  "
              f"{f(r.negative_k0kK, r.negative_k0kK_lo, r.negative_k0kK_hi)}  "
              f"{f(r.positive_k0kK, r.positive_k0kK_lo, r.positive_k0kK_hi)}")

    p("\n--- H2 by tone of the conversation (all option orders pooled) ---")
    p("Tone counts (items): VADER of the context text; MT-CSD also gold stance of the parent.")
    t = w.drop_duplicates("id")
    p(t.groupby(["grp", "tom_vader"]).id.count().to_string())
    p(t[t.grp == "MT-CSD"].groupby("pai").id.count().to_string())
    for med, lab in (("pai", "MT-CSD, gold stance of the comment above"), ("vader", "VADER tone of the context")):
        p(f"\n## {lab}")
        x = h2[h2.medida == med]
        for g in x.grp.unique():
            p(f"\n-- {g}")
            p(f"{'model':16s} {'tom':12s} {'n':>5s}  {'neg k0->kK':>24s}  {'neu k0->kK':>24s}  {'pos k0->kK':>24s}")
            for _, r in x[x.grp == g].iterrows():
                p(f"{r.model:16s} {str(r.tom):12s} {r.n:5d}  "
                  f"{f(r.negative_k0kK, r.negative_k0kK_lo, r.negative_k0kK_hi)}  "
                  f"{f(r.neutral_k0kK, r.neutral_k0kK_lo, r.neutral_k0kK_hi)}  "
                  f"{f(r.positive_k0kK, r.positive_k0kK_lo, r.positive_k0kK_hi)}")
print("done")
