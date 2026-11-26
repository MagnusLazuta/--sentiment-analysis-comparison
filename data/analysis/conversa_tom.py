"""Majority sentiment of the conversation before each item (2026-11-22).

MT-CSD: gold stance of every annotated context turn (replies are items of the dataset; the
thread head is not annotated and is skipped); majority among them -> negativo / neutro /
positivo, ties -> "misto"; no annotated turn -> "sem rótulo".
MUStARD++: context turns are not annotated; each turn gets a VADER polarity (compound
< -0.05 negativo, > 0.05 positivo, else neutro); majority among turns, ties -> "misto".
VADER is a lexical proxy for tone, not stance. Most MUStARD++ turns are short and VADER-neutral,
so the per-turn majority is almost always "neutro"; "tom_total" (VADER of the whole context
text) is used for MUStARD++ in the figures.

Run: uv run --with vaderSentiment python conversa_tom.py
Output: conversa_tom-20261122.csv (id, dataset, n_turnos, n_rotulados, neg, neu, pos, maioria, medida)
"""
import collections
import csv
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SRC = "../results/20261117-181106-838446/data/items.jsonl"
PT = {"negative": "negativo", "neutral": "neutro", "positive": "positivo"}
items = [json.loads(l) for l in open(SRC, encoding="utf-8")]
items = [it for it in items if it["dataset"] in ("mtcsd", "mustardpp")]

# MT-CSD: (target, chain texts) -> gold, to label each context turn
gold = {}
for it in items:
    if it["dataset"] == "mtcsd":
        gold[(it["target"],) + tuple(t["text"] for t in it["context"]) + (it["text"],)] = it["gold"]

sia = SentimentIntensityAnalyzer()


def vader(t):
    c = sia.polarity_scores(t)["compound"]
    return "negativo" if c < -0.05 else ("positivo" if c > 0.05 else "neutro")


rows = []
for it in items:
    ctx = it["context"] or []
    labs = []
    if it["dataset"] == "mtcsd":
        texts = [t["text"] for t in ctx]
        for j in range(1, len(texts)):  # turn j (j >= 1) is a reply whose own context is texts[:j]
            g = gold.get((it["target"],) + tuple(texts[:j + 1]))
            if g:
                labs.append(PT[g])
        medida = "postura anotada"
    else:
        labs = [vader(t["text"]) for t in ctx]
        medida = "VADER por turno"
    c = collections.Counter(labs)
    if not labs:
        maj = "sem rótulo"
    else:
        top = c.most_common()
        maj = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "misto"
    tot = vader(" ".join(t["text"] for t in ctx)) if ctx else "sem contexto"
    rows.append({"id": it["id"], "dataset": it["dataset"], "n_turnos": len(ctx), "n_rotulados": len(labs),
                 "neg": c["negativo"], "neu": c["neutro"], "pos": c["positivo"], "maioria": maj, "medida": medida,
                 "tom_total": tot})

with open("conversa_tom-20261122.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(collections.Counter((r["dataset"], r["maioria"]) for r in rows))
