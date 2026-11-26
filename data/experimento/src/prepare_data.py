"""Stage 1: convert the raw files into one items.jsonl with a common schema.

Usage: python prepare_data.py <raw_dir> <out_dir>

Each item:
  id, dataset, text, context (list of {speaker, text}), target (str or None),
  unit (utterance/comment/tweet/review), gold (negative/neutral/positive),
  literal (sentiment of the words, sarcastic MUStARD++ only, else None),
  sarcastic (bool), meta (dict).
Raw files are only read; extracted archives go to <out_dir>/extracted.
"""
import collections
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

from config import SARCASM_NLP_PER_GROUP, SEED

# MUStARD++ emotions -> sentiment. Surprise is ambiguous and dropped (gold) or None (literal).
EMOTION = {
    "Happiness": "positive", "Excitement": "positive",
    "Neutral": "neutral",
    "Anger": "negative", "Disgust": "negative", "Fear": "negative",
    "Frustration": "negative", "Sadness": "negative", "Ridicule": "negative",
}
STANCE = {"favor": "positive", "against": "negative", "none": "neutral"}
STARS = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
SARC_TAGS = re.compile(r"#(sarcastic|sarcasm|ironic|irony)|#not\b", re.I)
SARC_NLP = {"-1": "negative", "0": "neutral", "1": "positive"}


def mustardpp(raw):
    rows = list(csv.DictReader(open(raw / "mustardpp/mustard++_text.csv", encoding="utf-8")))
    ctx = collections.defaultdict(list)
    for r in rows:
        if "_c_" in r["KEY"]:
            ctx[r["SCENE"]].append(r)
    items, dropped = [], collections.Counter()
    for r in rows:
        if not r["KEY"].endswith("_u"):
            continue
        gold = EMOTION.get(r["Implicit_Emotion"])
        if gold is None:
            dropped[r["Implicit_Emotion"]] += 1
            continue
        sarcastic = r["Sarcasm"] == "1"
        # numeric order of the "_c_<n>" suffix (string order put c_10 before c_2)
        turns = sorted(ctx[r["SCENE"]], key=lambda c: int(c["KEY"].rsplit("_c_", 1)[1]))
        items.append({
            "id": f"mpp:{r['KEY']}", "dataset": "mustardpp",
            "text": r["SENTENCE"].strip(), "speaker": r["SPEAKER"],
            "context": [{"speaker": c["SPEAKER"], "text": c["SENTENCE"].strip()} for c in turns],
            "target": None, "unit": "utterance", "gold": gold,
            "literal": EMOTION.get(r["Explicit_Emotion"]) if sarcastic else None,
            "sarcastic": sarcastic, "subgroup": r["SHOW"],
            "meta": {"show": r["SHOW"], "implicit": r["Implicit_Emotion"],
                     "explicit": r["Explicit_Emotion"], "sarcasm_type": r["Sarcasm_Type"]},
        })
    return items, {"dropped_gold": dict(dropped)}


def mtcsd(raw):
    items, info = [], {}
    for target in ["Tesla", "SpaceX"]:
        text = {}
        with open(raw / f"mtcsd/data/{target}/text.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                text[r["id"]] = r["text"].strip()
        index = json.load(open(raw / f"mtcsd/data/{target}/test.json", encoding="utf-8"))
        no_ctx = 0
        for k, row in enumerate(index):
            chain = row["index"]
            if len(chain) < 2:  # the post itself: no context, identical in both conditions
                no_ctx += 1
                continue
            context = [{"speaker": "Post" if i == 0 else f"Reply {i}", "text": text[c]}
                       for i, c in enumerate(chain[:-1])]
            items.append({
                "id": f"mtcsd:{target}:{k}", "dataset": "mtcsd",
                "text": text[chain[-1]], "speaker": f"Reply {len(chain) - 1}",
                "context": context, "target": target, "unit": "comment",
                "gold": STANCE[row["stance"]], "literal": None, "sarcastic": False, "subgroup": target,
                "meta": {"target": target, "depth": len(chain), "stance": row["stance"]},
            })
        info[f"{target}_dropped_without_context"] = no_ctx
    return items, info


def sarcasm_nlp(raw):
    path = raw / "sarcasm_nlp/Human_Sentiment_Data_Analysis/Human_Sentiment_Agree.csv"
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["Sentiment"] in SARC_NLP and r["Sarcasm_Col"] in ("Sarcastic", "Non-Sarcastic")]
    rng = random.Random(SEED)
    items, tags_removed = [], collections.Counter()
    for group in ["Sarcastic", "Non-Sarcastic"]:
        pool = [r for r in rows if r["Sarcasm_Col"] == group]
        chosen = pool if SARCASM_NLP_PER_GROUP is None else rng.sample(pool, SARCASM_NLP_PER_GROUP)
        for r in chosen:
            t = r["Tweets"].strip()
            # sarcastic tweets are mostly wrapped in quotes (generation artefact): strip outer quotes
            if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
                t = t[1:-1].strip()
            # hashtags that name the label would leak it (counts in items_report.json)
            t2 = " ".join(SARC_TAGS.sub(" ", t).split())
            tags_removed[group] += t2 != t
            items.append({
                "id": f"snlp:{r['']}", "dataset": "sarcasm_nlp", "text": t2, "speaker": None,
                "context": [], "target": "the topic", "unit": "tweet",
                "gold": SARC_NLP[r["Sentiment"]], "literal": None,
                "sarcastic": group == "Sarcastic", "subgroup": r["Topic"] or "unknown",
                "meta": {"topic": r["Topic"]},
            })
    return items, {"pool": len(rows), "label_hashtags_removed": dict(tags_removed)}


def filatova(raw, out):
    ext = out / "extracted" / "filatova"
    items, bad = [], []
    for name, sarcastic in [("Ironic", True), ("Regular", False)]:
        d = ext / name
        if not d.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            rar = str(raw / f"filatova/{name}.rar")
            if shutil.which("bsdtar"):
                cmd = ["bsdtar", "-xf", rar, "-C", str(d.parent)]
            else:  # nodes without libarchive: static 7-Zip fetched by pipeline.sh (SEVENZIP)
                cmd = [os.environ["SEVENZIP"], "x", "-y", "-bd", f"-o{d.parent}", rar]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        for f in sorted(d.glob("*.txt")):
            raw_text = f.read_bytes()
            try:
                t = raw_text.decode("utf-8")
            except UnicodeDecodeError:
                t = raw_text.decode("latin-1")
            fields = {k: (re.search(rf"<{k}>(.*?)</{k}>", t, re.S) or [None, None])[1]
                      for k in ["STARS", "TITLE", "REVIEW", "PRODUCT"]}
            try:
                stars = int(float(fields["STARS"]))
            except (TypeError, ValueError):
                bad.append(f.name)
                continue
            title, review = (fields["TITLE"] or "").strip(), (fields["REVIEW"] or "").strip()
            items.append({
                "id": f"fil:{f.stem}", "dataset": "filatova",
                "text": f"{title}\n\n{review}" if title else review, "speaker": None,
                "context": [], "target": "the product", "unit": "review",
                "gold": STARS[stars], "literal": None, "sarcastic": sarcastic, "subgroup": f"{stars} stars",
                "meta": {"stars": stars, "product": (fields["PRODUCT"] or "").strip()},
            })
    return items, {"unparsed": bad}


def main(raw_dir, out_dir):
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_items, report = [], {}
    for name, fn in [("mustardpp", lambda: mustardpp(raw)), ("mtcsd", lambda: mtcsd(raw)),
                     ("sarcasm_nlp", lambda: sarcasm_nlp(raw)), ("filatova", lambda: filatova(raw, out))]:
        items, info = fn()
        counts = collections.Counter((i["sarcastic"], i["gold"]) for i in items)
        report[name] = {"n": len(items), **info,
                        "by_sarcasm_gold": {f"{'sarc' if s else 'nonsarc'}/{g}": c
                                            for (s, g), c in sorted(counts.items())}}
        all_items += items
        print(name, json.dumps(report[name]))
    ids = [i["id"] for i in all_items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for i in all_items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    (out / "items_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
