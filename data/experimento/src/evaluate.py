"""Stage 3: metrics from the prediction files (CPU only, no plots).

Usage: python evaluate.py <items.jsonl> <predictions_dir> <out_dir>

Writes:
  metrics.csv         one row per model x variant x condition x dataset x group
  context_effect.csv  paired with_context - no_context comparison (McNemar exact test)
Groups: all, sarcastic, non_sarcastic, subgroup=<x> and <sarcasm>|subgroup=<x>
(subgroup = Sarcasm_NLP topic, MT-CSD target, MUStARD++ show, Filatova stars). Task mapping (see README):
  T1 = mustardpp/non_sarcastic + mtcsd, no_context
  T2 = the same, with_context
  T3 = mustardpp, sarcasm_nlp, filatova (sarcastic vs non_sarcastic)
Bootstrap: 1000 resamples, seed config.SEED, 95% percentile intervals.
"""
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from config import LABELS, SEED

N_BINS, N_BOOT = 15, 1000


def macro_f1(golds, preds):
    f1s = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(golds, preds))
        fp = sum(g != lab and p == lab for g, p in zip(golds, preds))
        fn = sum(g == lab and p != lab for g, p in zip(golds, preds))
        if tp + fp + fn == 0:  # label absent from gold and predictions
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def ece(conf, correct):
    bins = defaultdict(list)
    for c, k in zip(conf, correct):
        bins[min(int(c * N_BINS), N_BINS - 1)].append((c, k))
    n = len(conf)
    return sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(k for _, k in b) / len(b))
               for b in bins.values())


def brier(rows):
    return sum(sum((r["probs"][l] - (l == r["gold"])) ** 2 for l in LABELS) for r in rows) / len(rows)


def ci(rows, fn, rng):
    vals = sorted(fn([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(N_BOOT))
    return vals[int(0.025 * N_BOOT)], vals[int(0.975 * N_BOOT) - 1]


def summarize(rows, rng):
    acc = lambda rs: sum(r["pred"] == r["gold"] for r in rs) / len(rs)
    f1 = lambda rs: macro_f1([r["gold"] for r in rs], [r["pred"] for r in rs])
    conf = [r["probs"][r["pred"]] for r in rows]
    correct = [r["pred"] == r["gold"] for r in rows]
    lit = [r for r in rows if r.get("literal") and r["literal"] != r["gold"]]
    out = {
        "n": len(rows),
        "gold_labels": "/".join(f"{l[:3]}={sum(r['gold'] == l for r in rows)}" for l in LABELS),
        "accuracy": acc(rows), "macro_f1": f1(rows),
        "ece": ece(conf, correct), "brier": brier(rows), "mean_confidence": sum(conf) / len(conf),
        "confident_errors_0.9": sum(c >= 0.9 and not k for c, k in zip(conf, correct)) / len(rows),
        "n_literal_differs": len(lit),
        "literal_fall": sum(r["pred"] == r["literal"] for r in lit) / len(lit) if lit else "",
    }
    out["accuracy_lo"], out["accuracy_hi"] = ci(rows, acc, rng)
    out["macro_f1_lo"], out["macro_f1_hi"] = ci(rows, f1, rng)
    return out


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(items_path, pred_dir, out_dir):
    items = {i["id"]: i for i in map(json.loads, open(items_path, encoding="utf-8"))}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    groups, by_pair = defaultdict(list), defaultdict(dict)
    for f in sorted(Path(pred_dir).glob("*.jsonl")):
        model = f.stem
        for r in map(json.loads, open(f, encoding="utf-8")):
            it = items[r["id"]]
            r.update(gold=it["gold"], literal=it["literal"])
            sub = "sarcastic" if it["sarcastic"] else "non_sarcastic"
            for g in ("all", sub, f"subgroup={it.get('subgroup', '')}", f"{sub}|subgroup={it.get('subgroup', '')}"):
                groups[(model, r["variant"], r["condition"], it["dataset"], g)].append(r)
            by_pair[(model, r["variant"], it["dataset"], sub)].setdefault(r["id"], {})[r["condition"]] = r

    with open(out / "metrics.csv", "w", newline="") as fh:
        w = None
        for key in sorted(groups):
            row = dict(zip(["model", "variant", "condition", "dataset", "group"], key))
            row.update(summarize(groups[key], rng))
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)

    with open(out / "context_effect.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "variant", "dataset", "group", "n_pairs", "acc_no_context",
                    "acc_with_context", "delta", "fixed_by_context", "broken_by_context", "mcnemar_p"])
        for key in sorted(by_pair):
            pairs = [p for p in by_pair[key].values() if len(p) == 2]
            if not pairs:
                continue
            ok = lambda r: r["pred"] == r["gold"]
            a0 = sum(ok(p["no_context"]) for p in pairs) / len(pairs)
            a1 = sum(ok(p["with_context"]) for p in pairs) / len(pairs)
            fixed = sum(not ok(p["no_context"]) and ok(p["with_context"]) for p in pairs)
            broken = sum(ok(p["no_context"]) and not ok(p["with_context"]) for p in pairs)
            w.writerow([*key, len(pairs), f"{a0:.4f}", f"{a1:.4f}", f"{a1 - a0:+.4f}",
                        fixed, broken, f"{mcnemar(fixed, broken):.4g}"])
    export_predictions(items, pred_dir, out)
    print("wrote", out / "metrics.csv", out / "context_effect.csv", out / "predictions.csv")


EXTRA = ["label_mass", "top_token", "tokens", "turns_dropped", "over_budget", "routed_to"]


def export_predictions(items, pred_dir, out):
    """Flat table: one row per model x item x condition x variant, with the item fields."""
    cols = ["model", "dataset", "id", "subgroup", "sarcastic", "gold", "literal", "condition", "variant", "pred",
            "correct", "confidence"] + [f"p_{l}" for l in LABELS] + EXTRA + ["target", "text", "context"]
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for f in sorted(Path(pred_dir).glob("*.jsonl")):
            for r in map(json.loads, open(f, encoding="utf-8")):
                it = items[r["id"]]
                row = {"model": f.stem, "dataset": it["dataset"], "id": r["id"], "subgroup": it.get("subgroup", ""),
                       "sarcastic": int(it["sarcastic"]), "gold": it["gold"], "literal": it["literal"] or "",
                       "condition": r["condition"], "variant": r["variant"], "pred": r["pred"],
                       "correct": int(r["pred"] == it["gold"]), "confidence": r["probs"][r["pred"]],
                       "target": it["target"] or "", "text": it["text"],
                       "context": "\n".join(f"{t['speaker']}: {t['text']}" for t in it["context"])
                       if r["condition"] == "with_context" else ""}
                row.update({f"p_{l}": r["probs"][l] for l in LABELS})
                row.update({k: r.get(k, "") for k in EXTRA})
                w.writerow(row)


if __name__ == "__main__":
    main(*sys.argv[1:4])
