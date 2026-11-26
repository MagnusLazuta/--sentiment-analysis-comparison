"""Stage 4: one PDF per model with the metrics and every classified instance.

Usage: python report.py <items.jsonl> <predictions_dir> <eval_dir> <out_dir>

Each PDF (report-<model>.pdf) has, per dataset:
  1. the metrics from eval/metrics.csv (all variants, conditions and groups);
  2. one row per instance and condition: id, text (shortened), gold label,
     sarcasm flag, and for each variant the predicted label and the full
     probability distribution (negative / neutral / positive).
"""
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle

from config import LABELS, SEED

ST = getSampleStyleSheet()
SMALL = ST["BodyText"].clone("small", fontSize=6.5, leading=7.5)
ABBR = {"negative": "neg", "neutral": "neu", "positive": "pos"}
METRIC_COLS = ["variant", "condition", "group", "n", "accuracy", "acc_ci", "macro_f1", "f1_ci",
               "ece", "brier", "mean_confidence", "confident_errors_0.9", "literal_fall", "gold_labels"]


def p(text, style=SMALL):
    return Paragraph(escape(str(text)), style)


def fmt(x):
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def table(rows, widths, header_bg=colors.lightgrey):
    t = LongTable(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return t


def metrics_table(rows):
    head = [p(c) for c in METRIC_COLS]
    body = []
    for r in rows:
        r = dict(r, acc_ci=f"[{fmt(r['accuracy_lo'])}, {fmt(r['accuracy_hi'])}]",
                 f1_ci=f"[{fmt(r['macro_f1_lo'])}, {fmt(r['macro_f1_hi'])}]")
        body.append([p(fmt(r[c]) if c not in ("variant", "condition", "group", "n", "gold_labels",
                                               "acc_ci", "f1_ci") else r[c]) for c in METRIC_COLS])
    w = [12, 20, 20, 10, 15, 25, 15, 25, 13, 13, 17, 17, 15, 45]
    return table([head] + body, [x * mm for x in w])


def instance_table(item_list, preds, variants):
    head = [p(h) for h in ["id", "text", "gold", "sarc", "condition"]] + \
           [p(f"{v}: pred (neg/neu/pos)") for v in variants]
    body = []
    for it in item_list:
        for cond in ("no_context", "with_context"):
            got = {v: preds.get((it["id"], cond, v)) for v in variants}
            if not any(got.values()):
                continue
            text = it["text"].replace("\n", " ")
            text = text if len(text) <= 220 else text[:217] + "..."
            row = [p(it["id"]), p(text), p(ABBR[it["gold"]]), p("yes" if it["sarcastic"] else "no"), p(cond)]
            for v in variants:
                r = got[v]
                if r is None:
                    row.append(p("-"))
                    continue
                mark = "" if r["pred"] == it["gold"] else " X"
                probs = "/".join(f"{r['probs'][l]:.2f}" for l in LABELS)
                row.append(p(f"{ABBR[r['pred']]}{mark} ({probs})"))
            body.append(row)
    vw = (277 - 30 - 95 - 9 - 8 - 17) / len(variants)
    return table([head] + body, [x * mm for x in [30, 95, 9, 8, 17] + [vw] * len(variants)])


def main(items_path, pred_dir, eval_dir, out_dir):
    items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
    metrics = list(csv.DictReader(open(Path(eval_dir) / "metrics.csv")))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(pred_dir).glob("*.jsonl")):
        model = f.stem
        preds = {}
        for r in map(json.loads, open(f, encoding="utf-8")):
            preds[(r["id"], r["condition"], r["variant"])] = r
        variants = sorted({k[2] for k in preds})
        by_ds = defaultdict(list)
        for it in items:
            if any((it["id"], c, v) in preds for c in ("no_context", "with_context") for v in variants):
                by_ds[it["dataset"]].append(it)
        doc = SimpleDocTemplate(str(out / f"report-{model}.pdf"), pagesize=landscape(A4),
                                leftMargin=10 * mm, rightMargin=10 * mm, topMargin=10 * mm,
                                bottomMargin=10 * mm, title=f"{model} predictions")
        story = [Paragraph(f"Model: {escape(model)}", ST["Title"]),
                 p(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Predictions: {len(preds)}. "
                   "Instance columns: predicted label (X = wrong) and probabilities negative/neutral/positive. "
                   f"Metrics: accuracy and macro-F1 with 95% bootstrap CI (1000 resamples, seed {SEED}); "
                   "ECE (15 bins); Brier; confident errors = share of wrong predictions with p >= 0.9; "
                   "literal_fall = share of sarcastic items (literal != intended) predicted with the literal label.",
                   ST["BodyText"])]
        for ds in sorted(by_ds):
            story += [PageBreak(), Paragraph(f"{escape(model)} / {escape(ds)}", ST["Heading1"]),
                      Paragraph("Metrics", ST["Heading2"]),
                      metrics_table([m for m in metrics if m["model"] == model and m["dataset"] == ds]),
                      Spacer(1, 4 * mm), Paragraph(f"Instances ({len(by_ds[ds])})", ST["Heading2"]),
                      instance_table(by_ds[ds], preds, variants)]
        doc.build(story)
        print("wrote", out / f"report-{model}.pdf", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
