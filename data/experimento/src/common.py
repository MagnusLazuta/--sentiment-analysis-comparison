"""Shared helpers for the inference stages."""
import argparse
import json
import math
from pathlib import Path

from config import LABELS
from prompts import conditions, render_state


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True, help="predictions jsonl (appended, resumable)")
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=0, help="max items per dataset (pilot)")
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


def load_items(path, limit=0):
    items = [json.loads(line) for line in open(path, encoding="utf-8")]
    if limit:
        seen = {}
        keep = []
        for it in items:
            seen[it["dataset"]] = seen.get(it["dataset"], 0) + 1
            if seen[it["dataset"]] <= limit:
                keep.append(it)
        items = keep
    return items


def jobs(items, variants):
    """(item, condition, variant) triples to run."""
    return [(it, c, v) for it in items for c in conditions(it) for v in variants]


class Writer:
    """Appends one json line per prediction; skips keys already written (resume)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            good = []
            for line in open(self.path, encoding="utf-8"):
                try:  # a job killed mid-write can leave a truncated last line
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                good.append(line if line.endswith("\n") else line + "\n")
                self.done.add((r["id"], r["condition"], r["variant"]))
            self.path.write_text("".join(good), encoding="utf-8")
        self.f = open(self.path, "a", encoding="utf-8")

    def todo(self, triples):
        return [t for t in triples if (t[0]["id"], t[1], t[2]) not in self.done]

    def write(self, item, condition, variant, probs, **extra):
        assert set(probs) == set(LABELS), probs
        s = sum(probs.values())
        assert all(math.isfinite(p) and p >= 0 for p in probs.values()) and abs(s - 1) < 1e-3, probs
        rec = {"id": item["id"], "condition": condition, "variant": variant,
               "probs": probs, "pred": max(LABELS, key=probs.get), **extra}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def fit_state(item, condition, n_tokens, budget):
    """Render the input, dropping the oldest context turns until it fits `budget` tokens.

    Returns (state, info). If the text alone is still too long, the caller's tokenizer
    truncates it and info['over_budget'] is True.
    """
    turns = list(item["context"]) if condition == "with_context" else []
    dropped = 0
    state = render_state(item, condition, turns)
    while turns and n_tokens(state) > budget:
        turns.pop(0)
        dropped += 1
        state = render_state(item, condition, turns) if turns else render_state(item, "no_context")
    n = n_tokens(state)
    return state, {"tokens": n, "turns_dropped": dropped, "over_budget": n > budget}
