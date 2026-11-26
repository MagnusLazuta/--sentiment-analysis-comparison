"""Inputs shown to the models.

Decisions (2026-11-15, after review):
  - One fixed wording per dataset, chosen to measure what the gold label measures
    (intended sentiment, or stance for MT-CSD). Variants v1..v3 change ONLY the
    option order, so they measure the same thing and can be compared.
  - MT-CSD is asked as stance (favor / against / none) and mapped back to
    positive / negative / neutral; the other datasets are asked as sentiment.
  - NLI scores each label separately, so the option order does not exist there:
    it runs a single variant (v1).
Conditions: no_context (all items) and with_context (MUStARD++, MT-CSD). MUStARD++
shows the speaker name in both conditions so only the context changes.
"""
from config import LABELS

VARIANTS = ["v1", "v2", "v3"]

# Option order per variant, over the canonical labels.
ORDER = {
    "v1": ["negative", "neutral", "positive"],
    "v2": ["positive", "neutral", "negative"],
    "v3": ["neutral", "positive", "negative"],
}

# Words shown to the model for each canonical label.
SENTIMENT_WORDS = {"negative": "negative", "neutral": "neutral", "positive": "positive"}
STANCE_WORDS = {"negative": "against", "neutral": "none", "positive": "favor"}


def is_stance(item):
    return item["dataset"] == "mtcsd"


def label_words(item):
    return STANCE_WORDS if is_stance(item) else SENTIMENT_WORDS


def descriptions(item):
    """Option descriptions (System One Choice criteria), keyed by canonical label."""
    if is_stance(item):
        t = item["target"]
        return {"positive": f"supports {t} or speaks well of it",
                "negative": f"opposes or criticises {t}",
                "neutral": f"takes no position on {t}, or is unrelated to it"}
    return {"negative": "unfavourable, critical, angry or unhappy",
            "neutral": "neither positive nor negative, or purely factual",
            "positive": "favourable, approving, happy or pleased"}


def conditions(item):
    if item.get("ablation_k"):  # context-length ablation copy (make_ablation.py)
        return ["with_context"]
    if item.get("extra_condition"):  # follow-up experiment copy (make_extra.py)
        return [item["extra_condition"]]
    return ["no_context", "with_context"] if item["context"] else ["no_context"]


def question(item):
    if is_stance(item):
        return f"What is the stance of the author of the comment toward {item['target']}?"
    if item["dataset"] == "mustardpp":
        return "How does the speaker of the utterance feel?"
    q = f"How does the author of the {item['unit']} feel about {item['target']}?"
    if item.get("question_note"):  # follow-up D (make_issue.py): extra annotation rule after the question
        q += " " + item["question_note"]
    return q


def hypothesis(item, label):
    """NLI hypothesis for one canonical label."""
    if is_stance(item):
        t = item["target"]
        return {"positive": f"The author of this comment is in favor of {t}.",
                "negative": f"The author of this comment is against {t}.",
                "neutral": f"The author of this comment takes no position on {t}."}[label]
    who = "speaker" if item["dataset"] == "mustardpp" else "author"
    about = f" about {item['target']}" if item["target"] else ""
    feel = "neither positive nor negative" if label == "neutral" else label
    return f"The {who} feels {feel}{about}."


def render_text(item):
    return f"{item['speaker']}: {item['text']}" if item["dataset"] == "mustardpp" else item["text"]


def render_context(turns):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


def render_state(item, condition, turns=None):
    """Plain-text input. `turns` overrides item['context'] (used when truncating)."""
    turns = (item["context"] if turns is None else turns) if condition == "with_context" else []
    if not turns:
        return render_text(item)
    head = "Conversation so far" if item["dataset"] == "mustardpp" else "Thread so far"
    return f"{head}:\n{render_context(turns)}\n\n{item['unit'].capitalize()} to classify:\n{render_text(item)}"


def llm_messages(state, item, variant):
    words = label_words(item)
    opts = ", ".join(words[l] for l in ORDER[variant])
    ask = question(item)
    if item.get("swap_criteria"):  # follow-up B (make_extra.py): show the System One descriptions
        desc = descriptions(item)
        ask += "\n" + "\n".join(f"- {words[l]}: {desc[l]}" for l in ORDER[variant])
    user = f"{state}\n\n{ask}\nAnswer with exactly one word: {opts}."
    return [{"role": "user", "content": user}]


def choice_question(item, variant):
    words, desc = label_words(item), descriptions(item)
    if item.get("swap_criteria"):  # follow-up B (make_extra.py): label words only, no descriptions
        desc = {l: words[l] for l in words}
    return {"type": "choice", "instructions": question(item),
            "criteria": {words[l]: desc[l] for l in ORDER[variant]}}


def to_canonical(item, probs_by_word):
    """Map a {shown word: p} dict back to {canonical label: p}, renormalised."""
    words = label_words(item)
    p = {l: float(probs_by_word[words[l]]) for l in LABELS}
    s = sum(p.values())
    return {l: v / s for l, v in p.items()}


assert all(sorted(o) == sorted(LABELS) for o in ORDER.values())
