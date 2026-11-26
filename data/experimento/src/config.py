"""Pinned sources, model revisions and study constants (single source of truth)."""

SEED = 5
LABELS = ["negative", "neutral", "positive"]

# Raw data: repository + commit; files are downloaded unchanged to $SCRATCH/data/raw.
# Commits resolved on 2026-11-15.
DATA = {
    "mustardpp": {
        "repo": "apoorva-nunna/MUStARD_Plus_Plus",
        "commit": "723267b3543ee51a3c0d628d6e45172d8261d875",
        "files": ["mustard++_text.csv"],
    },
    "mtcsd": {
        "repo": "nfq729/MT-CSD",
        "commit": "0707ec6372e747f59ada31f2301614413438e120",
        "files": ["data/Tesla/test.json", "data/Tesla/text.csv",
                  "data/SpaceX/test.json", "data/SpaceX/text.csv"],
    },
    "sarcasm_nlp": {
        "repo": "aims-umich/Sarcasm_NLP",
        "commit": "ff05349549a86cf87acc15a7beca8d0863bf9669",
        "files": ["Human_Sentiment_Data_Analysis/Human_Sentiment_Agree.csv"],
    },
    "filatova": {
        "repo": "ef2020/SarcasmAmazonReviewsCorpus",
        "commit": "7d59500a6344c5fc1eee55c6a855ec83991931cb",
        "files": ["Ironic.rar", "Regular.rar", "five_labels_plus_stars.xlsx"],
    },
}

# Sarcasm_NLP sample size per group (sarcastic / non-sarcastic); None = whole file
# (decided 2026-11-17: use all 5,929 agreed tweets).
SARCASM_NLP_PER_GROUP = None

# Models: Hugging Face revision resolved on 2026-11-15.
MODELS = {
    "twitter-roberta": {
        "kind": "classifier",
        "hf": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "revision": "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
        "max_tokens": 512,
        "weights": ["pytorch_model.bin"],  # this revision has no safetensors
    },
    "deberta-nli": {
        "kind": "nli",
        "hf": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "revision": "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
        "max_tokens": 512,
    },
    "qwen3.5-4b": {
        "kind": "llm",
        "hf": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "chat_kwargs": {"enable_thinking": False},
        "max_batch_tokens": 16000,
    },
    "mistral-7b": {
        "kind": "llm",
        "hf": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "chat_kwargs": {},
        "max_batch_tokens": 16000,
        "ignore_patterns": ["consolidated*"],  # duplicate of the sharded weights
    },
    "kev-4b": {
        "kind": "kev",
        "hf": "jaredpalmer/kev-4b",
        "run_spec": "jaredpalmer/kev-4b@v1.0",
        "code_repo": "jaredpalmer/kev",
        "code_commit": "2074cc44256200bb37540103fa4572901faa10f3",
    },
    "laya": {
        "kind": "laya",
        "pip": "laya==0.4.34",
        "hf": "convaiinnovations/laya",
        "revision": "e333cb7c823cd1787276228260f56d02fbf5180b",
        # English checkpoint: 512 tokens, 192 reserved for the question head.
        "state_tokenizer": "answerdotai/ModernBERT-large",
        "state_max_tokens": 320,
    },
    "nanojev": {
        "kind": "nanojev",
        "hf": "C-Tianyu/NanoJev",
        "revision": "unified-games-v1",
        "code_repo": "TianyuCodings/NanoJev",
        "code_commit": "45b629283ddeb3b3784806f352bbf5e97f71dd40",
        "max_length": 8192,  # checkpoint config value; inputs are never truncated by NanoJev
        "state_max_tokens": 6000,
    },
}
