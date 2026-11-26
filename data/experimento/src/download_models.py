"""Download the pinned model snapshots into the Hugging Face cache ($HF_HOME).

Usage: python download_models.py <nanojev_ckpt_dir> <manifest.json>
Kev (kev.serve --run ...@v1.0) and Laya (Router) fetch their own weights through the
same cache; their resolved revisions are recorded by the inference stages.
"""
import json
import sys

from huggingface_hub import snapshot_download

from config import MODELS


def main(nanojev_dir, manifest_path):
    manifest = {}
    for name, cfg in MODELS.items():
        try:
            fetch(name, cfg, nanojev_dir, manifest)
        except Exception as e:  # noqa: BLE001 - a failed download only skips that model
            manifest[name] = {"error": repr(e)}
            print(name, "FAILED", repr(e), flush=True)
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    if any("error" in v for v in manifest.values()):
        sys.exit(1)


def fetch(name, cfg, nanojev_dir, manifest):
    if True:
        if cfg["kind"] in ("classifier", "nli", "llm"):
            path = snapshot_download(cfg["hf"], revision=cfg["revision"],
                                     allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                                     "*.jinja", "*.py", "tokenizer*", "spm*"]
                                     + cfg.get("weights", []),
                                     ignore_patterns=cfg.get("ignore_patterns"))
            manifest[name] = {"hf": cfg["hf"], "revision": cfg["revision"], "path": path}
        elif cfg["kind"] == "nanojev":
            path = snapshot_download(cfg["hf"], revision=cfg["revision"], local_dir=nanojev_dir,
                                     allow_patterns=["best.safetensors", "config.json", "tokenizer/*",
                                                     "backbone_config/*"])
            manifest[name] = {"hf": cfg["hf"], "revision": cfg["revision"], "path": path}
        elif cfg["kind"] == "laya":
            path = snapshot_download(cfg["state_tokenizer"], allow_patterns=["tokenizer*", "*.json"])
            manifest[name + "-state-tokenizer"] = {"hf": cfg["state_tokenizer"], "path": path}
        print(name, "ok", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
