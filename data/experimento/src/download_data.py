"""Stage 0: download the raw dataset files, pinned by commit, without modifying them.

Usage: python download_data.py <raw_dir>
"""
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import DATA


def main(raw_dir):
    raw = Path(raw_dir)
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    for name, src in DATA.items():
        for f in src["files"]:
            dest = raw / name / f
            url = f"https://raw.githubusercontent.com/{src['repo']}/{src['commit']}/{f}"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.rename(dest)
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            manifest["files"].append({"dataset": name, "url": url, "path": str(dest.relative_to(raw)),
                                      "bytes": dest.stat().st_size, "sha256": sha})
            print(f"{name}: {f} {dest.stat().st_size} bytes")
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
