"""Download and verify the public UCI archives listed in data/datasets.json.

Raw data are never committed: the D1/D1b licence restricts use to research,
so the repository ships only URLs and SHA-256 digests.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "datasets.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def fetch(key: str, data_dir: Path | None = None, source: Path | None = None) -> Path:
    """Return the extracted directory for dataset `key`, downloading if needed.

    `source` may point to a local copy of the zip (e.g. a shared folder);
    it is verified against the manifest digest before use.
    """
    entry = manifest()[key]
    data_dir = Path(data_dir or ROOT / "data")
    raw = data_dir / "raw" / entry["file"]
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        if source is not None:
            shutil.copyfile(source, raw)
        else:
            urllib.request.urlretrieve(entry["url"], raw)
    digest = sha256(raw)
    if digest != entry["sha256"]:
        raise ValueError(f"{key}: SHA-256 mismatch {digest} != {entry['sha256']}")
    out = data_dir / "extracted" / key
    if not (out / ".complete").exists():
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw) as z:
            for member in z.namelist():
                target = (out / member).resolve()
                if not str(target).startswith(str(out.resolve())):
                    raise ValueError(f"unsafe path in archive: {member}")
            z.extractall(out)
        (out / ".complete").write_text(digest, encoding="utf-8")
    return out
