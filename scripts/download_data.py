"""Download and verify D1, D1b and D2.  Usage: python scripts/download_data.py [--from DIR]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enose_drift.fetch import fetch, manifest  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--from", dest="src", type=Path, help="directory holding already downloaded zips")
a = p.parse_args()
for key, entry in manifest().items():
    src = a.src / entry["file"] if a.src else None
    print(key, "->", fetch(key, source=src if src and src.exists() else None))
