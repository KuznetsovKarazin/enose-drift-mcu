"""Extract the 64 streaming features (8 per channel) for all D2 traces.

Usage: python scripts/extract_d2.py [--jobs 4]  ->  data/cache/d2_features.npz
"""
import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from enose_drift.data import list_d2  # noqa: E402
from enose_drift.features import extract  # noqa: E402


def one(path):
    tr = np.loadtxt(path)
    t = tr[:, 0]
    return extract(tr), len(t), float(np.diff(t).max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    recs = list_d2(ROOT / "data" / "extracted" / "D2")
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, [r.path for r in recs], chunksize=4))
    out = ROOT / "data" / "cache" / "d2_features.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.stack([r[0] for r in res]),
             board=np.array([r.board for r in recs]), gas=np.array([r.gas for r in recs]),
             level=np.array([r.level for r in recs]), repetition=np.array([r.repetition for r in recs]),
             concentration=np.array([r.concentration for r in recs]),
             n_samples=np.array([r[1] for r in res]), max_gap_s=np.array([r[2] for r in res]),
             filename=np.array([r.path.name for r in recs]))
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())
