"""E6 (revision, post hoc): the float32 C streaming extractor of the firmware run
over all 640 complete D2 recordings and compared with the features used in E4
(data/cache/d2_features.npz). Answers the fixture-length limitation of E5.

Usage: python experiments/e6/ext_full.py OUT_DIR
"""
from __future__ import annotations

import ctypes as C
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from enose_drift import f32 as F  # noqa: E402
from enose_drift.data import list_d2  # noqa: E402
from enose_drift.features import BASELINE_SECONDS  # noqa: E402


def one(path):
    lib = F._lib()
    tr = np.loadtxt(path)
    t = np.ascontiguousarray(tr[:, 0], np.float32)
    r = np.ascontiguousarray(tr[:, 1:], np.float32)
    f = np.zeros(64, np.float32)
    fp = C.POINTER(C.c_float)
    lib.f_ext_run(t.ctypes.data_as(fp), r.ctypes.data_as(fp), len(t), C.c_float(BASELINE_SECONDS), f.ctypes.data_as(fp))
    return path.name, f, len(t)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cache = np.load(ROOT / "data/cache/d2_features.npz", allow_pickle=True)
    ref = {n: x for n, x in zip(cache["filename"], cache["X"])}
    recs = list_d2(ROOT / "data/extracted/D2")
    F._lib()
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(one, [r.path for r in recs], chunksize=8))
    rel = np.array([np.max(np.abs(f - ref[n]) / np.maximum(1.0, np.abs(ref[n]))) for n, f, _ in res])
    exact = int(sum(np.array_equal(f, ref[n]) for n, f, _ in res))
    s = dict(n_recordings=len(res), total_samples=int(sum(n for *_, n in res)), bit_identical=exact,
             max_rel_diff=float(rel.max()), n_within_1e_5=int(np.sum(rel <= 1e-5)))
    (out / "extractor_full.json").write_text(json.dumps(s, indent=1))
    print(s)


if __name__ == "__main__":
    main()
