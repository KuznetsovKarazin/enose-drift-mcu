"""Precompute the host float32 reference outputs that mcu/hil.py compares with the board,
so that the board run needs no C compiler on the PC (the replica uses gcc via ctypes).

Usage: python mcu/make_hil_expected.py   ->  mcu/hil_expected.json.gz
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "e6"))
sys.path.insert(0, str(ROOT / "experiments" / "e4"))

import run_d1 as R  # noqa: E402
import run_d2 as R2  # noqa: E402
import run as E4  # noqa: E402

SEED, K = 20270131, 20
PLAN = [(n, 16) for n in ("M0", "M1", "M5", "M6", "M7", "M1+M5")] + [(n, 128) for n in ("M0", "M1", "M5", "M1+M5", "M6")]


def main():
    out = {"seed": SEED, "k": K, "d1": {}, "ds": {}}
    for name, d in PLAN:
        rows = R.stream32(name, d, R.HP2[str(d)]["methods"].get(name, {}), K, SEED)
        out["d1"][f"{name}|{d}"] = [dict(batch=r["batch"], accuracy=r["accuracy"], pred=r["pred"]) for r in rows]
        print(name, d, "done", flush=True)
    for (i, j) in ((1, 3), (1, 4), (1, 5)):
        out["ds"][f"{i}->{j}"] = R2.ds_f32_job((16, 16, E4.SEED_BASE, (i, j)))["acc_fused32"]
    p = HERE / "hil_expected.json.gz"
    with gzip.open(p, "wt") as fh:
        json.dump(out, fh)
    print(p)


if __name__ == "__main__":
    main()
