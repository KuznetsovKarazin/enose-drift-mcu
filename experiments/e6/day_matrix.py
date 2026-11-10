"""E6 (revision 2, post hoc): static head trained on one repetition (day) of a D2 board and
tested on every other repetition of the same board, d=16, frozen E4 head ridge parameter.
Shows whether the day-to-day conclusion of E4 depends on the assumed repetition order.

Usage: python experiments/e6/day_matrix.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments" / "e4"))
import run as E4  # noqa: E402
from enose_drift.stream import make_source  # noqa: E402

HP = json.loads((ROOT / "results/e4_20261109/hyperparameters.json").read_text())


def main():
    out = Path(sys.argv[1]); D = E4.load(); y = D["y"]; d = 16
    X = E4.feats(d); lam = HP[str(d)]["head_lambda"]; res = {}
    for b in range(1, 6):
        m = D["board"] == b; reps = sorted({int(r) for r in D["repetition"][m]})
        M = []
        for ri in reps:
            tr = m & (D["repetition"] == ri)
            src = make_source(X[tr], y[tr], lam, classes=E4.CLASSES)
            row = []
            for rj in reps:
                te = m & (D["repetition"] == rj)
                Z = src.pre(X[te]); A = np.c_[Z, np.ones(len(Z))]
                row.append(float(np.mean(np.argmax(A @ src.W, 1) == y[te])))
            M.append(row)
        off = [M[i][j] for i in range(len(reps)) for j in range(len(reps)) if i != j]
        res[str(b)] = dict(repetitions=reps, matrix=M, off_diagonal_min=min(off), off_diagonal_max=max(off))
    (out / "day_matrix.json").write_text(json.dumps(res, indent=1))
    print({b: (r["off_diagonal_min"], r["off_diagonal_max"]) for b, r in res.items()})


if __name__ == "__main__":
    main()
