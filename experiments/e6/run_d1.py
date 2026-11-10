"""E6 (revision, post hoc) on D1b: float32 replica of the E2 streams, harder
stream scenarios, intercept penalty of block least squares, RLS variants for
E2 and E3.

Usage: python experiments/e6/run_d1.py OUT_DIR [--seeds 20] [--jobs 4]

Everything here was added after the reviews of 2026-11-10 and is post hoc:
hyperparameters are the frozen E2/E3 values, nothing is re-tuned, and the
results are reported as sensitivity analyses, separately from E1-E5.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from enose_drift import methods as M  # noqa: E402
from enose_drift.data import load_d1  # noqa: E402
from enose_drift.stream import CLASSES, build, feature_index, make_source  # noqa: E402

E2_SEED_BASE = 20270131          # identical seeds to E2, so the float32 replica sees the same streams
E3_SEED_BASE = 20270131 + 300
HP2 = json.loads((ROOT / "results/e1e2_20261109/hyperparameters.json").read_text())
HP3 = json.loads((ROOT / "results/e3_20261109/hyperparameters.json").read_text())
BATCHES = list(range(2, 11))
DATA = None


def data(d):
    global DATA
    if DATA is None:
        DATA = load_d1(ROOT / "data" / "extracted" / "D1b")
    return DATA.X[:, feature_index(d)], DATA.gas, DATA.batch, DATA.concentration


# ---------------------------------------------------------------- variants
class BlockOLSBias(M.BlockOLS):
    """M7 with the intercept penalised like the weights (gamma on every row)."""

    def end_reference_session(self):
        if self.buf_y:
            Z = np.array(self.buf_z); A = M.augment(Z)
            Y = np.asarray(self.buf_y, float)[:, None] if self.classes is None else np.eye(self.classes)[self.buf_y]
            R = self.gamma * np.eye(A.shape[1])
            self.W = np.linalg.solve(A.T @ A + R, A.T @ Y + R @ self.W)
        self.buf_z, self.buf_y = [], []


class RLSBounded(M.RLS):
    """M6 with the covariance trace bounded by that of the warm start
    (constant-trace bound against wind-up; O(d^2) per update, no extra state)."""

    def __init__(self, W, classes, lam, P0):
        super().__init__(W, classes, lam, None, P0=P0)
        self.cap = float(np.trace(P0))

    def reference(self, z, y):
        super().reference(z, y)
        tr = float(np.trace(self.P))
        if tr > self.cap:
            self.P *= self.cap / tr


def make(name, src, hp, classes=CLASSES):
    if name == "M7b":
        return BlockOLSBias(src.W, classes, hp["gamma"])
    if name == "M6b":
        return RLSBounded(src.W, classes, hp["forget"], src.P0)
    if name.startswith("M6@"):
        return M.RLS(src.W, classes, float(name[3:]), None, P0=src.P0)
    return build(name, src, hp, classes=classes)


# ---------------------------------------------------------------- stream with scenarios
def order_batch(idx, y, rng, order, refs, k):
    """Return (references, rest) for one batch.
    order: 'shuffled' (E2) or 'grouped' (gases arrive in blocks, random gas order).
    refs: 'uniform' (E2: first k of the stream) or 'one_gas' (all k references of one
    randomly chosen gas present in the batch, drawn before the stream)."""
    idx = idx[rng.permutation(len(idx))]
    if refs == "uniform":
        ref, rest = idx[:k], idx[k:]
    else:
        gases = np.unique(y[idx]); g = gases[rng.integers(len(gases))]
        cand = idx[y[idx] == g]
        ref = cand[:k]
        rest = np.setdiff1d(idx, ref, assume_unique=True)
        rest = idx[np.isin(idx, rest)]
    if order == "grouped":
        gs = np.unique(y[rest]); gs = gs[rng.permutation(len(gs))]
        rest = np.concatenate([rest[y[rest] == g] for g in gs])
    return ref, rest


def stream64(name, d, hp, k, seed, order, refs, classes=CLASSES, Xy=None):
    X, y, b, _ = data(d) if Xy is None else Xy
    src = make_source(X[b == 1], y[b == 1], HP2[str(d)]["head_lambda"]) if Xy is None else Xy[4]
    m = make(name, src, hp, classes)
    Z = src.pre(X)
    rng = np.random.default_rng(seed)
    rows = []
    for bb in BATCHES:
        ref, rest = order_batch(np.flatnonzero(b == bb), y, rng, order, refs, k)
        for i in ref:
            m.reference(Z[i], int(y[i]))
        m.end_reference_session()
        pred = (np.array([m.observe(Z[i]) for i in rest]) if m.sequential else m.predict_many(Z[rest]))
        rows.append(dict(batch=bb, n=int(len(rest)), accuracy=float(np.mean(pred == y[rest])),
                         macro_f1=float(f1_score(y[rest], pred, labels=range(classes), average="macro", zero_division=0)),
                         pred=pred.astype(int).tolist() if order == "shuffled" and refs == "uniform" else None))
    return rows


def stream32(name, d, hp, k, seed):
    from enose_drift import f32 as F
    X, y, b, _ = data(d)
    src = make_source(X[b == 1], y[b == 1], HP2[str(d)]["head_lambda"])
    fs = F.F32Source(src)
    Z = fs.pre(X)                                  # float32 preprocessing in the C kernel
    m = F.F32Method(name, fs, hp)
    rng = np.random.default_rng(seed)
    rows = []
    for bb in BATCHES:
        ref, rest = order_batch(np.flatnonzero(b == bb), y, rng, "shuffled", "uniform", k)
        for i in ref:
            m.reference(Z[i], int(y[i]))
        m.end_reference_session()
        pred = m.run_unlabelled(Z[rest])
        rows.append(dict(batch=bb, n=int(len(rest)), accuracy=float(np.mean(pred == y[rest])),
                         macro_f1=float(f1_score(y[rest], pred, labels=range(CLASSES), average="macro", zero_division=0)),
                         pred=pred.astype(int).tolist()))
    return rows


def job(args):
    kind, name, d, hpname, k, seed, order, refs = args
    hp = HP2[str(d)]["methods"].get(hpname, {})
    if kind == "f32":
        rows = stream32(name, d, hp, k, seed)
    else:
        rows = stream64(name, d, hp, k, seed, order, refs)
    return dict(kind=kind, method=name, d=d, k=k, seed=seed, order=order, refs=refs, rows=rows)


# ---------------------------------------------------------------- E3 RLS variants
def e3_job(args):
    name, d, g, k, seed = args
    X, y, b, conc = data(d)
    sel = y == g
    Xg, bg, t = X[sel], b[sel], np.log10(conc[sel])
    hp = dict(HP3[str(d)]["methods"]["M6"])
    src = make_source(Xg[bg == 1], t[bg == 1], HP3[str(d)]["head_lambda"], classes=None)
    m = make(name, src, hp, classes=None)
    Z = src.pre(Xg)
    rng = np.random.default_rng(seed)
    rmse = []
    for bb in BATCHES:
        idx = np.flatnonzero(bg == bb)
        if len(idx) == 0:
            continue
        idx = idx[rng.permutation(len(idx))]
        for i in idx[:k]:
            m.reference(Z[i], t[i])
        m.end_reference_session()
        rest = idx[k:]
        if len(rest) == 0:
            continue
        pred = m.predict_many(Z[rest])
        if bb >= 3:
            rmse.append(float(np.sqrt(np.mean((pred - t[rest]) ** 2))))
    return dict(method=name, d=d, gas=int(g), k=k, seed=seed, rmse=float(np.mean(rmse)) if rmse else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = [E2_SEED_BASE + i for i in range(a.seeds)]
    jobs = []
    core = ("M0", "M1", "M5", "M6", "M7", "M1+M5", "M1+M7")
    for d in (16, 128):
        for k in (0, 1, 5, 20):
            for n in core:
                if k == 0 and n not in ("M0", "M1"):
                    continue
                for s in seeds:
                    jobs.append(("f32", n, d, n, k, s, "shuffled", "uniform"))
                    jobs.append(("f64", n, d, n, k, s, "shuffled", "uniform"))
        for order, refs in (("grouped", "uniform"), ("shuffled", "one_gas"), ("grouped", "one_gas")):
            for k in (0, 5, 20):
                for n in ("M0", "M1", "M5", "M6", "M7", "M1+M5"):
                    if k == 0 and n not in ("M0", "M1"):
                        continue
                    if refs == "one_gas" and k == 0:
                        continue
                    for s in seeds:
                        jobs.append(("f64", n, d, n, k, s, order, refs))
        for k in (1, 5, 20):
            for s in seeds:
                jobs.append(("f64", "M7b", d, "M7", k, s, "shuffled", "uniform"))
                jobs.append(("f64", "M6b", d, "M6", k, s, "shuffled", "uniform"))
                for beta in ("0.95", "0.99"):
                    jobs.append(("f64", f"M6@{beta}", d, "M6", k, s, "shuffled", "uniform"))
    with ProcessPoolExecutor(a.jobs) as pool:
        runs = list(pool.map(job, jobs, chunksize=4))
    with gzip.open(a.out / "d1_runs.json.gz", "wt") as fh:
        json.dump(runs, fh)
    e3_seeds = [E3_SEED_BASE + i for i in range(a.seeds)]
    e3_jobs = [(n, d, g, k, s) for n in ("M6", "M6b", "M6@0.9", "M6@0.95", "M6@0.99") for d in (16, 128)
               for g in range(6) for k in (1, 5, 20) for s in e3_seeds]
    with ProcessPoolExecutor(a.jobs) as pool:
        e3 = list(pool.map(e3_job, e3_jobs, chunksize=8))
    (a.out / "e3_rls_runs.json").write_text(json.dumps(e3))
    print(f"done in {time.time() - t0:.0f} s, {len(runs)} streams, {len(e3)} E3 runs")


if __name__ == "__main__":
    main()
