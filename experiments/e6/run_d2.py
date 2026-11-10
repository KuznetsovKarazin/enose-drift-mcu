"""E6 (revision, post hoc) on D2: float32 DS and DS fused into the head,
sensitivity of E4 to data exclusions, and board-level cluster bootstrap.

Usage: python experiments/e6/run_d2.py OUT_DIR [--seeds 20] [--jobs 4]

Post hoc: frozen E4 hyperparameters, same seeds as E4, nothing re-tuned.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "e4"))

import run as E4  # noqa: E402  (experiments/e4/run.py)
from enose_drift import f32 as F  # noqa: E402
from enose_drift import transfer as T  # noqa: E402
from enose_drift.stream import make_source  # noqa: E402

HP4 = json.loads((ROOT / "results/e4_20261109/hyperparameters.json").read_text())
SEEDS = None
EXCL = {"none": None, "no_b5_methane": "b5m", "no_gappy": "gappy", "both": "both"}


def apply_exclusion(key):
    """Restrict the module-level D2 table of experiments/e4/run.py."""
    E4.D = None
    D = E4.load()
    keep = np.ones(len(D["y"]), bool)
    if key in ("b5m", "both"):
        keep &= ~((D["board"] == 5) & (D["gas"] == "methane"))
    if key in ("gappy", "both"):
        keep &= D["n_samples"] >= 50000
    E4.D = {k: v[keep] for k, v in D.items()}


def sens_job(args):
    excl, d, name, k, seed, pair = args
    apply_exclusion(EXCL[excl])
    h = HP4[str(d)]
    r = E4.transfer_job((d, h["head_lambda"], name, h["methods"].get(name, {}), k, seed, tuple(pair)))
    r["exclusion"] = excl
    return r


def ds_f32_job(args):
    """Same draw as E4 M8 (seeded), DS fitted in float32 by the firmware kernel;
    predictions of the separate map and of the fused head are compared with float64."""
    d, k, seed, (i, j) = args
    apply_exclusion(None)
    D = E4.load(); X = E4.feats(d); y = D["y"]
    board, lev, rep, gas = D["board"], D["level"], D["repetition"], D["gas"]
    h = HP4[str(d)]; rho = h["methods"]["M8"]["rho"]
    src = make_source(X[board == i], y[board == i], h["head_lambda"], classes=E4.CLASSES)
    Z = src.pre(X)
    rng = np.random.default_rng(seed)
    conds = sorted({(g, l) for g, l in zip(gas[board == j], lev[board == j])})
    chosen = [conds[t] for t in rng.permutation(len(conds))[:k]]
    ds = T.DirectStandardisation(src.W, E4.CLASSES, rho)
    S, Mm, used = [], [], []
    for g, l in chosen:
        sj = np.flatnonzero((board == j) & (gas == g) & (lev == l) & (rep == 1))
        si = np.flatnonzero((board == i) & (gas == g) & (lev == l) & (rep == 1))
        if len(sj) == 0 or len(si) == 0:
            continue
        used.append(sj[0]); ds.pair(Z[sj[0]], Z[si[0]]); S.append(sj[0]); Mm.append(si[0])
    ds.fit()
    ev = np.setdiff1d(np.flatnonzero(board == j), used)
    ev = ev[rng.permutation(len(ev))]
    p64 = ds.predict_many(Z[ev])
    # fused float64 head: scores = [[z 1] F, 1] W = [z 1] (F W_lin) + w_bias
    Fm = ds.F if hasattr(ds, "F") else None
    Wf = Fm @ src.W[:-1]; Wf[-1] += src.W[-1]
    pf64 = np.argmax(np.hstack([Z[ev], np.ones((len(ev), 1))]) @ Wf, axis=1)
    # float32 kernels: pre, ridge to identity, apply + argmax; fused head in float32
    fs = F.F32Source(src)
    Z32 = fs.pre(X)
    Zs, Zm = F.f32(Z32[S]), F.f32(Z32[Mm])
    G = np.zeros((d + 1) ** 2, np.float32); F32m = np.zeros((d + 1) * d, np.float32)
    ok = F.LIB.f_ridge(F.p(Zs), F.p(Zm), len(S), d, d, np.float32(rho), None, 1, F.p(G), F.p(F32m))
    F32m = F32m.reshape(d + 1, d)
    U = np.empty((len(ev), d), np.float32)
    for t, e in enumerate(ev):
        u = np.empty(d, np.float32)
        F.LIB.f_ds_apply(F.p(F.f32(Z32[e])), F.p(F32m), d, F.p(u)); U[t] = u
    p32 = np.empty(len(ev), np.int32)
    F.LIB.f_argmax_many(F.p(F.f32(U)), F.p(fs.W), len(ev), d, E4.CLASSES, F.pi(p32))
    Wf32 = F.f32(F32m.astype(np.float32) @ fs.W[:-1]); Wf32[-1] += fs.W[-1]
    pf32 = np.empty(len(ev), np.int32)
    F.LIB.f_argmax_many(F.p(F.f32(Z32[ev])), F.p(F.f32(Wf32)), len(ev), d, E4.CLASSES, F.pi(pf32))
    yv = y[ev]
    return dict(d=d, k=k, seed=seed, pair=[i, j], ok=int(ok), n=int(len(ev)),
                acc64=float(np.mean(p64 == yv)), acc_fused64=float(np.mean(pf64 == yv)),
                acc32=float(np.mean(p32 == yv)), acc_fused32=float(np.mean(pf32 == yv)),
                agree_32_64=float(np.mean(p32 == p64)), agree_fused_64=float(np.mean(pf64 == p64)),
                agree_fused32_64=float(np.mean(pf32 == p64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = [E4.SEED_BASE + i for i in range(a.seeds)]
    pairs = [p for p in E4.PAIRS if p != E4.TUNE_PAIR]
    ds_jobs = [(d, k, s, p) for d in (16, 64) for k in (8, 16) for s in seeds for p in pairs]
    with ProcessPoolExecutor(a.jobs) as pool:
        ds = list(pool.map(ds_f32_job, ds_jobs, chunksize=8))
    (a.out / "d2_ds_f32_runs.json").write_text(json.dumps(ds))
    sjobs = []
    for excl in EXCL:
        for d in (16,):
            for name, k in (("M0", 0), ("M1", 0), ("M1", 16), ("M8", 16), ("M6", 16), ("M9", 16), ("M4", 16)):
                for s in (seeds if (k or name == "M1") else seeds[:1]):
                    for p in pairs:
                        sjobs.append((excl, d, name, k, s, p))
    with ProcessPoolExecutor(a.jobs) as pool:
        sens = list(pool.map(sens_job, sjobs, chunksize=16))
    (a.out / "d2_sensitivity_runs.json").write_text(json.dumps(sens))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
