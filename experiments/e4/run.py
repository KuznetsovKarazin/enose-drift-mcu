"""E4: calibration transfer between the five twin boards of D2, and
within-board day-to-day drift.

Usage: python experiments/e4/run.py OUT_DIR [--seeds 20] [--jobs 4]
Requires data/cache/d2_features.npz (scripts/extract_d2.py).

Transfer i -> j: the head (4-gas least-squares classifier) and the
preprocessing are fitted on all records of master board i. On slave board j,
k of the 40 (gas, level) conditions are drawn (seeded); their repetition-1
records on both boards form the transfer pairs and, with their gas labels,
the references for head-update methods. Evaluation: all other records of j.
Tuning: pair 1 -> 2 only, k = 8, 5 tuning seeds; reported: the other 19 pairs.
Day drift (boards 1-3): train on R1 of a board, stream R2..R4 with k references per day.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from enose_drift import transfer as T  # noqa: E402
from enose_drift.stream import build, make_source, run_stream  # noqa: E402

SEED_BASE = 20270131 + 400
TUNE_SEEDS = [SEED_BASE + 1000 + i for i in range(5)]
GASES = ("co", "ethanol", "ethylene", "methane")
CLASSES = 4
DIMS = (16, 64)
BUDGETS = (0, 2, 4, 8, 16, 40)
TUNE_K = 8
TUNE_PAIR = (1, 2)
PAIRS = [(i, j) for i in range(1, 6) for j in range(1, 6) if i != j]
GRIDS = {
    "M1": {"alpha": [0.01, 0.03, 0.1, 0.3]},
    "M4": {"rho": [0.01, 0.1, 1.0, 10.0]},
    "M5": {"mu": [0.03, 0.1, 0.3, 1.0]},
    "M6": {"forget": [0.9, 0.99, 1.0]},
    "M7": {"gamma": [0.1, 1.0, 10.0, 100.0]},
    "M8": {"rho": [0.01, 0.1, 1.0, 10.0]},
    "M9": {"rho": [0.01, 0.1, 1.0, 10.0]},
}
HEAD_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
TRANSFER = ("M4", "M8", "M9")
HEAD_UPDATE = ("M5", "M6", "M7")
CACHE = ROOT / "data" / "cache" / "d2_features.npz"
D = None


def load():
    global D
    if D is None:
        z = np.load(CACHE)
        D = {k: z[k] for k in z.files}
        D["y"] = np.array([GASES.index(g) for g in D["gas"]])
    return D


def feats(d):
    X = load()["X"].reshape(-1, 8, 8)
    return (X[:, :, :2] if d == 16 else X).reshape(len(X), -1)


def block(d):
    return d // 8


def make_method(name, src, hp, d):
    if name == "M4":
        return T.PairedAffine(src.W, CLASSES, hp["rho"])
    if name == "M8":
        return T.DirectStandardisation(src.W, CLASSES, hp["rho"])
    if name == "M9":
        return T.ChannelDS(src.W, CLASSES, hp["rho"], block(d))
    return build(name, src, hp, classes=CLASSES)


def transfer_job(args):
    d, lam, name, hp, k, seed, (i, j) = args
    Dd = load(); X = feats(d); y = Dd["y"]
    board, lev, rep, gas = Dd["board"], Dd["level"], Dd["repetition"], Dd["gas"]
    src = make_source(X[board == i], y[board == i], lam, classes=CLASSES)
    Z = src.pre(X)
    rng = np.random.default_rng(seed)
    conds = sorted({(g, l) for g, l in zip(gas[board == j], lev[board == j])})
    chosen = [conds[t] for t in rng.permutation(len(conds))[:k]]
    m = make_method(name, src, hp, d)
    used = []
    for g, l in chosen:
        sj = np.flatnonzero((board == j) & (gas == g) & (lev == l) & (rep == 1))
        si = np.flatnonzero((board == i) & (gas == g) & (lev == l) & (rep == 1))
        if len(sj) == 0 or len(si) == 0:
            continue
        used.append(sj[0])
        if name in TRANSFER:
            m.pair(Z[sj[0]], Z[si[0]])
        else:
            m.reference(Z[sj[0]], int(y[sj[0]]))
    if name in TRANSFER:
        m.fit()
    m.end_reference_session()
    ev = np.setdiff1d(np.flatnonzero(board == j), used)
    ev = ev[rng.permutation(len(ev))]
    if m.sequential:
        pred = np.array([m.observe(Z[t]) for t in ev])
    else:
        pred = m.predict_many(Z[ev])
    return dict(d=d, method=name, hp=hp, k=k, seed=seed, pair=[i, j], n_eval=int(len(ev)),
                accuracy=float(np.mean(pred == y[ev])),
                macro_f1=float(f1_score(y[ev], pred, labels=range(CLASSES), average="macro", zero_division=0)),
                state_bytes=m.state_bytes(), payload_bytes=m.payload_bytes())


def day_job(args):
    d, lam, name, hp, k, seed, b = args
    Dd = load(); X = feats(d); y = Dd["y"]
    mb = Dd["board"] == b
    Xb, yb, rb = X[mb], y[mb], Dd["repetition"][mb]
    src = make_source(Xb[rb == 1], yb[rb == 1], lam, classes=CLASSES)
    m = build(name, src, hp, classes=CLASSES)
    rows = run_stream(m, src.pre(Xb), yb, rb, [2, 3, 4], k, seed, classes=CLASSES)
    return dict(d=d, method=name, hp=hp, k=k, seed=seed, board=b, rows=rows, state_bytes=m.state_bytes())


def grid(name):
    g = GRIDS.get(name, {})
    return [dict(zip(g, v)) for v in itertools.product(*g.values())] if g else [{}]


def static_acc(d, lam, i, j):
    return transfer_job((d, lam, "M0", {}, 0, 0, (i, j)))["accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pool = ProcessPoolExecutor(a.jobs)
    lam, hp, tuning = {}, {}, []
    for d in DIMS:
        sc = {L: static_acc(d, L, *TUNE_PAIR) for L in HEAD_LAMBDAS}
        lam[d] = max(sc, key=lambda L: (sc[L], L))
        hp[d] = {"M0": {}}
        for name in GRIDS:
            k = 0 if name == "M1" else TUNE_K
            cands = grid(name)
            scores = []
            for c in cands:
                res = list(pool.map(transfer_job, [(d, lam[d], name, c, k, s, TUNE_PAIR) for s in TUNE_SEEDS]))
                scores.append(float(np.mean([r["accuracy"] for r in res])))
                tuning.append(dict(d=d, method=name, hp=c, k=k, mean_accuracy_pair_1_2=scores[-1]))
            hp[d][name] = cands[int(np.argmax(scores))]
    frozen = {str(d): dict(head_lambda=lam[d], methods=hp[d]) for d in DIMS}
    (a.out / "hyperparameters.json").write_text(json.dumps(frozen, indent=1))
    (a.out / "tuning.json").write_text(json.dumps(tuning, indent=1))

    seeds = [SEED_BASE + i for i in range(a.seeds)]
    pairs = [p for p in PAIRS if p != TUNE_PAIR]
    jobs = []
    for d in DIMS:
        for k in BUDGETS:
            names = ("M0", "M1") + ((TRANSFER + HEAD_UPDATE) if k else ())
            for n in names:
                for p in pairs:
                    for s in (seeds if (k or n == "M1") else seeds[:1]):
                        jobs.append((d, lam[d], n, hp[d][n], k, s, p))
    runs = list(pool.map(transfer_job, jobs, chunksize=16))
    (a.out / "e4_transfer_runs.json").write_text(json.dumps(runs))

    summary = []
    for d in DIMS:
        for k in BUDGETS:
            for n in ("M0", "M1") + TRANSFER + HEAD_UPDATE:
                rs = [r for r in runs if r["d"] == d and r["k"] == k and r["method"] == n]
                if not rs:
                    continue
                per_pair = {f"{p[0]}->{p[1]}": float(np.mean([r["accuracy"] for r in rs if tuple(r["pair"]) == p]))
                            for p in pairs}
                summary.append(dict(d=d, k=k, method=n, hp=rs[0]["hp"],
                                    accuracy=float(np.mean(list(per_pair.values()))),
                                    worst_pair=float(min(per_pair.values())),
                                    macro_f1=float(np.mean([r["macro_f1"] for r in rs])),
                                    per_pair=per_pair, state_bytes=max(r["state_bytes"] for r in rs),
                                    payload_bytes=rs[0]["payload_bytes"]))
    (a.out / "e4_transfer_summary.json").write_text(json.dumps(summary, indent=1))

    day_jobs = [(d, lam[d], n, hp[d][n], k, s, b) for d in DIMS for b in (1, 2, 3)
                for k in (0, 2, 4) for n in (("M0", "M1") + (HEAD_UPDATE if k else ())) for s in seeds]
    days = list(pool.map(day_job, day_jobs, chunksize=16))
    (a.out / "e4_days_runs.json").write_text(json.dumps(days))
    day_summary = []
    for d in DIMS:
        for k in (0, 2, 4):
            for n in ("M0", "M1") + HEAD_UPDATE:
                rs = [r for r in days if r["d"] == d and r["k"] == k and r["method"] == n]
                if rs:
                    day_summary.append(dict(d=d, k=k, method=n,
                        accuracy_by_repetition={rep: float(np.mean([x["accuracy"] for r in rs for x in r["rows"] if x["batch"] == rep])) for rep in (2, 3, 4)},
                        state_bytes=max(r["state_bytes"] for r in rs)))
    (a.out / "e4_days_summary.json").write_text(json.dumps(day_summary, indent=1))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (a.out / "run_info.json").write_text(json.dumps(dict(
        commit=commit, python=sys.version, platform=platform.platform(), numpy=np.__version__,
        seconds=round(time.time() - t0, 1), test_seeds=seeds, tune_seeds=TUNE_SEEDS, tune_pair=TUNE_PAIR,
        feature_cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest()), indent=1))
    print(json.dumps(frozen, indent=1), f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
