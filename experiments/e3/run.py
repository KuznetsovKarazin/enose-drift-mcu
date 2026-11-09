"""E3: log10-concentration regression under drift on D1b, per gas.

Usage: python experiments/e3/run.py OUT_DIR [--seeds 20] [--jobs 4]

Gas identity is taken as known (regression is evaluated per gas). Source:
batch-1 samples of the gas. Stream: batches 2-10, samples of that gas only;
the first k of each batch are references with known concentration.
Hyperparameters: chosen on batch 2 (mean RMSE over gases, 5 tuning seeds,
k=5; M1 at k=0), then frozen. Headline: batches 3-10.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from enose_drift.data import D1_GASES, load_d1  # noqa: E402
from enose_drift.stream import build, feature_index, make_source, run_stream  # noqa: E402

SEED_BASE = 20270131 + 300
TUNE_SEEDS = [SEED_BASE + 1000 + i for i in range(5)]
DIMS = (16, 128)
BUDGETS = (0, 1, 5, 20)
TUNE_K = 5
ZERO_LABEL = ("M0", "M1")
LABELLED = ("M5", "M6", "M7", "M1+M5", "M1+M7")
GRIDS = {
    "M1": {"alpha": [0.003, 0.01, 0.03, 0.1, 0.3]},
    "M5": {"mu": [0.01, 0.03, 0.1, 0.3, 1.0]},
    "M6": {"forget": [0.8, 0.9, 0.95, 0.99, 1.0]},
    "M7": {"gamma": [0.1, 1.0, 10.0, 100.0, 1000.0]},
    "M1+M5": {"mu": [0.01, 0.03, 0.1, 0.3, 1.0]},
    "M1+M7": {"gamma": [0.1, 1.0, 10.0, 100.0, 1000.0]},
}
HEAD_LAMBDAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
DATA = None


def data(d, g):
    global DATA
    if DATA is None:
        DATA = load_d1(ROOT / "data" / "extracted" / "D1b")
    m = DATA.gas == g
    return DATA.X[m][:, feature_index(d)], np.log10(DATA.concentration[m]), DATA.batch[m]


def job(args):
    d, g, lam, name, hp, k, seed, batches = args
    X, y, b = data(d, g)
    src = make_source(X[b == 1], y[b == 1], lam, classes=None)
    m = build(name, src, hp, classes=None)
    rows = run_stream(m, src.pre(X), y, b, batches, k, seed, classes=None)
    return dict(d=d, gas=D1_GASES[g], method=name, hp=hp, k=k, seed=seed, rows=rows,
                state_bytes=m.state_bytes(), payload_bytes=m.payload_bytes())


def static_rmse(d, g, lam, test_batch):
    X, y, b = data(d, g)
    if not (b == test_batch).any():
        return np.nan
    src = make_source(X[b == 1], y[b == 1], lam, classes=None)
    m = build("M0", src, {}, classes=None)
    e = m.predict_many(src.pre(X[b == test_batch])) - y[b == test_batch]
    return float(np.sqrt(np.mean(e ** 2)))


def grid(name):
    g = GRIDS.get(name, {})
    return [dict(zip(g, v)) for v in itertools.product(*g.values())] if g else [{}]


def mean_rmse(res, batches):
    vals = [x["rmse"] for r in res for x in r["rows"] if x["batch"] in batches]
    return float(np.mean(vals)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pool = ProcessPoolExecutor(a.jobs)
    gases = range(len(D1_GASES))
    lam, hp, tuning = {}, {}, []
    for d in DIMS:
        sc = {L: np.nanmean([static_rmse(d, g, L, 2) for g in gases]) for L in HEAD_LAMBDAS}
        lam[d] = min(sc, key=lambda L: (sc[L], -L))
        hp[d] = {"M0": {}}
        for name in GRIDS:
            k = 0 if name in ZERO_LABEL else TUNE_K
            cands = ([dict(c, alpha=hp[d]["M1"]["alpha"]) for c in grid(name)]
                     if name.startswith("M1+") else grid(name))
            scores = []
            for c in cands:
                res = list(pool.map(job, [(d, g, lam[d], name, c, k, s, [2]) for g in gases for s in TUNE_SEEDS]))
                per_gas = [mean_rmse([r for r in res if r["gas"] == D1_GASES[g]], [2]) for g in gases]
                scores.append(float(np.nanmean(per_gas)))
                tuning.append(dict(d=d, method=name, hp=c, k=k, mean_rmse_batch2=scores[-1]))
            hp[d][name] = cands[int(np.nanargmin(scores))]
    frozen = {str(d): dict(head_lambda=lam[d], methods=hp[d]) for d in DIMS}
    (a.out / "hyperparameters.json").write_text(json.dumps(frozen, indent=1))
    (a.out / "tuning.json").write_text(json.dumps(tuning, indent=1))

    seeds = [SEED_BASE + i for i in range(a.seeds)]
    jobs = [(d, g, lam[d], n, hp[d][n], k, s, list(range(2, 11)))
            for d in DIMS for k in BUDGETS for n in (ZERO_LABEL + (LABELLED if k else ()))
            for g in gases for s in seeds]
    runs = list(pool.map(job, jobs, chunksize=8))
    (a.out / "e3_runs.json").write_text(json.dumps(runs))

    summary = []
    for d in DIMS:
        for k in BUDGETS:
            for n in ZERO_LABEL + LABELLED:
                rs = [r for r in runs if r["d"] == d and r["k"] == k and r["method"] == n]
                if not rs:
                    continue
                per_gas = {}
                for g in D1_GASES:
                    rg = [r for r in rs if r["gas"] == g]
                    per_gas[g] = dict(
                        rmse=mean_rmse(rg, range(3, 11)),
                        median_rel_error=float(np.mean([x["median_rel_error"] for r in rg for x in r["rows"] if x["batch"] >= 3])))
                summary.append(dict(d=d, k=k, method=n, hp=rs[0]["hp"],
                                    rmse_mean_over_gases=float(np.nanmean([v["rmse"] for v in per_gas.values()])),
                                    median_rel_error_mean=float(np.nanmean([v["median_rel_error"] for v in per_gas.values()])),
                                    per_gas=per_gas, state_bytes=max(r["state_bytes"] for r in rs),
                                    payload_bytes=rs[0]["payload_bytes"]))
    (a.out / "e3_summary.json").write_text(json.dumps(summary, indent=1))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (a.out / "run_info.json").write_text(json.dumps(dict(
        commit=commit, python=sys.version, platform=platform.platform(), numpy=np.__version__,
        seconds=round(time.time() - t0, 1), test_seeds=seeds, tune_seeds=TUNE_SEEDS), indent=1))
    print(json.dumps(frozen, indent=1), f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
