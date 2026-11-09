"""E1 (static settings) and E2 (streaming, reference budget) on D1b.

Usage: python experiments/e1e2/run.py OUT_DIR [--seeds 20] [--jobs 4]

Phase 1 selects hyperparameters on batch 2 only (source = batch 1) with
tuning seeds; phase 2 freezes them and runs batches 2-10 with test seeds.
Batch 2 is part of the stream (state continuity) but excluded from the
headline averages, which cover batches 3-10.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from enose_drift.budget import head_bytes  # noqa: E402
from enose_drift.data import load_d1  # noqa: E402
from enose_drift.stream import CLASSES, build, feature_index, make_source, run_stream, setting_scores  # noqa: E402

SEED_BASE = 20270131
TUNE_SEEDS = [SEED_BASE + 1000 + i for i in range(5)]
DIMS = (16, 128)
BUDGETS = (0, 1, 5, 20)
TUNE_K = 5
ZERO_LABEL = ("M0", "M1", "M2")
LABELLED = ("M3", "M4", "M5", "M6", "M7", "M1+M5", "M1+M7")
GRIDS = {
    "M1": {"alpha": [0.001, 0.003, 0.01, 0.03, 0.1]},
    "M2": {"beta": [0.001, 0.003, 0.01, 0.03, 0.1]},
    "M3": {"r": [1, 2, 3], "eta": [0.1, 0.3, 1.0]},
    "M4": {},
    "M5": {"mu": [0.03, 0.1, 0.3, 0.6, 1.0]},
    "M6": {"forget": [0.9, 0.95, 0.99, 0.999, 1.0]},
    "M7": {"gamma": [0.1, 1.0, 10.0, 100.0]},
    "M1+M5": {"mu": [0.03, 0.1, 0.3, 0.6, 1.0]},
    "M1+M7": {"gamma": [0.1, 1.0, 10.0, 100.0]},
}
HEAD_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]

DATA = None


def data(d):
    global DATA
    if DATA is None:
        DATA = load_d1(ROOT / "data" / "extracted" / "D1b")
    X = DATA.X[:, feature_index(d)]
    return X, DATA.gas, DATA.batch


def source_for(d, lam):
    X, y, b = data(d)
    return make_source(X[b == 1], y[b == 1], lam)


def stream_job(args):
    d, lam, name, hp, k, seed, batches = args
    X, y, b = data(d)
    src = source_for(d, lam)
    m = build(name, src, hp)
    rows = run_stream(m, src.pre(X), y, b, batches, k, seed)
    return dict(d=d, method=name, hp=hp, k=k, seed=seed, rows=rows,
                state_bytes=m.state_bytes(), payload_bytes=m.payload_bytes())


def grid(name):
    g = GRIDS.get(name, {})
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*g.values())] if keys else [{}]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    test_seeds = [SEED_BASE + i for i in range(a.seeds)]
    pool = ProcessPoolExecutor(a.jobs)

    # Phase 1a: head ridge lambda, static head on batch 2.
    X1, y1, b1 = data(128)
    lam = {}
    for d in DIMS:
        X, y, b = data(d)
        scores = {L: setting_scores(X, y, b, L, 1, 2) for L in HEAD_LAMBDAS}
        lam[d] = max(scores, key=lambda L: (scores[L], -L))
    # Phase 1b: method hyperparameters on batch 2 with tuning seeds.
    hp, tuning = {}, []
    for d in DIMS:
        hp[d] = {"M0": {}}
        for name in [n for n in GRIDS]:
            k = 0 if name in ZERO_LABEL else TUNE_K
            if name.startswith("M1+"):
                cands = [dict(c, alpha=hp[d]["M1"]["alpha"]) for c in grid(name)]
            else:
                cands = grid(name)
            jobs = [(d, lam[d], name, c, k, s, [2]) for c in cands for s in TUNE_SEEDS]
            res = list(pool.map(stream_job, jobs))
            means = []
            for c in cands:
                acc = [r["rows"][0]["accuracy"] for r in res if r["hp"] == c]
                means.append(float(np.mean(acc)))
                tuning.append(dict(d=d, method=name, hp=c, k=k, mean_accuracy_batch2=means[-1]))
            hp[d][name] = cands[int(np.argmax(means))]
    frozen = {str(d): dict(head_lambda=lam[d], methods=hp[d]) for d in DIMS}
    (a.out / "hyperparameters.json").write_text(json.dumps(frozen, indent=1))
    (a.out / "tuning.json").write_text(json.dumps(tuning, indent=1))

    # Phase 2: E2 test runs.
    jobs = []
    for d in DIMS:
        for k in BUDGETS:
            names = ZERO_LABEL + (LABELLED if k > 0 else ())
            for name in names:
                for s in test_seeds:
                    jobs.append((d, lam[d], name, hp[d][name], k, s, list(range(2, 11))))
    runs = list(pool.map(stream_job, jobs, chunksize=4))
    (a.out / "e2_runs.json").write_text(json.dumps(runs))

    # Summary: mean over batches 3-10 per run, then mean and 95% bootstrap CI over seeds.
    rng = np.random.default_rng(SEED_BASE)
    summary = []
    for d in DIMS:
        for k in BUDGETS:
            for name in ZERO_LABEL + LABELLED:
                rs = [r for r in runs if r["d"] == d and r["k"] == k and r["method"] == name]
                if not rs:
                    continue
                acc = np.array([np.mean([x["accuracy"] for x in r["rows"] if x["batch"] >= 3]) for r in rs])
                f1 = np.array([np.mean([x["macro_f1"] for x in r["rows"] if x["batch"] >= 3]) for r in rs])
                per_batch = {b: float(np.mean([x["accuracy"] for r in rs for x in r["rows"] if x["batch"] == b]))
                             for b in range(2, 11)}
                boot = [rng.choice(acc, len(acc)).mean() for _ in range(2000)]
                summary.append(dict(d=d, k=k, method=name, accuracy=float(acc.mean()),
                                    ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                    macro_f1=float(f1.mean()), per_batch_accuracy=per_batch,
                                    state_bytes=max(r["state_bytes"] for r in rs),
                                    payload_bytes=rs[0]["payload_bytes"],
                                    head_bytes=head_bytes(d, CLASSES), hp=rs[0]["hp"]))
    (a.out / "e2_summary.json").write_text(json.dumps(summary, indent=1))
    with open(a.out / "e2_summary.csv", "w") as f:
        f.write("d,k,method,accuracy_b3_10,ci_low,ci_high,macro_f1,state_bytes,payload_bytes\n")
        for s in summary:
            f.write(f"{s['d']},{s['k']},{s['method']},{s['accuracy']:.4f},{s['ci95'][0]:.4f},"
                    f"{s['ci95'][1]:.4f},{s['macro_f1']:.4f},{s['state_bytes']},{s['payload_bytes']}\n")

    # E1: Settings 1 and 2, ridge and logistic heads, with/without per-batch statistics.
    e1 = []
    for d in DIMS:
        X, y, b = data(d)
        for head in ("ridge", "logistic"):
            for pbs in (False, True):
                s1 = [setting_scores(X, y, b, lam[d], 1, t, head, pbs) for t in range(2, 11)]
                s2 = [setting_scores(X, y, b, lam[d], t - 1, t, head, pbs) for t in range(2, 11)]
                e1.append(dict(d=d, head=head, per_batch_standardisation=pbs,
                               setting1=s1, setting2=s2,
                               setting1_mean=float(np.mean(s1)), setting2_mean=float(np.mean(s2))))
    (a.out / "e1.json").write_text(json.dumps(e1, indent=1))

    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    info = dict(commit=commit, dirty_worktree=dirty, python=sys.version, platform=platform.platform(),
                numpy=np.__version__, seconds=round(time.time() - t0, 1),
                test_seeds=test_seeds, tune_seeds=TUNE_SEEDS, budgets=BUDGETS, dims=DIMS,
                data_sha256=json.loads((ROOT / "data" / "datasets.json").read_text())["D1b"]["sha256"],
                files={p.name: sha(p) for p in sorted(a.out.glob("*.json")) if p.name != "run_info.json"})
    (a.out / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(frozen, indent=1))
    print(f"done in {info['seconds']} s")


if __name__ == "__main__":
    main()
