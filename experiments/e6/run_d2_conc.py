"""E6 (revision 2, post hoc): calibration transfer when the transfer pairs (or labelled
slave references) are concentrated on one or two gases or on the four lowest
concentration levels, instead of 16 conditions drawn from all 40.

Usage: python experiments/e6/run_d2_conc.py OUT_DIR [--seeds 20] [--jobs 4]
Frozen E4 hyperparameters and seeds; nothing re-tuned.
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
import run as E4  # noqa: E402

HP4 = json.loads((ROOT / "results/e4_20261109/hyperparameters.json").read_text())
SCEN = ("all_gases", "two_gases", "one_gas", "low_levels")


def restricted_job(args):
    """E4.transfer_job with the 40 gas-level conditions restricted before the seeded draw."""
    scen, d, name, k, seed, (i, j) = args
    h = HP4[str(d)]
    hp = h["methods"].get(name, {})
    Dd = E4.load(); X = E4.feats(d); y = Dd["y"]
    board, lev, rep, gas = Dd["board"], Dd["level"], Dd["repetition"], Dd["gas"]
    src = E4.make_source(X[board == i], y[board == i], h["head_lambda"], classes=E4.CLASSES)
    Z = src.pre(X)
    conds = sorted({(g, l) for g, l in zip(gas[board == j], lev[board == j])})
    srng = np.random.default_rng([seed, 7])
    gases = sorted({g for g, _ in conds})
    if scen == "two_gases":
        keep = set(srng.choice(gases, 2, replace=False).tolist())
        conds = [c for c in conds if c[0] in keep]
    elif scen == "one_gas":
        keep = {srng.choice(gases)}
        conds = [c for c in conds if c[0] in keep]
    elif scen == "low_levels":
        levels = sorted({l for _, l in conds})[:4]
        conds = [c for c in conds if c[1] in levels]
    rng = np.random.default_rng(seed)
    chosen = [conds[t] for t in rng.permutation(len(conds))[:k]]
    m = E4.make_method(name, src, hp, d)
    used = []
    for g, l in chosen:
        sj = np.flatnonzero((board == j) & (gas == g) & (lev == l) & (rep == 1))
        si = np.flatnonzero((board == i) & (gas == g) & (lev == l) & (rep == 1))
        if len(sj) == 0 or len(si) == 0:
            continue
        used.append(sj[0])
        if name in E4.TRANSFER:
            m.pair(Z[sj[0]], Z[si[0]])
        else:
            m.reference(Z[sj[0]], int(y[sj[0]]))
    if name in E4.TRANSFER:
        m.fit()
    m.end_reference_session()
    ev = np.setdiff1d(np.flatnonzero(board == j), used)
    ev = ev[rng.permutation(len(ev))]
    pred = np.array([m.observe(Z[t]) for t in ev]) if m.sequential else m.predict_many(Z[ev])
    return dict(scenario=scen, d=d, method=name, k=k, seed=seed, pair=[i, j], n_pairs_used=len(used),
                n_eval=int(len(ev)), eval_idx_hash=int(np.sum(ev)), accuracy=float(np.mean(pred == y[ev])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = [E4.SEED_BASE + s for s in range(a.seeds)]
    pairs = [(i, j) for i in range(1, 6) for j in range(1, 6) if i != j and (i, j) != (1, 2)]
    jobs = [(sc, 16, n, 16, s, p) for sc in SCEN for n in ("M0", "M1", "M4", "M9", "M8", "M6", "M7") for s in seeds for p in pairs]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(restricted_job, jobs, chunksize=16))
    (a.out / "d2_concentrated_runs.json").write_text(json.dumps(res))
    print(f"done in {time.time() - t0:.0f} s, {len(res)} runs")


if __name__ == "__main__":
    main()
