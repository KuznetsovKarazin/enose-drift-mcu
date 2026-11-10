"""Summaries of the revision analyses (E6) and of E2/E4 statistics.

Usage: python experiments/e6/analyse.py results/e6_20261110
Writes summary.json in that folder and prints the key tables.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
B_REPS = 10000


def mean_b3(rows, key="accuracy"):
    return float(np.mean([r[key] for r in rows if r["batch"] >= 3]))


def boot_ci(x, rng, reps=B_REPS):
    x = np.asarray(x, float)
    m = np.array([np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(reps)])
    return [float(np.mean(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def d1(out, rng):
    runs = json.loads(gzip.open(out / "d1_runs.json.gz", "rt").read())
    g = defaultdict(list)
    for r in runs:
        g[(r["kind"], r["method"], r["d"], r["k"], r["order"], r["refs"])].append(r)
    res = {}
    # float32 replica versus float64
    f32 = []
    for (kind, m, d, k, o, rf), rs in g.items():
        if kind != "f32":
            continue
        r64 = {r["seed"]: r for r in g[("f64", m, d, k, o, rf)]}
        agree, n = 0, 0
        for r in rs:
            for a, b in zip(r["rows"], r64[r["seed"]]["rows"]):
                agree += int(np.sum(np.array(a["pred"]) == np.array(b["pred"]))); n += len(a["pred"])
        f32.append(dict(method=m, d=d, k=k, acc_f32=float(np.mean([mean_b3(r["rows"]) for r in rs])),
                        acc_f64=float(np.mean([mean_b3(r["rows"]) for r in r64.values()])),
                        prediction_agreement=agree / n, n_predictions=n))
    res["f32_replica"] = f32
    # scenarios
    sc = []
    for (kind, m, d, k, o, rf), rs in g.items():
        if kind == "f64" and m in ("M0", "M1", "M5", "M6", "M7", "M1+M5") and k in (0, 5, 20):
            acc = [mean_b3(r["rows"]) for r in rs]
            f1 = [mean_b3(r["rows"], "macro_f1") for r in rs]
            sc.append(dict(method=m, d=d, k=k, order=o, refs=rf, accuracy=boot_ci(acc, rng, 2000),
                           macro_f1=float(np.mean(f1))))
    res["scenarios"] = sc
    var = []
    for (kind, m, d, k, o, rf), rs in g.items():
        if m in ("M7b", "M6b", "M6@0.95", "M6@0.99") or (m in ("M7", "M6") and kind == "f64" and o == "shuffled" and rf == "uniform"):
            var.append(dict(method=m, d=d, k=k, accuracy=float(np.mean([mean_b3(r["rows"]) for r in rs]))))
    res["variants"] = var
    e3 = json.loads((out / "e3_rls_runs.json").read_text())
    h = defaultdict(list)
    for r in e3:
        if r["rmse"] is not None:
            h[(r["method"], r["d"], r["k"], r["gas"])].append(r["rmse"])
    agg = defaultdict(list)
    for (m, d, k, gsel), v in h.items():
        agg[(m, d, k)].append(float(np.mean(v)))
    res["e3_rls"] = [dict(method=m, d=d, k=k, rmse_mean_over_gases=float(np.mean(v)), worst_gas=float(np.max(v)))
                     for (m, d, k), v in sorted(agg.items())]
    return res


def e2_stats(rng):
    """Per-batch accuracy, macro-F1 and paired differences, conditional on this dataset:
    intervals reflect the 20 random reference draws and orders, not new data."""
    runs = json.loads((R / "e1e2_20261109/e2_runs.json").read_text())
    by = defaultdict(dict)
    for r in runs:
        by[(r["d"], r["k"], r["method"])][r["seed"]] = r
    def per_batch(key):
        rs = by[key]
        bs = sorted({x["batch"] for r in rs.values() for x in r["rows"] if x["batch"] >= 3})
        return bs, np.array([[next(x["accuracy"] for x in r["rows"] if x["batch"] == b) for b in bs] for r in rs.values()])
    out = {"per_batch": [], "differences": [], "macro_f1": []}
    for key in by:
        bs, A = per_batch(key)
        F = np.array([[x["macro_f1"] for x in r["rows"] if x["batch"] >= 3] for r in by[key].values()])
        out["per_batch"].append(dict(d=key[0], k=key[1], method=key[2], batches=bs, accuracy=A.mean(0).round(4).tolist()))
        out["macro_f1"].append(dict(d=key[0], k=key[1], method=key[2], macro_f1=float(F.mean())))
    comps = [((16, 0, "M1"), (16, 0, "M0")), ((128, 0, "M1"), (128, 0, "M0")), ((128, 20, "M5"), (128, 20, "M0")),
             ((128, 20, "M6"), (128, 20, "M5")), ((16, 20, "M7"), (16, 20, "M6")), ((16, 20, "M6"), (16, 20, "M5"))]
    for a, b in comps:
        _, A = per_batch(a); _, Bm = per_batch(b)
        seeds = sorted(set(by[a]) & set(by[b]))
        if a[1] == 0 or b[1] == 0:
            A = A.mean(0, keepdims=True).repeat(len(seeds), 0) if A.shape[0] != len(seeds) else A
            Bm = Bm.mean(0, keepdims=True).repeat(len(seeds), 0) if Bm.shape[0] != len(seeds) else Bm
        diff_batch = (A - Bm).mean(0)
        diff_seed = (A - Bm).mean(1)
        out["differences"].append(dict(a=list(a), b=list(b), mean_diff=float(diff_batch.mean()),
                                       batches_better=int(np.sum(diff_batch > 0)), n_batches=len(diff_batch),
                                       min_batch_diff=float(diff_batch.min()), max_batch_diff=float(diff_batch.max()),
                                       ci_over_seeds=boot_ci(diff_seed, rng)[1:]))
    return out


def e4_stats(rng):
    """Board-level cluster bootstrap: resample the five boards with replacement and
    average the differences over all ordered pairs of distinct drawn boards (the
    tuning pair 1->2 excluded). Comparisons are made on identical test records."""
    S = {(r["d"], r["k"], r["method"]): r["per_pair"] for r in json.loads((R / "e4_20261109/e4_transfer_summary.json").read_text())}
    comps = [((16, 0, "M1"), (16, 0, "M0")), ((16, 16, "M8"), (16, 16, "M1")), ((16, 16, "M8"), (16, 16, "M6")),
             ((16, 16, "M8"), (64, 16, "M8")), ((16, 16, "M1"), (16, 0, "M1")), ((16, 8, "M8"), (16, 8, "M1"))]
    out = []
    for a, b in comps:
        pa, pb = S[a], S[b]
        diff = {tuple(map(int, k.split("->"))): pa[k] - pb[k] for k in pa}
        def stat(boards):
            v = [diff[(i, j)] for i in boards for j in boards if i != j and (i, j) in diff]
            return np.mean(v) if v else np.nan
        bs = np.array([stat(rng.integers(1, 6, 5)) for _ in range(B_REPS)])
        bs = bs[np.isfinite(bs)]
        out.append(dict(a=list(a), b=list(b), mean_diff=float(np.mean(list(diff.values()))),
                        pairs_better=int(np.sum(np.array(list(diff.values())) > 0)), n_pairs=len(diff),
                        ci_board_bootstrap=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]))
    return out


def d2(out):
    res = {}
    p = out / "d2_ds_f32_runs.json"
    if p.exists():
        ds = json.loads(p.read_text())
        g = defaultdict(list)
        for r in ds:
            g[(r["d"], r["k"])].append(r)
        res["ds_f32_fused"] = [dict(d=d, k=k, n=sum(r["n"] for r in rs),
                                    acc64=float(np.mean([r["acc64"] for r in rs])),
                                    acc_fused32=float(np.mean([r["acc_fused32"] for r in rs])),
                                    agree_32_64=float(np.average([r["agree_32_64"] for r in rs], weights=[r["n"] for r in rs])),
                                    agree_fused_64=float(np.average([r["agree_fused_64"] for r in rs], weights=[r["n"] for r in rs])),
                                    agree_fused32_64=float(np.average([r["agree_fused32_64"] for r in rs], weights=[r["n"] for r in rs])),
                                    fit_failures=int(sum(r["ok"] != 0 for r in rs)))
                               for (d, k), rs in sorted(g.items())]
    p = out / "d2_sensitivity_runs.json"
    if p.exists():
        sens = json.loads(p.read_text())
        g = defaultdict(lambda: defaultdict(list))
        for r in sens:
            g[(r["exclusion"], r["method"], r["k"])][tuple(r["pair"])].append(r["accuracy"])
        res["sensitivity"] = []
        for (ex, m, k), pp in sorted(g.items()):
            per = {kk: float(np.mean(v)) for kk, v in pp.items()}
            # Without board-5 methane a board-5 master head has no methane class, so the
            # like-for-like comparison uses the 15 pairs whose master is not board 5.
            m5 = [v for kk, v in per.items() if kk[0] != 5]
            res["sensitivity"].append(dict(exclusion=ex, method=m, k=k, accuracy=float(np.mean(list(per.values()))),
                                           accuracy_master_not5=float(np.mean(m5)), n_pairs_master_not5=len(m5),
                                           worst_pair=float(min(per.values()))))
    return res


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "e6_20261110"
    rng = np.random.default_rng(20270131 + 600)
    summary = dict(note="Post hoc revision analyses; frozen hyperparameters; see PROTOCOL_CHANGES.md")
    if (out / "d1_runs.json.gz").exists():
        summary["d1"] = d1(out, rng)
    summary["e2_stats"] = e2_stats(rng)
    summary["e4_stats"] = e4_stats(rng)
    summary["d2"] = d2(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for x in summary["e2_stats"]["differences"] + summary["e4_stats"]:
        print(x)
    if "d1" in summary:
        for x in summary["d1"]["f32_replica"]:
            print("f32", x)
        for x in sorted(summary["d1"]["scenarios"], key=lambda r: (r["d"], r["method"], r["k"], r["order"], r["refs"])):
            print("sc", x)
        for x in summary["d1"]["variants"] + summary["d1"]["e3_rls"]:
            print("var", x)
    for k, v in summary["d2"].items():
        for x in v:
            print(k, x)


if __name__ == "__main__":
    main()
