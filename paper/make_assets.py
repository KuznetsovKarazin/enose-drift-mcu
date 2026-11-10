"""Tables, figures and paired tests for the manuscript.

Reads only frozen results under results/ (E1-E5); refits nothing.
Usage: python paper/make_assets.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
R = ROOT / "results"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
NAMES = {"M0": "Static head", "M1": "EWMA standardisation", "M2": "Adaptive centroids", "M3": "Component correction",
         "M4": "Per-feature affine", "M5": "NLMS on head", "M6": "RLS on head", "M7": "Block OLS on head",
         "M8": "Direct standardisation", "M9": "Per-channel DS", "M1+M5": "EWMA + NLMS", "M1+M7": "EWMA + block OLS"}


def fmt_int(v):
    return f"{int(v):,}".replace(",", "\\,")


def load_e2():
    rows = list(csv.DictReader(open(R / "e1e2_20261109/e2_summary.csv")))
    return {(int(r["d"]), int(r["k"]), r["method"]): r for r in rows}


def e2_batch_means():
    runs = json.loads((R / "e1e2_20261109/e2_runs.json").read_text())
    acc = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for b in r["rows"]:
            if b["batch"] >= 3:
                acc[(r["d"], r["k"], r["method"])][b["batch"]].append(b["accuracy"])
    return {key: np.array([np.mean(v[b]) for b in sorted(v)]) for key, v in acc.items()}


def table_e2(e2):
    meth = ["M0", "M1", "M5", "M6", "M7", "M1+M5", "M1+M7"]
    lines = [r"\begin{table*}[!t]", r"\caption{Drift Compensation on the Gas Sensor Array Drift Data: Accuracy (Batches 3--10, 20 Seeds) and Persistent State}",
             r"\label{tab:e2}\centering\footnotesize", r"\begin{tabular}{l rrrr r @{\hspace{14pt}} rrrr r}\toprule",
             r" & \multicolumn{5}{c}{$d=16$} & \multicolumn{5}{c}{$d=128$}\\\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
             r"Method & $k=0$ & 1 & 5 & 20 & State (B) & $k=0$ & 1 & 5 & 20 & State (B)\\\midrule"]
    for m in meth:
        cells = [NAMES[m]]
        for d in (16, 128):
            for k in (0, 1, 5, 20):
                r = e2.get((d, k, m))
                cells.append(f"{float(r['accuracy_b3_10']):.3f}" if r else "--")
            cells.append(fmt_int(e2[(d, 20, m)]["state_bytes"]))
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table*}"]
    (P / "tables/e2.tex").write_text("\n".join(lines) + "\n")


SHORT = {"M0": "static", "M1": "EWMA", "M4": "affine", "M9": "chan. DS", "M8": "DS", "M5": "NLMS", "M6": "RLS", "M7": "block OLS"}


def load_e4():
    return {(r["d"], r["k"], r["method"]): r for r in json.loads((R / "e4_20261109/e4_transfer_summary.json").read_text())}


def table_e4(e4):
    meth = ["M0", "M1", "M4", "M9", "M8", "M5", "M6", "M7"]
    lines = [r"\begin{table}[t]", r"\caption{Calibration Transfer Between Twin Boards: Mean (Worst) Accuracy Over 19 Board Pairs, $d=16$}",
             r"\label{tab:e4}\centering\scriptsize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{l rrrr r}\toprule",
             r"Method & $k=0$ & 4 & 8 & 16 & State (B)\\\midrule"]
    for m in meth:
        cells = [f"{m} {SHORT.get(m, NAMES[m])}"]
        for k in (0, 4, 8, 16):
            r = e4.get((16, k, m))
            cells.append(f"{r['accuracy']:.3f} ({r['worst_pair']:.2f})" if r else "--")
        cells.append(fmt_int(e4[(16, 16, m)]["state_bytes"]))
        lines.append(" & ".join(cells) + r"\\")
    r64 = e4[(64, 16, "M8")]
    lines += [r"\midrule", f"M8 DS, $d=64$ & -- & {e4[(64, 4, 'M8')]['accuracy']:.3f} ({e4[(64, 4, 'M8')]['worst_pair']:.2f}) & "
              f"{e4[(64, 8, 'M8')]['accuracy']:.3f} ({e4[(64, 8, 'M8')]['worst_pair']:.2f}) & {r64['accuracy']:.3f} ({r64['worst_pair']:.2f}) & {fmt_int(r64['state_bytes'])}\\\\",
              r"\bottomrule\end{tabular}", r"\end{table}"]
    (P / "tables/e4.tex").write_text("\n".join(lines) + "\n")


def table_e3():
    s = {(r["d"], r["k"], r["method"]): r for r in json.loads((R / "e3_20261109/e3_summary.json").read_text())}
    meth = ["M0", "M1", "M5", "M6", "M7"]
    lines = [r"\begin{table}[t]", r"\caption{Concentration Regression Under Drift: RMSE in $\log_{10}$ ppmv (Batches 3--10, Six Gases, 20 Seeds)}",
             r"\label{tab:e3}\centering\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{l rrr rrr}\toprule",
             r" & \multicolumn{3}{c}{$d=16$} & \multicolumn{3}{c}{$d=128$}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & $k=1$ & 5 & 20 & $k=1$ & 5 & 20\\\midrule"]
    for m in meth:
        cells = [NAMES[m]]
        for d in (16, 128):
            for k in (1, 5, 20):
                r = s.get((d, k, m))
                cells.append(f"{r['rmse_mean_over_gases']:.3f}" if r else "--")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (P / "tables/e3.tex").write_text("\n".join(lines) + "\n")
    return s


MCU_ROWS = [("static16", "Static head, $d=16$", "classify"), ("static128", "Static head, $d=128$", "classify"),
            ("m1_16", "EWMA, $d=16$", "classify"), ("m1_128", "EWMA, $d=128$", "classify"),
            ("nlms16", "NLMS, $d=16$", "classify + update"), ("nlms128", "NLMS, $d=128$", "classify + update"),
            ("rls16", "RLS, $d=16$", "classify + update"), ("rls128", "RLS, $d=128$", "classify + update"),
            ("ols16", "Block OLS, $d=16$", "refit, $k=20$"), ("ds16_fit", "DS, $d=16$", "fit, 16 pairs"),
            ("ds16_apply", "DS, $d=16$", "map + classify"), ("extract", "Feature extractor", "one sample")]


def load_e5():
    rows = list(csv.DictReader(open(R / "e5_20261109/e5_timing.csv")))
    return {(r["board"], r["kernel"]): r for r in rows}


def table_mcu(t):
    lines = [r"\begin{table*}[!t]", r"\caption{Measured Time per Call on the Boards (Median of 20 Timings in Two Sessions, $\mu$s) and Persistent State}",
             r"\label{tab:mcu}\centering\footnotesize", r"\begin{tabular}{l l r r rrr}\toprule",
             r"Kernel & One call & State (B) & Host accuracy & Mega2560 & ESP32-C3 & ESP32-S3\\\midrule"]
    for key, name, unit in MCU_ROWS:
        s3 = t[("ESP32-S3", key)]
        acc = s3["host_accuracy"]
        sb = int(float(s3["state_bytes"])) if key != "extract" else None
        cells = [name, unit, fmt_int(sb) if sb is not None else "426/428", f"{float(acc):.3f}" if acc else "--"]
        for b in ("Mega2560", "ESP32-C3", "ESP32-S3"):
            r = t[(b, key)]
            if r["status"] != "measured":
                cells.append("does not fit")
            else:
                v = float(r["us_median"])
                cells.append(fmt_int(round(v)) if v >= 1000 else f"{v:.1f}" if v >= 10 else f"{v:.2f}")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table*}"]
    (P / "tables/mcu.tex").write_text("\n".join(lines) + "\n")


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def tests(e2b, e4):
    fam = []
    def add(label, a, b):
        res = wilcoxon(a, b, alternative="two-sided", method="exact" if len(a) <= 25 else "auto")
        fam.append(dict(test=label, n=len(a), median_diff=float(np.median(np.asarray(a) - np.asarray(b))),
                        wins=int(np.sum(np.asarray(a) > np.asarray(b))), p=float(res.pvalue)))
    add("E2 d16 k0 M1 vs M0", e2b[(16, 0, "M1")], e2b[(16, 0, "M0")])
    add("E2 d128 k0 M1 vs M0", e2b[(128, 0, "M1")], e2b[(128, 0, "M0")])
    add("E2 d128 k20 M5 vs M0", e2b[(128, 20, "M5")], e2b[(128, 20, "M0")])
    add("E2 d16 k20 M7 vs M6", e2b[(16, 20, "M7")], e2b[(16, 20, "M6")])
    add("E2 d128 k20 M6 vs M5", e2b[(128, 20, "M6")], e2b[(128, 20, "M5")])
    pairs = sorted(e4[(16, 16, "M8")]["per_pair"])
    pp = lambda d, k, m: np.array([e4[(d, k, m)]["per_pair"][p] for p in pairs])
    add("E4 d16 k0 M1 vs M0", pp(16, 0, "M1"), pp(16, 0, "M0"))
    add("E4 d16 k16 M8 vs M1", pp(16, 16, "M8"), pp(16, 0, "M1"))
    add("E4 k16 M8 d16 vs d64", pp(16, 16, "M8"), pp(64, 16, "M8"))
    add("E4 d16 k16 M8 vs M6", pp(16, 16, "M8"), pp(16, 16, "M6"))
    adj = holm(np.array([f["p"] for f in fam]))
    for f, a in zip(fam, adj):
        f["p_holm"] = float(a)
    (P / "stats.json").write_text(json.dumps(fam, indent=1))
    return fam


def fig_frontier(e2, e4):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    col = {"M0": "#777777", "M1": "#1f77b4", "M4": "#8c564b", "M5": "#2ca02c", "M6": "#d62728", "M7": "#9467bd",
           "M8": "#ff7f0e", "M9": "#17becf", "M1+M5": "#bcbd22", "M1+M7": "#e377c2"}
    def panel(ax, pts, title, small, big, label_keys):
        for (m, d, acc, sb) in pts:
            ax.scatter(max(sb, 2), acc, color=col[m], marker="o" if d == small else "s", s=24, zorder=3,
                       edgecolor="k", linewidth=0.3)
            if (m, d) in label_keys:
                ax.annotate(f"{m}", (max(sb, 2), acc), xytext=(4, -3), textcoords="offset points", fontsize=6.3)
        ax.axvline(8192, color="k", lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_xlabel("Persistent state (bytes, log scale; 2 B = none)")
        ax.set_title(title, fontsize=8); ax.grid(alpha=0.25, lw=0.4)
        ax.scatter([], [], marker="o", color="w", edgecolor="k", label=f"$d={small}$")
        ax.scatter([], [], marker="s", color="w", edgecolor="k", label=f"$d={big}$")
    pts = [(m, d, float(e2[(d, 20, m)]["accuracy_b3_10"]), int(e2[(d, 20, m)]["state_bytes"]))
           for d in (16, 128) for m in ("M0", "M1", "M5", "M6", "M7", "M1+M5", "M1+M7")]
    panel(axes[0], pts, "(a) Drift over 36 months, $k=20$ references per batch", 16, 128,
          {("M0", 16), ("M0", 128), ("M1", 128), ("M5", 128), ("M6", 128), ("M7", 16), ("M5", 16), ("M6", 16), ("M7", 128)})
    axes[0].set_ylabel("Accuracy"); axes[0].set_xlim(1.5, 2.5e5)
    pts = [(m, d, e4[(d, 16, m)]["accuracy"], e4[(d, 16, m)]["state_bytes"])
           for d in (16, 64) for m in ("M0", "M1", "M4", "M5", "M6", "M7", "M8", "M9")]
    panel(axes[1], pts, "(b) Board-to-board transfer, $k=16$ pairs", 16, 64,
          {("M0", 16), ("M5", 16), ("M8", 16), ("M8", 64), ("M6", 16), ("M6", 64), ("M1", 64)})
    axes[1].set_xlim(1.5, 6e4)
    for ax in axes:
        ax.text(8192 * 1.1, ax.get_ylim()[0] + 0.01, "8 kB SRAM\n(Mega2560)", fontsize=6, va="bottom")
    handles = [plt.Line2D([], [], marker="o", ls="", color=col[m], markeredgecolor="k", markeredgewidth=0.3,
                          label=f"{m} {NAMES[m]}") for m in col]
    handles += [plt.Line2D([], [], marker="o", ls="", color="w", markeredgecolor="k", label="compact features"),
                plt.Line2D([], [], marker="s", ls="", color="w", markeredgecolor="k", label="full features")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, -0.33))
    fig.savefig(P / "figures/frontier.pdf")
    plt.close(fig)


def fig_timing(t):
    keys = [k for k, _, _ in MCU_ROWS]
    labels = ["static 16", "static 128", "EWMA 16", "EWMA 128", "NLMS 16", "NLMS 128", "RLS 16", "RLS 128",
              "OLS refit", "DS fit", "DS apply", "extract"]
    fig, ax = plt.subplots(figsize=(7.16, 2.0))
    x = np.arange(len(keys)); w = 0.27
    for i, (b, c) in enumerate((("Mega2560", "#555555"), ("ESP32-C3", "#1f77b4"), ("ESP32-S3", "#ff7f0e"))):
        v = [float(t[(b, k)]["us_median"]) if t[(b, k)]["status"] == "measured" else np.nan for k in keys]
        ax.bar(x + (i - 1) * w, v, w, label=b, color=c)
        if b == "Mega2560":
            for xi, vi in zip(x, v):
                if np.isnan(vi):
                    ax.text(xi - w, 3, "n/f", ha="center", fontsize=6, rotation=90)
    ax.set_yscale("log"); ax.set_ylabel(r"Time per call ($\mu$s)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25, lw=0.4); ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper left")
    ax.set_ylim(1, 1e6)
    fig.savefig(P / "figures/timing.pdf")
    plt.close(fig)


def fig_graphical(e2, e4, t):
    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    pts = [("Static, 0 B", 0.498, 2, "#777777"), ("EWMA, 1 kB, 0 labels", 0.614, 1028, "#1f77b4"),
           ("NLMS, 4 B", 0.772, 4, "#2ca02c"), ("RLS d=16, 1.2 kB", 0.876, 1160, "#d62728"),
           ("RLS d=128, 66.6 kB", 0.923, 66568, "#d62728"), ("DS board transfer, 1.1 kB", 0.955, 1088, "#ff7f0e")]
    offs = {"RLS d=128, 66.6 kB": (-40, -10), "DS board transfer, 1.1 kB": (-100, -2), "RLS d=16, 1.2 kB": (4, -10)}
    for lab, a, s, c in pts:
        ax.scatter(s, a, color=c, s=26, edgecolor="k", linewidth=0.3, zorder=3)
        ax.annotate(lab, (s, a), xytext=offs.get(lab, (4, -3)), textcoords="offset points", fontsize=6.5)
    ax.axvspan(8192, 3e5, color="#f2c6c6", alpha=0.5, lw=0)
    ax.text(9500, 0.47, "does not fit\nMega2560", fontsize=6.3)
    ax.set_xscale("log"); ax.set_xlim(1.5, 3e5); ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Persistent state, bytes"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per byte of on-device state", fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(P / "figures/gagraphic_inline.pdf")
    fig.savefig(P / "figures/gagraphic.png", dpi=300)
    plt.close(fig)


def main():
    (P / "tables").mkdir(exist_ok=True); (P / "figures").mkdir(exist_ok=True)
    e2 = load_e2(); e4 = load_e4(); t = load_e5()
    table_e2(e2); table_e4(e4); table_e3(); table_mcu(t)
    fam = tests(e2_batch_means(), e4)
    fig_frontier(e2, e4); fig_timing(t); fig_graphical(e2, e4, t)
    for f in fam:
        print(f"{f['test']:28s} n={f['n']:2d} wins={f['wins']:2d} med={f['median_diff']:+.3f} p={f['p']:.4f} holm={f['p_holm']:.4f}")


if __name__ == "__main__":
    main()
