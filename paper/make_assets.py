"""Tables and figures for the manuscript (revision 1).

Reads only archived results under results/ (E1-E5 frozen, E6 post hoc); refits nothing.
Statistics are computed by experiments/e6/analyse.py (results/e6_20261110/summary.json).
Usage: python paper/make_assets.py
"""
from __future__ import annotations

import csv
import os
import json
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible PDF metadata
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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


MUTABLE = {"M5", "M6", "M7", "M1+M5", "M1+M7"}   # methods that modify the head, which must then live in RAM
E6 = R / "e6_20261110"


def head_bytes(d, C):
    return 4 * C * (d + 1)


def persistent_ram(m, d, C, state):
    """Bytes kept in RAM between samples: adaptation state plus the head if the method modifies it.
    DS (M8) is counted fused into the head (W' = F W), which replaces the map."""
    if m == "M8":
        return head_bytes(d, C)
    return state + (head_bytes(d, C) if m in MUTABLE else 0)


def load_summary():
    return json.loads((E6 / "summary.json").read_text())


def table_e2(e2, S):
    meth = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M1+M5", "M1+M7"]
    f1 = {(r["d"], r["k"], r["method"]): r["macro_f1"] for r in S["e2_stats"]["macro_f1"]}
    lines = [r"\begin{table*}[!t]", r"\caption{Drift Compensation on the Gas Sensor Array Drift Data: Accuracy (Batches 3--10, 20 Seeds), Macro-F1 at $k=20$ and Persistent RAM}",
             r"\label{tab:e2}\centering\footnotesize", r"\begin{tabular}{l rrrr r r @{\hspace{14pt}} rrrr r r}\toprule",
             r" & \multicolumn{6}{c}{$d=16$} & \multicolumn{6}{c}{$d=128$}\\\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
             r"Method & $k=0$ & 1 & 5 & 20 & F1$_{20}$ & RAM (B) & $k=0$ & 1 & 5 & 20 & F1$_{20}$ & RAM (B)\\\midrule"]
    for m in meth:
        cells = [f"{m} {NAMES[m]}"]
        for d in (16, 128):
            for k in (0, 1, 5, 20):
                r = e2.get((d, k, m))
                cells.append(f"{float(r['accuracy_b3_10']):.3f}" if r else "--")
            cells.append(f"{f1[(d, 20, m)]:.3f}")
            cells.append(fmt_int(persistent_ram(m, d, 6, int(e2[(d, 20, m)]["state_bytes"]))))
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\multicolumn{13}{p{0.95\textwidth}}{\scriptsize $k$: labelled references per batch (one calibration session). F1$_{20}$: macro-F1 at $k=20$. RAM: bytes kept between samples in float32, i.e., the adaptation state plus the head (408~B at $d=16$, 3\,096~B at $d=128$) for methods that modify it (M5--M7 and combinations); other heads can stay in flash. Unsupervised methods (M1, M2) do not use the references. M3 and M4 need at least one reference.}",
              r"\end{tabular}", r"\end{table*}"]
    (P / "tables/e2.tex").write_text("\n".join(lines) + "\n")


def table_stats(S):
    lines = [r"\begin{table}[t]", r"\caption{Paired Differences in Accuracy}", r"\label{tab:stats}\centering\scriptsize\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{l r r l}\toprule", r"Comparison & Mean & Better & Spread\\\midrule",
             r"\multicolumn{4}{l}{\emph{E2, drift stream: units are the 8 test batches; spread = range over batches}}\\"]
    for x in S["e2_stats"]["differences"]:
        a, b = x["a"], x["b"]
        lines.append(f"{a[2]} vs {b[2]}, $d={a[0]}$, $k={a[1]}$ & {x['mean_diff']:+.3f} & {x['batches_better']}/{x['n_batches']} & "
                     f"[{x['min_batch_diff']:+.2f}, {x['max_batch_diff']:+.2f}]\\\\")
    lines.append(r"\multicolumn{4}{l}{\emph{E4, transfer: units are the 19 board pairs; spread = 95\% board-bootstrap interval}}\\")
    for x in S["e4_stats"]:
        a, b = x["a"], x["b"]
        if a[2] == b[2] and a[0] == b[0]:
            continue      # M1 at k=16 vs k=0 would compare different test sets
        name = f"{a[2]} vs {b[2]}, $k={a[1]}$" if a[0] == b[0] else f"{a[2]} $d={a[0]}$ vs $d={b[0]}$, $k={a[1]}$"
        lo, hi = x["ci_board_bootstrap"]
        lines.append(f"{name} & {x['mean_diff']:+.3f} & {x['pairs_better']}/{x['n_pairs']} & [{lo:+.3f}, {hi:+.3f}]\\\\")
    lines += [r"\bottomrule", r"\multicolumn{4}{p{0.95\columnwidth}}{\scriptsize E2: the range of the per-batch differences is a description, not a confidence interval; intervals over the 20 reference draws describe only this one stream and are given in the repository. E4: cluster bootstrap resampling the five boards (10\,000 replicates); with five clusters the intervals are coarse. Both methods of a row are evaluated on identical test records.}",
              r"\end{tabular}", r"\end{table}"]
    (P / "tables/stats.tex").write_text("\n".join(lines) + "\n")


def table_conc(S):
    rows = {(r["scenario"], r["method"]): r for r in S["d2"]["concentrated"]}
    meth = ["M0", "M1", "M4", "M9", "M8", "M6", "M7"]
    names = {"all_gases": "16 of 40 conditions (E4)", "low_levels": "4 lowest levels, all gases", "two_gases": "Two gases, all levels",
             "one_gas": "One gas (10 pairs)"}
    lines = [r"\begin{table}[t]", r"\caption{Transfer With Transfer Pairs From Restricted Conditions ($d=16$, $k=16$, Post Hoc)}",
             r"\label{tab:conc}\centering\scriptsize\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{l rrrrrrr}\toprule",
             "Pairs drawn from & " + " & ".join(meth) + r"\\\midrule"]
    for sc in ("all_gases", "low_levels", "two_gases", "one_gas"):
        lines.append(names[sc] + " & " + " & ".join(f"{rows[(sc, m)]['accuracy']:.3f}" for m in meth) + r"\\")
    lines += [r"\bottomrule", r"\multicolumn{8}{p{0.95\columnwidth}}{\scriptsize Mean accuracy over 19 board pairs and 20 seeds; the gases are drawn per seed. M0 and M1 do not use the pairs and change only because the evaluated records differ slightly.}",
              r"\end{tabular}", r"\end{table}"]
    (P / "tables/conc.tex").write_text("\n".join(lines) + "\n")


def table_scenarios(S):
    sc = {(r["method"], r["d"], r["k"], r["order"], r["refs"]): r for r in S["d1"]["scenarios"]}
    cols = [("shuffled", "uniform"), ("grouped", "uniform"), ("shuffled", "one_gas"), ("grouped", "one_gas")]
    lines = [r"\begin{table}[t]", r"\caption{Drift Stream Under Less Favourable Calibration ($k=20$, Accuracy, Batches 3--10, Post Hoc)}",
             r"\label{tab:scen}\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l r rrrr}\toprule",
             r" & & \multicolumn{2}{c}{References of all gases} & \multicolumn{2}{c}{References of one gas}\\\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Method & $d$ & shuffled & gas blocks & shuffled & gas blocks\\\midrule"]
    for d in (16, 128):
        for m in ("M1", "M5", "M6", "M7", "M1+M5"):
            cells = [f"{m} {SHORT.get(m, NAMES[m])}", str(d)]
            for o, rf in cols:
                key = (m, d, 20 if m != "M1" else 0, o, rf) if rf == "uniform" and m == "M1" else (m, d, 20, o, rf)
                r = sc.get(key)
                cells.append(f"{r['accuracy'][0]:.3f}" if r else "--")
            lines.append(" & ".join(cells) + r"\\")
        if d == 16:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\multicolumn{6}{p{0.95\columnwidth}}{\scriptsize First column: protocol of E2. Gas blocks: within each batch the recordings arrive grouped by gas in a random gas order. One gas: all $k$ references of a session come from one randomly chosen gas. 20 seeds; the 95\% intervals over seeds are at most $\pm0.036$ and are given in the repository.}",
              r"\end{tabular}", r"\end{table}"]
    (P / "tables/scen.tex").write_text("\n".join(lines) + "\n")


def ram_rows():
    """RAM of the firmware configurations, float32 (mcu/src/hil/hil_main.cpp follows the same layout)."""
    rows = []
    def add(ds, m, label, d, C, k, state, scratch, calib):
        persistent = state + (head_bytes(d, C) if m in MUTABLE else 0)
        rows.append(dict(data=ds, method=m, label=label, d=d, head=head_bytes(d, C) if m in MUTABLE or m.startswith("M8f") else 0,
                         state=state, scratch=scratch, calib=calib, persistent=persistent if not m.startswith("M8f") else head_bytes(d, C),
                         peak=(persistent if not m.startswith("M8f") else head_bytes(d, C)) + scratch + calib))
    for d in (16, 128):
        n, C, k = d + 1, 6, 20
        z = 4 * d
        fit = 4 * n * n + 4 * n * C + 4 * k * C    # Gram matrix, solution, one-hot targets
        add("D1", "M0", "Static head", d, C, 0, 0, z, 0)
        add("D1", "M1", "EWMA", d, C, 0, 8 * d + 4, z, 0)
        add("D1", "M5", "NLMS", d, C, 0, 4, z, 0)
        add("D1", "M6", "RLS", d, C, 0, 4 * n * n + 4, z + 4 * n, 0)
        add("D1", "M7", "Block OLS", d, C, k, k * (4 * d + 1), z, fit)
        add("D1", "M1+M5", "EWMA + NLMS", d, C, 0, 8 * d + 8, z, 0)
        add("D1", "M1+M7", "EWMA + OLS", d, C, k, 8 * d + 4 + k * (4 * d + 1), z, fit)
    for d in (16, 64):
        n, C, k = d + 1, 4, 16
        z = 4 * d
        add("D2", "M8", "DS, map", d, C, k, 4 * n * d, 2 * z, 8 * k * d + 4 * n * n)
        add("D2", "M8f", "DS, fused", d, C, k, 0, z, 8 * k * d + 4 * n * n + 4 * n * d)
    return rows


MEGA_FREE = 8192 - 800   # SRAM minus the static use of the HIL firmware outside its buffers (6\,036 B - 5\,236 B)


def table_ram():
    rows = ram_rows()
    lines = [r"\begin{table}[t]", r"\caption{RAM of the Firmware Configurations in Bytes (float32)}", r"\label{tab:ram}\centering\scriptsize\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{l r r r r r r r}\toprule", r"Configuration & $d$ & Head & State & Persist. & Scratch & Calib. & Peak\\\midrule"]
    last = None
    for r in rows:
        if last and r["data"] != last:
            lines.append(r"\midrule")
        last = r["data"]
        peak = fmt_int(r["peak"]) + ("" if r["peak"] <= MEGA_FREE else "$^*$")
        lines.append(f"{r['method'].replace('M8f', 'M8')} {r['label']} & {r['d']} & {fmt_int(r['head']) if r['head'] else '--'} & {fmt_int(r['state']) if r['state'] else '--'} & "
                     f"{fmt_int(r['persistent'])} & {fmt_int(r['scratch'])} & {fmt_int(r['calib']) if r['calib'] else '--'} & {peak}\\\\")
    lines += [r"\bottomrule", r"\multicolumn{8}{p{0.97\columnwidth}}{\scriptsize Head: classifier in RAM because the method modifies it (otherwise it stays in flash). Persist.: persistent RAM, head plus state, kept between samples. Scratch: feature vector of the current sample (and the RLS gain or the DS output). Calib.: temporary memory at the end of a session (Gram matrix, solution, targets, transfer pairs). D1: $C=6$, $k=20$; D2: $C=4$, $k=16$. $^*$Exceeds the 7\,392~B of SRAM left by the firmware on the Mega2560.}",
              r"\end{tabular}", r"\end{table}"]
    (P / "tables/ram.tex").write_text("\n".join(lines) + "\n")
    (P / "ram.json").write_text(json.dumps(rows, indent=1))
    return rows


SHORT = {"M0": "static", "M1": "EWMA", "M4": "affine", "M9": "chan. DS", "M8": "DS", "M5": "NLMS", "M6": "RLS", "M7": "block OLS"}


def load_e4():
    return {(r["d"], r["k"], r["method"]): r for r in json.loads((R / "e4_20261109/e4_transfer_summary.json").read_text())}


def table_e4(e4):
    meth = ["M0", "M1", "M4", "M9", "M8", "M5", "M6", "M7"]
    lines = [r"\begin{table}[t]", r"\caption{Calibration Transfer Between Twin Boards: Mean (Worst) Accuracy Over 19 Board Pairs, $d=16$}",
             r"\label{tab:e4}\centering\scriptsize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{l rrrr r}\toprule",
             r"Method & $k=0$ & 4 & 8 & 16 & RAM (B)\\\midrule"]
    for m in meth:
        cells = [f"{m} {SHORT.get(m, NAMES[m])}"]
        for k in (0, 4, 8, 16):
            r = e4.get((16, k, m))
            cells.append(f"{r['accuracy']:.3f} ({r['worst_pair']:.2f})" if r else "--")
        cells.append(fmt_int(persistent_ram(m, 16, 4, e4[(16, 16, m)]["state_bytes"])))
        lines.append(" & ".join(cells) + r"\\")
    r64 = e4[(64, 16, "M8")]
    lines += [r"\midrule", f"M8 DS, $d=64$ & -- & {e4[(64, 4, 'M8')]['accuracy']:.3f} ({e4[(64, 4, 'M8')]['worst_pair']:.2f}) & "
              f"{e4[(64, 8, 'M8')]['accuracy']:.3f} ({e4[(64, 8, 'M8')]['worst_pair']:.2f}) & {r64['accuracy']:.3f} ({r64['worst_pair']:.2f}) & {fmt_int(persistent_ram('M8', 64, 4, 0))}\\\\",
              r"\bottomrule", r"\multicolumn{6}{p{0.95\columnwidth}}{\scriptsize $k$: transfer pairs (M4, M8, M9) or labelled slave references (M5--M7); M1 does not use them. RAM: bytes kept between samples; M8 is counted fused into the head ($4C(d+1)$~B), the separate map takes $4d(d+1)$~B.}",
              r"\end{tabular}", r"\end{table}"]
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
                mark = "$^\\dagger$" if (m == "M6" and k == 20) else ""
                cells.append(f"{r['rmse_mean_over_gases']:.3f}{mark}" if r else "--")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\multicolumn{7}{p{0.95\columnwidth}}{\scriptsize Gas identity known; $k$ references of that gas per batch. $^\dagger$RLS with the selected $\beta=0.8$ winds up at $k=20$ for toluene and ammonia (worst gas 0.95 at $d=16$); with $\beta=0.95$ (post hoc) it gives 0.171 and 0.124.}",
              r"\end{tabular}", r"\end{table}"]
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
    lines = [r"\begin{table*}[!t]", r"\caption{Measured Time per Call on the Boards (Median of 20 Timings in Two Sessions, $\mu$s) and Host Accuracy}",
             r"\label{tab:mcu}\centering\footnotesize", r"\begin{tabular}{l l r rrr}\toprule",
             r"Kernel & One call & Host accuracy & Mega2560 & ESP32-C3 & ESP32-S3\\\midrule"]
    for key, name, unit in MCU_ROWS:
        s3 = t[("ESP32-S3", key)]
        acc = s3["host_accuracy"]
        cells = [name, unit, f"{float(acc):.3f}" if acc else "--"]
        for b in ("Mega2560", "ESP32-C3", "ESP32-S3"):
            r = t[(b, key)]
            if r["status"] != "measured":
                cells.append("does not fit")
            else:
                v = float(r["us_median"])
                cells.append(fmt_int(round(v)) if v >= 1000 else f"{v:.1f}" if v >= 10 else f"{v:.2f}")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\multicolumn{6}{p{0.62\textwidth}}{\scriptsize Kernels are timed on fixtures of real samples stored in flash (E5); host accuracy is that of the same configuration in E2 ($k=20$) or E4 ($k=16$). RAM of each configuration is given in Table~\ref{tab:ram}.}",
              r"\end{tabular}", r"\end{table*}"]
    (P / "tables/mcu.tex").write_text("\n".join(lines) + "\n")


def fig_frontier(e2, e4):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    col = {"M0": "#777777", "M1": "#1f77b4", "M4": "#8c564b", "M5": "#2ca02c", "M6": "#d62728", "M7": "#9467bd",
           "M8": "#ff7f0e", "M9": "#17becf", "M1+M5": "#bcbd22", "M1+M7": "#e377c2"}
    def panel(ax, pts, title, small, big, label_keys):
        for (m, d, acc, sb) in pts:
            ax.scatter(max(sb, 1), acc, color=col[m], marker="o" if d == small else "s", s=24, zorder=3,
                       edgecolor="k", linewidth=0.3)
            if (m, d) in label_keys:
                ax.annotate(f"{m}", (max(sb, 1), acc), xytext=(4, -3), textcoords="offset points", fontsize=6.3)
        ax.axvline(8192, color="k", lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_xlabel("Persistent RAM (bytes, log scale)")
        ax.set_xticks([1, 10, 100, 1000, 10000, 100000]); ax.set_xticklabels(["0", "10", "100", "1k", "10k", "100k"])
        ax.set_title(title, fontsize=8); ax.grid(alpha=0.25, lw=0.4)
        ax.scatter([], [], marker="o", color="w", edgecolor="k", label=f"$d={small}$")
        ax.scatter([], [], marker="s", color="w", edgecolor="k", label=f"$d={big}$")
    pts = [(m, d, float(e2[(d, 20, m)]["accuracy_b3_10"]), persistent_ram(m, d, 6, int(e2[(d, 20, m)]["state_bytes"])))
           for d in (16, 128) for m in ("M0", "M1", "M5", "M6", "M7", "M1+M5", "M1+M7")]
    panel(axes[0], pts, "(a) Drift over 36 months, $k=20$ references per batch", 16, 128,
          {("M0", 16), ("M0", 128), ("M1", 128), ("M5", 128), ("M6", 128), ("M7", 16), ("M5", 16), ("M6", 16), ("M7", 128)})
    axes[0].set_ylabel("Accuracy"); axes[0].set_xlim(0.7, 2.5e5)
    pts = [(m, d, e4[(d, 16, m)]["accuracy"], persistent_ram(m, d, 4, e4[(d, 16, m)]["state_bytes"]))
           for d in (16, 64) for m in ("M0", "M1", "M4", "M5", "M6", "M7", "M8", "M9")]
    panel(axes[1], pts, "(b) Board-to-board transfer, $k=16$ pairs", 16, 64,
          {("M0", 16), ("M5", 16), ("M8", 16), ("M8", 64), ("M6", 16), ("M6", 64), ("M1", 64)})
    axes[1].set_xlim(0.7, 6e4)
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
    plt.rcParams["hatch.linewidth"] = 0.5
    for i, (b, c, h) in enumerate((("Mega2560", "#555555", ""), ("ESP32-C3", "#1f77b4", "////"),
                                   ("ESP32-S3", "#ff7f0e", "...."))):
        v = [float(t[(b, k)]["us_median"]) if t[(b, k)]["status"] == "measured" else np.nan for k in keys]
        ax.bar(x + (i - 1) * w, v, w, label=b, color=c, hatch=h, edgecolor="k", linewidth=0.3)
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
    pts = [("Static, 0 B", 0.498, 1, "#777777"), ("EWMA, 1.0 kB, no labels", 0.614, 1028, "#1f77b4"),
           ("NLMS, 3.1 kB", 0.772, 3100, "#2ca02c"), ("RLS d=16, 1.6 kB", 0.876, 1568, "#d62728"),
           ("RLS d=128, 69.7 kB", 0.923, 69664, "#d62728"), ("DS fused, 272 B (twin boards)", 0.955, 272, "#ff7f0e")]
    offs = {"RLS d=128, 69.7 kB": (-36, -14), "DS fused, 272 B (twin boards)": (-40, 6), "RLS d=16, 1.6 kB": (-78, -3),
            "NLMS, 3.1 kB": (5, -4), "EWMA, 1.0 kB, no labels": (5, -4)}
    for lab, a, s, c in pts:
        ax.scatter(s, a, color=c, s=26, edgecolor="k", linewidth=0.3, zorder=3)
        ax.annotate(lab, (s, a), xytext=offs.get(lab, (4, -3)), textcoords="offset points", fontsize=6.5)
    ax.axvspan(8192, 3e5, color="#f2c6c6", alpha=0.5, lw=0)
    ax.text(9500, 0.47, "does not fit\nMega2560", fontsize=6.3)
    ax.set_xscale("log"); ax.set_xlim(0.7, 3e5); ax.set_ylim(0.45, 1.02)
    ax.set_xticks([1, 100, 10000]); ax.set_xticklabels(["0", "100", "10k"])
    ax.set_xlabel("Persistent RAM, bytes"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy versus RAM on the device", fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(P / "figures/gagraphic_inline.pdf")
    fig.savefig(P / "figures/gagraphic.png", dpi=300)
    plt.close(fig)


def table_e7():
    """Table of the complete-stream board runs (results/e7_20261110/summary.json, from experiments/e7_summary.py)."""
    E = json.loads((ROOT / "results/e7_20261110/summary.json").read_text())
    B = ("Mega2560", "ESP32-C3", "ESP32-S3")

    def g(v, nd=3):
        if v >= 1000:
            return f"{round(v):,}".replace(",", "\\,")
        return f"{v:.{nd}g}"

    rows = [
        ("Complete D1 streams", [str(E[b]["streams"]) for b in B]),
        ("Predictions differing (all RLS)", [f"{E[b]['disagreements']} of {E[b]['predictions']:,}".replace(",", "\\,") for b in B]),
        ("DS pairs matching host accuracy", [f"{E[b]['ds_matches']}/{E[b]['ds_pairs']}" for b in B]),
        (r"Static head, $d=16$ ($\mu$s/sample)", [g(E[b]["static16_us"]) for b in B]),
        (r"RLS update, $d=16$ ($\mu$s/ref.)", [g(E[b]["rls16_ref_us"]) for b in B]),
        (r"Block OLS refit, $k=20$ (ms)", [g(E[b]["ols16_refit_us"] / 1000) for b in B]),
        ("DS fit + fusion, 16 pairs (ms)", [g(E[b]["ds_fit_fuse_us"] / 1000) for b in B]),
        (r"Fused DS head ($\mu$s/sample)", [g(E[b]["ds_apply_us"]) for b in B]),
        (r"Extractor ($\mu$s/sample)", [g(E[b]["extractor_us"]) for b in B]),
        ("Peak stack (B)", [g(E[b]["stack_peak"]) for b in B]),
        ("Largest test arena (B)", [g(E[b]["arena_max"]) for b in B]),
    ]
    lines = [r"\begin{table}[t]", r"\caption{Complete Streams on the Boards (E7)}",
             r"\label{tab:e7}\centering\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l r r r}\toprule", r" & Mega2560 & ESP32-C3 & ESP32-S3\\\midrule"]
    lines += [f"{lab} & " + " & ".join(v) + r"\\" for lab, v in rows]
    lines += [r"\bottomrule",
              r"\multicolumn{4}{p{0.95\columnwidth}}{\scriptsize D1: seed 0, $k=20$; Mega2560 runs the compact-feature configurations only. Times include feature preprocessing and are means over the complete streams; the stack peak covers the whole stack (Mega2560) or the Arduino loop task (ESP32). The test firmware keeps every head in RAM (Section~\ref{sec:memory}).}",
             r"\end{tabular}", r"\end{table}"]
    (P / "tables/e7.tex").write_text("\n".join(lines) + "\n")


def main():
    (P / "tables").mkdir(exist_ok=True); (P / "figures").mkdir(exist_ok=True)
    e2 = load_e2(); e4 = load_e4(); t = load_e5(); S = load_summary()
    table_e2(e2, S); table_e4(e4); table_e3(); table_mcu(t)
    table_stats(S); table_scenarios(S); table_conc(S); table_e7(); rows = table_ram()
    fig_frontier(e2, e4); fig_timing(t); fig_graphical(e2, e4, t)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
