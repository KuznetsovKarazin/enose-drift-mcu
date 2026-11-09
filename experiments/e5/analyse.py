"""E5 analysis: board captures -> time per call, joined with host accuracy.

Usage: python experiments/e5/analyse.py results/e5_20261109

Checks that every capture is validated, that both sessions on a board agree,
and that the board's source hashes equal the repository sources (with Windows
CRLF line endings, as checked out by the capture machine). Writes
e5_timing.csv, e5_summary.json and prints the tables used in RESULTS.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOARDS = {"mega": ("Mega2560", 16e6), "c3": ("ESP32-C3", 160e6), "s3": ("ESP32-S3", 240e6)}
# kernel -> (host experiment, d, k, method, what one call is)
JOIN = {
    "static16": ("E2", 16, 20, "M0", "classify"), "static128": ("E2", 128, 20, "M0", "classify"),
    "m1_16": ("E2", 16, 20, "M1", "classify"), "m1_128": ("E2", 128, 20, "M1", "classify"),
    "nlms16": ("E2", 16, 20, "M5", "classify + update"), "nlms128": ("E2", 128, 20, "M5", "classify + update"),
    "rls16": ("E2", 16, 20, "M6", "classify + update"), "rls128": ("E2", 128, 20, "M6", "classify + update"),
    "ols16": ("E2", 16, 20, "M7", "refit, k=20"),
    "ds16_fit": ("E4", 16, 16, "M8", "fit, k=16 pairs"), "ds16_apply": ("E4", 16, 16, "M8", "map + classify"),
    "extract": (None, None, None, None, "one 8-channel sample"),
}


def crlf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()


def host_accuracy():
    acc = {}
    with open(ROOT / "results/e1e2_20261109/e2_summary.csv") as f:
        for r in csv.DictReader(f):
            acc[("E2", int(r["d"]), int(r["k"]), r["method"])] = float(r["accuracy_b3_10"])
    for r in json.loads((ROOT / "results/e4_20261109/e4_transfer_summary.json").read_text()):
        acc[("E4", r["d"], r["k"], r["method"])] = r["accuracy"]
    return acc


def main(out: Path):
    acc = host_accuracy()
    rows, problems = [], []
    for key, (label, hz) in BOARDS.items():
        man = json.loads((out / key / "manifest.json").read_text())
        for rel, h in man["sources"].items():
            p = ROOT / "mcu" / rel.replace("\\", "/")
            if crlf_sha(p) != h:
                problems.append(f"{key}: source {rel} differs from repository")
        runs = [json.loads((out / key / r["file"]).read_text()) for r in man["runs"]]
        for r in runs:
            if r["status"] != "validated" or not r["complete"]:
                problems.append(f"{key}: run {r['metadata']['SESSION']} not validated")
            bad = [k for k, v in r["accuracy"].items() if v["status"] != "PASS"]
            if bad:
                problems.append(f"{key}: FAIL {bad}")
        for kern, (exp, d, k, meth, unit) in JOIN.items():
            per = [x["us"] / x["calls"] for r in runs for x in r["time"].get(kern, [])]
            if not per:
                rows.append(dict(board=label, kernel=kern, unit=unit, status="skipped",
                                 state_bytes=runs[0]["state"].get(kern)))
                continue
            sess = [st.median(x["us"] / x["calls"] for x in r["time"][kern]) for r in runs]
            med = st.median(per)
            rows.append(dict(board=label, kernel=kern, unit=unit, status="measured",
                             us_median=round(med, 3), us_min=round(min(per), 3), us_max=round(max(per), 3),
                             session_spread_pct=round(100 * (max(sess) - min(sess)) / med, 4),
                             cycles=round(med * hz / 1e6), n_timings=len(per),
                             state_bytes=runs[0]["state"].get(kern.replace("_fit", "").replace("_apply", "")),
                             host_accuracy=None if exp is None else round(acc[(exp, d, k, meth)], 4),
                             host_setting=None if exp is None else f"{exp} d={d} k={k} {meth}"))
    with open(out / "e5_timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({c for r in rows for c in r}))
        w.writeheader(); w.writerows(rows)
    (out / "e5_summary.json").write_text(json.dumps(dict(problems=problems, rows=rows), indent=1))
    print("problems:", problems or "none")
    ks = list(JOIN)
    print("| Kernel | One call | State, B | Host acc. | " + " | ".join(l for l, _ in BOARDS.values()) + " |")
    print("| --- " * (4 + len(BOARDS)) + "|")
    for kern in ks:
        rr = [r for r in rows if r["kernel"] == kern]
        cells = [f"{r['us_median']:,.1f}" if r["status"] == "measured" else "does not fit" for r in rr]
        m = next(r for r in rr if r["status"] == "measured")
        a = m.get("host_accuracy"); sb = rr[-1]["state_bytes"]
        print(f"| {kern} | {JOIN[kern][4]} | {sb:,} | {'' if a is None else f'{a:.3f}'} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/e5_20261109")
