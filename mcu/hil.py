"""E7 (revision): complete streams on a board flashed with the *_hil firmware.

Usage (repository root, after `python scripts/download_data.py` and
`python scripts/extract_d2.py`):
    python mcu/hil.py --board mega2560 --port COM7 --out runs/mega_e7
    python mcu/hil.py --board esp32c3  --port COM8 --out runs/c3_e7
    python mcu/hil.py --board esp32s3  --port COM9 --out runs/s3_e7

The script uploads the HIL firmware, then streams real samples to the board:
D1 batches 2-10 in the order of E2 seed 0 with k = 20 references per batch,
D2 transfer for the 19 board pairs with DS fitted on the board and fused into
the head, and complete 600 s D2 recordings through the streaming extractor.
The board returns its predictions and features; the script compares them with
the host float32 replica and records kernel time and peak stack depth.
On the Mega2560, d = 128 streams take about 15 min each; they run only with
--mega-d128.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "e6"))
sys.path.insert(0, str(ROOT / "experiments" / "e4"))

METHOD_ID = {"M0": 0, "M1": 1, "M5": 5, "M6": 6, "M7": 7, "M1+M5": 15}
SEED = 20270131          # E2 test seed 0
K = 20
BAUD = {"mega2560": 1_000_000, "esp32c3": 115200, "esp32s3": 115200}


EXPECTED = HERE / "hil_expected.json.gz"   # host float32 outputs (mcu/make_hil_expected.py); no C compiler needed


def expected():
    import gzip
    if not hasattr(expected, "cache"):
        expected.cache = json.loads(gzip.open(EXPECTED, "rt").read()) if EXPECTED.exists() else None
    return expected.cache


class Link:
    def __init__(self, port, board, log):
        self.s = serial.Serial(port, BAUD[board], timeout=30)
        self.log = log
        time.sleep(2.5)
        self.s.reset_input_buffer()
        for _ in range(20):
            self.s.write(b"H")
            ln = self.line()
            if ln.startswith("READY,ENOSE_HIL_V1"):
                return
        raise SystemExit("board did not answer READY (is the *_hil firmware flashed?)")

    def line(self):
        ln = self.s.readline().decode(errors="replace").rstrip("\r\n")
        self.log.append(ln)
        return ln

    def expect(self, prefix):
        ln = self.line()
        if not ln.startswith(prefix):
            raise SystemExit(f"expected {prefix!r}, got {ln!r}")
        return ln

    def send(self, b):
        self.s.write(b)


def f32bytes(a):
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def stats(link):
    link.send(b"S")
    p = link.expect("STATS").split(",")[1:]
    v = list(map(int, p))
    return dict(n_ref=v[0], us_ref=v[1], n_obs=v[2], us_obs=v[3], n_end=v[4], us_end=v[5], stack_peak_bytes=v[6])


def run_d1(link, board, name, d):
    import run_d1 as R
    X, y, b, _ = R.data(d)
    ex = expected()
    expected_rows = (ex["d1"][f"{name}|{d}"] if ex else
                     R.stream32(name, d, R.HP2[str(d)]["methods"].get(name, {}), K, SEED))
    link.send(b"I" + bytes([METHOD_ID[name], d]))
    init_bytes = int(link.expect("INIT").split(",")[1])
    rng = np.random.default_rng(SEED)
    nb = max(1, 128 // d) if board == "mega2560" else 32
    preds, agree, n = [], 0, 0
    for row, bb in zip(expected_rows, R.BATCHES):
        ref, rest = R.order_batch(np.flatnonzero(b == bb), y, rng, "shuffled", "uniform", K)
        for i in ref:
            link.send(b"R" + bytes([int(y[i])]) + f32bytes(X[i])); link.expect("r")
        link.send(b"E"); link.expect("e")
        got = []
        for s in range(0, len(rest), nb):
            chunk = rest[s:s + nb]
            link.send(b"B" + bytes([len(chunk)]) + f32bytes(X[chunk]))
            got += [int(ch) for ch in link.expect("P,")[2:]]
        got = np.array(got)
        exp = np.array(row["pred"])
        agree += int(np.sum(got == exp)); n += len(exp)
        preds.append(dict(batch=bb, n=len(rest), accuracy_board=float(np.mean(got == y[rest])),
                          accuracy_host_f32=row["accuracy"], agree=int(np.sum(got == exp))))
    st = stats(link)
    return dict(kind="d1_stream", method=name, d=d, k=K, seed=SEED, init_ram_bytes=init_bytes,
                agree=agree, n=n, batches=preds, **st)


def run_ds(link, board):
    import run as E4
    d = 16
    out = []
    E4.D = None
    D = E4.load(); X = E4.feats(d); y = D["y"]
    board_id, lev, rep, gas = D["board"], D["level"], D["repetition"], D["gas"]
    # The firmware holds the board-1 master head of the fixture: pairs 1->3, 1->4, 1->5
    # (1->2 was the tuning pair).
    for (i, j) in ((1, 3), (1, 4), (1, 5)):
        seed = E4.SEED_BASE
        rng = np.random.default_rng(seed)
        conds = sorted({(g, l) for g, l in zip(gas[board_id == j], lev[board_id == j])})
        chosen = [conds[t] for t in rng.permutation(len(conds))[:16]]
        link.send(b"I" + bytes([8, d])); init_bytes = int(link.expect("INIT").split(",")[1])
        used = []
        for g, l in chosen:
            sj = np.flatnonzero((board_id == j) & (gas == g) & (lev == l) & (rep == 1))
            si = np.flatnonzero((board_id == i) & (gas == g) & (lev == l) & (rep == 1))
            if len(sj) == 0 or len(si) == 0:
                continue
            used.append(sj[0])
            link.send(b"Q" + f32bytes(X[sj[0]]) + f32bytes(X[si[0]])); link.expect("q")
        link.send(b"E"); link.expect("e")
        ev = np.setdiff1d(np.flatnonzero(board_id == j), used)
        ev = ev[rng.permutation(len(ev))]
        nb = 8 if board == "mega2560" else 32
        got = []
        for s in range(0, len(ev), nb):
            chunk = ev[s:s + nb]
            link.send(b"B" + bytes([len(chunk)]) + f32bytes(X[chunk]))
            got += [int(ch) for ch in link.expect("P,")[2:]]
        ex = expected()
        if ex:
            acc_ref = ex["ds"][f"{i}->{j}"]
        else:
            import run_d2 as R2   # needs a C compiler for the float32 replica
            acc_ref = R2.ds_f32_job((d, 16, seed, (i, j)))["acc_fused32"]
        out.append(dict(kind="ds_fused", pair=[i, j], n=len(ev), init_ram_bytes=init_bytes,
                        accuracy_board=float(np.mean(np.array(got) == y[ev])),
                        accuracy_host_fused32=acc_ref, **stats(link)))
    return out


def run_extractor(link, board, n_traces):
    from enose_drift.data import list_d2
    from enose_drift.features import BASELINE_SECONDS, StreamingExtractor
    recs = list_d2(ROOT / "data/extracted/D2")
    pick = [recs[0], recs[len(recs) // 2], recs[-1]][:n_traces]
    rows_per_frame = 14 if board == "mega2560" else 32
    out = []
    for rec in pick:
        tr = rec.load().astype(np.float32)
        ex = StreamingExtractor(8)
        for row in tr:
            ex.push(np.float32(row[0]), row[1:])
        host = ex.features().astype(np.float32)
        link.send(b"X" + f32bytes(np.array([BASELINE_SECONDS]))); link.expect("x")
        for s in range(0, len(tr), rows_per_frame):
            chunk = tr[s:s + rows_per_frame]
            link.send(b"P" + bytes([len(chunk)]) + f32bytes(chunk)); link.expect("p")
        link.send(b"F")
        feat = np.array(list(map(float, link.expect("FEAT").split(",")[1:])), np.float32)
        n_ext, us = map(int, link.expect("EXTTIME").split(",")[1:])
        rel = float(np.max(np.abs(feat - host) / np.maximum(1.0, np.abs(host))))
        out.append(dict(kind="extractor", file=rec.path.name, samples=int(len(tr)), n_push=n_ext, us_push=us,
                        max_rel_diff=rel, **{k: v for k, v in stats(link).items() if k == "stack_peak_bytes"}))
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=list(BAUD), required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mega-d128", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--skip-d1", action="store_true", help="run only DS and the extractor")
    a = ap.parse_args()
    if a.out.exists():
        ap.error("choose a new output directory")
    a.out.mkdir(parents=True)
    env = f"{a.board}_hil"
    if not a.no_upload:
        up = subprocess.run([sys.executable, "-m", "platformio", "run", "-d", str(HERE), "-e", env, "-t", "upload",
                             "--upload-port", a.port], capture_output=True, text=True)
        (a.out / "upload.log").write_text(up.stdout + up.stderr, encoding="utf-8")
        if up.returncode:
            raise SystemExit("upload failed, see upload.log")
    log = []
    link = Link(a.port, a.board, log)
    results, t0 = [], time.time()
    plan = [(n, 16) for n in ("M0", "M1", "M5", "M6", "M7", "M1+M5")]
    if a.board != "mega2560" or a.mega_d128:
        plan += [(n, 128) for n in ("M0", "M1", "M5", "M1+M5")]
    if a.board != "mega2560":
        plan += [("M6", 128)]
    for name, d in ([] if a.skip_d1 else plan):
        print(f"D1 stream {name} d={d} ...", flush=True)
        r = run_d1(link, a.board, name, d); results.append(r)
        print(f"  agreement {r['agree']}/{r['n']}, stack {r['stack_peak_bytes']} B", flush=True)
        (a.out / "results.json").write_text(json.dumps(results, indent=1))
    print("D2 DS fused ...", flush=True)
    results += run_ds(link, a.board)
    print("D2 extractor, full recordings ...", flush=True)
    results += run_extractor(link, a.board, 3)
    link.s.close()
    (a.out / "raw.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        git = ""
    git = git or "not a git checkout (see source hashes)"
    build = HERE / ".pio" / "build" / env
    images = {n: sha(build / n) for n in ("firmware.elf", "firmware.hex", "firmware.bin") if (build / n).exists()}
    ok_d1 = all(r["agree"] == r["n"] for r in results if r["kind"] == "d1_stream")
    ok_ds = all(r["accuracy_board"] == r["accuracy_host_fused32"] for r in results if r["kind"] == "ds_fused")
    ok_ext = all(r["max_rel_diff"] <= 1e-5 for r in results if r["kind"] == "extractor")
    summary = dict(board=a.board, git_commit=git, utc=datetime.now(timezone.utc).isoformat(), seconds=round(time.time() - t0),
                   images=images, raw_sha256=sha(a.out / "raw.log"), all_d1_predictions_match=ok_d1,
                   extractor_within_1e_5=ok_ext, ds_accuracy_matches_host=ok_ds,
                   sources={str(p.relative_to(HERE)): sha(p) for p in sorted([*HERE.glob("include/*.h"), *HERE.glob("src/hil/*"), HERE / "platformio.ini", HERE / "hil.py"])})
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    (a.out / "manifest.json").write_text(json.dumps(summary, indent=1))
    print("Validated" if ok_d1 and ok_ext and ok_ds else "Finished with mismatches (please send the folder anyway)", a.out)


if __name__ == "__main__":
    main()
