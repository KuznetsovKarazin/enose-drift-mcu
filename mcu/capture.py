"""Run one command-driven E5 session on a flashed board and save raw log, parsed JSON and hashes.

Usage: python mcu/capture.py --port COM7 --out runs/mega_e5/run01.json
"""
import argparse
import hashlib
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import serial


def parse(lines):
    res = dict(metadata={}, state={}, accuracy={}, time={}, skip=[], complete=False)
    for ln in lines:
        p = ln.strip().split(",")
        if p[0] in ("SESSION", "BOARD", "CPU_HZ", "ARENA_BYTES"):
            res["metadata"][p[0]] = p[1]
        elif p[0] == "STATE":
            res["state"][p[1]] = int(p[2])
        elif p[0] == "ACCURACY":
            res["accuracy"][p[1]] = dict(match=int(p[2]), n=int(p[3]), value=float(p[4]), expected=float(p[5]), status=p[6])
        elif p[0] == "TIME":
            res["time"].setdefault(p[1], []).append(dict(rep=int(p[2]), calls=int(p[3]), us=int(p[4])))
        elif p[0] == "SKIP":
            res["skip"].append(p[1:])
        elif p[0] == "END_ENOSE_BENCH":
            res["complete"] = True
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("output exists; choose a new name")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    session = secrets.token_hex(16)
    lines = []
    with serial.Serial(a.port, 115200, timeout=1) as s:
        time.sleep(2.5)  # Mega resets on open
        s.reset_input_buffer()
        deadline = time.time() + 20
        while time.time() < deadline:
            s.write(b"H\n")
            ln = s.readline().decode(errors="replace").strip()
            if ln.startswith("READY,ENOSE_BENCH_V1"):
                break
        else:
            raise SystemExit("board did not answer READY")
        s.write(f"R,{session}\n".encode())
        deadline = time.time() + a.timeout
        while time.time() < deadline:
            ln = s.readline().decode(errors="replace").rstrip("\r\n")
            if ln:
                lines.append(ln)
            if ln == "END_ENOSE_BENCH":
                break
    raw = a.out.with_suffix(".log")
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = parse(lines)
    res["session_requested"] = session
    res["utc"] = datetime.now(timezone.utc).isoformat()
    ok = res["complete"] and res["metadata"].get("SESSION") == session and all(v["status"] == "PASS" for v in res["accuracy"].values())
    res["status"] = "validated" if ok else "invalid"
    res["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    a.out.write_text(json.dumps(res, indent=1))
    print(res["status"], a.out)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
