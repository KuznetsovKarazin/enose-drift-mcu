"""Build, upload and capture two E5 sessions for one board, keeping firmware images and hashes.

Usage (repository root):  python mcu/run_boards.py --board mega2560 --port COM7 --out runs/mega_e5
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=["mega2560", "esp32c3", "esp32s3"], required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("choose a new output directory")
    a.out.mkdir(parents=True)
    sources = {str(p.relative_to(HERE)): sha(p) for p in sorted([*HERE.glob("include/*.h"), *HERE.glob("src/*"), HERE / "platformio.ini"])}
    up = subprocess.run([sys.executable, "-m", "platformio", "run", "-d", str(HERE), "-e", a.board, "-t", "upload",
                         "--upload-port", a.port], capture_output=True, text=True)
    (a.out / "upload.log").write_text(up.stdout + up.stderr, encoding="utf-8")
    if up.returncode:
        raise SystemExit("upload failed, see upload.log")
    build = HERE / ".pio" / "build" / a.board
    images = {}
    for name in ("firmware.elf", "firmware.hex", "firmware.bin"):
        if (build / name).exists():
            shutil.copyfile(build / name, a.out / name)
            images[name] = sha(a.out / name)
    size = subprocess.run([sys.executable, "-m", "platformio", "run", "-d", str(HERE), "-e", a.board, "-t", "size"],
                          capture_output=True, text=True)
    (a.out / "size.log").write_text(size.stdout + size.stderr, encoding="utf-8")
    runs = []
    for i in range(1, a.runs + 1):
        out = a.out / f"run{i:02d}.json"
        r = subprocess.run([sys.executable, str(HERE / "capture.py"), "--port", a.port, "--out", str(out)])
        runs.append(dict(file=out.name, ok=r.returncode == 0))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    (a.out / "manifest.json").write_text(json.dumps(dict(board=a.board, git_commit=git, sources=sources,
                                                         images=images, runs=runs), indent=1))
    print("Validated" if all(r["ok"] for r in runs) else "Some runs FAILED", a.out)


if __name__ == "__main__":
    main()
