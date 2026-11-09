import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

MCU = Path(__file__).resolve().parents[1] / "mcu"


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
def test_host_build_reproduces_committed_expected_outputs(tmp_path):
    exe = tmp_path / "host_bench"
    subprocess.run(["gcc", "-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Werror",
                    "-I", str(MCU / "include"), str(MCU / "host/host_bench.c"), "-lm", "-o", str(exe)], check=True)
    out = {r["routine"]: r for r in map(json.loads, subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines())}
    exp = (MCU / "include/enose_expected.h").read_text()
    for name, r in out.items():
        if name == "extract":
            continue
        m = re.search(rf"EXP_{name.upper()}\[\d+\] EK_FLASH = {{([^}}]*)}}", exp)
        preds = [int(v) for v in m.group(1).split(",")] if r["pred"] else []
        assert preds == r["pred"] or (not r["pred"] and m.group(1) == "0"), name


def test_manifest_parity_is_complete():
    p = json.loads((MCU / "fixture_manifest.json").read_text())["parity_host_c_vs_python"]
    assert p["extract_bit_exact"]
    assert all(v["agree"] == v["n"] for k, v in p.items() if isinstance(v, dict))
