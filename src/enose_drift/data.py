"""Loaders for D1/D1b (UCI 224/270) and D2 (UCI 361)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

D1_GASES = ("ethanol", "ethylene", "ammonia", "acetaldehyde", "acetone", "toluene")
D1_FEATURES = ("dR", "dR_norm", "ema_i_0.001", "ema_i_0.01", "ema_i_0.1",
               "ema_d_0.001", "ema_d_0.01", "ema_d_0.1")
D2_GASES = {"Ea": "ethanol", "CO": "co", "Ey": "ethylene", "Me": "methane"}
# UCI 361 page: concentration codes F010..F100 map to ten levels per gas (ppm).
D2_LEVELS = {
    "ethanol": np.arange(1, 11) * 12.5,
    "ethylene": np.arange(1, 11) * 12.5,
    "co": np.arange(1, 11) * 25.0,
    "methane": np.arange(1, 11) * 25.0,
}
# Test days per board, from the UCI 361 description.
D2_DAYS = {1: (4, 10, 15, 21), 2: (1, 7, 11, 16), 3: (2, 8, 14, 17), 4: (3, 9), 5: (18, 22)}
D2_CHANNELS = ("TGS2611@5.65V", "TGS2612@5.65V", "TGS2610@5.65V", "TGS2602@5.65V",
               "TGS2611@5.00V", "TGS2612@5.00V", "TGS2610@5.00V", "TGS2602@5.00V")


@dataclass
class D1Data:
    X: np.ndarray          # (n, 128) float64, sensor-major: 16 sensors x 8 features
    gas: np.ndarray        # (n,) int 0..5
    batch: np.ndarray      # (n,) int 1..10
    concentration: np.ndarray | None  # ppmv, D1b only
    row: np.ndarray        # (n,) line index inside its batch file


def _parse_d1_line(line: str, with_conc: bool):
    parts = line.split()
    if with_conc:
        label, conc = parts[0].split(";")
        conc = float(conc)
    else:
        label, conc = parts[0], np.nan
    x = np.full(128, np.nan)
    for tok in parts[1:]:
        k, v = tok.split(":")
        x[int(k) - 1] = float(v)
    if np.isnan(x).any():
        raise ValueError("missing feature in D1 line")
    return int(label) - 1, conc, x


def load_d1(root: Path, with_concentration: bool = True) -> D1Data:
    """`root` is the extracted D1b directory (with concentration) or D1/Dataset."""
    root = Path(root)
    if (root / "Dataset").is_dir():
        root = root / "Dataset"
    X, g, b, c, r = [], [], [], [], []
    for batch in range(1, 11):
        for i, line in enumerate((root / f"batch{batch}.dat").read_text().splitlines()):
            if not line.strip():
                continue
            lab, conc, x = _parse_d1_line(line, with_concentration)
            X.append(x); g.append(lab); b.append(batch); c.append(conc); r.append(i)
    return D1Data(np.asarray(X), np.asarray(g), np.asarray(b),
                  np.asarray(c) if with_concentration else None, np.asarray(r))


D2_NAME = re.compile(r"B(?P<board>\d)_G(?P<gas>\w\w)_F(?P<level>\d{3})_R(?P<rep>\d)\.txt$")


@dataclass(frozen=True)
class D2Record:
    path: Path
    board: int
    gas: str
    level: int            # 1..10
    repetition: int
    concentration: float  # ppm

    def load(self) -> np.ndarray:
        """(n, 9): time [s] and 8 resistances [kOhm]."""
        return np.loadtxt(self.path)


def list_d2(root: Path) -> list[D2Record]:
    root = Path(root)
    if (root / "data1").is_dir():
        root = root / "data1"
    out = []
    for p in sorted(root.glob("B*_G*_F*_R*.txt")):
        m = D2_NAME.search(p.name)
        if not m:
            continue
        gas = D2_GASES[m["gas"]]
        level = int(m["level"]) // 10
        out.append(D2Record(p, int(m["board"]), gas, level, int(m["rep"]),
                            float(D2_LEVELS[gas][level - 1])))
    return out
