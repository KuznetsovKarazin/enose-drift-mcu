"""Streaming feature extraction for raw MOX resistance traces (D2, experiment E5).

The extractor mirrors the D1 feature definition (Vergara et al. 2012): a
steady-state change and exponential moving averages of the first difference,
y[k] = (1 - a) y[k-1] + a (r[k] - r[k-1]), summarised by their maximum and
minimum.  MOX resistance falls under reducing gases, so the steady-state
change is computed as R0 - min(R).  The state is O(1) per channel and is
laid out exactly as the C struct planned for the microcontroller kernels.
"""
from __future__ import annotations

import numpy as np

ALPHAS = (0.001, 0.01, 0.1)
FEATURES_PER_CHANNEL = 2 + 2 * len(ALPHAS)
BASELINE_SECONDS = 30.0   # 0-30 s precedes gas arrival in all 640 traces (earliest clear onset ~36 s, B3 ethylene R2); see PROTOCOL.md


class StreamingExtractor:
    """float32 state for C channels; one `push` per sample."""

    def __init__(self, channels: int, baseline_seconds: float = BASELINE_SECONDS):
        self.c = channels
        self.t_base = np.float32(baseline_seconds)
        f32 = np.float32
        self.base_sum = np.zeros(channels, f32)
        self.base_n = np.uint32(0)
        self.r0 = np.zeros(channels, f32)
        self.rmin = np.full(channels, np.inf, f32)
        self.prev = np.zeros(channels, f32)
        self.y = np.zeros((len(ALPHAS), channels), f32)
        self.ymax = np.full((len(ALPHAS), channels), -np.inf, f32)
        self.ymin = np.full((len(ALPHAS), channels), np.inf, f32)
        self.started = False
        self.a = np.asarray(ALPHAS, f32)[:, None]

    def state_bytes(self) -> int:
        arrays = (self.base_sum, self.r0, self.rmin, self.prev, self.y, self.ymax, self.ymin)
        return int(sum(a.nbytes for a in arrays) + 4 + 4 + 1)  # +count, +t_base, +started flag

    def push(self, t: float, r: np.ndarray) -> None:
        r = np.asarray(r, np.float32)
        if t < self.t_base:
            self.base_sum += r
            self.base_n += 1
        elif self.base_n and not self.r0.any():
            self.r0 = (self.base_sum / np.float32(self.base_n)).astype(np.float32)
        if self.started:
            self.y = ((np.float32(1) - self.a) * self.y + self.a * (r - self.prev)).astype(np.float32)
            np.maximum(self.ymax, self.y, out=self.ymax)
            np.minimum(self.ymin, self.y, out=self.ymin)
        self.prev = r
        self.started = True
        if t >= self.t_base:
            np.minimum(self.rmin, r, out=self.rmin)

    def features(self) -> np.ndarray:
        if not self.r0.any():
            self.r0 = (self.base_sum / np.float32(max(int(self.base_n), 1))).astype(np.float32)
        dr = self.r0 - self.rmin
        cols = [dr, dr / self.r0, *self.ymax, *self.ymin]
        return np.stack(cols, axis=1).reshape(-1).astype(np.float32)  # channel-major


def extract(trace: np.ndarray, baseline_seconds: float = BASELINE_SECONDS) -> np.ndarray:
    """Host equivalent of StreamingExtractor for an (n, 1+C) trace (same float32 order of operations)."""
    t = trace[:, 0]
    r = trace[:, 1:].astype(np.float32)
    base = t < np.float32(baseline_seconds)
    n = int(base.sum())
    if n:
        r0 = (np.add.accumulate(r[base], axis=0, dtype=np.float32)[-1] / np.float32(n)).astype(np.float32)
    else:
        r0 = r[0]
    rmin = r[~base].min(axis=0)
    a = np.asarray(ALPHAS, np.float32)[:, None]
    y = np.zeros((len(ALPHAS), r.shape[1]), np.float32)
    hi = np.full_like(y, -np.inf)
    lo = np.full_like(y, np.inf)
    for row in np.diff(r, axis=0):
        y = ((np.float32(1) - a) * y + a * row).astype(np.float32)
        np.maximum(hi, y, out=hi)
        np.minimum(lo, y, out=lo)
    dr = r0 - rmin
    return np.stack([dr, dr / r0, *hi, *lo], axis=1).reshape(-1).astype(np.float32)
