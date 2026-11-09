"""Calibration transfer between twin boards (E4): M4 affine, M8 DS, M9 per-channel DS.

All maps act in the master-preprocessed feature space and are fitted on k
paired samples (same gas and level measured on master and slave), by ridge
regression shrunk towards the identity map, so that k < d is well posed.
"""
from __future__ import annotations

import numpy as np

from . import methods as M

F32 = 4


def _ridge_to_identity(Zs: np.ndarray, Zm: np.ndarray, rho: float) -> np.ndarray:
    """F of shape (d+1, d): Zm ~ [Zs 1] F, penalty rho ||F - [I; 0]||^2 (bias free)."""
    d = Zs.shape[1]
    A = M.augment(Zs)
    R = rho * np.eye(d + 1); R[-1, -1] = 1e-9
    F0 = np.vstack([np.eye(d), np.zeros((1, d))])
    return np.linalg.solve(A.T @ A + R, A.T @ Zm + R @ F0)


class Transfer(M.Method):
    """Wraps a fixed master head with a slave-to-master feature map."""

    def __init__(self, W, classes, name):
        super().__init__(W, classes)
        self.name = name
        self.pairs_s, self.pairs_m = [], []

    def pair(self, z_slave, z_master):
        self.pairs_s.append(z_slave); self.pairs_m.append(z_master)

    def fit(self):
        raise NotImplementedError


class DirectStandardisation(Transfer):
    def __init__(self, W, classes, rho):
        super().__init__(W, classes, "M8")
        self.rho = rho
        d = W.shape[0] - 1
        self.F = np.vstack([np.eye(d), np.zeros((1, d))])

    def fit(self):
        if self.pairs_s:
            self.F = _ridge_to_identity(np.array(self.pairs_s), np.array(self.pairs_m), self.rho)

    def transform(self, Z):
        return M.augment(Z) @ self.F

    def state_bytes(self):
        return F32 * self.F.size


class ChannelDS(Transfer):
    """M9: one small DS map per sensor channel (block-diagonal F)."""

    def __init__(self, W, classes, rho, block):
        super().__init__(W, classes, "M9")
        self.rho, self.block = rho, block
        d = W.shape[0] - 1
        self.F = [np.vstack([np.eye(block), np.zeros((1, block))]) for _ in range(d // block)]

    def fit(self):
        if not self.pairs_s:
            return
        S, T = np.array(self.pairs_s), np.array(self.pairs_m)
        b = self.block
        self.F = [_ridge_to_identity(S[:, c * b:(c + 1) * b], T[:, c * b:(c + 1) * b], self.rho)
                  for c in range(len(self.F))]

    def transform(self, Z):
        b = self.block
        return np.hstack([M.augment(Z[:, c * b:(c + 1) * b]) @ F for c, F in enumerate(self.F)])

    def state_bytes(self):
        return F32 * sum(F.size for F in self.F)


class PairedAffine(Transfer):
    """M4 for transfer: per-feature gain and offset from the pairs, shrunk to (1, 0)."""

    def __init__(self, W, classes, rho):
        super().__init__(W, classes, "M4")
        self.rho = rho
        d = W.shape[0] - 1
        self.g, self.o = np.ones(d), np.zeros(d)

    def fit(self):
        if not self.pairs_s:
            return
        S, T = np.array(self.pairs_s), np.array(self.pairs_m)
        ms, mt = S.mean(0), T.mean(0)
        cov = ((S - ms) * (T - mt)).sum(0)
        var = ((S - ms) ** 2).sum(0)
        self.g = (cov + self.rho) / (var + self.rho)
        self.o = mt - self.g * ms

    def transform(self, Z):
        return Z * self.g + self.o

    def state_bytes(self):
        return 2 * F32 * len(self.g)
