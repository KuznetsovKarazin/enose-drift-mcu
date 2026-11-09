"""E1 and E2 on D1b: static settings and the streaming reference-budget protocol."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from . import methods as M

CLASSES = 6


def feature_index(d: int) -> np.ndarray:
    if d == 128:
        return np.arange(128)
    if d == 16:
        return np.arange(0, 128, 8)        # steady-state dR of each sensor
    raise ValueError(d)


@dataclass
class Source:
    pre: M.Preprocessor
    W: np.ndarray
    centroids: np.ndarray
    P0: np.ndarray        # (A^T A + R)^-1 of the source fit, RLS warm start
    lam: float


def make_source(X: np.ndarray, y: np.ndarray, lam: float, classes: int | None = CLASSES) -> Source:
    pre = M.Preprocessor(X)
    Z = pre(X)
    W = M.fit_ridge(Z, y, classes, lam)
    A = M.augment(Z)
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 1e-9
    P0 = np.linalg.inv(A.T @ A + R)
    cents = (np.zeros((1, Z.shape[1])) if classes is None
             else np.stack([Z[y == c].mean(axis=0) for c in range(classes)]))
    return Source(pre, W, cents, P0, lam)


def build(name: str, src: Source, hp: dict, classes: int | None = CLASSES) -> M.Method:
    W, d = src.W, src.W.shape[0] - 1
    if name == "M0":
        return M.Static(W, classes)
    if name == "M1":
        return M.EWMAStandardise(W, classes, d, hp["alpha"])
    if name == "M2":
        return M.AdaptiveCentroids(W, classes, src.centroids, hp["beta"])
    if name == "M3":
        return M.ComponentCorrection(W, classes, src.centroids, hp["r"], hp["eta"])
    if name == "M4":
        return M.AffineCorrection(W, classes, src.centroids)
    if name == "M5":
        return M.NLMS(W, classes, hp["mu"])
    if name == "M6":
        return M.RLS(W, classes, hp["forget"], None, P0=src.P0)
    if name == "M7":
        return M.BlockOLS(W, classes, hp["gamma"])
    if name == "M1+M5":
        return M.Combined(M.EWMAStandardise(W, classes, d, hp["alpha"]), M.NLMS(W, classes, hp["mu"]))
    if name == "M1+M7":
        return M.Combined(M.EWMAStandardise(W, classes, d, hp["alpha"]), M.BlockOLS(W, classes, hp["gamma"]))
    raise KeyError(name)


def run_stream(method: M.Method, Z: np.ndarray, y: np.ndarray, batch: np.ndarray,
               batches, k: int, seed: int, classes: int | None = CLASSES) -> list[dict]:
    """Batches in time order; inside a batch a seeded permutation.  The first k
    samples of each batch are labelled references (one calibration session);
    the remaining samples are predicted, unlabelled, in order."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in batches:
        idx = np.flatnonzero(batch == b)
        idx = idx[rng.permutation(len(idx))]
        ref, rest = idx[:k], idx[k:]
        for i in ref:
            method.reference(Z[i], y[i] if classes is None else int(y[i]))
        method.end_reference_session()
        if len(rest) == 0:
            continue
        if method.sequential:
            pred = np.array([method.observe(Z[i]) for i in rest])
        else:
            pred = method.predict_many(Z[rest])
        if classes is None:
            err = pred - y[rest]
            rows.append(dict(batch=int(b), n=int(len(rest)),
                             rmse=float(np.sqrt(np.mean(err ** 2))),
                             median_rel_error=float(np.median(np.abs(10 ** err - 1)))))
        else:
            rows.append(dict(batch=int(b), n=int(len(rest)),
                             accuracy=float(np.mean(pred == y[rest])),
                             macro_f1=float(f1_score(y[rest], pred, labels=range(classes),
                                                     average="macro", zero_division=0))))
    return rows


def setting_scores(X, y, batch, lam, fit_on, test_on, head="ridge", per_batch_stats=False):
    """E1: fit on batch `fit_on`, test on `test_on`.  With `per_batch_stats`
    the test batch is standardised with its own statistics (0 labels)."""
    tr, te = batch == fit_on, batch == test_on
    if head == "ridge":
        src = make_source(X[tr], y[tr], lam)
        if per_batch_stats:
            Zt = M.Preprocessor(X[te])(X[te])
        else:
            Zt = src.pre(X[te])
        pred = np.argmax(M.augment(Zt) @ src.W, axis=1)
    else:
        pre = M.Preprocessor(X[tr])
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(pre(X[tr]), y[tr])
        Zt = M.Preprocessor(X[te])(X[te]) if per_batch_stats else pre(X[te])
        pred = clf.predict(Zt)
    return float(np.mean(pred == y[te]))
