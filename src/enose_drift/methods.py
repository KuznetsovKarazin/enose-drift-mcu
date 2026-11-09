"""Drift-compensation methods M0-M9 over a common least-squares linear head.

Every method sees the same preprocessed features z = (phi(x) - mu_s) / sd_s,
where phi is the signed log transform and (mu_s, sd_s) come from the source
block.  The head is a ridge regression onto one-hot targets (a least-squares
classifier), so that NLMS, RLS and block OLS update exactly the model that is
deployed.  A multinomial logistic head is kept only as a host reference.

Host arithmetic is float64; float32 / Q15 replicas are planned for E2-E5.
Each method reports `state_bytes()` (persistent adaptation state, float32
layout) and `payload_bytes()` (constants in flash: head, source statistics,
source centroids).
"""
from __future__ import annotations

import numpy as np

F32 = 4


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    def __init__(self, X_source: np.ndarray):
        p = signed_log(X_source)
        self.mu = p.mean(axis=0)
        self.sd = p.std(axis=0) + 1e-9

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (signed_log(X) - self.mu) / self.sd

    def payload_bytes(self) -> int:
        return 2 * F32 * len(self.mu)


def augment(Z: np.ndarray) -> np.ndarray:
    Z = np.atleast_2d(Z)
    return np.hstack([Z, np.ones((len(Z), 1))])


def fit_ridge(Z: np.ndarray, y: np.ndarray, classes: int, lam: float,
              W0: np.ndarray | None = None) -> np.ndarray:
    """argmin ||Y - [Z 1] W||^2 + lam ||W - W0||^2 (bias not penalised)."""
    A = augment(Z)
    Y = np.asarray(y, float)[:, None] if classes is None else np.eye(classes)[y]
    R = lam * np.eye(A.shape[1])
    R[-1, -1] = 1e-9
    W0 = np.zeros((A.shape[1], Y.shape[1])) if W0 is None else W0
    return np.linalg.solve(A.T @ A + R, A.T @ Y + R @ W0)


class Method:
    """Interface: `reference(z, y)` consumes one labelled reference sample;
    `observe(z)` returns the class prediction for one unlabelled sample and may
    update unsupervised state; `predict_many(Z)` is used when `sequential` is
    False (the method does not change between references)."""

    name = "M0"
    sequential = False

    def __init__(self, W: np.ndarray, classes: int | None):
        """`classes=None` selects regression: one output, real-valued target."""
        self.W = W.copy()
        self.classes = classes

    def _t(self, y):
        return np.array([float(y)]) if self.classes is None else np.eye(self.classes)[y]

    def transform(self, Z):
        return Z

    def scores(self, Z):
        return augment(self.transform(Z)) @ self.W

    def predict_many(self, Z):
        s = self.scores(Z)
        return s[:, 0] if self.classes is None else np.argmax(s, axis=1)

    def observe(self, z):
        return self.predict_many(z[None])[0]

    def reference(self, z, y):
        pass

    def end_reference_session(self):
        pass

    def state_bytes(self) -> int:
        return 0

    def payload_bytes(self) -> int:
        return F32 * self.W.size


class Static(Method):
    name = "M0"


class EWMAStandardise(Method):
    """M1: running mean/variance replaces the source statistics (0 labels)."""

    name = "M1"
    sequential = True

    def __init__(self, W, classes, d, alpha):
        super().__init__(W, classes)
        self.alpha = alpha
        self.m = np.zeros(d)
        self.v = np.ones(d)

    def _update(self, z):
        a = self.alpha
        delta = z - self.m
        self.m = self.m + a * delta
        self.v = (1 - a) * (self.v + a * delta * delta)

    def transform(self, Z):
        return (Z - self.m) / np.sqrt(self.v + 1e-6)

    def observe(self, z):
        self._update(z)
        return super().observe(z)

    def reference(self, z, y):
        self._update(z)

    def state_bytes(self):
        return 2 * F32 * len(self.m) + F32


class AdaptiveCentroids(Method):
    """M2: nearest class mean; the winning centroid tracks the sample (0 labels)."""

    name = "M2"
    sequential = True

    def __init__(self, W, classes, centroids, beta):
        super().__init__(W, classes)
        self.c = centroids.copy()
        self.beta = beta

    def predict_many(self, Z):
        d2 = ((Z[:, None, :] - self.c[None]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1)

    def observe(self, z):
        k = int(self.predict_many(z[None])[0])
        self.c[k] += self.beta * (z - self.c[k])
        return k

    def state_bytes(self):
        return F32 * self.c.size + F32

    def payload_bytes(self):
        return 0


class ComponentCorrection(Method):
    """M3: drift directions learnt from reference residuals by Oja's rule
    (O(r d) state); their projections are removed before the head."""

    name = "M3"

    def __init__(self, W, classes, centroids, r, eta, seed=0):
        super().__init__(W, classes)
        rng = np.random.default_rng(seed)
        self.U = np.linalg.qr(rng.normal(size=(centroids.shape[1], r)))[0]
        self.centroids = centroids
        self.eta = eta

    def transform(self, Z):
        return Z - (Z @ self.U) @ self.U.T

    def reference(self, z, y):
        e = z - self.centroids[y]
        g = np.outer(e, e @ self.U)
        self.U, _ = np.linalg.qr(self.U + self.eta * g / (e @ e + 1e-9))

    def state_bytes(self):
        return F32 * self.U.size + F32

    def payload_bytes(self):
        return F32 * (self.W.size + self.centroids.size)


class AffineCorrection(Method):
    """M4: per-feature gain/offset fitted on the references of a calibration
    session against the source class centroids (offset only when one reference)."""

    name = "M4"

    def __init__(self, W, classes, centroids):
        super().__init__(W, classes)
        d = centroids.shape[1]
        self.centroids = centroids
        self.g = np.ones(d)
        self.o = np.zeros(d)
        self._reset_sums()

    def _reset_sums(self):
        d = len(self.g)
        self.n = 0
        self.sc = np.zeros(d); self.sz = np.zeros(d)
        self.scc = np.zeros(d); self.scz = np.zeros(d)

    def reference(self, z, y):
        c = self.centroids[y]
        self.n += 1
        self.sc += c; self.sz += z; self.scc += c * c; self.scz += c * z

    def end_reference_session(self):
        if self.n == 0:
            return
        n = self.n
        var = self.scc / n - (self.sc / n) ** 2
        cov = self.scz / n - (self.sc / n) * (self.sz / n)
        if n >= 3:
            g = np.where(var > 1e-3, cov / np.maximum(var, 1e-3), 1.0)
            self.g = np.clip(g, 0.2, 5.0)
        else:
            self.g = np.ones_like(self.g)
        self.o = self.sz / n - self.g * self.sc / n
        self._reset_sums()

    def transform(self, Z):
        return (Z - self.o) / self.g

    def state_bytes(self):
        return 2 * F32 * len(self.g)

    def scratch_bytes(self):
        return 4 * F32 * len(self.g) + F32

    def payload_bytes(self):
        return F32 * (self.W.size + self.centroids.size)


class NLMS(Method):
    """M5: normalised LMS on the least-squares head."""

    name = "M5"

    def __init__(self, W, classes, mu):
        super().__init__(W, classes)
        self.mu = mu

    def reference(self, z, y):
        a = augment(self.transform(z[None]))[0]
        err = self._t(y) - a @ self.W
        self.W += self.mu * np.outer(a, err) / (a @ a + 1e-6)

    def state_bytes(self):
        return F32


class RLS(Method):
    """M6: exponentially weighted RLS on the head, covariance shared by classes."""

    name = "M6"

    def __init__(self, W, classes, lam, delta, P0=None):
        super().__init__(W, classes)
        n = W.shape[0]
        self.lam = lam
        self.P = np.eye(n) / delta if P0 is None else P0.copy()

    def reference(self, z, y):
        a = augment(self.transform(z[None]))[0]
        Pa = self.P @ a
        gain = Pa / (self.lam + a @ Pa)
        err = self._t(y) - a @ self.W
        self.W += np.outer(gain, err)
        self.P = (self.P - np.outer(gain, Pa)) / self.lam
        self.P = (self.P + self.P.T) / 2

    def state_bytes(self):
        return F32 * self.P.size + F32


class BlockOLS(Method):
    """M7: buffer the references of a session, then refit the head by ridge
    regression shrunk towards the current head."""

    name = "M7"

    def __init__(self, W, classes, gamma):
        super().__init__(W, classes)
        self.gamma = gamma
        self.buf_z, self.buf_y = [], []
        self.max_k = 0

    def reference(self, z, y):
        self.buf_z.append(self.transform(z[None])[0]); self.buf_y.append(y)
        self.max_k = max(self.max_k, len(self.buf_y))

    def end_reference_session(self):
        if self.buf_y:
            self.W = fit_ridge(np.array(self.buf_z), np.array(self.buf_y),
                               self.classes, self.gamma, W0=self.W)
        self.buf_z, self.buf_y = [], []

    def state_bytes(self):
        d = self.W.shape[0] - 1
        return self.max_k * (F32 * d + 1)


class Combined(Method):
    """M1 standardisation in front of a label-driven method (M5 or M7)."""

    sequential = True

    def __init__(self, front: EWMAStandardise, back: Method):
        super().__init__(back.W, back.classes)
        self.front, self.back = front, back
        self.name = f"M1+{back.name}"

    def observe(self, z):
        self.front._update(z)
        return self.back.predict_many(self.front.transform(z[None]))[0]

    def reference(self, z, y):
        self.front._update(z)
        self.back.reference(self.front.transform(z[None])[0], y)

    def end_reference_session(self):
        self.back.end_reference_session()

    def state_bytes(self):
        return self.front.state_bytes() + self.back.state_bytes()

    def payload_bytes(self):
        return self.back.payload_bytes()
