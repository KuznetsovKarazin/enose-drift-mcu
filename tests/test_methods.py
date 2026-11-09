import numpy as np

from enose_drift import methods as M
from enose_drift.stream import make_source, run_stream


def data(n=300, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 6
    means = rng.normal(0, 2, (6, d))
    X = np.exp(means[y] * 0.3 + rng.normal(0, 0.1, (n, d)))
    return X, y


def test_rls_warm_start_equals_batch_ridge():
    X, y = data()
    src = make_source(X[:200], y[:200], 2.0)
    Z = src.pre(X)
    rls = M.RLS(src.W, 6, 1.0, None, P0=src.P0)
    for i in range(200, 260):
        rls.reference(Z[i], int(y[i]))
    full = M.fit_ridge(Z[:260], y[:260], 6, 2.0)
    assert np.allclose(rls.W, full, atol=1e-8)


def test_nlms_reduces_reference_error():
    X, y = data()
    src = make_source(X, y, 1.0)
    z = src.pre(X[:1])[0] + 3.0
    m = M.NLMS(src.W, 6, 0.5)
    before = np.linalg.norm(np.eye(6)[y[0]] - M.augment(z[None])[0] @ m.W)
    m.reference(z, int(y[0]))
    after = np.linalg.norm(np.eye(6)[y[0]] - M.augment(z[None])[0] @ m.W)
    assert after < before * 0.6


def test_ewma_tracks_offset():
    X, y = data()
    src = make_source(X, y, 1.0)
    m = M.EWMAStandardise(src.W, 6, X.shape[1], 0.05)
    Z = src.pre(X) + 4.0
    for z in Z:
        m.observe(z)
    assert np.abs(m.m - 4.0).max() < 0.5


def test_affine_correction_removes_offset_and_gain():
    X, y = data()
    src = make_source(X, y, 1.0)
    Z = src.centroids[y] * 1.5 + 2.0
    m = M.AffineCorrection(src.W, 6, src.centroids)
    for i in range(30):
        m.reference(Z[i], int(y[i]))
    m.end_reference_session()
    assert np.allclose(m.transform(Z), src.centroids[y], atol=1e-6)


def test_drift_recovered_by_offset_methods():
    X, y = data(n=600)
    b = np.repeat([1, 2], 300)
    src = make_source(X[b == 1], y[b == 1], 1.0)
    Z = src.pre(X); Z[b == 2] += 1.5
    static = run_stream(M.Static(src.W, 6), Z, y, b, [2], 0, 0)[0]["accuracy"]
    ewma = run_stream(M.EWMAStandardise(src.W, 6, 8, 0.05), Z, y, b, [2], 0, 0)[0]["accuracy"]
    aff = run_stream(M.AffineCorrection(src.W, 6, src.centroids), Z, y, b, [2], 12, 0)[0]["accuracy"]
    assert ewma > static and aff > static
