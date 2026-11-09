import numpy as np

from enose_drift import transfer as T


def test_ds_recovers_linear_map_with_enough_pairs():
    rng = np.random.default_rng(0)
    d = 6
    A = np.eye(d) + 0.2 * rng.normal(size=(d, d)); b = rng.normal(size=d)
    Zm = rng.normal(size=(60, d)); Zs = (Zm - b) @ np.linalg.inv(A)
    m = T.DirectStandardisation(np.zeros((d + 1, 3)), 3, 1e-8)
    for s, t in zip(Zs, Zm):
        m.pair(s, t)
    m.fit()
    assert np.allclose(m.transform(Zs), Zm, atol=1e-5)
    assert m.state_bytes() == 4 * (d + 1) * d


def test_identity_without_pairs_and_shrinkage():
    d = 8
    for m in (T.DirectStandardisation(np.zeros((d + 1, 2)), 2, 1.0),
              T.ChannelDS(np.zeros((d + 1, 2)), 2, 1.0, 4),
              T.PairedAffine(np.zeros((d + 1, 2)), 2, 1.0)):
        Z = np.random.default_rng(1).normal(size=(5, d))
        m.fit()
        assert np.allclose(m.transform(Z), Z)


def test_channel_ds_is_block_diagonal():
    rng = np.random.default_rng(2)
    Zs = rng.normal(size=(40, 8)); Zm = Zs.copy(); Zm[:, :4] *= 2.0; Zm[:, 4:] += 1.0
    m = T.ChannelDS(np.zeros((9, 2)), 2, 1e-8, 4)
    for s, t in zip(Zs, Zm):
        m.pair(s, t)
    m.fit()
    assert np.allclose(m.transform(Zs), Zm, atol=1e-6)
    assert m.state_bytes() == 4 * 2 * 5 * 4
