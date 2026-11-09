import numpy as np

from enose_drift import methods as M
from enose_drift.budget import head_bytes, state_bytes
from enose_drift.stream import build, make_source


def test_closed_form_values():
    assert head_bytes(128, 6) == 3096
    assert state_bytes("M1", 128) == 1028
    assert state_bytes("M2", 128) == 3076
    assert [state_bytes("M3", 128, r=r) for r in (1, 2, 3)] == [516, 1028, 1540]
    assert state_bytes("M6", 128) == 129 * 129 * 4 + 4
    assert state_bytes("M6", 16) == 17 * 17 * 4 + 4
    assert state_bytes("M7", 128, k=10) == 10 * 513
    assert state_bytes("M8", 8) == 256
    assert state_bytes("M9", 8, w=1) == 96


def test_budget_matches_method_objects():
    rng = np.random.default_rng(1)
    X = rng.lognormal(size=(120, 16)); y = np.arange(120) % 6
    src = make_source(X, y, 1.0)
    hp = dict(alpha=0.01, beta=0.01, r=2, eta=0.5, mu=0.3, forget=0.99, gamma=1.0)
    for name in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
        m = build(name, src, hp)
        assert m.state_bytes() == state_bytes(name, 16, r=2), name
    m = build("M7", src, hp)
    Z = src.pre(X)
    for i in range(5):
        m.reference(Z[i], int(y[i]))
    assert m.state_bytes() == state_bytes("M7", 16, k=5)
    assert head_bytes(16, 6) == M.F32 * src.W.size
