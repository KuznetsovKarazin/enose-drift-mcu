from pathlib import Path

import numpy as np
import pytest

from enose_drift.data import list_d2
from enose_drift.features import FEATURES_PER_CHANNEL, StreamingExtractor, extract

EXTRACTED = Path(__file__).resolve().parents[1] / "data" / "extracted"


def synthetic_trace(seed=0, n=6000, channels=8):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.1
    r0 = rng.uniform(5, 100, channels)
    drop = rng.uniform(0.1, 0.8, channels)
    gas = ((t >= 50) & (t < 150)).astype(float)
    resp = np.convolve(gas, np.exp(-np.arange(200) / 40) / 40, "full")[:n]
    r = r0 * (1 - drop * resp[:, None]) + rng.normal(0, 0.05, (n, channels))
    return np.column_stack([t, r])


def streaming(trace):
    ex = StreamingExtractor(trace.shape[1] - 1)
    for row in trace:
        ex.push(row[0], row[1:])
    return ex.features(), ex.state_bytes()


def test_streaming_matches_host_bit_exactly():
    tr = synthetic_trace()
    s, nbytes = streaming(tr)
    h = extract(tr)
    assert s.shape == (8 * FEATURES_PER_CHANNEL,)
    assert np.array_equal(s, h)
    assert nbytes == 8 * 4 * (4 + 9) + 9


def test_response_sign_and_scale():
    f = extract(synthetic_trace()).reshape(8, FEATURES_PER_CHANNEL)
    assert (f[:, 0] > 0).all() and ((f[:, 1] > 0) & (f[:, 1] < 1)).all()
    assert (f[:, 2:5] >= f[:, 5:8]).all()


@pytest.mark.skipif(not (EXTRACTED / "D2").is_dir(), reason="run scripts/download_data.py first")
def test_streaming_matches_host_on_real_file():
    rec = list_d2(EXTRACTED / "D2")[0]
    tr = rec.load()[:20000]
    s, _ = streaming(tr)
    assert np.array_equal(s, extract(tr))
