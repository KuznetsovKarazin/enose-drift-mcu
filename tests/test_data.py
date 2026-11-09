from pathlib import Path

import numpy as np
import pytest

from enose_drift.data import D2_DAYS, _parse_d1_line, list_d2, load_d1
from enose_drift.fetch import manifest, sha256

SHARED = Path("/mnt/project-files/enose-2027/data")
EXTRACTED = Path(__file__).resolve().parents[1] / "data" / "extracted"


def test_parse_d1_line_with_concentration():
    toks = " ".join(f"{i}:{i / 10}" for i in range(1, 129))
    lab, conc, x = _parse_d1_line(f"3;50.000000 {toks}", True)
    assert lab == 2 and conc == 50.0 and x.shape == (128,) and x[-1] == pytest.approx(12.8)


def test_parse_d1_line_rejects_missing_feature():
    toks = " ".join(f"{i}:1" for i in range(1, 128))
    with pytest.raises(ValueError):
        _parse_d1_line(f"1 {toks}", False)


def test_d2_name_parsing(tmp_path):
    for n in ("B1_GEa_F040_R2.txt", "B5_GMe_F100_R1.txt", "notes.txt"):
        (tmp_path / n).write_text("0 1 2 3 4 5 6 7 8\n")
    recs = list_d2(tmp_path)
    assert [(r.board, r.gas, r.level, r.repetition, r.concentration) for r in recs] == [
        (1, "ethanol", 4, 2, 50.0), (5, "methane", 10, 1, 250.0)]


def test_manifest_digests_match_shared_copies():
    if not SHARED.is_dir():
        pytest.skip("shared data folder not mounted")
    for entry in manifest().values():
        assert sha256(SHARED / entry["file"]) == entry["sha256"]


@pytest.mark.skipif(not (EXTRACTED / "D1b").is_dir(), reason="run scripts/download_data.py first")
def test_d1b_counts():
    d = load_d1(EXTRACTED / "D1b")
    assert d.X.shape == (13910, 128)
    assert np.bincount(d.batch)[1:].tolist() == [445, 1244, 1586, 161, 197, 2300, 3613, 294, 470, 3600]
    assert np.isfinite(d.concentration).all()


@pytest.mark.skipif(not (EXTRACTED / "D1").is_dir(), reason="run scripts/download_data.py first")
def test_d1_and_d1b_features_agree():
    a = load_d1(EXTRACTED / "D1", with_concentration=False)
    b = load_d1(EXTRACTED / "D1b")
    assert (a.gas == b.gas).all() and (a.batch == b.batch).all()
    assert np.allclose(a.X, b.X, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not (EXTRACTED / "D2").is_dir(), reason="run scripts/download_data.py first")
def test_d2_inventory():
    recs = list_d2(EXTRACTED / "D2")
    assert len(recs) == 640
    per_board = {b: sum(r.board == b for r in recs) for b in range(1, 6)}
    assert per_board == {1: 160, 2: 160, 3: 160, 4: 80, 5: 80}
    assert {b: len(d) for b, d in D2_DAYS.items()} == {b: max(r.repetition for r in recs if r.board == b) for b in D2_DAYS}
