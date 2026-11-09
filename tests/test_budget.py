from enose_drift.budget import head_bytes, state_bytes


def test_plan_table_values():
    assert head_bytes(128, 6) == 3096
    assert state_bytes("M1", 128) == 1024
    assert state_bytes("M2", 128) == 3072
    assert [state_bytes("M3", 128, r=r) for r in (1, 2, 3)] == [512, 1024, 1536]
    assert state_bytes("M6", 128) == 65536 + 4
    assert state_bytes("M6", 16) == 1024 + 4
    assert state_bytes("M7", 128, k=10) == 5160
    assert state_bytes("M8", 8) == 256
    assert state_bytes("M9", 8, w=1) == 96
