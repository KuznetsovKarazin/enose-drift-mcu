"""Byte-exact persistent-state accounting for the compensation methods M0-M9.

Sizes follow the float32 C structs planned for the MCU kernels (4 bytes per
coefficient).  Only *persistent* adaptation state is counted here; the
classifier/regressor head itself is reported separately as payload.
Stack and scratch are measured on hardware, not modelled.
"""
from __future__ import annotations

F32 = 4


def head_bytes(d: int, classes: int) -> int:
    """Linear head: weights and bias for each class (one output for regression)."""
    return F32 * classes * (d + 1)


def state_bytes(method: str, d: int, classes: int = 6, k: int = 0, r: int = 1, w: int = 1) -> int:
    m = method.upper()
    if m == "M0":   # static head, no adaptation
        return 0
    if m == "M1":   # EWMA mean and variance per feature + alpha
        return 2 * F32 * d + F32
    if m == "M2":   # adaptive class centroids + step
        return F32 * classes * d + F32
    if m == "M3":   # component correction: r drift directions + step
        return F32 * r * d + F32
    if m == "M4":   # per-feature gain and offset (session sums are scratch: 4d+1 floats)
        return 2 * F32 * d
    if m == "M5":   # NLMS: step size only (the head is updated in place)
        return F32
    if m == "M6":   # RLS: covariance over [z, 1], (d+1)^2, + forgetting factor
        return F32 * (d + 1) * (d + 1) + F32
    if m == "M7":   # block OLS: buffer of k (features, class byte) pairs
        return k * (F32 * d + 1)
    if m == "M8":   # direct standardisation matrix
        return F32 * d * d
    if m == "M9":   # piecewise DS with band half-width w
        return F32 * d * (2 * w + 1)
    raise KeyError(method)
