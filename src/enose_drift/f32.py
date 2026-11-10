"""Host float32 replica of the methods, built from the firmware C kernels.

The shared library is compiled from mcu/host/f32_lib.c with the firmware
header and flags, so every arithmetic step is the one executed on the boards
(up to libm differences in logf/sqrtf). Used for the revision experiments
(experiments/e6) to run complete streams in float32.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "mcu/host/f32_lib.c"
INC = ROOT / "mcu/include"
FLAGS = ["-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-shared"]

_fp = C.POINTER(C.c_float)
_ip = C.POINTER(C.c_int)


def _lib():
    h = hashlib.sha256(SRC.read_bytes() + (INC / "enose_kernels.h").read_bytes()).hexdigest()[:12]
    out = ROOT / "data/cache" / f"f32_lib_{h}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcc", *FLAGS, "-I", str(INC), str(SRC), "-lm", "-o", str(out)], check=True)
    lib = C.CDLL(str(out))
    sig = {
        "f_pre": [_fp, _fp, _fp, C.c_int, _fp],
        "f_argmax": [_fp, _fp, C.c_int, C.c_int],
        "f_ewma": [_fp, _fp, _fp, C.c_int, C.c_float],
        "f_nlms": [_fp, _fp, C.c_int, C.c_int, C.c_int, C.c_float],
        "f_rls": [_fp, _fp, _fp, C.c_int, C.c_int, C.c_int, C.c_float, _fp],
        "f_ridge": [_fp, _fp, C.c_int, C.c_int, C.c_int, C.c_float, _fp, C.c_int, _fp, _fp],
        "f_ds_apply": [_fp, _fp, C.c_int, _fp],
        "f_pre_many": [_fp, _fp, _fp, C.c_int, C.c_int, _fp],
        "f_argmax_many": [_fp, _fp, C.c_int, C.c_int, C.c_int, _ip],
        "f_ewma_run": [_fp, C.c_int, C.c_int, _fp, _fp, C.c_float, _fp, C.c_int, _fp, _ip],
        "f_ext_run": [_fp, _fp, C.c_int, C.c_float, _fp],
    }
    for name, args in sig.items():
        fn = getattr(lib, name); fn.argtypes = args
        fn.restype = C.c_int if name in ("f_argmax", "f_ridge") else None
    return lib


class _LazyLib:
    """Compile and load the library on first use, so that importing this module needs no C compiler."""
    _lib = None

    def __getattr__(self, name):
        if _LazyLib._lib is None:
            _LazyLib._lib = _lib()
        return getattr(_LazyLib._lib, name)


LIB = _LazyLib()


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def p(a):
    return a.ctypes.data_as(_fp)


def pi(a):
    return a.ctypes.data_as(_ip)


class F32Source:
    """Frozen source model in float32: mu, sd, W ((d+1) x C), P0."""

    def __init__(self, src):
        self.mu, self.sd, self.W = f32(src.pre.mu), f32(src.pre.sd), f32(src.W)
        self.P0 = f32(src.P0)
        self.d, self.C = self.W.shape[0] - 1, self.W.shape[1]

    def pre(self, X):
        X = f32(X); n = len(X)
        Z = np.empty((n, self.d), np.float32)
        LIB.f_pre_many(p(X), p(self.mu), p(self.sd), n, self.d, p(Z))
        return Z


class F32Method:
    """Float32 counterpart of methods.M0/M1/M5/M6/M7 and the M1+ combinations."""

    def __init__(self, name, fs: F32Source, hp):
        self.name, self.fs, self.hp = name, fs, hp
        d, Cc = fs.d, fs.C
        self.W = fs.W.copy()
        self.ewma = name in ("M1", "M1+M5", "M1+M7")
        self.back = name.replace("M1+", "") if "+" in name else (name if name != "M1" else "M0")
        if self.ewma:
            self.m = np.zeros(d, np.float32); self.v = np.ones(d, np.float32)
            self.a = np.float32(hp["alpha"])
        if self.back == "M6":
            self.P = fs.P0.copy(); self.pa = np.zeros(d + 1, np.float32)
        self.buf_z, self.buf_y = [], []
        self.zb = np.zeros(d, np.float32)

    def _front(self, z):
        z = z.copy()
        if self.ewma:
            LIB.f_ewma(p(z), p(self.m), p(self.v), self.fs.d, self.a)
        return z

    def reference(self, z, y):
        z = self._front(z)
        d, Cc = self.fs.d, self.fs.C
        if self.back == "M5":
            LIB.f_nlms(p(self.W), p(z), d, Cc, int(y), np.float32(self.hp["mu"]))
        elif self.back == "M6":
            LIB.f_rls(p(self.W), p(self.P), p(z), d, Cc, int(y), np.float32(self.hp["forget"]), p(self.pa))
        elif self.back == "M7":
            self.buf_z.append(z); self.buf_y.append(int(y))

    def end_reference_session(self):
        if self.back == "M7" and self.buf_y:
            d, Cc, k = self.fs.d, self.fs.C, len(self.buf_y)
            Z = f32(np.array(self.buf_z)); T = f32(np.eye(Cc)[self.buf_y])
            G = np.zeros((d + 1) ** 2, np.float32); X = np.zeros((d + 1) * Cc, np.float32)
            W0 = self.W.copy()
            ok = LIB.f_ridge(p(Z), p(T), k, d, Cc, np.float32(self.hp["gamma"]), p(W0), 0, p(G), p(X))
            if ok == 0:
                self.W = X.reshape(d + 1, Cc)
        self.buf_z, self.buf_y = [], []

    def run_unlabelled(self, Z):
        n = len(Z); out = np.empty(n, np.int32)
        if self.ewma:
            LIB.f_ewma_run(p(f32(Z)), n, self.fs.d, p(self.m), p(self.v), self.a, p(self.W), self.fs.C, p(self.zb), pi(out))
        else:
            LIB.f_argmax_many(p(f32(Z)), p(self.W), n, self.fs.d, self.fs.C, pi(out))
        return out
