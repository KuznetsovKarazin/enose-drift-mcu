/* Exported wrappers of the firmware kernels for host float32 replicas
 * (ctypes). Same source and flags as the firmware: -fno-fast-math -ffp-contract=off. */
#include "enose_kernels.h"
void f_pre(const float *x, const float *mu, const float *sd, int d, float *z) { ek_pre(x, mu, sd, d, z); }
int f_argmax(const float *z, const float *W, int d, int C) { return ek_argmax(z, W, d, C, 0); }
void f_ewma(float *z, float *m, float *v, int d, float a) { ek_ewma(z, m, v, d, a); }
void f_nlms(float *W, const float *z, int d, int C, int y, float mu) { ek_nlms(W, z, d, C, y, mu); }
void f_rls(float *W, float *P, const float *z, int d, int C, int y, float lam, float *pa) { ek_rls(W, P, z, d, C, y, lam, pa); }
int f_ridge(const float *Z, const float *T, int k, int d, int m, float reg, const float *X0, int x0_eye, float *G, float *X) {
  return ek_ridge(Z, T, k, d, m, reg, X0, x0_eye, G, X);
}
void f_ds_apply(const float *z, const float *F, int d, float *out) { ek_ds_apply(z, F, d, out); }
/* Batch helpers to keep the Python loop short. */
void f_pre_many(const float *X, const float *mu, const float *sd, int n, int d, float *Z) {
  for (int s = 0; s < n; ++s) ek_pre(X + s * d, mu, sd, d, Z + s * d);
}
void f_argmax_many(const float *Z, const float *W, int n, int d, int C, int *out) {
  for (int s = 0; s < n; ++s) out[s] = ek_argmax(Z + s * d, W, d, C, 0);
}
/* EWMA standardisation over a run of unlabelled samples, writing predictions. */
void f_ewma_run(const float *Z, int n, int d, float *m, float *v, float a, const float *W, int C, float *zbuf, int *out) {
  for (int s = 0; s < n; ++s) {
    for (int i = 0; i < d; ++i) zbuf[i] = Z[s * d + i];
    ek_ewma(zbuf, m, v, d, a);
    out[s] = ek_argmax(zbuf, W, d, C, 0);
  }
}
/* Streaming extractor over a complete recording: t (n), r (n x 8, row-major). */
void f_ext_run(const float *t, const float *r, int n, float t_base, float *f) {
  EkExtractor e;
  ek_ext_init(&e, t_base);
  for (int s = 0; s < n; ++s) ek_ext_push(&e, t[s], r + s * EK_CH);
  ek_ext_features(&e, f);
}
