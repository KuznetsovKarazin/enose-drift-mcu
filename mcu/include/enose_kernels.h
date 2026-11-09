/* float32 kernels for E5. Shared verbatim by the firmware and the host
 * reference build (host/host_bench.c), compiled with -fno-fast-math
 * -ffp-contract=off on both. Constants may live in flash: every read of a
 * `const float *` that can point to flash goes through RD(). */
#pragma once
#include <math.h>
#include <stdint.h>

#ifndef RD
#define RD(p) (*(p))
#endif
#ifndef RD8
#define RD8(p) (*(p))
#endif

/* Signed log transform and source standardisation. log(1+|x|) is computed as
 * logf(1 + |x|) on every target, because avr-libc has no log1pf. */
static inline void ek_pre(const float *x, const float *mu, const float *sd, int d, float *z) {
  for (int i = 0; i < d; ++i) {
    float v = RD(x + i);
    float s = v < 0.f ? -logf(1.f - v) : logf(1.f + v);
    z[i] = (s - RD(mu + i)) / RD(sd + i);
  }
}

/* Linear head, W row-major (d+1) x C, bias in the last row. */
static inline int ek_argmax(const float *z, const float *W, int d, int C, int w_in_flash) {
  int best = 0; float bs = -INFINITY;
  for (int c = 0; c < C; ++c) {
    float s = w_in_flash ? RD(W + d * C + c) : W[d * C + c];
    for (int i = 0; i < d; ++i) s += z[i] * (w_in_flash ? RD(W + i * C + c) : W[i * C + c]);
    if (s > bs) { bs = s; best = c; }
  }
  return best;
}

/* M1: EWMA mean/variance, update then transform (in place). */
static inline void ek_ewma(float *z, float *m, float *v, int d, float a) {
  for (int i = 0; i < d; ++i) {
    float delta = z[i] - m[i];
    m[i] += a * delta;
    v[i] = (1.f - a) * (v[i] + a * delta * delta);
    z[i] = (z[i] - m[i]) / sqrtf(v[i] + 1e-6f);
  }
}

/* M5: NLMS on the one-hot least-squares head. */
static inline void ek_nlms(float *W, const float *z, int d, int C, int y, float mu) {
  float aa = 1.f;  /* bias input */
  for (int i = 0; i < d; ++i) aa += z[i] * z[i];
  float g = mu / (aa + 1e-6f);
  for (int c = 0; c < C; ++c) {
    float s = W[d * C + c];
    for (int i = 0; i < d; ++i) s += z[i] * W[i * C + c];
    float e = (c == y ? 1.f : 0.f) - s;
    float ge = g * e;
    for (int i = 0; i < d; ++i) W[i * C + c] += ge * z[i];
    W[d * C + c] += ge;
  }
}

/* M6: exponentially weighted RLS, P is (d+1)^2, scratch pa has d+1 floats. */
static inline void ek_rls(float *W, float *P, const float *z, int d, int C, int y, float lam, float *pa) {
  const int n = d + 1;
  float den = lam;
  for (int r = 0; r < n; ++r) {
    float s = P[r * n + d];
    for (int i = 0; i < d; ++i) s += P[r * n + i] * z[i];
    pa[r] = s;
  }
  for (int i = 0; i < d; ++i) den += z[i] * pa[i];
  den += pa[d];
  float inv = 1.f / den;
  for (int c = 0; c < C; ++c) {
    float s = W[d * C + c];
    for (int i = 0; i < d; ++i) s += z[i] * W[i * C + c];
    float e = (c == y ? 1.f : 0.f) - s;
    for (int r = 0; r < n; ++r) W[r * C + c] += pa[r] * inv * e;
  }
  float il = 1.f / lam;
  for (int r = 0; r < n; ++r)
    for (int q = r; q < n; ++q) {
      float val = (P[r * n + q] - pa[r] * inv * pa[q]) * il;
      P[r * n + q] = val; P[q * n + r] = val;
    }
}

/* In-place Cholesky solve of G X = B; G n x n (lower part overwritten), B n x m. */
static inline int ek_chol_solve(float *G, float *B, int n, int m) {
  for (int j = 0; j < n; ++j) {
    float s = G[j * n + j];
    for (int k = 0; k < j; ++k) s -= G[j * n + k] * G[j * n + k];
    if (!(s > 0.f)) return -1;
    float l = sqrtf(s); G[j * n + j] = l;
    for (int i = j + 1; i < n; ++i) {
      float t = G[i * n + j];
      for (int k = 0; k < j; ++k) t -= G[i * n + k] * G[j * n + k];
      G[i * n + j] = t / l;
    }
  }
  for (int c = 0; c < m; ++c) {
    for (int i = 0; i < n; ++i) {
      float t = B[i * m + c];
      for (int k = 0; k < i; ++k) t -= G[i * n + k] * B[k * m + c];
      B[i * m + c] = t / G[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
      float t = B[i * m + c];
      for (int k = i + 1; k < n; ++k) t -= G[k * n + i] * B[k * m + c];
      B[i * m + c] = t / G[i * n + i];
    }
  }
  return 0;
}

/* Ridge fit with targets T (k x m) and prior X0 ((d+1) x m in RAM, or NULL = 0;
 * x0_eye = 1 uses the prior [I; 0] with m = d, as in direct standardisation):
 * X = argmin ||T - [Z 1] X||^2 + reg ||X - X0||^2, bias row penalised by 1e-9.
 * G: (d+1)^2 scratch, X: (d+1) x m output. */
static inline int ek_ridge(const float *Z, const float *T, int k, int d, int m, float reg,
                           const float *X0, int x0_eye, float *G, float *X) {
  const int n = d + 1;
  for (int r = 0; r < n; ++r) {
    for (int q = 0; q <= r; ++q) {
      float s = 0.f;
      for (int t = 0; t < k; ++t) {
        float ar = r < d ? Z[t * d + r] : 1.f, aq = q < d ? Z[t * d + q] : 1.f;
        s += ar * aq;
      }
      G[r * n + q] = s;
    }
    G[r * n + r] += r < d ? reg : 1e-9f;
    for (int c = 0; c < m; ++c) {
      float s = 0.f;
      for (int t = 0; t < k; ++t) s += (r < d ? Z[t * d + r] : 1.f) * T[t * m + c];
      float prior = X0 ? X0[r * m + c] : (x0_eye && r == c ? 1.f : 0.f);
      X[r * m + c] = s + (r < d ? reg : 1e-9f) * prior;
    }
  }
  return ek_chol_solve(G, X, n, m);
}

/* M8: apply DS map F ((d+1) x d) to z, out has d floats. */
static inline void ek_ds_apply(const float *z, const float *F, int d, float *out) {
  for (int j = 0; j < d; ++j) {
    float s = F[d * d + j];
    for (int i = 0; i < d; ++i) s += z[i] * F[i * d + j];
    out[j] = s;
  }
}

/* Streaming feature extractor, 8 channels, mirrors features.StreamingExtractor. */
#define EK_CH 8
typedef struct {
  float base_sum[EK_CH], r0[EK_CH], rmin[EK_CH], prev[EK_CH];
  float y[3][EK_CH], ymax[3][EK_CH], ymin[3][EK_CH];
  uint32_t base_n;
  float t_base;
  uint8_t started, have_r0;
} EkExtractor;

static inline void ek_ext_init(EkExtractor *e, float t_base) {
  for (int c = 0; c < EK_CH; ++c) {
    e->base_sum[c] = 0.f; e->r0[c] = 0.f; e->rmin[c] = INFINITY; e->prev[c] = 0.f;
    for (int k = 0; k < 3; ++k) { e->y[k][c] = 0.f; e->ymax[k][c] = -INFINITY; e->ymin[k][c] = INFINITY; }
  }
  e->base_n = 0; e->t_base = t_base; e->started = 0; e->have_r0 = 0;
}

static inline void ek_ext_push(EkExtractor *e, float t, const float *r) {
  static const float A[3] = {0.001f, 0.01f, 0.1f};
  if (t < e->t_base) {
    for (int c = 0; c < EK_CH; ++c) e->base_sum[c] += RD(r + c);
    e->base_n++;
  } else if (e->base_n && !e->have_r0) {
    for (int c = 0; c < EK_CH; ++c) e->r0[c] = e->base_sum[c] / (float)e->base_n;
    e->have_r0 = 1;
  }
  for (int c = 0; c < EK_CH; ++c) {
    float rc = RD(r + c);
    if (e->started)
      for (int k = 0; k < 3; ++k) {
        float yv = (1.f - A[k]) * e->y[k][c] + A[k] * (rc - e->prev[c]);
        e->y[k][c] = yv;
        if (yv > e->ymax[k][c]) e->ymax[k][c] = yv;
        if (yv < e->ymin[k][c]) e->ymin[k][c] = yv;
      }
    e->prev[c] = rc;
    if (t >= e->t_base && rc < e->rmin[c]) e->rmin[c] = rc;
  }
  e->started = 1;
}

static inline void ek_ext_features(EkExtractor *e, float *f) {
  if (!e->have_r0) {
    float n = e->base_n ? (float)e->base_n : 1.f;
    for (int c = 0; c < EK_CH; ++c) e->r0[c] = e->base_sum[c] / n;
    e->have_r0 = 1;
  }
  for (int c = 0; c < EK_CH; ++c) {
    float dr = e->r0[c] - e->rmin[c];
    float *o = f + 8 * c;
    o[0] = dr; o[1] = dr / e->r0[c];
    for (int k = 0; k < 3; ++k) { o[2 + k] = e->ymax[k][c]; o[5 + k] = e->ymin[k][c]; }
  }
}
