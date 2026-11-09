/* E5 suite: each routine initialises its state outside the timed region, runs
 * one kernel over the fixture, writes its outputs and returns the elapsed
 * microseconds of the timed loop only. Shared by firmware and host build.
 * Needs: RD(), NOW_US(), enose_fixture.h. */
#pragma once
#if defined(__AVR__)
#define EK_HAVE_RLS128 0
#define EK_ARENA_FLOATS 1150
#else
#define EK_HAVE_RLS128 1
#define EK_ARENA_FLOATS (129 * 129 + 129 * 6 + 2 * 129 + 8)
#endif
#include "enose_kernels.h"
#include "enose_fixture.h"


static float ek_arena[EK_ARENA_FLOATS];

static void ek_copy(float *dst, const float *src, int n) { for (int i = 0; i < n; ++i) dst[i] = RD(src + i); }

#define EK_PRED_MAX 64
typedef struct { uint8_t pred[EK_PRED_MAX]; int n; float aux; uint32_t us; uint32_t calls; } EkResult;

/* Static head: pre + argmax per sample. */
static void ek_static(int d, EkResult *r) {
  const float *X = d == 16 ? FX_D1_X16 : FX_D1_X128;
  const float *W = d == 16 ? FX_D1_W16 : FX_D1_W128;
  const float *mu = d == 16 ? FX_D1_MU16 : FX_D1_MU128, *sd = d == 16 ? FX_D1_SD16 : FX_D1_SD128;
  int n = d == 16 ? FX_N16 : FX_N128;
  float *z = ek_arena;
  uint32_t t0 = NOW_US();
  for (int s = 0; s < n; ++s) { ek_pre(X + s * d, mu, sd, d, z); r->pred[s] = (uint8_t)ek_argmax(z, W, d, 6, 1); }
  r->us = NOW_US() - t0; r->n = n; r->calls = n; r->aux = 0.f;
}

/* M1: pre + EWMA + argmax per sample. */
static void ek_m1(int d, EkResult *r) {
  const float *X = d == 16 ? FX_D1_X16 : FX_D1_X128;
  const float *W = d == 16 ? FX_D1_W16 : FX_D1_W128;
  const float *mu = d == 16 ? FX_D1_MU16 : FX_D1_MU128, *sd = d == 16 ? FX_D1_SD16 : FX_D1_SD128;
  int n = d == 16 ? FX_N16 : FX_N128;
  float a = d == 16 ? FX_M1_ALPHA16 : FX_M1_ALPHA128;
  float *z = ek_arena, *m = z + d, *v = m + d;
  for (int i = 0; i < d; ++i) { m[i] = 0.f; v[i] = 1.f; }
  uint32_t t0 = NOW_US();
  for (int s = 0; s < n; ++s) {
    ek_pre(X + s * d, mu, sd, d, z); ek_ewma(z, m, v, d, a);
    r->pred[s] = (uint8_t)ek_argmax(z, W, d, 6, 1);
  }
  r->us = NOW_US() - t0; r->n = n; r->calls = n;
  float cs = 0.f; for (int i = 0; i < d; ++i) cs += m[i] + v[i];
  r->aux = cs;
}

/* M5 / M6: predict, then update with the label (each sample is a reference).
 * Timed loop covers pre + predict + update. */
static void ek_adapt(int d, int rls, EkResult *r) {
  const float *X = d == 16 ? FX_D1_X16 : FX_D1_X128;
  const float *W0 = d == 16 ? FX_D1_W16 : FX_D1_W128;
  const float *mu = d == 16 ? FX_D1_MU16 : FX_D1_MU128, *sd = d == 16 ? FX_D1_SD16 : FX_D1_SD128;
  const uint8_t *Y = d == 16 ? FX_D1_Y16 : FX_D1_Y128;
  int n = d == 16 ? FX_N16 : FX_N128, nw = (d + 1) * 6, np = (d + 1) * (d + 1);
  float *W = ek_arena, *z = W + nw, *pa = z + d, *P = pa + d + 1;
  ek_copy(W, W0, nw);
#if EK_HAVE_RLS128
  if (rls) ek_copy(P, d == 16 ? FX_P0_16 : FX_P0_128, np);
#else
  if (rls) ek_copy(P, FX_P0_16, np);
#endif
  float par = rls ? (d == 16 ? FX_M6_FORGET16 : FX_M6_FORGET128) : (d == 16 ? FX_M5_MU16 : FX_M5_MU128);
  uint32_t t0 = NOW_US();
  for (int s = 0; s < n; ++s) {
    ek_pre(X + s * d, mu, sd, d, z);
    r->pred[s] = (uint8_t)ek_argmax(z, W, d, 6, 0);
    int y = RD8(Y + s);
    if (rls) ek_rls(W, P, z, d, 6, y, par, pa); else ek_nlms(W, z, d, 6, y, par);
  }
  r->us = NOW_US() - t0; r->n = n; r->calls = n;
  float cs = 0.f; for (int i = 0; i < nw; ++i) cs += W[i];
  r->aux = cs;
}

/* M7: buffer FX_OLS_K references, refit once (timed), predict the rest. */
static void ek_ols(EkResult *r) {
  const int d = 16, k = FX_OLS_K, n = FX_N16, nw = (d + 1) * 6;
  float *W0 = ek_arena, *W = W0 + nw, *Z = W + nw, *T = Z + k * d, *G = T + k * 6, *z = G + (d + 1) * (d + 1);
  ek_copy(W0, FX_D1_W16, nw);
  for (int s = 0; s < k; ++s) {
    ek_pre(FX_D1_X16 + s * d, FX_D1_MU16, FX_D1_SD16, d, Z + s * d);
    for (int c = 0; c < 6; ++c) T[s * 6 + c] = RD8(FX_D1_Y16 + s) == c ? 1.f : 0.f;
  }
  uint32_t t0 = NOW_US();
  int ok = ek_ridge(Z, T, k, d, 6, FX_M7_GAMMA16, W0, 0, G, W);
  r->us = NOW_US() - t0; r->calls = 1;
  for (int s = k; s < n; ++s) {
    ek_pre(FX_D1_X16 + s * d, FX_D1_MU16, FX_D1_SD16, d, z);
    r->pred[s - k] = (uint8_t)ek_argmax(z, W, d, 6, 0);
  }
  r->n = n - k;
  float cs = 0.f; for (int i = 0; i < nw; ++i) cs += W[i];
  r->aux = ok == 0 ? cs : NAN;
}

/* M8: fit DS from FX_DS_K pairs (timed separately: fit, then apply+predict). */
static void ek_ds(EkResult *fit, EkResult *app) {
  const int d = 16, k = FX_DS_K, n = FX_DS_N, nf = (d + 1) * d;
  float *Zs = ek_arena, *Zm = Zs + k * d, *G = Zm + k * d, *F = G + (d + 1) * (d + 1),
        *z = F + nf, *u = z + d;
  for (int s = 0; s < k; ++s) {
    ek_pre(FX_DS_SLAVE + s * d, FX_D2_MU, FX_D2_SD, d, Zs + s * d);
    ek_pre(FX_DS_MASTER + s * d, FX_D2_MU, FX_D2_SD, d, Zm + s * d);
  }
  uint32_t t0 = NOW_US();
  int ok = ek_ridge(Zs, Zm, k, d, d, FX_DS_RHO, 0, 1, G, F);
  fit->us = NOW_US() - t0; fit->calls = 1; fit->n = 0;
  float cs = 0.f; for (int i = 0; i < nf; ++i) cs += F[i];
  fit->aux = ok == 0 ? cs : NAN;
  t0 = NOW_US();
  for (int s = 0; s < n; ++s) {
    ek_pre(FX_DS_EVAL + s * d, FX_D2_MU, FX_D2_SD, d, z);
    ek_ds_apply(z, F, d, u);
    app->pred[s] = (uint8_t)ek_argmax(u, FX_D2_W, d, 4, 1);
  }
  app->us = NOW_US() - t0; app->calls = n; app->n = n; app->aux = 0.f;
}

/* Streaming extractor over FX_EXT_N samples of one raw D2 trace. */
static void ek_extract(float *features, EkResult *r) {
  EkExtractor *e = (EkExtractor *)(void *)ek_arena;
  ek_ext_init(e, FX_EXT_TBASE);
  uint32_t t0 = NOW_US();
  for (int s = 0; s < FX_EXT_N; ++s) ek_ext_push(e, RD(FX_EXT_T + s), FX_EXT_R + s * EK_CH);
  r->us = NOW_US() - t0; r->calls = FX_EXT_N; r->n = 0;
  ek_ext_features(e, features);
  float cs = 0.f; for (int i = 0; i < 64; ++i) cs += features[i];
  r->aux = cs;
}
