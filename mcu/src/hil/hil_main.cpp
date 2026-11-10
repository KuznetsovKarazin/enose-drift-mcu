// E7 firmware (revision): complete compensation streams on the board,
// samples sent by the host over USB. The board runs exactly the kernels of
// E5 on every sample, keeps all method state in its own RAM, and returns its
// predictions; the host compares them with the float32 host replica.
// Also: DS fitted on the board and fused into the head, the streaming
// extractor over complete 600 s recordings, timing of the kernel work per
// frame, and the peak stack depth by stack painting.
#if defined(HIL_HOST)
#include "hil_host_shim.h"
#else
#include <Arduino.h>
#endif
#if defined(__AVR__)
#include <avr/pgmspace.h>
#define EK_FLASH PROGMEM
#define FRD(p) pgm_read_float(p)     // constants in flash
#define EK_BOARD "mega2560"
#define HIL_ARENA_FLOATS 1180
#elif defined(HIL_HOST)
#define EK_BOARD "host"
#define HIL_ARENA_FLOATS (129 * 129 + 129 * 6 + 2 * 129 + 8)
#elif defined(CONFIG_IDF_TARGET_ESP32C3)
#define EK_BOARD "esp32c3"
#define HIL_ARENA_FLOATS (129 * 129 + 129 * 6 + 2 * 129 + 8)
#else
#define EK_BOARD "esp32s3"
#define HIL_ARENA_FLOATS (129 * 129 + 129 * 6 + 2 * 129 + 8)
#endif
#ifndef FRD
#define FRD(p) (*(p))
#endif
// Kernel inputs (samples, raw rows) live in RAM here, so RD is a plain read.
#define RD(p) (*(p))
#define RD8(p) (*(p))
#if defined(__AVR__)
#define EK_HAVE_RLS128 0
#else
#define EK_HAVE_RLS128 1
#endif
#include "enose_kernels.h"
#include "enose_fixture.h"
#if defined(HIL_HOST)
#include "hil_host_shim.h"
#define OUT Serial
static int rd() { return Serial.read(); }
#elif !defined(__AVR__)
#include "usb_bench_serial.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
static UsbBenchSerial io;
#define OUT io
static int rd() { return io.read(); }
#else
#define OUT Serial
static int rd() { return Serial.available() ? Serial.read() : -1; }
#endif

static float arena[HIL_ARENA_FLOATS];
#if defined(__AVR__)
#define HIL_BUF_FLOATS 129
#define HIL_BAUD 1000000
#else
#define HIL_BUF_FLOATS (32 * 128)
#endif
static float xbuf[HIL_BUF_FLOATS];   // a whole frame is read before any computation
static float *xin = xbuf;

// ---------------------------------------------------------------- byte input
static uint8_t rbyte() { int c; while ((c = rd()) < 0) {} return (uint8_t)c; }
static void rbytes(void *dst, size_t n) { uint8_t *p = (uint8_t *)dst; for (size_t i = 0; i < n; ++i) p[i] = rbyte(); }

// ---------------------------------------------------------------- stack painting
#if defined(__AVR__)
extern char __heap_start, *__brkval;
static uint8_t *stack_low() { return (uint8_t *)(__brkval ? __brkval : &__heap_start) + 16; }
#elif defined(HIL_HOST)
static uint8_t *stack_low() { volatile uint8_t m; return (uint8_t *)&m - 32768; }
#else
static uint8_t *stack_low() { return (uint8_t *)pxTaskGetStackStart(NULL) + 64; }
#endif
static uint8_t *paint_top;
static void __attribute__((noinline)) paint() {
  volatile uint8_t marker; uint8_t *sp = (uint8_t *)&marker;
  paint_top = sp - 96;
  for (uint8_t *p = stack_low(); p < paint_top; ++p) *p = 0xA5;
}
static long __attribute__((noinline)) stack_peak() {   // bytes below the painting point that were written
  uint8_t *p = stack_low();
  while (p < paint_top && *p == 0xA5) ++p;
  return (long)(paint_top - p);
}

// ---------------------------------------------------------------- method state
struct Hil {
  uint8_t method, d, C, k;            // method: 0 static, 1 EWMA, 5 NLMS, 6 RLS, 7 block LS, 15 EWMA+NLMS, 8 DS fused
  float *W, *m, *v, *P, *pa, *Z, *T, *G, *X, *z;
  uint8_t *lab;
  const float *mu, *sd;
  float a, mu_step, beta, gamma, rho;
  uint32_t us_ref, us_obs, us_end, n_ref, n_obs, n_end;
} H;

static void reply(const char *tag, long v) { OUT.print(tag); OUT.print(','); OUT.println(v); }

static void init_method(uint8_t method, uint8_t d) {
  memset(&H, 0, sizeof(H));
  H.method = method; H.d = d; H.C = method == 8 ? 4 : 6;
  float *p = arena;
  const int n = d + 1;
  H.z = p; p += d;
  if (method == 8) {
    // DS fused: slave/master pair buffers (16 pairs), G, F, then the fused head.
    H.mu = FX_D2_MU; H.sd = FX_D2_SD; H.rho = FX_DS_RHO;
    H.W = p; p += n * 4;
    for (int i = 0; i < n * 4; ++i) H.W[i] = FRD(FX_D2_W + i);
    H.Z = p; p += 16 * d; H.T = p; p += 16 * d; H.G = p; p += n * n; H.X = p; p += n * d;
  } else {
    H.mu = d == 16 ? FX_D1_MU16 : FX_D1_MU128; H.sd = d == 16 ? FX_D1_SD16 : FX_D1_SD128;
    const float *W0 = d == 16 ? FX_D1_W16 : FX_D1_W128;
    H.W = p; p += n * 6;
    for (int i = 0; i < n * 6; ++i) H.W[i] = FRD(W0 + i);
    if (method == 1 || method == 15) {
      H.m = p; p += d; H.v = p; p += d;
      for (int i = 0; i < d; ++i) { H.m[i] = 0.f; H.v[i] = 1.f; }
      H.a = d == 16 ? FX_M1_ALPHA16 : FX_M1_ALPHA128;
    }
    if (method == 5 || method == 15) H.mu_step = d == 16 ? FX_M5_MU16 : FX_M5_MU128;
    if (method == 6) {
      H.beta = d == 16 ? FX_M6_FORGET16 : FX_M6_FORGET128;
      H.pa = p; p += n; H.P = p; p += n * n;
#if EK_HAVE_RLS128
      const float *P0 = d == 16 ? FX_P0_16 : FX_P0_128;
#else
      const float *P0 = FX_P0_16;
#endif
      for (int i = 0; i < n * n; ++i) H.P[i] = FRD(P0 + i);
    }
    if (method == 7) {
      H.gamma = FX_M7_GAMMA16;
      H.Z = p; p += 20 * d; H.T = p; p += 20 * 6; H.G = p; p += n * n; H.X = p; p += n * 6;
      H.lab = (uint8_t *)p; p += 6;   // 20 label bytes
    }
  }
  reply("INIT", (long)((p - arena) * sizeof(float)));
}

// ek_pre with the statistics read from flash: same operations in the same order.
static void hpre(const float *x, const float *mu, const float *sd, int d, float *z) {
  for (int i = 0; i < d; ++i) {
    float v = x[i];
    float s = v < 0.f ? -logf(1.f - v) : logf(1.f + v);
    z[i] = (s - FRD(mu + i)) / FRD(sd + i);
  }
}
static void pre(const float *x, float *z) { hpre(x, H.mu, H.sd, H.d, z); }

static void do_reference(uint8_t y) {
  uint32_t t0 = micros();
  pre(xin, H.z);
  if (H.m) ek_ewma(H.z, H.m, H.v, H.d, H.a);
  if (H.method == 5 || H.method == 15) ek_nlms(H.W, H.z, H.d, 6, y, H.mu_step);
  else if (H.method == 6) ek_rls(H.W, H.P, H.z, H.d, 6, y, H.beta, H.pa);
  else if (H.method == 7 && H.k < 20) {
    for (int i = 0; i < H.d; ++i) H.Z[H.k * H.d + i] = H.z[i];
    H.lab[H.k++] = y;
  }
  H.us_ref += micros() - t0; H.n_ref++;
}

static void do_end() {
  uint32_t t0 = micros();
  if (H.method == 7 && H.k) {
    const int n = H.d + 1;
    for (int s = 0; s < H.k; ++s) for (int c = 0; c < 6; ++c) H.T[s * 6 + c] = H.lab[s] == c ? 1.f : 0.f;
    if (ek_ridge(H.Z, H.T, H.k, H.d, 6, H.gamma, H.W, 0, H.G, H.X) == 0)
      for (int i = 0; i < n * 6; ++i) H.W[i] = H.X[i];
    H.k = 0;
  }
  if (H.method == 8 && H.k) {
    // fit F, then fuse: W' = F W_lin, bias row += w_bias; F and buffers are then free.
    const int d = H.d, n = d + 1;
    int ok = ek_ridge(H.Z, H.T, H.k, d, d, H.rho, 0, 1, H.G, H.X);
    if (ok == 0) {
      float *Wf = H.G;   // reuse G as output of the fusion
      for (int r = 0; r < n; ++r)
        for (int c = 0; c < 4; ++c) {
          float s = r == d ? H.W[d * 4 + c] : 0.f;
          for (int j = 0; j < d; ++j) s += H.X[r * d + j] * H.W[j * 4 + c];
          Wf[r * 4 + c] = s;
        }
      for (int i = 0; i < n * 4; ++i) H.W[i] = Wf[i];
    }
    H.k = 0;
  }
  H.us_end += micros() - t0; H.n_end++;
}

static uint8_t do_observe(const float *x) {
  uint32_t t0 = micros();
  pre(x, H.z);
  if (H.m) ek_ewma(H.z, H.m, H.v, H.d, H.a);
  uint8_t c = (uint8_t)ek_argmax(H.z, H.W, H.d, H.C, 0);
  H.us_obs += micros() - t0; H.n_obs++;
  return c;
}

static EkExtractor ext;
static uint32_t us_ext, n_ext;

void setup() {
#if defined(__AVR__) || defined(HIL_HOST)
  Serial.begin(1000000);
#else
  io.begin();
#endif
}

void loop() {
  int c = rd();
  if (c < 0) return;
  switch (c) {
    case 'H': OUT.print("READY,ENOSE_HIL_V1,"); OUT.println(EK_BOARD); break;
    case 'I': { uint8_t m = rbyte(), d = rbyte(); paint(); init_method(m, d); break; }
    case 'R': { uint8_t y = rbyte(); rbytes(xin, 4 * H.d); do_reference(y); OUT.println("r"); break; }
    case 'Q': {   // DS pair: slave then master raw features
      rbytes(xin, 4 * H.d);
      if (H.k < 16) {
        hpre(xin, H.mu, H.sd, H.d, H.Z + H.k * H.d);
        rbytes(xin, 4 * H.d);
        hpre(xin, H.mu, H.sd, H.d, H.T + H.k * H.d);
        H.k++;
      } else rbytes(xin, 4 * H.d);
      OUT.println("q"); break;
    }
    case 'E': do_end(); OUT.println("e"); break;
    case 'B': {   // n observations, one prediction digit per sample in a single line
      uint8_t n = rbyte(); char line[34];
      if (n > 32 || (int)n * H.d > HIL_BUF_FLOATS) { OUT.println("ERROR"); break; }
      rbytes(xbuf, 4 * (size_t)n * H.d);
      for (uint8_t i = 0; i < n; ++i) line[i] = (char)('0' + do_observe(xbuf + i * H.d));
      line[n] = 0; OUT.print("P,"); OUT.println(line); break;
    }
    case 'X': { float tb; rbytes(&tb, 4); paint(); ek_ext_init(&ext, tb); us_ext = n_ext = 0; OUT.println("x"); break; }
    case 'P': {   // n samples of (t, r[8])
      uint8_t n = rbyte();
      if ((int)n * 9 > HIL_BUF_FLOATS) { OUT.println("ERROR"); break; }
      rbytes(xbuf, 4 * 9 * (size_t)n);
      for (uint8_t i = 0; i < n; ++i) {
        const float *row = xbuf + 9 * i;
        uint32_t t0 = micros(); ek_ext_push(&ext, row[0], row + 1); us_ext += micros() - t0; n_ext++;
      }
      OUT.println("p"); break;
    }
    case 'F': {
      float f[64]; ek_ext_features(&ext, f);
      OUT.print("FEAT");
      for (int i = 0; i < 64; ++i) { OUT.print(','); OUT.print(f[i], 9); }
      OUT.println();
      OUT.print("EXTTIME,"); OUT.print(n_ext); OUT.print(','); OUT.println(us_ext);
      break;
    }
    case 'S':
      OUT.print("STATS,"); OUT.print(H.n_ref); OUT.print(','); OUT.print(H.us_ref); OUT.print(',');
      OUT.print(H.n_obs); OUT.print(','); OUT.print(H.us_obs); OUT.print(',');
      OUT.print(H.n_end); OUT.print(','); OUT.print(H.us_end); OUT.print(',');
      OUT.println(stack_peak());
      break;
    default: OUT.println("ERROR"); break;
  }
}
