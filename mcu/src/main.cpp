// E5 firmware: drift-compensation kernels on Mega2560, ESP32-C3 and ESP32-S3.
// Command protocol and USB transport adapted from lut-kan-native (MIT).
#include <Arduino.h>
#if defined(__AVR__)
#include <avr/pgmspace.h>
#define EK_FLASH PROGMEM
#define RD(p) pgm_read_float(p)
#define RD8(p) pgm_read_byte(p)
#define EK_BOARD "mega2560"
#elif defined(CONFIG_IDF_TARGET_ESP32C3)
#define EK_BOARD "esp32c3"
#else
#define EK_BOARD "esp32s3"
#endif
#define NOW_US() ((uint32_t)micros())
#include "enose_suite.h"
#include "enose_expected.h"
#include "command_protocol.h"
#if defined(CONFIG_IDF_TARGET_ESP32C3) || defined(CONFIG_IDF_TARGET_ESP32S3)
#define EK_NATIVE_USB
#include "usb_bench_serial.h"
static UsbBenchSerial bench_usb;
#ifdef Serial
#undef Serial
#endif
#define Serial bench_usb
#endif

static const int REPS = 10;

static bool check(const char *name, const EkResult &r, const uint8_t *exp, float exp_aux) {
  int match = 0;
  for (int i = 0; i < r.n; ++i) match += r.pred[i] == RD8(exp + i);
  float tol = 1e-4f * fmaxf(1.f, fabsf(exp_aux));
  bool aux_ok = isfinite(r.aux) && fabsf(r.aux - exp_aux) <= tol;
  bool pass = match == r.n && aux_ok;
  Serial.print("ACCURACY,"); Serial.print(name); Serial.print(','); Serial.print(match); Serial.print(',');
  Serial.print(r.n); Serial.print(','); Serial.print(r.aux, 6); Serial.print(','); Serial.print(exp_aux, 6);
  Serial.print(','); Serial.println(pass ? "PASS" : "FAIL");
  return pass;
}

static void timing(const char *name, uint8_t rep, const EkResult &r) {
  Serial.print("TIME,"); Serial.print(name); Serial.print(','); Serial.print(rep); Serial.print(',');
  Serial.print(r.calls); Serial.print(','); Serial.println(r.us);
}

static void state(const char *name, long bytes) {
  Serial.print("STATE,"); Serial.print(name); Serial.print(','); Serial.println(bytes);
}

#define RUN(name, call, exp, aux)                         \
  do {                                                    \
    EkResult r; call;                                     \
    if (check(name, r, exp, aux))                         \
      for (int k = 0; k < REPS; ++k) { call; timing(name, k, r); delay(1); } \
  } while (0)

static void run_benchmark(const char *session) {
  Serial.println("ENOSE_BENCH_V1");
  Serial.print("SESSION,"); Serial.println(session);
  Serial.print("BOARD,"); Serial.println(EK_BOARD);
  Serial.print("CPU_HZ,"); Serial.println((unsigned long)F_CPU);
  Serial.print("ARENA_BYTES,"); Serial.println((unsigned long)sizeof(ek_arena));
  state("static16", 0); state("static128", 0);
  state("m1_16", 2 * 4 * 16 + 4); state("m1_128", 2 * 4 * 128 + 4);
  state("nlms16", 4); state("nlms128", 4);
  state("rls16", 4L * 17 * 17 + 4); state("rls128", EK_HAVE_RLS128 ? 4L * 129 * 129 + 4 : -1);
  state("ols16", (long)FX_OLS_K * (4 * 16 + 1)); state("ds16", 4L * 17 * 16); state("extract", (long)sizeof(EkExtractor));
  RUN("static16", ek_static(16, &r), EXP_STATIC16, EXP_STATIC16_AUX);
  RUN("static128", ek_static(128, &r), EXP_STATIC128, EXP_STATIC128_AUX);
  RUN("m1_16", ek_m1(16, &r), EXP_M1_16, EXP_M1_16_AUX);
  RUN("m1_128", ek_m1(128, &r), EXP_M1_128, EXP_M1_128_AUX);
  RUN("nlms16", ek_adapt(16, 0, &r), EXP_NLMS16, EXP_NLMS16_AUX);
  RUN("nlms128", ek_adapt(128, 0, &r), EXP_NLMS128, EXP_NLMS128_AUX);
  RUN("rls16", ek_adapt(16, 1, &r), EXP_RLS16, EXP_RLS16_AUX);
#if EK_HAVE_RLS128
  RUN("rls128", ek_adapt(128, 1, &r), EXP_RLS128, EXP_RLS128_AUX);
#else
  Serial.println("SKIP,rls128,does_not_fit_sram");
#endif
  RUN("ols16", ek_ols(&r), EXP_OLS16, EXP_OLS16_AUX);
  {
    EkResult fit, app; ek_ds(&fit, &app);
    bool ok = check("ds16_fit", fit, EXP_DS16_FIT, EXP_DS16_FIT_AUX) & check("ds16_apply", app, EXP_DS16_APPLY, EXP_DS16_APPLY_AUX);
    if (ok) for (int k = 0; k < REPS; ++k) { ek_ds(&fit, &app); timing("ds16_fit", k, fit); timing("ds16_apply", k, app); delay(1); }
  }
  {
    static float f[64]; EkResult r; ek_extract(f, &r);
    float gap = 0.f;
    for (int i = 0; i < 64; ++i) {
      float e = RD(EXP_FEATURES + i);
      gap = fmaxf(gap, fabsf(f[i] - e) / fmaxf(1.f, fabsf(e)));
    }
    bool pass = gap <= 1e-5f;
    Serial.print("ACCURACY,extract,"); Serial.print(pass ? 64 : 0); Serial.print(",64,"); Serial.print(gap, 9);
    Serial.print(",0,"); Serial.println(pass ? "PASS" : "FAIL");
    if (pass) for (int k = 0; k < REPS; ++k) { ek_extract(f, &r); timing("extract", k, r); delay(1); }
  }
  Serial.println("END_ENOSE_BENCH");
}

void setup() {
#if defined(EK_NATIVE_USB)
  bench_usb.begin();
#else
  Serial.begin(115200);
#endif
}

void loop() {
  static BenchCommand command; static char last_session[33] = {0}; int c;
#if defined(EK_NATIVE_USB)
  while ((c = bench_usb.read()) >= 0) {
#else
  while (Serial.available() && (c = Serial.read()) >= 0) {
#endif
    if (!command.push((char)c)) continue;
    if (command.hello()) { Serial.print("READY,ENOSE_BENCH_V1,"); Serial.println(EK_BOARD); }
    else if (const char *token = command.probe()) { Serial.print("ECHO,"); Serial.println(token); }
    else if (const char *session = command.session()) {
      if (strcmp(last_session, session) == 0) { Serial.print("ALREADY_RAN,"); Serial.println(session); }
      else { strcpy(last_session, session); Serial.print("ACCEPTED,"); Serial.println(last_session); run_benchmark(last_session); }
    } else Serial.println("ERROR,INVALID_COMMAND");
  }
  delay(1);
}
