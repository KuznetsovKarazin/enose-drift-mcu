/* Host reference build of the E5 suite: prints every routine's outputs as JSON. */
#include <stdio.h>
#define NOW_US() 0u
#include "enose_suite.h"

static void dump(const char *name, const EkResult *r) {
  printf("{\"routine\":\"%s\",\"aux\":%.9g,\"pred\":[", name, (double)r->aux);
  for (int i = 0; i < r->n; ++i) printf(i ? ",%d" : "%d", r->pred[i]);
  printf("]}\n");
}

int main(void) {
  EkResult r;
  ek_static(16, &r); dump("static16", &r);
  ek_static(128, &r); dump("static128", &r);
  ek_m1(16, &r); dump("m1_16", &r);
  ek_m1(128, &r); dump("m1_128", &r);
  ek_adapt(16, 0, &r); dump("nlms16", &r);
  ek_adapt(128, 0, &r); dump("nlms128", &r);
  ek_adapt(16, 1, &r); dump("rls16", &r);
  ek_adapt(128, 1, &r); dump("rls128", &r);
  ek_ols(&r); dump("ols16", &r);
  EkResult fit, app; ek_ds(&fit, &app); dump("ds16_fit", &fit); dump("ds16_apply", &app);
  float f[64]; ek_extract(f, &r);
  printf("{\"routine\":\"extract\",\"aux\":%.9g,\"features\":[", (double)r.aux);
  for (int i = 0; i < 64; ++i) printf(i ? ",%.9g" : "%.9g", (double)f[i]);
  printf("]}\n");
  return 0;
}
