/* Non-inlined copies of the kernels, compiled only to read per-function stack
 * frames (-fstack-usage). Not linked into the firmware. */
#include <stdint.h>
#include <stddef.h>
#define static
#define inline __attribute__((noinline))
#include "enose_kernels.h"
