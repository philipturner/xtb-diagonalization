// The bulge chase's memory pattern under Instruments' CPU counters: after each predecessor (nothing, dgemm, dsyr2k, a
// 64 MB read), five pattern runs, each a Points of Interest signpost interval named "pattern" with the message
// "<predecessor> <run>". The counters per interval separate the slow (~32 ms) and fast (~19 ms) states (accel_after.c).
//   clang -O2 chase_trace.c -framework Accelerate -o chase_trace
//   VECLIB_MAXIMUM_THREADS=1 xcrun xctrace record --template build/memory-counters.tracetemplate \
//     --output build/chase.trace --launch -- ./chase_trace 1204 48
#include <os/log.h>
#include <os/signpost.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef int lint;
extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *, const double *,
                   const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dsyr2k_(const char *, const char *, const lint *, const lint *, const double *, const double *, const lint *,
                    const double *, const lint *, const double *, double *, const lint *);

// As in accel_after.c: per bulge, the column segments rows rs ..< rs + b of columns rs - b ..< rs + 2b, lda = n.
static double chase_pattern(double *M, int n, int b) {
  double acc = 0;
  for (int s = 0; s < n - 2; s += 1)
    for (int rs = s + 1; rs < n; rs += b) {
      int lo = rs - b < 0 ? 0 : rs - b, hi = rs + 2 * b > n ? n : rs + 2 * b, len = rs + b > n ? n - rs : b;
      for (int j = lo; j < hi; j++) {
        double *c = M + rs + (size_t)j * n;
        for (int i = 0; i < len; i++) c[i] = c[i] * 0.999999 + 1e-9;
        acc += c[0];
      }
    }
  return acc;
}

int main(int argc, char **argv) {
  lint n = argc > 1 ? atoi(argv[1]) : 1204, k = argc > 2 ? atoi(argv[2]) : 48;
  double *A = calloc((size_t)n * n, 8), *M = calloc((size_t)n * n, 8);
  double *X = malloc((size_t)n * k * 8), *Y = malloc((size_t)n * k * 8);
  for (size_t i = 0; i < (size_t)n * k; i++) { X[i] = (double)(i % 97) / 97 - 0.5; Y[i] = (double)(i % 89) / 89 - 0.5; }
  size_t big = (size_t)8 << 20; double *E = calloc(big, 8);
  double one = 1, mone = -1, sink = 0;
  os_log_t log = os_log_create("chase_trace", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
  const char *names[4] = {"none", "dgemm", "dsyr2k", "read64MB"};
  for (int round = 0; round < 6; round++)
    for (int kind = 0; kind < 4; kind++) {
      if (kind == 1) dgemm_("N", "T", &n, &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      if (kind == 2) dsyr2k_("L", "N", &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      if (kind == 3) for (size_t i = 0; i < big; i += 16) sink += E[i];
      for (int r = 0; r < 5; r++) {
        os_signpost_id_t id = os_signpost_id_generate(log);
        os_signpost_interval_begin(log, id, "pattern", "%{public}s %d", names[kind], r);
        sink += chase_pattern(M, n, k);
        os_signpost_interval_end(log, id, "pattern");
      }
    }
  if (sink == 12345) printf("%f\n", sink);
  return 0;
}
