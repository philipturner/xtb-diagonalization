// What do Accelerate's symmetric BLAS3 kernels leave running after they return? The bulge chase right after one dsyr2k
// or dsymm call is ~25% slower, after dgemm it isn't (eigen-bench probe, LOG.md). Per kernel (stage-1 sizes, n x n
// trailing block, k = b columns): the process's thread count before and after, and the CPU time the process uses during
// the 50 ms sleep that follows (near 0 if nothing keeps running). Part 2: a chase-like workload (short FMA loops on
// L1-resident vectors of 48 doubles) timed right after each kernel. (An explicit AMX clr, 0x00201221 after
// github.com/corsix/amx, right after dsyr2k raised SIGILL: the kernel had already disabled the AMX unit.) Part 3: the
// chase's memory access pattern (no arithmetic to speak of) on a matrix of its own, timed right after each kernel.
//   clang -O2 accel_after.c -framework Accelerate -o accel_after
//   VECLIB_MAXIMUM_THREADS=1 ./accel_after 1204 48
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

typedef int lint;
extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *, const double *,
                   const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dsymm_(const char *, const char *, const lint *, const lint *, const double *, const double *, const lint *,
                   const double *, const lint *, const double *, double *, const lint *);
extern void dsyr2k_(const char *, const char *, const lint *, const lint *, const double *, const double *, const lint *,
                    const double *, const lint *, const double *, double *, const lint *);

static double cpu(void) {
  struct rusage r; getrusage(RUSAGE_SELF, &r);
  return r.ru_utime.tv_sec + 1e-6 * r.ru_utime.tv_usec + r.ru_stime.tv_sec + 1e-6 * r.ru_stime.tv_usec;
}
static int threads(void) {
  thread_act_array_t list; mach_msg_type_number_t count;
  task_threads(mach_task_self(), &list, &count);
  for (mach_msg_type_number_t i = 0; i < count; i++) mach_port_deallocate(mach_task_self(), list[i]);
  vm_deallocate(mach_task_self(), (vm_address_t)list, count * sizeof(thread_act_t));
  return (int)count;
}

// ~2 GFLOP/s-scale short-vector work like the chase's inner loops: y += a x, dot products, 48 elements each.
static double workload(void) {
  static double x[48], y[48];
  double acc = 0;
  for (int i = 0; i < 48; i++) { x[i] = 1e-3 * i; y[i] = 1.0; }
  for (int rep = 0; rep < 200000; rep++) {
    double a = 1e-9 * (rep & 7), d = 0;
    for (int i = 0; i < 48; i++) { y[i] += a * x[i]; d += x[i] * y[i]; }
    acc += d;
  }
  return acc;
}
// The chase's memory pattern without its arithmetic: for every bulge (sweep s, k = 0, 1, ...; rows rs = s + 1 + k b),
// read and update the column segments rows rs ..< rs + b of columns rs - b ..< rs + 2b of an n x n matrix (lda = n).
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
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  lint n = argc > 1 ? atoi(argv[1]) : 1204, k = argc > 2 ? atoi(argv[2]) : 48;
  double *A = calloc((size_t)n * n, 8), *X = malloc((size_t)n * k * 8), *Y = malloc((size_t)n * k * 8);
  for (size_t i = 0; i < (size_t)n * k; i++) { X[i] = (double)(i % 97) / 97 - 0.5; Y[i] = (double)(i % 89) / 89 - 0.5; }
  for (lint i = 0; i < n; i++) A[i + (size_t)i * n] = 1;
  double one = 1, mone = -1, zero = 0;
  const char *names[4] = {"none", "dgemm", "dsymm", "dsyr2k"};
  printf("n = %d, k = %d\n", n, k);
  for (int round = 0; round < 3; round++)
    for (int kind = 0; kind < 4; kind++) {
      int t0 = threads();
      if (kind == 1) dgemm_("N", "T", &n, &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      if (kind == 2) dsymm_("L", "L", &n, &k, &one, A, &n, X, &n, &zero, Y, &n);
      if (kind == 3) dsyr2k_("L", "N", &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      int t1 = threads();
      double c0 = cpu();
      usleep(50000);
      double c1 = cpu();
      int t2 = threads();
      printf("  round %d %-7s threads %d -> %d -> %d after 50 ms; CPU time during the sleep %6.1f ms\n", round,
             names[kind], t0, t1, t2, 1e3 * (c1 - c0));
    }

  // Part 2
  const char *names2[4] = {"none", "dgemm", "dsymm", "dsyr2k"};
  double t[4][15]; double sink = 0;
  for (int round = 0; round < 15; round++)
    for (int kind = 0; kind < 4; kind++) {
      if (kind == 1) dgemm_("N", "T", &n, &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      if (kind == 2) dsymm_("L", "L", &n, &k, &one, A, &n, X, &n, &zero, Y, &n);
      if (kind == 3) dsyr2k_("L", "N", &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      double t0 = now(); sink += workload(); t[kind][round] = 1e3 * (now() - t0);
    }
  printf("chase-like workload right after each kernel (median / min / max of 15, ms):\n");
  for (int kind = 0; kind < 4; kind++) {
    qsort(t[kind], 15, 8, cmp);
    printf("  after %-13s %7.2f %7.2f %7.2f\n", names2[kind], t[kind][7], t[kind][0], t[kind][14]);
  }
  // Part 3: the chase's memory pattern on its own n x n matrix (not touched by the kernels)
  double *M = calloc((size_t)n * n, 8), tm[4][9];
  for (int round = 0; round < 9; round++)
    for (int kind = 0; kind < 4; kind++) {
      if (kind == 1) dgemm_("N", "T", &n, &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      if (kind == 2) dsymm_("L", "L", &n, &k, &one, A, &n, X, &n, &zero, Y, &n);
      if (kind == 3) dsyr2k_("L", "N", &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      double t0 = now(); sink += chase_pattern(M, n, k); tm[kind][round] = 1e3 * (now() - t0);
    }
  printf("chase memory pattern right after each kernel (median / min / max of 9, ms):\n");
  for (int kind = 0; kind < 4; kind++) {
    qsort(tm[kind], 9, 8, cmp);
    printf("  after %-13s %7.2f %7.2f %7.2f\n", names2[kind], tm[kind][4], tm[kind][0], tm[kind][8]);
  }
  // Part 4: five consecutive pattern runs after each kernel, and after evicting the caches (64 MB read)
  printf("five consecutive pattern runs (ms):\n");
  size_t big = (size_t)8 << 20; double *E = calloc(big, 8);
  const char *names4[5] = {"none", "dgemm", "dsymm", "dsyr2k", "64 MB read"};
  for (int round = 0; round < 2; round++)
    for (int kind = 0; kind < 5; kind++) {
      if (kind == 1) dgemm_("N", "T", &n, &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      if (kind == 2) dsymm_("L", "L", &n, &k, &one, A, &n, X, &n, &zero, Y, &n);
      if (kind == 3) dsyr2k_("L", "N", &n, &k, &mone, X, &n, Y, &n, &one, A, &n);
      if (kind == 4) for (size_t i = 0; i < big; i += 16) sink += E[i];
      printf("  after %-11s", names4[kind]);
      for (int r = 0; r < 5; r++) { double t0 = now(); sink += chase_pattern(M, n, k); printf(" %6.2f", 1e3 * (now() - t0)); }
      printf("\n");
    }
  if (sink == 12345) printf("%f\n", sink);
  return 0;
}
