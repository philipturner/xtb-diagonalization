// Would running the bulge chase's sweeps in groups (a wavefront) keep its data in L1? The chase's memory pattern
// (lower triangle, full storage, lda = n) without its arithmetic: per bulge (sweep s, index k, rows rs = s + 1 + k b),
// read and update the left block (rows rs ..< rs + b, columns rs - b ..< rs), the diagonal triangle (rows and columns
// rs ..< rs + b) and the block below (rows rs + b ..< rs + 2b, columns rs ..< rs + b). Sweep orders:
//   g = 1  sequential, as BulgeChasing.swift: each sweep's bulges in turn
//   g > 1  groups of g consecutive sweeps as a wavefront, sweep s + j trailing sweep s by 2j bulges (sweep s + 1's
//          bulge k shares an element with sweep s's bulge k + 2, so a lag of 2 is the least that keeps the result)
// Median of 5 runs (after one warm-up) per (precision, b, g).
//   clang -O2 wavefront.c -o wavefront && ./wavefront 1204
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

#define PATTERN(T)                                                                                                    \
static void bulge_##T(T *M, int n, int b, int rs) {                                                                   \
  int re = rs + b > n ? n : rs + b, lo = rs - b < 0 ? 0 : rs - b, hi = re + b > n ? n : re + b;                      \
  for (int j = lo; j < rs; j++) { T *c = M + (size_t)j * n; for (int i = rs; i < re; i++) c[i] = c[i] * (T)0.999999 + (T)1e-9; } \
  for (int j = rs; j < re; j++) { T *c = M + (size_t)j * n; for (int i = j; i < hi; i++) c[i] = c[i] * (T)0.999999 + (T)1e-9; } \
}                                                                                                                     \
static double run_##T(T *M, int n, int b, int g) {                                                                    \
  int sweeps = n - 2, kmax = n / b + 1;                                                                               \
  double t0 = now();                                                                                                  \
  for (int s0 = 0; s0 < sweeps; s0 += g)                                                                              \
    for (int t = 0; t < kmax + 2 * g; t++)                                                                            \
      for (int j = 0; j < g; j++) {                                                                                   \
        int s = s0 + j, k = t - 2 * j;                                                                                \
        if (s >= sweeps || k < 0) continue;                                                                           \
        int rs = s + 1 + k * b;                                                                                       \
        if (rs < n) bulge_##T(M, n, b, rs);                                                                           \
      }                                                                                                               \
  return 1e3 * (now() - t0);                                                                                          \
}
PATTERN(double)
PATTERN(float)

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 1204;
  double *Md = calloc((size_t)n * n, 8); float *Mf = calloc((size_t)n * n, 4);
  int bs[3] = {32, 40, 48}, gs[5] = {1, 2, 3, 4, 6};
  printf("n = %d, ms (median of 5); speed-up over g = 1 in brackets\n%-5s %3s", n, "", "b");
  for (int gi = 0; gi < 5; gi++) printf("      g = %d    ", gs[gi]);
  printf("\n");
  for (int p = 0; p < 2; p++)
    for (int bi = 0; bi < 3; bi++) {
      double t[5][5], base = 0;
      printf("%-5s %3d", p ? "FP32" : "FP64", bs[bi]);
      for (int gi = 0; gi < 5; gi++) {
        for (int r = -1; r < 5; r++) {
          double x = p ? run_float(Mf, n, bs[bi], gs[gi]) : run_double(Md, n, bs[bi], gs[gi]);
          if (r >= 0) t[gi][r] = x;
        }
        qsort(t[gi], 5, 8, cmp);
        if (gi == 0) base = t[0][2];
        printf("  %6.1f (%4.2fx)", t[gi][2], base / t[gi][2]);
      }
      printf("\n");
    }
  return 0;
}
