// Can the 'l' stage-1 option (lower-triangle update) keep its saving without leaving the chase in the slow state
// (LOG.md: after Accelerate's dsyr2k/dsymm, L2-bound code runs up to 1.7x slower for ~30-150 ms; after dgemm not)?
// The trailing update A -= V W^T + W V^T (n x n, rank 2k) three ways, timed: dsyr2k on the lower triangle ('l' today),
// two full dgemm calls ('tcs'), and the lower triangle by dgemm per block column of nb (half the work, like dsyr2k).
// Then the chase's memory pattern (lower triangle, scripts/kernels/wavefront.c) three times right after each, and after
// dsyr2k followed by a candidate reset: a 12 MB copy, a small dgemm.
//   clang -O2 lower_update.c -framework Accelerate -o lower_update
//   VECLIB_MAXIMUM_THREADS=1 ./lower_update 1204 48
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int lint;
extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *, const double *,
                   const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dsyr2k_(const char *, const char *, const lint *, const lint *, const double *, const double *, const lint *,
                    const double *, const lint *, const double *, double *, const lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static lint n, k;
static double *A, *V, *W, *M, *C1, *C2;

static void syr2k(void) { double mone = -1, one = 1; dsyr2k_("L", "N", &n, &k, &mone, V, &n, W, &n, &one, A, &n); }
static void two_gemm(void) {
  double mone = -1, one = 1;
  dgemm_("N", "T", &n, &n, &k, &mone, V, &n, W, &n, &one, A, &n);
  dgemm_("N", "T", &n, &n, &k, &mone, W, &n, V, &n, &one, A, &n);
}
// Lower triangle by block columns: rows j ..< n of columns j ..< j + nb (the diagonal block in full, its upper part
// is never read), two dgemm calls each.
static void lower_gemm(lint nb) {
  double mone = -1, one = 1;
  for (lint j = 0; j < n; j += nb) {
    lint w = n - j < nb ? n - j : nb, h = n - j;
    dgemm_("N", "T", &h, &w, &k, &mone, V + j, &n, W + j, &n, &one, A + j + (size_t)j * n, &n);
    dgemm_("N", "T", &h, &w, &k, &mone, W + j, &n, V + j, &n, &one, A + j + (size_t)j * n, &n);
  }
}
static double pattern(lint b) {
  double t0 = now();
  for (lint s = 0; s < n - 2; s++)
    for (lint rs = s + 1; rs < n; rs += b) {
      lint re = rs + b > n ? n : rs + b, lo = rs - b < 0 ? 0 : rs - b, hi = re + b > n ? n : re + b;
      for (lint j = lo; j < rs; j++) { double *c = M + (size_t)j * n; for (lint i = rs; i < re; i++) c[i] = c[i] * 0.999999 + 1e-9; }
      for (lint j = rs; j < re; j++) { double *c = M + (size_t)j * n; for (lint i = j; i < hi; i++) c[i] = c[i] * 0.999999 + 1e-9; }
    }
  return 1e3 * (now() - t0);
}

int main(int argc, char **argv) {
  n = argc > 1 ? atoi(argv[1]) : 1204; k = argc > 2 ? atoi(argv[2]) : 48;
  size_t nn = (size_t)n * n;
  A = calloc(nn, 8); M = calloc(nn, 8); C1 = calloc(nn, 8); C2 = calloc(nn, 8);
  V = malloc((size_t)n * k * 8); W = malloc((size_t)n * k * 8);
  for (size_t i = 0; i < (size_t)n * k; i++) { V[i] = (double)(i % 97) / 97 - 0.5; W[i] = (double)(i % 89) / 89 - 0.5; }
  printf("n = %d, rank 2k = %d\n", n, 2 * k);

  // Part 1: cost of the update (median of 9)
  const char *un[5] = {"dsyr2k (l)", "2 x dgemm (tcs)", "lower dgemm nb=128", "lower dgemm nb=256", "lower dgemm nb=512"};
  for (int v = 0; v < 5; v++) {
    double t[10];
    for (int r = -1; r < 9; r++) {
      double t0 = now();
      if (v == 0) syr2k(); else if (v == 1) two_gemm(); else lower_gemm(v == 2 ? 128 : v == 3 ? 256 : 512);
      if (r >= 0) t[r] = 1e3 * (now() - t0);
    }
    qsort(t, 9, 8, cmp);
    printf("  update %-20s %7.2f ms\n", un[v], t[4]);
  }

  // Part 2: the chase's memory pattern right after each (three runs; rounds alternate the predecessors)
  const char *pn[7] = {"nothing", "dsyr2k", "2 x dgemm", "lower dgemm 256", "dsyr2k + 12 MB copy", "dsyr2k + small dgemm",
                       "dsyr2k + 3 ms wait"};
  double pt[7][8][3];
  for (int round = 0; round < 8; round++)
    for (int v = 0; v < 7; v++) {
      if (v == 1 || v >= 4) syr2k();
      if (v == 2) two_gemm();
      if (v == 3) lower_gemm(256);
      if (v == 4) memcpy(C2, C1, nn * 8);
      if (v == 5) { lint m = 256, kk = 48; double one = 1, zero = 0; dgemm_("N", "T", &m, &m, &kk, &one, V, &n, W, &n, &zero, C2, &m); }
      if (v == 6) { double t0 = now(); while (now() - t0 < 0.003) {} }
      for (int r = 0; r < 3; r++) pt[v][round][r] = pattern(k);
    }
  printf("chase memory pattern after each (median over 8 rounds of run 1 / run 2 / run 3, ms):\n");
  for (int v = 0; v < 7; v++) {
    printf("  after %-22s", pn[v]);
    for (int r = 0; r < 3; r++) {
      double x[8]; for (int q = 0; q < 8; q++) x[q] = pt[v][q][r];
      qsort(x, 8, 8, cmp); printf(" %6.2f", x[4]);
    }
    printf("\n");
  }
  return 0;
}
