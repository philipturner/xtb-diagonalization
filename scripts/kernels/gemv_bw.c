// Does Accelerate's full matrix-vector product read FP32 matrices ~3x faster than FP64 ones at every size?
// (scripts/kernels/unsym.c, n = 448: sgemv ~160 GB/s, dgemv ~55 GB/s.)
//   clang -O3 -mcpu=apple-m1 gemv_bw.c -framework Accelerate -o gemv_bw
//   VECLIB_MAXIMUM_THREADS=1 ./gemv_bw
// For square n x n column-major matrices (page-aligned, lda = n): xgemv 'N' and 'T', Accelerate's xsymv (upper), and
// the NEON symv of sytrd.c. Median of enough calls for ~0.2 s per kernel, after warm-up; GB/s = bytes the kernel must
// read (n^2 * size for gemv, half that for symv) / time.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define xtb_dsytrd_upper ref_dsytrd_upper
#define xtb_ssytrd_upper ref_ssytrd_upper
#include "../../xtb/src/mctc/lapack/sytrd.c"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(void) {
  int sizes[] = {64, 128, 200, 256, 300, 412, 448, 512, 619, 776, 1024, 1300, 1600, 2048, 3000};
  int ns = sizeof sizes / sizeof sizes[0];
  printf("GB/s (time per call in us); symv reads the upper triangle only\n");
  printf("%5s %9s | %-15s %-15s %-15s %-15s | %-15s %-15s %-15s %-15s\n", "n", "MB FP64", "dgemv N", "dgemv T",
         "dsymv", "NEON dsymv", "sgemv N", "sgemv T", "ssymv", "NEON ssymv");
  for (int si = 0; si < ns; si++) {
    lint n = sizes[si], i1 = 1;
    size_t nn = (size_t)n * n;
    double *A = NULL, *x = malloc(n * 8), *y = malloc(n * 8);
    float *As = NULL, *xs = malloc(n * 4), *ys = malloc(n * 4);
    posix_memalign((void **)&A, 16384, nn * 8); posix_memalign((void **)&As, 16384, nn * 4);
    srand(1);
    for (size_t k = 0; k < nn; k++) { A[k] = rand() / (double)RAND_MAX - 0.5; As[k] = A[k]; }
    for (lint k = 0; k < n; k++) { x[k] = rand() / (double)RAND_MAX; xs[k] = x[k]; }
    printf("%5d %9.2f |", n, nn * 8 / 1e6);
    for (int kern = 0; kern < 8; kern++) {
      double one = 1, zero = 0; float ones = 1, zeros = 0;
      int fp32 = kern >= 4, k4 = kern % 4;
      double bytes = (double)nn * (fp32 ? 4 : 8) * (k4 >= 2 ? 0.5 : 1);
      int reps = (int)(0.2 / (bytes / 50e9)); if (reps < 21) reps = 21; if (reps > 20001) reps = 20001;
      double *t = malloc(reps * 8);
      for (int r = -reps / 10 - 3; r < reps; r++) {
        double t0 = now();
        if (!fp32) switch (k4) {
          case 0: dgemv_("N", &n, &n, &one, A, &n, x, &i1, &zero, y, &i1); break;
          case 1: dgemv_("T", &n, &n, &one, A, &n, x, &i1, &zero, y, &i1); break;
          case 2: dsymv_("U", &n, &one, A, &n, x, &i1, &zero, y, &i1); break;
          case 3: symv(n, one, A, n, x, y); break;
        } else switch (k4) {
          case 0: sgemv_("N", &n, &n, &ones, As, &n, xs, &i1, &zeros, ys, &i1); break;
          case 1: sgemv_("T", &n, &n, &ones, As, &n, xs, &i1, &zeros, ys, &i1); break;
          case 2: ssymv_("U", &n, &ones, As, &n, xs, &i1, &zeros, ys, &i1); break;
          case 3: symv_f(n, ones, As, n, xs, ys); break;
        }
        if (r >= 0) t[r] = now() - t0;
      }
      qsort(t, reps, 8, cmp);
      double med = t[reps / 2];
      printf(" %6.1f (%6.1f)", bytes / med / 1e9, 1e6 * med);
      if (kern == 3) printf(" |");
      free(t);
    }
    printf("\n");
    free(A); free(As); free(x); free(y); free(xs); free(ys);
  }
  return 0;
}
