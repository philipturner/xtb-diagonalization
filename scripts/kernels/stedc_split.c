// How much of Accelerate's xstedc (COMPZ = 'I', all eigenvectors of the tridiagonal matrix) is the last merge of
// divide and conquer, the step whose eigenvector product could be limited to the m occupied columns xTB keeps?
// For the tridiagonal matrices of sytrd.c on random symmetric ones (n = 412-776): xstedc on all n, and on the two
// halves of the rank-one tearing (n1 = n / 2 and n - n1, the subproblems of the top merge); top merge = all - halves.
// Median of 21.
//   clang -O3 -mcpu=apple-m1 stedc_split.c -framework Accelerate -o stedc_split
//   VECLIB_MAXIMUM_THREADS=1 ./stedc_split 412 448 619 776
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../xtb/src/mctc/lapack/sytrd.c"

extern void dstedc_(const char *, const lint *, double *, double *, double *, const lint *, double *, const lint *,
                    lint *, const lint *, lint *);
extern void sstedc_(const char *, const lint *, float *, float *, float *, const lint *, float *, const lint *, lint *,
                    const lint *, lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  for (int ai = 1; ai < argc; ai++) {
    lint n = atoi(argv[ai]), lda = n;
    for (int prec = 0; prec < 2; prec++) {
      size_t sz = prec ? 4 : 8;
      void *A, *Z, *d0, *e0, *d, *e, *tau, *wk, *work; lint *iwork, info;
      posix_memalign(&A, 16384, (size_t)n * n * sz); posix_memalign(&Z, 16384, (size_t)n * n * sz);
      d0 = malloc(n * sz); e0 = malloc(n * sz); d = malloc(n * sz); e = malloc(n * sz); tau = malloc(n * sz);
      wk = malloc((size_t)n * 32 * sz);
      lint lwork = 1 + 4 * n + n * n, liwork = 3 + 5 * n;
      work = malloc(lwork * sz); iwork = malloc(liwork * sizeof(lint));
      srand(n);
      for (lint j = 0; j < n; j++)
        for (lint i = 0; i <= j; i++) {
          double v = (double)rand() / RAND_MAX - 0.5;
          if (prec) { ((float *)A)[i + j * n] = v; ((float *)A)[j + i * n] = v; }
          else { ((double *)A)[i + j * n] = v; ((double *)A)[j + i * n] = v; }
        }
      if (prec) xtb_ssytrd_upper(&n, A, &lda, d0, e0, tau, wk); else xtb_dsytrd_upper(&n, A, &lda, d0, e0, tau, wk);
      lint sizes[3] = {n, n / 2, n - n / 2}, offs[3] = {0, 0, n / 2};
      double med[3];
      for (int s = 0; s < 3; s++) {
        double t[21];
        for (int r = -2; r < 21; r++) {
          memcpy(d, (char *)d0 + offs[s] * sz, sizes[s] * sz); memcpy(e, (char *)e0 + offs[s] * sz, sizes[s] * sz);
          double t0 = now();
          if (prec) sstedc_("I", &sizes[s], d, e, Z, &sizes[s], work, &lwork, iwork, &liwork, &info);
          else dstedc_("I", &sizes[s], d, e, Z, &sizes[s], work, &lwork, iwork, &liwork, &info);
          if (r >= 0) t[r] = 1e3 * (now() - t0);
        }
        qsort(t, 21, 8, cmp); med[s] = t[10];
      }
      double top = med[0] - med[1] - med[2];
      printf("n = %4d %s: stedc %.3f ms, halves %.3f + %.3f, top merge %.3f ms (%.0f%%)\n", n, prec ? "FP32" : "FP64",
             med[0], med[1], med[2], top, 100 * top / med[0]);
      free(A); free(Z); free(d0); free(e0); free(d); free(e); free(tau); free(wk); free(work); free(iwork);
    }
  }
  return 0;
}
