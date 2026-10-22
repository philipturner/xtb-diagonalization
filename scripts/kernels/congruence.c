// The per-cycle congruence of xtb's eigensolve (eigensolve.F90 `congruence`, nb = 96): the upper triangle of
// Z = X^T H X, X = U^-1 upper triangular, by block columns j .. jj:
//   Z(1:jj, j:jj) = H(1:jj, j:jj) X(j:jj, j:jj)      trmm 'R', 'U', 'N', 'N'
//   Z(1:jj, j:jj) += H(1:jj, 1:j-1) X(1:j-1, j:jj)   gemm 'N', 'N'
//   Z(1:jj, j:jj) = X(1:jj, 1:jj)^T Z(1:jj, j:jj)    trmm 'L', 'U', 'T', 'N'
// and the final back-transformation trmm (C = X C, 'L', 'U', 'N', 'N', n x m). Each Accelerate call timed by kind,
// then alternatives for the transposed trmm, which xtb can set up once per single point (X is fixed across cycles):
//   L  X^T stored as a lower-triangular matrix, trmm 'L', 'L', 'N', 'N'
//   G  X^T stored dense (zeros above), gemm 'N', 'N' (twice the flops of the triangle)
// Median of 21; random H, X = inverse of a random well-conditioned upper triangle.
//   clang -O3 -mcpu=apple-m1 congruence.c -framework Accelerate -o congruence
//   VECLIB_MAXIMUM_THREADS=1 ./congruence 412 448 619 776
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int lint;
extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *, const double *,
                   const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const lint *, const lint *, const double *,
                   const double *, const lint *, double *, const lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static void *page(size_t bytes) { void *p; posix_memalign(&p, 16384, bytes); memset(p, 0, bytes); return p; }

static double acc[3];
#define TIMED(k, call) do { double t0_ = now(); call; acc[k] += now() - t0_; } while (0)

// variant 0: as xtb; 1: X^T lower (xl), trmm 'L','L','N'; 2: X^T dense (xd), gemm
static void congruence(int variant, lint n, const double *h, const double *x, const double *xl, const double *xd,
                       double *z) {
  const lint nb = 96; const double one = 1;
  for (lint j = 1; j <= n; j += nb) {
    lint jj = j + nb - 1 < n ? j + nb - 1 : n, k = jj - j + 1, jm1 = j - 1;
    for (lint c = j; c <= jj; c++) memcpy(z + (size_t)(c - 1) * n, h + (size_t)(c - 1) * n, jj * 8);
    TIMED(0, dtrmm_("R", "U", "N", "N", &jj, &k, &one, x + (j - 1) + (size_t)(j - 1) * n, &n, z + (size_t)(j - 1) * n, &n));
    if (jm1 > 0) TIMED(1, dgemm_("N", "N", &jj, &k, &jm1, &one, h, &n, x + (size_t)(j - 1) * n, &n, &one, z + (size_t)(j - 1) * n, &n));
    if (variant == 0) TIMED(2, dtrmm_("L", "U", "T", "N", &jj, &k, &one, x, &n, z + (size_t)(j - 1) * n, &n));
    else if (variant == 1) TIMED(2, dtrmm_("L", "L", "N", "N", &jj, &k, &one, xl, &n, z + (size_t)(j - 1) * n, &n));
    else {
      // Z block = X^T(1:jj, 1:jj) Zb: gemm into a copy (gemm can't run in place)
      static double *tmp; static lint cap;
      if (cap < jj * k) { free(tmp); tmp = page((size_t)jj * k * 8); cap = jj * k; }
      for (lint c = 0; c < k; c++) memcpy(tmp + (size_t)c * jj, z + (size_t)(j - 1 + c) * n, jj * 8);
      double zero = 0;
      TIMED(2, dgemm_("N", "N", &jj, &k, &jj, &one, xd, &n, tmp, &jj, &zero, z + (size_t)(j - 1) * n, &n));
    }
  }
}

int main(int argc, char **argv) {
  for (int ai = 1; ai < argc; ai++) {
    lint n = atoi(argv[ai]), m = n / 2;
    size_t nn = (size_t)n * n;
    double *h = page(nn * 8), *x = page(nn * 8), *xl = page(nn * 8), *xd = page(nn * 8), *z = page(nn * 8),
           *zr = page(nn * 8), *c = page((size_t)n * m * 8);
    srand(n);
    for (lint j = 0; j < n; j++)
      for (lint i = 0; i <= j; i++) { double v = (double)rand() / RAND_MAX - 0.5; h[i + j * n] = v; h[j + i * n] = v; }
    // X: upper triangular, unit-ish diagonal, small off-diagonal (like U^-1 of an overlap matrix)
    for (lint j = 0; j < n; j++)
      for (lint i = 0; i <= j; i++) x[i + j * n] = i == j ? 1 + 0.5 * rand() / RAND_MAX : 0.05 * ((double)rand() / RAND_MAX - 0.5);
    for (lint j = 0; j < n; j++)
      for (lint i = 0; i <= j; i++) { xl[j + i * n] = x[i + j * n]; xd[j + i * n] = x[i + j * n]; }
    for (size_t i = 0; i < (size_t)n * m; i++) c[i] = (double)rand() / RAND_MAX - 0.5;
    congruence(0, n, h, x, xl, xd, zr);
    printf("n = %4d:", n);
    for (int v = 0; v < 3; v++) {
      double t[21], parts[21][3];
      for (int r = -2; r < 21; r++) {
        memset(acc, 0, sizeof acc);
        double t0 = now();
        congruence(v, n, h, x, xl, xd, z);
        if (r >= 0) { t[r] = 1e3 * (now() - t0); for (int q = 0; q < 3; q++) parts[r][q] = 1e3 * acc[q]; }
      }
      double s[21]; memcpy(s, t, sizeof t); qsort(s, 21, 8, cmp);
      int med = 0; for (int r = 0; r < 21; r++) if (t[r] == s[10]) med = r;
      double err = 0;
      for (lint j = 0; j < n; j++) for (lint i = 0; i <= j; i++) { double d = fabs(z[i + j * n] - zr[i + j * n]); if (d > err) err = d; }
      printf("  %s %.3f ms (trmm R %.3f, gemm %.3f, %s %.3f; max|dZ| %.0e)", v == 0 ? "xtb" : v == 1 ? "L" : "G", t[med],
             parts[med][0], parts[med][1], v == 0 ? "trmm LUT" : v == 1 ? "trmm LLN" : "gemm", parts[med][2], err);
    }
    // back-transformation trmm, n x m
    double t[21]; const double one = 1;
    for (int r = -2; r < 21; r++) {
      double t0 = now();
      dtrmm_("L", "U", "N", "N", &n, &m, &one, x, &n, c, &n);
      if (r >= 0) t[r] = 1e3 * (now() - t0);
    }
    qsort(t, 21, 8, cmp);
    printf("  | back trmm LUN n x n/2 %.3f ms (%.0f GF/s)\n", t[10], (double)n * n * m / t[10] * 1e-6);
    free(h); free(x); free(xl); free(xd); free(z); free(zr); free(c);
  }
  return 0;
}
