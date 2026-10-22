// The 1-stage back-transformation, Z = Q Z with Q from the upper-triangle tridiagonal reduction (xormtr 'L', 'U', 'N',
// which xTB calls for the occupied eigenvectors): Accelerate's xormtr (xormql -> xlarft + xlarfb) against the same
// blocks applied by three gemm calls (WYTransform.swift's gemmWY: Accelerate's larfb is 2.2-4x slower than gemm on the
// bulge blocks, wy.c). Blocks of nb reflectors as in xormql: reflector i (1-based) has v(i) = 1 and v(1 .. i - 1) in
// A(1 .. i - 1, i + 1), zero below; block i .. i + ib - 1 acts on rows 1 .. i + ib - 1, blocks in increasing i;
// V explicit (mi x ib, unit entries, zeros below), T = larft('B', 'C') (lower), W = V^T Z, W = T W, Z -= V W.
// A from sytrd.c's xtb_dsytrd_upper on a random symmetric matrix; m = n / 2 columns of a random Z. Median of 21.
//   clang -O3 -mcpu=apple-m1 ormtr_gemm.c -framework Accelerate -o ormtr_gemm
//   VECLIB_MAXIMUM_THREADS=1 ./ormtr_gemm 412 448 619 776
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../xtb/src/mctc/lapack/sytrd.c"

extern void dormtr_(const char *, const char *, const char *, const lint *, const lint *, const double *, const lint *,
                    const double *, double *, const lint *, double *, const lint *, lint *);
extern void sormtr_(const char *, const char *, const char *, const lint *, const lint *, const float *, const lint *,
                    const float *, float *, const lint *, float *, const lint *, lint *);
extern void dlarft_(const char *, const char *, const lint *, const lint *, const double *, const lint *, const double *,
                    double *, const lint *);
extern void slarft_(const char *, const char *, const lint *, const lint *, const float *, const lint *, const float *,
                    float *, const lint *);
extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *, const double *,
                   const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void sgemm_(const char *, const char *, const lint *, const lint *, const lint *, const float *, const float *,
                   const lint *, const float *, const lint *, const float *, float *, const lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

// work: n * nb (V) + nb * nb (T) + 2 * nb * m (W, W2)
#define ORMTR_GEMM(NAME, T, larft, gemm)                                                                              \
static void NAME(lint n, lint m, const T *a, lint lda, const T *tau, T *z, lint ldz, lint nb, T *work, int mode) {             \
  lint k = n - 1;                                                                                                     \
  T *v = work, *t = v + (size_t)(n + 64) * nb, *w = t + nb * nb, *w2 = w + (size_t)nb * m, *vt = w2 + (size_t)nb * m; \
  lint al = 64 / sizeof(T);                                   \
  const T one = 1, zero = 0, mone = -1;                                                                               \
  for (lint i = 1; i <= k; i += nb) {                                                                                 \
    lint ib = k - i + 1 < nb ? k - i + 1 : nb, mi = i + ib - 1, ldv = mode & 1 ? (mi + al - 1) / al * al : mi;      \
    lint ldt = mode & 1 ? (ib + al - 1) / al * al : ib;                                                       \
    /* Reflector r = i + c (column c of the block) is column r + 1 of a: rows 1 .. r - 1, then 1 at row r,         */ \
    /* zeros below (rows of the block: 1 .. mi).                                                                    */ \
    for (lint c = 0; c < ib; c++) {                                                                                   \
      const T *src = a + (size_t)(i + c) * lda; T *dst = v + (size_t)c * ldv;                                          \
      lint top = i + c;                                                                                               \
      memcpy(dst, src, (top - 1) * sizeof(T));                                                                        \
      dst[top - 1] = 1;                                                                                               \
      for (lint r = top; r < mi; r++) dst[r] = 0;                                                                     \
    }                                                                                                                 \
    larft("B", "C", &mi, &ib, v, &ldv, tau + i - 1, t, &ib);                                                           \
    /* T lower triangular: clear above the diagonal for the dense gemm. */                                            \
    for (lint c = 1; c < ib; c++) for (lint r = 0; r < c; r++) t[r + c * ib] = 0;                                     \
    if (mode & 6) for (lint c = 0; c < ib; c++) for (lint r = 0; r < mi; r++) vt[c + r * ldt] = v[r + c * ldv];     \
    if (mode & 2) gemm("N", "N", &ib, &m, &mi, &one, vt, &ldt, z, &ldz, &zero, w, &ib);                              \
    else gemm("T", "N", &ib, &m, &mi, &one, v, &ldv, z, &ldz, &zero, w, &ib);                                        \
    gemm("N", "N", &ib, &m, &ib, &one, t, &ib, w, &ib, &zero, w2, &ib);                                               \
    if (mode & 4) gemm("T", "N", &mi, &m, &ib, &mone, vt, &ldt, w2, &ib, &one, z, &ldz);                             \
    else gemm("N", "N", &mi, &m, &ib, &mone, v, &ldv, w2, &ib, &one, z, &ldz);                                       \
  }                                                                                                                   \
}
ORMTR_GEMM(ormtr_gemm_d, double, dlarft_, dgemm_)
ORMTR_GEMM(ormtr_gemm_f, float, slarft_, sgemm_)

static void *page(size_t bytes) { void *p; posix_memalign(&p, 16384, bytes); memset(p, 0, bytes); return p; }

int main(int argc, char **argv) {
  lint nbs[] = {32, 64, 128};
  for (int ai = 1; ai < argc; ai++) {
    lint n = atoi(argv[ai]), m = n / 2, lda = n, info;
    size_t nn = (size_t)n * n;
    for (int prec = 0; prec < 2; prec++) {
      size_t sz = prec ? 4 : 8;
      void *A = page(nn * sz), *Z0 = page((size_t)n * m * sz), *Z = page((size_t)n * m * sz),
           *R = page((size_t)n * m * sz), *wk = page((size_t)n * 32 * sz), *d = page(n * sz), *e = page(n * sz),
           *tau = page(n * sz);
      lint lwork = n * 256; void *work = page(lwork * sz + (size_t)256 * 256 * sz + (size_t)4 * 256 * (m + n + 64) * sz);
      srand(n);
      for (lint j = 0; j < n; j++)
        for (lint i = 0; i <= j; i++) {
          double x = (double)rand() / RAND_MAX - 0.5;
          if (prec) { ((float *)A)[i + j * n] = x; ((float *)A)[j + i * n] = x; }
          else { ((double *)A)[i + j * n] = x; ((double *)A)[j + i * n] = x; }
        }
      for (size_t i = 0; i < (size_t)n * m; i++) {
        double x = (double)rand() / RAND_MAX - 0.5;
        if (prec) ((float *)Z0)[i] = x; else ((double *)Z0)[i] = x;
      }
      if (prec) xtb_ssytrd_upper(&n, A, &lda, d, e, tau, wk); else xtb_dsytrd_upper(&n, A, &lda, d, e, tau, wk);
      // reference result
      memcpy(R, Z0, (size_t)n * m * sz);
      if (prec) sormtr_("L", "U", "N", &n, &m, A, &lda, tau, R, &n, work, &lwork, &info);
      else dormtr_("L", "U", "N", &n, &m, A, &lda, tau, R, &n, work, &lwork, &info);
      if (!getenv("ONLY")) printf("n = %4d m = %4d %s:", n, m, prec ? "FP32" : "FP64");
      double base = 0;
      // ONLY=v (0 = xormtr, 1 + 8 * (nb == 64) + mode): run just that variant REPS times and print the seconds
      // (for 8 x 1 throughput: start 8 copies together).
      if (getenv("ONLY")) {
        int v = atoi(getenv("ONLY")), reps = getenv("REPS") ? atoi(getenv("REPS")) : 200;
        lint nbv = v == 0 ? 0 : (v - 1) / 8 == 0 ? 32 : 64; int mode = (v - 1) % 8;
        if (getenv("FP32") ? !prec : prec) continue;
        double t0 = now();
        for (int r = 0; r < reps; r++) {
          memcpy(Z, Z0, (size_t)n * m * sz);
          if (v == 0) {
            if (prec) sormtr_("L", "U", "N", &n, &m, A, &lda, tau, Z, &n, work, &lwork, &info);
            else dormtr_("L", "U", "N", &n, &m, A, &lda, tau, Z, &n, work, &lwork, &info);
          } else if (prec) ormtr_gemm_f(n, m, A, lda, tau, Z, n, nbv, work, mode);
          else ormtr_gemm_d(n, m, A, lda, tau, Z, n, nbv, work, mode);
        }
        printf("%.4f\n", (now() - t0) / reps * 1e3);
        continue;
      }
      for (int v = 0; v < 17; v++) {
        lint nbv = v == 0 ? 0 : (v - 1) / 8 == 0 ? 32 : 64; int mode = (v - 1) % 8;
        double t[21];
        for (int r = -2; r < 21; r++) {
          memcpy(Z, Z0, (size_t)n * m * sz);
          double t0 = now();
          if (v == 0) {
            if (prec) sormtr_("L", "U", "N", &n, &m, A, &lda, tau, Z, &n, work, &lwork, &info);
            else dormtr_("L", "U", "N", &n, &m, A, &lda, tau, Z, &n, work, &lwork, &info);
          } else if (prec) ormtr_gemm_f(n, m, A, lda, tau, Z, n, nbv, work, mode);
          else ormtr_gemm_d(n, m, A, lda, tau, Z, n, nbv, work, mode);
          if (r >= 0) t[r] = 1e3 * (now() - t0);
        }
        qsort(t, 21, 8, cmp);
        double err = 0;
        for (size_t i = 0; i < (size_t)n * m; i++) {
          double x = prec ? ((float *)Z)[i] - ((float *)R)[i] : ((double *)Z)[i] - ((double *)R)[i];
          if (fabs(x) > err) err = fabs(x);
        }
        if (v == 0) { base = t[10]; printf(" ormtr %.3f ms;", t[10]); }
        else printf(" %d/%d %.2fx%s", nbv, mode, base / t[10], err > (prec ? 1e-4 : 1e-12) ? "!" : "");
      }
      if (!getenv("ONLY")) printf("\n");
      free(A); free(Z0); free(Z); free(R); free(wk); free(d); free(e); free(tau); free(work);
    }
  }
  return 0;
}
