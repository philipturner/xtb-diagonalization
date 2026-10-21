// Prototype: LAPACK's blocked dsytrd (UPLO = 'U'; dlatrd panels + dsyr2k, dsytd2 for the last block), ported as is,
// with only the symmetric matrix-vector product replaced by a hand-vectorized kernel (symv_upper). Same output
// format as dsytrd (Householder vectors in A, tau), so dormtr back-transforms it.
//   sytrd_neon HS.bin [REPS]   (HS.bin from diag/dump)
// Times Accelerate dsytrd_ vs this port with Accelerate's dsymv vs this port with symv_upper, and checks eigenvalues
// (dsterf) and the back-transformation (dstedc + dormtr: max |H' C - C diag(w)|).
//   clang -O3 -mcpu=apple-m1 sytrd_neon.c -framework Accelerate -o sytrd_neon
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef int lint;
extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dtrtri_(const char *, const char *, const lint *, double *, const lint *, lint *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const lint *, const lint *, const double *,
                   const double *, const lint *, double *, const lint *);
extern void dsytrd_(const char *, const lint *, double *, const lint *, double *, double *, double *, double *,
                    const lint *, lint *);
extern void dsterf_(const lint *, double *, double *, lint *);
extern void dstedc_(const char *, const lint *, double *, double *, double *, const lint *, double *, const lint *,
                    lint *, const lint *, lint *);
extern void dormtr_(const char *, const char *, const char *, const lint *, const lint *, const double *,
                    const lint *, const double *, double *, const lint *, double *, const lint *, lint *);
extern void dgemv_(const char *, const lint *, const lint *, const double *, const double *, const lint *,
                   const double *, const lint *, const double *, double *, const lint *);
extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *,
                   const double *, const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dsymv_(const char *, const lint *, const double *, const double *, const lint *, const double *,
                   const lint *, const double *, double *, const lint *);
extern void dsyr2k_(const char *, const char *, const lint *, const lint *, const double *, const double *,
                    const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dsyr2_(const char *, const lint *, const double *, const double *, const lint *, const double *,
                   const lint *, double *, const lint *);
extern void dlarfg_(const lint *, double *, double *, const lint *, double *);
extern double ddot_(const lint *, const double *, const lint *, const double *, const lint *);
extern void daxpy_(const lint *, const double *, const double *, const lint *, double *, const lint *);
extern void dscal_(const lint *, const double *, double *, const lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

// y = alpha * A x, A symmetric, upper triangle, n x n, column-major with lda
static void symv_upper(int n, double alpha, const double *A, int lda, const double *x, double *y) {
  for (int i = 0; i < n; i++) y[i] = 0.0;
  int j0 = 0;
  for (; j0 + 4 <= n; j0 += 4) {
    const double *c0 = A + (size_t)j0 * lda, *c1 = c0 + lda, *c2 = c1 + lda, *c3 = c2 + lda;
    float64x2_t x0 = vdupq_n_f64(x[j0]), x1 = vdupq_n_f64(x[j0 + 1]), x2 = vdupq_n_f64(x[j0 + 2]),
                x3 = vdupq_n_f64(x[j0 + 3]);
    float64x2_t s0 = vdupq_n_f64(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 2 <= j0; i += 2) {
      float64x2_t a0 = vld1q_f64(c0 + i), a1 = vld1q_f64(c1 + i), a2 = vld1q_f64(c2 + i), a3 = vld1q_f64(c3 + i);
      float64x2_t xi = vld1q_f64(x + i), yi = vld1q_f64(y + i);
      yi = vfmaq_f64(yi, a0, x0); yi = vfmaq_f64(yi, a1, x1); yi = vfmaq_f64(yi, a2, x2); yi = vfmaq_f64(yi, a3, x3);
      vst1q_f64(y + i, yi);
      s0 = vfmaq_f64(s0, a0, xi); s1 = vfmaq_f64(s1, a1, xi); s2 = vfmaq_f64(s2, a2, xi); s3 = vfmaq_f64(s3, a3, xi);
    }
    double t0 = vaddvq_f64(s0), t1 = vaddvq_f64(s1), t2 = vaddvq_f64(s2), t3 = vaddvq_f64(s3);
    for (; i < j0; i++) {
      y[i] += c0[i] * x[j0] + c1[i] * x[j0 + 1] + c2[i] * x[j0 + 2] + c3[i] * x[j0 + 3];
      t0 += c0[i] * x[i]; t1 += c1[i] * x[i]; t2 += c2[i] * x[i]; t3 += c3[i] * x[i];
    }
    y[j0] += t0; y[j0 + 1] += t1; y[j0 + 2] += t2; y[j0 + 3] += t3;
    for (int c = 0; c < 4; c++) {
      const double *col = A + (size_t)(j0 + c) * lda;
      for (int r = 0; r < c; r++) { y[j0 + r] += col[j0 + r] * x[j0 + c]; y[j0 + c] += col[j0 + r] * x[j0 + r]; }
      y[j0 + c] += col[j0 + c] * x[j0 + c];
    }
  }
  for (; j0 < n; j0++) {
    const double *col = A + (size_t)j0 * lda;
    double s = 0;
    for (int i = 0; i < j0; i++) { y[i] += col[i] * x[j0]; s += col[i] * x[i]; }
    y[j0] += s + col[j0] * x[j0];
  }
  if (alpha != 1.0) for (int i = 0; i < n; i++) y[i] *= alpha;
}

static int use_neon = 1;
static void symv(lint n, double alpha, const double *A, lint lda, const double *x, double *y) {
  if (use_neon) { symv_upper(n, alpha, A, lda, x, y); return; }
  double zero = 0; lint one = 1;
  dsymv_("U", &n, &alpha, A, &lda, x, &one, &zero, y, &one);
}

#define A_(i, j) a[((i) - 1) + (size_t)((j) - 1) * lda]
#define W_(i, j) w[((i) - 1) + (size_t)((j) - 1) * ldw]

// LAPACK dlatrd, UPLO = 'U'
static void latrd_u(lint n, lint nb, double *a, lint lda, double *e, double *tau, double *w, lint ldw) {
  const double one = 1, mone = -1, zero = 0, half = 0.5; const lint i1 = 1;
  for (lint i = n; i >= n - nb + 1; i--) {
    lint iw = i - n + nb;
    if (i < n) {
      lint m = n - i;
      dgemv_("N", &i, &m, &mone, &A_(1, i + 1), &lda, &W_(i, iw + 1), &ldw, &one, &A_(1, i), &i1);
      dgemv_("N", &i, &m, &mone, &W_(1, iw + 1), &ldw, &A_(i, i + 1), &lda, &one, &A_(1, i), &i1);
    }
    if (i > 1) {
      lint im1 = i - 1;
      dlarfg_(&im1, &A_(i - 1, i), &A_(1, i), &i1, &tau[i - 2]);
      e[i - 2] = A_(i - 1, i);
      A_(i - 1, i) = one;
      symv(im1, one, a, lda, &A_(1, i), &W_(1, iw));
      if (i < n) {
        lint m = n - i;
        dgemv_("T", &im1, &m, &one, &W_(1, iw + 1), &ldw, &A_(1, i), &i1, &zero, &W_(i + 1, iw), &i1);
        dgemv_("N", &im1, &m, &mone, &A_(1, i + 1), &lda, &W_(i + 1, iw), &i1, &one, &W_(1, iw), &i1);
        dgemv_("T", &im1, &m, &one, &A_(1, i + 1), &lda, &A_(1, i), &i1, &zero, &W_(i + 1, iw), &i1);
        dgemv_("N", &im1, &m, &mone, &W_(1, iw + 1), &ldw, &W_(i + 1, iw), &i1, &one, &W_(1, iw), &i1);
      }
      dscal_(&im1, &tau[i - 2], &W_(1, iw), &i1);
      double alpha = -half * tau[i - 2] * ddot_(&im1, &W_(1, iw), &i1, &A_(1, i), &i1);
      daxpy_(&im1, &alpha, &A_(1, i), &i1, &W_(1, iw), &i1);
    }
  }
}

// LAPACK dsytd2, UPLO = 'U'
static void sytd2_u(lint n, double *a, lint lda, double *d, double *e, double *tau) {
  const double one = 1, mone = -1, half = 0.5; const lint i1 = 1;
  for (lint i = n - 1; i >= 1; i--) {
    double taui;
    dlarfg_(&i, &A_(i, i + 1), &A_(1, i + 1), &i1, &taui);
    e[i - 1] = A_(i, i + 1);
    if (taui != 0) {
      A_(i, i + 1) = one;
      symv(i, taui, a, lda, &A_(1, i + 1), tau);
      double alpha = -half * taui * ddot_(&i, tau, &i1, &A_(1, i + 1), &i1);
      daxpy_(&i, &alpha, &A_(1, i + 1), &i1, tau, &i1);
      dsyr2_("U", &i, &mone, &A_(1, i + 1), &i1, tau, &i1, a, &lda);
      A_(i, i + 1) = e[i - 1];
    }
    d[i] = A_(i + 1, i + 1);
    tau[i - 1] = taui;
  }
  d[0] = A_(1, 1);
}

// LAPACK dsytrd, UPLO = 'U', nb = 32, crossover nx = 32 (ilaenv defaults); work >= n * nb
static void sytrd_u(lint n, double *a, lint lda, double *d, double *e, double *tau, double *work) {
  const lint nb = 32, nx = 32; const double mone = -1, one = 1;
  lint kk = 0;
  if (n > nx) {
    kk = n - ((n - nx + nb - 1) / nb) * nb;
    for (lint i = n - nb + 1; i >= kk + 1; i -= nb) {
      lint m = i + nb - 1, im1 = i - 1;
      latrd_u(m, nb, a, lda, e, tau, work, n);
      dsyr2k_("U", "N", &im1, &nb, &mone, &A_(1, i), &lda, work, &n, &one, a, &lda);
      for (lint j = i; j <= i + nb - 1; j++) { A_(j - 1, j) = e[j - 2]; d[j - 1] = A_(j, j); }
    }
  } else kk = n;
  sytd2_u(kk, a, lda, d, e, tau);
}

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); int reps = argc > 2 ? atoi(argv[2]) : 20;
  int n4; fread(&n4, 4, 1, f); lint n = n4; size_t nn = (size_t)n * n;
  double *H = malloc(nn * 8), *S = malloc(nn * 8), *Hp = malloc(nn * 8), *A = malloc(nn * 8), *Z = malloc(nn * 8);
  fread(H, 8, nn, f); fread(S, 8, nn, f); fclose(f);
  lint info; double one = 1.0;
  dpotrf_("U", &n, S, &n, &info);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) S[i + j * n] = 0.0;
  dtrtri_("U", "N", &n, S, &n, &info);
  memcpy(Hp, H, nn * 8);
  dtrmm_("R", "U", "N", "N", &n, &n, &one, S, &n, Hp, &n);
  dtrmm_("L", "U", "T", "N", &n, &n, &one, S, &n, Hp, &n);
  double *d = malloc(n * 8), *e = malloc(n * 8), *tau = malloc(n * 8), *t = malloc(reps * 8), *w0 = malloc(n * 8);
  lint lwork = 64 * n + 100000; double *work = malloc(lwork * 8);
  lint *iwork = malloc((3 + 5 * n) * sizeof(lint)); lint liwork = 3 + 5 * n, ldc = 1 + 4 * n + n * n;
  double *workdc = malloc(ldc * 8);
  printf("n = %d\n", n);
  const char *name[3] = {"Accelerate dsytrd_              ", "port, Accelerate dsymv          ",
                         "port, hand-vectorized symv      "};
  for (int k = 0; k < 3; k++) {
    use_neon = (k == 2);
    for (int r = -1; r < reps; r++) {
      memcpy(A, Hp, nn * 8);
      double t0 = now();
      if (k == 0) dsytrd_("U", &n, A, &n, d, e, tau, work, &lwork, &info);
      else sytrd_u(n, A, n, d, e, tau, work);
      if (r >= 0) t[r] = 1e3 * (now() - t0);
    }
    qsort(t, reps, 8, cmp);
    // check: eigenvalues and back-transformed eigenvectors
    double *dd = malloc(n * 8), *ee = malloc(n * 8);
    memcpy(dd, d, n * 8); memcpy(ee, e, n * 8);
    dstedc_("I", &n, dd, ee, Z, &n, workdc, &ldc, iwork, &liwork, &info);
    dormtr_("L", "U", "N", &n, &n, A, &n, tau, Z, &n, work, &lwork, &info);
    if (k == 0) memcpy(w0, dd, n * 8);
    double zero = 0; double *R = malloc(nn * 8);
    dgemm_("N", "N", &n, &n, &n, &one, Hp, &n, Z, &n, &zero, R, &n);  // Hp full? use symmetric: Hp is full here
    double res = 0, dw = 0;
    for (lint j = 0; j < n; j++) { for (lint i = 0; i < n; i++) res = fmax(res, fabs(R[i + j * n] - dd[j] * Z[i + j * n])); dw = fmax(dw, fabs(dd[j] - w0[j])); }
    printf("%s median %8.3f ms  (min %8.3f)  eigenvalues vs Accelerate %.1e, residual |H'C - C w| %.1e\n", name[k],
           t[reps / 2], t[0], dw, res);
    free(dd); free(ee); free(R);
  }
  return 0;
}
