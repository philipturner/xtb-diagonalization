// Hand-vectorized symmetric matrix-vector product y = A x from the upper triangle (column-major, lda), against
// Accelerate's dsymv: does dsytrd's BLAS2 core have headroom on M1 Max?
// Four columns per pass: for rows i < j0, each loaded A(i, j0..j0+3) feeds both y(i) += sum_c A(i,c) x(c) and the
// column dot products s_c += A(i,c) x(i), so the triangle is read once.
//   clang -O3 -mcpu=apple-m1 symv_neon.c -framework Accelerate -o symv_neon
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

extern void dsymv_(const char *, const int *, const double *, const double *, const int *, const double *,
                   const int *, const double *, double *, const int *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

void symv_upper(int n, const double *A, int lda, const double *x, double *y) {
  for (int i = 0; i < n; i++) y[i] = 0.0;
  int j0 = 0;
  for (; j0 + 4 <= n; j0 += 4) {
    const double *c0 = A + (size_t)j0 * lda, *c1 = c0 + lda, *c2 = c1 + lda, *c3 = c2 + lda;
    float64x2_t x0 = vdupq_n_f64(x[j0]), x1 = vdupq_n_f64(x[j0 + 1]), x2 = vdupq_n_f64(x[j0 + 2]),
                x3 = vdupq_n_f64(x[j0 + 3]);
    float64x2_t s0 = vdupq_n_f64(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 2 <= j0; i += 2) {  // rows strictly above the 4x4 diagonal block
      float64x2_t a0 = vld1q_f64(c0 + i), a1 = vld1q_f64(c1 + i), a2 = vld1q_f64(c2 + i), a3 = vld1q_f64(c3 + i);
      float64x2_t xi = vld1q_f64(x + i), yi = vld1q_f64(y + i);
      yi = vfmaq_f64(yi, a0, x0); yi = vfmaq_f64(yi, a1, x1); yi = vfmaq_f64(yi, a2, x2); yi = vfmaq_f64(yi, a3, x3);
      vst1q_f64(y + i, yi);
      s0 = vfmaq_f64(s0, a0, xi); s1 = vfmaq_f64(s1, a1, xi); s2 = vfmaq_f64(s2, a2, xi); s3 = vfmaq_f64(s3, a3, xi);
    }
    double t0 = vaddvq_f64(s0), t1 = vaddvq_f64(s1), t2 = vaddvq_f64(s2), t3 = vaddvq_f64(s3);
    for (; i < j0; i++) {  // odd remainder row
      y[i] += c0[i] * x[j0] + c1[i] * x[j0 + 1] + c2[i] * x[j0 + 2] + c3[i] * x[j0 + 3];
      t0 += c0[i] * x[i]; t1 += c1[i] * x[i]; t2 += c2[i] * x[i]; t3 += c3[i] * x[i];
    }
    y[j0] += t0; y[j0 + 1] += t1; y[j0 + 2] += t2; y[j0 + 3] += t3;
    // the 4x4 diagonal block, upper triangle
    for (int c = 0; c < 4; c++) {
      const double *col = A + (size_t)(j0 + c) * lda;
      for (int r = 0; r < c; r++) {
        y[j0 + r] += col[j0 + r] * x[j0 + c];
        y[j0 + c] += col[j0 + r] * x[j0 + r];
      }
      y[j0 + c] += col[j0 + c] * x[j0 + c];
    }
  }
  for (; j0 < n; j0++) {  // remaining columns
    const double *col = A + (size_t)j0 * lda;
    double s = 0;
    for (int i = 0; i < j0; i++) { y[i] += col[i] * x[j0]; s += col[i] * x[i]; }
    y[j0] += s + col[j0] * x[j0];
  }
}

// Eight columns per pass: half the y loads/stores per matrix element of the four-column version.
void symv_upper8(int n, const double *A, int lda, const double *x, double *y) {
  for (int i = 0; i < n; i++) y[i] = 0.0;
  int j0 = 0;
  for (; j0 + 8 <= n; j0 += 8) {
    const double *c[8];
    float64x2_t xc[8], sc[8];
    for (int q = 0; q < 8; q++) { c[q] = A + (size_t)(j0 + q) * lda; xc[q] = vdupq_n_f64(x[j0 + q]); sc[q] = vdupq_n_f64(0); }
    int i = 0;
    for (; i + 2 <= j0; i += 2) {
      float64x2_t xi = vld1q_f64(x + i), yi = vld1q_f64(y + i);
      for (int q = 0; q < 8; q++) {
        float64x2_t a = vld1q_f64(c[q] + i);
        yi = vfmaq_f64(yi, a, xc[q]);
        sc[q] = vfmaq_f64(sc[q], a, xi);
      }
      vst1q_f64(y + i, yi);
    }
    double t[8];
    for (int q = 0; q < 8; q++) t[q] = vaddvq_f64(sc[q]);
    for (; i < j0; i++) {
      double acc = 0;
      for (int q = 0; q < 8; q++) { acc += c[q][i] * x[j0 + q]; t[q] += c[q][i] * x[i]; }
      y[i] += acc;
    }
    for (int q = 0; q < 8; q++) y[j0 + q] += t[q];
    for (int q = 0; q < 8; q++) {
      const double *col = c[q];
      for (int r = 0; r < q; r++) { y[j0 + r] += col[j0 + r] * x[j0 + q]; y[j0 + q] += col[j0 + r] * x[j0 + r]; }
      y[j0 + q] += col[j0 + q] * x[j0 + q];
    }
  }
  for (; j0 < n; j0++) {
    const double *col = A + (size_t)j0 * lda;
    double s = 0;
    for (int i = 0; i < j0; i++) { y[i] += col[i] * x[j0]; s += col[i] * x[i]; }
    y[j0] += s + col[j0] * x[j0];
  }
}

int main(void) {
  int sizes[] = {448, 300, 150, 776, 500, 250}, reps = 200;
  double *t = malloc(reps * 8);
  printf("    n   dsymv us  GB/s | neon us  GB/s | speedup | 8-col us GB/s | max|dy|\n");
  for (int s = 0; s < 6; s++) {
    int n = sizes[s], one = 1; double alpha = 1, beta = 0;
    double *A = malloc((size_t)n * n * 8), *x = malloc(n * 8), *y = malloc(n * 8), *y2 = malloc(n * 8);
    srand(1);
    for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) A[i + (size_t)j * n] = A[j + (size_t)i * n] = rand() / (double)RAND_MAX;
    for (int i = 0; i < n; i++) x[i] = rand() / (double)RAND_MAX;
    double m[3]; double *y3 = malloc(n * 8);
    for (int k = 0; k < 3; k++) {
      for (int r = -1; r < reps; r++) {
        double t0 = now();
        if (k == 0) dsymv_("U", &n, &alpha, A, &n, x, &one, &beta, y, &one);
        else if (k == 1) symv_upper(n, A, n, x, y2);
        else symv_upper8(n, A, n, x, y3);
        if (r >= 0) t[r] = 1e6 * (now() - t0);
      }
      qsort(t, reps, 8, cmp); m[k] = t[reps / 2];
    }
    double err = 0; for (int i = 0; i < n; i++) err = fmax(err, fmax(fabs(y[i] - y2[i]), fabs(y[i] - y3[i])));
    double bytes = 8.0 * n * (n + 1) / 2;
    printf("%5d %9.2f %5.1f | %7.2f %5.1f | %6.2fx | 8-col %7.2f %5.1f | %.1e\n", n, m[0], bytes / m[0] / 1e3, m[1],
           bytes / m[1] / 1e3, m[0] / m[1], m[2], bytes / m[2] / 1e3, err);
    free(A); free(x); free(y); free(y2); free(y3);
  }
  return 0;
}
