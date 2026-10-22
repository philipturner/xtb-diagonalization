// Is sytrd.c's NEON symv (four columns per pass) bound by the latency of its column dot products? Each s_c gets one
// fused multiply-add per vector of rows, a loop-carried chain of 4-cycle latency: one pass per 4 cycles, ~51 GB/s at
// 3.2 GHz, close to the ~40 GB/s sytrd_profile.c implies. Eight columns per pass doubles the independent chains in
// the same 4 cycles (16 FMAs, the 4 pipes' throughput). Same bits: rows below j0 see the eight columns in the same
// order as two passes of four; the second four's rows j0 ..< j0 + 4 run after the first four's diagonal block,
// as before. Checked bit for bit against sytrd.c's kernels; median time of 201 calls per size.
//   clang -O3 -mcpu=apple-m1 symv8.c -framework Accelerate -o symv8 && ./symv8
// Variants tried (no change, 8-column times within 1%): -DUNROLL=1 two vectors of rows per iteration with the two y
// chains interleaved (same bits); -DDUPREG=1 broadcasts kept in registers instead of by-element fmla. The 8-column
// kernel takes ~6.3 cycles per vector of rows (the 4-column one ~3.9, its latency bound); its limit wasn't found.
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef DUPREG
#define DUPREG 0
#endif
#ifndef UNROLL
#define UNROLL 0
#endif

#include "../../xtb/src/mctc/lapack/sytrd.c"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

// T scalar, V vector type, W lanes, sfx intrinsic suffix
#define SYMV8(NAME, T, V, W, sfx)                                                                                     \
static void NAME(int n, T alpha, const T *A, int lda, const T *x, T *y) {                                             \
  for (int i = 0; i < n; i++) y[i] = 0;                                                                               \
  int j0 = 0;                                                                                                         \
  for (; j0 + 8 <= n; j0 += 8) {                                                                                      \
    const T *c[8];                                                                                                    \
    for (int q = 0; q < 8; q++) c[q] = A + (size_t)(j0 + q) * lda;                                                    \
    V x0 = vdupq_n_##sfx(x[j0]), x1 = vdupq_n_##sfx(x[j0 + 1]), x2 = vdupq_n_##sfx(x[j0 + 2]),                        \
      x3 = vdupq_n_##sfx(x[j0 + 3]), x4 = vdupq_n_##sfx(x[j0 + 4]), x5 = vdupq_n_##sfx(x[j0 + 5]),                    \
      x6 = vdupq_n_##sfx(x[j0 + 6]), x7 = vdupq_n_##sfx(x[j0 + 7]);                                                   \
    if (DUPREG) { __asm__("" : "+w"(x0), "+w"(x1), "+w"(x2), "+w"(x3), "+w"(x4), "+w"(x5), "+w"(x6), "+w"(x7)); } \
    V s0 = vdupq_n_##sfx(0), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;                           \
    int i = 0;                                                                                                        \
    for (; UNROLL && i + 2 * W <= j0; i += 2 * W) {                                                                   \
      V a0 = vld1q_##sfx(c[0] + i), a1 = vld1q_##sfx(c[1] + i), a2 = vld1q_##sfx(c[2] + i), a3 = vld1q_##sfx(c[3] + i); \
      V a4 = vld1q_##sfx(c[4] + i), a5 = vld1q_##sfx(c[5] + i), a6 = vld1q_##sfx(c[6] + i), a7 = vld1q_##sfx(c[7] + i); \
      V b0 = vld1q_##sfx(c[0] + i + W), b1 = vld1q_##sfx(c[1] + i + W), b2 = vld1q_##sfx(c[2] + i + W), b3 = vld1q_##sfx(c[3] + i + W); \
      V b4 = vld1q_##sfx(c[4] + i + W), b5 = vld1q_##sfx(c[5] + i + W), b6 = vld1q_##sfx(c[6] + i + W), b7 = vld1q_##sfx(c[7] + i + W); \
      V xi = vld1q_##sfx(x + i), yi = vld1q_##sfx(y + i), xj = vld1q_##sfx(x + i + W), yj = vld1q_##sfx(y + i + W);   \
      yi = vfmaq_##sfx(yi, a0, x0); yj = vfmaq_##sfx(yj, b0, x0); yi = vfmaq_##sfx(yi, a1, x1); yj = vfmaq_##sfx(yj, b1, x1); \
      yi = vfmaq_##sfx(yi, a2, x2); yj = vfmaq_##sfx(yj, b2, x2); yi = vfmaq_##sfx(yi, a3, x3); yj = vfmaq_##sfx(yj, b3, x3); \
      yi = vfmaq_##sfx(yi, a4, x4); yj = vfmaq_##sfx(yj, b4, x4); yi = vfmaq_##sfx(yi, a5, x5); yj = vfmaq_##sfx(yj, b5, x5); \
      yi = vfmaq_##sfx(yi, a6, x6); yj = vfmaq_##sfx(yj, b6, x6); yi = vfmaq_##sfx(yi, a7, x7); yj = vfmaq_##sfx(yj, b7, x7); \
      vst1q_##sfx(y + i, yi); vst1q_##sfx(y + i + W, yj);                                                             \
      s0 = vfmaq_##sfx(s0, a0, xi); s1 = vfmaq_##sfx(s1, a1, xi); s2 = vfmaq_##sfx(s2, a2, xi); s3 = vfmaq_##sfx(s3, a3, xi); \
      s4 = vfmaq_##sfx(s4, a4, xi); s5 = vfmaq_##sfx(s5, a5, xi); s6 = vfmaq_##sfx(s6, a6, xi); s7 = vfmaq_##sfx(s7, a7, xi); \
      s0 = vfmaq_##sfx(s0, b0, xj); s1 = vfmaq_##sfx(s1, b1, xj); s2 = vfmaq_##sfx(s2, b2, xj); s3 = vfmaq_##sfx(s3, b3, xj); \
      s4 = vfmaq_##sfx(s4, b4, xj); s5 = vfmaq_##sfx(s5, b5, xj); s6 = vfmaq_##sfx(s6, b6, xj); s7 = vfmaq_##sfx(s7, b7, xj); \
    }                                                                                                                 \
    for (; i + W <= j0; i += W) {                                                                                     \
      V a0 = vld1q_##sfx(c[0] + i), a1 = vld1q_##sfx(c[1] + i), a2 = vld1q_##sfx(c[2] + i), a3 = vld1q_##sfx(c[3] + i); \
      V a4 = vld1q_##sfx(c[4] + i), a5 = vld1q_##sfx(c[5] + i), a6 = vld1q_##sfx(c[6] + i), a7 = vld1q_##sfx(c[7] + i); \
      V xi = vld1q_##sfx(x + i), yi = vld1q_##sfx(y + i);                                                             \
      yi = vfmaq_##sfx(yi, a0, x0); yi = vfmaq_##sfx(yi, a1, x1); yi = vfmaq_##sfx(yi, a2, x2); yi = vfmaq_##sfx(yi, a3, x3); \
      yi = vfmaq_##sfx(yi, a4, x4); yi = vfmaq_##sfx(yi, a5, x5); yi = vfmaq_##sfx(yi, a6, x6); yi = vfmaq_##sfx(yi, a7, x7); \
      vst1q_##sfx(y + i, yi);                                                                                         \
      s0 = vfmaq_##sfx(s0, a0, xi); s1 = vfmaq_##sfx(s1, a1, xi); s2 = vfmaq_##sfx(s2, a2, xi); s3 = vfmaq_##sfx(s3, a3, xi); \
      s4 = vfmaq_##sfx(s4, a4, xi); s5 = vfmaq_##sfx(s5, a5, xi); s6 = vfmaq_##sfx(s6, a6, xi); s7 = vfmaq_##sfx(s7, a7, xi); \
    }                                                                                                                 \
    /* First four columns: finish as a pass of four (no row tail: j0 is a multiple of 4). */                          \
    y[j0] += vaddvq_##sfx(s0); y[j0 + 1] += vaddvq_##sfx(s1); y[j0 + 2] += vaddvq_##sfx(s2); y[j0 + 3] += vaddvq_##sfx(s3); \
    for (int q = 0; q < 4; q++) {                                                                                     \
      const T *col = c[q];                                                                                            \
      for (int r = 0; r < q; r++) { y[j0 + r] += col[j0 + r] * x[j0 + q]; y[j0 + q] += col[j0 + r] * x[j0 + r]; }    \
      y[j0 + q] += col[j0 + q] * x[j0 + q];                                                                           \
    }                                                                                                                 \
    /* Last four: rows j0 ..< j0 + 4, then finish. */                                                                 \
    for (; i + W <= j0 + 4; i += W) {                                                                                 \
      V a4 = vld1q_##sfx(c[4] + i), a5 = vld1q_##sfx(c[5] + i), a6 = vld1q_##sfx(c[6] + i), a7 = vld1q_##sfx(c[7] + i); \
      V xi = vld1q_##sfx(x + i), yi = vld1q_##sfx(y + i);                                                             \
      yi = vfmaq_##sfx(yi, a4, x4); yi = vfmaq_##sfx(yi, a5, x5); yi = vfmaq_##sfx(yi, a6, x6); yi = vfmaq_##sfx(yi, a7, x7); \
      vst1q_##sfx(y + i, yi);                                                                                         \
      s4 = vfmaq_##sfx(s4, a4, xi); s5 = vfmaq_##sfx(s5, a5, xi); s6 = vfmaq_##sfx(s6, a6, xi); s7 = vfmaq_##sfx(s7, a7, xi); \
    }                                                                                                                 \
    y[j0 + 4] += vaddvq_##sfx(s4); y[j0 + 5] += vaddvq_##sfx(s5); y[j0 + 6] += vaddvq_##sfx(s6); y[j0 + 7] += vaddvq_##sfx(s7); \
    for (int q = 0; q < 4; q++) {                                                                                     \
      const T *col = c[4 + q];                                                                                        \
      for (int r = 0; r < q; r++) { y[j0 + 4 + r] += col[j0 + 4 + r] * x[j0 + 4 + q]; y[j0 + 4 + q] += col[j0 + 4 + r] * x[j0 + 4 + r]; } \
      y[j0 + 4 + q] += col[j0 + 4 + q] * x[j0 + 4 + q];                                                               \
    }                                                                                                                 \
  }                                                                                                                   \
  for (; j0 + 4 <= n; j0 += 4) {                                                                                      \
    const T *c0 = A + (size_t)j0 * lda, *c1 = c0 + lda, *c2 = c1 + lda, *c3 = c2 + lda;                               \
    V x0 = vdupq_n_##sfx(x[j0]), x1 = vdupq_n_##sfx(x[j0 + 1]), x2 = vdupq_n_##sfx(x[j0 + 2]), x3 = vdupq_n_##sfx(x[j0 + 3]); \
    V s0 = vdupq_n_##sfx(0), s1 = s0, s2 = s0, s3 = s0;                                                               \
    for (int i = 0; i + W <= j0; i += W) {                                                                            \
      V a0 = vld1q_##sfx(c0 + i), a1 = vld1q_##sfx(c1 + i), a2 = vld1q_##sfx(c2 + i), a3 = vld1q_##sfx(c3 + i);       \
      V xi = vld1q_##sfx(x + i), yi = vld1q_##sfx(y + i);                                                             \
      yi = vfmaq_##sfx(yi, a0, x0); yi = vfmaq_##sfx(yi, a1, x1); yi = vfmaq_##sfx(yi, a2, x2); yi = vfmaq_##sfx(yi, a3, x3); \
      vst1q_##sfx(y + i, yi);                                                                                         \
      s0 = vfmaq_##sfx(s0, a0, xi); s1 = vfmaq_##sfx(s1, a1, xi); s2 = vfmaq_##sfx(s2, a2, xi); s3 = vfmaq_##sfx(s3, a3, xi); \
    }                                                                                                                 \
    y[j0] += vaddvq_##sfx(s0); y[j0 + 1] += vaddvq_##sfx(s1); y[j0 + 2] += vaddvq_##sfx(s2); y[j0 + 3] += vaddvq_##sfx(s3); \
    for (int q = 0; q < 4; q++) {                                                                                     \
      const T *col = A + (size_t)(j0 + q) * lda;                                                                      \
      for (int r = 0; r < q; r++) { y[j0 + r] += col[j0 + r] * x[j0 + q]; y[j0 + q] += col[j0 + r] * x[j0 + r]; }    \
      y[j0 + q] += col[j0 + q] * x[j0 + q];                                                                           \
    }                                                                                                                 \
  }                                                                                                                   \
  for (; j0 < n; j0++) {                                                                                              \
    const T *col = A + (size_t)j0 * lda;                                                                              \
    T s = 0;                                                                                                          \
    for (int i = 0; i < j0; i++) { y[i] += col[i] * x[j0]; s += col[i] * x[i]; }                                      \
    y[j0] += s + col[j0] * x[j0];                                                                                     \
  }                                                                                                                   \
  if (alpha != 1) for (int i = 0; i < n; i++) y[i] *= alpha;                                                          \
}
SYMV8(symv8_d, double, float64x2_t, 2, f64)
SYMV8(symv8_f, float, float32x4_t, 4, f32)

int main(void) {
  int sizes[] = {5, 9, 13, 64, 131, 200, 300, 412, 448, 619, 776};
  printf("median us per call (GB/s on the upper triangle); bits: 8-column vs sytrd.c\n");
  for (int si = 0; si < (int)(sizeof sizes / sizeof sizes[0]); si++) {
    int n = sizes[si], lda = n;
    double *A, *x = malloc(n * 8), *y1 = malloc(n * 8), *y2 = malloc(n * 8);
    float *As, *xs = malloc(n * 4), *ys1 = malloc(n * 4), *ys2 = malloc(n * 4);
    posix_memalign((void **)&A, 16384, (size_t)n * n * 8); posix_memalign((void **)&As, 16384, (size_t)n * n * 4);
    srand(n);
    for (size_t i = 0; i < (size_t)n * n; i++) { A[i] = (double)rand() / RAND_MAX - 0.5; As[i] = (float)A[i]; }
    for (int i = 0; i < n; i++) { x[i] = (double)rand() / RAND_MAX - 0.5; xs[i] = (float)x[i]; }
    symv_upper(n, 0.7, A, lda, x, y1); symv8_d(n, 0.7, A, lda, x, y2);
    int same_d = memcmp(y1, y2, n * 8) == 0;
    symv_upper_f(n, 0.7f, As, lda, xs, ys1); symv8_f(n, 0.7f, As, lda, xs, ys2);
    int same_f = memcmp(ys1, ys2, n * 4) == 0;
    double t[4][201];
    for (int r = -5; r < 201; r++)
      for (int v = 0; v < 4; v++) {
        double t0 = now();
        if (v == 0) symv_upper(n, 1, A, lda, x, y1);
        else if (v == 1) symv8_d(n, 1, A, lda, x, y2);
        else if (v == 2) symv_upper_f(n, 1, As, lda, xs, ys1);
        else symv8_f(n, 1, As, lda, xs, ys2);
        if (r >= 0) t[v][r] = 1e6 * (now() - t0);
      }
    double med[4];
    for (int v = 0; v < 4; v++) { qsort(t[v], 201, 8, cmp); med[v] = t[v][100]; }
    double bd = 0.5 * n * (n + 1) * 8e-3, bf = bd / 2;  // bytes / 1e3 -> GB/s with us
    printf("n = %4d  FP64 4-col %8.2f (%5.1f)  8-col %8.2f (%5.1f)  %.2fx %s | FP32 4-col %8.2f (%5.1f)  8-col %8.2f (%5.1f)  %.2fx %s\n",
           n, med[0], bd / med[0], med[1], bd / med[1], med[0] / med[1], same_d ? "same bits" : "DIFFERENT",
           med[2], bf / med[2], med[3], bf / med[3], med[2] / med[3], same_f ? "same bits" : "DIFFERENT");
    free(A); free(As); free(x); free(y1); free(y2); free(xs); free(ys1); free(ys2);
  }
  return 0;
}
