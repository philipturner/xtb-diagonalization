// Where does the bulge back-transformation's time go? One bulge block: b reflectors on 2b - 1 rows (parallelogram V,
// dense storage) applied to a (2b - 1) x m block of Z, as backTransformBulgesBlocked does per block.
//   clang -O3 -mcpu=apple-m1 wy.c -framework Accelerate -o wy
//   VECLIB_MAXIMUM_THREADS=1 ./wy
// Times (median, us per block) of xlarft (T), xlarfb ('L', 'N'), and larfb's work as plain GEMMs: W = V^T Z (gemm),
// W = T W (trmm, or gemm with T's zero lower triangle), Z -= V W (gemm). GFLOP/s counts the dense V: 4 (2b - 1) b m.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../Sources/CLAPACK/include/CLAPACK.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

#define MEDIAN(expr, out) do { \
  double ts[41]; for (int r = 0; r < 41; r++) { double t0 = now(); for (int q = 0; q < reps; q++) { expr; } \
  ts[r] = (now() - t0) / reps; } qsort(ts, 41, sizeof(double), cmp); out = ts[20]; } while (0)

#define BENCH(T, LARFT, LARFB, GEMM, TRMM, NAME) \
static void bench_##T(int b, int m) { \
  lint h = 2 * b - 1, ldv = 2 * b, bb = b, mm = m, ldz = 2 * b, ldt = b, ldw = m; \
  T *v, *t, *tau, *z, *w; \
  posix_memalign((void **)&v, 16384, sizeof(T) * ldv * b); posix_memalign((void **)&t, 16384, sizeof(T) * b * b); \
  posix_memalign((void **)&tau, 16384, sizeof(T) * b); posix_memalign((void **)&z, 16384, sizeof(T) * ldz * m); \
  posix_memalign((void **)&w, 16384, sizeof(T) * m * b); \
  for (int i = 0; i < ldv * b; i++) v[i] = 0; \
  for (int c = 0; c < b; c++) { v[c + c * ldv] = 1; for (int i = 1; i < b; i++) v[c + i + c * ldv] = 0.1 * (rand() / (T)RAND_MAX - 0.5); tau[c] = 1.5; } \
  for (int i = 0; i < ldz * m; i++) z[i] = rand() / (T)RAND_MAX; \
  int reps = 1 + 20000 / (b * m / 16 + 1); \
  double tf, tb, tg1, tg2, ttr; \
  MEDIAN(LARFT("F", "C", &h, &bb, v, &ldv, tau, t, &ldt), tf); \
  MEDIAN(LARFB("L", "N", "F", "C", &h, &mm, &bb, v, &ldv, t, &ldt, z, &ldz, w, &ldw), tb); \
  T one = 1, zero = 0, mone = -1; lint bm = b; \
  MEDIAN(GEMM("T", "N", &bm, &mm, &h, &one, v, &ldv, z, &ldz, &zero, w, &bm), tg1); \
  MEDIAN(TRMM("L", "U", "N", "N", &bm, &mm, &one, t, &ldt, w, &bm), ttr); \
  MEDIAN(GEMM("N", "N", &h, &mm, &bm, &mone, v, &ldv, w, &bm, &one, z, &ldz), tg2); \
  T *w2; posix_memalign((void **)&w2, 16384, sizeof(T) * m * b); double tgt; \
  MEDIAN(GEMM("N", "N", &bm, &mm, &bm, &one, t, &ldt, w, &bm, &zero, w2, &bm), tgt); free(w2); \
  double fl = 4.0 * (2 * b - 1) * b * m; \
  printf("%s b=%2d m=%4d | larft %7.2f | larfb %7.2f (%5.1f GF/s) | gemm %7.2f + trmm %6.2f + gemm %7.2f = %7.2f (%5.1f GF/s) | T by gemm %6.2f\n", \
         NAME, b, m, 1e6 * tf, 1e6 * tb, fl / tb * 1e-9, 1e6 * tg1, 1e6 * ttr, 1e6 * tg2, 1e6 * (tg1 + ttr + tg2), \
         fl / (tg1 + ttr + tg2) * 1e-9, 1e6 * tgt); \
  free(v); free(t); free(tau); free(z); free(w); \
}

BENCH(double, dlarft_, dlarfb_, dgemm_, dtrmm_, "FP64")
BENCH(float, slarft_, slarfb_, sgemm_, strmm_, "FP32")

int main(void) {
  int bs[] = {16, 32, 48, 64}, ms[] = {294, 588, 1204};
  for (int mi = 0; mi < 3; mi++) for (int bi = 0; bi < 4; bi++) bench_double(bs[bi], ms[mi]);
  for (int mi = 0; mi < 3; mi++) for (int bi = 0; bi < 4; bi++) bench_float(bs[bi], ms[mi]);
  return 0;
}
