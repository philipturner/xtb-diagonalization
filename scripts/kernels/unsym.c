// Independent check of the Swift sytrd-bench result that the unsymmetrized reduction (full matrix: gemv, gemm, ger)
// costs about the same as the symmetric one in FP32 but 1.4-1.8x in FP64.
//   unsym HS.bin [REPS]   (HS.bin from diag/dump)
//   clang -O3 -mcpu=apple-m1 -ffp-contract=off unsym.c -framework Accelerate -o unsym
// Part 1, components at the sizes the reduction sees (m = trailing size): the hand-vectorized symv of sytrd.c vs
// Accelerate's xsymv vs Accelerate's full xgemv; xsyr2k (k = 32) vs two xgemm; xsyr2 vs two xger. Median of REPS calls,
// each kernel in its own loop.
// Part 2, the whole reduction in C (sytrd.c, included, with a symmetric/unsymmetrized switch): symmetric vs
// unsymmetrized, timed two ways: interleaved (round-robin, as sytrd-bench) and blocked (all reps of one variant, then the
// other). Eigenvalues (xsterf) compared between the two.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define xtb_dsytrd_upper ref_dsytrd_upper
#define xtb_ssytrd_upper ref_ssytrd_upper
#include "../../xtb/src/mctc/lapack/sytrd.c"

extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *,
                   const double *, const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void sgemm_(const char *, const char *, const lint *, const lint *, const lint *, const float *, const float *,
                   const lint *, const float *, const lint *, const float *, float *, const lint *);
extern void dger_(const lint *, const lint *, const double *, const double *, const lint *, const double *,
                  const lint *, double *, const lint *);
extern void sger_(const lint *, const lint *, const float *, const float *, const lint *, const float *, const lint *,
                  float *, const lint *);
extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dtrtri_(const char *, const char *, const lint *, double *, const lint *, lint *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const lint *, const lint *, const double *,
                   const double *, const lint *, double *, const lint *);
extern void dsterf_(const lint *, double *, double *, lint *);
extern void ssterf_(const lint *, float *, float *, lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double median(double *t, int n) { qsort(t, n, 8, cmp); return t[n / 2]; }

// ---- unsymmetrized reduction: sytrd.c's latrd_u / sytd2_u / sytrd_u with full-matrix BLAS ----
// UNSYM_GEMV=T multiplies by the transpose (the same product, A being symmetric): Accelerate's dgemv 'T' reads the
// matrix 2-3x faster than 'N' (scripts/kernels/gemv_bw.c).
static const char *gemv_trans = "N";
#define GEN(T, P, SYMV, DOT, SUF)                                                                                   \
static void latrd_##SUF(lint n, lint nb, T *a, lint lda, T *e, T *tau, T *w, lint ldw, int sym) {                    \
  const T one = 1, mone = -1, zero = 0, half = 0.5; const lint i1 = 1;                                             \
  for (lint i = n; i >= n - nb + 1; i--) {                                                                         \
    lint iw = i - n + nb;                                                                                          \
    if (i < n) {                                                                                                   \
      lint m = n - i;                                                                                              \
      P##gemv_("N", &i, &m, &mone, &A_(1, i + 1), &lda, &W_(i, iw + 1), &ldw, &one, &A_(1, i), &i1);              \
      P##gemv_("N", &i, &m, &mone, &W_(1, iw + 1), &ldw, &A_(i, i + 1), &lda, &one, &A_(1, i), &i1);              \
    }                                                                                                              \
    if (i > 1) {                                                                                                   \
      lint im1 = i - 1;                                                                                            \
      P##larfg_(&im1, &A_(i - 1, i), &A_(1, i), &i1, &tau[i - 2]);                                                 \
      e[i - 2] = A_(i - 1, i);                                                                                     \
      A_(i - 1, i) = one;                                                                                          \
      if (sym) SYMV(im1, one, a, lda, &A_(1, i), &W_(1, iw));                                                      \
      else P##gemv_(gemv_trans, &im1, &im1, &one, a, &lda, &A_(1, i), &i1, &zero, &W_(1, iw), &i1);                \
      if (i < n) {                                                                                                 \
        lint m = n - i;                                                                                            \
        P##gemv_("T", &im1, &m, &one, &W_(1, iw + 1), &ldw, &A_(1, i), &i1, &zero, &W_(i + 1, iw), &i1);          \
        P##gemv_("N", &im1, &m, &mone, &A_(1, i + 1), &lda, &W_(i + 1, iw), &i1, &one, &W_(1, iw), &i1);          \
        P##gemv_("T", &im1, &m, &one, &A_(1, i + 1), &lda, &A_(1, i), &i1, &zero, &W_(i + 1, iw), &i1);           \
        P##gemv_("N", &im1, &m, &mone, &W_(1, iw + 1), &ldw, &W_(i + 1, iw), &i1, &one, &W_(1, iw), &i1);          \
      }                                                                                                            \
      P##scal_(&im1, &tau[i - 2], &W_(1, iw), &i1);                                                                \
      T alpha = -half * tau[i - 2] * DOT(im1, &W_(1, iw), &A_(1, i));                                              \
      P##axpy_(&im1, &alpha, &A_(1, i), &i1, &W_(1, iw), &i1);                                                     \
    }                                                                                                              \
  }                                                                                                                \
}                                                                                                                  \
static void sytd2_##SUF(lint n, T *a, lint lda, T *d, T *e, T *tau, int sym) {                                       \
  const T one = 1, mone = -1, half = 0.5, zero = 0; const lint i1 = 1;                                             \
  for (lint i = n - 1; i >= 1; i--) {                                                                              \
    T taui;                                                                                                        \
    P##larfg_(&i, &A_(i, i + 1), &A_(1, i + 1), &i1, &taui);                                                       \
    e[i - 1] = A_(i, i + 1);                                                                                       \
    if (taui != 0) {                                                                                               \
      A_(i, i + 1) = one;                                                                                          \
      if (sym) SYMV(i, taui, a, lda, &A_(1, i + 1), tau);                                                          \
      else P##gemv_(gemv_trans, &i, &i, &taui, a, &lda, &A_(1, i + 1), &i1, &zero, tau, &i1);                      \
      T alpha = -half * taui * DOT(i, tau, &A_(1, i + 1));                                                         \
      P##axpy_(&i, &alpha, &A_(1, i + 1), &i1, tau, &i1);                                                          \
      if (sym) P##syr2_("U", &i, &mone, &A_(1, i + 1), &i1, tau, &i1, a, &lda);                                    \
      else { P##ger_(&i, &i, &mone, &A_(1, i + 1), &i1, tau, &i1, a, &lda);                                        \
             P##ger_(&i, &i, &mone, tau, &i1, &A_(1, i + 1), &i1, a, &lda); }                                      \
      A_(i, i + 1) = e[i - 1];                                                                                     \
    }                                                                                                              \
    d[i] = A_(i + 1, i + 1);                                                                                       \
    tau[i - 1] = taui;                                                                                             \
  }                                                                                                                \
  d[0] = A_(1, 1);                                                                                                 \
}                                                                                                                  \
static void sytrd_##SUF(lint n, T *a, lint lda, T *d, T *e, T *tau, T *work, int sym) {                              \
  const lint nb = 32, nx = 32; const T mone = -1, one = 1;                                                         \
  lint kk = n;                                                                                                     \
  if (n > nx) {                                                                                                    \
    kk = n - ((n - nx + nb - 1) / nb) * nb;                                                                        \
    for (lint i = n - nb + 1; i >= kk + 1; i -= nb) {                                                              \
      lint m = i + nb - 1, im1 = i - 1;                                                                            \
      latrd_##SUF(m, nb, a, lda, e, tau, work, n, sym);                                                            \
      if (sym) P##syr2k_("U", "N", &im1, &nb, &mone, &A_(1, i), &lda, work, &n, &one, a, &lda);                    \
      else { P##gemm_("N", "T", &im1, &im1, &nb, &mone, &A_(1, i), &lda, work, &n, &one, a, &lda);                 \
             P##gemm_("N", "T", &im1, &im1, &nb, &mone, work, &n, &A_(1, i), &lda, &one, a, &lda); }               \
      for (lint j = i; j <= i + nb - 1; j++) { A_(j - 1, j) = e[j - 2]; d[j - 1] = A_(j, j); }                     \
    }                                                                                                              \
  }                                                                                                                \
  sytd2_##SUF(kk, a, lda, d, e, tau, sym);                                                                         \
}

static double ddot1(lint n, const double *x, const double *y) { lint i1 = 1; return ddot_(&n, x, &i1, y, &i1); }
GEN(double, d, symv, ddot1, d)
#undef A_
#undef W_
#define A_(i, j) a[((i) - 1) + (size_t)((j) - 1) * lda]
#define W_(i, j) w[((i) - 1) + (size_t)((j) - 1) * ldw]
GEN(float, s, symv_f, dot_f, s)

// ---- part 1: components ----
static void components(int reps) {
  int sizes[] = {448, 300, 150};
  printf("Part 1: components, median of %d calls (us), effective GB/s on the n x n matrix in brackets\n", reps);
  printf("  %-5s %-4s %16s %16s %16s | %10s %10s | %9s %9s\n", "m", "prec", "NEON symv", "Accel symv", "Accel gemv full",
         "syr2k k=32", "2x gemm", "syr2", "2x ger");
  double *t = malloc(reps * 8);
  for (int si = 0; si < 3; si++) {
    lint m = sizes[si], i1 = 1, nb = 32;
    size_t mm = (size_t)m * m;
    for (int prec = 0; prec < 2; prec++) {
      size_t es = prec ? 4 : 8;
      void *A = malloc(mm * es), *x = malloc(m * es), *y = malloc(m * es), *V = malloc(m * nb * es),
           *W = malloc(m * nb * es);
      srand(1);
      for (size_t k = 0; k < mm; k++) { double v = rand() / (double)RAND_MAX - 0.5;
        if (prec) ((float *)A)[k] = v; else ((double *)A)[k] = v; }
      for (lint k = 0; k < m * nb; k++) { double v = rand() / (double)RAND_MAX - 0.5;
        if (prec) { ((float *)V)[k] = v; ((float *)W)[k] = -v; } else { ((double *)V)[k] = v; ((double *)W)[k] = -v; } }
      for (lint k = 0; k < m; k++) { double v = rand() / (double)RAND_MAX;
        if (prec) ((float *)x)[k] = v; else ((double *)x)[k] = v; }
      double r[7];
      for (int kern = 0; kern < 7; kern++) {
        for (int k = -5; k < reps; k++) {
          double t0 = now();
          if (prec == 0) {
            double one = 1, zero = 0, mone = -1, *a = A, *xx = x, *yy = y, *v = V, *w = W;
            switch (kern) {
              case 0: symv(m, one, a, m, xx, yy); break;
              case 1: dsymv_("U", &m, &one, a, &m, xx, &i1, &zero, yy, &i1); break;
              case 2: dgemv_("N", &m, &m, &one, a, &m, xx, &i1, &zero, yy, &i1); break;
              case 3: dsyr2k_("U", "N", &m, &nb, &mone, v, &m, w, &m, &one, a, &m); break;
              case 4: dgemm_("N", "T", &m, &m, &nb, &mone, v, &m, w, &m, &one, a, &m);
                      dgemm_("N", "T", &m, &m, &nb, &mone, w, &m, v, &m, &one, a, &m); break;
              case 5: dsyr2_("U", &m, &mone, xx, &i1, yy, &i1, a, &m); break;
              case 6: dger_(&m, &m, &mone, xx, &i1, yy, &i1, a, &m); dger_(&m, &m, &mone, yy, &i1, xx, &i1, a, &m); break;
            }
          } else {
            float one = 1, zero = 0, mone = -1, *a = A, *xx = x, *yy = y, *v = V, *w = W;
            switch (kern) {
              case 0: symv_f(m, one, a, m, xx, yy); break;
              case 1: ssymv_("U", &m, &one, a, &m, xx, &i1, &zero, yy, &i1); break;
              case 2: sgemv_("N", &m, &m, &one, a, &m, xx, &i1, &zero, yy, &i1); break;
              case 3: ssyr2k_("U", "N", &m, &nb, &mone, v, &m, w, &m, &one, a, &m); break;
              case 4: sgemm_("N", "T", &m, &m, &nb, &mone, v, &m, w, &m, &one, a, &m);
                      sgemm_("N", "T", &m, &m, &nb, &mone, w, &m, v, &m, &one, a, &m); break;
              case 5: ssyr2_("U", &m, &mone, xx, &i1, yy, &i1, a, &m); break;
              case 6: sger_(&m, &m, &mone, xx, &i1, yy, &i1, a, &m); sger_(&m, &m, &mone, yy, &i1, xx, &i1, a, &m); break;
            }
          }
          if (k >= 0) t[k] = 1e6 * (now() - t0);
          // keep the matrix bounded: the rank updates add the same terms every call
          if (kern >= 3 && k % 16 == 15) { for (size_t q = 0; q < mm; q += 1) {
            if (prec) ((float *)A)[q] *= 0.5f; else ((double *)A)[q] *= 0.5; } }
        }
        r[kern] = median(t, reps);
      }
      double gb = (double)mm * es / 1e3;  // bytes / 1e9 per us -> GB/s with /us
      printf("  %-5d %-4s %8.2f [%5.1f] %8.2f [%5.1f] %8.2f [%5.1f] | %10.2f %10.2f | %9.2f %9.2f\n", m,
             prec ? "FP32" : "FP64", r[0], gb / r[0], r[1], gb / r[1], r[2], gb / r[2], r[3], r[4], r[5], r[6]);
      free(A); free(x); free(y); free(V); free(W);
    }
  }
  free(t);
}

// ---- part 2: whole reduction ----
static void load_d(double *A, lint lda, const double *H, lint n) {
  for (lint j = 0; j < n; j++) memcpy(A + (size_t)j * lda, H + (size_t)j * n, n * 8);
}
static void load_s(float *A, lint lda, const float *H, lint n) {
  for (lint j = 0; j < n; j++) memcpy(A + (size_t)j * lda, H + (size_t)j * n, n * 4);
}
static void reduction(const char *path, int reps) {
  FILE *f = fopen(path, "rb");
  int n4; fread(&n4, 4, 1, f); lint n = n4; size_t nn = (size_t)n * n;
  double *H = malloc(nn * 8), *S = malloc(nn * 8);
  fread(H, 8, nn, f); fread(S, 8, nn, f); fclose(f);
  lint info; double one = 1;
  dpotrf_("U", &n, S, &n, &info);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) S[i + j * n] = 0;
  dtrtri_("U", "N", &n, S, &n, &info);
  dtrmm_("R", "U", "N", "N", &n, &n, &one, S, &n, H, &n);
  dtrmm_("L", "U", "T", "N", &n, &n, &one, S, &n, H, &n);
  float *Hs = malloc(nn * 4); for (size_t k = 0; k < nn; k++) Hs[k] = H[k];
  // UNSYM_OFFSET (bytes, default 0) shifts the matrix and workspace from malloc's page alignment, e.g. 32 as in a
  // Swift array (elements start 32 bytes after the allocation).
  size_t off = getenv("UNSYM_OFFSET") ? (size_t)atoi(getenv("UNSYM_OFFSET")) : 0;
  // UNSYM_PAD=1 rounds the leading dimension up to a multiple of 64 bytes (8 doubles / 16 floats).
  int pad = getenv("UNSYM_PAD") && atoi(getenv("UNSYM_PAD"));
  lint ldd = pad ? (n + 7) / 8 * 8 : n, lds = pad ? (n + 15) / 16 * 16 : n;
  nn = (size_t)(ldd > lds ? ldd : lds) * n;
  double *A = (double *)((char *)malloc(nn * 8 + off) + off), *d = malloc(n * 8), *e = malloc(n * 8),
         *tau = malloc(n * 8), *work = (double *)((char *)malloc(n * 32 * 8 + off) + off);
  float *As = (float *)((char *)malloc(nn * 4 + off) + off), *ds = malloc(n * 4), *es = malloc(n * 4),
        *taus = malloc(n * 4), *works = (float *)((char *)malloc(n * 32 * 4 + off) + off);
  printf("  matrix address mod 4096: %zu, workspace mod 4096: %zu, lda FP64 %d FP32 %d\n", (size_t)A % 4096,
         (size_t)work % 4096, ldd, lds);
  double *ts[2][2], ev[2][2][1000];
  for (int p = 0; p < 2; p++) for (int s = 0; s < 2; s++) ts[p][s] = malloc(reps * 8);
  printf("Part 2: %s  n = %d, median of %d (ms)\n", path, n, reps);
  for (int mode = 0; mode < 2; mode++) {  // 0 interleaved, 1 blocked
    for (int prec = 0; prec < 2; prec++) {
      if (mode == 0) {
        for (int r = -1; r < reps; r++) for (int sym = 1; sym >= 0; sym--) {
          double t0;
          if (prec == 0) { load_d(A, ldd, H, n); t0 = now(); sytrd_d(n, A, ldd, d, e, tau, work, sym); }
          else { load_s(As, lds, Hs, n); t0 = now(); sytrd_s(n, As, lds, ds, es, taus, works, sym); }
          if (r >= 0) ts[prec][sym][r] = 1e3 * (now() - t0);
          if (r == reps - 1) {
            if (prec == 0) { dsterf_(&n, d, e, &info); for (lint k = 0; k < n; k++) ev[prec][sym][k] = d[k]; }
            else { ssterf_(&n, ds, es, &info); for (lint k = 0; k < n; k++) ev[prec][sym][k] = ds[k]; }
          }
        }
      } else {
        for (int sym = 1; sym >= 0; sym--) for (int r = -1; r < reps; r++) {
          double t0;
          if (prec == 0) { load_d(A, ldd, H, n); t0 = now(); sytrd_d(n, A, ldd, d, e, tau, work, sym); }
          else { load_s(As, lds, Hs, n); t0 = now(); sytrd_s(n, As, lds, ds, es, taus, works, sym); }
          if (r >= 0) ts[prec][sym][r] = 1e3 * (now() - t0);
        }
      }
      double s1 = median(ts[prec][1], reps), s0 = median(ts[prec][0], reps);
      double dw = 0; for (lint k = 0; k < n; k++) dw = fmax(dw, fabs(ev[prec][0][k] - ev[prec][1][k]));
      printf("  %-11s %s  symmetric %7.3f  unsymmetrized %7.3f  ratio %.2f%s\n", mode ? "blocked" : "interleaved",
             prec ? "FP32" : "FP64", s1, s0, s0 / s1,
             mode ? "" : (dw < 1 ? "" : ""));
      if (mode == 0) printf("              eigenvalues unsymmetrized vs symmetric: max |dw| %.1e Eh\n", dw);
    }
  }
}

int main(int argc, char **argv) {
  int reps = argc > 2 ? atoi(argv[2]) : 21;
  if (getenv("UNSYM_GEMV")) gemv_trans = getenv("UNSYM_GEMV");
  printf("unsymmetrized gemv: '%s'\n", gemv_trans);
  components(201);
  if (argc > 1) reduction(argv[1], reps);
  return 0;
}
