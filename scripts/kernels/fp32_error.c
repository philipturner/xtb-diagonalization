// Where does the FP32 eigensolve's error in the occupied subspace come from, and does one FP64 correction step remove
// it? (FP32 early SCC cycles, XTB_SCC_FP32_RMSQ.)
//   fp32_error HS.bin NOCC
//   clang -O2 fp32_error.c -framework Accelerate -o fp32_error
// H' = X^T H X in FP64 (S = U^T U, X = U^-1), as xTB forms it. Reference: FP64 dsytrd + dstedc + dormtr. Variants, each
// stage in FP32 (the matrix or vectors rounded on entry, promoted on exit) or FP64:
//   all FP32           ssytrd + sstedc + sormtr (xTB's FP32 path, with Accelerate's reduction)
//   FP32 reduction     ssytrd; its d, e, V, tau promoted: dstedc + dormtr
//   FP32 D&C           dsytrd; d, e rounded: sstedc, Z promoted; dormtr
//   FP32 back-transf.  dsytrd + dstedc; V, tau, Z rounded: sormtr
//   all FP32 + 1 step  all FP32, then one FP64 first-order correction of the occupied block: residual
//                      R = H' C_occ - C_occ diag(e), Y = C_vir^T R, C_occ -= C_vir (Y_ai / (e_a - e_i)), Cholesky QR
// Errors of the lowest NOCC vectors against the reference: sin of the largest principal angle (2-norm of
// C_vir,ref^T C_occ after orthonormalizing C_occ), max |P - P_ref| with P = C_occ C_occ^T, max |eigenvalue error| over
// the occupied ones, and the HOMO-LUMO gap (the correction divides by gaps).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef int lint;
#define BLAS(T, P) \
extern void P##sytrd_(const char *, const lint *, T *, const lint *, T *, T *, T *, T *, const lint *, lint *); \
extern void P##stedc_(const char *, const lint *, T *, T *, T *, const lint *, T *, const lint *, lint *, \
                      const lint *, lint *); \
extern void P##ormtr_(const char *, const char *, const char *, const lint *, const lint *, const T *, const lint *, \
                      const T *, T *, const lint *, T *, const lint *, lint *);
BLAS(double, d)
BLAS(float, s)
extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dtrtri_(const char *, const char *, const lint *, double *, const lint *, lint *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const lint *, const lint *, const double *,
                   const double *, const lint *, double *, const lint *);
extern void dtrsm_(const char *, const char *, const char *, const char *, const lint *, const lint *, const double *,
                   const double *, const lint *, double *, const lint *);
extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *, const double *,
                   const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dsyrk_(const char *, const char *, const lint *, const lint *, const double *, const double *, const lint *,
                   const double *, double *, const lint *);
extern void dsyevd_(const char *, const char *, const lint *, double *, const lint *, double *, double *, const lint *,
                    lint *, const lint *, lint *);
extern void dgesvd_(const char *, const char *, const lint *, const lint *, double *, const lint *, double *, double *,
                    const lint *, double *, const lint *, double *, const lint *, lint *);

static lint n, m;
static double *Hp;  // H' (n x n, FP64)

// Full FP64 or mixed eigensolve: mode 0 all FP64, 1 all FP32, 2 FP32 reduction, 3 FP32 D&C, 4 FP32 back-transf.
static void solve(int mode, double *w, double *Z) {
  size_t nn = (size_t)n * n;
  lint info, lwork = 1 + 4 * n + n * n + 64 * n, liwork = 3 + 5 * n;
  double *A = malloc(nn * 8), *e = malloc(n * 8), *tau = malloc(n * 8), *work = malloc(lwork * 8);
  float *As = malloc(nn * 4), *ds = malloc(n * 4), *es = malloc(n * 4), *taus = malloc(n * 4), *Zs = malloc(nn * 4);
  float *works = malloc(lwork * 4);
  lint *iwork = malloc(liwork * sizeof(lint));
  if (mode == 1 || mode == 2) {
    for (size_t i = 0; i < nn; i++) As[i] = (float)Hp[i];
    ssytrd_("U", &n, As, &n, ds, es, taus, works, &lwork, &info);
    for (size_t i = 0; i < nn; i++) A[i] = As[i];
    for (lint i = 0; i < n; i++) { w[i] = ds[i]; e[i] = es[i]; tau[i] = taus[i]; }
  } else {
    memcpy(A, Hp, nn * 8);
    dsytrd_("U", &n, A, &n, w, e, tau, work, &lwork, &info);
  }
  if (mode == 1 || mode == 3) {
    for (lint i = 0; i < n; i++) { ds[i] = (float)w[i]; es[i] = (float)e[i]; }
    sstedc_("I", &n, ds, es, Zs, &n, works, &lwork, iwork, &liwork, &info);
    for (lint i = 0; i < n; i++) w[i] = ds[i];
    for (size_t i = 0; i < nn; i++) Z[i] = Zs[i];
  } else {
    dstedc_("I", &n, w, e, Z, &n, work, &lwork, iwork, &liwork, &info);
  }
  if (mode == 1 || mode == 4) {
    for (size_t i = 0; i < nn; i++) { As[i] = (float)A[i]; Zs[i] = (float)Z[i]; }
    for (lint i = 0; i < n; i++) taus[i] = (float)tau[i];
    sormtr_("L", "U", "N", &n, &n, As, &n, taus, Zs, &n, works, &lwork, &info);
    for (size_t i = 0; i < nn; i++) Z[i] = Zs[i];
  } else {
    dormtr_("L", "U", "N", &n, &n, A, &n, tau, Z, &n, work, &lwork, &info);
  }
  free(A); free(e); free(tau); free(work); free(As); free(ds); free(es); free(taus); free(Zs); free(works); free(iwork);
}

// Orthonormalize the first m columns of C (n x m) by Cholesky QR in FP64.
static void cholqr(double *C) {
  double one = 1, zero = 0; lint info;
  double *G = malloc((size_t)m * m * 8);
  dsyrk_("U", "T", &m, &n, &one, C, &n, &zero, G, &m);
  dpotrf_("U", &m, G, &m, &info);
  dtrsm_("R", "U", "N", "N", &n, &m, &one, G, &m, C, &n);
  free(G);
}

// One FP64 first-order correction of the occupied block of (w, Z).
static void correct(const double *w, double *Z) {
  double one = 1, zero = 0, mone = -1; lint nv = n - m;
  double *HC = malloc((size_t)n * m * 8), *Y = malloc((size_t)nv * m * 8);
  dgemm_("N", "N", &n, &m, &n, &one, Hp, &n, Z, &n, &zero, HC, &n);
  // Residual R = H' C_occ - C_occ diag(e): projecting H' C_occ alone would mix in the FP32 vectors' own
  // non-orthogonality (~1e-6, the size of the error being corrected).
  for (lint i = 0; i < m; i++)
    for (lint k = 0; k < n; k++) HC[k + (size_t)i * n] -= w[i] * Z[k + (size_t)i * n];
  dgemm_("T", "N", &nv, &m, &n, &one, Z + (size_t)m * n, &n, HC, &n, &zero, Y, &nv);
  for (lint i = 0; i < m; i++)
    for (lint a = 0; a < nv; a++) Y[a + (size_t)i * nv] /= (w[m + a] - w[i]);
  dgemm_("N", "N", &n, &m, &nv, &mone, Z + (size_t)m * n, &n, Y, &nv, &one, Z, &n);
  cholqr(Z);
  free(HC); free(Y);
}

// The cheaper correction: first-order step of each vector z_i against all vectors z_j, j != i (the virtual ones and
// the block's own; pairs closer than 1e-3 Eh skipped), Rayleigh quotients e_i + z_i^T R_i / z_i^T z_i for the eigenvalues, Cholesky QR; no Rayleigh-Ritz.
static void correct_all(double *w, double *Z) {
  double one = 1, zero = 0, mone = -1;
  double *R = malloc((size_t)n * m * 8), *Y = malloc((size_t)n * m * 8), *T = malloc((size_t)n * m * 8);
  double *rq = malloc(m * 8);
  dgemm_("N", "N", &n, &m, &n, &one, Hp, &n, Z, &n, &zero, R, &n);
  for (lint i = 0; i < m; i++) {
    double zz = 0, zr = 0;
    for (lint k = 0; k < n; k++) R[k + (size_t)i * n] -= w[i] * Z[k + (size_t)i * n];
    for (lint k = 0; k < n; k++) { zz += Z[k + (size_t)i * n] * Z[k + (size_t)i * n]; zr += Z[k + (size_t)i * n] * R[k + (size_t)i * n]; }
    rq[i] = w[i] + zr / zz;
  }
  dgemm_("T", "N", &n, &m, &n, &one, Z, &n, R, &n, &zero, Y, &n);
  for (lint i = 0; i < m; i++)
    for (lint j = 0; j < n; j++) {
      double g = w[j] - w[i];
      Y[j + (size_t)i * n] = (j == i || fabs(g) < 1e-3) ? 0 : Y[j + (size_t)i * n] / g;
    }
  memcpy(T, Z, (size_t)n * m * 8);
  dgemm_("N", "N", &n, &m, &n, &mone, Z, &n, Y, &n, &one, T, &n);
  memcpy(Z, T, (size_t)n * m * 8);
  cholqr(Z);
  for (lint i = 0; i < m; i++) w[i] = rq[i];
  // Rayleigh-Ritz on each cluster of consecutive eigenvalues closer than 1e-3 Eh (the skipped pairs)
  int clusters = 0, largest = 1;
  for (lint i0 = 0; i0 < m;) {
    lint i1 = i0 + 1;
    while (i1 < m && w[i1] - w[i1 - 1] < 1e-3) i1++;
    lint c = i1 - i0;
    if (c > 1) {
      clusters++; if (c > largest) largest = c;
      double *AZ = malloc((size_t)n * c * 8), *G = malloc((size_t)c * c * 8), *ev = malloc(c * 8);
      lint lw = 1 + 6 * c + 2 * c * c, liw = 3 + 5 * c, info; double *wk = malloc(lw * 8); lint *iw = malloc(liw * sizeof(lint));
      double *Zc = Z + (size_t)i0 * n;
      dgemm_("N", "N", &n, &c, &n, &one, Hp, &n, Zc, &n, &zero, AZ, &n);
      dgemm_("T", "N", &c, &c, &n, &one, Zc, &n, AZ, &n, &zero, G, &c);
      dsyevd_("V", "U", &c, G, &c, ev, wk, &lw, iw, &liw, &info);
      dgemm_("N", "N", &n, &c, &c, &one, Zc, &n, G, &c, &zero, AZ, &n);
      memcpy(Zc, AZ, (size_t)n * c * 8);
      for (lint j = 0; j < c; j++) w[i0 + j] = ev[j];
      free(AZ); free(G); free(ev); free(wk); free(iw);
    }
    i0 = i1;
  }
  printf("  (clusters below 1e-3 Eh: %d, largest %d)\n", clusters, largest);
  free(R); free(Y); free(T); free(rq);
}

static void report(const char *name, const double *w, double *Z, const double *wref, const double *Zref) {
  double one = 1, zero = 0; lint nv = n - m, info;
  double *C = malloc((size_t)n * m * 8); memcpy(C, Z, (size_t)n * m * 8);
  cholqr(C);
  // sin(largest principal angle) = || Zref_vir^T C ||_2
  double *M = malloc((size_t)nv * m * 8), *sv = malloc((nv < m ? nv : m) * 8);
  dgemm_("T", "N", &nv, &m, &n, &one, Zref + (size_t)m * n, &n, C, &n, &zero, M, &nv);
  lint lw = 5 * (nv + m) + 64 * (nv > m ? nv : m), ld1 = 1; double *wk = malloc(lw * 8), du;
  dgesvd_("N", "N", &nv, &m, M, &nv, sv, &du, &ld1, &du, &ld1, wk, &lw, &info);
  // max |P - Pref|
  double *P = malloc((size_t)n * n * 8), *Pr = malloc((size_t)n * n * 8);
  dsyrk_("U", "N", &n, &m, &one, C, &n, &zero, P, &n);
  dsyrk_("U", "N", &n, &m, &one, Zref, &n, &zero, Pr, &n);
  double dp = 0;
  for (lint j = 0; j < n; j++) for (lint i = 0; i <= j; i++) dp = fmax(dp, fabs(P[i + (size_t)j * n] - Pr[i + (size_t)j * n]));
  double dw = 0; for (lint i = 0; i < m; i++) dw = fmax(dw, fabs(w[i] - wref[i]));
  printf("  %-24s sin(angle) %.1e   max|dP| %.1e   occupied eigenvalues %.1e Eh\n", name, sv[0], dp, dw);
  free(C); free(M); free(sv); free(wk); free(P); free(Pr);
}

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); m = atoi(argv[2]);
  int n4; fread(&n4, 4, 1, f); n = n4; size_t nn = (size_t)n * n;
  double *H = malloc(nn * 8), *S = malloc(nn * 8);
  fread(H, 8, nn, f); fread(S, 8, nn, f); fclose(f);
  lint info; double one = 1.0;
  dpotrf_("U", &n, S, &n, &info);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) S[i + j * n] = 0.0;
  dtrtri_("U", "N", &n, S, &n, &info);
  dtrmm_("R", "U", "N", "N", &n, &n, &one, S, &n, H, &n);
  dtrmm_("L", "U", "T", "N", &n, &n, &one, S, &n, H, &n);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) H[i + j * n] = H[j + i * n];
  Hp = H;
  double *wref = malloc(n * 8), *Zref = malloc(nn * 8), *w = malloc(n * 8), *Z = malloc(nn * 8);
  solve(0, wref, Zref);
  printf("%s  n = %d, occupied %d, HOMO-LUMO gap %.4f Eh\n", argv[1], n, m, wref[m] - wref[m - 1]);
  static const char *names[5] = {"", "all FP32", "FP32 reduction", "FP32 D&C", "FP32 back-transformation"};
  for (int mode = 1; mode <= 4; mode++) { solve(mode, w, Z); report(names[mode], w, Z, wref, Zref); }
  solve(1, w, Z); correct(w, Z); report("all FP32 + 1 FP64 step", w, Z, wref, Zref);
  correct(w, Z); report("all FP32 + 2 FP64 steps", w, Z, wref, Zref);
  solve(1, w, Z); correct_all(w, Z); report("all FP32 + cheap step", w, Z, wref, Zref);
  return 0;
}
