// Divide and conquer for the lowest m eigenvectors only: how much of xstedc can a top merge that forms only the
// needed columns save? On the tridiagonal matrices of dumped GFN2 SCC cycles (H' = U^-T H U^-1, reduced by sytrd.c):
//   full     dstedc('I') on all n
//   tear     the rank-one tearing at n1 = n / 2 (as dlaed0), dstedc('I') on both halves
//   merge    LAPACK's top merge: dlaed2 (deflation) + this file's port of dlaed3 (dlaed4 per root, the stabilized
//            vectors of the rank-one problem, then the gemm Q2 S that forms the merged eigenvectors) + dlamrg
//   partial  the same merge with the gemm restricted to the roots among the m lowest eigenvalues (sorted order)
// Times per part (median of 21) and checks: eigenvalues vs dstedc, residual max|T z - lambda z| / max|lambda| and
// orthogonality max|Z^T Z - I| of the m lowest vectors from the partial merge. m = n / 2.
//   clang -O3 -mcpu=apple-m1 stedc_partial.c -framework Accelerate -o stedc_partial
//   VECLIB_MAXIMUM_THREADS=1 ./stedc_partial build/dumps/facet_closed/hs_1.bin ...
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../xtb/src/mctc/lapack/sytrd.c"

extern void dstedc_(const char *, const lint *, double *, double *, double *, const lint *, double *, const lint *,
                    lint *, const lint *, lint *);
extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dsygst_(const lint *, const char *, const lint *, double *, const lint *, const double *, const lint *,
                    lint *);
extern void dlaed2_(lint *, const lint *, const lint *, double *, double *, const lint *, lint *, double *, double *,
                    double *, double *, double *, lint *, lint *, lint *, lint *, lint *);
extern void dlaed4_(const lint *, const lint *, const double *, const double *, double *, const double *, double *,
                    lint *);
extern void dlamrg_(const lint *, const lint *, const double *, const lint *, const lint *, lint *);
extern double dnrm2_(const lint *, const double *, const lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

// Reference dlaed3 (LAPACK 3.12), with the final gemm restricted to the first kc of the K roots (kc = K: dlaed3).
// q (n x n, ldq): rows 1 .. K of columns 1 .. K receive the rank-one problem's vectors, then columns 1 .. kc the
// merged eigenvectors. s: workspace (n1 + n2) * K. Returns info.
static lint laed3(lint k, lint n, lint n1, double *d, double *q, lint ldq, double rho, double *dlamda,
                  const double *q2, const lint *indx, const lint *ctot, double *w, double *s, lint kc,
                  double *t_roots, double *t_gemm) {
  lint info = 0, i1 = 1;
  double t0 = now();
  for (lint j = 1; j <= k; j++) {
    dlaed4_(&k, &j, dlamda, w, q + (size_t)(j - 1) * ldq, &rho, &d[j - 1], &info);
    if (info) return info;
  }
  if (k == 2) {
    for (lint j = 0; j < k; j++) {
      w[0] = q[(size_t)j * ldq]; w[1] = q[1 + (size_t)j * ldq];
      q[(size_t)j * ldq] = w[indx[0] - 1]; q[1 + (size_t)j * ldq] = w[indx[1] - 1];
    }
  } else if (k > 2) {
    memcpy(s, w, k * 8);
    for (lint i = 0; i < k; i++) w[i] = q[i + (size_t)i * ldq];
    for (lint j = 0; j < k; j++) {
      for (lint i = 0; i < j; i++) w[i] *= q[i + (size_t)j * ldq] / (dlamda[i] - dlamda[j]);
      for (lint i = j + 1; i < k; i++) w[i] *= q[i + (size_t)j * ldq] / (dlamda[i] - dlamda[j]);
    }
    for (lint i = 0; i < k; i++) w[i] = copysign(sqrt(-w[i]), s[i]);
    for (lint j = 0; j < k; j++) {
      double *qj = q + (size_t)j * ldq;
      for (lint i = 0; i < k; i++) s[i] = w[i] / qj[i];
      double temp = dnrm2_(&k, s, &i1);
      for (lint i = 0; i < k; i++) qj[i] = s[indx[i] - 1] / temp;
    }
  }
  double t1 = now();
  lint n2 = n - n1, n12 = ctot[0] + ctot[1], n23 = ctot[1] + ctot[2];
  double one = 1, zero = 0;
  // S = Q(ctot1 + 1 .. ctot1 + n23, 1 .. kc), then Q(n1 + 1 .., 1 .. kc) = Q2(iq2) S
  for (lint j = 0; j < kc; j++) memcpy(s + (size_t)j * n23, q + ctot[0] + (size_t)j * ldq, n23 * 8);
  if (n23) dgemm_("N", "N", &n2, &kc, &n23, &one, q2 + n1 * n12, &n2, s, &n23, &zero, q + n1, &ldq);
  else for (lint j = 0; j < kc; j++) memset(q + n1 + (size_t)j * ldq, 0, n2 * 8);
  for (lint j = 0; j < kc; j++) memcpy(s + (size_t)j * n12, q + (size_t)j * ldq, n12 * 8);
  if (n12) dgemm_("N", "N", &n1, &kc, &n12, &one, q2, &n1, s, &n12, &zero, q, &ldq);
  else for (lint j = 0; j < kc; j++) memset(q + (size_t)j * ldq, 0, n1 * 8);
  *t_roots += t1 - t0; *t_gemm += now() - t1;
  return 0;
}

int main(int argc, char **argv) {
  for (int ai = 1; ai < argc; ai++) {
    FILE *f = fopen(argv[ai], "rb"); int n32; fread(&n32, 4, 1, f);
    lint n = n32, info, ldq = n;
    double *H = malloc((size_t)n * n * 8), *S = malloc((size_t)n * n * 8);
    fread(H, 8, (size_t)n * n, f); fread(S, 8, (size_t)n * n, f); fclose(f);
    lint one_i = 1;
    dpotrf_("U", &n, S, &n, &info); dsygst_(&one_i, "U", &n, H, &n, S, &n, &info);
    double *d0 = malloc(n * 8), *e0 = malloc(n * 8), *tau = malloc(n * 8), *wk = malloc((size_t)n * 32 * 8);
    xtb_dsytrd_upper(&n, H, &n, d0, e0, tau, wk);
    lint m = n / 2, n1 = n / 2, n2 = n - n1;
    lint lwork = 1 + 4 * n + 2 * n * n, liwork = 3 + 5 * n;
    double *d = malloc(n * 8), *e = malloc(n * 8), *Zr, *Q, *work = malloc(lwork * 8), *dref = malloc(n * 8);
    lint *iwork = malloc(liwork * sizeof(lint));
    posix_memalign((void **)&Zr, 16384, (size_t)n * n * 8); posix_memalign((void **)&Q, 16384, (size_t)n * n * 8);
    double *q2 = malloc(((size_t)n1 * n1 + (size_t)n2 * n2) * 8), *z = malloc(n * 8), *dlamda = malloc(n * 8),
           *w = malloc(n * 8), *s = malloc((size_t)n * n * 8), *dsort = malloc(n * 8);
    lint *indxq = malloc(n * sizeof(lint)), *indx = malloc(n * sizeof(lint)), *indxc = malloc(n * sizeof(lint)),
         *indxp = malloc(n * sizeof(lint)), *coltyp = malloc(4 * n * sizeof(lint));
    double tf[21], th[21], t2[21], tr[21], tg[21], tp[21], tgp[21];
    lint K = 0, kc = 0;
    for (int r = -2; r < 21; r++) {
      // full
      memcpy(d, d0, n * 8); memcpy(e, e0, n * 8);
      double t0 = now();
      dstedc_("I", &n, d, e, Zr, &n, work, &lwork, iwork, &liwork, &info);
      double t1 = now();
      memcpy(dref, d, n * 8);
      for (int pass = 0; pass < 2; pass++) {  // 0: full merge, 1: partial
        memcpy(d, d0, n * 8); memcpy(e, e0, n * 8);
        double ta = now();
        double rho = e[n1 - 1];
        d[n1 - 1] -= fabs(rho); d[n1] -= fabs(rho);
        memset(Q, 0, (size_t)n * n * 8);
        dstedc_("I", &n1, d, e, Q, &ldq, work, &lwork, iwork, &liwork, &info);
        dstedc_("I", &n2, d + n1, e + n1, Q + n1 + (size_t)n1 * ldq, &ldq, work, &lwork, iwork, &liwork, &info);
        double tb = now();
        for (lint i = 0; i < n1; i++) indxq[i] = i + 1;
        for (lint i = 0; i < n2; i++) indxq[n1 + i] = i + 1;
        for (lint i = 0; i < n1; i++) z[i] = Q[n1 - 1 + (size_t)i * ldq];
        for (lint i = 0; i < n2; i++) z[n1 + i] = Q[n1 + (size_t)(n1 + i) * ldq];
        dlaed2_(&K, &n, &n1, d, Q, &ldq, indxq, &rho, z, dlamda, w, q2, indx, indxc, indxp, coltyp, &info);
        double tc = now(), troots = 0, tgemm = 0;
        // pass 0: all K roots; pass 1: kc, the largest root index among the m lowest (from pass 0)
        if (pass == 0) kc = K;
        if (K) info = laed3(K, n, n1, d, Q, ldq, rho, dlamda, q2, indxc, coltyp, w, s, kc, &troots, &tgemm);
        lint k1 = K, k2 = n - K, m1 = 1, mm1 = -1;
        if (K) dlamrg_(&k1, &k2, d, &m1, &mm1, indxq); else for (lint i = 0; i < n; i++) indxq[i] = i + 1;
        double td = now();
        if (pass == 0) {
          // kc for the partial pass: the largest unsorted root index among the m lowest
          kc = 0;
          for (lint i = 0; i < m; i++) if (indxq[i] <= K && indxq[i] > kc) kc = indxq[i];
          if (r >= 0) { th[r] = 1e3 * (tb - ta); t2[r] = 1e3 * (tc - tb); tr[r] = 1e3 * troots; tg[r] = 1e3 * tgemm;
                        tf[r] = 1e3 * (t1 - t0); }
        } else if (r >= 0) { tp[r] = 1e3 * (td - ta); tgp[r] = 1e3 * tgemm; }
      }
    }
    // checks on the partial result: the m lowest in sorted order
    double ev = 0, res = 0, orth = 0, scale = 0;
    for (lint i = 0; i < n; i++) dsort[i] = d[indxq[i] - 1];
    for (lint i = 0; i < n; i++) { ev = fmax(ev, fabs(dsort[i] - dref[i])); scale = fmax(scale, fabs(dref[i])); }
    for (lint a = 0; a < m; a++) {
      const double *za = Q + (size_t)(indxq[a] - 1) * ldq;
      for (lint i = 0; i < n; i++) {
        double tz = d0[i] * za[i] + (i > 0 ? e0[i - 1] * za[i - 1] : 0) + (i < n - 1 ? e0[i] * za[i + 1] : 0);
        res = fmax(res, fabs(tz - dsort[a] * za[i]));
      }
      for (lint b = 0; b <= a; b++) {
        const double *zb = Q + (size_t)(indxq[b] - 1) * ldq; double g = 0;
        for (lint i = 0; i < n; i++) g += za[i] * zb[i];
        orth = fmax(orth, fabs(g - (a == b)));
      }
    }
    double *all[7] = {tf, th, t2, tr, tg, tp, tgp}, md[7];
    for (int q = 0; q < 7; q++) { qsort(all[q], 21, 8, cmp); md[q] = all[q][10]; }
    printf("%s n = %d, K = %d non-deflated, m = %d needs %d roots\n", argv[ai], n, K, m, kc);
    printf("  dstedc %.3f ms | halves %.3f + deflation %.3f + roots %.3f + gemm %.3f (merge total %.3f)"
           " | partial total %.3f (gemm %.3f): %.2fx\n", md[0], md[1], md[2], md[3], md[4], md[1] + md[2] + md[3] + md[4],
           md[5], md[6], md[0] / md[5]);
    printf("  partial: eigenvalues %.1e, residual %.1e, orthogonality %.1e\n", ev, res / scale, orth);
  }
  return 0;
}
