// Accelerate's legacy BLAS/LAPACK entry points (Fortran ABI, 32-bit integers, no hidden string lengths), declared
// like xtb/src/mctc/lapack/sytrd.c declares them. These are what libxtb links when built with BLA_VENDOR=Apple.
#ifndef CLAPACK_H
#define CLAPACK_H

typedef int lint;

// BLAS, FP64
void dgemv_(const char *trans, const lint *m, const lint *n, const double *alpha, const double *a, const lint *lda,
            const double *x, const lint *incx, const double *beta, double *y, const lint *incy);
void dsymv_(const char *uplo, const lint *n, const double *alpha, const double *a, const lint *lda, const double *x,
            const lint *incx, const double *beta, double *y, const lint *incy);
void dsyr2k_(const char *uplo, const char *trans, const lint *n, const lint *k, const double *alpha, const double *a,
             const lint *lda, const double *b, const lint *ldb, const double *beta, double *c, const lint *ldc);
void dsyr2_(const char *uplo, const lint *n, const double *alpha, const double *x, const lint *incx, const double *y,
            const lint *incy, double *a, const lint *lda);
double ddot_(const lint *n, const double *x, const lint *incx, const double *y, const lint *incy);
void daxpy_(const lint *n, const double *alpha, const double *x, const lint *incx, double *y, const lint *incy);
void dscal_(const lint *n, const double *alpha, double *x, const lint *incx);
void dgemm_(const char *transa, const char *transb, const lint *m, const lint *n, const lint *k, const double *alpha,
            const double *a, const lint *lda, const double *b, const lint *ldb, const double *beta, double *c,
            const lint *ldc);
void dger_(const lint *m, const lint *n, const double *alpha, const double *x, const lint *incx, const double *y,
           const lint *incy, double *a, const lint *lda);
void dsymm_(const char *side, const char *uplo, const lint *m, const lint *n, const double *alpha, const double *a,
            const lint *lda, const double *b, const lint *ldb, const double *beta, double *c, const lint *ldc);
void dtrmm_(const char *side, const char *uplo, const char *transa, const char *diag, const lint *m, const lint *n,
            const double *alpha, const double *a, const lint *lda, double *b, const lint *ldb);

// BLAS, FP32 (sdot_ is left out: its legacy ABI returns a double)
void sgemv_(const char *trans, const lint *m, const lint *n, const float *alpha, const float *a, const lint *lda,
            const float *x, const lint *incx, const float *beta, float *y, const lint *incy);
void ssyr2k_(const char *uplo, const char *trans, const lint *n, const lint *k, const float *alpha, const float *a,
             const lint *lda, const float *b, const lint *ldb, const float *beta, float *c, const lint *ldc);
void ssyr2_(const char *uplo, const lint *n, const float *alpha, const float *x, const lint *incx, const float *y,
            const lint *incy, float *a, const lint *lda);
void sgemm_(const char *transa, const char *transb, const lint *m, const lint *n, const lint *k, const float *alpha,
            const float *a, const lint *lda, const float *b, const lint *ldb, const float *beta, float *c,
            const lint *ldc);
void sger_(const lint *m, const lint *n, const float *alpha, const float *x, const lint *incx, const float *y,
           const lint *incy, float *a, const lint *lda);
void ssymm_(const char *side, const char *uplo, const lint *m, const lint *n, const float *alpha, const float *a,
            const lint *lda, const float *b, const lint *ldb, const float *beta, float *c, const lint *ldc);
void strmm_(const char *side, const char *uplo, const char *transa, const char *diag, const lint *m, const lint *n,
            const float *alpha, const float *a, const lint *lda, float *b, const lint *ldb);
void saxpy_(const lint *n, const float *alpha, const float *x, const lint *incx, float *y, const lint *incy);
void sscal_(const lint *n, const float *alpha, float *x, const lint *incx);

// LAPACK
void dlarfg_(const lint *n, double *alpha, double *x, const lint *incx, double *tau);
void slarfg_(const lint *n, float *alpha, float *x, const lint *incx, float *tau);
void dpotrf_(const char *uplo, const lint *n, double *a, const lint *lda, lint *info);
void dtrtri_(const char *uplo, const char *diag, const lint *n, double *a, const lint *lda, lint *info);
void dsytrd_(const char *uplo, const lint *n, double *a, const lint *lda, double *d, double *e, double *tau,
             double *work, const lint *lwork, lint *info);
void ssytrd_(const char *uplo, const lint *n, float *a, const lint *lda, float *d, float *e, float *tau, float *work,
             const lint *lwork, lint *info);
void dsterf_(const lint *n, double *d, double *e, lint *info);
void dgeqrf_(const lint *m, const lint *n, double *a, const lint *lda, double *tau, double *work, const lint *lwork,
             lint *info);
void sgeqrf_(const lint *m, const lint *n, float *a, const lint *lda, float *tau, float *work, const lint *lwork,
             lint *info);
void dlarft_(const char *direct, const char *storev, const lint *n, const lint *k, const double *v, const lint *ldv,
             const double *tau, double *t, const lint *ldt);
void slarft_(const char *direct, const char *storev, const lint *n, const lint *k, const float *v, const lint *ldv,
             const float *tau, float *t, const lint *ldt);
void dlarfb_(const char *side, const char *trans, const char *direct, const char *storev, const lint *m,
             const lint *n, const lint *k, const double *v, const lint *ldv, const double *t, const lint *ldt,
             double *c, const lint *ldc, double *work, const lint *ldwork);
void slarfb_(const char *side, const char *trans, const char *direct, const char *storev, const lint *m,
             const lint *n, const lint *k, const float *v, const lint *ldv, const float *t, const lint *ldt, float *c,
             const lint *ldc, float *work, const lint *ldwork);
void dstedc_(const char *compz, const lint *n, double *d, double *e, double *z, const lint *ldz, double *work,
             const lint *lwork, lint *iwork, const lint *liwork, lint *info);
void dormtr_(const char *side, const char *uplo, const char *trans, const lint *m, const lint *n, const double *a,
             const lint *lda, const double *tau, double *c, const lint *ldc, double *work, const lint *lwork,
             lint *info);
void sstedc_(const char *compz, const lint *n, float *d, float *e, float *z, const lint *ldz, float *work,
             const lint *lwork, lint *iwork, const lint *liwork, lint *info);
// The merge step of divide and conquer (ParallelDivideAndConquer.swift): deflation, one secular-equation root,
// merging two sorted lists. Bound to Accelerate's new LAPACK interface ($NEWLAPACK, 32-bit integers), which Apple
// documents as thread-safe; whether the legacy symbols are safe to call from several threads at once wasn't checked.
void dstedc_new(const char *compz, const lint *n, double *d, double *e, double *z, const lint *ldz, double *work,
                const lint *lwork, lint *iwork, const lint *liwork, lint *info) __asm__("_dstedc$NEWLAPACK");
void sstedc_new(const char *compz, const lint *n, float *d, float *e, float *z, const lint *ldz, float *work,
                const lint *lwork, lint *iwork, const lint *liwork, lint *info) __asm__("_sstedc$NEWLAPACK");
void dlaed2_(lint *k, const lint *n, const lint *n1, double *d, double *q, const lint *ldq, lint *indxq, double *rho,
             double *z, double *dlamda, double *w, double *q2, lint *indx, lint *indxc, lint *indxp, lint *coltyp,
             lint *info) __asm__("_dlaed2$NEWLAPACK");
void slaed2_(lint *k, const lint *n, const lint *n1, float *d, float *q, const lint *ldq, lint *indxq, float *rho,
             float *z, float *dlamda, float *w, float *q2, lint *indx, lint *indxc, lint *indxp, lint *coltyp,
             lint *info) __asm__("_slaed2$NEWLAPACK");
void dlaed4_(const lint *n, const lint *i, const double *d, const double *z, double *delta, const double *rho,
             double *dlam, lint *info) __asm__("_dlaed4$NEWLAPACK");
void slaed4_(const lint *n, const lint *i, const float *d, const float *z, float *delta, const float *rho, float *dlam,
             lint *info) __asm__("_slaed4$NEWLAPACK");
void dlamrg_(const lint *n1, const lint *n2, const double *a, const lint *dtrd1, const lint *dtrd2, lint *index) __asm__("_dlamrg$NEWLAPACK");
void slamrg_(const lint *n1, const lint *n2, const float *a, const lint *dtrd1, const lint *dtrd2, lint *index) __asm__("_slamrg$NEWLAPACK");
void sormtr_(const char *side, const char *uplo, const char *trans, const lint *m, const lint *n, const float *a,
             const lint *lda, const float *tau, float *c, const lint *ldc, float *work, const lint *lwork, lint *info);
void ssyevd_(const char *jobz, const char *uplo, const lint *n, float *a, const lint *lda, float *w, float *work,
             const lint *lwork, lint *iwork, const lint *liwork, lint *info);
void dsyevd_(const char *jobz, const char *uplo, const lint *n, double *a, const lint *lda, double *w, double *work,
             const lint *lwork, lint *iwork, const lint *liwork, lint *info);
void ssterf_(const lint *n, float *d, float *e, lint *info);

#endif
