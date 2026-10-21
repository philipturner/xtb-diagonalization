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
void ssterf_(const lint *n, float *d, float *e, lint *info);

#endif
