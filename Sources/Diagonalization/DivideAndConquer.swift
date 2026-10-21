// The tridiagonal eigenproblem between the reduction and the back-transformations: Accelerate's divide and conquer
// (xstedc, COMPZ = 'I'), unchanged. Eigenvalues ascending in d, eigenvectors of the tridiagonal matrix in z.
import CLAPACK

/// Workspace for divideAndConquer: (elements of Real, Int32 elements).
public func divideAndConquerWorkspace(_ n: Int) -> (Int, Int) { (1 + 4 * n + n * n, 3 + 5 * n) }

/// Diagonal d (n) and subdiagonal e (n - 1) -> eigenvalues in d, eigenvectors in z (n x n, ldz). Returns LAPACK's info.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func divideAndConquer<Real: BandReductionScalar>(
  _ n: Int, _ d: UnsafeMutablePointer<Real>, _ e: UnsafeMutablePointer<Real>, _ z: UnsafeMutablePointer<Real>,
  _ ldz: Int, _ work: UnsafeMutablePointer<Real>, _ lwork: Int, _ iwork: UnsafeMutablePointer<Int32>, _ liwork: Int
) -> Int32 {
  Real.stedc(n, d, e, z, ldz, work, lwork, iwork, liwork)
}
