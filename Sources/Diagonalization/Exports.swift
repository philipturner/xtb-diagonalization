// C entry points with the names and signatures of xtb/src/mctc/lapack/sytrd.c, so the static library links into
// libxtb in its place (xtb's CMake option XTB_SYTRD_LIBRARY, scripts/build_xtb_swift.sh). Fortran passes every
// argument by reference.

/// Fortran: call xtb_dsytrd_upper(n, a, lda, d, e, tau, work), work of size n * 32
@_cdecl("xtb_dsytrd_upper")
public func xtb_dsytrd_upper(
  _ n: UnsafePointer<Int32>, _ a: UnsafeMutablePointer<Double>, _ lda: UnsafePointer<Int32>,
  _ d: UnsafeMutablePointer<Double>, _ e: UnsafeMutablePointer<Double>, _ tau: UnsafeMutablePointer<Double>,
  _ work: UnsafeMutablePointer<Double>
) {
  tridiagonalize(Int(n.pointee), a, Int(lda.pointee), d, e, tau, work)
}

/// Fortran: call xtb_ssytrd_upper(n, a, lda, d, e, tau, work), work of size n * 32
@_cdecl("xtb_ssytrd_upper")
public func xtb_ssytrd_upper(
  _ n: UnsafePointer<Int32>, _ a: UnsafeMutablePointer<Float>, _ lda: UnsafePointer<Int32>,
  _ d: UnsafeMutablePointer<Float>, _ e: UnsafeMutablePointer<Float>, _ tau: UnsafeMutablePointer<Float>,
  _ work: UnsafeMutablePointer<Float>
) {
  tridiagonalize(Int(n.pointee), a, Int(lda.pointee), d, e, tau, work)
}
