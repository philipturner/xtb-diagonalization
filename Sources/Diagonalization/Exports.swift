// C entry points with the names and signatures of all functions xtb/src/mctc/lapack/sytrd.c exports, so the static
// library links into libxtb in its place (xtb's CMake option XTB_SYTRD_LIBRARIES, scripts/build_xtb_swift.sh).
// Fortran passes every argument by reference.

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

/// Fortran: call xtb_dsytrd_full(n, a, lda, d, e, tau, work): the same reduction on the full symmetric matrix
/// (both triangles stored and updated), work of size n * 32
@_cdecl("xtb_dsytrd_full")
public func xtb_dsytrd_full(
  _ n: UnsafePointer<Int32>, _ a: UnsafeMutablePointer<Double>, _ lda: UnsafePointer<Int32>,
  _ d: UnsafeMutablePointer<Double>, _ e: UnsafeMutablePointer<Double>, _ tau: UnsafeMutablePointer<Double>,
  _ work: UnsafeMutablePointer<Double>
) {
  tridiagonalize(Int(n.pointee), a, Int(lda.pointee), d, e, tau, work, symmetric: false)
}

/// Fortran: call xtb_dsymmetrize_pad(n, src, lds, dst, ldd): dst(1:n, 1:n) = the symmetric matrix whose upper
/// triangle is src's (src's lower triangle is not read), in 32 x 32 tiles
@_cdecl("xtb_dsymmetrize_pad")
public func xtb_dsymmetrize_pad(
  _ n: UnsafePointer<Int32>, _ src: UnsafePointer<Double>, _ lds: UnsafePointer<Int32>,
  _ dst: UnsafeMutablePointer<Double>, _ ldd: UnsafePointer<Int32>
) {
  let n = Int(n.pointee), lds = Int(lds.pointee), ldd = Int(ldd.pointee), tile = 32
  for j0 in stride(from: 0, to: n, by: tile) {
    let j1 = min(j0 + tile, n)
    for i0 in stride(from: 0, through: j0, by: tile) {
      let i1 = min(i0 + tile, n)
      for j in j0..<j1 {
        for i in i0..<min(i1, j + 1) {
          let v = src[i + j * lds]
          dst[i + j * ldd] = v
          dst[j + i * ldd] = v
        }
      }
    }
  }
}
