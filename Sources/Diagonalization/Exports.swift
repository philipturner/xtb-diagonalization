// C entry points with the names and signatures of all functions xtb/src/mctc/lapack/sytrd.c exports, so the static
// library links into libxtb in its place (xtb's CMake option XTB_SYTRD_LIBRARIES, scripts/build_xtb_swift.sh).
// Fortran passes every argument by reference.
import Darwin

/// Fortran: call xtb_dsytrd_upper(n, a, lda, d, e, tau, work), work of size n * 32; on the caller's OpenMP threads, at
/// most one cluster of performance cores (performanceClusterSize), from n = 256 (tridiagonalizeParallel), as sytrd.c
@_cdecl("xtb_dsytrd_upper")
public func xtb_dsytrd_upper(
  _ n: UnsafePointer<Int32>, _ a: UnsafeMutablePointer<Double>, _ lda: UnsafePointer<Int32>,
  _ d: UnsafeMutablePointer<Double>, _ e: UnsafeMutablePointer<Double>, _ tau: UnsafeMutablePointer<Double>,
  _ work: UnsafeMutablePointer<Double>
) {
  let threads = min(openMPThreads(), performanceClusterSize)
  tridiagonalizeParallel(Int(n.pointee), a, Int(lda.pointee), d, e, tau, work, threads: threads)
}

/// Fortran: call xtb_ssytrd_upper(n, a, lda, d, e, tau, work), work of size n * 32; as xtb_dsytrd_upper
@_cdecl("xtb_ssytrd_upper")
public func xtb_ssytrd_upper(
  _ n: UnsafePointer<Int32>, _ a: UnsafeMutablePointer<Float>, _ lda: UnsafePointer<Int32>,
  _ d: UnsafeMutablePointer<Float>, _ e: UnsafeMutablePointer<Float>, _ tau: UnsafeMutablePointer<Float>,
  _ work: UnsafeMutablePointer<Float>
) {
  let threads = min(openMPThreads(), performanceClusterSize)
  tridiagonalizeParallel(Int(n.pointee), a, Int(lda.pointee), d, e, tau, work, threads: threads)
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

/// As sytrd.c's xtb_xormtr_upper: two column blocks on two workers when the caller has more than one OpenMP thread
/// (each block with its own workspace), else serial with the caller's work.
func backTransformTridiagonalOpenMP<T: BandReductionScalar>(
  _ n: Int, _ m: Int, _ a: UnsafePointer<T>, _ lda: Int, _ tau: UnsafePointer<T>, _ z: UnsafeMutablePointer<T>,
  _ ldz: Int, _ work: UnsafeMutablePointer<T>
) {
  guard openMPThreads() > 1, m >= 4 else {
    backTransformTridiagonal(n, m, a, lda, tau, z, ldz, work)
    return
  }
  let width = (m + 1) / 2
  let slices = UnsafeMutablePointer<T>.allocate(capacity: 2 * tridiagonalBackTransformWorkspace(n, width))
  defer { slices.deallocate() }
  backTransformTridiagonalParallel(n, m, a, lda, tau, z, ldz, slices, threads: 2)
}

/// Fortran: call xtb_dormtr_upper(n, m, a, lda, tau, z, ldz, work): z = Q z for m columns, Q from xtb_dsytrd_upper
/// (dormtr 'L', 'U', 'N' by gemm), work of size 32 * (2 * n + 2 * m + 32)
@_cdecl("xtb_dormtr_upper")
public func xtb_dormtr_upper(
  _ n: UnsafePointer<Int32>, _ m: UnsafePointer<Int32>, _ a: UnsafePointer<Double>, _ lda: UnsafePointer<Int32>,
  _ tau: UnsafePointer<Double>, _ z: UnsafeMutablePointer<Double>, _ ldz: UnsafePointer<Int32>,
  _ work: UnsafeMutablePointer<Double>
) {
  backTransformTridiagonalOpenMP(Int(n.pointee), Int(m.pointee), a, Int(lda.pointee), tau, z, Int(ldz.pointee), work)
}

/// Fortran: call xtb_sormtr_upper(n, m, a, lda, tau, z, ldz, work): as xtb_dormtr_upper, Q from xtb_ssytrd_upper
@_cdecl("xtb_sormtr_upper")
public func xtb_sormtr_upper(
  _ n: UnsafePointer<Int32>, _ m: UnsafePointer<Int32>, _ a: UnsafePointer<Float>, _ lda: UnsafePointer<Int32>,
  _ tau: UnsafePointer<Float>, _ z: UnsafeMutablePointer<Float>, _ ldz: UnsafePointer<Int32>,
  _ work: UnsafeMutablePointer<Float>
) {
  backTransformTridiagonalOpenMP(Int(n.pointee), Int(m.pointee), a, Int(lda.pointee), tau, z, Int(ldz.pointee), work)
}

/// The OpenMP threads the calling thread would get (omp_get_max_threads, from the runtime libxtb loads), or 1 inside
/// a parallel region or without an OpenMP runtime: sytrd.c's xtb_omp_threads.
func openMPThreads() -> Int {
  typealias Query = @convention(c) () -> Int32
  let handle = UnsafeMutableRawPointer(bitPattern: -2)  // RTLD_DEFAULT
  guard let maxThreads = dlsym(handle, "omp_get_max_threads"),
        let inParallel = dlsym(handle, "omp_in_parallel") else { return 1 }
  if unsafeBitCast(inParallel, to: Query.self)() != 0 { return 1 }
  return Int(unsafeBitCast(maxThreads, to: Query.self)())
}

/// Fortran: call xtb_dstedc(n, d, e, z, ldz, work, lwork, iwork, liwork, info): dstedc('I', ...) on the caller's
/// OpenMP threads (divideAndConquerParallel), Accelerate's dstedc itself with one or below n = 256
@_cdecl("xtb_dstedc")
public func xtb_dstedc(
  _ n: UnsafePointer<Int32>, _ d: UnsafeMutablePointer<Double>, _ e: UnsafeMutablePointer<Double>,
  _ z: UnsafeMutablePointer<Double>, _ ldz: UnsafePointer<Int32>, _ work: UnsafeMutablePointer<Double>,
  _ lwork: UnsafePointer<Int32>, _ iwork: UnsafeMutablePointer<Int32>, _ liwork: UnsafePointer<Int32>,
  _ info: UnsafeMutablePointer<Int32>
) {
  let n = Int(n.pointee), threads = openMPThreads()
  info.pointee = threads > 1 && n >= 256
    ? divideAndConquerParallel(n, d, e, z, Int(ldz.pointee), threads: threads)
    : Double.stedc(n, d, e, z, Int(ldz.pointee), work, Int(lwork.pointee), iwork, Int(liwork.pointee))
}

/// Fortran: call xtb_sstedc(n, d, e, z, ldz, work, lwork, iwork, liwork, info): as xtb_dstedc in FP32
@_cdecl("xtb_sstedc")
public func xtb_sstedc(
  _ n: UnsafePointer<Int32>, _ d: UnsafeMutablePointer<Float>, _ e: UnsafeMutablePointer<Float>,
  _ z: UnsafeMutablePointer<Float>, _ ldz: UnsafePointer<Int32>, _ work: UnsafeMutablePointer<Float>,
  _ lwork: UnsafePointer<Int32>, _ iwork: UnsafeMutablePointer<Int32>, _ liwork: UnsafePointer<Int32>,
  _ info: UnsafeMutablePointer<Int32>
) {
  let n = Int(n.pointee), threads = openMPThreads()
  info.pointee = threads > 1 && n >= 256
    ? divideAndConquerParallel(n, d, e, z, Int(ldz.pointee), threads: threads)
    : Float.stedc(n, d, e, z, Int(ldz.pointee), work, Int(lwork.pointee), iwork, Int(liwork.pointee))
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
