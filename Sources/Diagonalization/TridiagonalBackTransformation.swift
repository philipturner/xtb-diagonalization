// Back-transformation through the 1-stage reduction (Tridiagonalization.swift): z = Q z with Q from tridiagonalize
// (upper triangle), what xormtr('L', 'U', 'N') computes, as xtb/src/mctc/lapack/sytrd.c's xtb_dormtr_upper does it:
// xormql's blocks of 32 reflectors (block i ..< i + ib, 0-based reflectors, acts on rows 0 ..< i + ib; blocks in
// increasing i), each applied by three gemm calls instead of Accelerate's larfb (WYTransform.swift's gemmWY), with V^T
// stored explicitly so that W = V^T z is gemm 'N', 'N' (Accelerate's 'T', 'N' runs slower). Measured against
// Accelerate's dormtr (scripts/kernels/ormtr_gemm.c, m = n / 2): 1.2-1.4x faster at n = 412-776 for one process,
// 1.1-1.25x with 8 processes x 1 thread. Same result up to rounding.

/// Workspace (elements) for backTransformTridiagonal.
public func tridiagonalBackTransformWorkspace(_ n: Int, _ m: Int) -> Int {
  32 * (2 * n + 2 * m + 32)
}

/// z = Q z for the first m columns of z (n x m, ldz), Q from tridiagonalize on a (n x n, lda: the Householder vectors
/// above the superdiagonal) and tau. work holds tridiagonalBackTransformWorkspace(n, m).
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
public func backTransformTridiagonal<T: BandReductionScalar>(
  _ n: Int, _ m: Int, _ a: UnsafePointer<T>, _ lda: Int, _ tau: UnsafePointer<T>, _ z: UnsafeMutablePointer<T>,
  _ ldz: Int, _ work: UnsafeMutablePointer<T>
) {
  let nb = 32, k = n - 1
  guard m > 0, k > 0 else { return }
  let v = work, vt = v + n * nb, t = vt + n * nb, w = t + nb * nb, w2 = w + nb * m
  for i in stride(from: 0, to: k, by: nb) {
    let ib = min(nb, k - i), rows = i + ib
    // Reflector r = i + c: rows 0 ..< r from column r + 1 of a, 1 at row r, zeros below.
    for c in 0..<ib {
      let r = i + c, col = v + c * rows, src = a + (r + 1) * lda
      for q in 0..<r { col[q] = src[q] }
      col[r] = 1
      for q in (r + 1)..<rows { col[q] = 0 }
    }
    for c in 0..<ib {
      for q in 0..<rows { vt[c + q * ib] = v[q + c * rows] }
    }
    T.larftBackward(rows, ib, v, rows, tau + i, t, ib)
    for c in 1..<ib {
      for q in 0..<c { t[q + c * ib] = 0 }
    }
    T.gemm("N", "N", ib, m, rows, 1, vt, ib, z, ldz, 0, w, ib)
    T.gemm("N", "N", ib, m, ib, 1, t, ib, w, ib, 0, w2, ib)
    T.gemm("N", "N", rows, m, ib, -1, v, rows, w2, ib, 1, z, ldz)
  }
}

/// backTransformTridiagonal over `threads` workers: the m columns in one block per worker (the blocks are
/// independent). work holds threads * tridiagonalBackTransformWorkspace(n, (m + threads - 1) / threads).
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
public func backTransformTridiagonalParallel<T: BandReductionScalar>(
  _ n: Int, _ m: Int, _ a: UnsafePointer<T>, _ lda: Int, _ tau: UnsafePointer<T>, _ z: UnsafeMutablePointer<T>,
  _ ldz: Int, _ work: UnsafeMutablePointer<T>, threads: Int
) {
  let threads = max(threads, 1)
  let width = (m + threads - 1) / threads
  guard threads > 1, width > 0 else {
    backTransformTridiagonal(n, m, a, lda, tau, z, ldz, work)
    return
  }
  let slice = tridiagonalBackTransformWorkspace(n, width)
  parallelBlocks((m + width - 1) / width, threads: threads) { blk in
    let c0 = blk * width
    backTransformTridiagonal(n, min(width, m - c0), a, lda, tau, z + c0 * ldz, ldz, work + blk * slice)
  }
}
