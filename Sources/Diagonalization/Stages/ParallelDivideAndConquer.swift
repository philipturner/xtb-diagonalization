// The tridiagonal eigenproblem on worker threads (SBROptions.threads): divide and conquer with the top levels of the
// recursion in Swift and the leaves in Accelerate's xstedc (its new interface, documented as thread-safe: the leaves
// run concurrently). Each level tears the matrix at n1 = n / 2 (a rank-one
// change, as LAPACK's dlaed0), solves the two halves at once on half the workers each, and merges them as dlaed1:
// deflation by LAPACK's xlaed2, then a port of dlaed3 whose independent parts run over the workers (one xlaed4 call
// per root, the stabilized eigenvectors of the rank-one problem per column, the gemm that forms the merged
// eigenvectors per block of columns). Accelerate's xstedc on all of n is about 40% top merge and 57% halves at
// n = 412-776 (scripts/kernels/stedc_split.c, stedc_partial.c); the halves and the roots are the parallel work.

/// Diagonal d (n) and subdiagonal e (n - 1) -> eigenvalues ascending in d, eigenvectors in z (n x n, ldz), on up to
/// `threads` workers (threads <= 1: Accelerate's xstedc). Returns LAPACK's info (0: success).
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func divideAndConquerParallel<Real: BandReductionScalar>(
  _ n: Int, _ d: UnsafeMutablePointer<Real>, _ e: UnsafeMutablePointer<Real>, _ z: UnsafeMutablePointer<Real>,
  _ ldz: Int, threads: Int
) -> Int32 {
  guard n > 1 else {
    if n == 1 { z[0] = 1 }
    return 0
  }
  // Scale to unit max norm, as xstedc does before dlaed0.
  var norm: Real = 0
  for i in 0..<n { norm = max(norm, abs(d[i])) }
  for i in 0..<(n - 1) { norm = max(norm, abs(e[i])) }
  for j in 0..<n {
    for i in 0..<n { z[i + j * ldz] = 0 }
  }
  guard norm > 0 else {
    for i in 0..<n { z[i + i * ldz] = 1 }
    return 0
  }
  for i in 0..<n { d[i] /= norm }
  for i in 0..<(n - 1) { e[i] /= norm }
  let info = divideAndConquerLevel(n, d, e, z, ldz, threads: threads)
  for i in 0..<n { d[i] *= norm }
  return info
}

/// One level of the recursion: the halves (recursively, or by xstedc), then the merge. q's off-diagonal blocks are
/// zero on entry. On return d is ascending and q's columns match.
func divideAndConquerLevel<Real: BandReductionScalar>(
  _ n: Int, _ d: UnsafeMutablePointer<Real>, _ e: UnsafeMutablePointer<Real>, _ q: UnsafeMutablePointer<Real>,
  _ ldq: Int, threads: Int
) -> Int32 {
  guard threads > 1, n >= 128 else {
    let (lwork, liwork) = divideAndConquerWorkspace(n)
    let work = UnsafeMutablePointer<Real>.allocate(capacity: lwork)
    let iwork = UnsafeMutablePointer<Int32>.allocate(capacity: liwork)
    defer { work.deallocate(); iwork.deallocate() }
    return Real.stedcThreadSafe(n, d, e, q, ldq, work, lwork, iwork, liwork)
  }
  // Tear: T = diag(T1, T2) + rho v v^T, v = e_{n1} + e_{n1 + 1} (rows n1 - 1, n1 here).
  let n1 = n / 2, n2 = n - n1
  let rho = e[n1 - 1]
  d[n1 - 1] -= abs(rho)
  d[n1] -= abs(rho)
  let left = threads / 2, right = threads - left
  var infos: (Int32, Int32) = (0, 0)
  withUnsafeMutablePointer(to: &infos) { infos in
    parallelBlocks(2, threads: 2) { half in
      if half == 0 {
        infos.pointee.0 = divideAndConquerLevel(n1, d, e, q, ldq, threads: left)
      } else {
        infos.pointee.1 = divideAndConquerLevel(
          n2, d + n1, e + n1, q + offset(n1, n1, ldq), ldq, threads: right)
      }
    }
  }
  if infos.0 != 0 { return infos.0 }
  if infos.1 != 0 { return infos.1 + Int32(n1) }
  return mergeRankOne(n, n1, d, q, ldq, rho, threads: threads)
}

/// LAPACK's dlaed1 on worker threads: merges the eigensystems of the halves (d ascending in each, q block diagonal)
/// across the rank-one tear rho, then sorts d and q's columns ascending.
func mergeRankOne<Real: BandReductionScalar>(
  _ n: Int, _ n1: Int, _ d: UnsafeMutablePointer<Real>, _ q: UnsafeMutablePointer<Real>, _ ldq: Int, _ rho: Real,
  threads: Int
) -> Int32 {
  let n2 = n - n1
  // Q2 as dlaed1 sizes it, n^2: xlaed2 also stores the deflated eigenvectors there with leading dimension n, past the
  // n1^2 + n2^2 its documentation gives (an overrun with many deflations, found by AddressSanitizer at FP32 n = 1788).
  let reals = UnsafeMutablePointer<Real>.allocate(capacity: 3 * n + n * n)
  let ints = UnsafeMutablePointer<Int32>.allocate(capacity: 8 * n)
  defer { reals.deallocate(); ints.deallocate() }
  let z = reals, dlamda = z + n, w = dlamda + n, q2 = w + n
  let indxq = ints, indx = ints + n, indxc = indx + n, indxp = indxc + n, coltyp = indxp + n
  for i in 0..<n1 { indxq[i] = Int32(i + 1) }
  for i in 0..<n2 { indxq[n1 + i] = Int32(i + 1) }
  // z: the last row of the first half's eigenvectors, the first row of the second half's.
  for i in 0..<n1 { z[i] = q[offset(n1 - 1, i, ldq)] }
  for i in 0..<n2 { z[n1 + i] = q[offset(n1, n1 + i, ldq)] }
  var k: Int32 = 0
  var rho = rho
  let info = Real.laed2(&k, n, n1, d, q, ldq, indxq, &rho, z, dlamda, w, q2, indx, indxc, indxp, coltyp)
  if info != 0 { return info }
  let kk = Int(k)
  if kk > 0 {
    let info = secularEigenvectors(kk, n, n1, d, q, ldq, rho, dlamda, q2, indxc, coltyp, w, threads: threads)
    if info != 0 { return info }
    Real.lamrg(kk, n - kk, d, 1, -1, indxq)
  } else {
    for i in 0..<n { indxq[i] = Int32(i + 1) }
  }
  // Sort: column i of the result is column indxq[i] of q.
  let sorted = UnsafeMutablePointer<Real>.allocate(capacity: n * n + n)
  defer { sorted.deallocate() }
  let dSorted = sorted + n * n
  for i in 0..<n { dSorted[i] = d[Int(indxq[i]) - 1] }
  let width = max(16, (n + 2 * threads - 1) / (2 * threads))
  parallelBlocks((n + width - 1) / width, threads: threads) { blk in
    for i in (blk * width)..<min(n, (blk + 1) * width) {
      (sorted + i * n).update(from: q + (Int(indxq[i]) - 1) * ldq, count: n)
    }
  }
  parallelBlocks((n + width - 1) / width, threads: threads) { blk in
    for i in (blk * width)..<min(n, (blk + 1) * width) {
      (q + i * ldq).update(from: sorted + i * n, count: n)
    }
  }
  d.update(from: dSorted, count: n)
  return 0
}

/// LAPACK's dlaed3 on worker threads: the k roots of the secular equation (into d), the eigenvectors of the rank-one
/// problem with Gu and Eisenstat's recomputed z, and the merged eigenvectors in q's first k columns.
func secularEigenvectors<Real: BandReductionScalar>(
  _ k: Int, _ n: Int, _ n1: Int, _ d: UnsafeMutablePointer<Real>, _ q: UnsafeMutablePointer<Real>, _ ldq: Int,
  _ rho: Real, _ dlamda: UnsafeMutablePointer<Real>, _ q2: UnsafePointer<Real>, _ indx: UnsafePointer<Int32>,
  _ ctot: UnsafePointer<Int32>, _ w: UnsafeMutablePointer<Real>, threads: Int
) -> Int32 {
  // The roots: column j of q receives dlamda - root j.
  let rootWidth = max(8, (k + 4 * threads - 1) / (4 * threads))
  let rootBlocks = (k + rootWidth - 1) / rootWidth
  let infos = UnsafeMutablePointer<Int32>.allocate(capacity: rootBlocks)
  defer { infos.deallocate() }
  parallelBlocks(rootBlocks, threads: threads) { blk in
    infos[blk] = 0
    for j in (blk * rootWidth)..<min(k, (blk + 1) * rootWidth) {
      let info = Real.laed4(k, j + 1, dlamda, w, q + j * ldq, rho, d + j)
      if info != 0 { infos[blk] = info; return }
    }
  }
  for blk in 0..<rootBlocks where infos[blk] != 0 { return infos[blk] }

  if k == 2 {
    for j in 0..<k {
      let w0 = q[offset(0, j, ldq)], w1 = q[offset(1, j, ldq)]
      let pair = [w0, w1]
      q[offset(0, j, ldq)] = pair[Int(indx[0]) - 1]
      q[offset(1, j, ldq)] = pair[Int(indx[1]) - 1]
    }
  } else if k > 2 {
    // z recomputed from the roots (w_i = -prod ...), each i on its own, the same product order as dlaed3.
    let signs = UnsafeMutablePointer<Real>.allocate(capacity: 2 * k)
    defer { signs.deallocate() }
    let wNew = signs + k
    signs.update(from: w, count: k)
    let width = max(16, (k + 2 * threads - 1) / (2 * threads))
    parallelBlocks((k + width - 1) / width, threads: threads) { blk in
      for i in (blk * width)..<min(k, (blk + 1) * width) {
        var product = q[offset(i, i, ldq)]
        for j in 0..<k where j != i {
          product *= q[offset(i, j, ldq)] / (dlamda[i] - dlamda[j])
        }
        let magnitude = (-product).squareRoot()
        wNew[i] = signs[i] < 0 ? -magnitude : magnitude
      }
    }
    // Column j: s = w / q(:, j), normalized, then permuted by indx.
    parallelBlocks((k + width - 1) / width, threads: threads) { blk in
      withUnsafeTemporaryAllocation(of: Real.self, capacity: k) { buffer in
        let s = buffer.baseAddress!
        for j in (blk * width)..<min(k, (blk + 1) * width) {
          let column = q + j * ldq
          var sum: Real = 0
          for i in 0..<k {
            s[i] = wNew[i] / column[i]
            sum += s[i] * s[i]
          }
          let scale = 1 / sum.squareRoot()
          for i in 0..<k { column[i] = s[Int(indx[i]) - 1] * scale }
        }
      }
    }
  }

  // q(:, 0 ..< k) = Q2 S by blocks of columns (dlaed3's two gemm calls).
  let n2 = n - n1
  let c0 = Int(ctot[0]), n12 = c0 + Int(ctot[1]), n23 = Int(ctot[1]) + Int(ctot[2])
  let width = max(16, (k + 2 * threads - 1) / (2 * threads))
  parallelBlocks((k + width - 1) / width, threads: threads) { blk in
    let j0 = blk * width, cols = min(width, k - j0)
    let s = UnsafeMutablePointer<Real>.allocate(capacity: max(n12, n23, 1) * cols)
    defer { s.deallocate() }
    for j in 0..<cols { (s + j * n23).update(from: q + c0 + (j0 + j) * ldq, count: n23) }
    if n23 > 0 {
      Real.gemm("N", "N", n2, cols, n23, 1, q2 + n1 * n12, n2, s, n23, 0, q + offset(n1, j0, ldq), ldq)
    } else {
      for j in 0..<cols { for i in 0..<n2 { q[offset(n1 + i, j0 + j, ldq)] = 0 } }
    }
    for j in 0..<cols { (s + j * n12).update(from: q + (j0 + j) * ldq, count: n12) }
    if n12 > 0 {
      Real.gemm("N", "N", n1, cols, n12, 1, q2, n1, s, n12, 0, q + offset(0, j0, ldq), ldq)
    } else {
      for j in 0..<cols { for i in 0..<n1 { q[offset(i, j0 + j, ldq)] = 0 } }
    }
  }
  return 0
}
