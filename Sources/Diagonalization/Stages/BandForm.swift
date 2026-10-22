// Step 1 of successive band reduction: a dense symmetric matrix -> bandwidth b (the first stage of every chain).
// Per panel of b columns: QR of the panel below the band (geqrf), its compact WY factor T (WYTransform.swift), and
// Q^T A Q on the rest of the matrix, either as two block reflectors (left, right) or as one symmetric rank-2b update
// (LAPACK's sy2sb). V and T are kept for BandFormBackTransformation.swift.
// Indices are 0-based, matrices column-major; the band is the lower band |i - j| <= b, mirrored above unless the
// options leave the upper triangle stale.

/// The panels of stage 1: column offsets j with at least two rows below the band (n - j - b >= 2).
@inline(__always)
func bandPanels(_ n: Int, _ b: Int) -> StrideTo<Int> {
  stride(from: 0, to: max(0, n - b - 1), by: b)
}

/// Workspace (elements) for reduceToBand and backTransformBand.
public func bandReductionWorkspace(_ n: Int, _ b: Int) -> Int {
  b + n * b + 64 * b + 2 * n * b + 2 * b * b
}

/// Stage 1: reduces the full symmetric matrix a (n x n, lda) to bandwidth b, in place. Entries outside the band are
/// set to zero. For the back-transformation it keeps, per panel at column offset j, the Householder vectors in
/// v1 (n x n, leading dimension n; rows j + b ..< n of columns j ..< j + k, geqrf layout, unit diagonal implied) and
/// the triangular factor T in t1 (b x n, leading dimension b; columns j ..< j + k), where k = min(b, n - j - b).
/// With options.gemmWY, V is stored explicitly (unit diagonal, zeros above; R goes only to a) and the block reflectors
/// are applied by three gemm calls. panelDone (optional) receives after each panel its frontier f: the panels still to
/// come change only entries whose row and column are both >= f, so every entry with a row or column below f is final.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func reduceToBand<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ a: UnsafeMutablePointer<Real>, _ lda: Int, _ v1: UnsafeMutablePointer<Real>,
  _ t1: UnsafeMutablePointer<Real>, _ work: UnsafeMutablePointer<Real>, _ options: SBROptions = SBROptions(),
  panelDone: ((Int) -> Void)? = nil
) {
  let tau = work, scratch = work + b
  let lscratch = n * b + 64 * b
  // combinedUpdate: V with explicit unit diagonal and zeros above (m x k), X (m x k), S and Y (k x k)
  let vf = scratch + lscratch, x = vf + n * b, sm = x + n * b, y = sm + b * b
  let trailing = options.trailingOnly || options.combinedUpdate
  let lower = options.combinedUpdate && options.lowerTriangle
  for j in bandPanels(n, b) {
    let r = j + b       // first row below the band
    let m = n - r       // rows of the panel
    let k = min(b, m)   // reflectors
    // Copy the panel below the band and factor it: Q R = A[r ..< n, j ..< j + b].
    let v = v1 + offset(r, j, n)
    for c in 0..<b {
      for i in 0..<m { v[offset(i, c, n)] = a[offset(r + i, j + c, lda)] }
    }
    Real.geqrf(m, b, v, n, tau, scratch, lscratch)
    let t = t1 + offset(0, j, b)
    wyFactor(m, k, v, n, tau, t, b)
    let g = options.gemmWY
    if g && !options.combinedUpdate { wyMakeExplicit(m, k, v, n) }
    if !trailing {
      // A = Q^T A Q on rows and columns r ..< n, applied to the whole matrix.
      wyApply("L", "T", m, n, k, v, n, t, b, a + r, lda, scratch, n, gemm: g)
      wyApply("R", "N", n, m, k, v, n, t, b, a + offset(0, r, lda), lda, scratch, n, gemm: g)
    } else if !options.combinedUpdate {
      // The same on rows and columns j ..< n only: everything before column j is already zero there.
      wyApply("L", "T", m, n - j, k, v, n, t, b, a + offset(r, j, lda), lda, scratch, n - j, gemm: g)
      wyApply("R", "N", n - j, m, k, v, n, t, b, a + offset(j, r, lda), lda, scratch, n - j, gemm: g)
    } else {
      // The panel below the band becomes R (both triangles).
      for c in 0..<b {
        for i in 0..<m {
          let value: Real = i <= c ? v[offset(i, c, n)] : 0
          a[offset(r + i, j + c, lda)] = value
          a[offset(j + c, r + i, lda)] = value
        }
      }
      // V with its unit diagonal and zeros above, m x k.
      for c in 0..<k {
        for i in 0..<m { vf[offset(i, c, m)] = i < c ? 0 : (i == c ? 1 : v[offset(i, c, n)]) }
      }
      if g { wyMakeExplicit(m, k, v, n) }
      // Trailing block A2 = A[r ..< n, r ..< n]: X = A2 V T, W = X - 1/2 V (T^T V^T X), A2 -= V W^T + W V^T.
      let a2 = a + offset(r, r, lda)
      // With options.threads > 1, the gemm calls below run on column (or row) blocks of `width` over the workers.
      let threads = options.bandThreads > 0 ? options.bandThreads : options.threads
      let width = threads > 1 ? max(32, min(128, (m + 2 * threads - 1) / (2 * threads) + 15) / 16 * 16) : m
      if lower {
        Real.symmLower(m, k, 1, a2, lda, vf, m, 0, x, m)
      } else if threads > 1 {
        parallelBlocks((m + width - 1) / width, threads: threads) { blk in
          let r0 = blk * width, h = min(width, m - r0)
          Real.gemm("N", "N", h, k, m, 1, a2 + r0, lda, vf, m, 0, x + r0, m)
        }
      } else {
        Real.gemm("N", "N", m, k, m, 1, a2, lda, vf, m, 0, x, m)
      }
      Real.trmm("R", "U", "N", "N", m, k, 1, t, b, x, m)
      Real.gemm("T", "N", k, k, m, 1, vf, m, x, m, 0, sm, k)
      for i in 0..<(k * k) { y[i] = sm[i] }
      Real.trmm("L", "U", "T", "N", k, k, 1, t, b, y, k)
      Real.gemm("N", "N", m, k, k, -0.5, vf, m, y, k, 1, x, m)
      if lower {
        if options.lowerGEMM {
          // The same lower triangle by gemm per block column (rows c ..< m of columns c ..< c + 128, the diagonal
          // block in full; its upper part is never read): 8-11% slower than Accelerate's syr2k, but syr2k leaves the
          // core in a state where the L2-bound bulge chase right after it runs 15-30% slower (LOG.md); gemm doesn't.
          let step = threads > 1 ? width : 128
          parallelBlocks((m + step - 1) / step, threads: threads) { blk in
            let c = blk * step, w = min(step, m - c), h = m - c
            Real.gemm("N", "T", h, w, k, -1, vf + c, m, x + c, m, 1, a2 + offset(c, c, lda), lda)
            Real.gemm("N", "T", h, w, k, -1, x + c, m, vf + c, m, 1, a2 + offset(c, c, lda), lda)
          }
        } else {
          Real.syr2kLower(m, k, -1, vf, m, x, m, 1, a2, lda)
        }
      } else if threads > 1 {
        parallelBlocks((m + width - 1) / width, threads: threads) { blk in
          let c = blk * width, w = min(width, m - c)
          Real.gemm("N", "T", m, w, k, -1, vf, m, x + c, m, 1, a2 + offset(0, c, lda), lda)
          Real.gemm("N", "T", m, w, k, -1, x, m, vf + c, m, 1, a2 + offset(0, c, lda), lda)
        }
      } else {
        Real.gemm("N", "T", m, m, k, -1, vf, m, x, m, 1, a2, lda)
        Real.gemm("N", "T", m, m, k, -1, x, m, vf, m, 1, a2, lda)
      }
    }
    panelDone?(j + b)
  }
  if panelDone != nil {
    // Overlapped with the chase (reduceToTridiagonalOverlapped): the combined update has already left every entry
    // outside the band exactly zero, so the passes below would change nothing, but they would race the chase.
    precondition(options.combinedUpdate, "panelDone needs the combined update")
    return
  }
  if lower && options.symmetricChase {
    // Stage 2 reads only the lower triangle, which is already exactly zero outside the band (the panels below it
    // were overwritten with R and zeros): no pass over the matrix. The upper triangle is left stale.
    return
  }
  if lower {
    // The lower triangle is current; mirror the band (everything else is cleared below).
    for j in 0..<n {
      for i in (j + 1)..<min(j + b + 1, n) { a[offset(j, i, lda)] = a[offset(i, j, lda)] }
    }
  }
  for j in 0..<n {
    for i in 0..<n where abs(i - j) > b { a[offset(i, j, lda)] = 0 }
  }
}
