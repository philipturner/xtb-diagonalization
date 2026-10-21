// Two-stage reduction of a symmetric matrix to tridiagonal form, and the back-transformation of its eigenvectors:
//   stage 1   dense -> band of bandwidth b: panel QR (geqrf) + two-sided block reflector (larfb)
//   stage 2   band -> tridiagonal: bulge chasing, one Householder reflector per bulge (Schwarz)
//   back-transformation of the tridiagonal eigenvectors: stage 2's reflectors in reverse, then stage 1's panels
// Development version (BRIEF.md Phase 4): the full n x n matrix is stored and updated, symmetry isn't exploited, the
// stage 1 updates span the whole matrix rather than the trailing block, and stage 2 and its back-transformation are
// plain loops, one reflector at a time. The structure follows swift-xtb's draft (nanotech-code-archive), column-major.
//
// Indices are 0-based. Matrices are column-major. The band is the lower band |i - j| <= b, mirrored above.
// The public functions are specialized for Double and Float, so callers in other modules don't run generic code.

/// Offset of element (i, j), 0-based, in a column-major matrix with leading dimension ld.
@inline(__always)
private func at(_ i: Int, _ j: Int, _ ld: Int) -> Int { i + j * ld }

/// The panels of stage 1: column offsets j with at least two rows below the band (n - j - b >= 2).
@inline(__always)
func bandPanels(_ n: Int, _ b: Int) -> StrideTo<Int> {
  stride(from: 0, to: max(0, n - b - 1), by: b)
}

/// Workspace (elements) for reduceToBand and backTransformBand.
public func bandReductionWorkspace(_ n: Int, _ b: Int) -> Int {
  b + n * b + 64 * b
}

/// Stage 1: reduces the full symmetric matrix a (n x n, lda) to bandwidth b, in place. Entries outside the band are
/// set to zero. For the back-transformation it keeps, per panel at column offset j, the Householder vectors in
/// v1 (n x n, leading dimension n; rows j + b ..< n of columns j ..< j + k, geqrf layout, unit diagonal implied) and
/// the triangular factor T in t1 (b x n, leading dimension b; columns j ..< j + k), where k = min(b, n - j - b).
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func reduceToBand<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ a: UnsafeMutablePointer<Real>, _ lda: Int, _ v1: UnsafeMutablePointer<Real>,
  _ t1: UnsafeMutablePointer<Real>, _ work: UnsafeMutablePointer<Real>
) {
  let tau = work, scratch = work + b
  let lscratch = n * b + 64 * b
  for j in bandPanels(n, b) {
    let r = j + b       // first row below the band
    let m = n - r       // rows of the panel
    let k = min(b, m)   // reflectors
    // Copy the panel below the band and factor it: Q R = A[r ..< n, j ..< j + b].
    let v = v1 + at(r, j, n)
    for c in 0..<b {
      for i in 0..<m { v[at(i, c, n)] = a[at(r + i, j + c, lda)] }
    }
    Real.geqrf(m, b, v, n, tau, scratch, lscratch)
    let t = t1 + at(0, j, b)
    Real.larft(m, k, v, n, tau, t, b)
    // A = Q^T A Q on rows and columns r ..< n, applied to the whole matrix.
    Real.larfb("L", "T", m, n, k, v, n, t, b, a + r, lda, scratch, n)
    Real.larfb("R", "N", n, m, k, v, n, t, b, a + at(0, r, lda), lda, scratch, n)
  }
  for j in 0..<n {
    for i in 0..<n where abs(i - j) > b { a[at(i, j, lda)] = 0 }
  }
}

/// Workspace (elements) for chaseBulges.
public func bulgeChasingWorkspace(_ b: Int) -> Int { 4 * b + 4 }

/// Stage 2: reduces the band matrix a (bandwidth b, full storage, entries outside the band zero) to tridiagonal form,
/// in place. Writes the diagonal to d (n) and the subdiagonal to e (n - 1). Keeps every reflector H = I - w w^T
/// (w = sqrt(tau) v, so ||w||^2 = 2 or w = 0) for the back-transformation in v2 (n x n, leading dimension n): the
/// reflectors of sweep s are in column s, the one acting on rows rs ..< re at those rows (bulgeReflectorRows).
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func chaseBulges<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ a: UnsafeMutablePointer<Real>, _ lda: Int, _ v2: UnsafeMutablePointer<Real>,
  _ d: UnsafeMutablePointer<Real>, _ e: UnsafeMutablePointer<Real>, _ work: UnsafeMutablePointer<Real>
) {
  let x = work               // the column segment to annihilate (b)
  let u = work + b           // A w for the right application (up to 3 b)
  if b > 1 {
    for s in 0..<max(0, n - 2) {
      var k = 0
      while true {
        let (rs, re) = bulgeReflectorRows(n, b, s, k)
        guard rs < n else { break }
        // The column to annihilate below its first entry: column s (k = 0), else the bulge's first column.
        let c = k == 0 ? s : rs - b
        let len = re - rs
        for i in 0..<len { x[i] = a[at(rs + i, c, lda)] }
        var tau: Real = 0
        Real.larfg(len, x, x + 1, &tau)
        let beta = x[0]
        // w = sqrt(tau) [1, x(1 ..< len)]
        let w = v2 + at(rs, s, n)
        let scale = tau.squareRoot()
        w[0] = scale
        for i in 1..<len { w[i] = scale * x[i] }
        if tau != 0 {
          // A = H A H on the window lo ..< hi, which holds every nonzero of rows and columns rs ..< re.
          let lo = max(rs - b, 0), hi = min(re + b, n)
          for j in lo..<hi {
            let col = a + at(rs, j, lda)
            var dot: Real = 0
            for i in 0..<len { dot += w[i] * col[i] }
            for i in 0..<len { col[i] -= w[i] * dot }
          }
          // u = A[lo ..< hi, rs ..< re] w, accumulated column by column (column-major order)
          for i in 0..<(hi - lo) { u[i] = 0 }
          for jj in 0..<len {
            let col = a + at(lo, rs + jj, lda)
            let wj = w[jj]
            for i in 0..<(hi - lo) { u[i] += col[i] * wj }
          }
          for jj in 0..<len {
            let col = a + at(lo, rs + jj, lda)
            let wj = w[jj]
            for i in 0..<(hi - lo) { col[i] -= u[i] * wj }
          }
        }
        // Exact zeros where the reflector annihilated, both triangles.
        a[at(rs, c, lda)] = beta
        a[at(c, rs, lda)] = beta
        for i in 1..<len {
          a[at(rs + i, c, lda)] = 0
          a[at(c, rs + i, lda)] = 0
        }
        k += 1
      }
    }
  }
  for i in 0..<n { d[i] = a[at(i, i, lda)] }
  for i in 0..<max(0, n - 1) { e[i] = a[at(i + 1, i, lda)] }
}

/// Rows rs ..< re of reflector k in sweep s of the bulge chase (rs >= n: no such reflector).
@inline(__always)
public func bulgeReflectorRows(_ n: Int, _ b: Int, _ s: Int, _ k: Int) -> (Int, Int) {
  let rs = s + 1 + k * b
  return (rs, min(rs + b, n))
}

/// Applies stage 2's reflectors to the first m columns of z (n x m, ldz): z = H_1 H_2 ... H_N z.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func backTransformBulges<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ v2: UnsafePointer<Real>, _ z: UnsafeMutablePointer<Real>, _ ldz: Int, _ m: Int
) {
  guard b > 1 else { return }
  for s in stride(from: n - 3, through: 0, by: -1) {
    var kmax = 0
    while bulgeReflectorRows(n, b, s, kmax + 1).0 < n { kmax += 1 }
    for k in stride(from: kmax, through: 0, by: -1) {
      let (rs, re) = bulgeReflectorRows(n, b, s, k)
      let len = re - rs
      let w = v2 + at(rs, s, n)
      for col in 0..<m {
        let zc = z + at(rs, col, ldz)
        var dot: Real = 0
        for i in 0..<len { dot += w[i] * zc[i] }
        for i in 0..<len { zc[i] -= w[i] * dot }
      }
    }
  }
}

/// Workspace (elements) for backTransformBulgesBlocked.
public func bulgeBackTransformWorkspace(_ b: Int, _ m: Int) -> Int {
  2 * b * b + b * b + b + m * b
}

/// backTransformBulges with the reflectors of b consecutive sweeps at the same bulge index k combined into one block
/// reflector I - V T V^T (V: 2b - 1 rows, b columns, each column shifted down one row from the previous), applied with
/// larfb (swift-xtb's draft). Order: k ascending, sweep blocks descending, which equals the reverse of the reduction
/// order because every pair it swaps acts on disjoint rows: H(s, k) and H(s', k') with s > s' and k > k' are at least
/// b + 1 rows apart, and the same sweep's reflectors are disjoint.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func backTransformBulgesBlocked<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ v2: UnsafePointer<Real>, _ z: UnsafeMutablePointer<Real>, _ ldz: Int, _ m: Int,
  _ work: UnsafeMutablePointer<Real>
) {
  guard b > 1, n > 2 else { return }
  let ldv = 2 * b
  let v = work, t = work + ldv * b, tau = t + b * b, scratch = tau + b
  let sweeps = n - 2
  var k = 0
  while bulgeReflectorRows(n, b, 0, k).0 < n {
    for s0 in stride(from: (sweeps - 1) / b * b, through: 0, by: -b) {
      let r0 = bulgeReflectorRows(n, b, s0, k).0
      guard r0 < n else { continue }
      let height = min(2 * b - 1, n - r0)
      // Reflectors of sweeps s0 ..< s0 + count, column c starting at row c of the block.
      let count = min(b, sweeps - s0, height)
      for c in 0..<count {
        let s = s0 + c
        let (rs, re) = bulgeReflectorRows(n, b, s, k)
        let col = v + c * ldv
        for i in 0..<height { col[i] = 0 }
        let w = v2 + at(rs, s, n)
        // w = sqrt(tau) [1, v(1 ...)]: tau = w0^2, v = w / w0.
        let w0 = w[0]
        if rs < n, w0 != 0 {
          tau[c] = w0 * w0
          col[c] = 1
          for i in 1..<(re - rs) { col[c + i] = w[i] / w0 }
        } else {
          tau[c] = 0
        }
      }
      Real.larft(height, count, v, ldv, tau, t, b)
      Real.larfb("L", "N", height, m, count, v, ldv, t, b, z + r0, ldz, scratch, max(m, 1))
    }
    k += 1
  }
}

/// Applies stage 1's panels to the first m columns of z (n x m, ldz): z = Q_1 Q_2 ... z.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func backTransformBand<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ v1: UnsafePointer<Real>, _ t1: UnsafePointer<Real>, _ z: UnsafeMutablePointer<Real>,
  _ ldz: Int, _ m: Int, _ work: UnsafeMutablePointer<Real>
) {
  for j in bandPanels(n, b).reversed() {
    let r = j + b
    let rows = n - r
    let k = min(b, rows)
    Real.larfb("L", "N", rows, m, k, v1 + at(r, j, n), n, t1 + at(0, j, b), b, z + r, ldz, work, max(m, 1))
  }
}
