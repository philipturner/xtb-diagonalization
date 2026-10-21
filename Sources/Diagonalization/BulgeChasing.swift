// Last step of successive band reduction: bandwidth b -> tridiagonal by bulge chasing (Schwarz), one Householder
// reflector per bulge, no WY blocking (each reflector is applied as soon as it's made). Sweep s annihilates column s
// below its first subdiagonal; the reflector's right application creates a bulge b rows further down, which the next
// reflector of the sweep removes, and so on to the end of the matrix. The reflectors (scaled: H = I - w w^T) are kept
// for BulgeBackTransformation.swift. SBROptions.symmetricChase works on the lower triangle only.

/// Workspace (elements) for chaseBulges.
public func bulgeChasingWorkspace(_ b: Int) -> Int { 4 * b + 4 }

/// Rows rs ..< re of reflector k in sweep s of the bulge chase (rs >= n: no such reflector).
@inline(__always)
public func bulgeReflectorRows(_ n: Int, _ b: Int, _ s: Int, _ k: Int) -> (Int, Int) {
  let rs = s + 1 + k * b
  return (rs, min(rs + b, n))
}

/// Stage 2: reduces the band matrix a (bandwidth b, full storage, entries outside the band zero) to tridiagonal form,
/// in place. Writes the diagonal to d (n) and the subdiagonal to e (n - 1). Keeps every reflector H = I - w w^T
/// (w = sqrt(tau) v, so ||w||^2 = 2 or w = 0) for the back-transformation in v2 (n x n, leading dimension n): the
/// reflectors of sweep s are in column s, the one acting on rows rs ..< re at those rows (bulgeReflectorRows).
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func chaseBulges<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ a: UnsafeMutablePointer<Real>, _ lda: Int, _ v2: UnsafeMutablePointer<Real>,
  _ d: UnsafeMutablePointer<Real>, _ e: UnsafeMutablePointer<Real>, _ work: UnsafeMutablePointer<Real>,
  _ options: SBROptions = SBROptions()
) {
  if options.symmetricChase {
    chaseBulgesLower(n, b, a, lda, v2, d, e, work)
    return
  }
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
        for i in 0..<len { x[i] = a[offset(rs + i, c, lda)] }
        var tau: Real = 0
        Real.larfg(len, x, x + 1, &tau)
        let beta = x[0]
        // w = sqrt(tau) [1, x(1 ..< len)]
        let w = v2 + offset(rs, s, n)
        let scale = tau.squareRoot()
        w[0] = scale
        for i in 1..<len { w[i] = scale * x[i] }
        if tau != 0 {
          // A = H A H on the window lo ..< hi, which holds every nonzero of rows and columns rs ..< re.
          let lo = max(rs - b, 0), hi = min(re + b, n)
          for j in lo..<hi {
            let col = a + offset(rs, j, lda)
            var dot: Real = 0
            for i in 0..<len { dot += w[i] * col[i] }
            for i in 0..<len { col[i] -= w[i] * dot }
          }
          // u = A[lo ..< hi, rs ..< re] w, accumulated column by column (column-major order)
          for i in 0..<(hi - lo) { u[i] = 0 }
          for jj in 0..<len {
            let col = a + offset(lo, rs + jj, lda)
            let wj = w[jj]
            for i in 0..<(hi - lo) { u[i] += col[i] * wj }
          }
          for jj in 0..<len {
            let col = a + offset(lo, rs + jj, lda)
            let wj = w[jj]
            for i in 0..<(hi - lo) { col[i] -= u[i] * wj }
          }
        }
        // Exact zeros where the reflector annihilated, both triangles.
        a[offset(rs, c, lda)] = beta
        a[offset(c, rs, lda)] = beta
        for i in 1..<len {
          a[offset(rs + i, c, lda)] = 0
          a[offset(c, rs + i, lda)] = 0
        }
        k += 1
      }
    }
  }
  for i in 0..<n { d[i] = a[offset(i, i, lda)] }
  for i in 0..<max(0, n - 1) { e[i] = a[offset(i + 1, i, lda)] }
}

/// chaseBulges on the lower triangle only (the upper triangle is neither read nor written). With H = I - w w^T acting
/// on rows R = rs ..< re, the window lo ..< hi splits into three blocks of the lower triangle:
///   columns lo ..< rs, rows R      A[R, i] = H A[R, i]                  (contiguous column segments)
///   rows re ..< hi, columns R      A[i, R] = A[i, R] H                  (y = A[re ..< hi, R] w, then a rank-1 update)
///   rows and columns R             A_RR = H A_RR H = A_RR - w q^T - q w^T, p = A_RR w, q = p - (w^T p) w / 2
/// which is about half the arithmetic and memory traffic of updating both triangles with left and right products.
func chaseBulgesLower<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ a: UnsafeMutablePointer<Real>, _ lda: Int, _ v2: UnsafeMutablePointer<Real>,
  _ d: UnsafeMutablePointer<Real>, _ e: UnsafeMutablePointer<Real>, _ work: UnsafeMutablePointer<Real>
) {
  let x = work, y = work + b, p = y + 2 * b
  if b > 1 {
    for s in 0..<max(0, n - 2) {
      var k = 0
      while true {
        let (rs, re) = bulgeReflectorRows(n, b, s, k)
        guard rs < n else { break }
        let c = k == 0 ? s : rs - b
        let len = re - rs
        for i in 0..<len { x[i] = a[offset(rs + i, c, lda)] }
        var tau: Real = 0
        Real.larfg(len, x, x + 1, &tau)
        let beta = x[0]
        let w = v2 + offset(rs, s, n)
        let scale = tau.squareRoot()
        w[0] = scale
        for i in 1..<len { w[i] = scale * x[i] }
        if tau != 0 {
          let lo = max(rs - b, 0), hi = min(re + b, n)
          // Columns lo ..< rs, rows R.
          for j in lo..<rs {
            let col = a + offset(rs, j, lda)
            let dot = vectorDot(len, w, col)
            for i in 0..<len { col[i] -= w[i] * dot }
          }
          // Rows re ..< hi, columns R.
          let below = hi - re
          if below > 0 {
            for i in 0..<below { y[i] = 0 }
            for jj in 0..<len {
              let col = a + offset(re, rs + jj, lda)
              let wj = w[jj]
              for i in 0..<below { y[i] += col[i] * wj }
            }
            for jj in 0..<len {
              let col = a + offset(re, rs + jj, lda)
              let wj = w[jj]
              for i in 0..<below { col[i] -= y[i] * wj }
            }
          }
          // The block R x R, lower triangle: p = A_RR w.
          for i in 0..<len { p[i] = 0 }
          for jj in 0..<len {
            let col = a + offset(rs, rs + jj, lda)
            let wj = w[jj]
            var acc = col[jj] * wj
            for i in (jj + 1)..<max(len, jj + 1) {
              p[i] += col[i] * wj
              acc += col[i] * w[i]
            }
            p[jj] += acc
          }
          let half = vectorDot(len, w, p) / 2
          for i in 0..<len { p[i] -= half * w[i] }   // p is now q
          for jj in 0..<len {
            let col = a + offset(rs, rs + jj, lda)
            let wj = w[jj], qj = p[jj]
            for i in jj..<len { col[i] -= w[i] * qj + p[i] * wj }
          }
        }
        a[offset(rs, c, lda)] = beta
        for i in 1..<len { a[offset(rs + i, c, lda)] = 0 }
        k += 1
      }
    }
  }
  for i in 0..<n { d[i] = a[offset(i, i, lda)] }
  for i in 0..<max(0, n - 1) { e[i] = a[offset(i + 1, i, lda)] }
}

/// x . y over n elements with vector accumulators (the compiler doesn't vectorize a scalar floating-point reduction,
/// which must keep its order).
@inline(__always)
func vectorDot<Real: BandReductionScalar>(_ n: Int, _ x: UnsafePointer<Real>, _ y: UnsafePointer<Real>) -> Real {
  let w = Real.Vector.scalarCount
  var acc0 = Real.Vector(), acc1 = Real.Vector()
  var i = 0
  while i + 2 * w <= n {
    acc0 = Real.fma(acc0, UnsafeRawPointer(x + i).loadUnaligned(as: Real.Vector.self),
                    UnsafeRawPointer(y + i).loadUnaligned(as: Real.Vector.self))
    acc1 = Real.fma(acc1, UnsafeRawPointer(x + i + w).loadUnaligned(as: Real.Vector.self),
                    UnsafeRawPointer(y + i + w).loadUnaligned(as: Real.Vector.self))
    i += 2 * w
  }
  var sum = Real.horizontalSum(acc0 + acc1)
  while i < n { sum += x[i] * y[i]; i += 1 }
  return sum
}
