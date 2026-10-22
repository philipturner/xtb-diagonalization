// Short vector loops with explicit NEON fused multiply-adds (CSIMD through TridiagonalScalar.fma), for the stages
// whose inner loops are BLAS1-sized (the bulge chase). Swift doesn't vectorize an ordered floating-point reduction,
// and SIMD addingProduct compiles to scalar fmadd. Two vector accumulators or two vectors per iteration, scalar tail.

@inline(__always)
private func load<Real: BandReductionScalar>(_ p: UnsafePointer<Real>) -> Real.Vector {
  UnsafeRawPointer(p).loadUnaligned(as: Real.Vector.self)
}

@inline(__always)
private func store<Real: BandReductionScalar>(_ v: Real.Vector, _ p: UnsafeMutablePointer<Real>) {
  UnsafeMutableRawPointer(p).storeBytes(of: v, as: Real.Vector.self)
}

/// x . y over n elements.
@inline(__always)
func vectorDot<Real: BandReductionScalar>(_ n: Int, _ x: UnsafePointer<Real>, _ y: UnsafePointer<Real>) -> Real {
  let w = Real.Vector.scalarCount
  var acc0 = Real.Vector(), acc1 = Real.Vector()
  var i = 0
  while i + 2 * w <= n {
    acc0 = Real.fma(acc0, load(x + i), load(y + i))
    acc1 = Real.fma(acc1, load(x + i + w), load(y + i + w))
    i += 2 * w
  }
  var sum = Real.horizontalSum(acc0 + acc1)
  while i < n { sum += x[i] * y[i]; i += 1 }
  return sum
}

/// y += alpha x over n elements.
@inline(__always)
func vectorAxpy<Real: BandReductionScalar>(
  _ n: Int, _ alpha: Real, _ x: UnsafePointer<Real>, _ y: UnsafeMutablePointer<Real>
) {
  let w = Real.Vector.scalarCount
  let a = Real.Vector(repeating: alpha)
  var i = 0
  while i + 2 * w <= n {
    store(Real.fma(load(y + i), a, load(x + i)), y + i)
    store(Real.fma(load(y + i + w), a, load(x + i + w)), y + i + w)
    i += 2 * w
  }
  while i < n { y[i] += alpha * x[i]; i += 1 }
}

/// y += a1 x1 + a2 x2 over n elements.
@inline(__always)
func vectorAxpy2<Real: BandReductionScalar>(
  _ n: Int, _ a1: Real, _ x1: UnsafePointer<Real>, _ a2: Real, _ x2: UnsafePointer<Real>,
  _ y: UnsafeMutablePointer<Real>
) {
  let w = Real.Vector.scalarCount
  let v1 = Real.Vector(repeating: a1), v2 = Real.Vector(repeating: a2)
  var i = 0
  while i + 2 * w <= n {
    store(Real.fma(Real.fma(load(y + i), v1, load(x1 + i)), v2, load(x2 + i)), y + i)
    store(Real.fma(Real.fma(load(y + i + w), v1, load(x1 + i + w)), v2, load(x2 + i + w)), y + i + w)
    i += 2 * w
  }
  while i < n { y[i] += a1 * x1[i] + a2 * x2[i]; i += 1 }
}

/// Returns x . u and does y += alpha x, over n elements, reading x once (a column of a symmetric matrix-vector
/// product's lower triangle).
@inline(__always)
func vectorDotAxpy<Real: BandReductionScalar>(
  _ n: Int, _ x: UnsafePointer<Real>, _ u: UnsafePointer<Real>, _ alpha: Real, _ y: UnsafeMutablePointer<Real>
) -> Real {
  let w = Real.Vector.scalarCount
  let a = Real.Vector(repeating: alpha)
  var acc0 = Real.Vector(), acc1 = Real.Vector()
  var i = 0
  while i + 2 * w <= n {
    let x0 = load(x + i), x1 = load(x + i + w)
    acc0 = Real.fma(acc0, x0, load(u + i))
    acc1 = Real.fma(acc1, x1, load(u + i + w))
    store(Real.fma(load(y + i), a, x0), y + i)
    store(Real.fma(load(y + i + w), a, x1), y + i + w)
    i += 2 * w
  }
  var sum = Real.horizontalSum(acc0 + acc1)
  while i < n { sum += x[i] * u[i]; y[i] += alpha * x[i]; i += 1 }
  return sum
}

// Four columns per pass (the bulge chase's left and below blocks): the shared vector is loaded once per element for
// all four columns instead of once per column.

/// (x . c0, x . c1, x . c2, x . c3) over n elements.
@inline(__always)
func vectorDot4<Real: BandReductionScalar>(
  _ n: Int, _ x: UnsafePointer<Real>, _ c0: UnsafePointer<Real>, _ c1: UnsafePointer<Real>,
  _ c2: UnsafePointer<Real>, _ c3: UnsafePointer<Real>
) -> (Real, Real, Real, Real) {
  let w = Real.Vector.scalarCount
  var a0 = Real.Vector(), a1 = Real.Vector(), a2 = Real.Vector(), a3 = Real.Vector()
  var i = 0
  while i + w <= n {
    let xv = load(x + i)
    a0 = Real.fma(a0, xv, load(c0 + i)); a1 = Real.fma(a1, xv, load(c1 + i))
    a2 = Real.fma(a2, xv, load(c2 + i)); a3 = Real.fma(a3, xv, load(c3 + i))
    i += w
  }
  var s0 = Real.horizontalSum(a0), s1 = Real.horizontalSum(a1)
  var s2 = Real.horizontalSum(a2), s3 = Real.horizontalSum(a3)
  while i < n {
    s0 += x[i] * c0[i]; s1 += x[i] * c1[i]; s2 += x[i] * c2[i]; s3 += x[i] * c3[i]
    i += 1
  }
  return (s0, s1, s2, s3)
}

/// c_j += a_j x for the four columns c0 ... c3, over n elements.
@inline(__always)
func vectorAxpy4<Real: BandReductionScalar>(
  _ n: Int, _ a0: Real, _ a1: Real, _ a2: Real, _ a3: Real, _ x: UnsafePointer<Real>,
  _ c0: UnsafeMutablePointer<Real>, _ c1: UnsafeMutablePointer<Real>, _ c2: UnsafeMutablePointer<Real>,
  _ c3: UnsafeMutablePointer<Real>
) {
  let w = Real.Vector.scalarCount
  let v0 = Real.Vector(repeating: a0), v1 = Real.Vector(repeating: a1)
  let v2 = Real.Vector(repeating: a2), v3 = Real.Vector(repeating: a3)
  var i = 0
  while i + w <= n {
    let xv = load(x + i)
    store(Real.fma(load(c0 + i), v0, xv), c0 + i); store(Real.fma(load(c1 + i), v1, xv), c1 + i)
    store(Real.fma(load(c2 + i), v2, xv), c2 + i); store(Real.fma(load(c3 + i), v3, xv), c3 + i)
    i += w
  }
  while i < n {
    c0[i] += a0 * x[i]; c1[i] += a1 * x[i]; c2[i] += a2 * x[i]; c3[i] += a3 * x[i]
    i += 1
  }
}

/// y += a0 c0 + a1 c1 + a2 c2 + a3 c3 over n elements.
@inline(__always)
func vectorGemv4<Real: BandReductionScalar>(
  _ n: Int, _ y: UnsafeMutablePointer<Real>, _ c0: UnsafePointer<Real>, _ c1: UnsafePointer<Real>,
  _ c2: UnsafePointer<Real>, _ c3: UnsafePointer<Real>, _ a0: Real, _ a1: Real, _ a2: Real, _ a3: Real
) {
  let w = Real.Vector.scalarCount
  let v0 = Real.Vector(repeating: a0), v1 = Real.Vector(repeating: a1)
  let v2 = Real.Vector(repeating: a2), v3 = Real.Vector(repeating: a3)
  var i = 0
  while i + w <= n {
    var yv = load(y + i)
    yv = Real.fma(yv, v0, load(c0 + i)); yv = Real.fma(yv, v1, load(c1 + i))
    yv = Real.fma(yv, v2, load(c2 + i)); yv = Real.fma(yv, v3, load(c3 + i))
    store(yv, y + i)
    i += w
  }
  while i < n {
    y[i] += a0 * c0[i] + a1 * c1[i] + a2 * c2[i] + a3 * c3[i]
    i += 1
  }
}
