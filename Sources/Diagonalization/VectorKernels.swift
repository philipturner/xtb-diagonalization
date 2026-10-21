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
