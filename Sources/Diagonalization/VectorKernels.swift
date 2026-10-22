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

// Eight columns per pass: about 17 of the 32 NEON registers, no spills (sixteen would spill).

/// (x . c[0], ..., x . c[7]) over n elements, c[j] = c0 + j ldc.
@inline(__always)
func vectorDot8<Real: BandReductionScalar>(
  _ n: Int, _ x: UnsafePointer<Real>, _ c0: UnsafePointer<Real>, _ ldc: Int
) -> SIMD8<Real> {
  let w = Real.Vector.scalarCount
  var a0 = Real.Vector(), a1 = Real.Vector(), a2 = Real.Vector(), a3 = Real.Vector()
  var a4 = Real.Vector(), a5 = Real.Vector(), a6 = Real.Vector(), a7 = Real.Vector()
  var i = 0
  while i + w <= n {
    let xv = load(x + i), c = c0 + i
    a0 = Real.fma(a0, xv, load(c)); a1 = Real.fma(a1, xv, load(c + ldc))
    a2 = Real.fma(a2, xv, load(c + 2 * ldc)); a3 = Real.fma(a3, xv, load(c + 3 * ldc))
    a4 = Real.fma(a4, xv, load(c + 4 * ldc)); a5 = Real.fma(a5, xv, load(c + 5 * ldc))
    a6 = Real.fma(a6, xv, load(c + 6 * ldc)); a7 = Real.fma(a7, xv, load(c + 7 * ldc))
    i += w
  }
  var s = SIMD8<Real>(
    Real.horizontalSum(a0), Real.horizontalSum(a1), Real.horizontalSum(a2), Real.horizontalSum(a3),
    Real.horizontalSum(a4), Real.horizontalSum(a5), Real.horizontalSum(a6), Real.horizontalSum(a7))
  while i < n {
    for j in 0..<8 { s[j] += x[i] * c0[i + j * ldc] }
    i += 1
  }
  return s
}

/// c[j] += a[j] x for the eight columns c[j] = c0 + j ldc, over n elements.
@inline(__always)
func vectorAxpy8<Real: BandReductionScalar>(
  _ n: Int, _ a: SIMD8<Real>, _ x: UnsafePointer<Real>, _ c0: UnsafeMutablePointer<Real>, _ ldc: Int
) {
  let w = Real.Vector.scalarCount
  let v0 = Real.Vector(repeating: a[0]), v1 = Real.Vector(repeating: a[1])
  let v2 = Real.Vector(repeating: a[2]), v3 = Real.Vector(repeating: a[3])
  let v4 = Real.Vector(repeating: a[4]), v5 = Real.Vector(repeating: a[5])
  let v6 = Real.Vector(repeating: a[6]), v7 = Real.Vector(repeating: a[7])
  var i = 0
  while i + w <= n {
    let xv = load(x + i), c = c0 + i
    store(Real.fma(load(c), v0, xv), c); store(Real.fma(load(c + ldc), v1, xv), c + ldc)
    store(Real.fma(load(c + 2 * ldc), v2, xv), c + 2 * ldc); store(Real.fma(load(c + 3 * ldc), v3, xv), c + 3 * ldc)
    store(Real.fma(load(c + 4 * ldc), v4, xv), c + 4 * ldc); store(Real.fma(load(c + 5 * ldc), v5, xv), c + 5 * ldc)
    store(Real.fma(load(c + 6 * ldc), v6, xv), c + 6 * ldc); store(Real.fma(load(c + 7 * ldc), v7, xv), c + 7 * ldc)
    i += w
  }
  while i < n {
    for j in 0..<8 { c0[i + j * ldc] += a[j] * x[i] }
    i += 1
  }
}

/// y += sum_j a[j] c[j] for the eight columns c[j] = c0 + j ldc, over n elements.
@inline(__always)
func vectorGemv8<Real: BandReductionScalar>(
  _ n: Int, _ y: UnsafeMutablePointer<Real>, _ c0: UnsafePointer<Real>, _ ldc: Int, _ a: SIMD8<Real>
) {
  let w = Real.Vector.scalarCount
  let v0 = Real.Vector(repeating: a[0]), v1 = Real.Vector(repeating: a[1])
  let v2 = Real.Vector(repeating: a[2]), v3 = Real.Vector(repeating: a[3])
  let v4 = Real.Vector(repeating: a[4]), v5 = Real.Vector(repeating: a[5])
  let v6 = Real.Vector(repeating: a[6]), v7 = Real.Vector(repeating: a[7])
  var i = 0
  while i + w <= n {
    let c = c0 + i
    var yv = load(y + i)
    yv = Real.fma(yv, v0, load(c)); yv = Real.fma(yv, v1, load(c + ldc))
    yv = Real.fma(yv, v2, load(c + 2 * ldc)); yv = Real.fma(yv, v3, load(c + 3 * ldc))
    yv = Real.fma(yv, v4, load(c + 4 * ldc)); yv = Real.fma(yv, v5, load(c + 5 * ldc))
    yv = Real.fma(yv, v6, load(c + 6 * ldc)); yv = Real.fma(yv, v7, load(c + 7 * ldc))
    store(yv, y + i)
    i += w
  }
  while i < n {
    for j in 0..<8 { y[i] += a[j] * c0[i + j * ldc] }
    i += 1
  }
}
