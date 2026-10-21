// Fused vector multiply-add on one NEON register, for Swift: SIMD2<Double>.addingProduct and SIMD4<Float>
// .addingProduct compile to scalar fmadd, these to fmla. The vector types import into Swift as SIMD2<Double> and
// SIMD4<Float>; the static inline functions are inlined into the Swift caller.
#ifndef CSIMD_H
#define CSIMD_H

typedef double csimd_double2 __attribute__((ext_vector_type(2)));
typedef float csimd_float4 __attribute__((ext_vector_type(4)));

// acc + a * b, rounded once
static inline __attribute__((always_inline)) csimd_double2 csimd_fma_f64(csimd_double2 acc, csimd_double2 a,
                                                                         csimd_double2 b) {
  return __builtin_elementwise_fma(a, b, acc);
}
static inline __attribute__((always_inline)) csimd_float4 csimd_fma_f32(csimd_float4 acc, csimd_float4 a,
                                                                        csimd_float4 b) {
  return __builtin_elementwise_fma(a, b, acc);
}

#endif
