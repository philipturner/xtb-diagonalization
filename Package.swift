// swift-tools-version: 5.9
// Swift prototypes of the eigensolver kernels, linkable into libxtb (see Sources/Diagonalization/Exports.swift and
// scripts/build_xtb_swift.sh). Build with `swift build -c release`.
import PackageDescription

let package = Package(
  name: "xtb-diagonalization",
  platforms: [.macOS(.v13)],
  products: [
    // Linked into libxtb.dylib in place of xtb/src/mctc/lapack/sytrd.c.
    .library(name: "Diagonalization", type: .static, targets: ["Diagonalization"]),
    .executable(name: "sytrd-bench", targets: ["SytrdBench"]),
  ],
  targets: [
    // The legacy Fortran-ABI BLAS/LAPACK symbols in Accelerate, the ones libxtb links (BLA_VENDOR=Apple).
    .target(
      name: "CLAPACK",
      linkerSettings: [.linkedFramework("Accelerate")]),
    // Vector fused multiply-add for the Swift kernels.
    .target(name: "CSIMD"),
    .target(
      name: "Diagonalization",
      dependencies: ["CLAPACK", "CSIMD"],
      swiftSettings: [.unsafeFlags(["-enforce-exclusivity=unchecked"])]),
    .executableTarget(
      name: "SytrdBench",
      dependencies: ["CLAPACK", "Diagonalization"]),
  ]
)
