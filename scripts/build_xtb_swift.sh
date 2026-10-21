#!/bin/sh
# Build libxtb.dylib with the Swift kernels of this repository's package (Sources/Diagonalization) in place of
# xtb/src/mctc/lapack/sytrd.c: the same recipe as build_xtb.sh, plus xtb's XTB_SYTRD_LIBRARIES hook. The Swift code is
# linked statically; the library then depends only on the Swift runtime in /usr/lib/swift (part of macOS).
#
#   scripts/build_xtb_swift.sh SRC_DIR BUILD_DIR [JOBS]     e.g. scripts/build_xtb_swift.sh xtb build/xtb-build-swift 4
#
# SwiftPM's own sandbox can't nest inside Claude Code's, hence --disable-sandbox. CMake configure downloads mctc-lib.
set -e
SRC=$1; BUILD=$2; JOBS=${3:-4}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
swift build --package-path "$ROOT" -c release --product Diagonalization --disable-sandbox
SWIFTLIB="$ROOT/.build/release/libDiagonalization.a"
SDK=$(xcrun --show-sdk-path)
# The Swift objects ask for their runtime libraries by name (-lswiftCore, ...); these are the directories to find them.
LIBS="$SWIFTLIB;-L$SDK/usr/lib/swift;-L/usr/lib/swift"
cmake -S "$SRC" -B "$BUILD" \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_COMPILER="gcc-16" \
 -DCMAKE_Fortran_COMPILER="gfortran-16" \
 -DWITH_TBLITE=OFF \
 -DWITH_CPCMX=OFF \
 -DWITH_TESTS=OFF \
 -DBLA_VENDOR=Apple \
 -DMCTCLIB_FIND_METHOD=fetch \
 -DXTB_SYTRD_LIBRARIES="$LIBS"
make -C "$BUILD" -j"$JOBS"
otool -L "$BUILD"/libxtb.*.dylib | grep -E 'Accelerate|swift|libxtb' || true
# The tridiagonal reduction must come from the Swift module, not from sytrd.c.
nm "$BUILD"/libxtb.*.dylib | grep -q 's15Diagonalization16xtb_dsytrd_upper' && echo "xtb_dsytrd_upper: Swift"
