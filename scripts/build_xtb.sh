#!/bin/sh
# Build libxtb.dylib against Accelerate the way production is built (swift-xtb compile-macos.sh, grimme-lab/xtb#1319):
# CMake, Homebrew GCC 16, BLA_VENDOR=Apple (legacy Accelerate LAPACK symbols), no tblite / CPCM-X / tests, mctc-lib
# fetched at the version xtb pins. Out-of-source: the source tree stays clean.
#
#   scripts/build_xtb.sh SRC_DIR BUILD_DIR [JOBS]     e.g. scripts/build_xtb.sh xtb build/xtb-build-main 4
#
# Downloads at configure time (mctc-lib and friends): run with network access to github.com / codeload.github.com.
set -e
SRC=$1; BUILD=$2; JOBS=${3:-4}
cmake -S "$SRC" -B "$BUILD" \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_COMPILER="gcc-16" \
 -DCMAKE_Fortran_COMPILER="gfortran-16" \
 -DWITH_TBLITE=OFF \
 -DWITH_CPCMX=OFF \
 -DWITH_TESTS=OFF \
 -DBLA_VENDOR=Apple \
 -DMCTCLIB_FIND_METHOD=fetch
make -C "$BUILD" -j"$JOBS"
otool -L "$BUILD"/libxtb.*.dylib | head -1
otool -L "$BUILD"/libxtb.*.dylib | grep -E 'Accelerate|openblas|lapack|blas' || true
