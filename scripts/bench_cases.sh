#!/bin/zsh
# The two benchmark cases (2026-10-22), on an otherwise idle machine (check with ~/bin/topcpu first: background daemons
# such as mediaanalysisd distort the 8-thread runs). Tables: scripts/cases_table.py. Libraries: production (the renderer's libxtb,
# FP64 only: no SCC precision API), lib-e2e5bee (this morning's tip, before today's speedups), lib-par (today: symv x8,
# ormtr by gemm, fused panel products, OpenMP divide and conquer and two-thread back-transformation).
# Case A, throughput: sweep.py 8 x 1 (8 processes, 1 thread each; jobs/s). Case B, latency: bench.py, one process with
# OMP_NUM_THREADS=8 (integrals, and lib-par's eigensolver steps), VECLIB_MAXIMUM_THREADS=1; median of 5 cold single
# points. Thread variables are set at launch. FP32 = SCC precision API, rmsq 1e-3 (FP64 restart on failure).
setopt NO_BG_NICE
cd /Users/philipturner/Documents/MolecularRenderer/xtb-diagonalization
PY=~/miniforge3/bin/python
PROD=../claude-molecular-renderer/libxtb.dylib
mkdir -p build/cases
CASES=facet_closed:0,facet_dbpair_tool:1,s3_S42_159:1,diamond233:0,sic_n1204:0,sic_n1788:0
# Case B: latency, 8 OpenMP threads
for L in prod e2e5bee par; do
  lib=build/lib-$L.dylib; [[ $L == prod ]] && lib=$PROD
  for mode in fp64 fp32; do
    [[ $L == prod && $mode == fp32 ]] && continue
    extra=(); [[ $mode == fp32 ]] && extra=(--fp32 1e-3)
    OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=1 $PY scripts/bench.py $lib build/cases/latency8-$L-$mode.json --cases $CASES --reps 5 $extra > build/cases/latency8-$L-$mode.log 2>&1
  done
done
# Case B references: production and lib-par with 1 thread
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 $PY scripts/bench.py $PROD build/cases/latency1-prod-fp64.json --cases $CASES --reps 5 > build/cases/latency1-prod-fp64.log 2>&1
for mode in fp64 fp32; do
  extra=(); [[ $mode == fp32 ]] && extra=(--fp32 1e-3)
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 $PY scripts/bench.py build/lib-par.dylib build/cases/latency1-par-$mode.json --cases $CASES --reps 5 $extra > build/cases/latency1-par-$mode.log 2>&1
done
# Case A: throughput 8 x 1
for c in facet_closed:0 s3_S42_159:1 diamond233:0 sic_n1204:0; do
  for L in prod e2e5bee par; do
    lib=build/lib-$L.dylib; [[ $L == prod ]] && lib=$PROD
    for mode in fp64 fp32; do
      [[ $L == prod && $mode == fp32 ]] && continue
      extra=(); [[ $mode == fp32 ]] && extra=(--fp32 1e-3)
      jobs=20; [[ $c == sic_n1204:0 ]] && jobs=3
      $PY scripts/sweep.py $lib build/cases/tput-${c%%:*}-$L-$mode.json --case $c --configs 8x1 --jobs $jobs $extra > build/cases/tput-${c%%:*}-$L-$mode.log 2>&1
    done
  done
done
rm -f .sccnotconverged
