# xtb-diagonalization

Faster GFN2-xTB single points on Apple silicon, by attacking the per-SCC-cycle eigensolve. `BRIEF.md` has the plan,
what is already known, the phases and the acceptance criteria.

```
git clone --recurse-submodules https://github.com/philipturner/xtb-diagonalization
```

- `xtb/`: submodule, fork of [grimme-lab/xtb](https://github.com/grimme-lab/xtb) at `68fe777`; add the upstream remote
  with `git -C xtb remote add upstream https://github.com/grimme-lab/xtb`.
- `structures/`: 30 test geometries (n = 124–776 basis functions; closed- and open-shell, small-gap, off-equilibrium),
  described in `structures/MANIFEST.tsv`.
- `reference/`: GFN2 results from the production library (xtb 6.7.1 + Accelerate) for every structure × two spin
  states × electronic temperature 100 / 300 / 1000 K: energy, gradient, charges, orbital eigenvalues and occupations,
  HOMO–LUMO gap and SCC iteration count. `reference/SUMMARY.tsv` has one line per run.
- `scripts/make_reference.py`: drives any `libxtb.dylib` through the xTB C API with the production call sequence
  (`--accuracy`, `--threads` for validation runs).
- `scripts/compare.py`: checks a run against the accuracy gates; `bench.py`, `sweep.py`, `warmstart.py`, `noise.py`:
  timing, throughput, warm starts, SCC noise floor; `scripts/kernels/`: eigensolver kernel studies.
- `FINDINGS.md`: answers to the brief's questions; `baseline.md`: Phase 1 measurements; `LOG.md`: session log.
