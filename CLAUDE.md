# Instructions for Claude: xtb-diagonalization

Goal: make GFN2-xTB single points on Apple silicon (M1 Max) at least 2x faster at n ≈ 410–650 basis functions by
attacking the per-SCC-cycle eigensolve, with every speedup measured and every accuracy gate passed. BRIEF.md is the
plan and the acceptance criteria; read it in full at the start of each session, then `LOG.md` if it exists.

## Hard rules
- Write only inside this repository (including `build/`, git-ignored). The mechanosynthesis repository
  `../claude-molecular-renderer` is read-only; never modify or replace anything in it. The baseline (production
  6.7.1 + Accelerate, behind `reference/`) is
  `../claude-molecular-renderer/.build/archive/2026-10-22-xtb-production/libxtb.dylib`. Since 2026-10-22 that
  repository's root `libxtb.dylib` is this fork's `diagonalization` build (FP32 on by default there), not the baseline.
  Never modify the Homebrew xtb install.
- xTB source changes go on the public branch `diagonalization` of the `xtb/` submodule (fork github.com/philipturner/xtb;
  remote `upstream` = grimme-lab/xtb), one commit per step with its measurement in the message, plus the matching
  submodule-pointer commit here. Never commit to the fork's `main`: it only fast-forwards to upstream `main`
  (the user runs `git -C xtb push origin upstream/main:main`). Claude commits; the user pushes (permissions deny
  `git push` in every form). `diagonalization` is never force-pushed (merge upstream, don't rebase). Ask the user before
  pull requests upstream or any other outward-facing step.
- Cold SCF is the default (fresh results object per single point). Warm starts (chained handle, or one fixed seed as in
  swift-xtb's `xTB_Seed`) are an optional engineering variable: report them separately, never in place of cold numbers.
- Never loosen SCC convergence thresholds. Report measured numbers only; say plainly what could not be measured.
- `reference/` holds the production library's results: never overwrite it. Runs with other libraries go to
  `make_reference.py --out build/...`.

## Running things
- Timed runs: `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`, median of 5, AC power, and only when the Mac is otherwise
  idle. Other Claude sessions (the mechanosynthesis agents) run long GFN2 jobs on this Mac: check the load first and
  ask the user for a quiet window when it is busy. Untimed work (builds, validation) can share the machine: keep to at
  most 4 processes.
- Python: `~/miniforge3/bin/python` (numpy). Don't wrap python in `timeout` (it runs under Rosetta and numpy fails).
- Downloads (meson subprojects, git fetch) need Bash `allowed_domains` (github.com, codeload.github.com,
  api.github.com); prefer curl or git over Python urllib, whose chunked reads break through the sandbox proxy.
- xtb writes `.sccnotconverged` into the working directory when an SCF fails; it is git-ignored, delete it anyway.

## Layout
`BRIEF.md` plan · `xtb/` fork submodule · `structures/` test set + `MANIFEST.tsv` · `reference/` production results +
`SUMMARY.tsv` · `scripts/make_reference.py` driver (`--lib`, `--out`, `--only`, `--workers`) ·
`scripts/provenance/` how the structures were made · `build/` builds and scratch.

## End of session
Append a short entry to `LOG.md` (what was measured, with numbers; what changed; next step) and commit your own files by
explicit path.
