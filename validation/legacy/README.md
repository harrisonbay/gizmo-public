# Legacy sound-wave oracle

This harness turns the smallest shipped public test case,
`scripts/test_problems/soundwave.params`, into a reproducible executable check.
The upstream repository supplies parameter files but no runner, initial
conditions, expected snapshots, or pass/fail criteria.

The harness generates a deterministic 128-particle linear wave derived from the
public test description, compiles the C code in an isolated temporary copy,
runs one MPI rank to the public test's `TimeMax=1.5`, and compares the final
density field with the analytic translated wave. It reports normalized L1
error, amplitude retention, and phase error.
These are useful differential-oracle quantities for a Rust implementation; a
passing result is compatibility evidence, not proof of every physics kernel.

This generated fixture deliberately uses amplitude `1e-2`, large enough for
stable error measurement at low particle count. It is not a byte-for-byte
replacement for Hopkins's public 2048-particle fixture, whose perturbation is
approximately `1e-6`. The independently pinned upstream asset lives under
`validation/oracles/soundwave`; Rust-vs-C snapshot comparisons must identify
which fixture they use.

On Apple Silicon macOS:

```sh
brew install open-mpi gsl hdf5
validation/legacy/run_soundwave.sh
```

The script also requires `make`, `rsync`, and `uv`; `uv` creates an ephemeral
Python environment containing NumPy and h5py. `fftw` is not needed because this
configuration disables gravity and does not enable `PMGRID`.

The shipped `MacBookCellar` Makefile stanza hard-codes specific Homebrew Cellar
versions. The harness intentionally overrides those include/library variables
with stable paths from `brew --prefix`.

The same shared timestep path can be compiled and evolved through the SPH
solver with:

```sh
LEGACY_CONFIG_PROFILE=validation/legacy/soundwave-sph.Config.sh \
  validation/legacy/run_soundwave.sh
```

The default gate checks density, velocity, pressure, mass, momentum, and total
energy. A three-resolution convergence gate additionally requires monotonic
error reduction and at least order `0.8` between 64, 128, and 256 particles:

```sh
validation/legacy/run_soundwave_convergence.sh
```

The verified 64/128/256 MFM run converges at orders `1.86/1.68` for density,
`1.88/1.68` for velocity, and `1.87/1.69` for pressure.
