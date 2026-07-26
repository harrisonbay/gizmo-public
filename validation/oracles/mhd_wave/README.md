# Pinned public linear fast-MHD-wave oracle

`assets.json` pins the only public initial-condition file by URL, byte size,
and SHA-256. Fetch it with:

```sh
python3 validation/oracles/fetch_assets.py mhd_wave
```

The fixture contains 2,048 gas particles and the uniform state
`rho=1`, `P=0.6`, `v=0`, `B=(1,sqrt(2),0.5)` in GIZMO's internal `mu0=1`
units. With `gamma=5/3`, its sound speed is 1 and its right-going fast speed
is exactly 2. The perturbation is a normalized fast eigenvector: its density
fundamental is `4.47213595644e-7`, although the Hopkins & Raives (2015) prose
describes `delta rho/rho=1e-6`. The pinned bytes are authoritative; the
discrepancy is not silently corrected.

`frontier-config.sh` and `frontier.params` are copied from the test at
`pfhopkins/gizmo@177ad43f57171c355fbcf163bd169eebe60afb06`
(2026-07-24). This newer harness is not part of the `gizmo-public` repository,
so it is recorded separately from the public provenance. It compares only the
initial and terminal snapshots with
`rtol=atol=1e-7`. Since `TimeMax=0.5` is exactly one box crossing at speed 2,
a program that merely advances the header time can satisfy that structure.
It is therefore a compatibility floor, not a correctness oracle.

The Rust port must additionally:

- evolve all 11 public output times and demonstrate nontrivial intermediate
  velocity and magnetic states;
- match analytic Fourier phase, amplitude, and fast-mode polarization;
- match ID-sorted corrected-C semantic tables at intermediate times;
- preserve the `mu0=1` magnetic convention (no extra `sqrt(4*pi)` factor);
- pass conservation, covariance, Dedner-cleaning, and resolution gates.

Run the complete Rust profile (2,048 particles, all 11 outputs, analytic
phase/amplitude at every output, and corrected-C L1 checkpoints) with:

```sh
validation/oracles/run_rust_mhd_wave.sh
```

The pinned optimized run completed 10,250 global KDK steps and 11 snapshots
through `t=0.5`. Every analytic gate passed; at the terminal time the largest
field L1 was `2.41e-9`, the fast-mode amplitudes retained at least `99.47%`,
and the largest phase error was `8.72e-4` radians. These intermediate gates
make a frozen/no-op implementation fail even though the terminal state is one
full wave period from the initial state.

The current `gizmo-public` C baseline is also not equivalent to the newer test:
with four MPI ranks its terminal snapshot violates the newer tolerance for
27 density, 12 internal-energy, and 10 magnetic components. This is recorded
as evidence, not used to weaken the analytic gates.

Reduce any raw snapshot to a deterministic table with:

```sh
uv run --with h5py --with numpy \
  python validation/oracles/reduce_mhd_wave_snapshot.py \
  snapshot_005.hdf5 evolution_t0.25.csv.gz
```

Inspect its analytic Fourier content with:

```sh
uv run --with h5py --with numpy \
  python validation/oracles/compare_mhd_wave_analytic.py snapshot_005.hdf5
```

The public paper's error norm is particle L1. Sparse kernel-support outliers
make a max-element norm misleading for this fixture: the pinned corrected-C
trajectory stays below `1.6e-8` L1 in every field over all 11 outputs even
though a few terminal elements differ by several `1e-7`.

## Corrected-C HLLD interface oracle

`hlld_flux_oracle.csv` is a direct differential oracle for the local MHD
Riemann solve. It was emitted by `generate_hlld_oracle.c`, which includes the
repository's actual `hydro/reimann.h`; the driver does not contain a translated
or shared reimplementation of HLLD. Its compatibility definitions select the
same public-wave mode:

- `MAGNETIC`
- `DIVBCLEANING_DEDNER`
- `HYDRO_MESHLESS_FINITE_MASS`
- `gamma=5/3`
- unconstrained-gradient Dedner limiter `0.75`

The four rows cover the constant fast-wave background, a nonzero-Phi and
discontinuous-normal-field interface, the zero-normal-field degeneracy, and a
Brio-Wu strong discontinuity. Each row records all conservative fluxes plus
contact speed, star total pressure, corrected normal magnetic field, both
Dedner interface terms, and corrected fast speeds. The Rust integration test
`hlld_corrected_c_oracle.rs` uses the contact frame and compares every recorded
output.

At generation, `hydro/reimann.h` was commit
`758fbc9f9e45d27bff3b85864669ccc0ffdd7de1`, with SHA-256
`8807762c1efd465ced84646dac3555d87720d9dfbb6780bdda5eab146117e4ba`.
Regenerate and check the pinned table with:

```sh
cc -std=c11 -O2 \
  validation/oracles/mhd_wave/generate_hlld_oracle.c -lm \
  -o /tmp/gizmo-hlld-oracle
/tmp/gizmo-hlld-oracle |
  diff -u validation/oracles/mhd_wave/hlld_flux_oracle.csv -
cargo test --manifest-path rust/Cargo.toml \
  -p gizmo-hydro --test hlld_corrected_c_oracle
```
