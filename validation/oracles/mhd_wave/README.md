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
