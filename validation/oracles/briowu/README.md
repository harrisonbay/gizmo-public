# Pinned public Brio-Wu oracle scaffold

`assets.json` pins the hosted public initial condition by URL, byte size, and
SHA-256. Fetch it with:

```sh
python3 validation/oracles/fetch_assets.py briowu
```

The file contains 50,176 float32 gas particles in 896 approximately vertical
columns of 56 particles. The periodic domain is `4 x 0.25`, not a square of
side `Header/BoxSize=0.25`: the x length is `BOX_LONG_X * BoxSize`. The states
are the standard Brio-Wu discontinuity at `x=2`, with `gamma=2`:

- left: `rho=1`, `P=1`, `v=(0,0,0)`, `B=(0.75,1,0)`;
- right: `rho=0.125`, `P=0.1`, `v=(0,0,0)`, `B=(0.75,-1,0)`.

The current public repository supplies example parameters, not a quantitative
test. A newer unpublished test checks only terminal time, finite fields,
positive density, and a broad density range. The initial file already satisfies
all field assertions, so a no-op that edits only the header time passes. This
is a liveness floor, not a correctness oracle.

`corrected-c-config.sh`, `generated-config.h`, `corrected-c.params`, and
`parameters-usedvalues` record the local reproducible C baseline. The latter is
GIZMO's own record of the subset of input keys consumed by this build.
`evolution-manifest.json` pins the source and executable, raw snapshot hashes,
build/run provenance, reducer, semantic tables, interface oracle, and
quantitative snapshot diagnostics.

The three raw snapshots were reduced to 896 equal-population x-rank bins with
`reduce_briowu_snapshot.py`. Rank bins avoid silently treating the 2-D sheet as
a 1-D particle set while giving a deterministic profile comparison. Sorting by
`x`, then `y`, then particle ID makes the compressed CSV independent of
MPI/HDF5 particle order; the gzip timestamp and embedded filename are fixed.

The file named `snapshot_000.hdf5` has header time zero but is **not** the
unevolved IC. `run.c` computes the initial force, enters the main loop, applies
`do_first_halfstep_kick()`, and only then services the scheduled `t=0` output
while drifting by zero elapsed time. Consequently its coordinates and internal
energy equal the IC, but its stored `vx` and `vy` are already half-kicked (up to
about `0.0182` and `0.0263`). `evolution_t0.csv.gz` preserves that phase and is
a useful first-force/integration oracle; it must not be compared to a
pre-force Rust initialization snapshot.

The corrected-C profile is differential evidence only. It cannot replace an
independent exact solution, convergence study, or the machine-readable curves
used for the published figures. Those missing upstream files are requested in
`../UPSTREAM_REQUEST.md`.

## Rust implementation status

The Rust path now owns the full 50,176-particle planar state: rectangular
periodic cell lists, the normalized 2-D cubic kernel, MFM density, conditioned
2-D MLS moments, batched limited gradients, vector face geometry,
arbitrary-normal HLLD, the MFM entropic/PdV energy correction, Powell/Dedner
terms and safety limiters, the density-loop `Particle_DivVel` estimator,
public-C adaptive smoothing lengths, and synchronized primitive KDK evolution.
The strict CLI accepts this profile without a one-dimensional projection and
writes `t=0`, `0.1`, and `0.2` snapshots.

Run the pinned initialization, three-pass adaptive-H, and full production
two-dimensional RHS gate with:

```sh
validation/oracles/run_rust_briowu_init.sh
```

This gate evaluates 501,760 unordered MFM face interactions and requires a
finite, nonzero force over the real fixture. It also contains regressions for
the oblique discontinuity HLLD branch, exact affine reconstruction, pair
limiters, CFL length, adaptive-H drift, and the C retry ladder.

This is not yet a terminal trajectory claim. Rust currently advances every
particle on a synchronized global step, whereas the C baseline uses
hierarchical individual-particle time bins. The corrected-C logs establish the
global synchronization cadence, but do not prove that every particle occupies
the same bin. Rust also still serializes the wrong leapfrog phase: its
`snapshot_000` is pre-kick and later snapshots are post-second-kick, while C
writes the mixed predictor/actual view during drift. The synchronized public
step currently discards that predictor view between calls, so multi-step
trajectory parity requires a persistent dual actual/predicted state before a
Rust `t=0.2` profile can be promoted to the 5% gate.

## Approximate published-figure reference

`published_exact_figure.csv` is a deterministic digitization of the seven black
dotted physical-variable paths in Figure 3 of Hopkins & Raives, *Accurate,
Meshless Methods for Magnetohydrodynamics* (arXiv:1505.02783v3). The caption
identifies the curves as the exact solution at `t=0.2`. The table contains
`vx`, `vy`, `bx`, `by`, `rho`, `u`, and `pressure`; the eighth panel is a
numerical divergence diagnostic and has no exact curve.

This is an **approximate plot oracle, not the author's numerical table**. The
extraction reads vector path vertices rather than raster pixels, but precision
is still limited by figure generation, axis calibration, path simplification,
and the plotted line itself. In particular, `bx` is represented by only its two
horizontal endpoints, and near-vertical discontinuities contain clustered x
coordinates that must not be casually smoothed or deduplicated. Use this table
for physics-level profile checks with documented tolerances, never bitwise
comparison. The independent source table remains requested upstream.
The CSV is in long form: `field`, zero-based `point_index`, calibrated `x` and
`value`, followed by the original page-space `svg_x` and `svg_y` coordinates
for auditability.

The source figure is `briowu_TP3.pdf` in the arXiv v3 source archive. Its
SHA-256 is
`f3354f2e92c6ff44042cef558910f27cef6cf098be66ef84481734e05d31bb3c`.
The checked conversion was made with:

```sh
pdftocairo -svg briowu_TP3.pdf tmp/pdfs/briowu.svg
```

using Poppler `pdftocairo 26.07.0`. A different converter version may emit
semantically equivalent but byte-different SVG; the extractor intentionally
rejects that ambiguity until its output is reviewed and re-pinned.
The checked SVG SHA-256 is
`64f6160efec9064557c1300d0fd3865bfbe6e8768f0b561fd53e774e8f557fc8`.
Regenerate and run all extraction self-checks with:

```sh
python3 validation/oracles/briowu/extract_published_exact_curves.py \
  tmp/pdfs/briowu.svg \
  validation/oracles/briowu/published_exact_figure.csv
python3 validation/oracles/briowu/extract_published_exact_curves.py \
  --check tmp/pdfs/briowu.svg \
  validation/oracles/briowu/published_exact_figure.csv
```

The script selects exactly seven full-panel `stroke-dasharray="1 3"` paths,
applies per-panel calibrations from labelled major ticks, and rejects the input
unless all fields are finite, x is nondecreasing, point indices are contiguous,
and both endpoint states agree with the Brio-Wu initial states. The rendered
page was also visually checked: the extracted paths correspond to the dotted
curves in all seven physical panels, including the constant `bx=0.75` panel.

Compare a reduced terminal profile with the digitized curves using:

```sh
python3 validation/oracles/compare_briowu_profile.py \
  validation/oracles/briowu/evolution_t0.2.csv.gz \
  --maximum-normalized-l1 vx=0.05 \
  --maximum-normalized-l1 vy=0.05 \
  --maximum-normalized-l1 bx=0.05 \
  --maximum-normalized-l1 by=0.05 \
  --maximum-normalized-l1 rho=0.05 \
  --maximum-normalized-l1 u=0.05 \
  --maximum-normalized-l1 pressure=0.05
```

The metric samples the rank-binned profile in the plotted x domain and reports
mean absolute error divided by each plotted reference range (or by one for the
constant `bx` curve). The corrected-C terminal profile is below 3% for every
field; `u` is largest at 2.93%. The 5% gate deliberately leaves room for
figure-digitization and integrator differences while still rejecting wrong
wave states. Elementwise maxima near vertical jumps are reported but are not
used as the acceptance norm because sub-bin shock-position offsets make them
unstable.

The terminal Rust acceptance gate must still require:

- nontrivial intermediate state at `t=0.1`;
- positivity and finite fields at every step;
- mass, momentum, and total-energy budgets;
- corrected-C profile agreement at `t=0.1` and `t=0.2`;
- an independent exact/profile or convergence oracle once upstream publishes
  it;
- equivalent handling or an explicitly quantified comparison of synchronized
  versus hierarchical time stepping.

The real fixture/rectangular-domain gate and the separate `gamma=2`
corrected-C HLLD table are already active. The existing
`mhd_wave/hlld_flux_oracle.csv` Brio-Wu-labelled row uses `gamma=5/3` and is
not this problem's interface.

Regenerate the semantic tables with:

```sh
uv run --with h5py --with numpy \
  python validation/oracles/reduce_briowu_snapshot.py \
  snapshot_000.hdf5 evolution_t0.csv.gz
uv run --with h5py --with numpy \
  python validation/oracles/reduce_briowu_snapshot.py \
  snapshot_001.hdf5 evolution_t0.1.csv.gz
uv run --with h5py --with numpy \
  python validation/oracles/reduce_briowu_snapshot.py \
  snapshot_002.hdf5 evolution_t0.2.csv.gz
```

Regenerate the interface table from the repository's actual
`hydro/reimann.h` with:

```sh
clang -std=c99 -O2 \
  validation/oracles/briowu/generate_hlld_gamma2_oracle.c \
  -lm -o generate_hlld_gamma2_oracle
./generate_hlld_gamma2_oracle \
  > validation/oracles/briowu/hlld_gamma2_flux_oracle.csv
```

Gate the Rust HLLD implementation against every recorded interface output with:

```sh
cargo test --manifest-path rust/Cargo.toml \
  -p gizmo-hydro --test hlld_brio_wu_gamma2_oracle
```
