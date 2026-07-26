# Pinned public Gresho-vortex scaffold

This directory pins the official **ring** initial condition and parameter file
hosted by the public GIZMO documentation. Fetch and byte-verify them with:

```sh
python3 validation/oracles/fetch_assets.py gresho
```

`public.params` is byte-for-byte identical to both the hosted
`gresho.params` and `scripts/test_problems/gresho.params`.
`public-config.sh` is the literal MFM example configuration printed in that
parameter file. It is not represented as an unpublished `Config.sh` used to
produce a reference trajectory.

The official ring fixture is a float32 HDF5 file with 4,092 type-0 particles,
particle IDs forming the permutation `0..4091`, `BoxSize=1`, and header time
zero. It has exactly two-dimensional coordinates and velocities, density
exactly one, and the public `gamma=1.4` equilibrium. Direct evaluation of the
analytic fields at the stored particle positions gives maximum componentwise
velocity error `5.73e-8` and maximum specific-internal-energy error `4.77e-7`,
consistent with float32 storage.

This is intentionally the ring IC. The separately hosted
`gresho_ics_grid.hdf5` encodes internal energy for `gamma=5/3`, conflicting
with the public parameter/configuration requirement `gamma=1.4`; it must not
silently replace this fixture.

## Analytic equilibrium

Let `r` be minimum-image distance from `(0.5, 0.5)`. Density is one, radial
velocity is zero, and the counter-clockwise tangential velocity is

```text
v_phi = 5 r          for r < 0.2
      = 2 - 5 r      for 0.2 <= r < 0.4
      = 0            otherwise.
```

The pressure balancing the centrifugal acceleration is

```text
P = 5 + 12.5 r^2                         for r < 0.2
  = 9 + 12.5 r^2 - 20 r + 4 log(5 r)    for 0.2 <= r < 0.4
  = 3 + 4 log(2)                         otherwise.
```

For the public ideal gas, `u=P/[(gamma-1) rho]`.

Inspect a snapshot, compare every particle directly with the analytic field,
and write a deterministic 64-bin radial reduction with:

```sh
uv run --with h5py --with numpy \
  python validation/oracles/compare_gresho_analytic.py \
  validation/oracles/gresho/gresho_ics.hdf5 \
  --require-time 0 \
  --official-initial-condition \
  --output-reduced /tmp/gresho-initial.csv.gz
```

The comparator accepts explicit `--maximum METRIC=LIMIT` gates. It deliberately
has no default terminal tolerance: the public materials publish no quantitative
acceptance threshold.

## Corrected-C differential

`evolution-manifest.json` pins a one-rank corrected-C run materialized from
the `harrisonbay/gizmo-public` correctness fork at commit
`f40611353d02f68d8714eb380e92fef1c1c34c9c`. The manifest separately records
the upstream base and every production-C correction in the fork. It used the literal
non-`DEVELOPER_MODE` configuration in `corrected-c-config.sh`, completed 8,192
shared `2^47`-tick steps, and wrote seven scheduled snapshots from zero through
`t=3`. Because developer mode was disabled, GIZMO ignored the input
`ErrTolIntAccuracy`, `CourantFac`, and `MaxRMSDisplacementFac` prompts; the
actual consumed subset is pinned verbatim in `parameters-usedvalues`.

The `t=0` corrected-C snapshot is not the hosted IC: it follows the initial
force and first half-kick. The semantic tables preserve this staggered phase.
At `t=3`, its mass-weighted errors against the analytic equilibrium are:

- tangential-velocity L1 `0.0376017196`;
- radial-velocity RMS `0.0300797144`;
- density L1 `0.00837691988`;
- absolute pressure L1 `0.0476735960`;
- peak tangential velocity `0.908619169`.

These are measured reference values, not public acceptance thresholds. A Rust
port should be gated independently against the analytic solution and then use
the ID-sorted corrected-C tables to localize integration differences.

Regenerate a semantic table with:

```sh
uv run --with h5py --with numpy \
  python -m validation.oracles.reduce_gresho_snapshot \
  snapshot_006.hdf5 evolution_t6.csv.gz
```

## Rust evolution gate

The Rust path is a genuine nonmagnetic, 2.5-D Euler MFM implementation. It uses
two-dimensional positions, kernels, volumes, and smoothing-length predictors
while retaining all three momentum components. Its force path uses the public
HLLC, KT, and exact-Euler retry chain rather than treating zero-field HLLD as
hydrodynamics. Individual particle bins retain inactive predictors and force
caches, evaluate directed active targets, process wakeups, and preserve the
public half-kick snapshot phase. Active-target force batches are threaded but
commit in deterministic particle order.

Run the complete 8,192-event release trajectory with:

```sh
validation/oracles/run_rust_gresho.sh
```

The gate writes and checks all seven public times. Every output must pass
schema, exact IDs, finiteness, positivity, unit-box, strict-2-D, and invariant
per-ID mass checks. The following analytic regression ceilings are physically
grounded but are not a convergence-order proof:

The strict runtime profile validates the literal public configuration, which
does not define `OUTPUT_IN_DOUBLEPRECISION`. The Rust oracle writer
intentionally stores its diagnostic snapshots as float64 so that comparison
and conservation gates do not acquire an additional float32 quantization
floor. This is an output-analysis choice only: it does not change the dynamics,
and snapshot byte/storage-width parity with the public configuration is not
claimed.

- density L1 `0.012`;
- pressure normalized L1 `0.08`;
- specific-internal-energy normalized L1 `0.045`;
- radial-velocity RMS `0.04`;
- tangential-velocity L1 `0.05`;
- full velocity-vector L1 `0.065`;
- annular peak tangential velocity at least `0.85`.

Mass is fixed, component momentum may drift by at most `1e-10`, and angular
momentum by at most 2%. The naive snapshot energy may drift by `1e-5` relative
because public outputs deliberately combine half-step velocity with predicted
thermal energy; corrected C itself varies by `3.3e-6` under that diagnostic.

As a differential guard, every Rust analytic error must also remain below 1.2
times the corrected-C error plus a small metric-specific roundoff/phase floor.
Per-ID position, velocity, density, internal-energy, and smoothing-length
differences localize failures. At every output, the corrected-C density L1,
normalized internal-energy L1, and mean relative smoothing-length difference
must remain below `0.01`, `0.04`, and `0.01`. The terminal per-ID position and
velocity L1 must also remain below `0.08` and `0.12`; a frozen copy of the
initial state misses those ceilings by more than a factor of two. Separate
non-equilibrium operator tests require nonzero evolution. These differential
checks do not replace the analytic criteria, and a pinned resolution sequence
is still required before making a formal convergence-order claim.
