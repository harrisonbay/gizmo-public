# Pinned public Sod shock-tube assets

The public documentation supplies a 320-particle equal-mass initial condition,
a 512-particle differential-mass initial condition, and a high-resolution PPM
reference table at `t=5`. `assets.json` pins their public URLs, byte sizes, and
SHA-256 digests. Fetch and verify them with:

```sh
python3 validation/oracles/fetch_assets.py shocktube
```

The equal-mass IC uses variable particle spacing; the differential-mass IC uses
uniform spacing with a mass jump. Both exercise discontinuous initial states
that the smooth sound-wave oracle does not cover. The reference table contains
position, density, pressure, entropy, and x velocity. Its header documents the
periodic coordinate wrapping used for comparison.

The corrected-C terminal state is also checked against that independent PPM
reference with periodic linear interpolation and particle-volume-weighted L1
errors. Density, pressure, entropy, and x-velocity errors are respectively
`0.00481671`, `0.00524103`, `0.00339320`, and `0.00899120`; the exact values
and comparison definition are pinned in the evolution manifest.

After GZ-0010 removes the unrelated top-tree capacity guard, the public
512-particle differential-mass IC also completes 8,192 steps. Its independent
manifest and `t=0`/`t=5` tables are pinned as
`diffmass-evolution-manifest.json`, `diffmass_t0.csv.gz`, and
`diffmass_t5.csv.gz`. Its PPM L1 errors for density, pressure, entropy, and
x-velocity are `0.00523322`, `0.00575777`, `0.00306382`, and `0.00425299`.

`evolution-manifest.json` additionally pins a corrected-C equal-mass run from
commit `66e1006e42a23019fa33da6cc7746457dc527bf9`. It completed 8,192 shared
`2^47`-tick steps on one MPI rank and wrote 11 scheduled snapshots. Reduced
ID-sorted state tables preserve the staggered drift semantics at `t=0` and
`t=5`; first-step tables additionally bracket the endpoint density/force work
with a normal scheduled drift and a post-second-kick observation snapshot.
`post-kick-snapshot.patch` is observation-only and its digest is pinned in the
manifest. The resolved parameter values are preserved verbatim so defaults
cannot silently change the oracle. Raw snapshots are identified by checksum
but omitted.

Run the release Rust/C differential with:

```sh
validation/oracles/run_rust_shocktube.sh
```
