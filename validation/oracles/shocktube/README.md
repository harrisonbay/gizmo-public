# Pinned public Sod shock-tube assets

The public documentation supplies two 320-particle initial conditions and a
high-resolution PPM reference table at `t=5`. `assets.json` pins their public
URLs, byte sizes, and SHA-256 digests. Fetch and verify them with:

```sh
python3 validation/oracles/fetch_assets.py shocktube
```

The equal-mass IC uses variable particle spacing; the differential-mass IC uses
uniform spacing with a mass jump. Both exercise discontinuous initial states
that the smooth sound-wave oracle does not cover. The reference table contains
position, density, pressure, entropy, and x velocity. Its header documents the
periodic coordinate wrapping used for comparison.

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
