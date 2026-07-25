# GIZMO Rust port

This workspace is the correctness-oriented Rust replacement for GIZMO. It
currently ports the sound-wave initialization, gradient, and face path: strict
configuration and parameter parsing, validated HDF5 gas input, the exact default
one-dimensional cubic kernel, density summation, adaptive smoothing-length
constraints, slope-limited moving-least-squares gradients, default MFM face
geometry, and pairwise primitive reconstruction. It does not evolve a simulation
yet.

```console
validation/oracles/run_rust_soundwave_init.sh
```

Initialization-only mode validates the sound-wave profile, parses the runtime
parameters, loads and ID-aligns the HDF5 state, recomputes density and adaptive
`Hsml`, reconstructs density, velocity, and pressure gradients, validates
one-dimensional face areas, and prints a deterministic JSON summary. Normal
invocations still exit with an explicit `not yet ported` error before evolution.

Crates are layered so that scientific code does not depend on command-line or
configuration parsing:

- `gizmo-core`: typed quantities, vector math, particle indices, and checked
  structure-of-arrays state.
- `gizmo-config`: strict legacy `Config.sh` parsing and deterministic SHA-256
  manifests.
- `gizmo-audit`: finding and provenance records used by correctness tooling.
- `gizmo-params`: strict typed parsing for the legacy runtime parameters used
  by the first vertical slice.
- `gizmo-io`: checked HDF5 sound-wave input with particle-ID alignment.
- `gizmo-hydro`: one-dimensional kernel, density, adaptive `Hsml` solve, and
  slope-limited moving-least-squares gradients, MFM faces, and primitive
  reconstruction.
- `gizmo-cli`: the compatibility command-line boundary.

The public sound-wave fixture is byte-pinned outside the Rust workspace. From
the repository root, run the pinned-data initialization oracle with:

```console
validation/oracles/run_rust_soundwave_init.sh
```

On the pinned fixture, the Rust density sum differs from the stored density by
at most `1.58e-12` relative. The fixture's producer commit is not published, so
this is pinned-data parity rather than proof of parity with our compiled C
baseline. The Rust adaptive solver satisfies `N_eff=4` to floating-point
precision; its smoothing lengths differ by at most `5.0e-4` relative from the
fixture values accepted under the legacy solver's looser neighbor tolerance.
The normalized mean errors of the reconstructed density, velocity, and pressure
gradients against the fitted analytic wave are respectively `3.99e-6`,
`9.99e-7`, and `1.09e-6`; adjacent face areas differ from the analytic unit
area by at most `4.07e-9`.
