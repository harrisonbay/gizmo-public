# GIZMO Rust port

This workspace is the correctness-oriented Rust replacement for GIZMO. It
currently ports the sound-wave initialization, gradient, and face path: strict
configuration and parameter parsing, validated HDF5 gas input, the exact default
one-dimensional cubic kernel, density summation, adaptive smoothing-length
constraints, slope-limited moving-least-squares gradients, default MFM face
geometry, pairwise primitive reconstruction, and the ideal-gas one-dimensional
MFM HLLC/KT/exact Riemann flux, including conservative pair orientation and
lab-frame deboost, face-closure fallback, and the low-contact-speed
entropic/PdV energy correction. It does not update particle states or evolve a
simulation yet.

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
  reconstruction, plus corrected ideal-gas MFM HLLC/KT/exact, pair, and
  entropic/PdV fluxes.
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
area by at most `4.07e-9`. All 2,048 adjacent pairs use HLLC and the entropic
branch, and both raw and corrected fluxes are exactly antisymmetric under pair
reversal in this fixture-backed invariant test. The full unordered-pair spatial
operator evaluates 4,096 interacting pairs and conserves global momentum and
energy rates to `8.9e-18` and `1.5e-24`.

The Riemann port preserves the corrected two-rarefaction vacuum criterion and
the legacy distinction between HLLC failure modes: negative or non-finite
pressure falls back to the MFM KT flux, while a positive pressure above the
configured limiter invokes a finite-checked exact ideal-gas solve. Exact-solver
nonconvergence is reported explicitly and cannot be misclassified as vacuum.
The pair API includes the legacy reconstruction retries, orientation, area
integration, lab-frame deboost, and the closure-leak rule that disables
reconstruction before solving. The entropic/PdV API then preserves the legacy
strict speed thresholds, condition-number override, independent kernel
derivatives, and KT-specific energy-delta semantics. Particle state updates and
CLI evolution remain behind the timestep-selection boundary described below.

The hydro crate also contains the first synchronized evolution slice: exact
unordered-pair accumulation into extensive momentum/total-energy rates,
conversion to acceleration and specific-internal-energy rate, the raw legacy
Courant estimate, and the noncosmological kick-drift-kick predictor ordering
with the half-loss energy limiter. The non-`LONG_INTEGER_TIME` synchronized
power-of-two timeline is also ported; it maps the fixture's raw
`3.6621e-5` Courant bound to the same initial `8192`-tick,
`2.288818359375e-5` step seen in the C run. This is not yet wired to normal CLI
execution. Before evolved snapshots can be called C-parity results, the
remaining acceleration/displacement timestep bounds must be proven inactive
or ported and the Rust result checked against the named-commit C evolution
tables in `validation/oracles/soundwave/evolution-manifest.json`. The baseline
run confirms all particles remain in the same `8192`-tick bin through all
65,536 steps, so global synchronized stepping is valid for this fixture.
