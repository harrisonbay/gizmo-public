# GIZMO Rust port

This workspace is the correctness-oriented Rust replacement for GIZMO. It
currently ports the public one-dimensional MFM sound-wave, both equal- and
differential-mass Sod shock-tube paths, and the Woodward-Colella interacting
blastwave, plus the two-fluid dusty-box and dusty-wave grain-drag problems: strict
configuration and parameter parsing, validated HDF5 gas input, the exact default
one-dimensional cubic kernel, density summation, adaptive smoothing-length
constraints, slope-limited moving-least-squares gradients, default MFM face
geometry, pairwise primitive reconstruction, and the ideal-gas one-dimensional
MFM HLLC/KT/exact Riemann flux, including conservative pair orientation and
lab-frame deboost, face-closure fallback, and the low-contact-speed
entropic/PdV energy correction. The strict restart-0 CLI evolves that profile
with synchronized KDK stepping and writes checked, upstream-compatible HDF5
snapshots. The Brio-Wu slice additionally implements rectangular-periodic 2-D
MFM geometry, adaptive smoothing lengths, limited planar MLS gradients,
arbitrary-normal HLLD, Powell/Dedner terms, synchronized 2-D KDK, exact
per-particle time-bin selection, and the hierarchical initial kick/drift
event. Other physics/configuration profiles fail closed.

```console
validation/oracles/run_rust_soundwave_init.sh
validation/oracles/run_rust_shocktube.sh
validation/oracles/run_rust_interactblast.sh
validation/oracles/run_rust_dustybox.sh
validation/oracles/run_rust_dustywave.sh
validation/oracles/run_rust_briowu_init.sh
```

The complete 65,536-step corrected-C differential is intentionally separate
from the fast initialization check:

```console
validation/oracles/run_rust_soundwave_long.sh
```

The shock-tube differential covers nonuniform particle spacing, a discontinuous
state, unequal particle masses, the public C face-closure correction to the
adaptive neighbor target,
8,192 synchronized KDK steps, and all 11 scheduled outputs through `t=5`.
Against the pinned corrected-C terminal drift, its maximum absolute
position/velocity differences are `4.27e-14`/`1.91e-13`; maximum relative
density/internal-energy/smoothing-length differences are
`1.31e-13`/`1.97e-13`/`4.67e-13`.
For the 512-particle differential-mass case, terminal absolute
position/velocity errors are `3.56e-14`/`4.16e-13`, with density,
internal-energy, and smoothing-length absolute errors below `5.04e-13`.
Both terminal states are also gated against the independent public PPM table
with volume-weighted L1 norms.

The interacting-blast slice adds non-periodic neighbor geometry and exact
public-C reflective-wall drift semantics, including the legacy ID-dependent
inward nudge and predictor reset. Its corrected-C oracle runs 262,144 shared
steps through `t=0.038` and is additionally gated against the supplied
20,000-zone reference solution. The hosted IC has 512 particles, contrary to
the public documentation's statement that it has 400; this discrepancy is
pinned as oracle metadata rather than silently normalized.

The dusty-wave slice adds validated type-3 grain HDF5 state, gas-property
interpolation at grain kernels, nonlinear finite-step Epstein drag, and
kernel-weighted gas backreaction; drag heating is absent, matching the disabled
term in the public C path. Its corrected-C oracle contains 64 gas particles
plus 64 grains, 32,768 synchronized steps, and initial, `t=1.2`, and terminal
differentials.
At the public reference time, Rust and corrected C differ by only
`1.54e-10` gas-velocity RMS and `1.48e-11` grain-velocity RMS. Both are also
gated against the supplied 512-point solution, and every drag batch checks
total momentum.

The dusty-box fixture independently exercises the same coupling at an initial
relative velocity of one rather than `~1e-4`. Its public analytic Epstein
solution gates the entire 251-output decay curve, while corrected-C tables pin
per-particle initial, midpoint, and terminal states. The run exposed and fixed
GZ-0012: accumulated output-time roundoff previously created a duplicate final
snapshot whose staggered grain/gas phases violated total momentum by
`2.37e-7`. Rust and corrected C now emit one terminal drift at exact `t=2.5`;
their gas and grain velocities agree to below `7e-14`, and the corrected-C
trajectory stays within `5.31e-5` of the analytic solution.

Initialization-only mode validates either supported profile, parses the runtime
parameters, loads and ID-aligns the HDF5 state, recomputes density and adaptive
`Hsml`, reconstructs density, velocity, and pressure gradients, validates
one-dimensional face areas, and prints a deterministic JSON summary. A normal
restart-0 invocation evolves the same checked state through `TimeMax` and writes
the configured snapshot sequence.

Crates are layered so that scientific code does not depend on command-line or
configuration parsing:

- `gizmo-core`: typed quantities, vector math, particle indices, and checked
  structure-of-arrays state.
- `gizmo-config`: strict legacy `Config.sh` parsing and deterministic SHA-256
  manifests.
- `gizmo-audit`: finding and provenance records used by correctness tooling.
- `gizmo-params`: strict typed parsing for the legacy runtime parameters used
  by the first vertical slice.
- `gizmo-io`: checked HDF5 gas input with particle-ID alignment and
  upstream-compatible snapshot output and metadata.
- `gizmo-hydro`: one-dimensional kernel, density, adaptive `Hsml` solve, and
  slope-limited moving-least-squares gradients, MFM faces, and primitive
  reconstruction, plus corrected ideal-gas MFM HLLC/KT/exact, pair, and
  entropic/PdV fluxes; it also owns the current two-dimensional MFM/MHD slice.
- `gizmo-cli`: the strict initialization/evolution compatibility boundary.

The public sound-wave fixture is byte-pinned outside the Rust workspace. From
the repository root, run the pinned-data initialization oracle with:

```console
validation/oracles/run_rust_soundwave_init.sh
```

On the pinned fixture, the Rust density sum differs from the stored density by
at most `4.44e-16` relative. The fixture's producer commit is not published, so
this is pinned-data parity rather than proof of parity with our compiled C
baseline. The Rust adaptive solver satisfies `N_eff=4` to floating-point
precision; its smoothing lengths differ by at most `5.0e-4` relative from the
fixture values accepted under the legacy solver's looser neighbor tolerance.
For restart-0 compatibility, the CLI also reproduces the public C gravity-tree
initial guess bit-for-bit and its completed smoothing lengths to `4.45e-16`
relative against the corrected-C `t=0` table.
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
CLI evolution use the complete timestep-selection path described below.

The hydro crate also contains synchronized sound-wave evolution: exact
unordered-pair accumulation into extensive momentum/total-energy rates,
conversion to acceleration and specific-internal-energy rate, the raw legacy
Courant estimate, and the noncosmological kick-drift-kick predictor ordering
with the half-loss energy limiter. The default `LONG_INTEGER_TIME` synchronized
power-of-two timeline is also ported; it maps the fixture's raw
`3.6621e-5` Courant bound to the same initial `2^44`-tick,
`2.288818359375e-5` step seen in the C run. The strict restart-0 CLI uses this
path, including the complete public sound-wave timestep bounds and checked HDF5
snapshot output. The baseline C run confirms all particles remain in the same
`2^44`-tick bin through all 65,536 steps, so global synchronized stepping is
valid for this fixture. The opt-in long differential checks the Rust trajectory
against the named-commit C evolution tables in
`validation/oracles/soundwave/evolution-manifest.json`.
