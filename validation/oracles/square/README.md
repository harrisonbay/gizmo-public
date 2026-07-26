# Pinned public advected-Square oracle

This bundle pins the hosted Square initial condition and literal public
parameter file. Fetch or byte-verify them with:

```sh
python3 validation/oracles/fetch_assets.py square
```

The hosted fixture does **not** match the documentation. The prose says
`64^2` particles and boost `(142.3,-31.4)`; the file contains a `128^2`
cell-centered lattice and boost `(1243,-358)`. This oracle validates the
hosted file explicitly instead of silently mixing those problems.

The float32 fixture has 16,384 IDs `0..16383`. The central 64-by-64 square has
four times the particle mass and one quarter the specific internal energy, so
pressure is uniform. Restart flag zero ignores the stored density and
smoothing length. The public three-pass cubic initialization produces

```text
rho_low  = 0.007814580197309413
rho_high = 0.03125832078923765
h        = 0.01526682428894447
pi h^2 rho / m = 12.00004427713
```

## Exact trajectory gate

Every particle has the analytic planar trajectory

```text
x_i(t) = (x_i(0) + 1243 t) mod 1
y_i(t) = (y_i(0) -  358 t) mod 1.
```

The full gate checks all 21 public outputs at `t=0,0.5,...,10`, sorting each
snapshot by ID. It requires exact per-ID masses, minimum-image position error
`<=1e-9`, velocity error `<=2e-9`, internal-energy error `<=1e-9`, relative
density and smoothing-length drift `<=1e-9`, relative pressure span `<=2e-9`,
effective neighbors in `[11.999,12.001]`, mass drift `<=1e-15`, component
momentum drift `<=1e-10`, and relative energy drift `<=2e-12`.
The density and smoothing-length gates are both relative to snapshot zero and
absolute against the post-initialization values above, so a compensating
rescaling cannot preserve the neighbor count and evade the gate.

Odd snapshots have a half-box x translation and therefore reject a frozen
run. The public cadence aliases the y translation to an integer and the
terminal state wraps to the initial phase, so a separate Rust hierarchy test
uses a non-aliased large y boost. This equilibrium problem validates
initialization, drift/output phase, Galilean invariance, and absence of contact
diffusion; it does not by itself prove nonzero force evolution.

The corrected-C reference completed 4,096 hierarchy events. Its maximum
position error is `5.52e-13`, velocity drift is zero, internal-energy error is
`5.03e-13`, relative density/H drift remains below `2.3e-13`, and conserved
totals are unchanged. `corrected-c-manifest.json` pins the retained artifacts
and records the unavailable source-commit/compiler provenance explicitly. The
raw endpoint snapshots are digest-pinned but not committed because of their
size. `reduce_square_trajectory.py` is the pinned deterministic reducer used
to generate the committed 21-row table; given the raw output directory,
`--check corrected-c-trajectory.csv` verifies it byte for byte.

Run Rust and the gate with:

```sh
validation/oracles/run_rust_square.sh
```

To check an existing output directory:

```sh
GIZMO_SQUARE_OUTPUT_DIR=/path/to/output \
  validation/oracles/run_rust_square.sh
```
