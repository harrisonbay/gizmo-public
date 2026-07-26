# Pinned public McNally Kelvin--Helmholtz oracle

This directory pins the public GIZMO particle initial condition and parameter
file together with the benchmark author's two machine-readable reference
curves. Fetch or byte-verify all four public artifacts with:

```sh
python3 validation/oracles/fetch_assets.py kh_mcnally
```

`public.params` is byte-for-byte identical to both the hosted parameter file
and `scripts/test_problems/kh_mcnally_2d.params`. `public-config.sh` is the
literal MFM example configuration printed in that file; it is not an
unpublished configuration used to create the reference curves. In particular,
neither `VISCOSITY` nor `CONDUCTION` is enabled, so the coefficient prompts
present in `public.params` do not make this a diffusive benchmark.

The literal GIZMO parameter file has `TimeMax 10`. That is preserved here.
The independent reference tables contain 76 samples at approximately `0.02`
intervals and end at **t=1.5**. They must never be extrapolated to the public
run's `t=10` endpoint. The later inviscid nonlinear flow is not a converged
reference solution. `reference.params` changes only `TimeMax` to `1.5`; it is
the trajectory-gate input, while `public.params` remains the literal supported
public profile.

## Provenance and diagnostics

The benchmark is defined by McNally, Lyra, and Passy, *A Well-Posed
Kelvin-Helmholtz Instability Test and Comparison*, ApJS 201, 18 (2012),
[doi:10.1088/0067-0049/201/2/18](https://doi.org/10.1088/0067-0049/201/2/18),
[arXiv:1111.1764](https://arxiv.org/abs/1111.1764). The authors publish the
reference files at <https://www.colinmcnally.ca/khcomp/>.

For a meshless quadrature point, equations 14--17 of the paper use its volume
`w_i`. For the GIZMO particle state the analyzer uses `w_i=m_i/rho_i`, not
equal particle weighting and not `h_i^2`:

```text
e_i = exp(-4 pi |y_i - 0.25|)                 y_i < 0.5
    = exp(-4 pi |(1-y_i) - 0.25|)             y_i >= 0.5
s_i = v_y,i w_i sin(4 pi x_i) e_i
c_i = v_y,i w_i cos(4 pi x_i) e_i
d_i = w_i e_i
M   = 2 sqrt[(sum(s_i)/sum(d_i))^2 + (sum(c_i)/sum(d_i))^2].
```

The second diagnostic is `max_i(0.5 rho_i v_y,i^2)`. The paper calls this
kinetic-energy density, while the download page and filename sometimes call it
"specific" kinetic energy. The factor of density in the paper and table header
is authoritative.

Analyze a GIZMO-format snapshot with:

```sh
uv run --with h5py --with numpy \
  python validation/oracles/analyze_kh_mcnally.py snapshot.hdf5
```

Add `--official-initial-condition` when checking the pinned IC. The analyzer
interpolates each table independently only for snapshot times in `[0, 1.5]`;
outside that interval it reports that no reference comparison is available.
It deliberately publishes measurements rather than inventing an acceptance
tolerance. The GCI uncertainty describes the high-resolution Pencil reference,
not the much larger discretization error expected from this 66,868-particle
fixture.

## Strict trajectory gate

`check_kh_mcnally_trajectory.py` checks snapshots `000` through `015`, at
exactly `t=0,0.1,...,1.5`. Later snapshots are deliberately ignored: the
published curves do not support extrapolation beyond `t=1.5`.

The primary mode-amplitude gate is the published curve itself. Each point must
be within `max(2.5e-4, 20% of the published amplitude)`. The independently run
corrected C trajectory is also checked with a secondary
absolute `5e-4` proximity band. Corrected C's
largest absolute error is `0.01768671` at `t=1.5`; its largest relative error
is `13.95%` at `t=1.3`. These limits contain measured resolution error without
accepting a frozen perturbation, while the tighter corrected-C gate verifies
implementation equivalence. The terminal mode must additionally grow by at
least `8x`; corrected C grows by `13.03x`.

The maximum vertical kinetic-energy density is an intentionally noisy extrema
diagnostic. Corrected C reaches `4.72x` the published value near `t=0.9`.
Accordingly the independent published curve has only a broad blow-up ceiling,
`max(0.025, 3x the published value)`, rather than a false precision band. The
port-equivalence check additionally requires proximity to corrected C within
`max(2.5e-5, 15% of the corrected-C value)`, preventing a solver that
artificially suppresses the instability from passing.

Every snapshot must retain all 66,868 IDs exactly as `1..66868`, preserve its
per-ID masses bit-for-bit, remain finite and strictly positive where required,
keep `pi h^2 rho/m` in the broad effective-neighbor range `[39.5,40.5]`, and
satisfy:

- relative total-mass drift `<= 1e-12`;
- absolute component-momentum drift `<= 1e-10`;
- relative diagnostic total-energy drift `<= 1e-4`.

The corrected C measurements are respectively `0`, `2.78e-17`, and
`3.37e-7`. The energy limit is broader because public snapshots combine
half-step velocities with predicted thermal state. Corrected C's actual
effective-neighbor extrema over all sixteen snapshots are `39.94894` and
`40.05304`.

Run the Rust trajectory and gate it with:

```sh
validation/oracles/run_rust_kh_mcnally.sh
```

The runner launches `reference.params`, whose vocabulary is identical to the
public file but whose `TimeMax` is shortened from `10` to the published
reference endpoint `1.5`. `public.params` remains the literal hosted file and
continues to describe the full public run.

To gate an existing output directory without launching another trajectory:

```sh
GIZMO_KH_OUTPUT_DIR=/path/to/output \
  validation/oracles/run_rust_kh_mcnally.sh
```

`corrected-c-manifest.json` pins the 4-rank corrected-C run's configuration,
parameter, and reduced-CSV digests. The retained run directory lacked Git
metadata, so the manifest leaves `source_commit` null instead of asserting an
unverifiable revision. It also leaves the compiler version null because that
record was not retained; the MPI rank count and launch command are pinned.

## Known phase mismatch

The paper writes the outer stream as `(rho, v_x)=(1,+0.5)` and the middle
stream as `(2,-0.5)`. The hosted GIZMO IC has the opposite assignment:
approximately `(2,-0.5)` outside and `(1,+0.5)` in the middle. This is exactly
the paper state translated by `0.5` in periodic `y`, not a different physical
problem. The seeded `v_y=0.01 sin(4 pi x)` and both diagnostics above are
unchanged by that translation. The official-IC check evaluates the stored
GIZMO phase explicitly so a future, non-equivalent fixture change cannot hide
behind periodicity.

The hosted IC contains 66,868 disordered float32 particles, IDs `1..66868`,
and quadrature volume summing to about `1.00027846`. Its initial measured mode
amplitude is `0.01000008645`; the reference value is `0.01000000000`. Its
maximum vertical kinetic-energy density is `9.99966748e-5`; the reference
value is `9.99988650e-5`. These small differences are fixture sampling and
float32 effects, not reference tolerances.
