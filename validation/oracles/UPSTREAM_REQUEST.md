# Request for the missing public validation suite

The repository contains 29 example parameter files but not enough material to
reproduce their pass/fail results. Please publish, for every supported example:

1. The exact `Config.sh`, initial-condition file or deterministic generator,
   generator version, random seed, and input checksum.
2. The runtime parameter file after defaults are resolved (the generated
   `*-usedvalues` file is sufficient).
3. The analysis script and quantitative acceptance criteria: conserved
   quantities, norms, convergence order, comparison times, phase conventions,
   and absolute/relative tolerances.
4. Small canonical output snapshots or reduced reference tables, with
   checksums, plus the GIZMO commit, compiler, precision, MPI rank count, and
   thread count that produced them.
5. The `scripts/pipelines/run_isodisk_testprob.sh` and
   `scripts/pipelines/compare_isodisk_output.sh` files referenced by
   `scripts/pipelines/bitbucket-pipelines.yml`, plus the isodisk reference
   output they compare against. Those scripts are absent from the public tree.

For `soundwave` specifically, please clarify which definition is authoritative.
The prose says mean pressure `3/5` and sound speed `1`, while the currently
hosted `soundwave_ics.hdf5` has mean pressure `4/15`, sound speed `2/3`, and a
`1.5` box-crossing time. Its SHA-256 is
`d13c5f1bb916490037c96f2d3e4fa37b1121e09b352bd6bc15de80bee9f35b35`.

For `shocktube`, please include the actual `Config.sh` and resolved parameters
used to produce the published comparison. The checked-in parameter file omits
tags required by a strict developer-mode run. Please also provide a known-good
reference run for `shocktube_ics_diffmass.hdf5`. The public source required a
correction to the force-tree node-capacity guard before that hosted IC could
complete under the otherwise equivalent one-rank profile.

For `interactblast`, please clarify the particle count and acceptance
criterion. The prose says 400 evenly spaced, equal-mass particles; the hosted
`interactblast_ics.hdf5` contains 512 particles and has adjusted edge masses
and smoothing lengths. Please provide the exact configuration and compiler
profile used for the published result, plus the quantitative norm and
tolerance applied to `interactblast_exact.txt`. A known-good reduced snapshot
at `t=0.038` would also disambiguate reflective-wall and output-phase
semantics.

For `dustywave`, please provide the exact positive grain numerical-size setup
used for the published curve. The hosted IC has no type-3
`SmoothingLength`, while `dustyboxwave.params` leaves
`Softening_Type3 0.001` commented out. Running those public files literally
makes the grain Courant length and desired timestep zero; after 2,532
synchronization points the run has advanced only
`1.098079960293319e-14`. Enabling that commented prompt produces a practical
32,768-step run, but it is not clear whether this was the authoritative setup.
Please also provide the generator/analyzer for `dustwave_exact.txt` and the
intended gas/dust velocity norms and tolerances.

For `dustybox`, please provide the initial-condition generator and the exact
acceptance norm used for the published analytic comparison. Please also clarify
the intended terminal-output phase. With `TimeBetSnapshot 0.01` and
`TimeMax 2.5`, repeated floating-point addition places the scheduled terminal
output just below the final integer tick; the uncorrected code then writes a
second endpoint file after gas backreaction but before the matching grain
half-kick. Those two files differ in total gas-plus-dust momentum by
`2.36915e-7` for the hosted IC.

For `mhd_wave`, please publish the initial-condition generator, analytic
analyzer, exact configuration, and acceptance norm. The hosted fixture uses a
normalized fast-mode eigenvector whose density amplitude is
`1e-6/sqrt(5)`, while Hopkins & Raives (2015) describes
`delta rho/rho=1e-6`. Please clarify which convention is authoritative. The
public `TimeMax 0.5` is exactly one wavelength at fast speed 2, so comparing
only the initial and final states cannot distinguish a real solver from a
no-op; intermediate-time phase and polarization criteria are essential.
Please also clarify whether the intended norm is the paper's particle L1 norm:
the current public C trajectory has small L1 errors but sparse elementwise
outliers large enough to fail a `1e-7` max/allclose-style gate.

For `square`, please clarify which problem is authoritative. The prose
describes `64^2` particles and boost `(142.3,-31.4)`, while the hosted IC has
`128^2` particles and boost `(1243,-358)` with differently normalized density
and pressure. Please also provide a non-aliased y-phase output: the hosted
`v_y=-358` wraps by an integer at every public `0.5` output interval. A
copy/paste request naming the exact artifacts and fixture checksum is in
`square/UPSTREAM_REQUEST.md`.

For `sedov`, please clarify the factor-`10^10` mass-unit discrepancy, the
30 Myr analytic-table time versus `TimeMax 0.03`, the 1 K versus 10 K ambient
temperature, and the documented 64-particle top hat versus the hosted tapered
278-particle thermal excess. Please provide the exact analyzer, tolerances,
build provenance, and reduced radial plus angular-symmetry trajectory. A
copy/paste request with the hosted checksums is in
`sedov/UPSTREAM_REQUEST.md`.

For `gresho`, please replace or explain `gresho_ics_grid.hdf5`. Its internal
energy encodes the documented equilibrium for gamma `5/3`, while the supplied
configuration and parameter-file comments require gamma `1.4`. The ring IC is
consistent with gamma `1.4`. Please also provide the exact `Config.sh`,
resolved `parameters-usedvalues`, analyzer, norm, tolerances, and reduced
outputs at `t=0,0.5,...,3` used for the published ring result. In the literal
non-`DEVELOPER_MODE` public profile, `ErrTolIntAccuracy`, `CourantFac`, and
`MaxRMSDisplacementFac` in `gresho.params` are not consumed, so the intended
effective defaults should be stated explicitly.

For `kh_mcnally_2d`, please provide the GIZMO-specific analyzer, acceptance
thresholds, reproducible reduced trajectory, exact build provenance, and IC
generator. The benchmark authors' independent curves stop at `t=1.5`, while
the public GIZMO parameter file runs to `t=10`. A copy/paste request naming the
exact artifacts and hosted fixture checksum is in
`kh_mcnally/UPSTREAM_REQUEST.md`.

For `brio-wu` and `toth`, please provide the machine-readable exact/reference
curves used in the published plots; no corresponding tables are present in the
hosted directory. The newer Brio-Wu harness checks only terminal time, finite
fields, positive density, and a broad density range already satisfied by the
unevolved IC, so it is not a correctness oracle. A copy/paste request naming
the exact Brio-Wu artifacts and fixture checksum is in
`briowu/UPSTREAM_REQUEST.md`. For `zeldovich`, please document the headerless
reference table's column transforms and explain why the hosted IC contains
262,208 particles rather than `64^3`.

The most useful delivery is a versioned archive or repository containing the
generators, configs, analyzers, thresholds, and a checksum manifest. Large raw
snapshots are optional if compact reference tables preserve the scientific
oracles.
