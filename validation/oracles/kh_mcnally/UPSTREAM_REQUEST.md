# Copy/paste request for the McNally KH validation files

Could you publish the validation bundle used to verify GIZMO's public
`kh_mcnally_2d` result? The repository provides an initial condition and
example parameters, and the benchmark authors provide independent reference
curves through `t=1.5`, but the GIZMO-specific analyzer, pass/fail criterion,
and reproducible reference trajectory are not in the public tree.

The minimum useful bundle would be:

1. The exact `Config.sh`, resolved `parameters-usedvalues`, GIZMO commit,
   compiler and precision settings, MPI rank count, thread count, and launch
   command.
2. The analysis script and exact norms, sampling times, and absolute/relative
   tolerances used for the mode amplitude, vertical kinetic-energy density,
   conserved quantities, and any other asserted result.
3. Particle-ID-keyed reduced outputs at `t=0,0.1,...,1.5`, including
   coordinates, velocity, density, internal energy, smoothing length, mass,
   and SHA-256 checksums. Raw HDF5 snapshots are optional.
4. A first-force/RHS table, first half-kick/drift state, and timestep-bin
   assignments, if available. These would localize kernel, gradient, face,
   kick/drift, and timestep discrepancies without requiring raw snapshots.
5. The initial-condition generator, version, random seed, and generation
   command. Please also state whether the intended GIZMO phase is the hosted
   fixture's half-box translation of the phase written in McNally et al.
6. Clarification of whether GIZMO's validation ends at the published reference
   endpoint `t=1.5` or runs the example parameter file through `TimeMax 10`.
   The benchmark authors' machine-readable curves stop at `t=1.5`, so a
   quantitative gate after that point needs a separate reference.

For fixture identification, the currently hosted
`kh_mcnally_2d_ics.hdf5` is 2,948,736 bytes with SHA-256
`55d2a579bdcc3e85abf2baa624f45ffdbd4a03f36316084544a698e85c8f7777`.
It contains 66,868 float32 gas particles with IDs `1..66868`.

If the files used for the original result are unavailable, a newly versioned
bundle from the current supported commit, with explicit tolerances and a
checksum manifest, would still close the main reproducibility gap.
