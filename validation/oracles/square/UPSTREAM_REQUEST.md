# Copy/paste request for the Square validation files

Could you publish the exact validation bundle used for GIZMO's public Square
advection result?

The minimum useful bundle would include:

1. The IC generator, version, random seed, exact `Config.sh`, resolved
   `parameters-usedvalues`, GIZMO commit, compiler/precision settings, MPI
   ranks, threads, and launch command.
2. The analyzer plus exact per-ID position, velocity, density, pressure,
   internal-energy, interface-shape, and conservation norms and tolerances.
3. ID-keyed reduced outputs at a non-return phase such as `t=0.5` and at
   `t=10`, plus a first-force/half-kick table and SHA-256 checksums.
4. Clarification of which setup is authoritative. The prose says `64^2`
   particles and boost `(142.3,-31.4)`, while the hosted fixture has `128^2`
   particles and boost `(1243,-358)` with correspondingly scaled density,
   pressure, and mass.
5. A non-aliased y-phase output. At the public `0.5` snapshot cadence,
   `v_y=-358` produces an integer wrapped displacement at every output, so the
   published trajectory cannot detect a missing y drift.

The currently hosted `square_ics.hdf5` is 727,440 bytes with SHA-256
`dd21e566f647405e62e59a0a5124a4081fcdd780f8489ae92447293f03bcc182`.
If the original files are unavailable, a newly versioned bundle from the
current supported commit with explicit tolerances would still close the gap.
