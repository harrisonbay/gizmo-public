# Copy/paste request for the Gresho validation files

Could you publish the validation bundle used for the public Gresho-vortex
result? The repository provides an IC and example parameters, but not the
analyzer, quantitative pass/fail criterion, or reference trajectory needed to
verify the result.

The minimum useful bundle would be:

1. The exact `Config.sh`, resolved `parameters-usedvalues`, GIZMO commit,
   compiler and precision settings, MPI rank count, and thread count.
2. The analysis script plus the exact density, pressure, radial-velocity,
   tangential-velocity, and full-vector norms and tolerances.
3. Particle-ID-keyed reduced outputs at `t=0,0.5,...,3`, with coordinates,
   velocity, density, internal energy, smoothing length, mass, and SHA-256
   checksums. Raw HDF5 is optional.
4. A first-force/RHS table and first half-kick/drift state, if available. This
   disambiguates the staggered `t=0` output and localizes operator errors.
5. The intended effective values of `ErrTolIntAccuracy`, `CourantFac`, and
   `MaxRMSDisplacementFac`. The literal public non-`DEVELOPER_MODE` build does
   not consume those prompts from `gresho.params`.
6. The generator and intended gamma for `gresho_ics_grid.hdf5`. Its internal
   energy encodes the documented equilibrium for gamma `5/3`, while the public
   configuration requires gamma `1.4`. The ring IC is consistent with `1.4`.

For fixture identification, the currently hosted ring
`gresho_ics.hdf5` is 186,624 bytes with SHA-256
`9109bbce2a58eef3cb13d665582fa636e667a1098ae491b80f8f39c2fae1a43c`.
It contains 4,092 float32 gas particles with IDs `0..4091` in the periodic
unit box.

If the original publication files are unavailable, a newly versioned bundle
from the current supported commit—with explicit tolerances—would still close
the key correctness gap.
