# Copy/paste request for the Brio-Wu validation files

Could you publish the validation bundle used for the Brio-Wu result in
Hopkins & Raives (2015), especially the material behind `briowu_TP3.pdf`?
The public repository has the IC and example parameters, while the newer test
at commit `177ad43f57171c355fbcf163bd169eebe60afb06` checks only that the run
finishes with finite fields and a broad density range. Those assertions are
also satisfied by the unevolved IC, so they cannot verify the wave solution.

The minimum useful bundle would be:

1. The machine-readable x/profile tables plotted for `rho`, `P`, `u`, `vx`,
   `vy`, `Bx`, and `By` at `t=0.2`, including the conventional compound-wave
   reference branch, plus the corresponding Powell-only curves.
2. The script that generated or read the reference solution and the exact
   definitions of binning, norms, comparison region, wave-location metrics,
   and pass/fail tolerances.
3. The exact `Config.sh`, resolved parameter file, GIZMO commit, IC generator,
   compiler/precision settings, MPI rank count, and thread count used for the
   plotted result.
4. Reduced GIZMO outputs at `t=0`, `0.1`, and `0.2`, preferably preserving
   particle IDs and the primitive fields above, with SHA-256 checksums. Raw
   HDF5 snapshots are welcome but not necessary.
5. If available, a first-force/RHS table and the first drift/post-kick state.
   These are much better at locating an operator error than a terminal plot.
6. The intended treatment of the IC datasets
   `DivBcleaningFunctionPhi`, `DivergenceOfMagneticField`, and the misspelled
   `DivBcleaningFunctionGadPhi`: should they be consumed as restart state,
   ignored, or recomputed on initialization?

For unambiguous fixture identification, the currently hosted
`briowu_ics.hdf5` is 3,821,968 bytes with SHA-256
`300364564285270ed6e2f435786728683aeccd23c7c1eeb34d11eee9fade6f01`.
It contains 50,176 float32 gas particles in a `4 x 0.25` periodic domain
(`896 x 56`), with `gamma=2` and the discontinuity at `x=2`.

If the original analysis files are no longer available, a newly versioned
reference bundle from the current supported GIZMO commit—with explicit
acceptance tolerances—would still close the most important correctness gap.
