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

The most useful delivery is a versioned archive or repository containing the
generators, configs, analyzers, thresholds, and a checksum manifest. Large raw
snapshots are optional if compact reference tables preserve the scientific
oracles.
