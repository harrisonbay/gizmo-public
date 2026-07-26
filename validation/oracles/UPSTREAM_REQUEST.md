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

The most useful delivery is a versioned archive or repository containing the
generators, configs, analyzers, thresholds, and a checksum manifest. Large raw
snapshots are optional if compact reference tables preserve the scientific
oracles.
