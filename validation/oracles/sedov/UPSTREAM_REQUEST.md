# Copy/paste request for the Sedov validation files

Could you publish the exact validation bundle used for GIZMO's public Sedov
result: IC/table generators, `Config.sh`, resolved `parameters-usedvalues`,
source commit, compiler/precision, MPI/thread counts, reduced snapshots,
analyzer, norms, and quantitative tolerances?

Please also clarify four correctness-relevant inconsistencies in the public
materials:

1. `sedov.params` has `UnitMass_in_g 1.989e43`, but the documented ambient
   density and `6.78e46 J` explosion agree with the hosted IC only for
   `1.989e33 g`, a factor of `10^10` smaller.
2. `sedov_exact.txt` states that it is evaluated at 30 Myr, while
   `TimeMax 0.03` is about 29.34 Myr using the public year constant. A direct
   terminal comparison must transform the self-similar table; the shock radius
   changes from about `1.39500` to `1.38266`.
3. The table's ambient temperature is 1 K, while the documentation, IC, and
   `MinGasTemp` specify 10 K.
4. The prose describes a 64-particle top-hat injection. The hosted IC has 64
   extremely hot particles but a tapered thermal excess extending across 278
   particles.

For identification, the hosted artifacts are:

- `sedov_ics.hdf5`: 92,281,232 bytes, SHA-256
  `f2aedf59b8d44eb652728f7d2527751031d1778d29baf7861433070d646fdb03`;
- `sedov_exact.txt`: 195,142 bytes, SHA-256
  `3b9063434370fbc946aa1c1e211f7f2515a0bb96cf868c79fb2a24845a439cfb`;
- `sedov.params`: 1,063 bytes, SHA-256
  `2d828a0328cdaf2450116f3380957c7bacd6c9b8c07270798044a547439d99ea`.

A useful reduced trajectory should include radial density/velocity/thermal
profiles, absolute shock radius, total mass/energy/three-vector momentum, and
angular-sector or quadrupole symmetry diagnostics. Radial averaging alone can
hide Cartesian lobes or a carbuncle instability.
