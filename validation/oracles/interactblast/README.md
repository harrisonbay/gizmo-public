# Pinned public interacting-blast assets

This directory pins the public Woodward-Colella interacting-blast initial
condition and its high-resolution reference solution at `t=0.038`. Fetch and
verify the ignored binary/table assets with:

```sh
python3 validation/oracles/fetch_assets.py interactblast
```

The hosted HDF5 initial condition contains 512 unique gas particles, numbered
1 through 512. This disagrees with the public documentation's statement that
the test uses 400 particles. It also means the documented
`BOX_BND_PARTICLES` option, which freezes only particles with ID zero, is inert
for this hosted IC. The active wall behavior comes from `BOX_REFLECT_X` and
non-periodic x-distance calculations.

`legacy-config.sh` and `legacy.params` make the otherwise implicit public test
configuration explicit for a reproducible corrected-C oracle. They retain the
public fixed minimum and maximum timestep and do not add
`FORCE_EQUAL_TIMESTEPS`.

The corrected-C run completes 262,144 shared `2^42`-tick steps and writes 11
scheduled snapshots. `evolution-manifest.json` pins the complete schedule,
resolved parameters, initial/midpoint/terminal state tables, and
public-reference metrics. The
terminal volume-weighted L1 density error against the supplied 20,000-zone
solution is `0.0462596080`; the public materials do not publish an acceptance
tolerance.
