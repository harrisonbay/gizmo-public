# Pinned public dusty-wave assets

This directory pins the public one-dimensional dusty-wave initial condition
and its tabulated gas and grain velocity solution at `t=1.2`. Fetch and verify
the ignored HDF5/table assets with:

```sh
python3 validation/oracles/fetch_assets.py dustywave
```

The public table contains 512 samples and states that only about four
significant figures are authoritative. Its velocity columns are normalized by
the initial perturbation amplitude, `1e-4`.

The hosted IC contains 64 gas particles and 64 type-3 grains, but no grain
smoothing length. Running the published parameter file literally therefore
gives every grain zero numerical size: the grain Courant timestep becomes zero
and the run advances on the minimum integer time bin. The public parameter
file comments out a suggested `Softening_Type3 0.001` prompt.
`legacy.params` enables that documented prompt explicitly. This choice is
necessary and reproducible, but is not uniquely specified by the public test.

With that prompt, the corrected-C profile completes 32,768 synchronized
`2^45`-tick steps. `evolution-manifest.json` pins its resolved parameters,
initial, reference-time, and scheduled terminal state tables. At `t=1.2`, the
corrected-C peak-normalized velocity RMS errors are `0.00075099` for grains and
`0.0119618` for gas. The public materials publish no acceptance tolerance.

The historical corrected-C run also exposed GZ-0012: accumulated output-time
roundoff caused a duplicate endpoint snapshot in a staggered, nonconservative
velocity phase. The Rust gate therefore compares the scheduled terminal drift,
not that duplicate final table, and requires exactly 251 outputs ending at
`t=2.5`.
