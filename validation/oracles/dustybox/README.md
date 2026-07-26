# Pinned public dusty-box oracle

This directory pins the public one-dimensional dusty-box initial condition,
the exact Epstein-drag solution documented by GIZMO, and three compact,
ID-sorted states from corrected C.

Fetch and verify the ignored public HDF5 asset with:

```sh
python3 validation/oracles/fetch_assets.py dustybox
```

The public `dustyboxwave.params` file is shared with dusty wave: it says to
select `dustybox_ics`, but defaults to `dustywave_ics`. The hosted dusty-box IC
contains 64 gas particles and 64 type-3 grains and, like the dusty-wave IC,
omits grain `SmoothingLength`. `legacy.params` makes the documented selection
and enables the documented `Softening_Type3 0.001` prompt. It also records
otherwise implicit legacy defaults needed for deterministic replay.

The corrected-C run uses one shared `2^45`-tick time bin for all particles,
completes 32,768 synchronized steps, and writes 251 scheduled snapshots. Its
terminal-output tick is snapped exactly to `TimeMax`, so `snapshot_250.hdf5`
has header time `2.5` rather than a floating-point value just below it.

For equal gas and dust density, the analytic Epstein solution is

```text
alpha = 15*pi/128
psi = exp(-2*t)/(1 + sqrt(1 + alpha))
v_dust - v_gas = 2*psi/(1 - alpha*psi^2)
v_gas = (1 - (v_dust-v_gas))/2
v_dust = (1 + (v_dust-v_gas))/2
```

`evolution_curve.csv.gz` checks all 251 corrected-C outputs. The maximum
velocity RMS error is `5.30014e-5` at snapshot 13 (`t=0.13`); by `t=1.25` it
is `3.81457e-6`, and at `t=2.5` it is `1.23367e-7`. Total momentum stays
within `1.41e-14` of its initial value. The public materials publish the
analytic formula but no acceptance threshold, so these measurements are
recorded rather than presented as an upstream pass/fail rule.

Run the complete oracle and ignored Rust end-to-end gate with:

```sh
validation/oracles/run_rust_dustybox.sh
```
