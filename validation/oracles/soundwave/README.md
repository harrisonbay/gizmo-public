# Pinned public sound-wave oracle

`assets.json` locks the public initial-condition file by URL, byte size, and
SHA-256. Fetch it with:

```sh
python3 validation/oracles/fetch_assets.py soundwave
```

The upstream prose and the actual file disagree: the file has a pressure mean
of `0.2666666667`, sound speed `2/3`, and perturbation amplitude near `1e-6`.
Those values make `TimeMax=1.5` one box crossing. Comparisons therefore treat
the pinned bytes and analytic invariants as authoritative and record the prose
discrepancy instead of silently changing either one.

Downloaded binary assets are intentionally ignored by Git. The manifest is the
reproducible identity.

Run the Rust HDF5, density, and adaptive-smoothing-length parity gate with:

```sh
validation/oracles/run_rust_soundwave_init.sh
```
