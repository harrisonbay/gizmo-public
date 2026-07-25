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

`evolution-manifest.json` pins a fresh corrected-C run from commit
`f408a498ec42990fa6b3cc413ded2223f3d33dc2`. The run used one MPI rank,
completed all 65,536 shared `8192`-tick steps to `t=1.5`, and kept every gas
particle in timebin 44. ID-sorted semantic tables at `t=0`, `0.1`, and `1.5`
contain position, velocity, density, specific internal energy, smoothing
length, and mass. Raw snapshots are identified by checksum but omitted.

Regenerate a semantic table from a raw snapshot with:

```sh
uv run --with h5py --with numpy \
  python validation/oracles/reduce_soundwave_snapshot.py \
  snapshot_001.hdf5 evolution_t0.1.csv.gz
```

Normal GIZMO snapshots are written during the drift, before the endpoint force
and second kick, and `io.c` warns that their velocity is staggered. Therefore
`evolution_step1.csv.gz` is generated with the pinned
`post-kick-snapshot.patch`, which writes one additional snapshot immediately
after the first `do_second_halfstep_kick()`. The patch only observes state and
is checksum-pinned in the manifest. The ignored Rust integration test compares
this completed 8192-tick state against the original IC. Position, density,
internal energy, and the first half-kick are strict gates. The second half-kick
currently has a localized corrected-C/Rust force divergence recorded as
GZ-0009, so the test asserts that the discrepancy remains visible instead of
silently claiming endpoint velocity parity.
