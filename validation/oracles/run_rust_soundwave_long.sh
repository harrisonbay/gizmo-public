#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle_dir="$repo_root/validation/oracles/soundwave"
oracle_tmp=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-rust-long-oracle.XXXXXX")
trap 'rm -rf "$oracle_tmp"' EXIT HUP INT TERM

for phase in t0 t0.1 t1.5
do
  gzip -dc "$oracle_dir/evolution_${phase}.csv.gz" \
    > "$oracle_tmp/evolution_${phase}.csv"
done

python3 "$repo_root/validation/oracles/fetch_assets.py" soundwave
GIZMO_SOUNDWAVE_IC="$oracle_dir/soundwave_ics.hdf5" \
GIZMO_SOUNDWAVE_C_T0="$oracle_tmp/evolution_t0.csv" \
GIZMO_SOUNDWAVE_C_T01="$oracle_tmp/evolution_t0.1.csv" \
GIZMO_SOUNDWAVE_C_TMAX="$oracle_tmp/evolution_t1.5.csv" \
  cargo test \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-hydro \
  --test public_soundwave \
  rust_long_evolution_matches_corrected_c_snapshots \
  -- \
  --ignored \
  --exact \
  --nocapture
