#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
fixture="$repo_root/validation/oracles/soundwave/soundwave_ics.hdf5"
oracle_dir="$repo_root/validation/oracles/soundwave"
oracle_tmp=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-rust-oracle.XXXXXX")
trap 'rm -rf "$oracle_tmp"' EXIT HUP INT TERM

gzip -dc "$oracle_dir/evolution_t0.csv.gz" > "$oracle_tmp/evolution_t0.csv"
gzip -dc "$oracle_dir/evolution_step1.csv.gz" > "$oracle_tmp/evolution_step1.csv"

python3 "$repo_root/validation/oracles/fetch_assets.py" soundwave
GIZMO_SOUNDWAVE_IC="$fixture" \
GIZMO_SOUNDWAVE_C_T0="$oracle_tmp/evolution_t0.csv" \
GIZMO_SOUNDWAVE_C_STEP1="$oracle_tmp/evolution_step1.csv" \
  cargo test \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-hydro \
  --test public_soundwave \
  -- \
  --ignored \
  --nocapture

(
  cd "$repo_root/validation/oracles/soundwave"
  cargo run \
    --quiet \
    --manifest-path "$repo_root/rust/Cargo.toml" \
    -p gizmo-cli \
    -- \
    --initialize-only \
    --config legacy-config.sh \
    legacy.params
)
