#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle_dir="$repo_root/validation/oracles/shocktube"
oracle_tmp=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-rust-shocktube.XXXXXX")
trap 'rm -rf "$oracle_tmp"' EXIT HUP INT TERM

gzip -dc "$oracle_dir/evolution_t0.csv.gz" > "$oracle_tmp/evolution_t0.csv"
gzip -dc "$oracle_dir/evolution_step1_drift.csv.gz" \
  > "$oracle_tmp/evolution_step1_drift.csv"
gzip -dc "$oracle_dir/evolution_step1_postkick.csv.gz" \
  > "$oracle_tmp/evolution_step1_postkick.csv"
gzip -dc "$oracle_dir/evolution_t5.csv.gz" > "$oracle_tmp/evolution_t5.csv"

python3 "$repo_root/validation/oracles/fetch_assets.py" shocktube
GIZMO_SHOCKTUBE_IC="$oracle_dir/shocktube_ics_emass.hdf5" \
GIZMO_SHOCKTUBE_C_T0="$oracle_tmp/evolution_t0.csv" \
GIZMO_SHOCKTUBE_C_STEP1_DRIFT="$oracle_tmp/evolution_step1_drift.csv" \
GIZMO_SHOCKTUBE_C_STEP1_POSTKICK="$oracle_tmp/evolution_step1_postkick.csv" \
GIZMO_SHOCKTUBE_C_T5="$oracle_tmp/evolution_t5.csv" \
  cargo test \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-hydro \
  --test public_shocktube \
  rust_equal_mass_shocktube_matches_corrected_c \
  -- \
  --ignored \
  --exact \
  --nocapture
