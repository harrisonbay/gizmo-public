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
gzip -dc "$oracle_dir/diffmass_t0.csv.gz" > "$oracle_tmp/diffmass_t0.csv"
gzip -dc "$oracle_dir/diffmass_t5.csv.gz" > "$oracle_tmp/diffmass_t5.csv"

python3 "$repo_root/validation/oracles/fetch_assets.py" shocktube
PYTHONPATH="$repo_root" \
  python3 -m unittest validation.tests.test_shocktube_oracle
python3 "$repo_root/validation/oracles/compare_shocktube_reference.py" \
  "$oracle_dir/evolution_t5.csv.gz" \
  "$oracle_dir/shocktube_exact.txt" \
  --check-manifest "$oracle_dir/evolution-manifest.json"
python3 "$repo_root/validation/oracles/compare_shocktube_reference.py" \
  "$oracle_dir/diffmass_t5.csv.gz" \
  "$oracle_dir/shocktube_exact.txt" \
  --check-manifest "$oracle_dir/diffmass-evolution-manifest.json"
GIZMO_SHOCKTUBE_IC="$oracle_dir/shocktube_ics_emass.hdf5" \
GIZMO_SHOCKTUBE_C_T0="$oracle_tmp/evolution_t0.csv" \
GIZMO_SHOCKTUBE_C_STEP1_DRIFT="$oracle_tmp/evolution_step1_drift.csv" \
GIZMO_SHOCKTUBE_C_STEP1_POSTKICK="$oracle_tmp/evolution_step1_postkick.csv" \
GIZMO_SHOCKTUBE_C_T5="$oracle_tmp/evolution_t5.csv" \
GIZMO_SHOCKTUBE_EXACT="$oracle_dir/shocktube_exact.txt" \
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
GIZMO_SHOCKTUBE_DIFFMASS_IC="$oracle_dir/shocktube_ics_diffmass.hdf5" \
GIZMO_SHOCKTUBE_C_DIFFMASS_T0="$oracle_tmp/diffmass_t0.csv" \
GIZMO_SHOCKTUBE_C_DIFFMASS_T5="$oracle_tmp/diffmass_t5.csv" \
GIZMO_SHOCKTUBE_EXACT="$oracle_dir/shocktube_exact.txt" \
  cargo test \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-hydro \
  --test public_shocktube \
  rust_differential_mass_shocktube_matches_corrected_c \
  -- \
  --ignored \
  --exact \
  --nocapture
GIZMO_SHOCKTUBE_IC="$oracle_dir/shocktube_ics_emass.hdf5" \
GIZMO_SHOCKTUBE_CONFIG="$repo_root/validation/legacy/shocktube.Config.sh" \
GIZMO_SHOCKTUBE_PARAMS="$repo_root/validation/legacy/shocktube.params" \
GIZMO_SHOCKTUBE_C_T0="$oracle_tmp/evolution_t0.csv" \
GIZMO_SHOCKTUBE_C_T5="$oracle_tmp/evolution_t5.csv" \
  cargo test \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-cli \
  --test public_shocktube_cli \
  strict_equal_mass_shocktube_cli_matches_corrected_c \
  -- \
  --ignored \
  --exact \
  --nocapture
GIZMO_SHOCKTUBE_DIFFMASS_IC="$oracle_dir/shocktube_ics_diffmass.hdf5" \
GIZMO_SHOCKTUBE_CONFIG="$repo_root/validation/legacy/shocktube.Config.sh" \
GIZMO_SHOCKTUBE_DIFFMASS_PARAMS="$repo_root/validation/legacy/shocktube-diffmass.params" \
GIZMO_SHOCKTUBE_C_DIFFMASS_T0="$oracle_tmp/diffmass_t0.csv" \
GIZMO_SHOCKTUBE_C_DIFFMASS_T5="$oracle_tmp/diffmass_t5.csv" \
  cargo test \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-cli \
  --test public_shocktube_cli \
  strict_differential_mass_shocktube_cli_matches_corrected_c \
  -- \
  --ignored \
  --exact \
  --nocapture
