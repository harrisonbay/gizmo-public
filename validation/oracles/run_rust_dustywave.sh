#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle_dir="$repo_root/validation/oracles/dustywave"
oracle_tmp=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-rust-dustywave.XXXXXX")
trap 'rm -rf "$oracle_tmp"' EXIT HUP INT TERM

gzip -dc "$oracle_dir/evolution_t0.csv.gz" > "$oracle_tmp/evolution_t0.csv"
gzip -dc "$oracle_dir/evolution_t1.2.csv.gz" > "$oracle_tmp/evolution_t1.2.csv"
gzip -dc "$oracle_dir/evolution_t2.5.csv.gz" > "$oracle_tmp/evolution_t2.5.csv"

python3 "$repo_root/validation/oracles/fetch_assets.py" dustywave
PYTHONPATH="$repo_root" \
  python3 -m unittest validation.tests.test_dustywave_oracle
python3 "$repo_root/validation/oracles/compare_dustywave_reference.py" \
  "$oracle_dir/evolution_t1.2.csv.gz" \
  "$oracle_dir/dustwave_exact.txt" \
  --check-manifest "$oracle_dir/evolution-manifest.json"
GIZMO_DUSTYWAVE_IC="$oracle_dir/dustywave_ics.hdf5" \
GIZMO_DUSTYWAVE_CONFIG="$oracle_dir/legacy-config.sh" \
GIZMO_DUSTYWAVE_PARAMS="$oracle_dir/legacy.params" \
GIZMO_DUSTYWAVE_C_T0="$oracle_tmp/evolution_t0.csv" \
GIZMO_DUSTYWAVE_C_T1_2="$oracle_tmp/evolution_t1.2.csv" \
GIZMO_DUSTYWAVE_C_T2_5="$oracle_tmp/evolution_t2.5.csv" \
GIZMO_DUSTYWAVE_EXACT="$oracle_dir/dustwave_exact.txt" \
  cargo test \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-cli \
  --test public_dustywave_cli \
  strict_dustywave_cli_matches_corrected_c_and_public_reference \
  -- \
  --ignored \
  --exact \
  --nocapture
