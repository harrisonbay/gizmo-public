#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle_dir="$repo_root/validation/oracles/dustybox"
oracle_tmp=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-rust-dustybox.XXXXXX")
trap 'rm -rf "$oracle_tmp"' EXIT HUP INT TERM

gzip -dc "$oracle_dir/evolution_t0.csv.gz" > "$oracle_tmp/evolution_t0.csv"
gzip -dc "$oracle_dir/evolution_t1.25.csv.gz" > "$oracle_tmp/evolution_t1.25.csv"
gzip -dc "$oracle_dir/evolution_t2.5.csv.gz" > "$oracle_tmp/evolution_t2.5.csv"

python3 "$repo_root/validation/oracles/fetch_assets.py" dustybox
PYTHONPATH="$repo_root" \
  python3 -m unittest validation.tests.test_dustybox_oracle
for time_and_table in \
  "0 evolution_t0.csv.gz" \
  "1.2500000000000009 evolution_t1.25.csv.gz" \
  "2.5 evolution_t2.5.csv.gz"
do
  set -- $time_and_table
  python3 "$repo_root/validation/oracles/compare_dustybox_analytic.py" \
    "$oracle_dir/$2" \
    --time "$1" \
    --check-manifest "$oracle_dir/evolution-manifest.json"
done
GIZMO_DUSTYBOX_IC="$oracle_dir/dustybox_ics.hdf5" \
GIZMO_DUSTYBOX_CONFIG="$oracle_dir/legacy-config.sh" \
GIZMO_DUSTYBOX_PARAMS="$oracle_dir/legacy.params" \
GIZMO_DUSTYBOX_C_T0="$oracle_tmp/evolution_t0.csv" \
GIZMO_DUSTYBOX_C_T1_25="$oracle_tmp/evolution_t1.25.csv" \
GIZMO_DUSTYBOX_C_T2_5="$oracle_tmp/evolution_t2.5.csv" \
  cargo test \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-cli \
  --test public_dustybox_cli \
  strict_dustybox_cli_matches_corrected_c_and_analytic_solution \
  -- \
  --ignored \
  --exact \
  --nocapture
