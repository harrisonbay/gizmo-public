#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle_dir="$repo_root/validation/oracles/interactblast"
oracle_tmp=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-rust-interactblast.XXXXXX")
trap 'rm -rf "$oracle_tmp"' EXIT HUP INT TERM

gzip -dc "$oracle_dir/evolution_t0.csv.gz" > "$oracle_tmp/evolution_t0.csv"
gzip -dc "$oracle_dir/evolution_tmid.csv.gz" > "$oracle_tmp/evolution_tmid.csv"
gzip -dc "$oracle_dir/evolution_tfinal.csv.gz" > "$oracle_tmp/evolution_tfinal.csv"

python3 "$repo_root/validation/oracles/fetch_assets.py" interactblast
PYTHONPATH="$repo_root" \
  python3 -m unittest validation.tests.test_interactblast_oracle
python3 "$repo_root/validation/oracles/compare_interactblast_reference.py" \
  "$oracle_dir/evolution_tfinal.csv.gz" \
  "$oracle_dir/interactblast_exact.txt" \
  --check-manifest "$oracle_dir/evolution-manifest.json"
GIZMO_INTERACTBLAST_IC="$oracle_dir/interactblast_ics.hdf5" \
GIZMO_INTERACTBLAST_CONFIG="$oracle_dir/legacy-config.sh" \
GIZMO_INTERACTBLAST_PARAMS="$oracle_dir/legacy.params" \
GIZMO_INTERACTBLAST_C_T0="$oracle_tmp/evolution_t0.csv" \
GIZMO_INTERACTBLAST_C_TMID="$oracle_tmp/evolution_tmid.csv" \
GIZMO_INTERACTBLAST_C_TFINAL="$oracle_tmp/evolution_tfinal.csv" \
GIZMO_INTERACTBLAST_EXACT="$oracle_dir/interactblast_exact.txt" \
  cargo test \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-cli \
  --test public_shocktube_cli \
  strict_interacting_blast_cli_matches_corrected_c \
  -- \
  --ignored \
  --exact \
  --nocapture
