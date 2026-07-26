#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle_dir="$repo_root/validation/oracles/gresho"
oracle_tmp=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-rust-gresho.XXXXXX")
trap 'rm -rf "$oracle_tmp"' EXIT HUP INT TERM

python3 "$repo_root/validation/oracles/fetch_assets.py" gresho
PYTHONPATH="$repo_root" \
  python3 -m unittest validation.tests.test_gresho_oracle

cargo build \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-cli
cp "$oracle_dir/gresho_ics.hdf5" "$oracle_tmp/gresho_ics.hdf5"
(
  cd "$oracle_tmp"
  GIZMO_ACTIVE_TARGET_THREADS="${GIZMO_ACTIVE_TARGET_THREADS:-4}" \
    "$repo_root/rust/target/release/gizmo" \
    --config "$oracle_dir/public-config.sh" \
    "$oracle_dir/public.params" \
    0
)

PYTHONPATH="$repo_root" \
  python3 -m validation.oracles.check_gresho_trajectory \
  "$oracle_tmp/output"
