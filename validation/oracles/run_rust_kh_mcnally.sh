#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle_dir="$repo_root/validation/oracles/kh_mcnally"

python3 "$repo_root/validation/oracles/fetch_assets.py" kh_mcnally
PYTHONPATH="$repo_root" \
  python3 -m unittest validation.tests.test_kh_mcnally_oracle

if [ -n "${GIZMO_KH_OUTPUT_DIR:-}" ]; then
  output_dir=$GIZMO_KH_OUTPUT_DIR
else
  oracle_tmp=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-rust-kh-mcnally.XXXXXX")
  trap 'rm -rf "$oracle_tmp"' EXIT HUP INT TERM
  cargo build \
    --release \
    --manifest-path "$repo_root/rust/Cargo.toml" \
    -p gizmo-cli
  cp "$oracle_dir/kh_mcnally_2d_ics.hdf5" \
    "$oracle_tmp/kh_mcnally_2d_ics.hdf5"
  (
    cd "$oracle_tmp"
    GIZMO_ACTIVE_TARGET_THREADS="${GIZMO_ACTIVE_TARGET_THREADS:-4}" \
      "$repo_root/rust/target/release/gizmo" \
      --config "$oracle_dir/public-config.sh" \
      "$oracle_dir/reference.params" \
      0
  )
  output_dir=$oracle_tmp/output
fi

PYTHONPATH="$repo_root" \
  python3 -m validation.oracles.check_kh_mcnally_trajectory \
  "$output_dir"
