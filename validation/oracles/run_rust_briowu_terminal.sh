#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle_dir="$repo_root/validation/oracles/briowu"
fixture="$oracle_dir/briowu_ics.hdf5"

python3 "$repo_root/validation/oracles/fetch_assets.py" briowu
cargo build \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-cli \
  --bin gizmo

if [ "${GIZMO_BRIOWU_RUN_DIR:-}" ]; then
  run_dir=$GIZMO_BRIOWU_RUN_DIR
  mkdir -p "$run_dir"
  if [ -e "$run_dir/output" ] || [ -e "$run_dir/briowu_ics.hdf5" ]; then
    echo "refusing to overwrite existing Brio-Wu run directory: $run_dir" >&2
    exit 1
  fi
else
  run_dir=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-briowu-terminal.XXXXXX")
fi
cp "$fixture" "$run_dir/briowu_ics.hdf5"

if [ "${GIZMO_ACTIVE_TARGET_THREADS:-}" ]; then
  force_threads=$GIZMO_ACTIVE_TARGET_THREADS
else
  force_threads=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi

echo "Brio-Wu terminal run directory: $run_dir"
echo "Brio-Wu active-target workers: $force_threads"
(
  cd "$run_dir"
  GIZMO_ACTIVE_TARGET_THREADS=$force_threads \
    "$repo_root/rust/target/release/gizmo" \
    --config "$oracle_dir/frontier-config.sh" \
    "$oracle_dir/frontier.params" \
    0
)

"$repo_root/validation/oracles/validate_rust_briowu_outputs.sh" "$run_dir"

echo "Brio-Wu terminal oracle passed; artifacts retained at $run_dir"
