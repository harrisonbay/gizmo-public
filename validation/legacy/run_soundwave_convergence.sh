#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-soundwave-convergence.XXXXXX")
cache_dir="${LEGACY_UV_CACHE_DIR:-/tmp/gizmo-legacy-uv-cache}"

for particles in 64 128 256; do
  metrics="$work_dir/metrics-$particles.json"
  LEGACY_UV_CACHE_DIR="$cache_dir" \
    AMPLITUDE=1.0e-4 \
    PARTICLES="$particles" \
    METRICS_OUTPUT="$metrics" \
    "$repo_root/validation/legacy/run_soundwave.sh"
done

python3 "$repo_root/validation/legacy/check_convergence.py" \
  "$work_dir/metrics-64.json" \
  "$work_dir/metrics-128.json" \
  "$work_dir/metrics-256.json"

echo "legacy sound-wave convergence artifacts: $work_dir"
