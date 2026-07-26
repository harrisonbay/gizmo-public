#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
fixture="$repo_root/validation/oracles/mhd_wave/mhd_wave_ics.hdf5"

python3 "$repo_root/validation/oracles/fetch_assets.py" mhd_wave
GIZMO_MHD_WAVE_IC="$fixture" \
  cargo test \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-io \
  reads_external_mhd_wave_fixture_when_configured \
  -- \
  --ignored \
  --nocapture
