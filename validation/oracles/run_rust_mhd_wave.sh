#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

python3 "$repo_root/validation/oracles/fetch_assets.py" mhd_wave
cargo test \
  --release \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-cli \
  --test public_mhd_wave_cli \
  -- \
  --ignored \
  --nocapture
