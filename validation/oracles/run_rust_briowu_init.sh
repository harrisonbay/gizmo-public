#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
fixture="$repo_root/validation/oracles/briowu/briowu_ics.hdf5"

python3 "$repo_root/validation/oracles/fetch_assets.py" briowu

GIZMO_BRIOWU_FIXTURE="$fixture" \
  cargo test \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-cli \
  --test public_briowu_cli \
  -- \
  --ignored \
  --nocapture

GIZMO_BRIOWU_FIXTURE="$fixture" \
  cargo test \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-hydro \
  --test public_briowu_geometry \
  -- \
  --ignored \
  --nocapture
