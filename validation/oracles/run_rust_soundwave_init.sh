#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
fixture="$repo_root/validation/oracles/soundwave/soundwave_ics.hdf5"

python3 "$repo_root/validation/oracles/fetch_assets.py" soundwave
GIZMO_SOUNDWAVE_IC="$fixture" \
  cargo test \
  --manifest-path "$repo_root/rust/Cargo.toml" \
  -p gizmo-hydro \
  --test public_soundwave \
  -- \
  --ignored \
  --nocapture

(
  cd "$repo_root/validation/oracles/soundwave"
  cargo run \
    --quiet \
    --manifest-path "$repo_root/rust/Cargo.toml" \
    -p gizmo-cli \
    -- \
    --initialize-only \
    --config legacy-config.sh \
    legacy.params
)
