#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 RUN_DIRECTORY" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
oracle_dir="$repo_root/validation/oracles/briowu"
run_dir=$1

for index in 000 001 002; do
  snapshot="$run_dir/output/snapshot_$index.hdf5"
  if [ ! -f "$snapshot" ]; then
    echo "missing terminal-run snapshot: $snapshot" >&2
    exit 1
  fi
done
if [ -e "$run_dir/output/snapshot_003.hdf5" ]; then
  echo "unexpected fourth Brio-Wu snapshot" >&2
  exit 1
fi

uv run --with h5py --with numpy python \
  "$repo_root/validation/oracles/check_briowu_snapshots.py" \
  "$run_dir/output/snapshot_000.hdf5" \
  "$run_dir/output/snapshot_001.hdf5" \
  "$run_dir/output/snapshot_002.hdf5" \
  --maximum-relative-mass-change 0 \
  --maximum-absolute-momentum 2e-8 \
  --maximum-relative-diagnostic-energy-change 0.005

for observation in 0 0.1 0.2; do
  case "$observation" in
    0) index=000 ;;
    0.1) index=001 ;;
    0.2) index=002 ;;
  esac
  uv run --with h5py --with numpy python \
    "$repo_root/validation/oracles/reduce_briowu_snapshot.py" \
    "$run_dir/output/snapshot_$index.hdf5" \
    "$run_dir/evolution_t$observation.csv.gz"
done

python3 "$repo_root/validation/oracles/compare_briowu_reduced.py" \
  "$run_dir/evolution_t0.csv.gz" \
  "$oracle_dir/evolution_t0.csv.gz" \
  --maximum-normalized-l1 vx=0.00005 \
  --maximum-normalized-l1 vy=0.00005 \
  --maximum-normalized-l1 bx=0.00005 \
  --maximum-normalized-l1 by=0.00005 \
  --maximum-normalized-l1 rho=0.00005 \
  --maximum-normalized-l1 u=0.00005 \
  --maximum-normalized-l1 pressure=0.00005 \
  --maximum-mean-relative-error hsml=0.001

for observation in 0.1 0.2; do
  python3 "$repo_root/validation/oracles/compare_briowu_reduced.py" \
    "$run_dir/evolution_t$observation.csv.gz" \
    "$oracle_dir/evolution_t$observation.csv.gz" \
    --maximum-normalized-l1 vx=0.05 \
    --maximum-normalized-l1 vy=0.05 \
    --maximum-normalized-l1 bx=0.05 \
    --maximum-normalized-l1 by=0.05 \
    --maximum-normalized-l1 rho=0.05 \
    --maximum-normalized-l1 u=0.05 \
    --maximum-normalized-l1 pressure=0.05 \
    --maximum-mean-relative-error hsml=0.05
done

python3 "$repo_root/validation/oracles/compare_briowu_profile.py" \
  "$run_dir/evolution_t0.2.csv.gz" \
  --maximum-normalized-l1 vx=0.05 \
  --maximum-normalized-l1 vy=0.05 \
  --maximum-normalized-l1 bx=0.05 \
  --maximum-normalized-l1 by=0.05 \
  --maximum-normalized-l1 rho=0.05 \
  --maximum-normalized-l1 u=0.05 \
  --maximum-normalized-l1 pressure=0.05

echo "Brio-Wu terminal outputs passed: $run_dir"
