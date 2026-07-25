#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/gizmo-soundwave.XXXXXX")
source_dir="$work_dir/source"
run_dir="$work_dir/run"
export UV_CACHE_DIR="${LEGACY_UV_CACHE_DIR:-$work_dir/uv-cache}"
config_profile="${LEGACY_CONFIG_PROFILE:-validation/legacy/soundwave.Config.sh}"

mkdir -p "$source_dir" "$run_dir/output"
rsync -a \
  --exclude .git \
  --exclude target \
  --exclude __pycache__ \
  --exclude '*.o' \
  --exclude GIZMO_config.h \
  --exclude compile_time_info.c \
  --exclude GIZMO \
  --exclude GIZMO-soundwave \
  "$repo_root/" "$source_dir/"

for tool in brew make mpicc mpirun uv; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "missing required command: $tool" >&2
    exit 2
  }
done

gsl_prefix=$(brew --prefix gsl)
hdf5_prefix=$(brew --prefix hdf5)

make -C "$source_dir" \
  CONFIG="$config_profile" \
  EXEC=GIZMO-soundwave \
  'SYSTYPE="MacBookCellar"' \
  GSL_INCL="-I$gsl_prefix/include" \
  GSL_LIBS="-L$gsl_prefix/lib" \
  HDF5INCL="-I$hdf5_prefix/include -DH5_USE_16_API" \
  HDF5LIB="-L$hdf5_prefix/lib -lhdf5 -lz" \
  -j"${JOBS:-4}"

cp "$source_dir/validation/legacy/soundwave.params" "$run_dir/soundwave.params"
uv run --with h5py --with numpy \
  python "$source_dir/validation/legacy/generate_soundwave_ic.py" \
  "$run_dir/soundwave_ics.hdf5" \
  --count "${PARTICLES:-128}" \
  --amplitude "${AMPLITUDE:-1.0e-2}"

(
  cd "$run_dir"
  mpirun -np "${MPI_RANKS:-1}" "$source_dir/GIZMO-soundwave" soundwave.params \
    >gizmo.stdout 2>gizmo.stderr
)

latest_snapshot=$(find "$run_dir/output" -name 'snapshot_*.hdf5' -type f | sort | tail -n 1)
initial_snapshot=$(find "$run_dir/output" -name 'snapshot_*.hdf5' -type f | sort | head -n 1)
test -n "$latest_snapshot" || {
  echo "GIZMO emitted no HDF5 snapshot" >&2
  exit 1
}
set -- \
  "$latest_snapshot" \
  --reference "$initial_snapshot" \
  --amplitude "${AMPLITUDE:-1.0e-2}"
if test -n "${METRICS_OUTPUT:-}"; then
  set -- "$@" --json-output "$METRICS_OUTPUT"
fi
uv run --with h5py --with numpy \
  python "$source_dir/validation/legacy/analyze_soundwave.py" "$@"

echo "legacy sound-wave artifacts: $work_dir"
