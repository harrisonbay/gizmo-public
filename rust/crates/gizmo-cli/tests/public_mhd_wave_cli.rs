use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use gizmo_io::{MhdWaveSnapshot, read_mhd_wave};

const BASE: [f64; 8] = [1.0, 0.0, 0.0, 0.0, 0.9, 1.0, std::f64::consts::SQRT_2, 0.5];
const EIGENVECTOR: [f64; 8] = [
    0.447_213_595_644e-6,
    -0.894_427_191_000e-6,
    0.421_637_021_356e-6,
    0.149_071_198_500e-6,
    0.268_324_803_201e-6,
    0.0,
    0.843_274_042_711e-6,
    0.298_142_396_999e-6,
];

#[test]
#[ignore = "runs the pinned 2048-particle MHD wave through t=0.5"]
fn strict_mhd_wave_cli_matches_intermediate_analytic_and_corrected_c_oracles() {
    let oracle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../validation/oracles/mhd_wave");
    let temporary = TemporaryDirectory::new();
    std::fs::copy(
        oracle.join("mhd_wave_ics.hdf5"),
        temporary.path.join("mhd_wave_ics.hdf5"),
    )
    .expect("MHD IC must copy into the isolated CLI directory");
    let result = Command::new(env!("CARGO_BIN_EXE_gizmo"))
        .current_dir(&temporary.path)
        .arg("--config")
        .arg(oracle.join("frontier-config.sh"))
        .arg(oracle.join("frontier.params"))
        .arg("0")
        .output()
        .expect("strict MHD-wave CLI must launch");
    assert!(
        result.status.success(),
        "CLI failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stderr).contains("synchronized MHD KDK steps"));

    let mut snapshots: Vec<PathBuf> = std::fs::read_dir(temporary.path.join("output"))
        .expect("MHD output directory")
        .map(|entry| entry.expect("MHD output entry").path())
        .collect();
    snapshots.sort();
    assert_eq!(snapshots.len(), 11);
    let c_tables = [
        (0, oracle.join("evolution_t0.csv.gz")),
        (5, oracle.join("evolution_t0.25.csv.gz")),
        (10, oracle.join("evolution_t0.5.csv.gz")),
    ];
    for (index, path) in snapshots.iter().enumerate() {
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some(format!("snapshot_{index:03}.hdf5").as_str())
        );
        let snapshot = read_mhd_wave(path).expect("Rust MHD output must be readable");
        #[allow(clippy::cast_precision_loss)]
        let expected_time = index as f64 * 0.05;
        assert!((snapshot.header.time - expected_time).abs() <= 8.0 * f64::EPSILON);
        assert_eq!(snapshot.header.num_part_total, [2048, 0, 0, 0, 0, 0]);
        assert!(snapshot.header.double_precision);
        assert_analytic_wave(&snapshot);
    }
    for (index, table) in c_tables {
        let snapshot = read_mhd_wave(&snapshots[index]).expect("MHD checkpoint");
        assert_corrected_c_l1(&snapshot, &table);
    }
}

fn assert_analytic_wave(snapshot: &MhdWaveSnapshot) {
    let time = snapshot.header.time;
    let mut error = [0.0; 8];
    for index in 0..snapshot.gas.len() {
        let phase = std::f64::consts::TAU * (snapshot.gas.coordinates[index][0] + 2.0 * time);
        let actual = [
            snapshot.gas.density[index],
            snapshot.gas.velocities[index][0],
            snapshot.gas.velocities[index][1],
            snapshot.gas.velocities[index][2],
            snapshot.gas.internal_energy[index],
            snapshot.gas.magnetic_field[index][0],
            snapshot.gas.magnetic_field[index][1],
            snapshot.gas.magnetic_field[index][2],
        ];
        for field in 0..actual.len() {
            error[field] +=
                (actual[field] - (BASE[field] + EIGENVECTOR[field] * phase.sin())).abs();
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let count = snapshot.gas.len() as f64;
    for (field, total) in error.into_iter().enumerate() {
        let limit = if field == 5 { 1.0e-8 } else { 3.0e-8 };
        assert!(
            total / count <= limit,
            "analytic field {field} L1 {} exceeds {limit}",
            total / count
        );
    }
}

fn assert_corrected_c_l1(snapshot: &MhdWaveSnapshot, path: &Path) {
    let decompressed = Command::new("gzip")
        .args(["-cd"])
        .arg(path)
        .output()
        .expect("gzip must read corrected-C table");
    assert!(decompressed.status.success());
    let contents = String::from_utf8(decompressed.stdout).expect("C table must be UTF-8");
    let mut rows = contents.lines();
    let header = rows.next().expect("C table header");
    assert!(header.starts_with("particle_id,x,velocity_x"));
    let mut errors = [0.0; 11];
    for (index, line) in rows.enumerate() {
        let columns: Vec<&str> = line.split(',').collect();
        let id: u64 = columns[0].parse().expect("numeric C particle ID");
        let values: Vec<f64> = columns
            .iter()
            .skip(1)
            .map(|value| value.parse().expect("numeric C table value"))
            .collect();
        assert_eq!(snapshot.gas.ids[index], id);
        let actual = [
            snapshot.gas.coordinates[index][0],
            snapshot.gas.velocities[index][0],
            snapshot.gas.velocities[index][1],
            snapshot.gas.velocities[index][2],
            snapshot.gas.density[index],
            snapshot.gas.internal_energy[index],
            snapshot.gas.smoothing_length[index],
            snapshot.gas.masses[index],
            snapshot.gas.magnetic_field[index][0],
            snapshot.gas.magnetic_field[index][1],
            snapshot.gas.magnetic_field[index][2],
        ];
        let expected_columns = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11];
        for field in 0..actual.len() {
            errors[field] += (actual[field] - values[expected_columns[field]]).abs();
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let count = snapshot.gas.len() as f64;
    let limits = [
        2.0e-9, 1.0e-8, 1.5e-8, 5.0e-9, 3.0e-8, 2.0e-8, 2.0e-7, 0.0, 1.0e-8, 1.5e-8, 5.0e-9,
    ];
    for field in 0..errors.len() {
        assert!(
            errors[field] / count <= limits[field],
            "corrected-C field {field} L1 {} exceeds {}",
            errors[field] / count,
            limits[field]
        );
    }
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock follows Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("gizmo-mhd-wave-cli-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path).expect("isolated MHD CLI directory");
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
