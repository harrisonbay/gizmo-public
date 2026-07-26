use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use gizmo_io::{DustyWaveSnapshot, read_dustywave};

#[test]
#[ignore = "requires pinned dusty-box assets; run via validation oracle script"]
fn strict_dustybox_cli_matches_corrected_c_and_analytic_solution() {
    let fixture = required_path("GIZMO_DUSTYBOX_IC");
    let config = required_path("GIZMO_DUSTYBOX_CONFIG");
    let parameters = required_path("GIZMO_DUSTYBOX_PARAMS");
    let expected = [
        read_evolution_table(&required_path("GIZMO_DUSTYBOX_C_T0")),
        read_evolution_table(&required_path("GIZMO_DUSTYBOX_C_T1_25")),
        read_evolution_table(&required_path("GIZMO_DUSTYBOX_C_T2_5")),
    ];
    let temporary = TemporaryDirectory::new();
    std::fs::copy(&fixture, temporary.path.join("dustybox_ics.hdf5"))
        .expect("dusty-box IC must copy into isolated CLI directory");
    let result = Command::new(env!("CARGO_BIN_EXE_gizmo"))
        .current_dir(&temporary.path)
        .arg("--config")
        .arg(config)
        .arg(parameters)
        .arg("0")
        .output()
        .expect("strict dusty-box CLI must launch");
    assert!(
        result.status.success(),
        "CLI failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("completed 32768 synchronized dusty-box steps"));
    assert!(stderr.contains("maximum drag momentum residual="));

    let output = temporary.path.join("output");
    let mut snapshots: Vec<PathBuf> = std::fs::read_dir(&output)
        .expect("CLI output directory must exist")
        .map(|entry| entry.expect("output entry must be readable").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "hdf5")
        })
        .collect();
    snapshots.sort();
    assert_eq!(snapshots.len(), 251);
    for (number, path) in snapshots.iter().enumerate() {
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some(format!("snapshot_{number:03}.hdf5").as_str())
        );
    }
    for (snapshot_index, expected_index, expected_time) in [
        (0, 0, 0.0_f64),
        (125, 1, 1.250_000_000_000_000_9),
        (250, 2, 2.5),
    ] {
        let snapshot = read_dustywave(&snapshots[snapshot_index])
            .expect("Rust dusty-box output must be readable");
        assert_eq!(snapshot.header.num_part_total, [64, 0, 0, 64, 0, 0]);
        assert!(snapshot.header.double_precision);
        assert_eq!(snapshot.header.time.to_bits(), expected_time.to_bits());
        assert_state_matches(&snapshot, &expected[expected_index]);
    }
    for path in &snapshots {
        let snapshot = read_dustywave(path).expect("Rust dusty-box output must be readable");
        assert_analytic_solution(&snapshot, snapshot.header.time);
    }
}

fn assert_state_matches(snapshot: &DustyWaveSnapshot, expected: &BTreeMap<u64, EvolutionRow>) {
    let gas_density = snapshot.gas.density.as_deref().expect("gas density");
    let gas_hsml = snapshot
        .gas
        .smoothing_length
        .as_deref()
        .expect("gas smoothing length");
    for index in 0..snapshot.gas.len() {
        let id = snapshot.gas.ids[index];
        let row = expected.get(&id).expect("corrected-C gas particle");
        assert_eq!(row.particle_type, 0);
        assert_close(
            id,
            "gas x",
            snapshot.gas.coordinates[index][0],
            row.x,
            1.0e-10,
        );
        assert_close(
            id,
            "gas velocity",
            snapshot.gas.velocities[index][0],
            row.velocity,
            5.0e-10,
        );
        assert_close(id, "gas mass", snapshot.gas.masses[index], row.mass, 0.0);
        assert_close(
            id,
            "gas smoothing length",
            gas_hsml[index],
            row.smoothing_length,
            5.0e-6,
        );
        assert_close(
            id,
            "gas density",
            gas_density[index],
            row.density.expect("corrected-C density"),
            5.0e-10,
        );
        assert_close(
            id,
            "gas internal energy",
            snapshot.gas.internal_energy[index],
            row.internal_energy.expect("corrected-C internal energy"),
            5.0e-10,
        );
    }
    let grain_hsml = snapshot
        .grains
        .smoothing_length
        .as_deref()
        .expect("grain smoothing length");
    for (index, &id) in snapshot.grains.ids.iter().enumerate() {
        let row = expected.get(&id).expect("corrected-C grain particle");
        assert_eq!(row.particle_type, 3);
        assert_close(
            id,
            "grain x",
            snapshot.grains.coordinates[index][0],
            row.x,
            1.0e-10,
        );
        assert_close(
            id,
            "grain velocity",
            snapshot.grains.velocities[index][0],
            row.velocity,
            1.0e-10,
        );
        assert_close(
            id,
            "grain mass",
            snapshot.grains.masses[index],
            row.mass,
            0.0,
        );
        assert_close(
            id,
            "grain smoothing length",
            grain_hsml[index],
            row.smoothing_length,
            3.0e-6,
        );
        assert_close(
            id,
            "grain size",
            snapshot.grains.grain_size[index],
            row.grain_size.expect("corrected-C grain size"),
            1.0e-14,
        );
    }
}

fn assert_analytic_solution(snapshot: &DustyWaveSnapshot, time: f64) {
    let alpha = 15.0 * std::f64::consts::PI / 128.0;
    let psi = (-2.0 * time).exp() / (1.0 + (1.0 + alpha).sqrt());
    let relative_velocity = 2.0 * psi / (1.0 - alpha * psi * psi);
    let expected = [
        0.5 * (1.0 - relative_velocity),
        0.5 * (1.0 + relative_velocity),
    ];
    let rms = |velocities: &[[f64; 3]], expected: f64| {
        #[allow(clippy::cast_precision_loss)]
        let count = velocities.len() as f64;
        velocities
            .iter()
            .map(|velocity| (velocity[0] - expected).powi(2))
            .sum::<f64>()
            .sqrt()
            / count.sqrt()
    };
    let gas_rms = rms(&snapshot.gas.velocities, expected[0]);
    let grain_rms = rms(&snapshot.grains.velocities, expected[1]);
    assert!(
        gas_rms <= 6.0e-5,
        "gas analytic RMS {gas_rms:.17e} exceeds corrected-C envelope"
    );
    assert!(
        grain_rms <= 6.0e-5,
        "grain analytic RMS {grain_rms:.17e} exceeds corrected-C envelope"
    );
    let momentum: f64 = snapshot
        .gas
        .masses
        .iter()
        .zip(&snapshot.gas.velocities)
        .chain(
            snapshot
                .grains
                .masses
                .iter()
                .zip(&snapshot.grains.velocities),
        )
        .map(|(mass, velocity)| mass * velocity[0])
        .sum();
    assert!(
        (momentum - 1.0).abs() <= 2.0e-14,
        "total momentum {momentum:.17e} moved outside corrected-C envelope"
    );
}

fn assert_close(id: u64, field: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "particle {id} {field}: {actual:.17e} != {expected:.17e} (limit {tolerance:.3e})"
    );
}

#[derive(Debug)]
struct EvolutionRow {
    particle_type: u8,
    x: f64,
    velocity: f64,
    mass: f64,
    smoothing_length: f64,
    density: Option<f64>,
    internal_energy: Option<f64>,
    grain_size: Option<f64>,
}

fn read_evolution_table(path: &Path) -> BTreeMap<u64, EvolutionRow> {
    let contents = std::fs::read_to_string(path).expect("evolution table must be readable");
    let mut lines = contents.lines();
    assert_eq!(
        lines.next(),
        Some(
            "particle_id,particle_type,x,velocity_x,mass,smoothing_length,density,\
             specific_internal_energy,grain_size"
        )
    );
    lines
        .map(|line| {
            let columns: Vec<&str> = line.split(',').collect();
            let optional = |value: &str| (!value.is_empty()).then(|| value.parse().unwrap());
            (
                columns[0].parse().unwrap(),
                EvolutionRow {
                    particle_type: columns[1].parse().unwrap(),
                    x: columns[2].parse().unwrap(),
                    velocity: columns[3].parse().unwrap(),
                    mass: columns[4].parse().unwrap(),
                    smoothing_length: columns[5].parse().unwrap(),
                    density: optional(columns[6]),
                    internal_energy: optional(columns[7]),
                    grain_size: optional(columns[8]),
                },
            )
        })
        .collect()
}

fn required_path(variable: &str) -> PathBuf {
    std::env::var_os(variable).map_or_else(
        || panic!("{variable} must identify a pinned oracle input"),
        PathBuf::from,
    )
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must follow Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("gizmo-dustybox-cli-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path).expect("isolated CLI directory must be created");
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
