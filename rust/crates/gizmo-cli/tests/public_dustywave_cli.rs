use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use gizmo_io::{DustyWaveSnapshot, read_dustywave};

#[test]
#[ignore = "requires pinned dusty-wave assets; run via validation oracle script"]
fn strict_dustywave_cli_matches_corrected_c_and_public_reference() {
    let fixture = required_path("GIZMO_DUSTYWAVE_IC");
    let config = required_path("GIZMO_DUSTYWAVE_CONFIG");
    let parameters = required_path("GIZMO_DUSTYWAVE_PARAMS");
    let expected = [
        read_evolution_table(&required_path("GIZMO_DUSTYWAVE_C_T0")),
        read_evolution_table(&required_path("GIZMO_DUSTYWAVE_C_T1_2")),
        read_evolution_table(&required_path("GIZMO_DUSTYWAVE_C_T2_5")),
    ];
    let temporary = TemporaryDirectory::new();
    std::fs::copy(&fixture, temporary.path.join("dustywave_ics.hdf5"))
        .expect("dusty-wave IC must copy into isolated CLI directory");
    let result = Command::new(env!("CARGO_BIN_EXE_gizmo"))
        .current_dir(&temporary.path)
        .arg("--config")
        .arg(config)
        .arg(parameters)
        .arg("0")
        .output()
        .expect("strict dusty-wave CLI must launch");
    assert!(
        result.status.success(),
        "CLI failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("completed 32768 synchronized dusty-wave steps"));
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
        (120, 1, 1.200_000_000_000_000_8),
        (250, 2, 2.5),
    ] {
        let snapshot = read_dustywave(&snapshots[snapshot_index])
            .expect("Rust dusty-wave output must be readable");
        assert_eq!(snapshot.header.num_part_total, [64, 0, 0, 64, 0, 0]);
        assert!(snapshot.header.double_precision);
        assert_eq!(snapshot.header.time.to_bits(), expected_time.to_bits());
        assert_state_matches(&snapshot, &expected[expected_index]);
    }
    assert_public_reference(
        &read_dustywave(&snapshots[120]).expect("reference-time snapshot must read"),
        &required_path("GIZMO_DUSTYWAVE_EXACT"),
    );
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

fn assert_close(id: u64, field: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "particle {id} {field}: {actual:.17e} != {expected:.17e} (limit {tolerance:.3e})"
    );
}

fn assert_public_reference(snapshot: &DustyWaveSnapshot, path: &Path) {
    let mut reference: Vec<[f64; 3]> = std::fs::read_to_string(path)
        .expect("public dusty-wave reference must be readable")
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .map(|line| {
            let values: Vec<f64> = line
                .split_whitespace()
                .map(|value| value.parse().expect("numeric reference value"))
                .collect();
            [values[0], values[1] * 1.0e-4, values[2] * 1.0e-4]
        })
        .collect();
    reference.sort_by(|left, right| left[0].total_cmp(&right[0]));
    let rms = |positions: &[[f64; 3]], velocities: &[[f64; 3]], column: usize| {
        let square_sum: f64 = positions
            .iter()
            .zip(velocities)
            .map(|(position, velocity)| {
                let expected = periodic_interpolate(&reference, position[0], column);
                (velocity[0] - expected).powi(2)
            })
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let count = positions.len() as f64;
        square_sum.sqrt() / count.sqrt()
    };
    let actual = [
        rms(&snapshot.grains.coordinates, &snapshot.grains.velocities, 1),
        rms(&snapshot.gas.coordinates, &snapshot.gas.velocities, 2),
    ];
    let corrected_c = [2.822_548_887_959_355_2e-8, 4.468_233_561_692_998_3e-7];
    for (species, (actual, baseline)) in ["grain", "gas"]
        .into_iter()
        .zip(actual.into_iter().zip(corrected_c))
    {
        assert!(
            (actual - baseline).abs() <= baseline * 1.0e-3,
            "{species} reference RMS moved outside the 0.1% corrected-C band: \
             actual={actual:.17e}, baseline={baseline:.17e}"
        );
    }
}

fn periodic_interpolate(reference: &[[f64; 3]], position: f64, column: usize) -> f64 {
    let right = reference.partition_point(|row| row[0] < position);
    let left = (right + reference.len() - 1) % reference.len();
    let right = right % reference.len();
    let x_left = reference[left][0];
    let mut x_right = reference[right][0];
    let mut x = position;
    if right == 0 {
        x_right += 1.0;
    }
    if x < x_left {
        x += 1.0;
    }
    let fraction = (x - x_left) / (x_right - x_left);
    reference[left][column] + fraction * (reference[right][column] - reference[left][column])
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
        let path = std::env::temp_dir().join(format!(
            "gizmo-dustywave-cli-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("isolated CLI directory must be created");
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
