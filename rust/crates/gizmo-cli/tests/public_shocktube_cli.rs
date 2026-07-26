use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use gizmo_io::read_soundwave;

const HEADER_ATTRIBUTES: [&str; 28] = [
    "BoxSize",
    "ComovingIntegrationOn",
    "Effective_Kernel_NeighborNumber",
    "Fixed_ForceSoftening_Keplerian_Kernel_Extent",
    "Flag_Cooling",
    "Flag_DoublePrecision",
    "Flag_Feedback",
    "Flag_IC_Info",
    "Flag_Metals",
    "Flag_Sfr",
    "Flag_StellarAge",
    "GIZMO_RustPort_version",
    "GIZMO_version",
    "Gravitational_Constant_In_Code_Inits",
    "HubbleParam",
    "Kernel_Function_ID",
    "MassTable",
    "Maximum_Mass_For_Cell_Split",
    "Minimum_Mass_For_Cell_Merge",
    "NumFilesPerSnapshot",
    "NumPart_ThisFile",
    "NumPart_Total",
    "NumPart_Total_HighWord",
    "Redshift",
    "Time",
    "UnitLength_In_CGS",
    "UnitMass_In_CGS",
    "UnitVelocity_In_CGS",
];

#[test]
#[ignore = "requires pinned shock-tube assets; run via validation oracle script"]
fn strict_equal_mass_shocktube_cli_matches_corrected_c() {
    run_shocktube_cli_case(&ShocktubeCase {
        fixture_variable: "GIZMO_SHOCKTUBE_IC",
        parameter_variable: "GIZMO_SHOCKTUBE_PARAMS",
        t0_variable: "GIZMO_SHOCKTUBE_C_T0",
        tmid_variable: None,
        t5_variable: "GIZMO_SHOCKTUBE_C_T5",
        fixture_name: "shocktube_ics_emass.hdf5",
        particle_count: 320,
        box_size: 80.0,
        snapshot_cadence: 0.5,
        synchronized_steps: 8_192,
        c_parity_tolerances: [1.0e-10; 6],
        public_reference_variable: None,
    });
}

#[test]
#[ignore = "requires pinned shock-tube assets; run via validation oracle script"]
fn strict_differential_mass_shocktube_cli_matches_corrected_c() {
    run_shocktube_cli_case(&ShocktubeCase {
        fixture_variable: "GIZMO_SHOCKTUBE_DIFFMASS_IC",
        parameter_variable: "GIZMO_SHOCKTUBE_DIFFMASS_PARAMS",
        t0_variable: "GIZMO_SHOCKTUBE_C_DIFFMASS_T0",
        tmid_variable: None,
        t5_variable: "GIZMO_SHOCKTUBE_C_DIFFMASS_T5",
        fixture_name: "shocktube_ics_diffmass.hdf5",
        particle_count: 512,
        box_size: 80.0,
        snapshot_cadence: 0.5,
        synchronized_steps: 8_192,
        c_parity_tolerances: [1.0e-10; 6],
        public_reference_variable: None,
    });
}

#[test]
#[ignore = "requires pinned interacting-blast assets; run via validation oracle script"]
fn strict_interacting_blast_cli_matches_corrected_c() {
    run_shocktube_cli_case(&ShocktubeCase {
        fixture_variable: "GIZMO_INTERACTBLAST_IC",
        parameter_variable: "GIZMO_INTERACTBLAST_PARAMS",
        t0_variable: "GIZMO_INTERACTBLAST_C_T0",
        tmid_variable: Some("GIZMO_INTERACTBLAST_C_TMID"),
        t5_variable: "GIZMO_INTERACTBLAST_C_TFINAL",
        fixture_name: "interactblast_ics.hdf5",
        particle_count: 512,
        box_size: 1.0,
        snapshot_cadence: 0.0038,
        synchronized_steps: 262_144,
        // Strong interacting shocks amplify roundoff and solver-branch timing;
        // the independent public reference remains the macroscopic quality gate.
        // Normalized limits for x, vx, rho, u, H, and mass. These retain at
        // least ~40% margin over the pinned macOS corrected-C differential.
        c_parity_tolerances: [5.0e-6, 5.0e-4, 1.5e-3, 1.5e-3, 1.0e-4, 0.0],
        public_reference_variable: Some("GIZMO_INTERACTBLAST_EXACT"),
    });
}

struct ShocktubeCase {
    fixture_variable: &'static str,
    parameter_variable: &'static str,
    t0_variable: &'static str,
    tmid_variable: Option<&'static str>,
    t5_variable: &'static str,
    fixture_name: &'static str,
    particle_count: u64,
    box_size: f64,
    snapshot_cadence: f64,
    synchronized_steps: u64,
    c_parity_tolerances: [f64; 6],
    public_reference_variable: Option<&'static str>,
}

#[allow(clippy::too_many_lines)]
fn run_shocktube_cli_case(case: &ShocktubeCase) {
    let fixture = required_path(case.fixture_variable);
    let config = required_path(if case.fixture_name.starts_with("interactblast") {
        "GIZMO_INTERACTBLAST_CONFIG"
    } else {
        "GIZMO_SHOCKTUBE_CONFIG"
    });
    let parameters = required_path(case.parameter_variable);
    let expected_t0 = read_evolution_table(&required_path(case.t0_variable));
    let expected_tmid = case
        .tmid_variable
        .map(|variable| read_evolution_table(&required_path(variable)));
    let expected_t5 = read_evolution_table(&required_path(case.t5_variable));
    let temporary = TemporaryDirectory::new();
    std::fs::copy(&fixture, temporary.path.join(case.fixture_name))
        .expect("shock-tube IC must copy into isolated CLI directory");

    let result = Command::new(env!("CARGO_BIN_EXE_gizmo"))
        .current_dir(&temporary.path)
        .arg("--config")
        .arg(config)
        .arg(parameters)
        .arg("0")
        .output()
        .expect("strict shock-tube CLI must launch");
    assert!(
        result.status.success(),
        "CLI failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stderr).contains(&format!(
        "completed {} synchronized steps",
        case.synchronized_steps
    )));

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
    assert_eq!(snapshots.len(), 11);
    for (index, path) in snapshots.iter().enumerate() {
        let expected_name = format!("snapshot_{index:03}.hdf5");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some(expected_name.as_str())
        );
        assert_snapshot_schema(path, index, case);
    }
    assert_snapshot_state(&snapshots[0], &expected_t0, case.c_parity_tolerances);
    if let Some(expected) = &expected_tmid {
        assert_snapshot_state(&snapshots[5], expected, case.c_parity_tolerances);
    }
    assert_snapshot_state(&snapshots[10], &expected_t5, case.c_parity_tolerances);
    if let Some(variable) = case.public_reference_variable {
        assert_interactblast_reference(&snapshots[10], &required_path(variable));
    }
}

fn required_path(variable: &str) -> PathBuf {
    std::env::var_os(variable).map_or_else(
        || panic!("{variable} must identify a pinned oracle input"),
        PathBuf::from,
    )
}

#[allow(clippy::cast_precision_loss)]
fn assert_snapshot_schema(path: &Path, index: usize, case: &ShocktubeCase) {
    let snapshot = read_soundwave(path).expect("CLI snapshot must be readable");
    let expected_time = index as f64 * case.snapshot_cadence;
    assert!((snapshot.header.time - expected_time).abs() < f64::EPSILON * 8.0);
    assert_eq!(snapshot.header.box_size.to_bits(), case.box_size.to_bits());
    assert_eq!(
        snapshot.header.num_part_total,
        [case.particle_count, 0, 0, 0, 0, 0]
    );
    assert!(snapshot.header.double_precision);
    assert_eq!(
        snapshot.header.effective_kernel_neighbors.map(f64::to_bits),
        Some(4.0_f64.to_bits())
    );

    let file = hdf5::File::open(path).expect("CLI snapshot HDF5 must open");
    let mut root_names = file.member_names().expect("root members must be readable");
    root_names.sort();
    assert_eq!(root_names, ["Header", "PartType0"]);
    let header = file.group("Header").expect("Header group must exist");
    let mut attributes = header
        .attr_names()
        .expect("Header attributes must be readable");
    attributes.sort();
    assert_eq!(attributes, HEADER_ATTRIBUTES);
    let gas = file.group("PartType0").expect("PartType0 group must exist");
    let mut datasets = gas.member_names().expect("gas datasets must be readable");
    datasets.sort();
    assert_eq!(
        datasets,
        [
            "Coordinates",
            "Density",
            "InternalEnergy",
            "Masses",
            "ParticleIDs",
            "SmoothingLength",
            "Velocities",
        ]
    );
    let particle_count =
        usize::try_from(case.particle_count).expect("particle count must fit usize");
    for dataset_name in ["Density", "InternalEnergy", "Masses", "SmoothingLength"] {
        let dataset = gas
            .dataset(dataset_name)
            .expect("scalar gas dataset must exist");
        assert_eq!(dataset.shape(), [particle_count]);
        assert!(
            dataset
                .dtype()
                .expect("scalar gas dataset type must be readable")
                .is::<f64>()
        );
    }
    for dataset_name in ["Coordinates", "Velocities"] {
        let dataset = gas
            .dataset(dataset_name)
            .expect("vector gas dataset must exist");
        assert_eq!(dataset.shape(), [particle_count, 3]);
        assert!(
            dataset
                .dtype()
                .expect("vector gas dataset type must be readable")
                .is::<f64>()
        );
    }
    let particle_ids = gas.dataset("ParticleIDs").expect("ParticleIDs must exist");
    assert_eq!(particle_ids.shape(), [particle_count]);
    assert!(
        particle_ids
            .dtype()
            .expect("ParticleIDs type must be readable")
            .is::<u32>(),
        "legacy-compatible ParticleIDs must be uint32"
    );
}

fn assert_snapshot_state(path: &Path, expected: &EvolutionTable, tolerances: [f64; 6]) {
    let actual = read_soundwave(path).expect("CLI snapshot must be readable");
    let density = actual
        .gas
        .density
        .as_deref()
        .expect("CLI density must be present");
    let smoothing_length = actual
        .gas
        .smoothing_length
        .as_deref()
        .expect("CLI Hsml must be present");
    let mut actual_indices: Vec<usize> = (0..actual.gas.ids.len()).collect();
    actual_indices.sort_unstable_by_key(|&index| actual.gas.ids[index]);
    let actual_ids: Vec<u64> = actual_indices
        .iter()
        .map(|&index| actual.gas.ids[index])
        .collect();
    assert!(
        actual_ids.windows(2).all(|pair| pair[0] != pair[1]),
        "CLI snapshot must not contain duplicate particle IDs"
    );
    assert_eq!(actual_ids, expected.ids);
    for (expected_index, &actual_index) in actual_indices.iter().enumerate() {
        for ((field, actual, expected_value), tolerance) in [
            (
                "x",
                actual.gas.coordinates[actual_index][0],
                expected.positions[expected_index],
            ),
            (
                "velocity_x",
                actual.gas.velocities[actual_index][0],
                expected.velocities[expected_index],
            ),
            (
                "density",
                density[actual_index],
                expected.densities[expected_index],
            ),
            (
                "specific_internal_energy",
                actual.gas.internal_energy[actual_index],
                expected.specific_internal_energy[expected_index],
            ),
            (
                "smoothing_length",
                smoothing_length[actual_index],
                expected.smoothing_lengths[expected_index],
            ),
            (
                "mass",
                actual.gas.masses[actual_index],
                expected.masses[expected_index],
            ),
        ]
        .into_iter()
        .zip(tolerances)
        {
            let scale = expected_value.abs().max(1.0);
            assert!(
                (actual - expected_value).abs() <= tolerance * scale,
                "particle {} {field}: {actual} != {expected_value}",
                expected.ids[expected_index]
            );
        }
    }
}

fn assert_interactblast_reference(path: &Path, reference_path: &Path) {
    let snapshot = read_soundwave(path).expect("CLI snapshot must be readable");
    let density = snapshot
        .gas
        .density
        .as_deref()
        .expect("CLI density must be present");
    let rows: Vec<[f64; 4]> = std::fs::read_to_string(reference_path)
        .expect("public reference must be readable")
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .map(|line| {
            let values: Vec<f64> = line
                .split_whitespace()
                .map(|value| value.parse().expect("reference value must be numeric"))
                .collect();
            assert_eq!(values.len(), 7);
            [values[1], values[2], values[3], values[6]]
        })
        .collect();
    assert_eq!(rows.len(), 20_000);

    let mut weighted_error = [0.0_f64; 4];
    let mut total_volume = 0.0;
    for (index, &actual_density) in density.iter().enumerate() {
        let position = snapshot.gas.coordinates[index][0];
        let right = rows.partition_point(|row| row[0] < position);
        let (left, right) = if right == 0 {
            (0, 1)
        } else if right == rows.len() {
            (rows.len() - 2, rows.len() - 1)
        } else {
            (right - 1, right)
        };
        let fraction = (position - rows[left][0]) / (rows[right][0] - rows[left][0]);
        let interpolate = |column: usize| {
            rows[left][column] + fraction * (rows[right][column] - rows[left][column])
        };
        let expected_density = interpolate(1);
        let expected_velocity = interpolate(2);
        let expected_pressure = interpolate(3);
        let actual_pressure = 0.4 * actual_density * snapshot.gas.internal_energy[index];
        let actual_entropy = actual_pressure / actual_density.powf(1.4);
        let expected_entropy = expected_pressure / expected_density.powf(1.4);
        let actual = [
            actual_density,
            actual_pressure,
            actual_entropy,
            snapshot.gas.velocities[index][0],
        ];
        let expected = [
            expected_density,
            expected_pressure,
            expected_entropy,
            expected_velocity,
        ];
        let volume = snapshot.gas.masses[index] / actual_density;
        for field in 0..4 {
            weighted_error[field] += volume * (actual[field] - expected[field]).abs();
        }
        total_volume += volume;
    }
    for error in &mut weighted_error {
        *error /= total_volume;
    }
    let corrected_c_baseline = [
        0.046_259_608_024_094_4,
        3.720_250_090_920_468,
        15.666_363_547_865_624,
        0.191_381_933_286_983_2,
    ];
    for (field, (actual, baseline)) in ["density", "pressure", "entropy", "velocity_x"]
        .into_iter()
        .zip(weighted_error.into_iter().zip(corrected_c_baseline))
    {
        assert!(
            (actual - baseline).abs() <= baseline * 1.0e-3,
            "{field} public-reference L1 moved outside the 0.1% corrected-C band: \
             actual={actual}, baseline={baseline}"
        );
    }
}

struct EvolutionTable {
    ids: Vec<u64>,
    positions: Vec<f64>,
    velocities: Vec<f64>,
    densities: Vec<f64>,
    specific_internal_energy: Vec<f64>,
    smoothing_lengths: Vec<f64>,
    masses: Vec<f64>,
}

struct EvolutionRow {
    id: u64,
    position: f64,
    velocity: f64,
    density: f64,
    specific_internal_energy: f64,
    smoothing_length: f64,
    mass: f64,
}

fn read_evolution_table(path: &Path) -> EvolutionTable {
    let contents = std::fs::read_to_string(path).expect("evolution table must be readable");
    let mut lines = contents.lines();
    assert_eq!(
        lines.next(),
        Some("particle_id,x,velocity_x,density,specific_internal_energy,smoothing_length,mass")
    );
    let mut rows = Vec::new();
    for line in lines {
        let columns: Vec<&str> = line.split(',').collect();
        assert_eq!(columns.len(), 7);
        rows.push(EvolutionRow {
            id: columns[0].parse().unwrap(),
            position: columns[1].parse().unwrap(),
            velocity: columns[2].parse().unwrap(),
            density: columns[3].parse().unwrap(),
            specific_internal_energy: columns[4].parse().unwrap(),
            smoothing_length: columns[5].parse().unwrap(),
            mass: columns[6].parse().unwrap(),
        });
    }
    rows.sort_unstable_by_key(|row| row.id);
    assert!(
        rows.windows(2).all(|pair| pair[0].id != pair[1].id),
        "evolution table must not contain duplicate particle IDs"
    );
    EvolutionTable {
        ids: rows.iter().map(|row| row.id).collect(),
        positions: rows.iter().map(|row| row.position).collect(),
        velocities: rows.iter().map(|row| row.velocity).collect(),
        densities: rows.iter().map(|row| row.density).collect(),
        specific_internal_energy: rows
            .iter()
            .map(|row| row.specific_internal_energy)
            .collect(),
        smoothing_lengths: rows.iter().map(|row| row.smoothing_length).collect(),
        masses: rows.iter().map(|row| row.mass).collect(),
    }
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
            "gizmo-shocktube-cli-{}-{nonce}",
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
