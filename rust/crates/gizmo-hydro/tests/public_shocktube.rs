use gizmo_hydro::{
    LEGACY_TIMEBASE_TICKS, MfmDriftState1d, MfmEvolvingState1d, SynchronizedTimeline1d,
    begin_mfm_kdk_1d, density_at_hsml_1d, finish_mfm_kdk_1d, mfm_spatial_rates_1d,
    select_public_soundwave_timestep_1d, solve_public_c_initial_smoothing_lengths_1d,
};
use gizmo_io::read_soundwave;
use std::path::{Path, PathBuf};

const TIME_MAX: f64 = 5.0;
const OUTPUT_INTERVAL: f64 = 0.5;
const MAXIMUM_TIMESTEP: f64 = 1.0e-3;
const COURANT_FACTOR: f64 = 0.05;
const INTEGRATION_ACCURACY: f64 = 0.0025;
const DESIRED_NEIGHBORS: f64 = 4.0;
const NEIGHBOR_TOLERANCE: f64 = 0.05;
const EXPECTED_STEP_TICKS: u64 = 1_u64 << 47;
const EXPECTED_STEPS: u64 = 8_192;
const EXPECTED_OUTPUTS: u32 = 11;
// Corrected-C baselines against the public high-resolution PPM table. Rust
// may exceed each only by this small cross-platform interpolation allowance.
const PUBLIC_REFERENCE_ALLOWANCE: f64 = 1.0e-4;
const EQUAL_MASS_PUBLIC_REFERENCE_BASELINES: [f64; 4] = [
    0.004_816_710_484_520_634,
    0.005_241_033_668_199_458,
    0.003_393_203_763_173_442,
    0.008_991_198_440_929_557,
];
const DIFFERENTIAL_MASS_PUBLIC_REFERENCE_BASELINES: [f64; 4] = [
    0.005_233_216_905_994_052,
    0.005_757_768_534_014_373,
    0.003_063_822_487_191_69,
    0.004_252_986_117_800_043,
];
const C_PARITY_TOLERANCE: f64 = 1.0e-10;

#[test]
#[ignore = "requires pinned shock-tube IC and corrected-C tables; run via validation oracle script"]
fn rust_equal_mass_shocktube_matches_corrected_c() {
    run_shocktube_case(&ShocktubeCase {
        fixture_variable: "GIZMO_SHOCKTUBE_IC",
        t0_variable: "GIZMO_SHOCKTUBE_C_T0",
        t5_variable: "GIZMO_SHOCKTUBE_C_T5",
        first_drift_variable: Some("GIZMO_SHOCKTUBE_C_STEP1_DRIFT"),
        first_postkick_variable: Some("GIZMO_SHOCKTUBE_C_STEP1_POSTKICK"),
        particle_count: 320,
        public_reference_baselines: EQUAL_MASS_PUBLIC_REFERENCE_BASELINES,
    });
}

#[test]
#[ignore = "requires pinned shock-tube IC and corrected-C tables; run via validation oracle script"]
fn rust_differential_mass_shocktube_matches_corrected_c() {
    run_shocktube_case(&ShocktubeCase {
        fixture_variable: "GIZMO_SHOCKTUBE_DIFFMASS_IC",
        t0_variable: "GIZMO_SHOCKTUBE_C_DIFFMASS_T0",
        t5_variable: "GIZMO_SHOCKTUBE_C_DIFFMASS_T5",
        first_drift_variable: None,
        first_postkick_variable: None,
        particle_count: 512,
        public_reference_baselines: DIFFERENTIAL_MASS_PUBLIC_REFERENCE_BASELINES,
    });
}

struct ShocktubeCase {
    fixture_variable: &'static str,
    t0_variable: &'static str,
    t5_variable: &'static str,
    first_drift_variable: Option<&'static str>,
    first_postkick_variable: Option<&'static str>,
    particle_count: usize,
    public_reference_baselines: [f64; 4],
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn run_shocktube_case(case: &ShocktubeCase) {
    let fixture_path = required_path(case.fixture_variable);
    let initialized_path = required_path(case.t0_variable);
    let terminal_path = required_path(case.t5_variable);
    let first_drift = case
        .first_drift_variable
        .map(|variable| read_evolution_table(Path::new(&required_path(variable))));
    let first_postkick = case
        .first_postkick_variable
        .map(|variable| read_evolution_table(Path::new(&required_path(variable))));
    let exact_path = std::env::var_os("GIZMO_SHOCKTUBE_EXACT")
        .expect("GIZMO_SHOCKTUBE_EXACT must identify the public PPM reference");
    let fixture = read_soundwave(fixture_path).expect("public shock-tube IC must be valid");
    let initialized = read_evolution_table(Path::new(&initialized_path));
    let terminal = read_evolution_table(Path::new(&terminal_path));
    let exact_reference = read_exact_reference(Path::new(&exact_path));
    assert_eq!(fixture.gas.len(), case.particle_count);
    assert_eq!(initialized.positions.len(), fixture.gas.len());
    assert_eq!(terminal.positions.len(), fixture.gas.len());
    assert_eq!(fixture.header.box_size.to_bits(), 80.0_f64.to_bits());

    let positions: Vec<f64> = fixture
        .gas
        .coordinates
        .iter()
        .map(|coordinate| coordinate[0])
        .collect();
    let velocities: Vec<f64> = fixture
        .gas
        .velocities
        .iter()
        .map(|velocity| velocity[0])
        .collect();
    assert!(max_absolute_error(&positions, &initialized.positions) < 1.0e-15);
    assert!(max_relative_error(&fixture.gas.masses, &initialized.masses) < 1.0e-15);
    assert!(
        max_relative_error(
            &fixture.gas.internal_energy,
            &initialized.specific_internal_energy,
        ) < 1.0e-15
    );

    let solved = solve_public_c_initial_smoothing_lengths_1d(
        &positions,
        &fixture.gas.masses,
        fixture.header.box_size,
        DESIRED_NEIGHBORS,
        NEIGHBOR_TOLERANCE,
    )
    .expect("public-C shock-tube smoothing initialization must converge");
    let smoothing_lengths: Vec<f64> = solved
        .iter()
        .map(|particle| particle.smoothing_length)
        .collect();
    let initial_densities: Vec<f64> = solved
        .iter()
        .map(|particle| particle.estimate.density)
        .collect();
    let initial_hsml_error = max_relative_error(&smoothing_lengths, &initialized.smoothing_lengths);
    let initial_density_error = max_relative_error(&initial_densities, &initialized.densities);
    eprintln!(
        "corrected-C shock-tube initialization: max rel density/Hsml=\
         {initial_density_error:.12e}/{initial_hsml_error:.12e}"
    );
    assert!(initial_density_error < 1.0e-12);
    assert!(initial_hsml_error < 1.0e-12);
    let density_at_c_hsml: Vec<f64> = density_at_hsml_1d(
        &positions,
        &fixture.gas.masses,
        &initialized.smoothing_lengths,
        fixture.header.box_size,
    )
    .expect("corrected-C t=0 Hsml must define valid Rust densities")
    .into_iter()
    .map(|estimate| estimate.density)
    .collect();
    assert!(
        max_relative_error(&density_at_c_hsml, &initialized.densities) < 1.0e-12,
        "Rust density math must match corrected C when initialized on the same Hsml branch"
    );

    let mut state = MfmEvolvingState1d {
        positions,
        masses: fixture.gas.masses,
        velocities: velocities.clone(),
        specific_internal_energy: fixture.gas.internal_energy,
        smoothing_lengths,
        box_size: fixture.header.box_size,
        gamma: 1.4,
    };
    let mut rates =
        mfm_spatial_rates_1d(state.as_view()).expect("initial shock-tube RHS must be valid");
    let mut timeline =
        SynchronizedTimeline1d::new(0.0, TIME_MAX).expect("LONG timeline must be valid");
    let tick_duration = TIME_MAX / LEGACY_TIMEBASE_TICKS as f64;
    let mut next_output_time = 0.0_f64;
    let mut next_output_tick = Some(0_u64);
    let mut output_count = 0_u32;
    let mut step_count = 0_u64;
    let mut compared_initial = false;
    let mut compared_terminal = false;

    while !timeline.is_finished() {
        let selected = select_public_soundwave_timestep_1d(
            state.as_view(),
            &rates,
            MAXIMUM_TIMESTEP,
            COURANT_FACTOR,
            INTEGRATION_ACCURACY,
        )
        .expect("complete public timestep selector must succeed");
        let synchronized = timeline
            .select_step(selected.duration, MAXIMUM_TIMESTEP)
            .expect("selected timestep must quantize on the LONG timeline");
        assert_eq!(
            synchronized.ticks, EXPECTED_STEP_TICKS,
            "corrected C kept every particle in timebin 47"
        );
        let start_tick = timeline.current_tick();
        let end_tick = start_tick + synchronized.ticks;
        let mut prepared = begin_mfm_kdk_1d(&state, &rates, synchronized.duration, 0.0)
            .expect("shock-tube first kick and drift preparation must succeed");
        if step_count == 0 {
            if let Some(expected) = &first_drift {
                let mut diagnostic = prepared.clone();
                let first_endpoint = diagnostic
                    .drift_state(synchronized.duration)
                    .expect("first shock-tube endpoint drift must be valid");
                assert_drift_matches("first endpoint drift", &first_endpoint, expected);
            }
        }

        while next_output_tick.is_some_and(|tick| tick <= end_tick) {
            let output_tick = next_output_tick.expect("checked above");
            assert!(output_tick >= start_tick);
            let elapsed = (output_tick - start_tick) as f64 * tick_duration;
            let drift = prepared
                .drift_state(elapsed)
                .expect("scheduled shock-tube drift must be valid");
            if output_tick == 0 {
                assert_drift_matches("t=0", &drift, &initialized);
                let kick_signal = max_absolute_error(&initialized.velocities, &velocities);
                let kick_error =
                    max_absolute_error(&drift.conserved_velocities, &initialized.velocities);
                eprintln!(
                    "corrected-C shock-tube first half-kick error/signal=\
                     {kick_error:.12e}/{kick_signal:.12e}"
                );
                compared_initial = true;
            }
            if output_tick == LEGACY_TIMEBASE_TICKS {
                assert_drift_matches("t=5", &drift, &terminal);
                assert_public_reference_norms(
                    &drift,
                    &state.masses,
                    &exact_reference,
                    case.public_reference_baselines,
                    state.box_size,
                );
                compared_terminal = true;
            }
            output_count += 1;
            next_output_time += OUTPUT_INTERVAL;
            next_output_tick = (next_output_time <= TIME_MAX)
                .then(|| legacy_output_tick(next_output_time, tick_duration));
            assert!(next_output_tick.is_none_or(|tick| tick > output_tick));
        }

        let (endpoint, new_rates) =
            finish_mfm_kdk_1d(prepared, DESIRED_NEIGHBORS, NEIGHBOR_TOLERANCE)
                .expect("shock-tube endpoint force and second kick must succeed");
        if step_count == 0 {
            if let Some(expected) = &first_postkick {
                assert_completed_matches("first completed step", &endpoint, expected);
            }
        }
        state = endpoint;
        rates = new_rates;
        timeline
            .advance(synchronized)
            .expect("completed step must advance the LONG timeline");
        step_count += 1;
    }

    assert!(compared_initial);
    assert!(compared_terminal);
    assert_eq!(step_count, EXPECTED_STEPS);
    assert_eq!(output_count, EXPECTED_OUTPUTS);
}

fn assert_completed_matches(phase: &str, actual: &MfmEvolvingState1d, expected: &EvolutionTable) {
    let densities: Vec<f64> = density_at_hsml_1d(
        &actual.positions,
        &actual.masses,
        &actual.smoothing_lengths,
        actual.box_size,
    )
    .expect("completed shock-tube state must have valid densities")
    .into_iter()
    .map(|estimate| estimate.density)
    .collect();
    let position_error = max_absolute_error(&actual.positions, &expected.positions);
    let velocity_error = max_absolute_error(&actual.velocities, &expected.velocities);
    let density_error = max_relative_error(&densities, &expected.densities);
    let energy_error = max_relative_error(
        &actual.specific_internal_energy,
        &expected.specific_internal_energy,
    );
    let smoothing_error =
        max_relative_error(&actual.smoothing_lengths, &expected.smoothing_lengths);
    eprintln!(
        "corrected-C shock-tube {phase}: max |dx|/|dv|=\
         {position_error:.12e}/{velocity_error:.12e}, max rel density/u/Hsml=\
         {density_error:.12e}/{energy_error:.12e}/{smoothing_error:.12e}"
    );
    assert!(
        position_error < C_PARITY_TOLERANCE,
        "{phase} position parity"
    );
    assert!(
        velocity_error < C_PARITY_TOLERANCE,
        "{phase} velocity parity"
    );
    assert!(density_error < C_PARITY_TOLERANCE, "{phase} density parity");
    assert!(
        energy_error < C_PARITY_TOLERANCE,
        "{phase} internal-energy parity"
    );
    assert!(smoothing_error < C_PARITY_TOLERANCE, "{phase} Hsml parity");
}

fn assert_drift_matches(phase: &str, actual: &MfmDriftState1d, expected: &EvolutionTable) {
    let position_error = max_absolute_error(&actual.positions, &expected.positions);
    let velocity_error = max_absolute_error(&actual.conserved_velocities, &expected.velocities);
    let density_error = max_relative_error(&actual.predicted_density, &expected.densities);
    let energy_error = max_relative_error(
        &actual.predicted_specific_internal_energy,
        &expected.specific_internal_energy,
    );
    let smoothing_error = max_relative_error(
        &actual.predicted_smoothing_lengths,
        &expected.smoothing_lengths,
    );
    eprintln!(
        "corrected-C shock-tube {phase}: max |dx|/|dv|=\
         {position_error:.12e}/{velocity_error:.12e}, max rel density/u/Hsml=\
         {density_error:.12e}/{energy_error:.12e}/{smoothing_error:.12e}"
    );
    assert!(
        position_error < C_PARITY_TOLERANCE,
        "{phase} position parity"
    );
    assert!(
        velocity_error < C_PARITY_TOLERANCE,
        "{phase} velocity parity"
    );
    assert!(density_error < C_PARITY_TOLERANCE, "{phase} density parity");
    assert!(
        energy_error < C_PARITY_TOLERANCE,
        "{phase} internal-energy parity"
    );
    assert!(smoothing_error < C_PARITY_TOLERANCE, "{phase} Hsml parity");
}

fn read_exact_reference(path: &Path) -> Vec<[f64; 5]> {
    let contents = std::fs::read_to_string(path).expect("public PPM reference must be readable");
    let rows: Vec<[f64; 5]> = contents
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let values: Vec<f64> = line
                .split_whitespace()
                .map(|value| value.parse().expect("PPM value must be numeric"))
                .collect();
            assert_eq!(values.len(), 5);
            Some([values[0], values[1], values[2], values[3], values[4]])
        })
        .collect();
    assert!(rows.len() > 2);
    assert!(rows.windows(2).all(|pair| pair[0][0] < pair[1][0]));
    rows
}

fn periodic_reference_value(
    reference: &[[f64; 5]],
    position: f64,
    column: usize,
    box_size: f64,
) -> f64 {
    let right_index = reference.partition_point(|row| row[0] < position);
    let (left_x, left_value, right_x, right_value) = if right_index == 0 {
        let left = reference.last().expect("reference is nonempty");
        (
            left[0] - box_size,
            left[column],
            reference[0][0],
            reference[0][column],
        )
    } else if right_index == reference.len() {
        let left = reference.last().expect("reference is nonempty");
        (
            left[0],
            left[column],
            reference[0][0] + box_size,
            reference[0][column],
        )
    } else {
        let left = reference[right_index - 1];
        let right = reference[right_index];
        (left[0], left[column], right[0], right[column])
    };
    let fraction = (position - left_x) / (right_x - left_x);
    left_value + fraction * (right_value - left_value)
}

fn assert_public_reference_norms(
    state: &MfmDriftState1d,
    masses: &[f64],
    reference: &[[f64; 5]],
    baselines: [f64; 4],
    box_size: f64,
) {
    assert_eq!(state.positions.len(), masses.len());
    let mut weighted_errors = [0.0; 4];
    let mut total_volume = 0.0;
    for (index, mass) in masses.iter().enumerate() {
        let density = state.predicted_density[index];
        let internal_energy = state.predicted_specific_internal_energy[index];
        let volume = mass / density;
        let actual = [
            density,
            0.4 * density * internal_energy,
            0.4 * internal_energy * density.powf(-0.4),
            state.conserved_velocities[index],
        ];
        for field in 0..4 {
            let expected =
                periodic_reference_value(reference, state.positions[index], field + 1, box_size);
            weighted_errors[field] += volume * (actual[field] - expected).abs();
        }
        total_volume += volume;
    }
    let measured = weighted_errors.map(|error| error / total_volume);
    eprintln!(
        "public PPM volume-weighted L1 rho/P/entropy/v={:.12e}/{:.12e}/{:.12e}/{:.12e}",
        measured[0], measured[1], measured[2], measured[3]
    );
    for (field, (actual, baseline)) in ["density", "pressure", "entropy", "velocity_x"]
        .into_iter()
        .zip(measured.into_iter().zip(baselines))
    {
        assert!(
            actual <= baseline + PUBLIC_REFERENCE_ALLOWANCE,
            "{field} public-reference L1 {actual} exceeds corrected-C baseline {baseline} \
             plus allowance {PUBLIC_REFERENCE_ALLOWANCE}"
        );
    }
}

fn required_path(variable: &str) -> PathBuf {
    std::env::var_os(variable).map_or_else(
        || panic!("{variable} must identify a pinned oracle input"),
        PathBuf::from,
    )
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn legacy_output_tick(output_time: f64, tick_duration: f64) -> u64 {
    (output_time / tick_duration) as u64
}

#[derive(Debug)]
struct EvolutionTable {
    positions: Vec<f64>,
    velocities: Vec<f64>,
    densities: Vec<f64>,
    specific_internal_energy: Vec<f64>,
    smoothing_lengths: Vec<f64>,
    masses: Vec<f64>,
}

fn read_evolution_table(path: &Path) -> EvolutionTable {
    let contents = std::fs::read_to_string(path).expect("evolution table must be readable");
    let mut lines = contents.lines();
    assert_eq!(
        lines.next(),
        Some("particle_id,x,velocity_x,density,specific_internal_energy,smoothing_length,mass")
    );
    let mut table = EvolutionTable {
        positions: Vec::new(),
        velocities: Vec::new(),
        densities: Vec::new(),
        specific_internal_energy: Vec::new(),
        smoothing_lengths: Vec::new(),
        masses: Vec::new(),
    };
    for (expected_id, line) in lines.enumerate() {
        let columns: Vec<&str> = line.split(',').collect();
        assert_eq!(columns.len(), 7);
        assert_eq!(columns[0].parse::<usize>().unwrap(), expected_id);
        table.positions.push(columns[1].parse().unwrap());
        table.velocities.push(columns[2].parse().unwrap());
        table.densities.push(columns[3].parse().unwrap());
        table
            .specific_internal_energy
            .push(columns[4].parse().unwrap());
        table.smoothing_lengths.push(columns[5].parse().unwrap());
        table.masses.push(columns[6].parse().unwrap());
    }
    table
}

fn max_absolute_error(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let error = (actual - expected).abs();
            if error.is_finite() {
                error
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

fn max_relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let error = (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
            if error.is_finite() {
                error
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}
