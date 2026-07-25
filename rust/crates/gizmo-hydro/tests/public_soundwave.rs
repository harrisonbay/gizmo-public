use gizmo_hydro::{
    EntropicPoint1d, GradientEstimate, MeshlessPoint1d, MfmEvolvingState1d, MfmState1d,
    PrimitiveState1d, ReconstructedPoint1d, RiemannMethod, SynchronizedTimeline1d,
    advance_mfm_kdk_1d, apply_entropic_pdv_1d, cubic_kernel_1d, density_at_hsml_1d,
    face_closure_errors_1d, global_courant_timestep_1d, gradients_at_hsml_1d, inverse_moments_1d,
    meshless_face_geometry_1d, mfm_pair_flux_1d, mfm_spatial_rates_1d, solve_smoothing_lengths_1d,
};
use gizmo_io::read_soundwave;
use std::path::Path;

#[test]
#[ignore = "requires GIZMO_SOUNDWAVE_IC; run via validation oracle script"]
#[allow(clippy::too_many_lines)]
fn rust_density_matches_pinned_public_soundwave_state() {
    let path = std::env::var_os("GIZMO_SOUNDWAVE_IC")
        .expect("GIZMO_SOUNDWAVE_IC must identify the pinned fixture");
    let snapshot = read_soundwave(path).expect("pinned sound-wave fixture must be valid");
    let expected_density = snapshot
        .gas
        .density
        .as_deref()
        .expect("pinned fixture must contain Density");
    let smoothing_lengths = snapshot
        .gas
        .smoothing_length
        .as_deref()
        .expect("pinned fixture must contain SmoothingLength");
    let positions: Vec<f64> = snapshot
        .gas
        .coordinates
        .iter()
        .map(|coordinate| coordinate[0])
        .collect();
    let estimates = density_at_hsml_1d(
        &positions,
        &snapshot.gas.masses,
        smoothing_lengths,
        snapshot.header.box_size,
    )
    .expect("Rust density summation must accept the pinned state");

    let relative_errors: Vec<f64> = estimates
        .iter()
        .zip(expected_density)
        .map(|(estimate, expected)| (estimate.density - expected).abs() / expected)
        .collect();
    let particle_count =
        u32::try_from(relative_errors.len()).expect("sound-wave particle count fits in u32");
    let particle_count_float = f64::from(particle_count);
    let max_relative_error = relative_errors.iter().copied().fold(0.0, f64::max);
    let mean_relative_error = relative_errors.iter().sum::<f64>() / particle_count_float;
    let max_neighbor_deviation = estimates
        .iter()
        .map(|estimate| (estimate.effective_neighbors - 4.0).abs())
        .fold(0.0, f64::max);
    eprintln!(
        "public fixture density parity: mean={mean_relative_error:.12e}, \
         max={max_relative_error:.12e}, max |N_eff - 4|={max_neighbor_deviation:.12e}"
    );

    assert!(
        max_relative_error < 1.0e-10,
        "Rust density kernel diverged from pinned fixture: max relative error {max_relative_error}"
    );

    let initial_hsml =
        vec![2.0 * snapshot.header.box_size / particle_count_float; snapshot.gas.len()];
    let solved = solve_smoothing_lengths_1d(
        &positions,
        &snapshot.gas.masses,
        &initial_hsml,
        snapshot.header.box_size,
        4.0,
        1.0e-8,
    )
    .expect("adaptive Rust smoothing-length solve must converge");
    let solved_max_neighbor_deviation = solved
        .iter()
        .map(|particle| (particle.estimate.effective_neighbors - 4.0).abs())
        .fold(0.0, f64::max);
    let max_hsml_relative_difference = solved
        .iter()
        .zip(smoothing_lengths)
        .map(|(particle, legacy)| (particle.smoothing_length - legacy).abs() / legacy)
        .fold(0.0, f64::max);
    eprintln!(
        "adaptive Hsml parity: max |N_eff - 4|={solved_max_neighbor_deviation:.12e}, \
         max relative difference from accepted legacy Hsml={max_hsml_relative_difference:.12e}"
    );
    assert!(solved_max_neighbor_deviation <= 1.0e-8);
    assert!(
        max_hsml_relative_difference <= 1.0e-3,
        "adaptive Hsml diverged from the legacy-accepted state: \
         max relative difference {max_hsml_relative_difference}"
    );

    assert_public_gradients(
        &positions,
        expected_density,
        &snapshot.gas.velocities,
        &snapshot.gas.internal_energy,
        smoothing_lengths,
        snapshot.header.box_size,
    );
    assert_public_faces(
        &positions,
        &snapshot.gas.masses,
        expected_density,
        smoothing_lengths,
        snapshot.header.box_size,
    );
    assert_public_pair_fluxes(
        &positions,
        &snapshot.gas.masses,
        expected_density,
        &snapshot.gas.velocities,
        &snapshot.gas.internal_energy,
        smoothing_lengths,
        snapshot.header.box_size,
    );
    assert_corrected_c_first_step();
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
        assert_eq!(
            columns[0].parse::<usize>().expect("particle ID"),
            expected_id
        );
        table
            .positions
            .push(columns[1].parse().expect("x coordinate"));
        table
            .velocities
            .push(columns[2].parse().expect("x velocity"));
        table.densities.push(columns[3].parse().expect("density"));
        table
            .specific_internal_energy
            .push(columns[4].parse().expect("specific internal energy"));
        table
            .smoothing_lengths
            .push(columns[5].parse().expect("smoothing length"));
        table.masses.push(columns[6].parse().expect("mass"));
    }
    assert_eq!(table.positions.len(), 2048);
    table
}

fn max_relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let error = (actual - expected).abs() / expected.abs().max(1.0e-30);
            assert!(error.is_finite());
            error
        })
        .fold(0.0, f64::max)
}

fn max_absolute_error(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let error = (actual - expected).abs();
            assert!(error.is_finite());
            error
        })
        .fold(0.0, f64::max)
}

fn assert_corrected_c_first_step() {
    let initialized_path = std::env::var_os("GIZMO_SOUNDWAVE_C_T0")
        .expect("corrected-C initialized table is required");
    let expected_path = std::env::var_os("GIZMO_SOUNDWAVE_C_STEP1")
        .expect("corrected-C first-step table is required");
    let initialized = read_evolution_table(Path::new(&initialized_path));
    let expected = read_evolution_table(Path::new(&expected_path));
    let fixture_path =
        std::env::var_os("GIZMO_SOUNDWAVE_IC").expect("pinned public fixture is required");
    let initial = read_soundwave(fixture_path).expect("pinned public fixture must be valid");
    let initial_positions: Vec<f64> = initial
        .gas
        .coordinates
        .iter()
        .map(|coordinate| coordinate[0])
        .collect();
    let initial_velocities: Vec<f64> = initial
        .gas
        .velocities
        .iter()
        .map(|velocity| velocity[0])
        .collect();
    assert!(max_absolute_error(&initial_positions, &initialized.positions) < 1.0e-15);
    assert!(max_relative_error(&initial.gas.masses, &initialized.masses) < 1.0e-15);
    let mut state = MfmEvolvingState1d {
        positions: initial_positions.clone(),
        masses: initial.gas.masses.clone(),
        velocities: initial_velocities.clone(),
        specific_internal_energy: initial.gas.internal_energy.clone(),
        smoothing_lengths: initialized.smoothing_lengths.clone(),
        box_size: initial.header.box_size,
        gamma: 5.0 / 3.0,
    };
    let old_rates = mfm_spatial_rates_1d(state.as_view()).expect("initial Rust RHS must be valid");
    let timestep = 8192.0 * 1.5 / 536_870_912.0;
    let rust_half_velocity: Vec<f64> = initial_velocities
        .iter()
        .zip(&old_rates.acceleration)
        .map(|(velocity, acceleration)| velocity + 0.5 * timestep * acceleration)
        .collect();
    let new_rates = advance_mfm_kdk_1d(&mut state, &old_rates, timestep, 4.0, 0.05, 0.0)
        .expect("Rust first KDK step must succeed");
    let densities: Vec<f64> = density_at_hsml_1d(
        &state.positions,
        &state.masses,
        &state.smoothing_lengths,
        state.box_size,
    )
    .expect("endpoint density must be valid")
    .into_iter()
    .map(|estimate| estimate.density)
    .collect();

    let position_error = max_absolute_error(&state.positions, &expected.positions);
    let velocity_error = max_absolute_error(&state.velocities, &expected.velocities);
    let density_error = max_relative_error(&densities, &expected.densities);
    let energy_error = max_relative_error(
        &state.specific_internal_energy,
        &expected.specific_internal_energy,
    );
    let smoothing_error = max_relative_error(&state.smoothing_lengths, &expected.smoothing_lengths);
    let position_signal = max_absolute_error(&expected.positions, &initial_positions);
    let velocity_signal = max_absolute_error(&expected.velocities, &initial_velocities);
    let density_signal = max_relative_error(&expected.densities, &initialized.densities);
    let energy_signal = max_relative_error(
        &expected.specific_internal_energy,
        &initial.gas.internal_energy,
    );
    let smoothing_signal =
        max_relative_error(&expected.smoothing_lengths, &initialized.smoothing_lengths);
    let first_kick_error = max_absolute_error(&rust_half_velocity, &initialized.velocities);
    let first_kick_signal = max_absolute_error(&initialized.velocities, &initial_velocities);
    let rust_second_half_velocity: Vec<f64> = rust_half_velocity
        .iter()
        .zip(&new_rates.acceleration)
        .map(|(velocity, acceleration)| velocity + 0.5 * timestep * acceleration)
        .collect();
    let second_kick_error = max_absolute_error(&rust_second_half_velocity, &expected.velocities);
    let second_kick_signal = max_absolute_error(&expected.velocities, &initialized.velocities);
    eprintln!(
        "corrected-C first-step parity: max |dx|={position_error:.12e}, \
         max |dv|={velocity_error:.12e}, max rel density/u/Hsml=\
         {density_error:.12e}/{energy_error:.12e}/{smoothing_error:.12e}; \
         C step signals={position_signal:.12e}/{velocity_signal:.12e}/\
         {density_signal:.12e}/{energy_signal:.12e}/{smoothing_signal:.12e}; \
         first/second kick error-to-signal={first_kick_error:.12e}/\
         {first_kick_signal:.12e}, {second_kick_error:.12e}/{second_kick_signal:.12e}"
    );

    assert!(position_error < 0.01 * position_signal);
    assert!(density_error < 0.01 * density_signal);
    assert!(energy_error < 0.01 * energy_signal);
    assert!(smoothing_error < 1.0e-9);
    assert!(first_kick_error < 0.01 * first_kick_signal);
    assert!(
        second_kick_error > 0.1 * second_kick_signal,
        "GZ-0009 remains under investigation; do not silently claim full endpoint parity"
    );
}

fn assert_public_gradients(
    positions: &[f64],
    density: &[f64],
    velocities: &[[f64; 3]],
    internal_energy: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) {
    let velocity: Vec<f64> = velocities.iter().map(|components| components[0]).collect();
    let pressure: Vec<f64> = density
        .iter()
        .zip(internal_energy)
        .map(|(density, internal_energy)| (2.0 / 3.0) * density * internal_energy)
        .collect();
    for (name, values, shoot_tolerance, positivity_preserving) in [
        ("density", density, 0.0, true),
        ("velocity", velocity.as_slice(), 0.1, false),
        ("pressure", pressure.as_slice(), 0.1, true),
    ] {
        let gradients = gradients_at_hsml_1d(
            positions,
            values,
            smoothing_lengths,
            box_size,
            shoot_tolerance,
            positivity_preserving,
        )
        .expect("moving-least-squares gradient must accept the pinned state");
        let error = fundamental_mode_gradient_error(positions, values, &gradients, box_size);
        eprintln!("public fixture {name} gradient normalized L1 error={error:.12e}");
        assert!(error < 1.0e-4, "{name} gradient error {error}");
    }
}

fn fundamental_mode_gradient_error(
    positions: &[f64],
    values: &[f64],
    gradients: &[GradientEstimate],
    box_size: f64,
) -> f64 {
    let count = u32::try_from(positions.len()).expect("particle count fits in u32");
    let count_float = f64::from(count);
    let sine = 2.0
        * positions
            .iter()
            .zip(values)
            .map(|(position, value)| value * (std::f64::consts::TAU * position / box_size).sin())
            .sum::<f64>()
        / count_float;
    let cosine = 2.0
        * positions
            .iter()
            .zip(values)
            .map(|(position, value)| value * (std::f64::consts::TAU * position / box_size).cos())
            .sum::<f64>()
        / count_float;
    let amplitude = sine.hypot(cosine);
    positions
        .iter()
        .zip(gradients)
        .map(|(position, estimate)| {
            let wave_number = std::f64::consts::TAU / box_size;
            let phase = wave_number * position;
            let expected = wave_number * (sine * phase.cos() - cosine * phase.sin());
            (estimate.limited - expected).abs()
        })
        .sum::<f64>()
        / (count_float * (std::f64::consts::TAU / box_size) * amplitude)
}

fn assert_public_faces(
    positions: &[f64],
    masses: &[f64],
    density: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) {
    let inverse_moments = inverse_moments_1d(positions, smoothing_lengths, box_size)
        .expect("public geometry must have invertible moments");
    let point = |index| MeshlessPoint1d {
        position: positions[index],
        mass: masses[index],
        density: density[index],
        smoothing_length: smoothing_lengths[index],
        inverse_moment: inverse_moments[index],
    };
    let mut spatial_order: Vec<usize> = (0..positions.len()).collect();
    spatial_order.sort_unstable_by(|left, right| positions[*left].total_cmp(&positions[*right]));
    let max_area_deviation = spatial_order
        .iter()
        .enumerate()
        .map(|(order_index, &index)| {
            let neighbor = spatial_order[(order_index + 1) % spatial_order.len()];
            let face = meshless_face_geometry_1d(point(index), point(neighbor), box_size)
                .expect("adjacent public particles must form a valid face");
            (face.area - 1.0).abs()
        })
        .fold(0.0, f64::max);
    eprintln!("public fixture max 1-D face-area deviation={max_area_deviation:.12e}");
    assert!(
        max_area_deviation < 1.0e-6,
        "public face areas diverged from unit geometry: {max_area_deviation}"
    );
}

#[allow(clippy::too_many_lines)]
fn assert_public_pair_fluxes(
    positions: &[f64],
    masses: &[f64],
    density: &[f64],
    velocities: &[[f64; 3]],
    internal_energy: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) {
    let velocity: Vec<f64> = velocities.iter().map(|components| components[0]).collect();
    let pressure: Vec<f64> = density
        .iter()
        .zip(internal_energy)
        .map(|(density, internal_energy)| (2.0 / 3.0) * density * internal_energy)
        .collect();
    let density_gradients =
        gradients_at_hsml_1d(positions, density, smoothing_lengths, box_size, 0.0, true)
            .expect("public density gradients must be valid");
    let velocity_gradients = gradients_at_hsml_1d(
        positions,
        &velocity,
        smoothing_lengths,
        box_size,
        0.1,
        false,
    )
    .expect("public velocity gradients must be valid");
    let pressure_gradients =
        gradients_at_hsml_1d(positions, &pressure, smoothing_lengths, box_size, 0.1, true)
            .expect("public pressure gradients must be valid");
    let inverse_moments = inverse_moments_1d(positions, smoothing_lengths, box_size)
        .expect("public geometry must have invertible moments");
    let closure_errors = face_closure_errors_1d(positions, smoothing_lengths, box_size)
        .expect("public geometry must have finite face-closure errors");
    let density_estimates = density_at_hsml_1d(positions, masses, smoothing_lengths, box_size)
        .expect("public density factors must be valid");
    let geometry = |index: usize| MeshlessPoint1d {
        position: positions[index],
        mass: masses[index],
        density: density[index],
        smoothing_length: smoothing_lengths[index],
        inverse_moment: inverse_moments[index],
    };
    let reconstructed = |index: usize| ReconstructedPoint1d {
        primitive: PrimitiveState1d {
            density: density[index],
            velocity: velocity[index],
            pressure: pressure[index],
        },
        density_gradient: density_gradients[index].limited,
        velocity_gradient: velocity_gradients[index].limited,
        pressure_gradient: pressure_gradients[index].limited,
        face_closure_error: closure_errors[index],
    };
    let mut spatial_order: Vec<usize> = (0..positions.len()).collect();
    spatial_order.sort_unstable_by(|left, right| positions[*left].total_cmp(&positions[*right]));

    let mut max_momentum_swap_error = 0.0_f64;
    let mut max_raw_energy_swap_error = 0.0_f64;
    let mut max_corrected_energy_swap_error = 0.0_f64;
    let mut hllc_pairs = 0_usize;
    let mut entropic_pairs = 0_usize;
    for (order_index, &index) in spatial_order.iter().enumerate() {
        let neighbor = spatial_order[(order_index + 1) % spatial_order.len()];
        let face = meshless_face_geometry_1d(geometry(index), geometry(neighbor), box_size)
            .expect("adjacent public particles must form a valid face");
        let reverse_face = meshless_face_geometry_1d(geometry(neighbor), geometry(index), box_size)
            .expect("reversed public pair must form a valid face");
        let flux = mfm_pair_flux_1d(
            reconstructed(index),
            reconstructed(neighbor),
            face,
            5.0 / 3.0,
        )
        .expect("public adjacent pair flux must be valid");
        let reverse = mfm_pair_flux_1d(
            reconstructed(neighbor),
            reconstructed(index),
            reverse_face,
            5.0 / 3.0,
        )
        .expect("reversed public adjacent pair flux must be valid");
        assert!(flux.mass.abs() <= f64::EPSILON);
        assert!(reverse.mass.abs() <= f64::EPSILON);
        max_momentum_swap_error =
            max_momentum_swap_error.max((flux.momentum + reverse.momentum).abs());
        max_raw_energy_swap_error =
            max_raw_energy_swap_error.max((flux.energy + reverse.energy).abs());
        let distance = face.distance_from_i.abs() + face.distance_from_j.abs();
        let entropic_point = |particle: usize| EntropicPoint1d {
            velocity: velocity[particle],
            density: density[particle],
            pressure: pressure[particle],
            sound_speed: ((5.0 / 3.0) * pressure[particle] / density[particle]).sqrt(),
            volume: masses[particle] / density[particle],
            dhsml_factor: density_estimates[particle].dhsml_factor,
            kernel_radial_derivative: cubic_kernel_1d(distance, smoothing_lengths[particle])
                .expect("public pair kernel derivative must be valid")
                .radial_derivative,
            condition_number: 1.0,
            face_closure_error: closure_errors[particle],
        };
        let (corrected, selected) =
            apply_entropic_pdv_1d(flux, face, entropic_point(index), entropic_point(neighbor))
                .expect("public pair entropic correction must be valid");
        let (corrected_reverse, reverse_selected) = apply_entropic_pdv_1d(
            reverse,
            reverse_face,
            entropic_point(neighbor),
            entropic_point(index),
        )
        .expect("reversed public pair entropic correction must be valid");
        assert_eq!(selected, reverse_selected);
        entropic_pairs += usize::from(selected);
        max_corrected_energy_swap_error = max_corrected_energy_swap_error
            .max((corrected.energy + corrected_reverse.energy).abs());
        if flux.method == RiemannMethod::Hllc {
            hllc_pairs += 1;
        }
    }
    eprintln!(
        "public fixture pair fluxes: HLLC={hllc_pairs}/{}, entropic={entropic_pairs}/{}, \
         max swap momentum error={max_momentum_swap_error:.12e}, \
         max raw/corrected swap energy error={max_raw_energy_swap_error:.12e}/\
         {max_corrected_energy_swap_error:.12e}",
        positions.len(),
        positions.len(),
    );
    assert_eq!(hllc_pairs, positions.len());
    assert_eq!(entropic_pairs, positions.len());
    assert!(max_momentum_swap_error < 1.0e-12);
    assert!(max_raw_energy_swap_error < 1.0e-12);
    assert!(max_corrected_energy_swap_error < 1.0e-12);

    let state = MfmState1d {
        positions,
        masses,
        velocities: &velocity,
        specific_internal_energy: internal_energy,
        smoothing_lengths,
        box_size,
        gamma: 5.0 / 3.0,
    };
    let rates = mfm_spatial_rates_1d(state).expect("public full spatial RHS must be valid");
    let net_momentum_rate: f64 = rates.momentum.iter().sum();
    let net_energy_rate: f64 = rates.total_energy.iter().sum();
    let courant = global_courant_timestep_1d(state, &rates, 0.05)
        .expect("public Courant timestep must be valid");
    let timeline_step = SynchronizedTimeline1d::new(0.0, 1.5)
        .expect("public timeline must be valid")
        .select_step(courant, 1.0e-3)
        .expect("public initial timeline step must be valid");
    eprintln!(
        "public fixture spatial RHS: pairs={}, entropic={}, \
         net momentum/energy rate={net_momentum_rate:.12e}/{net_energy_rate:.12e}, \
         Courant/quantized dt={courant:.12e}/{:.12e} ({} ticks)",
        rates.pair_count, rates.entropic_pair_count, timeline_step.duration, timeline_step.ticks,
    );
    assert_eq!(rates.pair_count, 2 * positions.len());
    assert_eq!(rates.entropic_pair_count, rates.pair_count);
    assert!(net_momentum_rate.abs() < 1.0e-12);
    assert!(net_energy_rate.abs() < 1.0e-12);
    assert_eq!(timeline_step.ticks, 8192);
}
