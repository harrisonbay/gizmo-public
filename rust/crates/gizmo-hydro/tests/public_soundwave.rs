use gizmo_hydro::{
    EntropicPoint1d, GradientEstimate, MeshlessPoint1d, MfmState1d, PrimitiveState1d,
    ReconstructedPoint1d, RiemannMethod, apply_entropic_pdv_1d, cubic_kernel_1d,
    density_at_hsml_1d, face_closure_errors_1d, gradients_at_hsml_1d, inverse_moments_1d,
    meshless_face_geometry_1d, mfm_pair_flux_1d, mfm_spatial_rates_1d, solve_smoothing_lengths_1d,
};
use gizmo_io::read_soundwave;

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

    let rates = mfm_spatial_rates_1d(MfmState1d {
        positions,
        masses,
        velocities: &velocity,
        specific_internal_energy: internal_energy,
        smoothing_lengths,
        box_size,
        gamma: 5.0 / 3.0,
    })
    .expect("public full spatial RHS must be valid");
    let net_momentum_rate: f64 = rates.momentum.iter().sum();
    let net_energy_rate: f64 = rates.total_energy.iter().sum();
    eprintln!(
        "public fixture spatial RHS: pairs={}, entropic={}, \
         net momentum/energy rate={net_momentum_rate:.12e}/{net_energy_rate:.12e}",
        rates.pair_count, rates.entropic_pair_count
    );
    assert_eq!(rates.pair_count, 2 * positions.len());
    assert_eq!(rates.entropic_pair_count, rates.pair_count);
    assert!(net_momentum_rate.abs() < 1.0e-12);
    assert!(net_energy_rate.abs() < 1.0e-12);
}
