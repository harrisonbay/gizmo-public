use gizmo_hydro::{density_at_hsml_1d, solve_smoothing_lengths_1d};
use gizmo_io::read_soundwave;

#[test]
#[ignore = "requires GIZMO_SOUNDWAVE_IC; run via validation oracle script"]
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
}
