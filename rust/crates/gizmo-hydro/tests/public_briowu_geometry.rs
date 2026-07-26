use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::PathBuf;

use gizmo_hydro::SynchronizedTimeline1d;
use gizmo_hydro::individual_timeline::IndividualParticleTimeline;
use gizmo_hydro::meshless_2d::{
    Box2d, Vector2, density_at_hsml_2d, face_closure_diagnostics_2d, inverse_moments_2d,
    solve_public_c_smoothing_lengths_from_seeds_2d,
};
use gizmo_hydro::mhd::Vector3;
use gizmo_hydro::mhd_evolution_2d::{
    DivergenceControl2d, MhdMfmState2d, begin_public_mhd_kdk_adaptive_2d,
    finish_public_mhd_kdk_adaptive_2d, mhd_mfm_spatial_rates_2d,
    public_mhd_particle_timestep_bounds_2d, quantize_public_mhd_initial_timebins_2d,
};
use gizmo_io::read_mhd_wave;

#[test]
#[ignore = "requires GIZMO_BRIOWU_FIXTURE; evaluates the full 50,176-particle 2-D sheet"]
#[allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]
fn public_briowu_fixture_exercises_real_two_dimensional_geometry() {
    let path = std::env::var_os("GIZMO_BRIOWU_FIXTURE")
        .map(PathBuf::from)
        .expect("set GIZMO_BRIOWU_FIXTURE to the pinned public HDF5 file");
    let snapshot = read_mhd_wave(path).expect("read the pinned Brio-Wu fixture");
    assert_eq!(snapshot.gas.len(), 50_176);
    let positions: Vec<_> = snapshot
        .gas
        .coordinates
        .iter()
        .map(|coordinate| Vector2::new(coordinate[0], coordinate[1]))
        .collect();
    let domain = Box2d::new(4.0, 0.25).unwrap();
    let density = density_at_hsml_2d(
        &positions,
        &snapshot.gas.masses,
        &snapshot.gas.smoothing_length,
        domain,
    )
    .expect("evaluate full 2-D MFM density");
    let moments = inverse_moments_2d(&positions, &snapshot.gas.smoothing_length, domain)
        .expect("invert every 2-D MLS moment");
    let closure = face_closure_diagnostics_2d(
        &positions,
        &snapshot.gas.masses,
        &snapshot.gas.smoothing_length,
        domain,
    )
    .expect("evaluate every 2-D meshless face and closure");

    assert_eq!(density.len(), 50_176);
    assert_eq!(moments.len(), 50_176);
    assert_eq!(closure.len(), 50_176);
    assert!(
        moments
            .iter()
            .all(|moment| moment.matrix.is_finite() && moment.condition_number.is_finite())
    );
    assert!(closure.iter().all(|value| {
        value.net_area_vector.is_finite()
            && value.relative_net_area.is_finite()
            && value.legacy_dimensionless_leak.is_finite()
    }));

    let range = |values: &[f64]| {
        values.iter().copied().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        )
    };
    let density_values: Vec<_> = density.iter().map(|value| value.density).collect();
    let neighbor_values: Vec<_> = density
        .iter()
        .map(|value| value.effective_neighbors)
        .collect();
    let condition_values: Vec<_> = moments.iter().map(|value| value.condition_number).collect();
    let relative_closure: Vec<_> = closure
        .iter()
        .map(|value| value.relative_net_area)
        .collect();
    let legacy_closure: Vec<_> = closure
        .iter()
        .map(|value| value.legacy_dimensionless_leak)
        .collect();
    let density_range = range(&density_values);
    let neighbor_range = range(&neighbor_values);
    let condition_range = range(&condition_values);
    let relative_closure_range = range(&relative_closure);
    let legacy_closure_range = range(&legacy_closure);
    assert!(
        density_range.0 > 0.124_999_9 && density_range.1 < 1.000_000_1,
        "{density_range:?}"
    );
    assert!(
        neighbor_range.0 > 16.017 && neighbor_range.1 < 16.019,
        "{neighbor_range:?}"
    );
    assert!(condition_range.0 >= 1.0 && condition_range.1 < 1.0001);
    assert!(relative_closure_range.1 < 3.0e-5);
    assert!(legacy_closure_range.1 < 2.0e-5);

    // Restart-0 invokes density three times before writing snapshot 000. Each
    // pass resets its brackets and retains the preceding accepted Hsml.
    let mut seeds = snapshot.gas.smoothing_length.clone();
    let mut solved = Vec::new();
    let corrected_c = std::env::var_os("GIZMO_BRIOWU_C_T0")
        .map(|path| read_mhd_wave(path).expect("read corrected-C Brio-Wu snapshot 000"));
    for _ in 0..3 {
        solved = solve_public_c_smoothing_lengths_from_seeds_2d(
            &positions,
            &snapshot.gas.masses,
            &seeds,
            domain,
            20.0,
            0.05,
        )
        .expect("run one public-C 2-D density/Hsml pass");
        seeds = solved
            .iter()
            .map(|particle| particle.smoothing_length)
            .collect();
    }
    let maximum_neighbor_error = solved
        .iter()
        .map(|particle| (particle.estimate.effective_neighbors - 20.0).abs())
        .fold(0.0_f64, f64::max);
    assert!(maximum_neighbor_error <= 0.05, "{maximum_neighbor_error}");

    if let Some(corrected_c) = corrected_c {
        assert_eq!(corrected_c.gas.ids, snapshot.gas.ids);
        let maximum_hsml_error = solved
            .iter()
            .zip(&corrected_c.gas.smoothing_length)
            .map(|(particle, expected)| (particle.smoothing_length - expected).abs())
            .fold(0.0_f64, f64::max);
        let maximum_density_error = solved
            .iter()
            .zip(&corrected_c.gas.density)
            .map(|(particle, expected)| (particle.estimate.density - expected).abs())
            .fold(0.0_f64, f64::max);
        let solved_hsml_range = solved.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), particle| {
                (
                    minimum.min(particle.smoothing_length),
                    maximum.max(particle.smoothing_length),
                )
            },
        );
        let corrected_hsml_range = range(&corrected_c.gas.smoothing_length);
        let solved_density_values: Vec<_> = solved
            .iter()
            .map(|particle| particle.estimate.density)
            .collect();
        let solved_density_range = range(&solved_density_values);
        let corrected_density_range = range(&corrected_c.gas.density);
        let solved_neighbor_values: Vec<_> = solved
            .iter()
            .map(|particle| particle.estimate.effective_neighbors)
            .collect();
        let solved_neighbor_range = range(&solved_neighbor_values);
        let corrected_neighbor_values: Vec<_> = corrected_c
            .gas
            .smoothing_length
            .iter()
            .zip(&corrected_c.gas.density)
            .zip(&corrected_c.gas.masses)
            .map(|((&hsml, &density), &mass)| PI * hsml * hsml * density / mass)
            .collect();
        let corrected_neighbor_range = range(&corrected_neighbor_values);
        eprintln!(
            "corrected-C maximum errors: H={maximum_hsml_error} density={maximum_density_error}; \
             H ranges {solved_hsml_range:?} vs {corrected_hsml_range:?}; density ranges \
             {solved_density_range:?} vs {corrected_density_range:?}; neighbors \
             {solved_neighbor_range:?} vs {corrected_neighbor_range:?}"
        );
        // These are exact ranges from the pinned corrected-C snapshot. The C
        // density accumulators are single precision and MPI-order-dependent,
        // so the f64 Rust solve need only land in the same accepted ±0.05
        // neighbor band, not reproduce an arbitrary reduction order bitwise.
        assert!((corrected_hsml_range.0 - 0.011_262_498_313_321_88).abs() < 5.0e-16);
        assert!((corrected_hsml_range.1 - 0.011_262_793_369_662_273).abs() < 5.0e-16);
        assert!((corrected_density_range.0 - 0.124_765_691_784_384_54).abs() < 5.0e-15);
        assert!((corrected_density_range.1 - 0.998_132_302_614_668_3).abs() < 5.0e-15);
        assert!(maximum_hsml_error < 1.1e-5, "{maximum_hsml_error}");
        assert!(maximum_density_error < 8.0e-6, "{maximum_density_error}");
    }

    // Exercise the actual production 2-D MFM/MHD RHS over every public
    // particle. This prevents the initialization gate from passing with a
    // no-op or a hidden one-dimensional projection.
    let velocities: Vec<_> = snapshot
        .gas
        .velocities
        .iter()
        .map(|value| Vector3::new(value[0], value[1], value[2]))
        .collect();
    let magnetic: Vec<_> = snapshot
        .gas
        .magnetic_field
        .iter()
        .map(|value| Vector3::new(value[0], value[1], value[2]))
        .collect();
    let state = MhdMfmState2d::from_primitive(
        positions.clone(),
        snapshot.gas.masses.clone(),
        velocities,
        snapshot.gas.internal_energy.clone(),
        solved
            .iter()
            .map(|particle| particle.smoothing_length)
            .collect(),
        &magnetic,
        &vec![0.0; snapshot.gas.len()],
        domain,
        2.0,
    )
    .expect("construct the full adaptive-H 2-D MHD state");
    let rates = mhd_mfm_spatial_rates_2d(
        &state,
        DivergenceControl2d {
            hyperbolic_sigma: 1.0,
            parabolic_sigma: 1.0,
            ..Default::default()
        },
    )
    .expect("evaluate the full public Brio-Wu 2-D MHD RHS");
    assert!(rates.pair_count > snapshot.gas.len());
    assert!(
        rates
            .acceleration
            .iter()
            .any(|value| value.x.abs() > 1.0e-8 || value.y.abs() > 1.0e-8)
    );
    assert!(
        rates.acceleration.iter().all(|value| value.is_finite())
            && rates
                .specific_internal_energy
                .iter()
                .all(|value| value.is_finite())
            && rates.magnetic_volume.iter().all(|value| value.is_finite())
    );
    assert!(rates.entropic_pair_count > 0);
    let particle_bounds = public_mhd_particle_timestep_bounds_2d(&state, &rates, 0.2, 0.01)
        .expect("evaluate every enabled public Brio-Wu particle timestep bound");
    let physical_bound = particle_bounds
        .iter()
        .map(|bound| bound.selected)
        .fold(f64::INFINITY, f64::min);
    let initial_timebins =
        quantize_public_mhd_initial_timebins_2d(&particle_bounds, 0.0, 0.2, 0.04)
            .expect("assign the initial public-C integer time bins");
    let timebin_counts =
        initial_timebins
            .iter()
            .fold(BTreeMap::<u32, usize>::new(), |mut counts, timebin| {
                *counts.entry(timebin.time_bin).or_default() += 1;
                counts
            });
    assert_eq!(initial_timebins.len(), snapshot.gas.len());
    assert!(
        initial_timebins
            .iter()
            .all(|timebin| timebin.ticks.is_power_of_two())
    );
    assert_eq!(
        timebin_counts,
        BTreeMap::from([(49_u32, 25_088_usize), (50, 25_088)])
    );
    let mut individual_timeline = IndividualParticleTimeline::from_initial_steps(
        0.0,
        0.2,
        &initial_timebins
            .iter()
            .map(|timebin| timebin.ticks)
            .collect::<Vec<_>>(),
    )
    .expect("construct the exact initial active-set schedule");
    assert!(
        individual_timeline
            .active_mask()
            .into_iter()
            .all(|active| active)
    );
    let first_active = individual_timeline
        .advance_to_next_sync()
        .expect("select the earliest occupied time-bin endpoint");
    assert_eq!(
        first_active.into_iter().filter(|&active| active).count(),
        25_088
    );
    let first_step = SynchronizedTimeline1d::new(0.0, 0.2)
        .unwrap()
        .select_step(physical_bound, 0.04)
        .unwrap();
    assert_eq!(
        initial_timebins
            .iter()
            .map(|timebin| timebin.duration)
            .fold(f64::INFINITY, f64::min)
            .to_bits(),
        first_step.duration.to_bits()
    );
    assert_eq!(first_step.duration.to_bits(), (0.2_f64 / 2048.0).to_bits());
    if std::env::var_os("GIZMO_BRIOWU_ADVANCE_ONE_STEP").is_some() {
        let mut step = begin_public_mhd_kdk_adaptive_2d(
            &state,
            &rates,
            first_step.duration,
            0.0,
            DivergenceControl2d::default(),
            20.0,
            0.05,
            0.2,
        )
        .expect("apply the context-free initial force in the first half-kick");
        let initial_output = step
            .drift_state(0.0)
            .expect("expose the corrected-C t=0 output phase");
        let maximum_abs_vx = initial_output
            .actual_velocities
            .iter()
            .map(|velocity| velocity.x.abs())
            .fold(0.0_f64, f64::max);
        let maximum_abs_vy = initial_output
            .actual_velocities
            .iter()
            .map(|velocity| velocity.y.abs())
            .fold(0.0_f64, f64::max);
        // The x maximum is half the corrected-C value while y already agrees.
        // This synchronized runner applies the minimum cadence to every
        // particle; corrected C therefore assigns the x-extremum particles a
        // two-times-larger individual bin while the y extrema stay in the
        // minimum bin.
        assert!(
            (0.009..0.0092).contains(&maximum_abs_vx),
            "{maximum_abs_vx}"
        );
        assert!((0.026..0.027).contains(&maximum_abs_vy), "{maximum_abs_vy}");
        let advanced = finish_public_mhd_kdk_adaptive_2d(step)
            .expect("finish one phase-aware public Brio-Wu KDK step");
        assert!(
            advanced
                .state
                .positions
                .iter()
                .zip(&state.positions)
                .any(|(actual, initial)| (*actual - *initial).norm() > 0.0)
        );
        assert!(
            advanced
                .state
                .specific_internal_energy
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
    }
    eprintln!(
        "density={:?} neighbors={:?} condition={:?} relative_closure={:?} \
         legacy_closure={:?} solved_neighbor_error={maximum_neighbor_error} rhs_pairs={} \
         entropic_pairs={} physical_dt={physical_bound} synchronized_dt={} timebins={:?}",
        density_range,
        neighbor_range,
        condition_range,
        relative_closure_range,
        legacy_closure_range,
        rates.pair_count,
        rates.entropic_pair_count,
        first_step.duration,
        timebin_counts,
    );
}
