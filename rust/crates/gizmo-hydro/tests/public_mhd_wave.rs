#![allow(clippy::cast_precision_loss)]

use std::f64::consts::{SQRT_2, TAU};

use gizmo_hydro::mhd::{IdealMhdPrimitive1d, Vector3, fast_magnetosonic_speed};
use gizmo_io::read_mhd_wave;

const GAMMA: f64 = 5.0 / 3.0;
const PARTICLE_COUNT: usize = 2048;
const NORMALIZED_AMPLITUDE: f64 = 4.472_135_954_999_579e-7;

#[test]
#[ignore = "requires GIZMO_MHD_WAVE_IC; run via validation oracle script"]
fn public_fixture_pins_units_sorting_and_fast_eigenvector() {
    let path = std::env::var_os("GIZMO_MHD_WAVE_IC")
        .expect("GIZMO_MHD_WAVE_IC must identify the pinned fixture");
    let snapshot = read_mhd_wave(path).expect("pinned MHD-wave fixture must be valid");
    assert_eq!(snapshot.gas.len(), PARTICLE_COUNT);
    assert_eq!(
        snapshot.gas.ids,
        (0_u64..PARTICLE_COUNT as u64).collect::<Vec<_>>()
    );

    let count = PARTICLE_COUNT as f64;
    let mean_density = snapshot.gas.density.iter().sum::<f64>() / count;
    let mean_internal_energy = snapshot.gas.internal_energy.iter().sum::<f64>() / count;
    let mean_magnetic = mean_vector(&snapshot.gas.magnetic_field);
    assert!((mean_density - 1.0).abs() < 2.0e-15);
    assert!((mean_internal_energy - 0.9).abs() < 6.0e-13);
    assert!((mean_magnetic[0] - 1.0).abs() < 1.0e-12);
    assert!((mean_magnetic[1] - SQRT_2).abs() < 1.0e-12);
    assert!((mean_magnetic[2] - 0.5).abs() < 1.0e-12);

    let base = IdealMhdPrimitive1d {
        density: 1.0,
        velocity: Vector3::ZERO,
        gas_pressure: 0.6,
        magnetic: Vector3::new(1.0, SQRT_2, 0.5),
        cleaning_scalar: 0.0,
    };
    assert_eq!(
        fast_magnetosonic_speed(base, GAMMA)
            .expect("the pinned base state is physical")
            .to_bits(),
        2.0_f64.to_bits()
    );

    let x: Vec<f64> = snapshot
        .gas
        .coordinates
        .iter()
        .map(|coordinate| coordinate[0])
        .collect();
    for (index, &position) in x.iter().enumerate() {
        let expected = (index as f64 + 0.5) / count;
        assert_eq!(position.to_bits(), expected.to_bits());
    }
    let density_amplitude = sine_amplitude(&snapshot.gas.density, &x, 1.0);
    let velocity_x: Vec<f64> = snapshot
        .gas
        .velocities
        .iter()
        .map(|velocity| velocity[0])
        .collect();
    let magnetic_y: Vec<f64> = snapshot
        .gas
        .magnetic_field
        .iter()
        .map(|magnetic| magnetic[1])
        .collect();
    assert!((density_amplitude - NORMALIZED_AMPLITUDE).abs() < 2.0e-15);
    assert!((sine_amplitude(&velocity_x, &x, 0.0) + 2.0 * NORMALIZED_AMPLITUDE).abs() < 2.0e-15);
    assert!(
        (sine_amplitude(&magnetic_y, &x, SQRT_2) - 4.0 * SQRT_2 * NORMALIZED_AMPLITUDE / 3.0).abs()
            < 2.0e-15
    );

    // Restart-0 C initialization discards these diagnostic IC columns and
    // starts the Dedner state at zero. Keeping them distinct in I/O prevents a
    // caller from accidentally treating stored divB roundoff as Phi.
    assert!(
        snapshot
            .gas
            .cleaning_phi
            .as_ref()
            .is_some_and(|phi| { phi.iter().all(|&value| value == 0.0) })
    );
    assert!(
        snapshot
            .gas
            .divergence_of_magnetic_field
            .as_ref()
            .is_some_and(|divergence| divergence.iter().any(|&value| value != 0.0))
    );
}

fn mean_vector(values: &[[f64; 3]]) -> [f64; 3] {
    let count = values.len() as f64;
    let sum = values.iter().fold([0.0; 3], |mut sum, value| {
        for component in 0..3 {
            sum[component] += value[component];
        }
        sum
    });
    [sum[0] / count, sum[1] / count, sum[2] / count]
}

fn sine_amplitude(values: &[f64], positions: &[f64], mean: f64) -> f64 {
    let count = values.len() as f64;
    2.0 * values
        .iter()
        .zip(positions)
        .map(|(&value, &position)| (value - mean) * (TAU * position).sin())
        .sum::<f64>()
        / count
}
