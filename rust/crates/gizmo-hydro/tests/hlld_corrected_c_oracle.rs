use gizmo_hydro::mhd::{
    hlld_riemann, FluxFrame1d, HlldOptions, IdealMhdPrimitive1d, MhdRiemannMethod, Vector3,
};

const GAMMA: f64 = 5.0 / 3.0;
const ORACLE: &str = include_str!("../../../../validation/oracles/mhd_wave/hlld_flux_oracle.csv");

#[test]
fn hlld_flux_and_dedner_interface_match_corrected_c() {
    for (line_number, line) in ORACLE.lines().enumerate().skip(1) {
        let columns: Vec<&str> = line.split(',').collect();
        assert_eq!(columns.len(), 34, "oracle row {}", line_number + 1);
        let case = columns[0];
        let values: Vec<f64> = columns[1..]
            .iter()
            .map(|value| {
                value
                    .parse::<f64>()
                    .unwrap_or_else(|error| panic!("{case}: invalid oracle value {value}: {error}"))
            })
            .collect();
        let primitive = |offset: usize| IdealMhdPrimitive1d {
            density: values[offset],
            gas_pressure: values[offset + 1],
            velocity: Vector3::new(values[offset + 2], values[offset + 3], values[offset + 4]),
            magnetic: Vector3::new(values[offset + 5], values[offset + 6], values[offset + 7]),
            cleaning_scalar: values[offset + 8],
        };
        let result = hlld_riemann(
            primitive(0),
            primitive(9),
            GAMMA,
            HlldOptions {
                frame: FluxFrame1d::Contact,
                maximum_star_total_pressure: Some(1.0e100),
                ..HlldOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("{case}: Rust HLLD solve failed: {error}"));
        assert_eq!(result.method, MhdRiemannMethod::Hlld, "{case}");

        let actual = [
            result.flux.mass,
            result.flux.momentum.x,
            result.flux.momentum.y,
            result.flux.momentum.z,
            result.flux.total_energy,
            result.flux.magnetic.x,
            result.flux.magnetic.y,
            result.flux.magnetic.z,
            result.face_velocity,
            result.star_total_pressure,
            result.corrected_normal_b,
            result.phi_mean,
            result.phi_db,
            result.fast_speed_left,
            result.fast_speed_right,
        ];
        let expected = &values[18..33];
        for (component, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_close(case, component, actual, expected);
        }
    }
}

fn assert_close(case: &str, component: usize, actual: f64, expected: f64) {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    let tolerance = 2.0e-13 * scale;
    assert!(
        (actual - expected).abs() <= tolerance,
        "{case}: component {component} differs: Rust={actual:.17e}, C={expected:.17e}, \
         tolerance={tolerance:.3e}"
    );
}
