//! One-dimensional gas-grain drag operators.
//!
//! These routines preserve the operator split used by
//! `solids/grain_physics.c`: gas properties are first interpolated to a grain,
//! the nonlinear Epstein equation is integrated over a finite step, and the
//! opposite grain impulse is distributed to gas with the same kernel. Drag
//! heating is intentionally absent because the public C path leaves that term
//! disabled.
//!
//! Arithmetic and stored values are `f64`, consistently with the rest of the
//! Rust hydro port. The public dustywave configuration stores the analogous C
//! fields as `f32`; an end-to-end C differential must therefore treat its
//! field-store rounding as an oracle boundary instead of attributing those
//! rounding differences to the drag equations.

use std::error::Error;
use std::fmt;

use crate::{HydroError, cubic_kernel_1d, periodic_displacement_1d};

const EPSTEIN_SPEED_COEFFICIENT: f64 = 0.469_993;
const EPSTEIN_STOPPING_COEFFICIENT: f64 = 1.595_77;
const SATURATED_STOPPING_STEP: f64 = 100.0;

/// Gas state needed by the one-dimensional grain operators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrainGasPoint1d {
    pub position: f64,
    pub mass: f64,
    /// Gas velocity sampled by interpolation; pass the KDK predictor value
    /// when reproducing the public C scheduling.
    pub velocity: f64,
    /// Gas internal energy sampled by interpolation; pass the predictor value
    /// when reproducing the public C scheduling.
    pub specific_internal_energy: f64,
}

/// Mutable one-dimensional grain state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrainPoint1d {
    pub position: f64,
    pub mass: f64,
    pub velocity: f64,
    /// Full compact-support radius of the grain's gas interpolation kernel.
    pub smoothing_length: f64,
    /// Physical grain radius expressed in code-length units.
    pub radius: f64,
}

/// Material and equation-of-state parameters for Epstein drag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpsteinDragParameters {
    pub gamma: f64,
    /// Intrinsic material density of a grain in code-density units.
    pub grain_internal_density: f64,
}

/// Kernel-interpolated gas properties at a grain position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolatedGrainGas1d {
    pub density: f64,
    pub velocity: f64,
    pub specific_internal_energy: f64,
    pub neighbor_count: usize,
}

/// The finite-step Epstein update before gas backreaction is applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpsteinImpulse1d {
    pub inverse_stopping_time: f64,
    pub dimensionless_relative_speed: f64,
    pub velocity_relaxation_fraction: f64,
    pub grain_velocity_delta: f64,
    pub grain_momentum_delta: f64,
}

/// Result of applying one grain drag step and its gas backreaction.
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedEpsteinStep1d {
    pub interpolated_gas: InterpolatedGrainGas1d,
    pub impulse: EpsteinImpulse1d,
    pub gas_velocity_deltas: Vec<f64>,
    /// Change of total gas-plus-grain momentum from floating-point summation.
    pub momentum_residual: f64,
}

/// Per-grain result from a frozen-gas drag calculation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComputedEpsteinGrain1d {
    pub interpolated_gas: InterpolatedGrainGas1d,
    pub impulse: EpsteinImpulse1d,
}

/// Frozen-gas impulses ready for an integrator to schedule and apply.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputedEpsteinBatch1d {
    pub grains: Vec<ComputedEpsteinGrain1d>,
    pub gas_velocity_deltas: Vec<f64>,
    /// Change of total gas-plus-grain momentum from floating-point summation.
    pub momentum_residual: f64,
}

/// Invalid input or non-finite result from a grain operator.
#[derive(Clone, Debug, PartialEq)]
pub enum GrainError {
    InvalidInput {
        field: &'static str,
        value: f64,
    },
    InvalidGasPoint {
        index: usize,
        field: &'static str,
        value: f64,
    },
    NonFiniteResult {
        field: &'static str,
        value: f64,
    },
    CoincidentGasNeighbor {
        grain_index: usize,
        gas_index: usize,
    },
    MissingBatchResult,
    Hydro(HydroError),
}

impl fmt::Display for GrainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field, value } => {
                write!(formatter, "invalid grain input {field}={value}")
            }
            Self::InvalidGasPoint {
                index,
                field,
                value,
            } => write!(formatter, "gas point {index} has invalid {field}={value}"),
            Self::NonFiniteResult { field, value } => {
                write!(
                    formatter,
                    "grain operator produced non-finite {field}={value}"
                )
            }
            Self::CoincidentGasNeighbor {
                grain_index,
                gas_index,
            } => write!(
                formatter,
                "grain {grain_index} and gas point {gas_index} are coincident; \
                 the public C stencil cannot conserve their drag impulse"
            ),
            Self::MissingBatchResult => {
                write!(formatter, "one-grain batch returned no grain result")
            }
            Self::Hydro(source) => write!(formatter, "grain kernel operation failed: {source}"),
        }
    }
}

impl Error for GrainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Hydro(source) => Some(source),
            _ => None,
        }
    }
}

impl From<HydroError> for GrainError {
    fn from(value: HydroError) -> Self {
        Self::Hydro(value)
    }
}

/// Interpolate density, velocity, and specific internal energy from gas to a
/// grain with the grain's smoothing length.
///
/// Density is `sum(m_j W_j)`. Velocity and internal energy are divided by that
/// same density after accumulating `m_j W_j q_j`, exactly as in
/// `hydro/density.c`. The public C loop excludes a zero-separation neighbor
/// from the velocity and energy numerators while retaining it in density; this
/// routine deliberately preserves that edge-case behavior.
///
/// # Errors
///
/// Returns an error for invalid box, grain, or gas inputs, or a non-finite
/// kernel result.
pub fn interpolate_gas_to_grain_1d(
    grain_position: f64,
    smoothing_length: f64,
    gas: &[GrainGasPoint1d],
    box_size: f64,
) -> Result<InterpolatedGrainGas1d, GrainError> {
    validate_position("grain_position", grain_position, box_size)?;
    validate_positive("smoothing_length", smoothing_length)?;
    validate_positive("box_size", box_size)?;
    validate_gas(gas, box_size)?;

    let mut density = 0.0;
    let mut weighted_velocity = 0.0;
    let mut weighted_internal_energy = 0.0;
    let mut neighbor_count = 0;
    for point in gas {
        let displacement = periodic_displacement_1d(grain_position, point.position, box_size)?;
        let radius = displacement.abs();
        if radius >= smoothing_length {
            continue;
        }
        let weight = cubic_kernel_1d(radius, smoothing_length)?.weight;
        let mass_weight = point.mass * weight;
        density += mass_weight;
        neighbor_count += 1;
        if radius > 0.0 {
            weighted_velocity += mass_weight * point.velocity;
            weighted_internal_energy += mass_weight * point.specific_internal_energy;
        }
    }

    let result = if density > 0.0 {
        InterpolatedGrainGas1d {
            density,
            velocity: weighted_velocity / density,
            specific_internal_energy: weighted_internal_energy / density,
            neighbor_count,
        }
    } else {
        InterpolatedGrainGas1d {
            density: 0.0,
            velocity: 0.0,
            specific_internal_energy: 0.0,
            neighbor_count: 0,
        }
    };
    validate_interpolated(result)?;
    Ok(result)
}

/// Integrate the nonlinear Epstein drag equation over one finite timestep.
///
/// This is the uncharged, non-Stokes specialization of
/// `apply_grain_dragforce`. Both velocities and all material quantities must
/// use a consistent system of code units.
///
/// # Errors
///
/// Returns an error for invalid state or parameters, or a non-finite result.
pub fn epstein_drag_impulse_1d(
    grain_mass: f64,
    grain_velocity: f64,
    grain_radius: f64,
    gas: InterpolatedGrainGas1d,
    timestep: f64,
    parameters: EpsteinDragParameters,
) -> Result<EpsteinImpulse1d, GrainError> {
    validate_positive("grain_mass", grain_mass)?;
    validate_finite("grain_velocity", grain_velocity)?;
    validate_positive("grain_radius", grain_radius)?;
    validate_nonnegative("timestep", timestep)?;
    validate_positive("gas_density", gas.density)?;
    validate_positive("gas_specific_internal_energy", gas.specific_internal_energy)?;
    validate_finite("gas_velocity", gas.velocity)?;
    if !parameters.gamma.is_finite() || parameters.gamma <= 1.0 {
        return Err(GrainError::InvalidInput {
            field: "gamma",
            value: parameters.gamma,
        });
    }
    validate_positive("grain_internal_density", parameters.grain_internal_density)?;

    let relative_velocity = gas.velocity - grain_velocity;
    let relative_speed = relative_velocity.abs();
    if timestep == 0.0 || relative_speed == 0.0 {
        return Ok(EpsteinImpulse1d {
            inverse_stopping_time: 0.0,
            dimensionless_relative_speed: 0.0,
            velocity_relaxation_fraction: 0.0,
            grain_velocity_delta: 0.0,
            grain_momentum_delta: 0.0,
        });
    }

    let gamma = parameters.gamma;
    let sound_speed = (gamma * (gamma - 1.0) * gas.specific_internal_energy).sqrt();
    let dimensionless_relative_speed =
        EPSTEIN_SPEED_COEFFICIENT * gamma.sqrt() * relative_speed / sound_speed;
    let inverse_stopping_time =
        EPSTEIN_STOPPING_COEFFICIENT / gamma.sqrt() * gas.density * sound_speed
            / (grain_radius * parameters.grain_internal_density);
    let stopping_step = timestep * inverse_stopping_time;

    let final_dimensionless_speed = if stopping_step < SATURATED_STOPPING_STEP {
        let x = dimensionless_relative_speed;
        if x >= 1.0 {
            // Since q=exp(-asinh(1/x)) and x=2q/(1-q^2), evolving
            // q -> q*exp(-s) gives this form. It remains accurate when both
            // 1/x and s are smaller than an ulp at one, where q would round
            // to exactly one.
            1.0 / (stopping_step + (1.0 / x).asinh()).sinh()
        } else {
            // Algebraically identical to C's C1/C2 expression, but this
            // bounded variable avoids cancellation for small x.
            let transformed_speed = x / (1.0 + x.hypot(1.0));
            let evolved_speed = transformed_speed * (-stopping_step).exp();
            2.0 * evolved_speed / (1.0 - evolved_speed * evolved_speed)
        }
    } else {
        0.0
    };
    let velocity_relaxation_fraction =
        1.0 - final_dimensionless_speed / dimensionless_relative_speed;
    let grain_velocity_delta = velocity_relaxation_fraction * relative_velocity;
    let grain_momentum_delta = grain_mass * grain_velocity_delta;

    for (field, value) in [
        ("sound_speed", sound_speed),
        ("dimensionless_relative_speed", dimensionless_relative_speed),
        ("inverse_stopping_time", inverse_stopping_time),
        ("velocity_relaxation_fraction", velocity_relaxation_fraction),
        ("grain_velocity_delta", grain_velocity_delta),
        ("grain_momentum_delta", grain_momentum_delta),
    ] {
        if !value.is_finite() {
            return Err(GrainError::NonFiniteResult { field, value });
        }
    }

    Ok(EpsteinImpulse1d {
        inverse_stopping_time,
        dimensionless_relative_speed,
        velocity_relaxation_fraction,
        grain_velocity_delta,
        grain_momentum_delta,
    })
}

/// Distribute the opposite of a grain impulse to its gas neighbors.
///
/// Each gas velocity increment is `-W_j * delta_p_grain / rho_gas`, matching
/// `grain_backrx_evaluate`. Zero-separation neighbors are excluded just as in
/// the C loop.
///
/// # Errors
///
/// Returns an error for invalid inputs or a non-finite velocity increment.
pub fn distribute_grain_backreaction_1d(
    grain_position: f64,
    smoothing_length: f64,
    gas_density: f64,
    grain_momentum_delta: f64,
    gas: &[GrainGasPoint1d],
    box_size: f64,
) -> Result<Vec<f64>, GrainError> {
    validate_position("grain_position", grain_position, box_size)?;
    validate_positive("smoothing_length", smoothing_length)?;
    validate_positive("gas_density", gas_density)?;
    validate_finite("grain_momentum_delta", grain_momentum_delta)?;
    validate_gas(gas, box_size)?;

    let mut deltas = vec![0.0; gas.len()];
    for (index, point) in gas.iter().enumerate() {
        let radius = periodic_displacement_1d(grain_position, point.position, box_size)?.abs();
        if radius == 0.0 || radius >= smoothing_length {
            continue;
        }
        let weight = cubic_kernel_1d(radius, smoothing_length)?.weight;
        let delta = -weight * grain_momentum_delta / gas_density;
        if !delta.is_finite() {
            return Err(GrainError::NonFiniteResult {
                field: "gas_velocity_delta",
                value: delta,
            });
        }
        deltas[index] = delta;
    }
    Ok(deltas)
}

/// Apply one finite Epstein drag step to a grain and its gas stencil.
///
/// Gas internal energy is not modified. This mirrors the public C build, where
/// drag-heating terms in the backreaction loop are commented out. For more
/// than one grain, use [`apply_epstein_drag_steps_1d`]: sequential calls to
/// this one-grain wrapper would not preserve the C operator's frozen-gas
/// gather/compute/scatter ordering.
///
/// # Errors
///
/// Returns an error for invalid inputs or a non-finite intermediate result.
pub fn apply_epstein_drag_step_1d(
    grain: &mut GrainPoint1d,
    gas: &mut [GrainGasPoint1d],
    box_size: f64,
    timestep: f64,
    parameters: EpsteinDragParameters,
) -> Result<AppliedEpsteinStep1d, GrainError> {
    let mut system = apply_epstein_drag_steps_1d(
        std::slice::from_mut(grain),
        gas,
        box_size,
        timestep,
        parameters,
    )?;
    let Some(grain_result) = system.grains.pop() else {
        return Err(GrainError::MissingBatchResult);
    };
    Ok(AppliedEpsteinStep1d {
        interpolated_gas: grain_result.interpolated_gas,
        impulse: grain_result.impulse,
        gas_velocity_deltas: system.gas_velocity_deltas,
        momentum_residual: system.momentum_residual,
    })
}

/// Apply Epstein drag to all grains with the public C operator ordering.
///
/// All gas properties are gathered before any velocity is changed. Grain
/// impulses are then computed from that frozen state, all gas backreaction
/// increments are accumulated, and the grain and gas velocities are committed
/// together. This makes results independent of grain iteration order apart
/// from floating-point summation order.
///
/// Gas internal energy is not modified.
///
/// # Errors
///
/// Returns an error for invalid inputs or a non-finite intermediate result.
/// No particle is mutated unless the entire batch validates successfully.
pub fn apply_epstein_drag_steps_1d(
    grains: &mut [GrainPoint1d],
    gas: &mut [GrainGasPoint1d],
    box_size: f64,
    timestep: f64,
    parameters: EpsteinDragParameters,
) -> Result<ComputedEpsteinBatch1d, GrainError> {
    let computed = compute_epstein_drag_batch_1d(grains, gas, box_size, timestep, parameters)?;

    let mut grain_velocities_new = Vec::with_capacity(grains.len());
    for (grain, result) in grains.iter().zip(&computed.grains) {
        let velocity_new = grain.velocity + result.impulse.grain_velocity_delta;
        if !velocity_new.is_finite() {
            return Err(GrainError::NonFiniteResult {
                field: "grain_velocity_new",
                value: velocity_new,
            });
        }
        grain_velocities_new.push(velocity_new);
    }
    let mut gas_velocities_new = Vec::with_capacity(gas.len());
    for (point, velocity_delta) in gas.iter().zip(&computed.gas_velocity_deltas) {
        let velocity_new = point.velocity + velocity_delta;
        if !velocity_new.is_finite() {
            return Err(GrainError::NonFiniteResult {
                field: "gas_velocity_new",
                value: velocity_new,
            });
        }
        gas_velocities_new.push(velocity_new);
    }

    for (grain, velocity_new) in grains.iter_mut().zip(grain_velocities_new) {
        grain.velocity = velocity_new;
    }
    for (point, velocity_new) in gas.iter_mut().zip(gas_velocities_new) {
        point.velocity = velocity_new;
    }
    Ok(computed)
}

/// Gather and compute a complete drag batch without mutating particle state.
///
/// This is the scheduling boundary required by the C leapfrog: callers can add
/// each returned grain `grain_velocity_delta / timestep` to the appropriate
/// acceleration channel while applying the gas backreaction deltas to both
/// stored and predicted gas velocities at the matching kick boundary.
///
/// # Errors
///
/// Returns an error for invalid inputs or a non-finite intermediate result.
pub fn compute_epstein_drag_batch_1d(
    grains: &[GrainPoint1d],
    gas: &[GrainGasPoint1d],
    box_size: f64,
    timestep: f64,
    parameters: EpsteinDragParameters,
) -> Result<ComputedEpsteinBatch1d, GrainError> {
    validate_positive("box_size", box_size)?;
    validate_nonnegative("timestep", timestep)?;
    validate_gas(gas, box_size)?;

    let mut grain_results = Vec::with_capacity(grains.len());
    for (grain_index, grain) in grains.iter().enumerate() {
        validate_position("grain_position", grain.position, box_size)?;
        validate_positive("grain_mass", grain.mass)?;
        validate_finite("grain_velocity", grain.velocity)?;
        validate_positive("grain_radius", grain.radius)?;
        validate_positive("smoothing_length", grain.smoothing_length)?;

        let interpolated =
            interpolate_gas_to_grain_1d(grain.position, grain.smoothing_length, gas, box_size)?;
        if timestep > 0.0 {
            for (gas_index, point) in gas.iter().enumerate() {
                let radius =
                    periodic_displacement_1d(grain.position, point.position, box_size)?.abs();
                if radius <= 0.0 && radius < grain.smoothing_length {
                    return Err(GrainError::CoincidentGasNeighbor {
                        grain_index,
                        gas_index,
                    });
                }
            }
        }
        let impulse = if timestep == 0.0 || interpolated.density == 0.0 {
            zero_epstein_impulse()
        } else {
            epstein_drag_impulse_1d(
                grain.mass,
                grain.velocity,
                grain.radius,
                interpolated,
                timestep,
                parameters,
            )?
        };
        grain_results.push(ComputedEpsteinGrain1d {
            interpolated_gas: interpolated,
            impulse,
        });
    }

    let mut gas_velocity_deltas = vec![0.0; gas.len()];
    for (grain, result) in grains.iter().zip(&grain_results) {
        if timestep <= 0.0 || result.interpolated_gas.density <= 0.0 {
            continue;
        }
        let contribution = distribute_grain_backreaction_1d(
            grain.position,
            grain.smoothing_length,
            result.interpolated_gas.density,
            result.impulse.grain_momentum_delta,
            gas,
            box_size,
        )?;
        for (total, delta) in gas_velocity_deltas.iter_mut().zip(contribution) {
            *total += delta;
            if !total.is_finite() {
                return Err(GrainError::NonFiniteResult {
                    field: "accumulated_gas_velocity_delta",
                    value: *total,
                });
            }
        }
    }

    let mut grain_momentum_delta = 0.0;
    for result in &grain_results {
        grain_momentum_delta += result.impulse.grain_momentum_delta;
    }

    let mut gas_momentum_delta = 0.0;
    for (point, velocity_delta) in gas.iter().zip(&gas_velocity_deltas) {
        gas_momentum_delta += point.mass * velocity_delta;
    }
    let momentum_residual = grain_momentum_delta + gas_momentum_delta;
    if !momentum_residual.is_finite() {
        return Err(GrainError::NonFiniteResult {
            field: "momentum_residual",
            value: momentum_residual,
        });
    }

    Ok(ComputedEpsteinBatch1d {
        grains: grain_results,
        gas_velocity_deltas,
        momentum_residual,
    })
}

const fn zero_epstein_impulse() -> EpsteinImpulse1d {
    EpsteinImpulse1d {
        inverse_stopping_time: 0.0,
        dimensionless_relative_speed: 0.0,
        velocity_relaxation_fraction: 0.0,
        grain_velocity_delta: 0.0,
        grain_momentum_delta: 0.0,
    }
}

fn validate_gas(gas: &[GrainGasPoint1d], box_size: f64) -> Result<(), GrainError> {
    for (index, point) in gas.iter().enumerate() {
        for (field, value) in [
            ("position", point.position),
            ("mass", point.mass),
            ("velocity", point.velocity),
            ("specific_internal_energy", point.specific_internal_energy),
        ] {
            let valid = value.is_finite()
                && match field {
                    "position" => value >= 0.0 && value < box_size,
                    "mass" | "specific_internal_energy" => value > 0.0,
                    _ => true,
                };
            if !valid {
                return Err(GrainError::InvalidGasPoint {
                    index,
                    field,
                    value,
                });
            }
        }
    }
    Ok(())
}

fn validate_interpolated(value: InterpolatedGrainGas1d) -> Result<(), GrainError> {
    for (field, scalar) in [
        ("gas_density", value.density),
        ("gas_velocity", value.velocity),
        (
            "gas_specific_internal_energy",
            value.specific_internal_energy,
        ),
    ] {
        if !scalar.is_finite() {
            return Err(GrainError::NonFiniteResult {
                field,
                value: scalar,
            });
        }
    }
    Ok(())
}

fn validate_position(field: &'static str, value: f64, box_size: f64) -> Result<(), GrainError> {
    if !value.is_finite() || value < 0.0 || value >= box_size {
        return Err(GrainError::InvalidInput { field, value });
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), GrainError> {
    if !value.is_finite() {
        return Err(GrainError::InvalidInput { field, value });
    }
    Ok(())
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), GrainError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(GrainError::InvalidInput { field, value });
    }
    Ok(())
}

fn validate_nonnegative(field: &'static str, value: f64) -> Result<(), GrainError> {
    if !value.is_finite() || value < 0.0 {
        return Err(GrainError::InvalidInput { field, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    fn uniform_periodic_gas() -> Vec<GrainGasPoint1d> {
        [0.125, 0.375, 0.625, 0.875]
            .map(|position| GrainGasPoint1d {
                position,
                mass: 0.25,
                velocity: 0.25,
                specific_internal_energy: 1.0,
            })
            .to_vec()
    }

    #[test]
    fn gas_interpolation_uses_mass_weighted_grain_kernel() {
        let gas = uniform_periodic_gas();
        let interpolated = interpolate_gas_to_grain_1d(0.0, 0.4, &gas, 1.0).unwrap();
        let near = cubic_kernel_1d(0.125, 0.4).unwrap().weight;
        let far = cubic_kernel_1d(0.375, 0.4).unwrap().weight;
        let expected_density = 0.25 * (2.0 * near + 2.0 * far);

        assert_eq!(interpolated.neighbor_count, 4);
        assert_close(interpolated.density, expected_density, 1.0e-15);
        assert_close(interpolated.velocity, 0.25, 1.0e-15);
        assert_close(interpolated.specific_internal_energy, 1.0, 1.0e-15);
    }

    #[test]
    fn interpolation_preserves_public_c_zero_lag_semantics() {
        let gas = [GrainGasPoint1d {
            position: 0.5,
            mass: 1.0,
            velocity: 3.0,
            specific_internal_energy: 2.0,
        }];
        let interpolated = interpolate_gas_to_grain_1d(0.5, 0.25, &gas, 1.0).unwrap();

        assert!(interpolated.density > 0.0);
        assert_close(interpolated.velocity, 0.0, 0.0);
        assert_close(interpolated.specific_internal_energy, 0.0, 0.0);
    }

    #[test]
    fn nonlinear_epstein_step_matches_public_c_reference_value() {
        let gas = InterpolatedGrainGas1d {
            density: 1.0,
            velocity: 0.0,
            specific_internal_energy: 1.0,
            neighbor_count: 4,
        };
        let impulse = epstein_drag_impulse_1d(
            1.0,
            1.0,
            1.236_08,
            gas,
            0.1,
            EpsteinDragParameters {
                gamma: 5.0 / 3.0,
                grain_internal_density: 1.0,
            },
        )
        .unwrap();

        assert_close(
            impulse.dimensionless_relative_speed,
            0.575_621_516_339_947_1,
            2.0e-15,
        );
        assert_close(
            impulse.inverse_stopping_time,
            1.054_090_956_044_137_4,
            2.0e-15,
        );
        assert_close(
            impulse.velocity_relaxation_fraction,
            0.113_012_000_620_125_51,
            2.0e-15,
        );
        assert_close(
            impulse.grain_velocity_delta,
            -0.113_012_000_620_125_51,
            2.0e-15,
        );
        assert_close(
            impulse.grain_momentum_delta,
            -0.113_012_000_620_125_51,
            2.0e-15,
        );
    }

    #[test]
    fn stiff_epstein_step_saturates_at_the_interpolated_gas_velocity() {
        let gas = InterpolatedGrainGas1d {
            density: 1.0,
            velocity: -2.0,
            specific_internal_energy: 1.0,
            neighbor_count: 1,
        };
        let impulse = epstein_drag_impulse_1d(
            2.0,
            3.0,
            1.0e-3,
            gas,
            1.0,
            EpsteinDragParameters {
                gamma: 5.0 / 3.0,
                grain_internal_density: 1.0,
            },
        )
        .unwrap();

        assert_close(impulse.velocity_relaxation_fraction, 1.0, 0.0);
        assert_close(impulse.grain_velocity_delta, -5.0, 0.0);
        assert_close(impulse.grain_momentum_delta, -10.0, 0.0);
    }

    #[test]
    fn epstein_transform_is_finite_at_tiny_and_huge_relative_speed() {
        let gas = InterpolatedGrainGas1d {
            density: 1.0,
            velocity: 0.0,
            specific_internal_energy: 1.0,
            neighbor_count: 1,
        };
        let parameters = EpsteinDragParameters {
            gamma: 5.0 / 3.0,
            grain_internal_density: 1.0,
        };
        let tiny = epstein_drag_impulse_1d(1.0, 1.0e-200, 1.236_08, gas, 0.1, parameters).unwrap();
        let expected_linear_fraction = 1.0 - (-0.1 * tiny.inverse_stopping_time).exp();
        assert_close(
            tiny.velocity_relaxation_fraction,
            expected_linear_fraction,
            2.0e-16,
        );
        assert!(tiny.grain_velocity_delta.is_finite());

        let huge = epstein_drag_impulse_1d(1.0, 1.0e300, 1.236_08, gas, 0.1, parameters).unwrap();
        assert!(huge.velocity_relaxation_fraction.is_finite());
        assert!(huge.velocity_relaxation_fraction > 0.99);
        assert!(huge.grain_velocity_delta.is_finite());

        let target_x = 1.0e20;
        let sound_speed = (parameters.gamma * (parameters.gamma - 1.0)).sqrt();
        let grain_velocity =
            target_x * sound_speed / (EPSTEIN_SPEED_COEFFICIENT * parameters.gamma.sqrt());
        let probe =
            epstein_drag_impulse_1d(1.0, grain_velocity, 1.236_08, gas, 1.0, parameters).unwrap();
        let tiny_stopping_step = 1.0e-20;
        let timestep = tiny_stopping_step / probe.inverse_stopping_time;
        let combined =
            epstein_drag_impulse_1d(1.0, grain_velocity, 1.236_08, gas, timestep, parameters)
                .unwrap();
        assert!(combined.velocity_relaxation_fraction.is_finite());
        assert_close(combined.velocity_relaxation_fraction, 0.5, 2.0e-15);
    }

    #[test]
    fn backreaction_is_kernel_weighted_and_conserves_total_momentum() {
        let mut gas = uniform_periodic_gas();
        let original_internal_energy: Vec<_> = gas
            .iter()
            .map(|point| point.specific_internal_energy)
            .collect();
        let initial_gas_momentum: f64 = gas.iter().map(|point| point.mass * point.velocity).sum();
        let mut grain = GrainPoint1d {
            position: 0.0,
            mass: 0.5,
            velocity: 1.0,
            smoothing_length: 0.4,
            radius: 1.236_08,
        };
        let initial_grain_momentum = grain.mass * grain.velocity;

        let applied = apply_epstein_drag_step_1d(
            &mut grain,
            &mut gas,
            1.0,
            0.1,
            EpsteinDragParameters {
                gamma: 5.0 / 3.0,
                grain_internal_density: 1.0,
            },
        )
        .unwrap();

        assert!(applied.impulse.grain_momentum_delta < 0.0);
        assert_close(
            applied.gas_velocity_deltas[0],
            applied.gas_velocity_deltas[3],
            0.0,
        );
        assert_close(
            applied.gas_velocity_deltas[1],
            applied.gas_velocity_deltas[2],
            0.0,
        );
        assert!(
            applied.gas_velocity_deltas[0] > applied.gas_velocity_deltas[1],
            "near gas must receive a larger velocity increment"
        );

        let final_gas_momentum: f64 = gas.iter().map(|point| point.mass * point.velocity).sum();
        let final_grain_momentum = grain.mass * grain.velocity;
        assert_close(
            final_gas_momentum + final_grain_momentum,
            initial_gas_momentum + initial_grain_momentum,
            2.0e-16,
        );
        assert_close(applied.momentum_residual, 0.0, 2.0e-17);
        assert_eq!(
            gas.iter()
                .map(|point| point.specific_internal_energy)
                .collect::<Vec<_>>(),
            original_internal_energy,
            "the public C operator does not apply drag heating"
        );
    }

    #[test]
    fn batched_operator_freezes_gas_and_is_grain_order_independent() {
        let gas_initial = uniform_periodic_gas();
        let grains_initial = [
            GrainPoint1d {
                position: 0.0,
                mass: 0.25,
                velocity: 1.0,
                smoothing_length: 0.4,
                radius: 1.236_08,
            },
            GrainPoint1d {
                position: 0.5,
                mass: 0.25,
                velocity: -0.5,
                smoothing_length: 0.4,
                radius: 1.236_08,
            },
        ];
        let parameters = EpsteinDragParameters {
            gamma: 5.0 / 3.0,
            grain_internal_density: 1.0,
        };

        let mut gas_forward = gas_initial.clone();
        let mut grains_forward = grains_initial;
        let forward = apply_epstein_drag_steps_1d(
            &mut grains_forward,
            &mut gas_forward,
            1.0,
            0.1,
            parameters,
        )
        .unwrap();

        let mut gas_reverse = gas_initial;
        let mut grains_reverse = [grains_initial[1], grains_initial[0]];
        let reverse = apply_epstein_drag_steps_1d(
            &mut grains_reverse,
            &mut gas_reverse,
            1.0,
            0.1,
            parameters,
        )
        .unwrap();

        for (forward_gas, reverse_gas) in gas_forward.iter().zip(&gas_reverse) {
            assert_close(forward_gas.velocity, reverse_gas.velocity, 2.0e-16);
            assert_close(
                forward_gas.specific_internal_energy,
                reverse_gas.specific_internal_energy,
                0.0,
            );
        }
        assert_close(grains_forward[0].velocity, grains_reverse[1].velocity, 0.0);
        assert_close(grains_forward[1].velocity, grains_reverse[0].velocity, 0.0);
        assert_close(forward.momentum_residual, 0.0, 5.0e-17);
        assert_close(reverse.momentum_residual, 0.0, 5.0e-17);
    }

    #[test]
    fn no_neighbor_and_zero_step_paths_are_noops() {
        let mut gas = uniform_periodic_gas();
        let original_gas = gas.clone();
        let mut grain = GrainPoint1d {
            position: 0.0,
            mass: 1.0,
            velocity: 1.0,
            smoothing_length: 0.01,
            radius: 1.0,
        };
        let original_grain = grain;
        let parameters = EpsteinDragParameters {
            gamma: 5.0 / 3.0,
            grain_internal_density: 1.0,
        };
        let no_neighbors =
            apply_epstein_drag_step_1d(&mut grain, &mut gas, 1.0, 0.1, parameters).unwrap();
        assert_close(no_neighbors.interpolated_gas.density, 0.0, 0.0);
        assert_eq!(grain, original_grain);
        assert_eq!(gas, original_gas);
        assert!(apply_epstein_drag_step_1d(&mut grain, &mut gas, 1.0, -0.1, parameters).is_err());

        grain.smoothing_length = 0.4;
        let before_zero_step = grain;
        let zero_step =
            apply_epstein_drag_step_1d(&mut grain, &mut gas, 1.0, 0.0, parameters).unwrap();
        assert_close(zero_step.impulse.grain_velocity_delta, 0.0, 0.0);
        assert_eq!(grain, before_zero_step);
        assert_eq!(gas, original_gas);
    }

    #[test]
    fn batched_operator_rejects_public_c_colocation_conservation_hole() {
        let mut gas = [GrainGasPoint1d {
            position: 0.5,
            mass: 1.0,
            velocity: 0.0,
            specific_internal_energy: 1.0,
        }];
        let mut grains = [GrainPoint1d {
            position: 0.5,
            mass: 1.0,
            velocity: 1.0,
            smoothing_length: 0.25,
            radius: 1.0,
        }];
        let error = apply_epstein_drag_steps_1d(
            &mut grains,
            &mut gas,
            1.0,
            0.1,
            EpsteinDragParameters {
                gamma: 5.0 / 3.0,
                grain_internal_density: 1.0,
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GrainError::CoincidentGasNeighbor {
                grain_index: 0,
                gas_index: 0
            }
        ));
    }
}
