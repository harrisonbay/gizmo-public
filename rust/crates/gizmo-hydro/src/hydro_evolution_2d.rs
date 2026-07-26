//! Two-dimensional rectangular-periodic nonmagnetic MFM Euler evolution.
//!
//! The local Riemann problem is solved in the moving face-normal frame with
//! the corrected public HLLC/KT/exact hierarchy from [`crate::ideal_gas_mfm_flux_1d`].
//! Unlike the magnetic planar operator, this module is a genuine pure-Euler
//! path: it has no zero-field HLLD dependency or magnetic conservative state.

use std::error::Error;
use std::fmt;

use crate::individual_timeline::{IndividualParticleTimeline, IndividualTimelineError};
use crate::meshless_2d::{
    Box2d, FaceClosure2d, GeometryError, InteractionPair2d, InverseMoment2d, MeshlessFace2d,
    MeshlessPoint2d, Vector2, cubic_kernel_2d, density_at_hsml_2d, face_closure_diagnostics_2d,
    interacting_pairs_2d, interacting_pairs_for_targets_2d, inverse_moments_2d,
    meshless_face_geometry_2d, particle_divergence_at_hsml_2d,
    particle_divergence_at_hsml_for_targets_2d, scalar_gradients_batch_with_moments_2d,
    solve_public_c_smoothing_lengths_from_seeds_2d,
    solve_public_c_smoothing_lengths_from_seeds_for_targets_2d,
};
use crate::mhd::Vector3;
use crate::{
    HydroError, PrimitiveState1d, RiemannMethod, ideal_gas_mfm_flux_1d, legacy_float_equal,
};

const LOCAL_GRADIENT_LIMITER_DISTANCE_FRACTION: f64 = 0.25;
const MIN_REAL_NUMBER: f64 = 1.0e-56;
const EPSILON_ENTROPIC_BIG: f64 = 0.5;
const EPSILON_ENTROPIC_SMALL: f64 = 1.0e-3;
const CONDITION_NUMBER_DANGER_SQUARED: f64 = 1.0e6;
const MAX_ACTIVE_TARGET_WORKERS: usize = 64;

/// Primitive Euler state in the simulation frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EulerPrimitive2d {
    pub density: f64,
    pub velocity: Vector3,
    pub pressure: f64,
}

/// Per-unit-area MFM Euler flux oriented along a planar face normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EulerMfmFlux2d {
    pub mass: f64,
    pub momentum: Vector3,
    pub total_energy: f64,
    pub star_pressure: f64,
    /// Contact or fallback signal speed in the moving face frame.
    pub solver_speed: f64,
    pub method: RiemannMethod,
}

/// Owned primitive state for synchronized nonmagnetic 2-D evolution.
#[derive(Clone, Debug, PartialEq)]
pub struct HydroMfmState2d {
    pub positions: Vec<Vector2>,
    pub masses: Vec<f64>,
    pub velocities: Vec<Vector3>,
    pub specific_internal_energy: Vec<f64>,
    /// Full compact-support radii.
    pub smoothing_lengths: Vec<f64>,
    pub domain: Box2d,
    pub gamma: f64,
}

/// Density-loop primitive fields recovered from [`HydroMfmState2d`].
#[derive(Clone, Debug, PartialEq)]
pub struct HydroPrimitiveColumns2d {
    pub density: Vec<f64>,
    pub dhsml_factor: Vec<f64>,
    pub pressure: Vec<f64>,
}

/// Extensive conservative rates and their primitive equivalents.
#[derive(Clone, Debug, PartialEq)]
pub struct HydroMfmRates2d {
    pub momentum: Vec<Vector3>,
    pub total_energy: Vec<f64>,
    pub acceleration: Vec<Vector3>,
    pub specific_internal_energy: Vec<f64>,
    pub maximum_signal_speed: Vec<f64>,
    pub velocity_divergence: Vec<f64>,
    pub pair_count: usize,
    pub entropic_pair_count: usize,
    pub hllc_pair_count: usize,
    pub kt_pair_count: usize,
    pub exact_pair_count: usize,
    pub vacuum_pair_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HydroKdkResult2d {
    pub state: HydroMfmState2d,
    /// Endpoint rates used by the second half-kick.
    pub rates: HydroMfmRates2d,
}

/// Per-particle non-cosmological timestep criteria used by public hydro.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublicHydroTimestepBounds2d {
    pub acceleration: f64,
    pub courant: f64,
    pub velocity_divergence: f64,
    pub selected: f64,
}

/// Initial power-of-two particle step on the public `2^60` timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublicHydroInitialTimebin2d {
    pub bounded_timestep: f64,
    pub raw_ticks: u64,
    pub ticks: u64,
    pub time_bin: u32,
    pub duration: f64,
}

/// Mixed actual/predicted state retained during hierarchical lazy drifts.
#[derive(Clone, Debug, PartialEq)]
pub struct PublicHydroDriftState2d {
    pub positions: Vec<Vector2>,
    pub actual_velocities: Vec<Vector3>,
    pub predicted_velocities: Vec<Vector3>,
    pub predicted_specific_internal_energy: Vec<f64>,
    pub predicted_density: Vec<f64>,
    pub predicted_smoothing_lengths: Vec<f64>,
}

/// Public individual-bin KDK state, including retained inactive caches.
#[derive(Clone, Debug, PartialEq)]
pub struct PublicHydroInitialHierarchy2d {
    timeline: IndividualParticleTimeline,
    start: HydroMfmState2d,
    old_rates: HydroMfmRates2d,
    actual_internal_energy: Vec<f64>,
    drift: PublicHydroDriftState2d,
    predictor_ticks: Vec<u64>,
    primitive_cache: HydroPrimitiveColumns2d,
    gradient_cache: PrimitiveGradients2d,
    moment_cache: Vec<InverseMoment2d>,
    face_closure_cache: Vec<FaceClosure2d>,
    minimum_specific_internal_energy: f64,
    awaiting_second_kick: bool,
    prepared_next_drift: bool,
    refreshed_cache_tick: Option<u64>,
    pending_wakeup: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublicHydroHierarchySync2d {
    pub tick: u64,
    pub time: f64,
    pub active: Vec<bool>,
    pub drift: PublicHydroDriftState2d,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublicHydroActiveRateResult2d {
    pub rates: HydroMfmRates2d,
    pub wakeup: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublicHydroHierarchyKickResult2d {
    pub state: HydroMfmState2d,
    pub rates: HydroMfmRates2d,
    pub wakeup: Vec<bool>,
}

#[derive(Debug, PartialEq)]
struct ActiveHydroTargetBatch2d {
    target: usize,
    momentum: Vector3,
    total_energy: f64,
    maximum_signal_speed: f64,
    wakeup_neighbors: Vec<usize>,
    entropic_pair_count: usize,
    method_counts: [usize; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub enum HydroEvolution2dError {
    MismatchedLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidState {
        index: Option<usize>,
        field: &'static str,
        value: f64,
    },
    Geometry(GeometryError),
    Riemann(HydroError),
    Timeline(IndividualTimelineError),
}

impl fmt::Display for HydroEvolution2dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedLength {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "2-D hydro column `{field}` has length {actual}, expected {expected}"
            ),
            Self::InvalidState {
                index,
                field,
                value,
            } => write!(
                formatter,
                "invalid 2-D hydro `{field}` at {index:?}: {value}"
            ),
            Self::Geometry(error) => error.fmt(formatter),
            Self::Riemann(error) => error.fmt(formatter),
            Self::Timeline(error) => error.fmt(formatter),
        }
    }
}

impl Error for HydroEvolution2dError {}

impl From<GeometryError> for HydroEvolution2dError {
    fn from(error: GeometryError) -> Self {
        Self::Geometry(error)
    }
}

impl From<HydroError> for HydroEvolution2dError {
    fn from(error: HydroError) -> Self {
        Self::Riemann(error)
    }
}

impl From<IndividualTimelineError> for HydroEvolution2dError {
    fn from(error: IndividualTimelineError) -> Self {
        Self::Timeline(error)
    }
}

impl HydroMfmState2d {
    /// Construct and validate a primitive nonmagnetic state.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Self::validate`].
    #[allow(clippy::too_many_arguments)]
    pub fn from_primitive(
        positions: Vec<Vector2>,
        masses: Vec<f64>,
        velocities: Vec<Vector3>,
        specific_internal_energy: Vec<f64>,
        smoothing_lengths: Vec<f64>,
        domain: Box2d,
        gamma: f64,
    ) -> Result<Self, HydroEvolution2dError> {
        let state = Self {
            positions,
            masses,
            velocities,
            specific_internal_energy,
            smoothing_lengths,
            domain,
            gamma,
        };
        state.validate()?;
        Ok(state)
    }

    /// Validate a complete state without silently repairing invalid columns.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched columns, positions outside the periodic
    /// box, invalid thermodynamics, or a non-finite velocity.
    pub fn validate(&self) -> Result<(), HydroEvolution2dError> {
        let count = self.positions.len();
        for (field, actual) in [
            ("masses", self.masses.len()),
            ("velocities", self.velocities.len()),
            (
                "specific_internal_energy",
                self.specific_internal_energy.len(),
            ),
            ("smoothing_lengths", self.smoothing_lengths.len()),
        ] {
            if actual != count {
                return Err(HydroEvolution2dError::MismatchedLength {
                    field,
                    expected: count,
                    actual,
                });
            }
        }
        if !self.gamma.is_finite() || self.gamma <= 1.0 {
            return Err(invalid(None, "gamma", self.gamma));
        }
        for i in 0..count {
            if !self.domain.contains(self.positions[i]) {
                return Err(invalid(Some(i), "position", f64::NAN));
            }
            if !self.velocities[i].is_finite() {
                return Err(invalid(Some(i), "velocity", f64::NAN));
            }
            for (field, value) in [
                ("mass", self.masses[i]),
                ("specific_internal_energy", self.specific_internal_energy[i]),
                ("smoothing_length", self.smoothing_lengths[i]),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    return Err(invalid(Some(i), field, value));
                }
            }
        }
        Ok(())
    }

    /// Recompute density, the adaptive-H correction, and ideal-gas pressure.
    ///
    /// # Errors
    ///
    /// Returns an error if the state or density geometry is invalid.
    pub fn primitive_columns(&self) -> Result<HydroPrimitiveColumns2d, HydroEvolution2dError> {
        self.validate()?;
        let estimates = density_at_hsml_2d(
            &self.positions,
            &self.masses,
            &self.smoothing_lengths,
            self.domain,
        )?;
        let density: Vec<_> = estimates.iter().map(|value| value.density).collect();
        let dhsml_factor = estimates.iter().map(|value| value.dhsml_factor).collect();
        let pressure = density
            .iter()
            .zip(&self.specific_internal_energy)
            .map(|(&rho, &internal)| (self.gamma - 1.0) * rho * internal)
            .collect::<Vec<_>>();
        for (i, &value) in pressure.iter().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(Some(i), "pressure", value));
            }
        }
        Ok(HydroPrimitiveColumns2d {
            density,
            dhsml_factor,
            pressure,
        })
    }

    #[must_use]
    pub fn total_momentum(&self) -> Vector3 {
        self.masses
            .iter()
            .zip(&self.velocities)
            .fold(Vector3::ZERO, |sum, (&mass, &velocity)| {
                sum + velocity * mass
            })
    }

    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.masses
            .iter()
            .zip(&self.velocities)
            .zip(&self.specific_internal_energy)
            .map(|((&mass, &velocity), &internal)| {
                mass * (internal + 0.5 * velocity.squared_norm())
            })
            .sum()
    }
}

/// Return the global nonmagnetic acoustic CFL bound.
///
/// This uses the public 2-D MFM cell scale `(mass / density)^0.5` and the
/// pairwise maximum signal speeds retained by [`hydro_mfm_spatial_rates_2d`].
///
/// # Errors
///
/// Returns an error for invalid state/rate columns or a Courant factor outside
/// `(0, 0.5]`.
pub fn global_hydro_courant_timestep_2d(
    state: &HydroMfmState2d,
    rates: &HydroMfmRates2d,
    courant_factor: f64,
) -> Result<f64, HydroEvolution2dError> {
    state.validate()?;
    validate_rate_lengths(rates, state.positions.len())?;
    if !courant_factor.is_finite() || courant_factor <= 0.0 || courant_factor > 0.5 {
        return Err(invalid(None, "courant_factor", courant_factor));
    }
    let primitive = state.primitive_columns()?;
    let mut timestep = f64::INFINITY;
    for (i, &density) in primitive.density.iter().enumerate() {
        let particle_size = (state.masses[i] / density).sqrt();
        let candidate = courant_factor * particle_size / (0.5 * rates.maximum_signal_speed[i]);
        if !candidate.is_finite() || candidate <= 0.0 {
            return Err(invalid(Some(i), "courant_timestep", candidate));
        }
        timestep = timestep.min(candidate);
    }
    if !timestep.is_finite() {
        return Err(invalid(None, "courant_timestep", timestep));
    }
    Ok(timestep)
}

/// Evaluate the public acceleration, acoustic-Courant, and compression bounds.
///
/// # Errors
///
/// Returns an error for invalid state, retained rates, or policy constants.
pub fn public_hydro_particle_timestep_bounds_2d(
    state: &HydroMfmState2d,
    rates: &HydroMfmRates2d,
    courant_factor: f64,
    integration_accuracy: f64,
) -> Result<Vec<PublicHydroTimestepBounds2d>, HydroEvolution2dError> {
    let primitive = state.primitive_columns()?;
    public_hydro_particle_timestep_bounds_from_primitive_2d(
        state,
        &primitive,
        rates,
        courant_factor,
        integration_accuracy,
    )
}

/// Evaluate public hydro timestep bounds against a retained lazy-drift cache.
///
/// # Errors
///
/// Returns an error for invalid cached primitives, rates, or policy constants.
pub fn public_hydro_particle_timestep_bounds_from_primitive_2d(
    state: &HydroMfmState2d,
    primitive: &HydroPrimitiveColumns2d,
    rates: &HydroMfmRates2d,
    courant_factor: f64,
    integration_accuracy: f64,
) -> Result<Vec<PublicHydroTimestepBounds2d>, HydroEvolution2dError> {
    state.validate()?;
    validate_rate_lengths(rates, state.positions.len())?;
    validate_lengths(
        state.positions.len(),
        &[
            ("primitive_density", primitive.density.len()),
            ("primitive_dhsml_factor", primitive.dhsml_factor.len()),
            ("primitive_pressure", primitive.pressure.len()),
        ],
    )?;
    if !courant_factor.is_finite() || courant_factor <= 0.0 || courant_factor > 0.5 {
        return Err(invalid(None, "courant_factor", courant_factor));
    }
    if !integration_accuracy.is_finite() || integration_accuracy <= 0.0 {
        return Err(invalid(None, "integration_accuracy", integration_accuracy));
    }
    let mut result = Vec::with_capacity(state.positions.len());
    for i in 0..state.positions.len() {
        let acceleration = rates.acceleration[i].squared_norm().sqrt().max(1.0e-30);
        let acceleration_bound =
            (integration_accuracy * state.smoothing_lengths[i] / acceleration).sqrt();
        let effective_neighbor_root =
            (std::f64::consts::PI * state.smoothing_lengths[i].powi(2) * primitive.density[i]
                / state.masses[i])
                .sqrt();
        let particle_size = 1.77245 * state.smoothing_lengths[i] / effective_neighbor_root;
        let courant_bound = courant_factor * particle_size / (0.5 * rates.maximum_signal_speed[i]);
        let divergence_bound = if rates.velocity_divergence[i] == 0.0 {
            f64::INFINITY
        } else {
            1.5 / rates.velocity_divergence[i].abs()
        };
        let selected = acceleration_bound.min(courant_bound).min(divergence_bound);
        for (field, value) in [
            ("acceleration_timestep", acceleration_bound),
            ("courant_timestep", courant_bound),
            ("velocity_divergence_timestep", divergence_bound),
        ] {
            if value.is_nan() || value <= 0.0 {
                return Err(invalid(Some(i), field, value));
            }
        }
        if !selected.is_finite() || selected <= 0.0 {
            return Err(invalid(Some(i), "public_hydro_timestep", selected));
        }
        result.push(PublicHydroTimestepBounds2d {
            acceleration: acceleration_bound,
            courant: courant_bound,
            velocity_divergence: divergence_bound,
            selected,
        });
    }
    Ok(result)
}

/// Quantize initial particle requests onto the public power-of-two timebase.
///
/// # Errors
///
/// Returns an error for invalid timeline bounds or timestep requests.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn quantize_public_hydro_initial_timebins_2d(
    bounds: &[PublicHydroTimestepBounds2d],
    time_begin: f64,
    time_max: f64,
    maximum_timestep: f64,
) -> Result<Vec<PublicHydroInitialTimebin2d>, HydroEvolution2dError> {
    if !time_begin.is_finite()
        || !time_max.is_finite()
        || time_max <= time_begin
        || !maximum_timestep.is_finite()
        || maximum_timestep <= 0.0
    {
        return Err(invalid(None, "public_hydro_timeline", time_max));
    }
    let tick_duration = (time_max - time_begin) / crate::LEGACY_TIMEBASE_TICKS as f64;
    bounds
        .iter()
        .enumerate()
        .map(|(i, bound)| {
            if !bound.selected.is_finite() || bound.selected <= 0.0 {
                return Err(invalid(Some(i), "public_hydro_timestep", bound.selected));
            }
            let bounded_timestep = bound.selected.min(maximum_timestep);
            let requested = (bounded_timestep / tick_duration).trunc();
            if !requested.is_finite() || requested < 0.0 {
                return Err(invalid(Some(i), "public_hydro_timestep_ticks", requested));
            }
            let raw_ticks = (requested as u64).max(2);
            if raw_ticks >= crate::LEGACY_TIMEBASE_TICKS {
                return Err(invalid(Some(i), "public_hydro_timestep_ticks", requested));
            }
            let time_bin = raw_ticks.ilog2();
            let ticks = 1_u64 << time_bin;
            Ok(PublicHydroInitialTimebin2d {
                bounded_timestep,
                raw_ticks,
                ticks,
                time_bin,
                duration: ticks as f64 * tick_duration,
            })
        })
        .collect()
}

/// Solve the pure-Euler MFM face problem in an arbitrary planar orientation.
///
/// `interface_velocity` is the moving face velocity in the simulation frame.
/// Both input velocities are deboosted and projected onto `unit_normal`; the
/// corrected HLLC/KT/exact hierarchy is then solved locally and its momentum
/// and energy fluxes are rotated/deboosted into the simulation frame.
///
/// # Errors
///
/// Returns an error for an invalid primitive, non-unit normal, pressure limit,
/// or Riemann arithmetic.
pub fn euler_mfm_flux_2d(
    left: EulerPrimitive2d,
    right: EulerPrimitive2d,
    unit_normal: Vector2,
    interface_velocity: Vector3,
    gamma: f64,
    pressure_limit: f64,
) -> Result<EulerMfmFlux2d, HydroEvolution2dError> {
    validate_primitive(left)?;
    validate_primitive(right)?;
    let normal_length = unit_normal.norm();
    if !unit_normal.is_finite()
        || !normal_length.is_finite()
        || (normal_length - 1.0).abs() > 128.0 * f64::EPSILON
    {
        return Err(invalid(None, "unit_normal", normal_length));
    }
    if !interface_velocity.is_finite() {
        return Err(invalid(None, "interface_velocity", f64::NAN));
    }
    let left_relative = left.velocity - interface_velocity;
    let right_relative = right.velocity - interface_velocity;
    let left_normal = planar_dot(left_relative, unit_normal);
    let right_normal = planar_dot(right_relative, unit_normal);
    let normal3 = Vector3::new(unit_normal.x, unit_normal.y, 0.0);
    let local = vector_mfm_flux(
        left,
        right,
        left_relative,
        right_relative,
        left_normal,
        right_normal,
        normal3,
        gamma,
        pressure_limit,
    )?;
    let result = EulerMfmFlux2d {
        mass: local.mass,
        momentum: local.momentum,
        total_energy: interface_velocity.dot(local.momentum) + local.energy,
        star_pressure: local.star_pressure,
        solver_speed: local.solver_speed,
        method: local.method,
    };
    if !result.momentum.is_finite() || !result.total_energy.is_finite() || !result.mass.is_finite()
    {
        return Err(invalid(None, "face_flux", f64::NAN));
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct VectorMfmFlux {
    mass: f64,
    momentum: Vector3,
    energy: f64,
    star_pressure: f64,
    solver_speed: f64,
    method: RiemannMethod,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn vector_mfm_flux(
    left: EulerPrimitive2d,
    right: EulerPrimitive2d,
    left_relative: Vector3,
    right_relative: Vector3,
    left_normal: f64,
    right_normal: f64,
    normal: Vector3,
    gamma: f64,
    pressure_limit: f64,
) -> Result<VectorMfmFlux, HydroEvolution2dError> {
    let sound_left = (gamma * left.pressure / left.density).sqrt();
    let sound_right = (gamma * right.pressure / right.density).sqrt();
    let left_1d = PrimitiveState1d {
        density: left.density,
        velocity: left_normal,
        pressure: left.pressure,
    };
    let right_1d = PrimitiveState1d {
        density: right.density,
        velocity: right_normal,
        pressure: right.pressure,
    };
    let vacuum_threshold = 2.0 * (sound_left + sound_right) / (gamma - 1.0);
    if right_normal - left_normal > vacuum_threshold {
        return scalar_flux_as_vector(
            ideal_gas_mfm_flux_1d(left_1d, right_1d, gamma, pressure_limit)?,
            normal,
        );
    }

    // reimann.h:344-347 and :563-580 retain all three kinetic components in
    // enthalpy and Roe averaging even though the Riemann normal is planar.
    let enthalpy_left = left.pressure / left.density
        + left.pressure / ((gamma - 1.0) * left.density)
        + 0.5 * left_relative.squared_norm();
    let enthalpy_right = right.pressure / right.density
        + right.pressure / ((gamma - 1.0) * right.density)
        + 0.5 * right_relative.squared_norm();
    let sound_maximum = sound_left.max(sound_right);
    let mut wave_left = left_normal.min(right_normal) - sound_maximum;
    let mut wave_right = left_normal.max(right_normal) + sound_maximum;
    let mut density_weight_left = left.density * (wave_left - left_normal);
    let mut density_weight_right = right.density * (wave_right - right_normal);
    let mut contact_speed = ((right.pressure - left.pressure) + density_weight_left * left_normal
        - density_weight_right * right_normal)
        / (density_weight_left - density_weight_right);
    let mut star_pressure = (left.pressure * density_weight_right
        - right.pressure * density_weight_left
        + density_weight_left * density_weight_right * (right_normal - left_normal))
        / (density_weight_right - density_weight_left);
    if valid_star(star_pressure, contact_speed, pressure_limit) {
        return Ok(hllc_vector_flux(normal, star_pressure, contact_speed));
    }

    let sqrt_density_left = left.density.sqrt();
    let sqrt_density_right = right.density.sqrt();
    let inverse_sqrt_density_sum = (sqrt_density_left + sqrt_density_right).recip();
    let roe_velocity = (left_relative * sqrt_density_left + right_relative * sqrt_density_right)
        * inverse_sqrt_density_sum;
    let roe_normal = roe_velocity.dot(normal);
    let roe_enthalpy = (sqrt_density_left * enthalpy_left + sqrt_density_right * enthalpy_right)
        * inverse_sqrt_density_sum;
    let roe_sound =
        ((gamma - 1.0) * (roe_enthalpy - 0.5 * roe_velocity.squared_norm()).max(1.0e-30)).sqrt();
    wave_right = (right_normal + sound_right).max(roe_normal + roe_sound);
    wave_left = (left_normal - sound_left).min(roe_normal - roe_sound);
    density_weight_right = right.density * (wave_right - right_normal);
    density_weight_left = -left.density * (wave_left - left_normal);
    contact_speed =
        (density_weight_right * right_normal + density_weight_left * left_normal + left.pressure
            - right.pressure)
            / (density_weight_right + density_weight_left);
    star_pressure =
        left.density * (left_normal - wave_left) * (left_normal - contact_speed) + left.pressure;
    if valid_star(star_pressure, contact_speed, pressure_limit) {
        return Ok(hllc_vector_flux(normal, star_pressure, contact_speed));
    }

    star_pressure = 0.5
        * (left.pressure
            + right.pressure
            + (left_normal - right_normal)
                * 0.25
                * (left.density + right.density)
                * (sound_left + sound_right));
    contact_speed = 0.5 * (right_normal + left_normal)
        + 2.0 * (left.pressure - right.pressure)
            / ((left.density + right.density) * (sound_left + sound_right));
    let signal_speed = [
        (left_normal - sound_left).abs(),
        (right_normal - sound_right).abs(),
        (left_normal + sound_left).abs(),
        (right_normal + sound_right).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max);
    contact_speed = contact_speed.clamp(-signal_speed, signal_speed);
    if star_pressure.is_finite() && contact_speed.is_finite() && star_pressure >= 0.0 {
        if star_pressure > pressure_limit {
            return forced_exact_vector_flux(left_1d, right_1d, normal, gamma);
        }
        return Ok(hllc_vector_flux(normal, star_pressure, contact_speed));
    }

    // reimann.h:500-526 uses the full momentum-jump norm to suppress KT
    // diffusion in shear, then uses that same S_M in P*, momentum, and energy.
    let momentum_jump = right_relative * right.density - left_relative * left.density;
    let normal_momentum_jump = right.density * right_normal - left.density * left_normal;
    let threshold = 0.001 * 0.5 * (left.density + right.density) * 0.5 * (sound_left + sound_right);
    let alpha =
        normal_momentum_jump.abs() / (threshold * threshold + momentum_jump.squared_norm()).sqrt();
    let diffusion_speed =
        (alpha * sound_left + left_normal.abs()).max(alpha * sound_right + right_normal.abs());
    let reported_signal_speed =
        (sound_left + left_normal.abs()).max(sound_right + right_normal.abs());
    let weighted_left = left.density * (left_normal + diffusion_speed);
    let weighted_right = right.density * (right_normal - diffusion_speed);
    let denominator_base = left.density * left_normal - right.density * right_normal
        + diffusion_speed * (left.density + right.density);
    if !denominator_base.is_finite() || denominator_base == 0.0 {
        return forced_exact_vector_flux(left_1d, right_1d, normal, gamma);
    }
    let denominator = denominator_base.recip();
    let weighted_product = weighted_left * weighted_right;
    star_pressure = (weighted_left * right.pressure - weighted_right * left.pressure) * denominator;
    if !star_pressure.is_finite() || star_pressure < 0.0 || star_pressure > pressure_limit {
        return forced_exact_vector_flux(left_1d, right_1d, normal, gamma);
    }
    if star_pressure == 0.0 {
        return Ok(VectorMfmFlux {
            mass: 0.0,
            momentum: Vector3::ZERO,
            energy: 0.0,
            star_pressure,
            solver_speed: reported_signal_speed,
            method: RiemannMethod::KurganovTadmor,
        });
    }
    let momentum = (right_relative - left_relative) * (weighted_product * denominator)
        + normal * star_pressure;
    let energy = (diffusion_speed
        * (weighted_left * right.pressure + weighted_right * left.pressure)
        + (enthalpy_right - enthalpy_left) * weighted_product)
        * denominator;
    if [
        alpha,
        diffusion_speed,
        reported_signal_speed,
        denominator,
        energy,
        star_pressure,
    ]
    .iter()
    .any(|value| !value.is_finite())
        || !momentum.is_finite()
    {
        return Err(invalid(None, "kt_flux", f64::NAN));
    }
    Ok(VectorMfmFlux {
        mass: 0.0,
        momentum,
        energy,
        star_pressure,
        solver_speed: reported_signal_speed,
        method: RiemannMethod::KurganovTadmor,
    })
}

fn valid_star(star_pressure: f64, contact_speed: f64, pressure_limit: f64) -> bool {
    star_pressure.is_finite()
        && contact_speed.is_finite()
        && star_pressure > 0.0
        && star_pressure <= pressure_limit
}

fn hllc_vector_flux(normal: Vector3, star_pressure: f64, contact_speed: f64) -> VectorMfmFlux {
    VectorMfmFlux {
        mass: 0.0,
        momentum: normal * star_pressure,
        energy: star_pressure * contact_speed,
        star_pressure,
        solver_speed: contact_speed,
        method: RiemannMethod::Hllc,
    }
}

fn forced_exact_vector_flux(
    left: PrimitiveState1d,
    right: PrimitiveState1d,
    normal: Vector3,
    gamma: f64,
) -> Result<VectorMfmFlux, HydroEvolution2dError> {
    scalar_flux_as_vector(
        ideal_gas_mfm_flux_1d(left, right, gamma, MIN_REAL_NUMBER)?,
        normal,
    )
}

fn scalar_flux_as_vector(
    flux: crate::MfmFlux1d,
    normal: Vector3,
) -> Result<VectorMfmFlux, HydroEvolution2dError> {
    if !matches!(
        flux.method,
        RiemannMethod::Hllc | RiemannMethod::Exact | RiemannMethod::Vacuum
    ) {
        return Err(invalid(None, "forced_exact_method", f64::NAN));
    }
    Ok(VectorMfmFlux {
        mass: flux.mass,
        momentum: normal * flux.momentum,
        energy: flux.energy,
        star_pressure: flux.star_pressure,
        solver_speed: flux.solver_speed,
        method: flux.method,
    })
}

/// Evaluate the semidiscrete rectangular-periodic nonmagnetic MFM operator.
///
/// Every unordered pair is solved once. Momentum and total-energy rates are
/// accumulated antisymmetrically, so their global sums vanish to roundoff.
///
/// # Errors
///
/// Returns an error for invalid state, meshless geometry, reconstruction, or
/// local Riemann arithmetic.
#[allow(clippy::too_many_lines)]
pub fn hydro_mfm_spatial_rates_2d(
    state: &HydroMfmState2d,
) -> Result<HydroMfmRates2d, HydroEvolution2dError> {
    state.validate()?;
    let primitive = state.primitive_columns()?;
    let moments = inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    let closure = face_closure_diagnostics_2d(
        &state.positions,
        &state.masses,
        &state.smoothing_lengths,
        state.domain,
    )?;
    let gradients = primitive_gradients(state, &primitive, &moments)?;
    let planar_velocities = state
        .velocities
        .iter()
        .map(|value| Vector2::new(value.x, value.y))
        .collect::<Vec<_>>();
    let velocity_divergence = particle_divergence_at_hsml_2d(
        &state.positions,
        &planar_velocities,
        &state.smoothing_lengths,
        &primitive.dhsml_factor,
        state.domain,
    )?;
    let count = state.positions.len();
    let mut momentum = vec![Vector3::ZERO; count];
    let mut total_energy = vec![0.0; count];
    let mut maximum_signal_speed = (0..count)
        .map(|i| (state.gamma * primitive.pressure[i] / primitive.density[i]).sqrt())
        .collect::<Vec<_>>();
    let mut method_counts = [0_usize; 4];
    let mut entropic_pair_count = 0_usize;
    let pairs = interacting_pairs_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    for pair in &pairs {
        let point = |i: usize| MeshlessPoint2d {
            position: state.positions[i],
            mass: state.masses[i],
            density: primitive.density[i],
            smoothing_length: state.smoothing_lengths[i],
            inverse_moment: moments[i].matrix,
            condition_number: moments[i].condition_number,
        };
        let face = meshless_face_geometry_2d(point(pair.i), point(pair.j), state.domain)?;
        let flux = solve_pair_with_retries(
            state, &primitive, &gradients, &closure, pair.i, pair.j, face,
        )?;
        let pair_momentum = flux.momentum * face.area;
        let displacement = state
            .domain
            .displacement(state.positions[pair.i], state.positions[pair.j])?;
        let (pair_energy, selected_entropic) = apply_entropic_pdv_energy_2d(
            flux.total_energy * face.area,
            flux,
            face,
            displacement,
            state,
            &primitive,
            &moments,
            &closure,
            pair.i,
            pair.j,
        )?;
        momentum[pair.i] = momentum[pair.i] + pair_momentum;
        momentum[pair.j] = momentum[pair.j] - pair_momentum;
        total_energy[pair.i] += pair_energy;
        total_energy[pair.j] -= pair_energy;
        entropic_pair_count += usize::from(selected_entropic);
        method_counts[method_index(flux.method)] += 1;

        let radial = displacement / displacement.norm();
        let radial_approach =
            planar_dot(state.velocities[pair.i] - state.velocities[pair.j], radial).min(0.0);
        let sound_i = (state.gamma * primitive.pressure[pair.i] / primitive.density[pair.i]).sqrt();
        let sound_j = (state.gamma * primitive.pressure[pair.j] / primitive.density[pair.j]).sqrt();
        let signal = sound_i + sound_j - radial_approach;
        maximum_signal_speed[pair.i] = maximum_signal_speed[pair.i].max(signal);
        maximum_signal_speed[pair.j] = maximum_signal_speed[pair.j].max(signal);
    }

    let mut acceleration = Vec::with_capacity(count);
    let mut specific_internal_energy = Vec::with_capacity(count);
    for i in 0..count {
        let inverse_mass = 1.0 / state.masses[i];
        let particle_acceleration = momentum[i] * inverse_mass;
        let internal_rate = (total_energy[i] - state.velocities[i].dot(momentum[i])) * inverse_mass;
        if !particle_acceleration.is_finite() || !internal_rate.is_finite() {
            return Err(invalid(Some(i), "primitive_rate", f64::NAN));
        }
        acceleration.push(particle_acceleration);
        specific_internal_energy.push(internal_rate);
    }
    Ok(HydroMfmRates2d {
        momentum,
        total_energy,
        acceleration,
        specific_internal_energy,
        maximum_signal_speed,
        velocity_divergence,
        pair_count: pairs.len(),
        entropic_pair_count,
        hllc_pair_count: method_counts[0],
        kt_pair_count: method_counts[1],
        exact_pair_count: method_counts[2],
        vacuum_pair_count: method_counts[3],
    })
}

/// Evaluate every particle as an independent active MFM target.
///
/// Public GIZMO's hydro force loop deliberately does not credit the neighbor
/// with the negated target flux. Even at a full synchronization point, the
/// reverse orientation is reconstructed and solved independently. Use this
/// path for public-C trajectory parity; [`hydro_mfm_spatial_rates_2d`] remains
/// the exactly conservative unordered-pair reference operator.
///
/// # Errors
///
/// Returns an error for invalid state, meshless geometry, reconstruction, or
/// local Riemann arithmetic.
pub fn hydro_mfm_directed_spatial_rates_2d(
    state: &HydroMfmState2d,
) -> Result<HydroMfmRates2d, HydroEvolution2dError> {
    state.validate()?;
    let count = state.positions.len();
    let primitive = state.primitive_columns()?;
    let moments = inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    let closure = face_closure_diagnostics_2d(
        &state.positions,
        &state.masses,
        &state.smoothing_lengths,
        state.domain,
    )?;
    let gradients = primitive_gradients(state, &primitive, &moments)?;
    let retained = HydroMfmRates2d {
        momentum: vec![Vector3::ZERO; count],
        total_energy: vec![0.0; count],
        acceleration: vec![Vector3::ZERO; count],
        specific_internal_energy: vec![0.0; count],
        maximum_signal_speed: vec![0.0; count],
        velocity_divergence: vec![0.0; count],
        pair_count: 0,
        entropic_pair_count: 0,
        hllc_pair_count: 0,
        kt_pair_count: 0,
        exact_pair_count: 0,
        vacuum_pair_count: 0,
    };
    let active = vec![true; count];
    Ok(hydro_mfm_active_target_rates_with_cache_2d(
        state, &primitive, &gradients, &moments, &closure, &retained, &active,
    )?
    .rates)
}

/// Advance one synchronized conservative KDK step while retaining current H.
///
/// # Errors
///
/// Returns an error for invalid inputs, force evaluation failure, or loss of
/// positive internal energy.
pub fn advance_hydro_kdk_2d(
    state: &HydroMfmState2d,
    old_rates: &HydroMfmRates2d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
) -> Result<HydroKdkResult2d, HydroEvolution2dError> {
    advance_hydro_kdk_impl_2d(
        state,
        old_rates,
        timestep,
        minimum_specific_internal_energy,
        None,
    )
}

/// Advance one synchronized conservative KDK step and repeat the public-C
/// adaptive smoothing-length solve after the drift.
///
/// # Errors
///
/// Returns an error for invalid inputs, an unconverged H solve, force
/// evaluation failure, or loss of positive internal energy.
#[allow(clippy::too_many_arguments)]
pub fn advance_hydro_kdk_adaptive_2d(
    state: &HydroMfmState2d,
    old_rates: &HydroMfmRates2d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
) -> Result<HydroKdkResult2d, HydroEvolution2dError> {
    advance_hydro_kdk_impl_2d(
        state,
        old_rates,
        timestep,
        minimum_specific_internal_energy,
        Some((desired_neighbors, neighbor_tolerance)),
    )
}

fn advance_hydro_kdk_impl_2d(
    state: &HydroMfmState2d,
    old_rates: &HydroMfmRates2d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    adaptive_h: Option<(f64, f64)>,
) -> Result<HydroKdkResult2d, HydroEvolution2dError> {
    state.validate()?;
    validate_rate_lengths(old_rates, state.positions.len())?;
    if !timestep.is_finite() || timestep <= 0.0 {
        return Err(invalid(None, "timestep", timestep));
    }
    if !minimum_specific_internal_energy.is_finite() || minimum_specific_internal_energy < 0.0 {
        return Err(invalid(
            None,
            "minimum_specific_internal_energy",
            minimum_specific_internal_energy,
        ));
    }
    let (momentum, total_energy) = extensive_state(state);
    let half_momentum = kick_vectors(&momentum, &old_rates.momentum, 0.5 * timestep);
    let half_energy = kick_scalars(&total_energy, &old_rates.total_energy, 0.5 * timestep);
    let half_velocity: Vec<_> = half_momentum
        .iter()
        .zip(&state.masses)
        .map(|(&value, &mass)| value / mass)
        .collect();
    let positions = state
        .positions
        .iter()
        .zip(&half_velocity)
        .map(|(&position, &velocity)| {
            state
                .domain
                .wrap(position + Vector2::new(velocity.x, velocity.y) * timestep)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let smoothing_lengths = if let Some((desired, tolerance)) = adaptive_h {
        solve_public_c_smoothing_lengths_from_seeds_2d(
            &positions,
            &state.masses,
            &state.smoothing_lengths,
            state.domain,
            desired,
            tolerance,
        )?
        .into_iter()
        .map(|estimate| estimate.smoothing_length)
        .collect()
    } else {
        state.smoothing_lengths.clone()
    };
    let half_state = recover_state(
        positions,
        state.masses.clone(),
        smoothing_lengths,
        state.domain,
        state.gamma,
        &half_momentum,
        &half_energy,
        minimum_specific_internal_energy,
    )?;
    let rates = hydro_mfm_spatial_rates_2d(&half_state)?;
    let final_momentum = kick_vectors(&half_momentum, &rates.momentum, 0.5 * timestep);
    let final_energy = kick_scalars(&half_energy, &rates.total_energy, 0.5 * timestep);
    let final_state = recover_state(
        half_state.positions,
        half_state.masses,
        half_state.smoothing_lengths,
        half_state.domain,
        half_state.gamma,
        &final_momentum,
        &final_energy,
        minimum_specific_internal_energy,
    )?;
    Ok(HydroKdkResult2d {
        state: final_state,
        rates,
    })
}

/// Prepare public-C individual-bin hydro evolution from a fully synchronized state.
///
/// # Errors
///
/// Returns an error for invalid state, rates, timeline bins, or kick arithmetic.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]
pub fn begin_public_hydro_initial_hierarchy_2d(
    state: &HydroMfmState2d,
    old_rates: &HydroMfmRates2d,
    timebins: &[PublicHydroInitialTimebin2d],
    time_begin: f64,
    time_max: f64,
    minimum_specific_internal_energy: f64,
) -> Result<PublicHydroInitialHierarchy2d, HydroEvolution2dError> {
    state.validate()?;
    validate_rate_lengths(old_rates, state.positions.len())?;
    if timebins.len() != state.positions.len() {
        return Err(HydroEvolution2dError::MismatchedLength {
            field: "initial_timebins",
            expected: state.positions.len(),
            actual: timebins.len(),
        });
    }
    if !minimum_specific_internal_energy.is_finite() || minimum_specific_internal_energy < 0.0 {
        return Err(invalid(
            None,
            "minimum_specific_internal_energy",
            minimum_specific_internal_energy,
        ));
    }
    let steps = timebins.iter().map(|bin| bin.ticks).collect::<Vec<_>>();
    let timeline = IndividualParticleTimeline::from_initial_steps(time_begin, time_max, &steps)?;
    let tick_duration = timeline.duration_for_ticks(1);
    for (i, bin) in timebins.iter().enumerate() {
        let requested = (bin.bounded_timestep / tick_duration).trunc();
        let expected_raw = if requested.is_finite() && requested >= 0.0 {
            (requested as u64).max(2)
        } else {
            0
        };
        if bin.time_bin != bin.ticks.ilog2()
            || !bin.bounded_timestep.is_finite()
            || bin.bounded_timestep <= 0.0
            || bin.raw_ticks != expected_raw
            || bin.raw_ticks < bin.ticks
            || bin.raw_ticks >= 2 * bin.ticks
            || !legacy_float_equal(bin.duration, timeline.duration_for_ticks(bin.ticks))
        {
            return Err(invalid(Some(i), "initial_timebin", bin.duration));
        }
    }
    let primitive_cache = state.primitive_columns()?;
    let moment_cache =
        inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    let gradient_cache = primitive_gradients(state, &primitive_cache, &moment_cache)?;
    let face_closure_cache = face_closure_diagnostics_2d(
        &state.positions,
        &state.masses,
        &state.smoothing_lengths,
        state.domain,
    )?;
    let mut actual_velocities = Vec::with_capacity(state.positions.len());
    let mut actual_internal_energy = Vec::with_capacity(state.positions.len());
    for (i, bin) in timebins.iter().enumerate() {
        let half = 0.5 * timeline.duration_for_ticks(bin.ticks);
        actual_velocities.push(state.velocities[i] + old_rates.acceleration[i] * half);
        actual_internal_energy.push(limited_internal_energy_update_2d(
            state.specific_internal_energy[i],
            old_rates.specific_internal_energy[i],
            half,
            minimum_specific_internal_energy,
        )?);
    }
    let count = state.positions.len();
    Ok(PublicHydroInitialHierarchy2d {
        timeline,
        start: state.clone(),
        old_rates: old_rates.clone(),
        actual_internal_energy,
        drift: PublicHydroDriftState2d {
            positions: state.positions.clone(),
            actual_velocities,
            predicted_velocities: state.velocities.clone(),
            predicted_specific_internal_energy: state.specific_internal_energy.clone(),
            predicted_density: primitive_cache.density.clone(),
            predicted_smoothing_lengths: state.smoothing_lengths.clone(),
        },
        predictor_ticks: vec![0; count],
        primitive_cache,
        gradient_cache,
        moment_cache,
        face_closure_cache,
        minimum_specific_internal_energy,
        awaiting_second_kick: false,
        prepared_next_drift: false,
        refreshed_cache_tick: None,
        pending_wakeup: vec![false; count],
    })
}

impl PublicHydroInitialHierarchy2d {
    #[must_use]
    pub fn current_tick(&self) -> u64 {
        self.timeline.current_tick()
    }

    #[must_use]
    pub fn current_time(&self) -> f64 {
        self.timeline.current_time()
    }

    /// Physical time of the next occupied-bin synchronization.
    ///
    /// # Errors
    ///
    /// Returns an error when the timeline is finished or has no occupied bin.
    pub fn next_sync_time(&self) -> Result<f64, HydroEvolution2dError> {
        let next_tick = self.timeline.next_sync_tick()?;
        Ok(self.timeline.current_time()
            + self
                .timeline
                .duration_for_ticks(next_tick - self.timeline.current_tick()))
    }

    #[must_use]
    pub fn active_mask(&self) -> Vec<bool> {
        self.timeline.active_mask()
    }

    #[must_use]
    pub fn current_drift_state(&self) -> &PublicHydroDriftState2d {
        &self.drift
    }

    /// Convert a requested output time to the public integer timeline.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid nonterminal time or terminal snap.
    pub fn output_time_to_integer_tick(
        &self,
        output_time: f64,
        terminal: bool,
    ) -> Result<u64, HydroEvolution2dError> {
        Ok(self
            .timeline
            .output_time_to_integer_tick(output_time, terminal)?)
    }

    /// Return the physical time encoded by an output tick.
    ///
    /// # Errors
    ///
    /// Returns an error for a tick beyond the public timebase.
    pub fn output_physical_time_at_tick(&self, tick: u64) -> Result<f64, HydroEvolution2dError> {
        Ok(self.timeline.physical_time_at_tick(tick)?)
    }

    /// Rebin, process wakeups, and apply the next first half-kicks once.
    ///
    /// The hierarchy then owns a mutable scheduled-drift interval. Call
    /// [`Self::drift_prepared_to_output_tick`] for every crossed output in
    /// ascending integer-tick order, followed by
    /// [`Self::finish_prepared_next_sync`].
    ///
    /// # Errors
    ///
    /// Returns an error for invalid phase, bounds, wakeup, or kick arithmetic.
    pub fn prepare_next_drift(
        &mut self,
        active_bounds: &[Option<f64>],
    ) -> Result<u64, HydroEvolution2dError> {
        let mut next = self.clone();
        next.prepare_next_first_kicks_in_place(active_bounds, true)?;
        let next_tick = next.timeline.next_sync_tick()?;
        *self = next;
        Ok(next_tick)
    }

    /// Mutably drift every predictor to a crossed integer output tick.
    ///
    /// Repeated calls intentionally preserve the public non-semigroup
    /// internal-energy, density, and H clamps across output boundaries.
    ///
    /// # Errors
    ///
    /// Returns an error unless a next drift is prepared, ticks are monotonic,
    /// and the requested tick is no later than the next force sync.
    pub fn drift_prepared_to_output_tick(
        &mut self,
        output_tick: u64,
    ) -> Result<PublicHydroDriftState2d, HydroEvolution2dError> {
        let mut next = self.clone();
        if !next.prepared_next_drift {
            return Err(invalid(
                None,
                "hierarchy_unprepared_output_drift",
                next.current_time(),
            ));
        }
        let next_sync = next.timeline.next_sync_tick()?;
        if output_tick < next.timeline.current_tick() || output_tick > next_sync {
            return Err(invalid(
                None,
                "scheduled_output_tick",
                next.timeline.physical_time_at_tick(output_tick)?,
            ));
        }
        for i in 0..next.start.positions.len() {
            next.drift_particle_to_tick(i, output_tick)?;
        }
        let drift = next.drift.clone();
        *self = next;
        Ok(drift)
    }

    /// Resume a prepared scheduled drift to its unchanged next force sync.
    ///
    /// # Errors
    ///
    /// Returns an error unless the interval was prepared or for drift/timeline
    /// arithmetic.
    pub fn finish_prepared_next_sync(
        &mut self,
    ) -> Result<PublicHydroHierarchySync2d, HydroEvolution2dError> {
        let mut next = self.clone();
        if !next.prepared_next_drift {
            return Err(invalid(
                None,
                "hierarchy_unprepared_next_sync",
                next.current_time(),
            ));
        }
        let arriving = next.timeline.advance_to_next_sync()?;
        for (i, &is_active) in arriving.iter().enumerate() {
            if is_active {
                next.drift_particle_to_current(i)?;
            }
        }
        next.prepared_next_drift = false;
        next.awaiting_second_kick = true;
        next.refreshed_cache_tick = None;
        let sync = next.sync(arriving);
        *self = next;
        Ok(sync)
    }

    #[must_use]
    pub fn retained_rates(&self) -> &HydroMfmRates2d {
        &self.old_rates
    }

    #[must_use]
    pub fn predictor_ticks(&self) -> &[u64] {
        &self.predictor_ticks
    }

    #[must_use]
    pub fn predicted_primitive_cache(&self) -> &HydroPrimitiveColumns2d {
        &self.primitive_cache
    }

    #[must_use]
    pub fn actual_specific_internal_energy(&self) -> &[f64] {
        &self.actual_internal_energy
    }

    /// Drift the earliest occupied bin from the initial synchronization.
    ///
    /// # Errors
    ///
    /// Returns an error outside the initial event or for predictor arithmetic.
    pub fn drift_to_first_sync(
        &mut self,
    ) -> Result<PublicHydroHierarchySync2d, HydroEvolution2dError> {
        if self.timeline.current_tick() != 0 {
            return Err(invalid(
                None,
                "initial_hierarchy_tick",
                self.timeline.current_time(),
            ));
        }
        let active = self.timeline.advance_to_next_sync()?;
        for (i, &arrives) in active.iter().enumerate() {
            if arrives {
                self.drift_particle_to_current(i)?;
            }
        }
        self.awaiting_second_kick = true;
        self.refreshed_cache_tick = None;
        Ok(self.sync(active))
    }

    /// Lazily drift one encountered neighbor to the current force epoch.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid index or predictor arithmetic.
    pub fn drift_neighbor_to_current(&mut self, index: usize) -> Result<(), HydroEvolution2dError> {
        self.drift_particle_to_current(index)
    }

    /// Materialize all lazy predictors at the current event.
    ///
    /// # Errors
    ///
    /// Returns an error for predictor arithmetic.
    pub fn drift_all_particles_to_current(&mut self) -> Result<(), HydroEvolution2dError> {
        for i in 0..self.start.positions.len() {
            self.drift_particle_to_current(i)?;
        }
        Ok(())
    }

    /// Build the retained mixed-epoch predicted state.
    ///
    /// # Errors
    ///
    /// Returns an error if predicted columns are non-physical.
    pub fn predicted_state(&self) -> Result<HydroMfmState2d, HydroEvolution2dError> {
        let state = HydroMfmState2d {
            positions: self.drift.positions.clone(),
            masses: self.start.masses.clone(),
            velocities: self.drift.predicted_velocities.clone(),
            specific_internal_energy: self.drift.predicted_specific_internal_energy.clone(),
            smoothing_lengths: self.drift.predicted_smoothing_lengths.clone(),
            domain: self.start.domain,
            gamma: self.start.gamma,
        };
        state.validate()?;
        Ok(state)
    }

    fn projected_search_state_at_current(&self) -> Result<HydroMfmState2d, HydroEvolution2dError> {
        let mut state = self.predicted_state()?;
        let target = self.timeline.current_tick();
        for i in 0..state.positions.len() {
            let source = self.predictor_ticks[i];
            if source > target {
                return Err(invalid(
                    Some(i),
                    "hierarchy_predictor_tick",
                    self.current_time(),
                ));
            }
            if source == target {
                continue;
            }
            let segment = self.timeline.duration_for_ticks(target - source);
            let velocity = self.drift.actual_velocities[i];
            state.positions[i] = state
                .domain
                .wrap(state.positions[i] + Vector2::new(velocity.x, velocity.y) * segment)?;
            let divergence = (self.old_rates.velocity_divergence[i] * segment).clamp(-0.3, 0.3);
            state.smoothing_lengths[i] *= (0.5 * divergence).exp();
        }
        state.validate()?;
        Ok(state)
    }

    fn required_active_neighbors(
        &self,
        active: &[bool],
    ) -> Result<Vec<bool>, HydroEvolution2dError> {
        let state = self.projected_search_state_at_current()?;
        let pairs = interacting_pairs_for_targets_2d(
            &state.positions,
            &state.smoothing_lengths,
            state.domain,
            active,
        )?;
        let mut required = active.to_vec();
        for pair in pairs {
            if active[pair.i] {
                required[pair.j] = true;
            }
            if active[pair.j] {
                required[pair.i] = true;
            }
        }
        Ok(required)
    }

    /// Drift required neighbors and refresh only arriving active cache slots.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid adaptive-H controls or cache geometry.
    pub fn refresh_arriving_active_caches(
        &mut self,
        desired_neighbors: f64,
        neighbor_tolerance: f64,
    ) -> Result<(), HydroEvolution2dError> {
        if !self.awaiting_second_kick {
            return Err(invalid(None, "hierarchy_force_phase", self.current_time()));
        }
        self.refreshed_cache_tick = None;
        let active = self.timeline.active_mask();
        let count = self.start.positions.len();
        for _ in 0..=count {
            let required = self.required_active_neighbors(&active)?;
            for (i, &needed) in required.iter().enumerate() {
                if needed {
                    self.drift_particle_to_current(i)?;
                }
            }
            let mut state = self.projected_search_state_at_current()?;
            let old_support = state.smoothing_lengths.clone();
            refresh_hydro_active_target_caches_2d(
                &mut state,
                &mut self.primitive_cache,
                &mut self.gradient_cache,
                &mut self.moment_cache,
                &mut self.face_closure_cache,
                &active,
                desired_neighbors,
                neighbor_tolerance,
            )?;
            for (i, &is_active) in active.iter().enumerate() {
                if is_active {
                    self.drift.predicted_density[i] = self.primitive_cache.density[i];
                    self.drift.predicted_smoothing_lengths[i] = state.smoothing_lengths[i];
                }
            }
            let expanded = active
                .iter()
                .enumerate()
                .any(|(i, &yes)| yes && state.smoothing_lengths[i] > old_support[i]);
            let required = self.required_active_neighbors(&active)?;
            let has_stale_neighbor = required
                .iter()
                .zip(&self.predictor_ticks)
                .any(|(&needed, &tick)| needed && tick != self.timeline.current_tick());
            if !expanded || !has_stale_neighbor {
                self.refreshed_cache_tick = Some(self.timeline.current_tick());
                return Ok(());
            }
        }
        Err(invalid(
            None,
            "hierarchy_neighbor_materialization",
            self.current_time(),
        ))
    }

    /// Evaluate fresh rates for active targets while retaining inactive slots.
    ///
    /// # Errors
    ///
    /// Returns an error unless active caches were refreshed at this event.
    pub fn evaluate_arriving_active_rates(
        &self,
    ) -> Result<PublicHydroActiveRateResult2d, HydroEvolution2dError> {
        if !self.awaiting_second_kick
            || self.refreshed_cache_tick != Some(self.timeline.current_tick())
        {
            return Err(invalid(
                None,
                "hierarchy_unrefreshed_active_cache",
                self.current_time(),
            ));
        }
        hydro_mfm_active_target_rates_with_cache_2d(
            &self.projected_search_state_at_current()?,
            &self.primitive_cache,
            &self.gradient_cache,
            &self.moment_cache,
            &self.face_closure_cache,
            &self.old_rates,
            &self.timeline.active_mask(),
        )
    }

    /// Apply endpoint half-kicks only to particles active at this sync.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid retained inactive rates or kick arithmetic.
    pub fn finish_arriving_active_kicks(
        &mut self,
        endpoint: PublicHydroActiveRateResult2d,
    ) -> Result<PublicHydroHierarchyKickResult2d, HydroEvolution2dError> {
        let mut next = self.clone();
        let result = next.finish_arriving_active_kicks_in_place(endpoint)?;
        *self = next;
        Ok(result)
    }

    fn finish_arriving_active_kicks_in_place(
        &mut self,
        endpoint: PublicHydroActiveRateResult2d,
    ) -> Result<PublicHydroHierarchyKickResult2d, HydroEvolution2dError> {
        if !self.awaiting_second_kick {
            return Err(invalid(
                None,
                "hierarchy_second_kick_phase",
                self.current_time(),
            ));
        }
        let count = self.start.positions.len();
        validate_rate_lengths(&endpoint.rates, count)?;
        if endpoint.wakeup.len() != count {
            return Err(HydroEvolution2dError::MismatchedLength {
                field: "wakeup",
                expected: count,
                actual: endpoint.wakeup.len(),
            });
        }
        let active = self.timeline.active_mask();
        for (i, &is_active) in active.iter().enumerate() {
            if !is_active && !same_particle_rate(&endpoint.rates, &self.old_rates, i) {
                return Err(invalid(
                    Some(i),
                    "endpoint_inactive_retained_rate",
                    f64::NAN,
                ));
            }
            if !is_active {
                continue;
            }
            let half = 0.5
                * self
                    .timeline
                    .duration_for_ticks(self.timeline.step_ticks()[i]);
            self.drift.actual_velocities[i] =
                self.drift.actual_velocities[i] + endpoint.rates.acceleration[i] * half;
            self.actual_internal_energy[i] = limited_internal_energy_update_2d(
                self.actual_internal_energy[i],
                endpoint.rates.specific_internal_energy[i],
                half,
                self.minimum_specific_internal_energy,
            )?;
            self.drift.predicted_velocities[i] = self.drift.actual_velocities[i];
            self.drift.predicted_specific_internal_energy[i] = self.actual_internal_energy[i];
            self.primitive_cache.pressure[i] = (self.start.gamma - 1.0)
                * self.primitive_cache.density[i]
                * self.actual_internal_energy[i];
        }
        let state = HydroMfmState2d {
            positions: self.drift.positions.clone(),
            masses: self.start.masses.clone(),
            velocities: self.drift.actual_velocities.clone(),
            specific_internal_energy: self.actual_internal_energy.clone(),
            smoothing_lengths: self.drift.predicted_smoothing_lengths.clone(),
            domain: self.start.domain,
            gamma: self.start.gamma,
        };
        state.validate()?;
        self.old_rates = endpoint.rates.clone();
        self.pending_wakeup.clone_from(&endpoint.wakeup);
        self.awaiting_second_kick = false;
        self.prepared_next_drift = false;
        self.refreshed_cache_tick = None;
        Ok(PublicHydroHierarchyKickResult2d {
            state,
            rates: endpoint.rates,
            wakeup: endpoint.wakeup,
        })
    }

    /// Rebin active particles, process wakeups, kick, and drift to the next sync.
    ///
    /// # Errors
    ///
    /// Returns an error for timeline, wakeup, kick, or drift arithmetic.
    pub fn begin_next_sync(
        &mut self,
        active_bounds: &[Option<f64>],
    ) -> Result<PublicHydroHierarchySync2d, HydroEvolution2dError> {
        self.begin_next_sync_impl(active_bounds, true)
    }

    /// Advance without wakeups, rejecting any pending wakeup flags.
    ///
    /// # Errors
    ///
    /// Returns an error if wakeups are pending or for timeline arithmetic.
    pub fn begin_next_sync_without_wakeups(
        &mut self,
        active_bounds: &[Option<f64>],
    ) -> Result<PublicHydroHierarchySync2d, HydroEvolution2dError> {
        self.begin_next_sync_impl(active_bounds, false)
    }

    fn begin_next_sync_impl(
        &mut self,
        active_bounds: &[Option<f64>],
        process_wakeups: bool,
    ) -> Result<PublicHydroHierarchySync2d, HydroEvolution2dError> {
        let mut next = self.clone();
        let result = next.begin_next_sync_in_place(active_bounds, process_wakeups)?;
        *self = next;
        Ok(result)
    }

    fn begin_next_sync_in_place(
        &mut self,
        active_bounds: &[Option<f64>],
        process_wakeups: bool,
    ) -> Result<PublicHydroHierarchySync2d, HydroEvolution2dError> {
        self.prepare_next_first_kicks_in_place(active_bounds, process_wakeups)?;
        let arriving = self.timeline.advance_to_next_sync()?;
        for (i, &is_active) in arriving.iter().enumerate() {
            if is_active {
                self.drift_particle_to_current(i)?;
            }
        }
        self.prepared_next_drift = false;
        self.awaiting_second_kick = true;
        self.refreshed_cache_tick = None;
        Ok(self.sync(arriving))
    }

    fn prepare_next_first_kicks_in_place(
        &mut self,
        active_bounds: &[Option<f64>],
        process_wakeups: bool,
    ) -> Result<(), HydroEvolution2dError> {
        if self.awaiting_second_kick || self.prepared_next_drift {
            return Err(invalid(
                None,
                "hierarchy_first_kick_phase",
                self.current_time(),
            ));
        }
        let active = self.timeline.active_mask();
        self.timeline.reassign_active_bounds(active_bounds)?;
        if process_wakeups {
            for transition in self.timeline.apply_wakeups(&self.pending_wakeup)? {
                self.reverse_wakeup_kick_to_current(
                    transition.index,
                    transition.old_endpoint_tick,
                    transition.new_step_active_at_current,
                )?;
            }
            self.pending_wakeup.fill(false);
        } else if self.pending_wakeup.iter().any(|&flag| flag) {
            return Err(invalid(
                None,
                "hierarchy_pending_wakeup",
                self.current_time(),
            ));
        }
        for (i, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            let half = 0.5
                * self
                    .timeline
                    .duration_for_ticks(self.timeline.step_ticks()[i]);
            self.drift.actual_velocities[i] =
                self.drift.actual_velocities[i] + self.old_rates.acceleration[i] * half;
            self.actual_internal_energy[i] = limited_internal_energy_update_2d(
                self.actual_internal_energy[i],
                self.old_rates.specific_internal_energy[i],
                half,
                self.minimum_specific_internal_energy,
            )?;
        }
        self.prepared_next_drift = true;
        Ok(())
    }

    fn reverse_wakeup_kick_to_current(
        &mut self,
        i: usize,
        old_endpoint_tick: u64,
        new_step_active_at_current: bool,
    ) -> Result<(), HydroEvolution2dError> {
        let current = self.timeline.current_tick();
        if self.predictor_ticks[i] != current {
            return Err(invalid(
                Some(i),
                "wakeup_neighbor_predictor_tick",
                self.current_time(),
            ));
        }
        let reverse_start = old_endpoint_tick.max(self.predictor_ticks[i]);
        if new_step_active_at_current && current < reverse_start {
            let duration = -self.timeline.duration_for_ticks(reverse_start - current);
            self.drift.actual_velocities[i] =
                self.drift.actual_velocities[i] + self.old_rates.acceleration[i] * duration;
            self.actual_internal_energy[i] = limited_internal_energy_update_signed_public_2d(
                self.actual_internal_energy[i],
                self.old_rates.specific_internal_energy[i],
                duration,
                self.minimum_specific_internal_energy,
            )?;
            self.drift.predicted_velocities[i] = self.drift.actual_velocities[i];
            self.drift.predicted_specific_internal_energy[i] = self.actual_internal_energy[i];
            self.primitive_cache.pressure[i] = (self.start.gamma - 1.0)
                * self.primitive_cache.density[i]
                * self.actual_internal_energy[i];
        }
        Ok(())
    }

    fn drift_particle_to_current(&mut self, i: usize) -> Result<(), HydroEvolution2dError> {
        self.drift_particle_to_tick(i, self.timeline.current_tick())
    }

    fn drift_particle_to_tick(
        &mut self,
        i: usize,
        target: u64,
    ) -> Result<(), HydroEvolution2dError> {
        if i >= self.start.positions.len() {
            return Err(invalid(Some(i), "hierarchy_particle_index", f64::NAN));
        }
        let source = self.predictor_ticks[i];
        if source > target || target > crate::LEGACY_TIMEBASE_TICKS {
            return Err(invalid(
                Some(i),
                "hierarchy_predictor_tick",
                self.timeline.physical_time_at_tick(target)?,
            ));
        }
        let segment = self.timeline.duration_for_ticks(target - source);
        if segment == 0.0 {
            return Ok(());
        }
        let velocity = self.drift.actual_velocities[i];
        self.drift.positions[i] = self
            .start
            .domain
            .wrap(self.drift.positions[i] + Vector2::new(velocity.x, velocity.y) * segment)?;
        self.drift.predicted_velocities[i] =
            self.drift.predicted_velocities[i] + self.old_rates.acceleration[i] * segment;
        self.drift.predicted_specific_internal_energy[i] = limited_internal_energy_update_2d(
            self.drift.predicted_specific_internal_energy[i],
            self.old_rates.specific_internal_energy[i],
            segment,
            self.minimum_specific_internal_energy,
        )?;
        let divergence = (self.old_rates.velocity_divergence[i] * segment).clamp(-0.3, 0.3);
        self.drift.predicted_density[i] *= (-divergence).exp();
        // Two-dimensional H evolves as rho^(-1/2), not rho^(-1/3).
        self.drift.predicted_smoothing_lengths[i] *= (0.5 * divergence).exp();
        self.primitive_cache.density[i] = self.drift.predicted_density[i];
        self.primitive_cache.pressure[i] = (self.start.gamma - 1.0)
            * self.drift.predicted_density[i]
            * self.drift.predicted_specific_internal_energy[i];
        self.predictor_ticks[i] = target;
        Ok(())
    }

    fn sync(&self, active: Vec<bool>) -> PublicHydroHierarchySync2d {
        PublicHydroHierarchySync2d {
            tick: self.timeline.current_tick(),
            time: self.timeline.current_time(),
            active,
            drift: self.drift.clone(),
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn refresh_hydro_active_target_caches_2d(
    state: &mut HydroMfmState2d,
    primitive: &mut HydroPrimitiveColumns2d,
    gradients: &mut PrimitiveGradients2d,
    moments: &mut [InverseMoment2d],
    closure: &mut [FaceClosure2d],
    active: &[bool],
    desired_neighbors: f64,
    neighbor_tolerance: f64,
) -> Result<(), HydroEvolution2dError> {
    let count = state.positions.len();
    validate_lengths(
        count,
        &[
            ("active_mask", active.len()),
            ("primitive_density", primitive.density.len()),
            ("primitive_dhsml_factor", primitive.dhsml_factor.len()),
            ("primitive_pressure", primitive.pressure.len()),
            ("gradient_density", gradients.density.len()),
            ("gradient_pressure", gradients.pressure.len()),
            ("gradient_velocity_x", gradients.velocity_x.len()),
            ("gradient_velocity_y", gradients.velocity_y.len()),
            ("gradient_velocity_z", gradients.velocity_z.len()),
            ("inverse_moments", moments.len()),
            ("face_closure", closure.len()),
        ],
    )?;
    let solved = solve_public_c_smoothing_lengths_from_seeds_for_targets_2d(
        &state.positions,
        &state.masses,
        &state.smoothing_lengths,
        state.domain,
        desired_neighbors,
        neighbor_tolerance,
        active,
    )?;
    for (i, &is_active) in active.iter().enumerate() {
        if is_active {
            let estimate = solved[i]
                .ok_or_else(|| invalid(Some(i), "missing active smoothing estimate", f64::NAN))?;
            state.smoothing_lengths[i] = estimate.smoothing_length;
            primitive.density[i] = estimate.estimate.density;
            primitive.dhsml_factor[i] = estimate.estimate.dhsml_factor;
            primitive.pressure[i] =
                (state.gamma - 1.0) * primitive.density[i] * state.specific_internal_energy[i];
        }
    }
    // The dense geometry calls are deterministic references; only active cache
    // slots are committed, preserving inactive retained values bitwise.
    let fresh_moments =
        inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    for (i, &is_active) in active.iter().enumerate() {
        if is_active {
            moments[i] = fresh_moments[i];
        }
    }
    let fresh_closure = face_closure_diagnostics_2d(
        &state.positions,
        &state.masses,
        &state.smoothing_lengths,
        state.domain,
    )?;
    for (i, &is_active) in active.iter().enumerate() {
        if is_active {
            closure[i] = fresh_closure[i];
        }
    }
    let fresh_gradients = primitive_gradients(state, primitive, moments)?;
    for (i, &is_active) in active.iter().enumerate() {
        if is_active {
            gradients.density[i] = fresh_gradients.density[i];
            gradients.pressure[i] = fresh_gradients.pressure[i];
            gradients.velocity_x[i] = fresh_gradients.velocity_x[i];
            gradients.velocity_y[i] = fresh_gradients.velocity_y[i];
            gradients.velocity_z[i] = fresh_gradients.velocity_z[i];
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_active_hydro_target_batch_2d(
    state: &HydroMfmState2d,
    primitive: &HydroPrimitiveColumns2d,
    gradients: &PrimitiveGradients2d,
    moments: &[InverseMoment2d],
    closure: &[FaceClosure2d],
    retained: &HydroMfmRates2d,
    active: &[bool],
    target: usize,
    neighbors: &[(usize, usize)],
) -> Result<ActiveHydroTargetBatch2d, (usize, HydroEvolution2dError)> {
    let mut batch = ActiveHydroTargetBatch2d {
        target,
        momentum: Vector3::ZERO,
        total_energy: 0.0,
        maximum_signal_speed: (state.gamma * primitive.pressure[target]
            / primitive.density[target])
            .sqrt(),
        wakeup_neighbors: Vec::new(),
        entropic_pair_count: 0,
        method_counts: [0; 4],
    };
    for &(neighbor, ordinal) in neighbors {
        let contribution = (|| {
            let point = |i: usize| MeshlessPoint2d {
                position: state.positions[i],
                mass: state.masses[i],
                density: primitive.density[i],
                smoothing_length: state.smoothing_lengths[i],
                inverse_moment: moments[i].matrix,
                condition_number: moments[i].condition_number,
            };
            let face = meshless_face_geometry_2d(point(target), point(neighbor), state.domain)?;
            let flux = solve_pair_with_retries(
                state, primitive, gradients, closure, target, neighbor, face,
            )?;
            let displacement = state
                .domain
                .displacement(state.positions[target], state.positions[neighbor])?;
            let (energy, entropic) = apply_entropic_pdv_energy_2d(
                flux.total_energy * face.area,
                flux,
                face,
                displacement,
                state,
                primitive,
                moments,
                closure,
                target,
                neighbor,
            )?;
            let radial = displacement / displacement.norm();
            let approach = planar_dot(
                state.velocities[target] - state.velocities[neighbor],
                radial,
            )
            .min(0.0);
            let sound_i =
                (state.gamma * primitive.pressure[target] / primitive.density[target]).sqrt();
            let sound_j =
                (state.gamma * primitive.pressure[neighbor] / primitive.density[neighbor]).sqrt();
            Ok::<_, HydroEvolution2dError>((
                flux.momentum * face.area,
                energy,
                sound_i + sound_j - approach,
                entropic,
                flux.method,
            ))
        })()
        .map_err(|error| (ordinal, error))?;
        batch.momentum = batch.momentum + contribution.0;
        batch.total_energy += contribution.1;
        batch.maximum_signal_speed = batch.maximum_signal_speed.max(contribution.2);
        batch.entropic_pair_count += usize::from(contribution.3);
        batch.method_counts[method_index(contribution.4)] += 1;
        if !active[neighbor] && contribution.2 > 4.1 * retained.maximum_signal_speed[neighbor] {
            batch.wakeup_neighbors.push(neighbor);
        }
    }
    Ok(batch)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_active_hydro_target_batches_2d(
    state: &HydroMfmState2d,
    primitive: &HydroPrimitiveColumns2d,
    gradients: &PrimitiveGradients2d,
    moments: &[InverseMoment2d],
    closure: &[FaceClosure2d],
    retained: &HydroMfmRates2d,
    active: &[bool],
    pairs: &[InteractionPair2d],
    requested_workers: usize,
) -> Result<Vec<ActiveHydroTargetBatch2d>, HydroEvolution2dError> {
    let mut neighbors = vec![Vec::new(); state.positions.len()];
    let mut directed_ordinal = 0_usize;
    for pair in pairs {
        for (target, neighbor) in [(pair.i, pair.j), (pair.j, pair.i)] {
            if active[target] {
                neighbors[target].push((neighbor, directed_ordinal));
                directed_ordinal += 1;
            }
        }
    }
    let active_targets = active
        .iter()
        .enumerate()
        .filter_map(|(i, &is_active)| is_active.then_some(i))
        .collect::<Vec<_>>();
    if active_targets.is_empty() {
        return Ok(Vec::new());
    }
    let workers = requested_workers.max(1).min(active_targets.len());
    let chunk_size = active_targets.len().div_ceil(workers);
    let chunks = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for targets in active_targets.chunks(chunk_size) {
            let neighbor_lists = &neighbors;
            handles.push(scope.spawn(move || {
                targets
                    .iter()
                    .map(|&target| {
                        evaluate_active_hydro_target_batch_2d(
                            state,
                            primitive,
                            gradients,
                            moments,
                            closure,
                            retained,
                            active,
                            target,
                            &neighbor_lists[target],
                        )
                    })
                    .collect::<Vec<_>>()
            }));
        }
        handles
            .into_iter()
            .map(std::thread::ScopedJoinHandle::join)
            .collect::<Vec<_>>()
    });
    let mut batches = Vec::with_capacity(active_targets.len());
    let mut earliest_failure: Option<(usize, HydroEvolution2dError)> = None;
    for chunk in chunks {
        let chunk =
            chunk.map_err(|_| invalid(None, "active_target_force_worker_panicked", f64::NAN))?;
        for batch in chunk {
            match batch {
                Ok(batch) => batches.push(batch),
                Err((ordinal, error)) => {
                    if earliest_failure
                        .as_ref()
                        .is_none_or(|(earliest, _)| ordinal < *earliest)
                    {
                        earliest_failure = Some((ordinal, error));
                    }
                }
            }
        }
    }
    if let Some((_, error)) = earliest_failure {
        return Err(error);
    }
    Ok(batches)
}

fn hydro_active_target_worker_count() -> usize {
    std::env::var("GIZMO_ACTIVE_TARGET_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&workers| workers > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
        .min(MAX_ACTIVE_TARGET_WORKERS)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn hydro_mfm_active_target_rates_with_cache_2d(
    state: &HydroMfmState2d,
    primitive: &HydroPrimitiveColumns2d,
    gradients: &PrimitiveGradients2d,
    moments: &[InverseMoment2d],
    closure: &[FaceClosure2d],
    retained: &HydroMfmRates2d,
    active: &[bool],
) -> Result<PublicHydroActiveRateResult2d, HydroEvolution2dError> {
    state.validate()?;
    validate_rate_lengths(retained, state.positions.len())?;
    let count = state.positions.len();
    validate_lengths(
        count,
        &[
            ("active_mask", active.len()),
            ("primitive_density", primitive.density.len()),
            ("primitive_dhsml_factor", primitive.dhsml_factor.len()),
            ("primitive_pressure", primitive.pressure.len()),
            ("inverse_moments", moments.len()),
            ("face_closure", closure.len()),
        ],
    )?;
    let planar = state
        .velocities
        .iter()
        .map(|value| Vector2::new(value.x, value.y))
        .collect::<Vec<_>>();
    let divergence = particle_divergence_at_hsml_for_targets_2d(
        &state.positions,
        &planar,
        &state.smoothing_lengths,
        &primitive.dhsml_factor,
        state.domain,
        active,
    )?;
    let mut updated = retained.clone();
    for (i, &is_active) in active.iter().enumerate() {
        if is_active {
            updated.momentum[i] = Vector3::ZERO;
            updated.total_energy[i] = 0.0;
            updated.acceleration[i] = Vector3::ZERO;
            updated.specific_internal_energy[i] = 0.0;
            updated.maximum_signal_speed[i] =
                (state.gamma * primitive.pressure[i] / primitive.density[i]).sqrt();
            updated.velocity_divergence[i] = divergence[i];
        }
    }
    let pairs = interacting_pairs_for_targets_2d(
        &state.positions,
        &state.smoothing_lengths,
        state.domain,
        active,
    )?;
    let mut wakeup = vec![false; count];
    let mut method_counts = [0_usize; 4];
    let mut entropic_count = 0_usize;
    let batches = evaluate_active_hydro_target_batches_2d(
        state,
        primitive,
        gradients,
        moments,
        closure,
        retained,
        active,
        &pairs,
        hydro_active_target_worker_count(),
    )?;
    let mut directed_pair_count = 0_usize;
    for batch in batches {
        let target = batch.target;
        directed_pair_count += batch.method_counts.iter().sum::<usize>();
        updated.momentum[target] = batch.momentum;
        updated.total_energy[target] = batch.total_energy;
        updated.maximum_signal_speed[target] = batch.maximum_signal_speed;
        entropic_count += batch.entropic_pair_count;
        for (sum, count) in method_counts.iter_mut().zip(batch.method_counts) {
            *sum += count;
        }
        for neighbor in batch.wakeup_neighbors {
            wakeup[neighbor] = true;
        }
    }
    for (i, &is_active) in active.iter().enumerate() {
        if is_active {
            updated.acceleration[i] = updated.momentum[i] / state.masses[i];
            updated.specific_internal_energy[i] = (updated.total_energy[i]
                - state.velocities[i].dot(updated.momentum[i]))
                / state.masses[i];
        }
    }
    updated.pair_count = directed_pair_count;
    updated.entropic_pair_count = entropic_count;
    updated.hllc_pair_count = method_counts[0];
    updated.kt_pair_count = method_counts[1];
    updated.exact_pair_count = method_counts[2];
    updated.vacuum_pair_count = method_counts[3];
    Ok(PublicHydroActiveRateResult2d {
        rates: updated,
        wakeup,
    })
}

fn same_particle_rate(left: &HydroMfmRates2d, right: &HydroMfmRates2d, i: usize) -> bool {
    left.momentum[i] == right.momentum[i]
        && left.total_energy[i].to_bits() == right.total_energy[i].to_bits()
        && left.acceleration[i] == right.acceleration[i]
        && left.specific_internal_energy[i].to_bits() == right.specific_internal_energy[i].to_bits()
        && left.maximum_signal_speed[i].to_bits() == right.maximum_signal_speed[i].to_bits()
        && left.velocity_divergence[i].to_bits() == right.velocity_divergence[i].to_bits()
}

fn limited_internal_energy_update_2d(
    previous: f64,
    rate: f64,
    duration: f64,
    floor: f64,
) -> Result<f64, HydroEvolution2dError> {
    if !duration.is_finite() || duration < 0.0 {
        return Err(invalid(None, "internal_energy_kick_duration", duration));
    }
    limited_internal_energy_update_signed_public_2d(previous, rate, duration, floor)
}

fn limited_internal_energy_update_signed_public_2d(
    previous: f64,
    rate: f64,
    duration: f64,
    floor: f64,
) -> Result<f64, HydroEvolution2dError> {
    let value = (previous + rate * duration).max(0.5 * previous).max(floor);
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(None, "specific_internal_energy", value));
    }
    Ok(value)
}

#[allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn apply_entropic_pdv_energy_2d(
    raw_energy: f64,
    flux: EulerMfmFlux2d,
    face: MeshlessFace2d,
    displacement: Vector2,
    state: &HydroMfmState2d,
    primitive: &HydroPrimitiveColumns2d,
    moments: &[InverseMoment2d],
    closure: &[FaceClosure2d],
    i: usize,
    j: usize,
) -> Result<(f64, bool), HydroEvolution2dError> {
    let face_velocity_i = planar_dot(state.velocities[i], face.unit_normal);
    let face_velocity_j = planar_dot(state.velocities[j], face.unit_normal);
    let face_velocity = 0.5 * (face_velocity_i + face_velocity_j);
    let sound_i = (state.gamma * primitive.pressure[i] / primitive.density[i]).sqrt();
    let sound_j = (state.gamma * primitive.pressure[j] / primitive.density[j]).sqrt();
    let speed_ratio = flux.solver_speed.abs() / sound_i.min(sound_j);
    let closure_leak =
        0.5 * (closure[i].legacy_dimensionless_leak + closure[j].legacy_dimensionless_leak);
    if speed_ratio >= EPSILON_ENTROPIC_BIG && closure_leak <= 1.0 {
        return Ok((raw_energy, false));
    }

    let distance = displacement.norm();
    let radial = displacement / distance;
    let relative_radial_velocity = planar_dot(state.velocities[i] - state.velocities[j], radial);
    let kernel_i = cubic_kernel_2d(distance, state.smoothing_lengths[i])?;
    let kernel_j = cubic_kernel_2d(distance, state.smoothing_lengths[j])?;
    let volume_i = state.masses[i] / primitive.density[i];
    let volume_j = state.masses[j] / primitive.density[j];
    let pressure_area = flux.star_pressure * face.area;
    let pdv_factor = flux.star_pressure * relative_radial_velocity;
    let pdv_i =
        kernel_i.radial_derivative * volume_i * volume_i * primitive.dhsml_factor[i] * pdv_factor;
    let pdv_j =
        kernel_j.radial_derivative * volume_j * volume_j * primitive.dhsml_factor[j] * pdv_factor;
    let old_energy = pressure_area * (flux.solver_speed + face_velocity);
    let new_energy = 0.5 * (pdv_i - pdv_j + pressure_area * (face_velocity_i + face_velocity_j));
    if !old_energy.is_finite() || !new_energy.is_finite() {
        return Err(invalid(Some(i), "entropic_pair_energy", f64::NAN));
    }

    let condition_i_squared = moments[i].condition_number.powi(2);
    let condition_j_squared = moments[j].condition_number.powi(2);
    let condition_threshold = CONDITION_NUMBER_DANGER_SQUARED - condition_i_squared;
    let mut use_entropic_energy = true;
    if speed_ratio > EPSILON_ENTROPIC_SMALL
        && condition_j_squared < condition_threshold
        && primitive.pressure[i] / primitive.density[i]
            != primitive.pressure[j] / primitive.density[j]
    {
        if primitive.pressure[i] / primitive.density[i]
            > primitive.pressure[j] / primitive.density[j]
        {
            let thermal_change_j = -old_energy + pressure_area * face_velocity_j;
            if thermal_change_j > 0.0
                || (thermal_change_j < 0.0
                    && thermal_change_j > -new_energy + pressure_area * face_velocity_j)
            {
                use_entropic_energy = false;
            }
        } else {
            let thermal_change_i = old_energy - pressure_area * face_velocity_i;
            if thermal_change_i > 0.0
                || (thermal_change_i < 0.0
                    && thermal_change_i > new_energy - pressure_area * face_velocity_i)
            {
                use_entropic_energy = false;
            }
        }
    }
    if condition_j_squared >= condition_threshold {
        use_entropic_energy = true;
    }
    let energy = if use_entropic_energy {
        raw_energy + new_energy - old_energy
    } else {
        raw_energy
    };
    if !energy.is_finite() {
        return Err(invalid(Some(i), "entropic_pair_energy", energy));
    }
    Ok((energy, use_entropic_energy))
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrimitiveGradients2d {
    density: Vec<Vector2>,
    pressure: Vec<Vector2>,
    velocity_x: Vec<Vector2>,
    velocity_y: Vec<Vector2>,
    velocity_z: Vec<Vector2>,
}

fn primitive_gradients(
    state: &HydroMfmState2d,
    primitive: &HydroPrimitiveColumns2d,
    moments: &[InverseMoment2d],
) -> Result<PrimitiveGradients2d, HydroEvolution2dError> {
    let velocity_x = state
        .velocities
        .iter()
        .map(|value| value.x)
        .collect::<Vec<_>>();
    let velocity_y = state
        .velocities
        .iter()
        .map(|value| value.y)
        .collect::<Vec<_>>();
    let velocity_z = state
        .velocities
        .iter()
        .map(|value| value.z)
        .collect::<Vec<_>>();
    let fields: [&[f64]; 5] = [
        &primitive.density,
        &primitive.pressure,
        &velocity_x,
        &velocity_y,
        &velocity_z,
    ];
    let raw = scalar_gradients_batch_with_moments_2d(
        &state.positions,
        &fields,
        &state.smoothing_lengths,
        state.domain,
        moments,
    )?;
    let pairs = interacting_pairs_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    let mut maximum_distance = vec![0.0_f64; state.positions.len()];
    for pair in &pairs {
        maximum_distance[pair.i] = maximum_distance[pair.i].max(pair.distance);
        maximum_distance[pair.j] = maximum_distance[pair.j].max(pair.distance);
    }
    let limit =
        |values: &[f64], gradients: Vec<Vector2>, positive: bool, overshoot_tolerance: f64| {
            limit_gradients(
                state,
                values,
                gradients,
                moments,
                &pairs,
                &maximum_distance,
                positive,
                overshoot_tolerance,
            )
        };
    Ok(PrimitiveGradients2d {
        density: limit(&primitive.density, raw[0].clone(), true, 0.0),
        pressure: limit(&primitive.pressure, raw[1].clone(), true, 0.1),
        velocity_x: limit(&velocity_x, raw[2].clone(), false, 0.1),
        velocity_y: limit(&velocity_y, raw[3].clone(), false, 0.1),
        velocity_z: limit(&velocity_z, raw[4].clone(), false, 0.1),
    })
}

#[allow(clippy::too_many_arguments)]
fn limit_gradients(
    state: &HydroMfmState2d,
    values: &[f64],
    mut gradients: Vec<Vector2>,
    moments: &[InverseMoment2d],
    pairs: &[crate::meshless_2d::InteractionPair2d],
    maximum_neighbor_distance: &[f64],
    preserve_positivity: bool,
    overshoot_tolerance: f64,
) -> Vec<Vector2> {
    let mut minima = vec![0.0_f64; state.positions.len()];
    let mut maxima = vec![0.0_f64; state.positions.len()];
    for pair in pairs {
        let delta = values[pair.j] - values[pair.i];
        minima[pair.i] = minima[pair.i].min(delta);
        maxima[pair.i] = maxima[pair.i].max(delta);
        minima[pair.j] = minima[pair.j].min(-delta);
        maxima[pair.j] = maxima[pair.j].max(-delta);
    }
    for i in 0..gradients.len() {
        let distance_fraction = if moments[i].condition_number > 100.0 {
            (LOCAL_GRADIENT_LIMITER_DISTANCE_FRACTION
                + 0.25 * (moments[i].condition_number - 100.0) / 100.0)
                .min(0.5)
        } else {
            LOCAL_GRADIENT_LIMITER_DISTANCE_FRACTION
        };
        let limiting_length = state.smoothing_lengths[i].max(maximum_neighbor_distance[i]);
        gradients[i] = local_slope_limiter(
            gradients[i],
            maxima[i],
            minima[i],
            distance_fraction,
            limiting_length,
            preserve_positivity,
            values[i],
            overshoot_tolerance,
        );
    }
    gradients
}

#[allow(clippy::too_many_arguments)]
fn local_slope_limiter(
    gradient: Vector2,
    maximum_delta: f64,
    minimum_delta: f64,
    distance_fraction: f64,
    limiting_length: f64,
    preserve_positivity: bool,
    central_value: f64,
    overshoot_tolerance: f64,
) -> Vector2 {
    let magnitude = gradient.norm();
    if magnitude == 0.0 {
        return gradient;
    }
    let larger_delta = maximum_delta.abs().max(minimum_delta.abs());
    let smaller_delta = maximum_delta.abs().min(minimum_delta.abs());
    let allowed_delta = (smaller_delta + overshoot_tolerance * larger_delta).min(larger_delta);
    let mut correction = allowed_delta / (distance_fraction * limiting_length * magnitude);
    if preserve_positivity {
        let minimum_value =
            central_value.min(0.0_f64.max(
                (MIN_REAL_NUMBER * central_value).max(
                    (0.5 * (central_value + minimum_delta)).min(central_value - allowed_delta),
                ),
            ));
        correction =
            correction.min((central_value - minimum_value) / (limiting_length * magnitude));
    }
    if correction < 1.0 {
        gradient * correction
    } else {
        gradient
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_pair_with_retries(
    state: &HydroMfmState2d,
    primitive: &HydroPrimitiveColumns2d,
    gradients: &PrimitiveGradients2d,
    closure: &[FaceClosure2d],
    i: usize,
    j: usize,
    face: MeshlessFace2d,
) -> Result<EulerMfmFlux2d, HydroEvolution2dError> {
    let interface_velocity = (state.velocities[i] + state.velocities[j]) * 0.5;
    let n = face.unit_normal;
    let displacement = state
        .domain
        .displacement(state.positions[i], state.positions[j])?;
    let radial = displacement / displacement.norm();
    let difference = state.velocities[i] - state.velocities[j];
    let face_approach = planar_dot(difference, n).min(0.0);
    let radial_approach = planar_dot(difference, radial).min(0.0);
    let approach = face_approach.min(0.0).min(radial_approach);
    let approach_squared = approach * approach;
    let pressure_limit = 1.1
        * (primitive.pressure[i] + primitive.density[i] * approach_squared)
            .max(primitive.pressure[j] + primitive.density[j] * approach_squared);
    if !pressure_limit.is_finite() || pressure_limit <= 0.0 {
        return Err(invalid(None, "pair_pressure_limit", pressure_limit));
    }
    let centered_left = primitive_at(state, primitive, j);
    let centered_right = primitive_at(state, primitive, i);
    let closure_leak =
        0.5 * (closure[i].legacy_dimensionless_leak + closure[j].legacy_dimensionless_leak);
    let reconstructed = (closure_leak <= 1.0).then(|| {
        let left = reconstruct_primitive(
            state,
            primitive,
            gradients,
            j,
            i,
            face.offset_from_j,
            face.offset_from_i,
        );
        euler_mfm_flux_2d(
            left.0,
            left.1,
            n,
            interface_velocity,
            state.gamma,
            pressure_limit,
        )
    });
    match reconstructed {
        Some(Ok(flux)) if flux.star_pressure <= 1.4 * pressure_limit => Ok(flux),
        Some(Ok(_) | Err(_)) | None => {
            if let Ok(flux) = euler_mfm_flux_2d(
                centered_left,
                centered_right,
                n,
                interface_velocity,
                state.gamma,
                1.4 * pressure_limit,
            ) {
                Ok(flux)
            } else {
                let quiet = interface_velocity;
                euler_mfm_flux_2d(
                    EulerPrimitive2d {
                        velocity: quiet,
                        ..centered_left
                    },
                    EulerPrimitive2d {
                        velocity: quiet,
                        ..centered_right
                    },
                    n,
                    interface_velocity,
                    state.gamma,
                    2.0 * pressure_limit,
                )
            }
        }
    }
}

fn reconstruct_primitive(
    state: &HydroMfmState2d,
    primitive: &HydroPrimitiveColumns2d,
    gradients: &PrimitiveGradients2d,
    left: usize,
    right: usize,
    offset_left: Vector2,
    offset_right: Vector2,
) -> (EulerPrimitive2d, EulerPrimitive2d) {
    let reconstruct = |value_left, gradient_left, value_right, gradient_right| {
        reconstruct_face_states(
            value_left,
            gradient_left,
            value_right,
            gradient_right,
            offset_left,
            offset_right,
        )
    };
    let density = reconstruct(
        primitive.density[left],
        gradients.density[left],
        primitive.density[right],
        gradients.density[right],
    );
    let pressure = reconstruct(
        primitive.pressure[left],
        gradients.pressure[left],
        primitive.pressure[right],
        gradients.pressure[right],
    );
    let velocity_x = reconstruct(
        state.velocities[left].x,
        gradients.velocity_x[left],
        state.velocities[right].x,
        gradients.velocity_x[right],
    );
    let velocity_y = reconstruct(
        state.velocities[left].y,
        gradients.velocity_y[left],
        state.velocities[right].y,
        gradients.velocity_y[right],
    );
    let velocity_z = reconstruct(
        state.velocities[left].z,
        gradients.velocity_z[left],
        state.velocities[right].z,
        gradients.velocity_z[right],
    );
    (
        EulerPrimitive2d {
            density: density.0,
            velocity: Vector3::new(velocity_x.0, velocity_y.0, velocity_z.0),
            pressure: pressure.0,
        },
        EulerPrimitive2d {
            density: density.1,
            velocity: Vector3::new(velocity_x.1, velocity_y.1, velocity_z.1),
            pressure: pressure.1,
        },
    )
}

#[allow(clippy::float_cmp)]
fn reconstruct_face_states(
    value_i: f64,
    gradient_i: Vector2,
    value_j: f64,
    gradient_j: Vector2,
    offset_i: Vector2,
    offset_j: Vector2,
) -> (f64, f64) {
    if value_i == value_j {
        return (value_i, value_i);
    }
    let mut face_i = value_i + gradient_i.dot(offset_i);
    let mut face_j = value_j + gradient_j.dot(offset_j);
    let minimum = value_i.min(value_j);
    let maximum = value_i.max(value_j);
    let midpoint = 0.5 * (value_i + value_j);
    let difference = maximum - minimum;
    let mut effective_maximum = maximum + 0.5 * difference;
    let mut effective_minimum = minimum - 0.5 * difference;
    if maximum < 0.0 && effective_maximum > 0.0 {
        effective_maximum = maximum * maximum / (maximum - (effective_maximum - maximum));
    }
    if minimum > 0.0 && effective_minimum < 0.0 {
        effective_minimum = minimum * minimum / (minimum + (minimum - effective_minimum));
    }
    let midpoint_maximum = (midpoint + 0.375 * difference).min(effective_maximum);
    let midpoint_minimum = (midpoint - 0.375 * difference).max(effective_minimum);
    if value_i < value_j {
        face_i = face_i.clamp(effective_minimum, midpoint_maximum);
        face_j = face_j.clamp(midpoint_minimum, effective_maximum);
    } else {
        face_i = face_i.clamp(midpoint_minimum, effective_maximum);
        face_j = face_j.clamp(effective_minimum, midpoint_maximum);
    }
    (face_i, face_j)
}

fn primitive_at(
    state: &HydroMfmState2d,
    primitive: &HydroPrimitiveColumns2d,
    index: usize,
) -> EulerPrimitive2d {
    EulerPrimitive2d {
        density: primitive.density[index],
        velocity: state.velocities[index],
        pressure: primitive.pressure[index],
    }
}

fn validate_primitive(value: EulerPrimitive2d) -> Result<(), HydroEvolution2dError> {
    if !value.density.is_finite() || value.density <= 0.0 {
        return Err(invalid(None, "density", value.density));
    }
    if !value.velocity.is_finite() {
        return Err(invalid(None, "velocity", f64::NAN));
    }
    if !value.pressure.is_finite() || value.pressure <= 0.0 {
        return Err(invalid(None, "pressure", value.pressure));
    }
    Ok(())
}

fn planar_dot(vector: Vector3, direction: Vector2) -> f64 {
    vector.x.mul_add(direction.x, vector.y * direction.y)
}

fn method_index(method: RiemannMethod) -> usize {
    match method {
        RiemannMethod::Hllc => 0,
        RiemannMethod::KurganovTadmor => 1,
        RiemannMethod::Exact => 2,
        RiemannMethod::Vacuum => 3,
    }
}

fn extensive_state(state: &HydroMfmState2d) -> (Vec<Vector3>, Vec<f64>) {
    let momentum = state
        .masses
        .iter()
        .zip(&state.velocities)
        .map(|(&mass, &velocity)| velocity * mass)
        .collect();
    let total_energy = state
        .masses
        .iter()
        .zip(&state.velocities)
        .zip(&state.specific_internal_energy)
        .map(|((&mass, &velocity), &internal)| mass * (internal + 0.5 * velocity.squared_norm()))
        .collect();
    (momentum, total_energy)
}

#[allow(clippy::too_many_arguments)]
fn recover_state(
    positions: Vec<Vector2>,
    masses: Vec<f64>,
    smoothing_lengths: Vec<f64>,
    domain: Box2d,
    gamma: f64,
    momentum: &[Vector3],
    total_energy: &[f64],
    minimum_specific_internal_energy: f64,
) -> Result<HydroMfmState2d, HydroEvolution2dError> {
    if momentum.len() != masses.len() || total_energy.len() != masses.len() {
        return Err(HydroEvolution2dError::MismatchedLength {
            field: "conserved_state",
            expected: masses.len(),
            actual: momentum.len().min(total_energy.len()),
        });
    }
    let mut velocities = Vec::with_capacity(masses.len());
    let mut specific_internal_energy = Vec::with_capacity(masses.len());
    for i in 0..masses.len() {
        let velocity = momentum[i] / masses[i];
        let internal = total_energy[i] / masses[i] - 0.5 * velocity.squared_norm();
        if !internal.is_finite() || internal < minimum_specific_internal_energy {
            return Err(invalid(Some(i), "recovered_internal_energy", internal));
        }
        velocities.push(velocity);
        specific_internal_energy.push(internal);
    }
    let state = HydroMfmState2d {
        positions,
        masses,
        velocities,
        specific_internal_energy,
        smoothing_lengths,
        domain,
        gamma,
    };
    state.validate()?;
    Ok(state)
}

fn kick_vectors(values: &[Vector3], rates: &[Vector3], duration: f64) -> Vec<Vector3> {
    values
        .iter()
        .zip(rates)
        .map(|(&value, &rate)| value + rate * duration)
        .collect()
}

fn kick_scalars(values: &[f64], rates: &[f64], duration: f64) -> Vec<f64> {
    values
        .iter()
        .zip(rates)
        .map(|(&value, &rate)| rate.mul_add(duration, value))
        .collect()
}

fn validate_rate_lengths(
    rates: &HydroMfmRates2d,
    expected: usize,
) -> Result<(), HydroEvolution2dError> {
    for (field, actual) in [
        ("rate.momentum", rates.momentum.len()),
        ("rate.total_energy", rates.total_energy.len()),
        ("rate.acceleration", rates.acceleration.len()),
        (
            "rate.specific_internal_energy",
            rates.specific_internal_energy.len(),
        ),
        (
            "rate.maximum_signal_speed",
            rates.maximum_signal_speed.len(),
        ),
        ("rate.velocity_divergence", rates.velocity_divergence.len()),
    ] {
        if actual != expected {
            return Err(HydroEvolution2dError::MismatchedLength {
                field,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn validate_lengths(
    expected: usize,
    fields: &[(&'static str, usize)],
) -> Result<(), HydroEvolution2dError> {
    for &(field, actual) in fields {
        if actual != expected {
            return Err(HydroEvolution2dError::MismatchedLength {
                field,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn invalid(index: Option<usize>, field: &'static str, value: f64) -> HydroEvolution2dError {
    HydroEvolution2dError::InvalidState {
        index,
        field,
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:.17e} != {expected:.17e} within {tolerance:.3e}"
        );
    }

    #[allow(clippy::cast_precision_loss)]
    fn lattice_state(side: usize, boost: Vector3) -> HydroMfmState2d {
        let domain = Box2d::new(1.0, 1.0).unwrap();
        let count = side * side;
        let mass = 1.0 / count as f64;
        let mut positions = Vec::with_capacity(count);
        let mut velocities = Vec::with_capacity(count);
        let mut internal = Vec::with_capacity(count);
        for y in 0..side {
            for x in 0..side {
                let position = Vector2::new(
                    (x as f64 + 0.5) / side as f64,
                    (y as f64 + 0.5) / side as f64,
                );
                let offset = position - Vector2::new(0.5, 0.5);
                let radius = offset.norm();
                let azimuthal = if radius < 0.2 {
                    5.0 * radius
                } else if radius < 0.4 {
                    2.0 - 5.0 * radius
                } else {
                    0.0
                };
                let pressure = if radius < 0.2 {
                    5.0 + 12.5 * radius * radius
                } else if radius < 0.4 {
                    9.0 + 12.5 * radius * radius - 20.0 * radius + 4.0 * (5.0 * radius).ln()
                } else {
                    3.0 + 4.0 * 2.0_f64.ln()
                };
                let tangent = if radius > 0.0 {
                    Vector3::new(-offset.y / radius, offset.x / radius, 0.0)
                } else {
                    Vector3::ZERO
                };
                positions.push(position);
                velocities.push(tangent * azimuthal + boost);
                internal.push(pressure / 0.4);
            }
        }
        HydroMfmState2d {
            positions,
            masses: vec![mass; count],
            velocities,
            specific_internal_energy: internal,
            smoothing_lengths: vec![0.32; count],
            domain,
            gamma: 1.4,
        }
    }

    #[test]
    fn arbitrary_normal_face_uses_hllc_and_is_covariant_under_boost() {
        let normal = Vector2::new(0.6, 0.8);
        let velocity = Vector3::new(0.7, -0.2, 0.35);
        let primitive = EulerPrimitive2d {
            density: 1.0,
            velocity,
            pressure: 0.6,
        };
        let flux =
            euler_mfm_flux_2d(primitive, primitive, normal, velocity, 5.0 / 3.0, 1.0).unwrap();
        assert_eq!(flux.method, RiemannMethod::Hllc);
        assert_close(flux.mass, 0.0, 0.0);
        assert_close(flux.momentum.x, 0.36, 1.0e-15);
        assert_close(flux.momentum.y, 0.48, 1.0e-15);
        assert_close(
            flux.total_energy,
            0.6 * planar_dot(velocity, normal),
            1.0e-15,
        );

        let boost = Vector3::new(3.0, -4.0, 2.5);
        let boosted = EulerPrimitive2d {
            velocity: velocity + boost,
            ..primitive
        };
        let boosted_flux =
            euler_mfm_flux_2d(boosted, boosted, normal, velocity + boost, 5.0 / 3.0, 1.0).unwrap();
        assert_eq!(boosted_flux.momentum, flux.momentum);
        assert_close(
            boosted_flux.total_energy - flux.total_energy,
            boost.dot(flux.momentum),
            2.0e-15,
        );
    }

    #[test]
    fn arbitrary_normal_face_preserves_corrected_kt_fallback() {
        let normal = Vector2::new(0.8, 0.6);
        let normal3 = Vector3::new(normal.x, normal.y, 0.0);
        let gamma = 5.0 / 3.0;
        let left = EulerPrimitive2d {
            density: 1.0,
            velocity: normal3 * -1.0 + Vector3::new(0.0, 0.0, 0.5),
            pressure: 0.6,
        };
        let right = EulerPrimitive2d {
            velocity: normal3 + Vector3::new(0.0, 0.0, -0.25),
            ..left
        };
        let flux = euler_mfm_flux_2d(left, right, normal, Vector3::ZERO, gamma, 1.0).unwrap();
        assert_eq!(flux.method, RiemannMethod::KurganovTadmor);

        // reimann.h:500-526: alpha's numerator is the normal momentum jump,
        // while its denominator contains the full 2.5-D momentum-jump norm.
        let normal_left = left.velocity.dot(normal3);
        let normal_right = right.velocity.dot(normal3);
        let sound_left = (gamma * left.pressure / left.density).sqrt();
        let sound_right = (gamma * right.pressure / right.density).sqrt();
        let momentum_jump = right.velocity * right.density - left.velocity * left.density;
        let normal_jump = right.density * normal_right - left.density * normal_left;
        let threshold =
            0.001 * 0.5 * (left.density + right.density) * 0.5 * (sound_left + sound_right);
        let alpha =
            normal_jump.abs() / (threshold * threshold + momentum_jump.squared_norm()).sqrt();
        let diffusion =
            (alpha * sound_left + normal_left.abs()).max(alpha * sound_right + normal_right.abs());
        let weighted_left = left.density * (normal_left + diffusion);
        let weighted_right = right.density * (normal_right - diffusion);
        let denominator = (left.density * normal_left - right.density * normal_right
            + diffusion * (left.density + right.density))
            .recip();
        let product = weighted_left * weighted_right;
        let expected_pressure =
            (weighted_left * right.pressure - weighted_right * left.pressure) * denominator;
        let expected_momentum = (right.velocity - left.velocity) * (product * denominator)
            + normal3 * expected_pressure;
        let enthalpy = |value: EulerPrimitive2d| {
            value.pressure / value.density
                + value.pressure / ((gamma - 1.0) * value.density)
                + 0.5 * value.velocity.squared_norm()
        };
        let expected_energy = (diffusion
            * (weighted_left * right.pressure + weighted_right * left.pressure)
            + (enthalpy(right) - enthalpy(left)) * product)
            * denominator;
        assert_close(flux.star_pressure, expected_pressure, 2.0e-15);
        assert_close(flux.momentum.x, expected_momentum.x, 2.0e-15);
        assert_close(flux.momentum.y, expected_momentum.y, 2.0e-15);
        assert_close(flux.momentum.z, expected_momentum.z, 2.0e-15);
        assert_close(flux.total_energy, expected_energy, 2.0e-15);
    }

    #[test]
    fn collinear_vector_flux_preserves_scalar_solver_parity() {
        let normal = Vector2::new(0.6, 0.8);
        let normal3 = Vector3::new(normal.x, normal.y, 0.0);
        let gamma = 1.4;
        let left = EulerPrimitive2d {
            density: 0.7,
            velocity: normal3 * -0.3,
            pressure: 0.9,
        };
        let right = EulerPrimitive2d {
            density: 1.2,
            velocity: normal3 * 0.15,
            pressure: 0.4,
        };
        let vector = euler_mfm_flux_2d(left, right, normal, Vector3::ZERO, gamma, 2.0).unwrap();
        let scalar = ideal_gas_mfm_flux_1d(
            PrimitiveState1d {
                density: left.density,
                velocity: -0.3,
                pressure: left.pressure,
            },
            PrimitiveState1d {
                density: right.density,
                velocity: 0.15,
                pressure: right.pressure,
            },
            gamma,
            2.0,
        )
        .unwrap();
        assert_eq!(vector.method, scalar.method);
        assert_close(vector.star_pressure, scalar.star_pressure, 2.0e-15);
        assert_close(vector.solver_speed, scalar.solver_speed, 2.0e-15);
        assert_close(vector.momentum.dot(normal3), scalar.momentum, 2.0e-15);
        assert_close(vector.total_energy, scalar.energy, 2.0e-15);
    }

    #[test]
    fn gresho_operator_is_conservative_and_galilean_covariant() {
        let state = lattice_state(8, Vector3::ZERO);
        let rates = hydro_mfm_spatial_rates_2d(&state).unwrap();
        assert!(rates.pair_count > state.positions.len());
        let momentum_sum = rates
            .momentum
            .iter()
            .copied()
            .fold(Vector3::ZERO, |sum, value| sum + value);
        let energy_sum: f64 = rates.total_energy.iter().sum();
        assert!(momentum_sum.squared_norm().sqrt() < 2.0e-13);
        assert!(energy_sum.abs() < 2.0e-13);

        let boost = Vector3::new(3.0, -1.25, 0.75);
        let boosted = lattice_state(8, boost);
        let boosted_rates = hydro_mfm_spatial_rates_2d(&boosted).unwrap();
        for i in 0..state.positions.len() {
            assert_close(
                boosted_rates.acceleration[i].x,
                rates.acceleration[i].x,
                5.0e-11,
            );
            assert_close(
                boosted_rates.acceleration[i].y,
                rates.acceleration[i].y,
                5.0e-11,
            );
            assert_close(
                boosted_rates.acceleration[i].z,
                rates.acceleration[i].z,
                5.0e-11,
            );
            assert_close(
                boosted_rates.specific_internal_energy[i],
                rates.specific_internal_energy[i],
                2.0e-10,
            );
        }
    }

    #[test]
    fn adaptive_kdk_moves_vortex_and_conserves_extensive_totals() {
        let state = lattice_state(8, Vector3::ZERO);
        let rates = hydro_mfm_spatial_rates_2d(&state).unwrap();
        let courant = global_hydro_courant_timestep_2d(&state, &rates, 0.025).unwrap();
        assert!(courant.is_finite() && courant > 0.0);
        let initial_momentum = state.total_momentum();
        let initial_energy = state.total_energy();
        let result = advance_hydro_kdk_adaptive_2d(&state, &rates, 1.0e-4, 0.0, 20.0, 0.5).unwrap();
        assert!(
            result
                .state
                .positions
                .iter()
                .zip(&state.positions)
                .any(|(&after, &before)| after != before)
        );
        assert!(
            result
                .state
                .smoothing_lengths
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        assert!(
            (result.state.total_momentum() - initial_momentum)
                .squared_norm()
                .sqrt()
                < 2.0e-13
        );
        assert_close(result.state.total_energy(), initial_energy, 2.0e-13);
    }

    #[test]
    fn non_equilibrium_planar_perturbation_has_nonzero_rates_and_evolves() {
        let mut state = lattice_state(8, Vector3::ZERO);
        state.specific_internal_energy[9] *= 1.2;
        state.velocities[18].z = 0.35;
        let rates = hydro_mfm_spatial_rates_2d(&state).unwrap();
        assert!(
            rates
                .acceleration
                .iter()
                .any(|rate| rate.squared_norm() > 1.0e-16)
        );
        assert!(
            rates
                .specific_internal_energy
                .iter()
                .any(|rate| rate.abs() > 1.0e-12)
        );
        let result = advance_hydro_kdk_2d(&state, &rates, 1.0e-5, 0.0).unwrap();
        assert_ne!(result.state.positions, state.positions);
        assert_ne!(result.state.velocities, state.velocities);
        assert_ne!(
            result.state.specific_internal_energy,
            state.specific_internal_energy
        );
    }

    fn alternating_timebins(count: usize) -> Vec<PublicHydroInitialTimebin2d> {
        let bounds = (0..count)
            .map(|i| {
                let selected = if i % 2 == 0 { 0.125 } else { 0.25 };
                PublicHydroTimestepBounds2d {
                    acceleration: selected,
                    courant: selected,
                    velocity_divergence: selected,
                    selected,
                }
            })
            .collect::<Vec<_>>();
        quantize_public_hydro_initial_timebins_2d(&bounds, 0.0, 1.0, 0.5).unwrap()
    }

    #[test]
    fn hierarchy_keeps_inactive_predictors_and_caches_lazy() {
        let state = lattice_state(8, Vector3::new(0.0, 0.0, 0.25));
        let rates = hydro_mfm_spatial_rates_2d(&state).unwrap();
        let bins = alternating_timebins(state.positions.len());
        let mut hierarchy =
            begin_public_hydro_initial_hierarchy_2d(&state, &rates, &bins, 0.0, 1.0, 0.0).unwrap();
        let sync = hierarchy.drift_to_first_sync().unwrap();
        assert!(sync.active.iter().any(|&active| active));
        assert!(sync.active.iter().any(|&active| !active));
        for (i, &active) in sync.active.iter().enumerate() {
            assert_eq!(
                hierarchy.predictor_ticks()[i],
                if active { sync.tick } else { 0 }
            );
        }
        let old_gradients = hierarchy.gradient_cache.clone();
        let old_moments = hierarchy.moment_cache.clone();
        let old_closure = hierarchy.face_closure_cache.clone();
        hierarchy.refresh_arriving_active_caches(20.0, 0.5).unwrap();
        for (i, &active) in sync.active.iter().enumerate() {
            if !active {
                assert_eq!(
                    hierarchy.primitive_cache.density[i].to_bits(),
                    hierarchy.drift.predicted_density[i].to_bits()
                );
                assert_eq!(
                    hierarchy.gradient_cache.density[i],
                    old_gradients.density[i]
                );
                assert_eq!(hierarchy.moment_cache[i], old_moments[i]);
                assert_eq!(hierarchy.face_closure_cache[i], old_closure[i]);
            }
        }
    }

    #[test]
    fn active_force_retains_inactive_rates_and_flags_wakeups() {
        let state = lattice_state(8, Vector3::ZERO);
        let rates = hydro_mfm_spatial_rates_2d(&state).unwrap();
        let bins = alternating_timebins(state.positions.len());
        let mut hierarchy =
            begin_public_hydro_initial_hierarchy_2d(&state, &rates, &bins, 0.0, 1.0, 0.0).unwrap();
        let sync = hierarchy.drift_to_first_sync().unwrap();
        for (i, &active) in sync.active.iter().enumerate() {
            if !active {
                hierarchy.old_rates.maximum_signal_speed[i] = 1.0e-6;
            }
        }
        hierarchy.refresh_arriving_active_caches(20.0, 0.5).unwrap();
        let endpoint = hierarchy.evaluate_arriving_active_rates().unwrap();
        for (i, &active) in sync.active.iter().enumerate() {
            if !active {
                assert!(same_particle_rate(&endpoint.rates, &hierarchy.old_rates, i));
            }
        }
        assert!(
            endpoint
                .wakeup
                .iter()
                .zip(&sync.active)
                .any(|(&wakeup, &active)| wakeup && !active)
        );
    }

    #[test]
    fn wakeup_reverses_the_inactive_particles_old_half_kick() {
        let state = lattice_state(8, Vector3::ZERO);
        let rates = hydro_mfm_spatial_rates_2d(&state).unwrap();
        let bins = alternating_timebins(state.positions.len());
        let mut hierarchy =
            begin_public_hydro_initial_hierarchy_2d(&state, &rates, &bins, 0.0, 1.0, 0.0).unwrap();
        let sync = hierarchy.drift_to_first_sync().unwrap();
        for (i, &active) in sync.active.iter().enumerate() {
            if !active {
                hierarchy.old_rates.maximum_signal_speed[i] = 1.0e-6;
            }
        }
        hierarchy.refresh_arriving_active_caches(20.0, 0.5).unwrap();
        let endpoint = hierarchy.evaluate_arriving_active_rates().unwrap();
        let woken = endpoint
            .wakeup
            .iter()
            .zip(&sync.active)
            .position(|(&wakeup, &active)| wakeup && !active)
            .unwrap();
        hierarchy.finish_arriving_active_kicks(endpoint).unwrap();
        let before = hierarchy.drift.actual_velocities[woken];
        let old_acceleration = hierarchy.old_rates.acceleration[woken];
        let reverse_duration = -0.125;
        let bounds = sync
            .active
            .iter()
            .map(|&active| active.then_some(0.0625))
            .collect::<Vec<_>>();
        hierarchy.begin_next_sync(&bounds).unwrap();
        assert_eq!(
            hierarchy.drift.actual_velocities[woken],
            before + old_acceleration * reverse_duration
        );
    }

    #[test]
    fn active_target_batches_are_bitwise_deterministic_across_worker_counts() {
        let state = lattice_state(8, Vector3::new(0.1, -0.2, 0.3));
        let retained = hydro_mfm_spatial_rates_2d(&state).unwrap();
        let primitive = state.primitive_columns().unwrap();
        let moments =
            inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain).unwrap();
        let closure = face_closure_diagnostics_2d(
            &state.positions,
            &state.masses,
            &state.smoothing_lengths,
            state.domain,
        )
        .unwrap();
        let gradients = primitive_gradients(&state, &primitive, &moments).unwrap();
        let active = (0..state.positions.len())
            .map(|i| i % 3 != 1)
            .collect::<Vec<_>>();
        let pairs = interacting_pairs_for_targets_2d(
            &state.positions,
            &state.smoothing_lengths,
            state.domain,
            &active,
        )
        .unwrap();
        let serial = evaluate_active_hydro_target_batches_2d(
            &state, &primitive, &gradients, &moments, &closure, &retained, &active, &pairs, 1,
        )
        .unwrap();
        let threaded = evaluate_active_hydro_target_batches_2d(
            &state, &primitive, &gradients, &moments, &closure, &retained, &active, &pairs, 4,
        )
        .unwrap();
        assert_eq!(threaded, serial);
    }

    #[test]
    fn production_directed_operator_solves_each_target_orientation() {
        let mut state = lattice_state(8, Vector3::ZERO);
        let retained = hydro_mfm_spatial_rates_2d(&state).unwrap();
        let pairs =
            interacting_pairs_2d(&state.positions, &state.smoothing_lengths, state.domain).unwrap();
        let pair = pairs[0];
        // Put this pair close to the pressure/retry machinery and add 2.5-D
        // shear. The assertion below compares each target against its own
        // directed solve, never against a negated canonical solve.
        state.specific_internal_energy[pair.i] = 0.08;
        state.specific_internal_energy[pair.j] = 4.0;
        state.velocities[pair.i] = Vector3::new(0.017, -0.004, 0.009);
        state.velocities[pair.j] = Vector3::new(-0.012, 0.006, -0.007);
        let primitive = state.primitive_columns().unwrap();
        let moments =
            inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain).unwrap();
        let closure = face_closure_diagnostics_2d(
            &state.positions,
            &state.masses,
            &state.smoothing_lengths,
            state.domain,
        )
        .unwrap();
        let gradients = primitive_gradients(&state, &primitive, &moments).unwrap();
        let active = vec![false; state.positions.len()];

        let directed = |target: usize, neighbor: usize| {
            evaluate_active_hydro_target_batch_2d(
                &state,
                &primitive,
                &gradients,
                &moments,
                &closure,
                &retained,
                &active,
                target,
                &[(neighbor, 0)],
            )
            .unwrap()
        };
        let forward = directed(pair.i, pair.j);
        let reverse = directed(pair.j, pair.i);

        let direct_flux = |target: usize, neighbor: usize| {
            let point = |i: usize| MeshlessPoint2d {
                position: state.positions[i],
                mass: state.masses[i],
                density: primitive.density[i],
                smoothing_length: state.smoothing_lengths[i],
                inverse_moment: moments[i].matrix,
                condition_number: moments[i].condition_number,
            };
            let face =
                meshless_face_geometry_2d(point(target), point(neighbor), state.domain).unwrap();
            let flux = solve_pair_with_retries(
                &state, &primitive, &gradients, &closure, target, neighbor, face,
            )
            .unwrap();
            let displacement = state
                .domain
                .displacement(state.positions[target], state.positions[neighbor])
                .unwrap();
            let energy = apply_entropic_pdv_energy_2d(
                flux.total_energy * face.area,
                flux,
                face,
                displacement,
                &state,
                &primitive,
                &moments,
                &closure,
                target,
                neighbor,
            )
            .unwrap()
            .0;
            (flux.momentum * face.area, energy, flux.method)
        };
        let expected_forward = direct_flux(pair.i, pair.j);
        let expected_reverse = direct_flux(pair.j, pair.i);
        assert_eq!(forward.momentum, expected_forward.0);
        assert_eq!(forward.total_energy.to_bits(), expected_forward.1.to_bits());
        assert_eq!(forward.method_counts[method_index(expected_forward.2)], 1);
        assert_eq!(reverse.momentum, expected_reverse.0);
        assert_eq!(reverse.total_energy.to_bits(), expected_reverse.1.to_bits());
        assert_eq!(reverse.method_counts[method_index(expected_reverse.2)], 1);

        let production = hydro_mfm_directed_spatial_rates_2d(&state).unwrap();
        let conservative = hydro_mfm_spatial_rates_2d(&state).unwrap();
        assert_eq!(production.pair_count, 2 * conservative.pair_count);
        assert!(
            production.momentum != conservative.momentum
                || production.total_energy != conservative.total_energy,
            "the public directed operator must not collapse to unordered antisymmetry"
        );
    }

    #[test]
    fn active_target_refresh_retains_every_inactive_rate_slot() {
        let state = lattice_state(8, Vector3::new(0.1, -0.2, 0.3));
        let retained = hydro_mfm_spatial_rates_2d(&state).unwrap();
        let primitive = state.primitive_columns().unwrap();
        let moments =
            inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain).unwrap();
        let closure = face_closure_diagnostics_2d(
            &state.positions,
            &state.masses,
            &state.smoothing_lengths,
            state.domain,
        )
        .unwrap();
        let gradients = primitive_gradients(&state, &primitive, &moments).unwrap();
        let active = (0..state.positions.len())
            .map(|i| i % 7 == 0)
            .collect::<Vec<_>>();
        let updated = hydro_mfm_active_target_rates_with_cache_2d(
            &state, &primitive, &gradients, &moments, &closure, &retained, &active,
        )
        .unwrap();
        for (i, &is_active) in active.iter().enumerate() {
            if !is_active {
                assert!(same_particle_rate(&updated.rates, &retained, i));
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn next_scheduled_drift_uses_the_next_half_step_velocity() {
        let state = HydroMfmState2d::from_primitive(
            vec![Vector2::new(0.25, 0.25)],
            vec![1.0],
            vec![Vector3::ZERO],
            vec![1.0],
            vec![0.5],
            Box2d::new(1.0, 1.0).unwrap(),
            5.0 / 3.0,
        )
        .unwrap();
        let rates = HydroMfmRates2d {
            momentum: vec![Vector3::new(2.0, 0.0, 0.0)],
            total_energy: vec![0.0],
            acceleration: vec![Vector3::new(2.0, 0.0, 0.0)],
            specific_internal_energy: vec![-100.0],
            maximum_signal_speed: vec![1.0],
            velocity_divergence: vec![10.0],
            pair_count: 0,
            entropic_pair_count: 0,
            hllc_pair_count: 0,
            kt_pair_count: 0,
            exact_pair_count: 0,
            vacuum_pair_count: 0,
        };
        let bins = quantize_public_hydro_initial_timebins_2d(
            &[PublicHydroTimestepBounds2d {
                acceleration: 0.25,
                courant: 0.25,
                velocity_divergence: 0.25,
                selected: 0.25,
            }],
            0.0,
            1.0,
            0.5,
        )
        .unwrap();
        let primitive = state.primitive_columns().unwrap();
        let mut hierarchy = PublicHydroInitialHierarchy2d {
            timeline: IndividualParticleTimeline::from_initial_steps(0.0, 1.0, &[bins[0].ticks])
                .unwrap(),
            start: state.clone(),
            old_rates: rates.clone(),
            actual_internal_energy: vec![1.0],
            drift: PublicHydroDriftState2d {
                positions: state.positions.clone(),
                actual_velocities: vec![Vector3::new(0.25, 0.0, 0.0)],
                predicted_velocities: state.velocities.clone(),
                predicted_specific_internal_energy: vec![1.0],
                predicted_density: primitive.density.clone(),
                predicted_smoothing_lengths: state.smoothing_lengths.clone(),
            },
            predictor_ticks: vec![0],
            primitive_cache: primitive,
            gradient_cache: PrimitiveGradients2d {
                density: vec![Vector2::ZERO],
                pressure: vec![Vector2::ZERO],
                velocity_x: vec![Vector2::ZERO],
                velocity_y: vec![Vector2::ZERO],
                velocity_z: vec![Vector2::ZERO],
            },
            moment_cache: vec![InverseMoment2d {
                matrix: crate::meshless_2d::Matrix2::ZERO,
                condition_number: 1.0,
                diagonal_regularization: 0.0,
            }],
            face_closure_cache: vec![FaceClosure2d::default()],
            minimum_specific_internal_energy: 0.0,
            awaiting_second_kick: false,
            prepared_next_drift: false,
            refreshed_cache_tick: None,
            pending_wakeup: vec![false],
        };
        hierarchy.drift_to_first_sync().unwrap();
        hierarchy
            .finish_arriving_active_kicks(PublicHydroActiveRateResult2d {
                rates: rates.clone(),
                wakeup: vec![false],
            })
            .unwrap();
        let integer_velocity = hierarchy.drift.actual_velocities[0];
        assert_eq!(integer_velocity.x.to_bits(), 0.5_f64.to_bits());
        let before = hierarchy.clone();
        let mut phase_correct = hierarchy.clone();
        assert!(phase_correct.prepare_next_drift(&[None]).is_err());
        assert_eq!(phase_correct, hierarchy);
        let next_tick = phase_correct.prepare_next_drift(&[Some(0.25)]).unwrap();
        assert_eq!(next_tick, bins[0].ticks * 2);
        let output_tick = phase_correct
            .output_time_to_integer_tick(0.375, false)
            .unwrap();
        let projected = phase_correct
            .drift_prepared_to_output_tick(output_tick)
            .unwrap();
        assert_eq!(hierarchy, before);
        assert_eq!(
            projected.actual_velocities[0].x.to_bits(),
            0.75_f64.to_bits()
        );
        assert_eq!(
            projected.predicted_velocities[0].x.to_bits(),
            0.75_f64.to_bits()
        );
        // x=0.25 + 0.25*0.25 for the first drift, then 0.75*0.125.
        assert_eq!(projected.positions[0].x.to_bits(), 0.40625_f64.to_bits());
        let integer_velocity_position = 0.3125 + integer_velocity.x * 0.125;
        assert_ne!(
            projected.positions[0].x.to_bits(),
            integer_velocity_position.to_bits()
        );

        let mut segmented = before.clone();
        let unchanged_next_tick = segmented.prepare_next_drift(&[Some(0.25)]).unwrap();
        let first_output_tick = segmented
            .output_time_to_integer_tick(0.3125, false)
            .unwrap();
        let second_output_tick = segmented.output_time_to_integer_tick(0.375, false).unwrap();
        segmented
            .drift_prepared_to_output_tick(first_output_tick)
            .unwrap();
        assert_eq!(segmented.predictor_ticks(), &[first_output_tick]);
        let first_internal = segmented.drift.predicted_specific_internal_energy[0];
        let first_density = segmented.drift.predicted_density[0];
        let first_h = segmented.drift.predicted_smoothing_lengths[0];
        segmented
            .drift_prepared_to_output_tick(second_output_tick)
            .unwrap();
        assert_eq!(segmented.predictor_ticks(), &[second_output_tick]);
        assert_eq!(
            segmented.drift.predicted_specific_internal_energy[0].to_bits(),
            (0.5 * first_internal).to_bits()
        );
        assert_eq!(
            segmented.drift.predicted_density[0].to_bits(),
            (first_density * (-0.3_f64).exp()).to_bits()
        );
        assert_eq!(
            segmented.drift.predicted_smoothing_lengths[0].to_bits(),
            (first_h * 0.15_f64.exp()).to_bits()
        );
        let resumed = segmented.finish_prepared_next_sync().unwrap();
        assert_eq!(resumed.tick, unchanged_next_tick);
        assert_eq!(resumed.tick, bins[0].ticks * 2);
    }
}
