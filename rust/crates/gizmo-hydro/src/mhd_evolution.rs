//! One-dimensional periodic meshless finite-mass ideal-MHD evolution.
//!
//! This module deliberately stores the magnetic conservative variable as
//! `volume * B`.  Density is recomputed from the particle geometry, so exposing
//! primitive `B` without that conversion would silently violate flux freezing.
//! Momentum, total energy, and `volume * B` pair rates are accumulated once per
//! unordered pair and are exactly antisymmetric.

use std::error::Error;
use std::fmt;

use crate::mhd::{
    DednerOptions, FluxFrame1d, HlldOptions, IdealMhdPrimitive1d, MhdError, MhdRiemannMethod,
    Vector3, dedner_hyperbolic_source, dedner_parabolic_source, fast_magnetosonic_speed,
    hlld_riemann,
};
use crate::{
    BoundaryMode1d, HydroError, MeshlessFace1d, MeshlessPoint1d, ReconstructionOrder,
    density_at_hsml_1d, face_closure_errors_1d, gradients_at_hsml_1d, interacting_pairs_1d,
    inverse_moments_1d, meshless_face_geometry_1d, particle_divergence_at_hsml_1d,
    reconstruct_face_states_1d, solve_public_c_smoothing_lengths_from_seeds_1d,
};

const MHD_GRADIENT_LIMITER: f64 = 0.25;

/// Controls the non-conservative divergence-control terms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DivergenceControl1d {
    /// Enable the Powell eight-wave source.
    pub powell: bool,
    /// Enable the two-wave Dedner interface correction and local sources.
    pub dedner: bool,
    /// Coefficient in `d(phi)/dt = -sigma_h c_h^2 div(B)`.
    pub hyperbolic_sigma: f64,
    /// Coefficient in the local parabolic decay of `phi`.
    pub parabolic_sigma: f64,
    /// Dedner interface correction limiter.
    pub implicit_limiter: f64,
}

impl Default for DivergenceControl1d {
    fn default() -> Self {
        Self {
            powell: true,
            dedner: true,
            hyperbolic_sigma: 1.0,
            parabolic_sigma: 0.2,
            implicit_limiter: 0.75,
        }
    }
}

/// Owned state for synchronized one-dimensional MFM ideal-MHD evolution.
#[derive(Clone, Debug, PartialEq)]
pub struct MhdMfmState1d {
    pub positions: Vec<f64>,
    pub masses: Vec<f64>,
    pub velocities: Vec<Vector3>,
    pub specific_internal_energy: Vec<f64>,
    pub smoothing_lengths: Vec<f64>,
    /// Extensive magnetic conservative variable, `mass / density * B`.
    pub magnetic_volume: Vec<Vector3>,
    /// Mass-based Dedner variable, `mass * phi`.
    pub cleaning_mass: Vec<f64>,
    pub box_size: f64,
    pub gamma: f64,
}

/// Primitive particle columns recovered at the current geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct MhdPrimitiveColumns1d {
    pub density: Vec<f64>,
    pub pressure: Vec<f64>,
    pub magnetic: Vec<Vector3>,
    pub cleaning_scalar: Vec<f64>,
}

/// Extensive semidiscrete rates plus predictor and timestep diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct MhdMfmRates1d {
    pub momentum: Vec<Vector3>,
    pub total_energy: Vec<f64>,
    pub magnetic_volume: Vec<Vector3>,
    pub cleaning_mass: Vec<f64>,
    pub acceleration: Vec<Vector3>,
    pub specific_internal_energy: Vec<f64>,
    pub maximum_signal_speed: Vec<f64>,
    pub velocity_divergence: Vec<f64>,
    pub magnetic_divergence: Vec<f64>,
    pub pair_count: usize,
}

/// A completed synchronized KDK step and the force evaluated at its endpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct MhdKdkResult1d {
    pub state: MhdMfmState1d,
    pub rates: MhdMfmRates1d,
}

/// Errors from the MHD evolution layer.
#[derive(Debug)]
pub enum MhdEvolutionError {
    Hydro(HydroError),
    Mhd(MhdError),
    InvalidState {
        index: Option<usize>,
        field: &'static str,
        value: f64,
    },
    MismatchedLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for MhdEvolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hydro(error) => write!(formatter, "{error}"),
            Self::Mhd(error) => write!(formatter, "{error}"),
            Self::InvalidState {
                index,
                field,
                value,
            } => write!(formatter, "invalid MHD state {field}={value} at {index:?}"),
            Self::MismatchedLength {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} has length {actual}, expected {expected}"
            ),
        }
    }
}

impl Error for MhdEvolutionError {}

impl From<HydroError> for MhdEvolutionError {
    fn from(value: HydroError) -> Self {
        Self::Hydro(value)
    }
}

impl From<MhdError> for MhdEvolutionError {
    fn from(value: MhdError) -> Self {
        Self::Mhd(value)
    }
}

impl MhdMfmState1d {
    /// Construct an evolving state from primitive magnetic and cleaning fields.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid columns or non-physical primitive fields.
    #[allow(clippy::too_many_arguments)]
    pub fn from_primitive(
        positions: Vec<f64>,
        masses: Vec<f64>,
        velocities: Vec<Vector3>,
        specific_internal_energy: Vec<f64>,
        smoothing_lengths: Vec<f64>,
        magnetic: &[Vector3],
        cleaning_scalar: &[f64],
        box_size: f64,
        gamma: f64,
    ) -> Result<Self, MhdEvolutionError> {
        let particle_count = positions.len();
        validate_lengths(
            particle_count,
            &[
                ("masses", masses.len()),
                ("velocities", velocities.len()),
                ("specific_internal_energy", specific_internal_energy.len()),
                ("smoothing_lengths", smoothing_lengths.len()),
                ("magnetic", magnetic.len()),
                ("cleaning_scalar", cleaning_scalar.len()),
            ],
        )?;
        let density = density_values(&positions, &masses, &smoothing_lengths, box_size)?;
        let magnetic_volume = magnetic
            .iter()
            .zip(&masses)
            .zip(&density)
            .map(|((&field, &mass), &rho)| field * (mass / rho))
            .collect();
        let cleaning_mass = cleaning_scalar
            .iter()
            .zip(&masses)
            .map(|(&phi, &mass)| phi * mass)
            .collect();
        let state = Self {
            positions,
            masses,
            velocities,
            specific_internal_energy,
            smoothing_lengths,
            magnetic_volume,
            cleaning_mass,
            box_size,
            gamma,
        };
        state.validate()?;
        Ok(state)
    }

    /// Recover density, pressure, primitive magnetic field, and Dedner scalar.
    ///
    /// # Errors
    ///
    /// Returns an error if geometry or a recovered primitive is non-physical.
    pub fn primitive_columns(&self) -> Result<MhdPrimitiveColumns1d, MhdEvolutionError> {
        self.validate()?;
        let density = density_values(
            &self.positions,
            &self.masses,
            &self.smoothing_lengths,
            self.box_size,
        )?;
        let mut pressure = Vec::with_capacity(self.positions.len());
        let mut magnetic = Vec::with_capacity(self.positions.len());
        let mut cleaning_scalar = Vec::with_capacity(self.positions.len());
        for (index, &rho) in density.iter().enumerate() {
            let volume = self.masses[index] / rho;
            let field = self.magnetic_volume[index] / volume;
            let phi = self.cleaning_mass[index] / self.masses[index];
            let gas_pressure = (self.gamma - 1.0) * rho * self.specific_internal_energy[index];
            let primitive = IdealMhdPrimitive1d {
                density: rho,
                velocity: self.velocities[index],
                gas_pressure,
                magnetic: field,
                cleaning_scalar: phi,
            };
            // The conversion is also a centralized physical-state validator.
            primitive.to_conserved(self.gamma)?;
            pressure.push(gas_pressure);
            magnetic.push(field);
            cleaning_scalar.push(phi);
        }
        Ok(MhdPrimitiveColumns1d {
            density,
            pressure,
            magnetic,
            cleaning_scalar,
        })
    }

    /// Sum particle total energy, including magnetic energy.
    ///
    /// # Errors
    ///
    /// Returns an error if primitives cannot be recovered.
    pub fn total_energy(&self) -> Result<f64, MhdEvolutionError> {
        let primitive = self.primitive_columns()?;
        Ok((0..self.positions.len())
            .map(|index| {
                let volume = self.masses[index] / primitive.density[index];
                self.masses[index] * self.specific_internal_energy[index]
                    + 0.5 * self.masses[index] * self.velocities[index].squared_norm()
                    + 0.5 * volume * primitive.magnetic[index].squared_norm()
            })
            .sum())
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

    fn validate(&self) -> Result<(), MhdEvolutionError> {
        let particle_count = self.positions.len();
        validate_lengths(
            particle_count,
            &[
                ("masses", self.masses.len()),
                ("velocities", self.velocities.len()),
                (
                    "specific_internal_energy",
                    self.specific_internal_energy.len(),
                ),
                ("smoothing_lengths", self.smoothing_lengths.len()),
                ("magnetic_volume", self.magnetic_volume.len()),
                ("cleaning_mass", self.cleaning_mass.len()),
            ],
        )?;
        if !self.box_size.is_finite() || self.box_size <= 0.0 {
            return Err(invalid(None, "box_size", self.box_size));
        }
        if !self.gamma.is_finite() || self.gamma <= 1.0 {
            return Err(invalid(None, "gamma", self.gamma));
        }
        for index in 0..particle_count {
            for (field, value) in [
                ("position", self.positions[index]),
                ("mass", self.masses[index]),
                (
                    "specific_internal_energy",
                    self.specific_internal_energy[index],
                ),
                ("smoothing_length", self.smoothing_lengths[index]),
                ("cleaning_mass", self.cleaning_mass[index]),
            ] {
                let positive = matches!(
                    field,
                    "mass" | "specific_internal_energy" | "smoothing_length"
                );
                if !value.is_finite() || (positive && value <= 0.0) {
                    return Err(invalid(Some(index), field, value));
                }
            }
            if !(0.0..self.box_size).contains(&self.positions[index]) {
                return Err(invalid(Some(index), "position", self.positions[index]));
            }
            if !self.velocities[index].is_finite() {
                return Err(invalid(Some(index), "velocity", f64::NAN));
            }
            if !self.magnetic_volume[index].is_finite() {
                return Err(invalid(Some(index), "magnetic_volume", f64::NAN));
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Gradients {
    density: Vec<f64>,
    pressure: Vec<f64>,
    velocity: [Vec<f64>; 3],
    magnetic: [Vec<f64>; 3],
    cleaning: Vec<f64>,
}

/// Evaluate the periodic semidiscrete 1-D MFM ideal-MHD operator.
///
/// Pair fluxes use the HLLD contact frame. The only non-antisymmetric terms
/// are the explicitly selected Powell and local Dedner sources.
///
/// # Errors
///
/// Returns an error for invalid state, geometry, reconstruction, or Riemann
/// arithmetic.
#[allow(clippy::too_many_lines)]
pub fn mhd_mfm_spatial_rates_1d(
    state: &MhdMfmState1d,
    controls: DivergenceControl1d,
) -> Result<MhdMfmRates1d, MhdEvolutionError> {
    state.validate()?;
    validate_controls(controls)?;
    let primitive = state.primitive_columns()?;
    let gradients = primitive_gradients(state, &primitive)?;
    let inverse_moments =
        inverse_moments_1d(&state.positions, &state.smoothing_lengths, state.box_size)?;
    let closure_errors =
        face_closure_errors_1d(&state.positions, &state.smoothing_lengths, state.box_size)?;
    let particle_count = state.positions.len();
    let mut momentum = vec![Vector3::ZERO; particle_count];
    let mut total_energy = vec![0.0; particle_count];
    let mut magnetic_volume = vec![Vector3::ZERO; particle_count];
    let mut magnetic_divergence_volume = vec![0.0; particle_count];
    let mut dedner_magnetic_correction = vec![Vector3::ZERO; particle_count];
    let mut cleaning_mass = vec![0.0; particle_count];
    let mut maximum_signal_speed = Vec::with_capacity(particle_count);
    for index in 0..particle_count {
        maximum_signal_speed.push(fast_magnetosonic_speed(
            primitive_at(state, &primitive, index),
            state.gamma,
        )?);
    }

    let mut pair_count = 0;
    for (i, j, _displacement) in interacting_pairs_1d(
        &state.positions,
        &state.smoothing_lengths,
        state.box_size,
        BoundaryMode1d::Periodic,
    )? {
        let geometry = |index: usize| MeshlessPoint1d {
            position: state.positions[index],
            mass: state.masses[index],
            density: primitive.density[index],
            smoothing_length: state.smoothing_lengths[index],
            inverse_moment: inverse_moments[index],
        };
        let face = meshless_face_geometry_1d(geometry(i), geometry(j), state.box_size)?;
        let (left, right) =
            reconstruct_pair(state, &primitive, &gradients, i, j, face, &closure_errors)?;
        let normal = face.signed_area.signum();
        let local_left = orient_primitive(left, normal);
        let local_right = orient_primitive(right, normal);
        let approach_velocity =
            (state.velocities[i].x * normal - state.velocities[j].x * normal).min(0.0);
        let approach_speed_squared = approach_velocity * approach_velocity;
        let maximum_star_total_pressure = 2.2
            * (primitive_at(state, &primitive, i).total_pressure()
                + primitive.density[i] * approach_speed_squared
                + primitive_at(state, &primitive, j).total_pressure()
                + primitive.density[j] * approach_speed_squared);
        let options = HlldOptions {
            frame: FluxFrame1d::Contact,
            dedner: controls.dedner.then_some(DednerOptions {
                implicit_limiter: controls.implicit_limiter,
            }),
            maximum_star_total_pressure: Some(maximum_star_total_pressure),
        };
        let mut result = match hlld_riemann(local_left, local_right, state.gamma, options) {
            Ok(result) => result,
            Err(MhdError::NoAdmissibleContactFlux) => {
                // A fixed-mass method cannot accept HLLE's nonzero mass flux.
                // Retry with the particle-centered states, matching the
                // reconstruction fallback policy in the scalar MFM path.
                let centered_left = orient_primitive(primitive_at(state, &primitive, j), normal);
                let centered_right = orient_primitive(primitive_at(state, &primitive, i), normal);
                hlld_riemann(centered_left, centered_right, state.gamma, options)?
            }
            Err(error) => return Err(error.into()),
        };
        let mass_roundoff = 128.0
            * f64::EPSILON
            * result.fast_speed_left.max(result.fast_speed_right).max(1.0)
            * left.density.max(right.density);
        if result.method != MhdRiemannMethod::Hlld || result.flux.mass.abs() > mass_roundoff {
            return Err(MhdError::NoAdmissibleContactFlux.into());
        }
        // HLLD's contact frame is mathematically fixed-mass; erase only its
        // last-bit cancellation residue so it cannot enter an extensive rate.
        result.flux.mass = 0.0;
        let mut magnetic_flux = orient_vector(result.flux.magnetic, normal) * face.area;
        let mut dedner_mean_flux = Vector3::ZERO;
        let mut dedner_jump_flux = Vector3::ZERO;
        if controls.dedner {
            // The C meshless path applies the mean cleaning flux immediately,
            // but accumulates the B-jump correction in a separate particle
            // column for a nonlinear cap after every pair is known.
            dedner_mean_flux.x = result.phi_mean * face.signed_area;
            dedner_jump_flux.x = result.phi_db * face.signed_area;
            magnetic_flux = magnetic_flux + dedner_mean_flux;
        }
        let pair_momentum = orient_vector(result.flux.momentum, normal) * face.area;
        let pair_energy = result.flux.total_energy * face.area;
        momentum[i] = momentum[i] + pair_momentum;
        momentum[j] = momentum[j] - pair_momentum;
        total_energy[i] += pair_energy;
        total_energy[j] -= pair_energy;
        magnetic_volume[i] = magnetic_volume[i] + magnetic_flux;
        magnetic_volume[j] = magnetic_volume[j] - magnetic_flux;
        // Use the same corrected normal field and meshless face as the
        // induction flux. A generic MLS gradient of Bx is not the discrete
        // divergence controlled by GIZMO's Powell/Dedner terms.
        let pair_divergence = -result.corrected_normal_b * face.area;
        magnetic_divergence_volume[i] += pair_divergence;
        magnetic_divergence_volume[j] -= pair_divergence;
        if controls.dedner {
            dedner_magnetic_correction[i] = dedner_magnetic_correction[i] + dedner_jump_flux;
            dedner_magnetic_correction[j] = dedner_magnetic_correction[j] - dedner_jump_flux;
            // Mirror the C B.dB coupling. The later primitive conversion
            // subtracts B.d(VB), so this prevents cleaning from being
            // spuriously interpreted as thermal heating/cooling.
            total_energy[i] += primitive.magnetic[i].dot(dedner_mean_flux);
            total_energy[j] -= primitive.magnetic[j].dot(dedner_mean_flux);
        }
        let pair_signal = (result.fast_speed_left + result.fast_speed_right)
            + (left.velocity.x - right.velocity.x).abs();
        maximum_signal_speed[i] = maximum_signal_speed[i].max(pair_signal);
        maximum_signal_speed[j] = maximum_signal_speed[j].max(pair_signal);
        pair_count += 1;
    }

    let velocity_divergence = particle_divergence_at_hsml_1d(
        &state.positions,
        &state
            .velocities
            .iter()
            .map(|value| value.x)
            .collect::<Vec<_>>(),
        &state.masses,
        &state.smoothing_lengths,
        state.box_size,
    )?;
    let magnetic_divergence: Vec<f64> = magnetic_divergence_volume
        .iter()
        .enumerate()
        .map(|(index, &integrated)| integrated / (state.masses[index] / primitive.density[index]))
        .collect();

    for index in 0..particle_count {
        let volume = state.masses[index] / primitive.density[index];
        let cleaning_speed = maximum_signal_speed[index];
        if controls.powell {
            let source_scale = -volume * magnetic_divergence[index];
            momentum[index] = momentum[index] + primitive.magnetic[index] * source_scale;
            total_energy[index] +=
                source_scale * state.velocities[index].dot(primitive.magnetic[index]);
            magnetic_volume[index] =
                magnetic_volume[index] + state.velocities[index] * source_scale;
        }
        if controls.dedner {
            let raw_squared = magnetic_volume[index].squared_norm();
            let cleaning_gradient_scale =
                0.1 * 0.5 * maximum_signal_speed[index] / state.smoothing_lengths[index];
            let regularization_squared =
                primitive.magnetic[index].squared_norm() * cleaning_gradient_scale.powi(2);
            let uncorrected_fourth = (raw_squared + regularization_squared).powi(2);
            let correction_fourth = dedner_magnetic_correction[index].squared_norm().powi(2);
            let tolerance_squared = 100.0;
            let correction_scale = if correction_fourth > tolerance_squared * uncorrected_fourth
                && correction_fourth > 0.0
                && uncorrected_fourth > 0.0
            {
                tolerance_squared * uncorrected_fourth / correction_fourth
            } else {
                1.0
            };
            let applied_correction = dedner_magnetic_correction[index] * correction_scale;
            magnetic_volume[index] = magnetic_volume[index] + applied_correction;
            total_energy[index] += primitive.magnetic[index].dot(applied_correction);

            let phi = primitive.cleaning_scalar[index];
            let hyperbolic = dedner_hyperbolic_source(
                magnetic_divergence[index],
                0.5 * cleaning_speed,
                controls.hyperbolic_sigma,
            )?;
            let parabolic = dedner_parabolic_source(
                phi,
                cleaning_speed,
                state.smoothing_lengths[index],
                controls.parabolic_sigma,
            )?;
            cleaning_mass[index] += state.masses[index] * (hyperbolic + parabolic);
        }
    }

    let mut acceleration = Vec::with_capacity(particle_count);
    let mut specific_internal_energy = Vec::with_capacity(particle_count);
    for index in 0..particle_count {
        let mass = state.masses[index];
        let volume = mass / primitive.density[index];
        let velocity = state.velocities[index];
        let field = primitive.magnetic[index];
        let acceleration_i = momentum[index] / mass;
        // d(VB magnetic energy) = B.d(VB) - 0.5 B^2 dV.
        // Particle volume obeys dV/dt = V div(v).
        let internal_rate = (total_energy[index]
            - velocity.dot(momentum[index])
            - field.dot(magnetic_volume[index])
            + 0.5 * field.squared_norm() * volume * velocity_divergence[index])
            / mass;
        if !acceleration_i.is_finite() || !internal_rate.is_finite() {
            return Err(invalid(Some(index), "primitive_rate", f64::NAN));
        }
        acceleration.push(acceleration_i);
        specific_internal_energy.push(internal_rate);
    }
    Ok(MhdMfmRates1d {
        momentum,
        total_energy,
        magnetic_volume,
        cleaning_mass,
        acceleration,
        specific_internal_energy,
        maximum_signal_speed,
        velocity_divergence,
        magnetic_divergence,
        pair_count,
    })
}

/// Global fast-wave CFL bound using the legacy effective particle size.
///
/// # Errors
///
/// Returns an error for invalid state, rates, or Courant factor.
pub fn global_mhd_courant_timestep_1d(
    state: &MhdMfmState1d,
    rates: &MhdMfmRates1d,
    courant_factor: f64,
) -> Result<f64, MhdEvolutionError> {
    state.validate()?;
    if !courant_factor.is_finite() || courant_factor <= 0.0 || courant_factor > 0.5 {
        return Err(invalid(None, "courant_factor", courant_factor));
    }
    validate_lengths(
        state.positions.len(),
        &[("maximum_signal_speed", rates.maximum_signal_speed.len())],
    )?;
    let density = density_at_hsml_1d(
        &state.positions,
        &state.masses,
        &state.smoothing_lengths,
        state.box_size,
    )?;
    let mut timestep = f64::INFINITY;
    for (index, estimate) in density.iter().enumerate() {
        let signal = rates.maximum_signal_speed[index];
        let particle_size = 2.0 * state.smoothing_lengths[index] / estimate.effective_neighbors;
        let candidate = courant_factor * particle_size / (0.5 * signal);
        if !candidate.is_finite() || candidate <= 0.0 {
            return Err(invalid(Some(index), "courant_timestep", candidate));
        }
        timestep = timestep.min(candidate);
    }
    Ok(timestep)
}

/// Advance one periodic synchronized kick-drift-kick step.
///
/// The kick acts on extensive conserved variables. Positions drift with the
/// half-kicked velocity; endpoint density and smoothing length are then solved
/// before primitive recovery and the second force evaluation.
///
/// # Errors
///
/// Returns an error for invalid inputs, failed smoothing-length solve, or loss
/// of positivity during conservative-to-primitive recovery.
#[allow(clippy::too_many_arguments)]
pub fn advance_mhd_kdk_1d(
    state: &MhdMfmState1d,
    old_rates: &MhdMfmRates1d,
    timestep: f64,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
    minimum_specific_internal_energy: f64,
    controls: DivergenceControl1d,
) -> Result<MhdKdkResult1d, MhdEvolutionError> {
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
    let primitive = state.primitive_columns()?;
    let conserved = extensive_columns(state, &primitive);
    let half = kick_extensive(&conserved, old_rates, 0.5 * timestep);
    let half_velocity: Vec<Vector3> = half
        .momentum
        .iter()
        .zip(&state.masses)
        .map(|(&momentum, &mass)| momentum / mass)
        .collect();
    let endpoint_positions: Vec<f64> = state
        .positions
        .iter()
        .zip(&half_velocity)
        .map(|(&position, &velocity)| (position + timestep * velocity.x).rem_euclid(state.box_size))
        .collect();
    let solved = solve_public_c_smoothing_lengths_from_seeds_1d(
        &endpoint_positions,
        &state.masses,
        &state.smoothing_lengths,
        state.box_size,
        desired_neighbors,
        neighbor_tolerance,
    )?;
    let endpoint_hsml: Vec<f64> = solved
        .iter()
        .map(|particle| particle.smoothing_length)
        .collect();
    let half_state = recover_state(
        endpoint_positions,
        state.masses.clone(),
        endpoint_hsml,
        state.box_size,
        state.gamma,
        &half,
        minimum_specific_internal_energy,
    )?;
    let new_rates = mhd_mfm_spatial_rates_1d(&half_state, controls)?;
    let final_conserved = kick_extensive(&half, &new_rates, 0.5 * timestep);
    let final_state = recover_state(
        half_state.positions,
        half_state.masses,
        half_state.smoothing_lengths,
        half_state.box_size,
        half_state.gamma,
        &final_conserved,
        minimum_specific_internal_energy,
    )?;
    Ok(MhdKdkResult1d {
        state: final_state,
        rates: new_rates,
    })
}

#[derive(Clone)]
struct ExtensiveColumns {
    momentum: Vec<Vector3>,
    energy: Vec<f64>,
    magnetic: Vec<Vector3>,
    cleaning: Vec<f64>,
}

fn extensive_columns(state: &MhdMfmState1d, primitive: &MhdPrimitiveColumns1d) -> ExtensiveColumns {
    let mut momentum = Vec::with_capacity(state.positions.len());
    let mut energy = Vec::with_capacity(state.positions.len());
    for index in 0..state.positions.len() {
        let mass = state.masses[index];
        let volume = mass / primitive.density[index];
        momentum.push(state.velocities[index] * mass);
        energy.push(
            mass * state.specific_internal_energy[index]
                + 0.5 * mass * state.velocities[index].squared_norm()
                + 0.5 * volume * primitive.magnetic[index].squared_norm(),
        );
    }
    ExtensiveColumns {
        momentum,
        energy,
        magnetic: state.magnetic_volume.clone(),
        cleaning: state.cleaning_mass.clone(),
    }
}

fn kick_extensive(
    state: &ExtensiveColumns,
    rates: &MhdMfmRates1d,
    duration: f64,
) -> ExtensiveColumns {
    ExtensiveColumns {
        momentum: state
            .momentum
            .iter()
            .zip(&rates.momentum)
            .map(|(&value, &rate)| value + rate * duration)
            .collect(),
        energy: state
            .energy
            .iter()
            .zip(&rates.total_energy)
            .map(|(&value, &rate)| value + rate * duration)
            .collect(),
        magnetic: state
            .magnetic
            .iter()
            .zip(&rates.magnetic_volume)
            .map(|(&value, &rate)| value + rate * duration)
            .collect(),
        cleaning: state
            .cleaning
            .iter()
            .zip(&rates.cleaning_mass)
            .map(|(&value, &rate)| value + rate * duration)
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn recover_state(
    positions: Vec<f64>,
    masses: Vec<f64>,
    smoothing_lengths: Vec<f64>,
    box_size: f64,
    gamma: f64,
    conserved: &ExtensiveColumns,
    minimum_specific_internal_energy: f64,
) -> Result<MhdMfmState1d, MhdEvolutionError> {
    let density = density_values(&positions, &masses, &smoothing_lengths, box_size)?;
    let mut velocities = Vec::with_capacity(positions.len());
    let mut internal_energy = Vec::with_capacity(positions.len());
    for index in 0..positions.len() {
        let mass = masses[index];
        let volume = mass / density[index];
        let velocity = conserved.momentum[index] / mass;
        let field = conserved.magnetic[index] / volume;
        let thermal = conserved.energy[index]
            - 0.5 * mass * velocity.squared_norm()
            - 0.5 * volume * field.squared_norm();
        let specific = thermal / mass;
        if !specific.is_finite() || specific < minimum_specific_internal_energy {
            return Err(invalid(
                Some(index),
                "recovered_specific_internal_energy",
                specific,
            ));
        }
        velocities.push(velocity);
        internal_energy.push(specific);
    }
    let state = MhdMfmState1d {
        positions,
        masses,
        velocities,
        specific_internal_energy: internal_energy,
        smoothing_lengths,
        magnetic_volume: conserved.magnetic.clone(),
        cleaning_mass: conserved.cleaning.clone(),
        box_size,
        gamma,
    };
    state.validate()?;
    Ok(state)
}

fn primitive_gradients(
    state: &MhdMfmState1d,
    primitive: &MhdPrimitiveColumns1d,
) -> Result<Gradients, MhdEvolutionError> {
    let velocity_columns = vector_columns(&state.velocities);
    let magnetic_columns = vector_columns(&primitive.magnetic);
    Ok(Gradients {
        density: scalar_gradient(state, &primitive.density, true)?,
        pressure: scalar_gradient(state, &primitive.pressure, true)?,
        velocity: [
            scalar_gradient(state, &velocity_columns[0], false)?,
            scalar_gradient(state, &velocity_columns[1], false)?,
            scalar_gradient(state, &velocity_columns[2], false)?,
        ],
        magnetic: [
            scalar_gradient(state, &magnetic_columns[0], false)?,
            scalar_gradient(state, &magnetic_columns[1], false)?,
            scalar_gradient(state, &magnetic_columns[2], false)?,
        ],
        cleaning: scalar_gradient(state, &primitive.cleaning_scalar, false)?,
    })
}

fn scalar_gradient(
    state: &MhdMfmState1d,
    values: &[f64],
    positivity_preserving: bool,
) -> Result<Vec<f64>, MhdEvolutionError> {
    Ok(gradients_at_hsml_1d(
        &state.positions,
        values,
        &state.smoothing_lengths,
        state.box_size,
        0.0,
        positivity_preserving,
    )?
    .into_iter()
    .map(|gradient| MHD_GRADIENT_LIMITER * gradient.limited)
    .collect())
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_pair(
    state: &MhdMfmState1d,
    primitive: &MhdPrimitiveColumns1d,
    gradients: &Gradients,
    i: usize,
    j: usize,
    face: MeshlessFace1d,
    closure_errors: &[f64],
) -> Result<(IdealMhdPrimitive1d, IdealMhdPrimitive1d), MhdEvolutionError> {
    // Disable reconstruction in a geometrically unclosed neighborhood, as in
    // the scalar MFM pair path.
    let order = if 0.5 * (closure_errors[i] + closure_errors[j]) <= 1.0 {
        ReconstructionOrder::First
    } else {
        ReconstructionOrder::Zeroth
    };
    let reconstruct_ordered = |values: &[f64], slopes: &[f64]| {
        reconstruct_face_states_1d(values[i], slopes[i], values[j], slopes[j], face, order)
    };
    let rho = reconstruct_ordered(&primitive.density, &gradients.density)?;
    let pressure = reconstruct_ordered(&primitive.pressure, &gradients.pressure)?;
    let velocity_components = vector_columns(&state.velocities);
    let magnetic_components = vector_columns(&primitive.magnetic);
    let velocity_x = reconstruct_ordered(&velocity_components[0], &gradients.velocity[0])?;
    let velocity_y = reconstruct_ordered(&velocity_components[1], &gradients.velocity[1])?;
    let velocity_z = reconstruct_ordered(&velocity_components[2], &gradients.velocity[2])?;
    let magnetic_x = reconstruct_ordered(&magnetic_components[0], &gradients.magnetic[0])?;
    let magnetic_y = reconstruct_ordered(&magnetic_components[1], &gradients.magnetic[1])?;
    let magnetic_z = reconstruct_ordered(&magnetic_components[2], &gradients.magnetic[2])?;
    let phi = reconstruct_ordered(&primitive.cleaning_scalar, &gradients.cleaning)?;
    let left = IdealMhdPrimitive1d {
        density: rho.left,
        velocity: Vector3::new(velocity_x.left, velocity_y.left, velocity_z.left),
        gas_pressure: pressure.left,
        magnetic: Vector3::new(magnetic_x.left, magnetic_y.left, magnetic_z.left),
        cleaning_scalar: phi.left,
    };
    let right = IdealMhdPrimitive1d {
        density: rho.right,
        velocity: Vector3::new(velocity_x.right, velocity_y.right, velocity_z.right),
        gas_pressure: pressure.right,
        magnetic: Vector3::new(magnetic_x.right, magnetic_y.right, magnetic_z.right),
        cleaning_scalar: phi.right,
    };
    Ok((left, right))
}

fn primitive_at(
    state: &MhdMfmState1d,
    primitive: &MhdPrimitiveColumns1d,
    index: usize,
) -> IdealMhdPrimitive1d {
    IdealMhdPrimitive1d {
        density: primitive.density[index],
        velocity: state.velocities[index],
        gas_pressure: primitive.pressure[index],
        magnetic: primitive.magnetic[index],
        cleaning_scalar: primitive.cleaning_scalar[index],
    }
}

fn orient_primitive(mut primitive: IdealMhdPrimitive1d, normal: f64) -> IdealMhdPrimitive1d {
    primitive.velocity.x *= normal;
    primitive.magnetic.x *= normal;
    primitive
}

fn orient_vector(mut vector: Vector3, normal: f64) -> Vector3 {
    vector.x *= normal;
    vector
}

fn vector_columns(values: &[Vector3]) -> [Vec<f64>; 3] {
    [
        values.iter().map(|value| value.x).collect(),
        values.iter().map(|value| value.y).collect(),
        values.iter().map(|value| value.z).collect(),
    ]
}

fn density_values(
    positions: &[f64],
    masses: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) -> Result<Vec<f64>, MhdEvolutionError> {
    Ok(
        density_at_hsml_1d(positions, masses, smoothing_lengths, box_size)?
            .into_iter()
            .map(|estimate| estimate.density)
            .collect(),
    )
}

fn validate_controls(controls: DivergenceControl1d) -> Result<(), MhdEvolutionError> {
    for (field, value) in [
        ("hyperbolic_sigma", controls.hyperbolic_sigma),
        ("parabolic_sigma", controls.parabolic_sigma),
        ("implicit_limiter", controls.implicit_limiter),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(invalid(None, field, value));
        }
    }
    Ok(())
}

fn validate_lengths(
    expected: usize,
    columns: &[(&'static str, usize)],
) -> Result<(), MhdEvolutionError> {
    for &(field, actual) in columns {
        if actual != expected {
            return Err(MhdEvolutionError::MismatchedLength {
                field,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn validate_rate_lengths(rates: &MhdMfmRates1d, expected: usize) -> Result<(), MhdEvolutionError> {
    validate_lengths(
        expected,
        &[
            ("momentum_rates", rates.momentum.len()),
            ("total_energy_rates", rates.total_energy.len()),
            ("magnetic_volume_rates", rates.magnetic_volume.len()),
            ("cleaning_mass_rates", rates.cleaning_mass.len()),
            ("acceleration", rates.acceleration.len()),
            (
                "specific_internal_energy_rates",
                rates.specific_internal_energy.len(),
            ),
            ("maximum_signal_speed", rates.maximum_signal_speed.len()),
            ("velocity_divergence", rates.velocity_divergence.len()),
            ("magnetic_divergence", rates.magnetic_divergence.len()),
        ],
    )
}

fn invalid(index: Option<usize>, field: &'static str, value: f64) -> MhdEvolutionError {
    MhdEvolutionError::InvalidState {
        index,
        field,
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    const GAMMA: f64 = 5.0 / 3.0;

    fn lattice_state(count: usize, amplitude: f64) -> MhdMfmState1d {
        let count_f = f64::from(u32::try_from(count).unwrap());
        let positions: Vec<f64> = (0..count)
            .map(|index| (f64::from(u32::try_from(index).unwrap()) + 0.5) / count_f)
            .collect();
        let masses = vec![1.0 / count_f; count];
        let hsml = vec![4.0 / count_f; count];
        let velocities: Vec<Vector3> = positions
            .iter()
            .map(|&x| {
                let wave = amplitude * (TAU * x).sin();
                Vector3::new(-2.0 * wave, 0.942_809_041_58 * wave, wave / 3.0)
            })
            .collect();
        let magnetic: Vec<Vector3> = positions
            .iter()
            .map(|&x| {
                let wave = amplitude * (TAU * x).sin();
                Vector3::new(
                    1.0,
                    2.0_f64.sqrt() + 1.885_618_083_16 * wave,
                    0.5 + 2.0 * wave / 3.0,
                )
            })
            .collect();
        let internal_energy: Vec<f64> = positions
            .iter()
            .map(|&x| 0.9 + 0.6 * amplitude * (TAU * x).sin())
            .collect();
        MhdMfmState1d::from_primitive(
            positions,
            masses,
            velocities,
            internal_energy,
            hsml,
            &magnetic,
            &vec![0.0; count],
            1.0,
            GAMMA,
        )
        .unwrap()
    }

    fn no_sources() -> DivergenceControl1d {
        DivergenceControl1d {
            powell: false,
            dedner: false,
            ..DivergenceControl1d::default()
        }
    }

    fn sum_vectors(values: &[Vector3]) -> Vector3 {
        values
            .iter()
            .copied()
            .fold(Vector3::ZERO, |sum, value| sum + value)
    }

    fn max_abs(vector: Vector3) -> f64 {
        vector.x.abs().max(vector.y.abs()).max(vector.z.abs())
    }

    #[test]
    fn uniform_stationary_state_has_zero_induction_and_net_pair_rates() {
        let mut state = lattice_state(32, 0.0);
        state.velocities.fill(Vector3::ZERO);
        let rates = mhd_mfm_spatial_rates_1d(&state, no_sources()).unwrap();
        assert!(max_abs(sum_vectors(&rates.momentum)) < 1.0e-12);
        assert!(rates.total_energy.iter().sum::<f64>().abs() < 1.0e-12);
        assert!(max_abs(sum_vectors(&rates.magnetic_volume)) < 1.0e-12);
        assert!(
            rates
                .magnetic_volume
                .iter()
                .all(|rate| max_abs(*rate) < 1.0e-12)
        );
    }

    #[test]
    fn unordered_pair_accumulation_is_extensively_conservative() {
        let state = lattice_state(32, 1.0e-4);
        let rates = mhd_mfm_spatial_rates_1d(&state, no_sources()).unwrap();
        assert!(rates.pair_count > 0);
        assert!(max_abs(sum_vectors(&rates.momentum)) < 1.0e-12);
        assert!(rates.total_energy.iter().sum::<f64>().abs() < 1.0e-12);
        assert!(max_abs(sum_vectors(&rates.magnetic_volume)) < 1.0e-12);
    }

    #[test]
    fn one_dimensional_induction_preserves_normal_magnetic_flux() {
        let state = lattice_state(32, 1.0e-6);
        let rates = mhd_mfm_spatial_rates_1d(&state, no_sources()).unwrap();
        let primitive = state.primitive_columns().unwrap();
        // `V Bx` changes when a Lagrangian particle volume changes.  The
        // physical invariant is primitive Bx: d(VBx)-Bx*dV must vanish.
        for index in 0..state.positions.len() {
            let volume = state.masses[index] / primitive.density[index];
            let primitive_bx_rate = (rates.magnetic_volume[index].x
                - primitive.magnetic[index].x * volume * rates.velocity_divergence[index])
                / volume;
            assert!(
                primitive_bx_rate.abs() < 2.0e-6,
                "particle {index}: dBx/dt={primitive_bx_rate}, dVB/dt={}, B={}, V={}, divv={}",
                rates.magnetic_volume[index].x,
                primitive.magnetic[index].x,
                volume,
                rates.velocity_divergence[index]
            );
        }
        assert!(
            rates
                .magnetic_volume
                .iter()
                .any(|rate| rate.y.abs() > 1.0e-12)
        );
    }

    #[test]
    fn fast_wave_generates_real_transverse_rhs() {
        let state = lattice_state(32, 1.0e-4);
        let rates = mhd_mfm_spatial_rates_1d(&state, no_sources()).unwrap();
        assert!(rates.acceleration.iter().any(|rate| rate.y.abs() > 1.0e-9));
        assert!(
            rates
                .magnetic_volume
                .iter()
                .any(|rate| rate.y.abs() > 1.0e-9)
        );
    }

    #[test]
    fn courant_bound_uses_fast_magnetosonic_signal() {
        let state = lattice_state(32, 0.0);
        let rates = mhd_mfm_spatial_rates_1d(&state, no_sources()).unwrap();
        let primitive = state.primitive_columns().unwrap();
        let cfast = fast_magnetosonic_speed(primitive_at(&state, &primitive, 0), GAMMA).unwrap();
        assert!((cfast - 2.0).abs() < 1.0e-12);
        assert!(
            rates
                .maximum_signal_speed
                .iter()
                .all(|&speed| speed >= cfast)
        );
        let dt = global_mhd_courant_timestep_1d(&state, &rates, 0.2).unwrap();
        assert!(dt.is_finite() && dt > 0.0 && dt < 0.02);
    }

    #[test]
    fn dedner_source_damps_phi_and_responds_to_divergence() {
        let mut uniform = lattice_state(32, 0.0);
        for index in 0..uniform.positions.len() {
            uniform.cleaning_mass[index] = uniform.masses[index] * 1.0e-3;
        }
        let uniform_rates =
            mhd_mfm_spatial_rates_1d(&uniform, DivergenceControl1d::default()).unwrap();
        assert!(uniform_rates.cleaning_mass.iter().all(|&rate| rate < 0.0));

        let mut state = lattice_state(32, 0.0);
        let density = state.primitive_columns().unwrap().density;
        for (((&mass, &rho), &position), magnetic_volume) in state
            .masses
            .iter()
            .zip(&density)
            .zip(&state.positions)
            .zip(&mut state.magnetic_volume)
        {
            let volume = mass / rho;
            magnetic_volume.x = volume * (1.0 + 1.0e-3 * (TAU * position).sin());
        }
        let rates = mhd_mfm_spatial_rates_1d(&state, DivergenceControl1d::default()).unwrap();
        assert!(
            rates
                .magnetic_divergence
                .iter()
                .any(|value| value.abs() > 1.0e-5)
        );
        assert!(rates.cleaning_mass.iter().any(|value| value.abs() > 1.0e-8));
    }

    #[test]
    fn short_periodic_kdk_is_finite_conservative_and_not_a_noop() {
        let mut state = lattice_state(32, 1.0e-5);
        let initial_momentum = state.total_momentum();
        let initial_energy = state.total_energy().unwrap();
        let initial_transverse = state.velocities[0].y;
        let mut rates = mhd_mfm_spatial_rates_1d(&state, no_sources()).unwrap();
        for _ in 0..3 {
            let dt = global_mhd_courant_timestep_1d(&state, &rates, 0.1)
                .unwrap()
                .min(2.0e-4);
            let result =
                advance_mhd_kdk_1d(&state, &rates, dt, 4.0, 1.0e-3, 1.0e-12, no_sources()).unwrap();
            state = result.state;
            rates = result.rates;
        }
        assert!(state.velocities.iter().all(|value| value.is_finite()));
        assert!(
            state
                .primitive_columns()
                .unwrap()
                .magnetic
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(max_abs(state.total_momentum() - initial_momentum) < 1.0e-12);
        assert!((state.total_energy().unwrap() - initial_energy).abs() < 1.0e-10);
        assert!((state.velocities[0].y - initial_transverse).abs() > f64::EPSILON);
    }
}
