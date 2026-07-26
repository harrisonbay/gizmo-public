//! Two-dimensional rectangular-periodic meshless finite-mass ideal-MHD evolution.
//!
//! This is a genuine planar operator: particle geometry, gradients, face
//! vectors, drift, and Riemann normals all retain both spatial coordinates.
//! Magnetic flux is stored as the extensive `V B` variable used by the public
//! MFM equations. Both fixed-H and public-C adaptive-H synchronized KDK entry
//! points are available; neither emulates the public code's hierarchical
//! individual-particle time bins.

use std::error::Error;
use std::fmt;

use crate::meshless_2d::{
    Box2d, GeometryError, InteractionPair2d, MeshlessPoint2d, Vector2, density_at_hsml_2d,
    interacting_pairs_2d, inverse_moments_2d, meshless_face_geometry_2d,
    scalar_gradients_at_hsml_2d, solve_public_c_smoothing_lengths_from_seeds_2d,
};
use crate::mhd::{
    DednerOptions, FluxFrame1d, HlldOptions, IdealMhdPrimitive1d, MhdError, MhdRiemannMethod,
    Vector3, dedner_hyperbolic_source, dedner_parabolic_source, fast_magnetosonic_speed,
};
use crate::mhd_2d::{HlldResult2d, Mhd2dError, hlld_riemann_2d};

const LOCAL_GRADIENT_LIMITER_DISTANCE_FRACTION: f64 = 0.25;
const MIN_REAL_NUMBER: f64 = 1.0e-56;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DivergenceControl2d {
    pub powell: bool,
    pub dedner: bool,
    pub hyperbolic_sigma: f64,
    pub parabolic_sigma: f64,
    pub implicit_limiter: f64,
}

impl Default for DivergenceControl2d {
    fn default() -> Self {
        Self {
            powell: true,
            dedner: true,
            hyperbolic_sigma: 1.0,
            parabolic_sigma: 1.0,
            implicit_limiter: 0.75,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MhdMfmState2d {
    pub positions: Vec<Vector2>,
    pub masses: Vec<f64>,
    pub velocities: Vec<Vector3>,
    pub specific_internal_energy: Vec<f64>,
    /// Full compact-support radii. [`advance_mhd_kdk_2d`] retains these, while
    /// [`advance_mhd_kdk_adaptive_2d`] resolves them after each drift.
    pub smoothing_lengths: Vec<f64>,
    /// Extensive magnetic conservative variable, `mass / density * B`.
    pub magnetic_volume: Vec<Vector3>,
    /// Mass-based Dedner variable, `mass * phi`.
    pub cleaning_mass: Vec<f64>,
    pub domain: Box2d,
    pub gamma: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MhdPrimitiveColumns2d {
    pub density: Vec<f64>,
    pub pressure: Vec<f64>,
    pub magnetic: Vec<Vector3>,
    pub cleaning_scalar: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MhdMfmRates2d {
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

#[derive(Clone, Debug, PartialEq)]
pub struct MhdKdkResult2d {
    pub state: MhdMfmState2d,
    pub rates: MhdMfmRates2d,
}

#[derive(Debug)]
pub enum MhdEvolution2dError {
    Geometry(GeometryError),
    Mhd(MhdError),
    Mhd2d(Mhd2dError),
    PairRiemann {
        i: usize,
        j: usize,
        error: Mhd2dError,
    },
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

impl fmt::Display for MhdEvolution2dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(error) => write!(formatter, "{error}"),
            Self::Mhd(error) => write!(formatter, "{error}"),
            Self::Mhd2d(error) => write!(formatter, "{error}"),
            Self::PairRiemann { i, j, error } => {
                write!(
                    formatter,
                    "2-D MHD Riemann failure for pair ({i}, {j}): {error}"
                )
            }
            Self::InvalidState {
                index,
                field,
                value,
            } => {
                write!(
                    formatter,
                    "invalid 2-D MHD state {field}={value} at {index:?}"
                )
            }
            Self::MismatchedLength {
                field,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "{field} has length {actual}, expected {expected}"
                )
            }
        }
    }
}

impl Error for MhdEvolution2dError {}

impl From<GeometryError> for MhdEvolution2dError {
    fn from(value: GeometryError) -> Self {
        Self::Geometry(value)
    }
}
impl From<MhdError> for MhdEvolution2dError {
    fn from(value: MhdError) -> Self {
        Self::Mhd(value)
    }
}
impl From<Mhd2dError> for MhdEvolution2dError {
    fn from(value: Mhd2dError) -> Self {
        Self::Mhd2d(value)
    }
}

impl MhdMfmState2d {
    /// Construct an owned conservative state from primitive particle columns.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched columns, invalid geometry, or a
    /// non-physical primitive state.
    #[allow(clippy::too_many_arguments)]
    pub fn from_primitive(
        positions: Vec<Vector2>,
        masses: Vec<f64>,
        velocities: Vec<Vector3>,
        specific_internal_energy: Vec<f64>,
        smoothing_lengths: Vec<f64>,
        magnetic: &[Vector3],
        cleaning_scalar: &[f64],
        domain: Box2d,
        gamma: f64,
    ) -> Result<Self, MhdEvolution2dError> {
        validate_lengths(
            positions.len(),
            &[
                ("masses", masses.len()),
                ("velocities", velocities.len()),
                ("specific_internal_energy", specific_internal_energy.len()),
                ("smoothing_lengths", smoothing_lengths.len()),
                ("magnetic", magnetic.len()),
                ("cleaning_scalar", cleaning_scalar.len()),
            ],
        )?;
        let density = density_values(&positions, &masses, &smoothing_lengths, domain)?;
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
            domain,
            gamma,
        };
        state.validate()?;
        Ok(state)
    }

    /// Recover density and primitive thermodynamic, magnetic, and cleaning fields.
    ///
    /// # Errors
    ///
    /// Returns an error when density reconstruction or primitive validation fails.
    pub fn primitive_columns(&self) -> Result<MhdPrimitiveColumns2d, MhdEvolution2dError> {
        self.validate()?;
        let density = density_values(
            &self.positions,
            &self.masses,
            &self.smoothing_lengths,
            self.domain,
        )?;
        let mut pressure = Vec::with_capacity(self.positions.len());
        let mut magnetic = Vec::with_capacity(self.positions.len());
        let mut cleaning_scalar = Vec::with_capacity(self.positions.len());
        for (index, &rho) in density.iter().enumerate() {
            let volume = self.masses[index] / rho;
            let field = self.magnetic_volume[index] / volume;
            let phi = self.cleaning_mass[index] / self.masses[index];
            let gas_pressure = (self.gamma - 1.0) * rho * self.specific_internal_energy[index];
            IdealMhdPrimitive1d {
                density: rho,
                velocity: self.velocities[index],
                gas_pressure,
                magnetic: field,
                cleaning_scalar: phi,
            }
            .to_conserved(self.gamma)?;
            pressure.push(gas_pressure);
            magnetic.push(field);
            cleaning_scalar.push(phi);
        }
        Ok(MhdPrimitiveColumns2d {
            density,
            pressure,
            magnetic,
            cleaning_scalar,
        })
    }

    /// Sum thermal, kinetic, and magnetic extensive energy.
    ///
    /// # Errors
    ///
    /// Returns an error when primitive recovery fails.
    pub fn total_energy(&self) -> Result<f64, MhdEvolution2dError> {
        let primitive = self.primitive_columns()?;
        Ok((0..self.positions.len())
            .map(|i| {
                let volume = self.masses[i] / primitive.density[i];
                self.masses[i] * self.specific_internal_energy[i]
                    + 0.5 * self.masses[i] * self.velocities[i].squared_norm()
                    + 0.5 * volume * primitive.magnetic[i].squared_norm()
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

    fn validate(&self) -> Result<(), MhdEvolution2dError> {
        let count = self.positions.len();
        validate_lengths(
            count,
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
        if !self.gamma.is_finite() || self.gamma <= 1.0 {
            return Err(invalid(None, "gamma", self.gamma));
        }
        for i in 0..count {
            if !self.domain.contains(self.positions[i]) {
                return Err(invalid(Some(i), "position", f64::NAN));
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
            if !self.cleaning_mass[i].is_finite() {
                return Err(invalid(Some(i), "cleaning_mass", self.cleaning_mass[i]));
            }
            if !self.velocities[i].is_finite() || !self.magnetic_volume[i].is_finite() {
                return Err(invalid(Some(i), "vector_column", f64::NAN));
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Gradients2d {
    density: Vec<Vector2>,
    pressure: Vec<Vector2>,
    velocity: [Vec<Vector2>; 3],
    magnetic: [Vec<Vector2>; 3],
    cleaning: Vec<Vector2>,
}

struct GradientLimiterGeometry2d {
    pairs: Vec<InteractionPair2d>,
    maximum_neighbor_distance: Vec<f64>,
    condition_number: Vec<f64>,
}

#[derive(Clone, Copy)]
enum GradientConstraint2d {
    Positive,
    Signed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RiemannRetry2d {
    Reconstructed,
    Centered,
    QuietCentered,
}

#[derive(Clone, Copy)]
enum FaceLimiterMode2d {
    Standard,
    Cleaning,
}

#[allow(clippy::too_many_lines)]
/// Evaluate the semidiscrete rectangular-periodic 2-D MFM ideal-MHD operator.
///
/// # Errors
///
/// Returns an error for invalid particle state, meshless geometry, or Riemann
/// arithmetic.
pub fn mhd_mfm_spatial_rates_2d(
    state: &MhdMfmState2d,
    controls: DivergenceControl2d,
) -> Result<MhdMfmRates2d, MhdEvolution2dError> {
    state.validate()?;
    validate_controls(controls)?;
    let primitive = state.primitive_columns()?;
    let gradients = primitive_gradients(state, &primitive)?;
    let moments = inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    let count = state.positions.len();
    let mut momentum = vec![Vector3::ZERO; count];
    let mut total_energy = vec![0.0; count];
    let mut magnetic_volume = vec![Vector3::ZERO; count];
    let mut cleaning_mass = vec![0.0; count];
    let mut magnetic_divergence_volume = vec![0.0; count];
    let mut dedner_jump = vec![Vector3::ZERO; count];
    let mut maximum_signal_speed = (0..count)
        .map(|i| fast_magnetosonic_speed(primitive_at(state, &primitive, i), state.gamma))
        .collect::<Result<Vec<_>, _>>()?;

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
        // Face normal points j -> i, hence HLLD left is j and right is i.
        let (left, right) = reconstruct_pair(
            state,
            &primitive,
            &gradients,
            pair.j,
            pair.i,
            face.offset_from_j,
            face.offset_from_i,
        );
        let n = face.unit_normal;
        let displacement = state
            .domain
            .displacement(state.positions[pair.i], state.positions[pair.j])?;
        let radial = displacement / displacement.norm();
        let velocity_difference = state.velocities[pair.i] - state.velocities[pair.j];
        let face_approach = velocity_difference
            .x
            .mul_add(n.x, velocity_difference.y * n.y)
            .min(0.0);
        let radial_approach = velocity_difference
            .x
            .mul_add(radial.x, velocity_difference.y * radial.y)
            .min(0.0);
        let approach_squared =
            (face_approach * face_approach).max(radial_approach * radial_approach);
        let pressure_cap = 2.2
            * (primitive_at(state, &primitive, pair.i).total_pressure()
                + primitive.density[pair.i] * approach_squared
                + primitive_at(state, &primitive, pair.j).total_pressure()
                + primitive.density[pair.j] * approach_squared);
        let centered_left = primitive_at(state, &primitive, pair.j);
        let centered_right = primitive_at(state, &primitive, pair.i);
        let (result, _) = solve_hlld_with_public_retries_2d(
            left,
            right,
            centered_left,
            centered_right,
            n,
            state.gamma,
            controls,
            pressure_cap,
        )
        .map_err(|error| match error {
            MhdEvolution2dError::Mhd2d(error) => MhdEvolution2dError::PairRiemann {
                i: pair.i,
                j: pair.j,
                error,
            },
            other => other,
        })?;
        let mass_roundoff = 128.0
            * f64::EPSILON
            * result.fast_speed_left.max(result.fast_speed_right).max(1.0)
            * left.density.max(right.density);
        if result.method != MhdRiemannMethod::Hlld || result.flux.mass.abs() > mass_roundoff {
            return Err(MhdError::NoAdmissibleContactFlux.into());
        }
        let pair_momentum = result.flux.momentum * face.area;
        let pair_energy = result.flux.total_energy * face.area;
        let mut pair_magnetic = result.flux.magnetic * face.area;
        let normal3 = Vector3::new(n.x, n.y, 0.0);
        if controls.dedner {
            let mean = normal3 * (result.phi_mean * face.area);
            let jump = normal3 * (result.phi_db * face.area);
            pair_magnetic = pair_magnetic + mean;
            dedner_jump[pair.i] = dedner_jump[pair.i] + jump;
            dedner_jump[pair.j] = dedner_jump[pair.j] - jump;
            total_energy[pair.i] += primitive.magnetic[pair.i].dot(mean);
            total_energy[pair.j] -= primitive.magnetic[pair.j].dot(mean);
        }
        momentum[pair.i] = momentum[pair.i] + pair_momentum;
        momentum[pair.j] = momentum[pair.j] - pair_momentum;
        total_energy[pair.i] += pair_energy;
        total_energy[pair.j] -= pair_energy;
        magnetic_volume[pair.i] = magnetic_volume[pair.i] + pair_magnetic;
        magnetic_volume[pair.j] = magnetic_volume[pair.j] - pair_magnetic;
        let pair_divergence = -result.corrected_normal_b * face.area;
        magnetic_divergence_volume[pair.i] += pair_divergence;
        magnetic_divergence_volume[pair.j] -= pair_divergence;
        let pair_signal =
            directional_fast_speed(primitive_at(state, &primitive, pair.i), radial, state.gamma)
                + directional_fast_speed(
                    primitive_at(state, &primitive, pair.j),
                    radial,
                    state.gamma,
                )
                - radial_approach;
        maximum_signal_speed[pair.i] = maximum_signal_speed[pair.i].max(pair_signal);
        maximum_signal_speed[pair.j] = maximum_signal_speed[pair.j].max(pair_signal);
    }

    let global_fastest_wave_speed = (0..count).fold(0.0_f64, |maximum, i| {
        let isotropic_fast = ((state.gamma * primitive.pressure[i]
            + primitive.magnetic[i].squared_norm())
            / primitive.density[i])
            .sqrt();
        maximum.max(isotropic_fast.max(0.5 * maximum_signal_speed[i]))
    });
    let velocity_divergence = velocity_divergence(state)?;
    let magnetic_divergence: Vec<f64> = magnetic_divergence_volume
        .iter()
        .enumerate()
        .map(|(i, &integrated)| integrated / (state.masses[i] / primitive.density[i]))
        .collect();
    for i in 0..count {
        let volume = state.masses[i] / primitive.density[i];
        let particle_size = volume.sqrt();
        let cleaning_speed = maximum_signal_speed[i];
        if controls.powell {
            let scale = -volume * magnetic_divergence[i];
            momentum[i] = momentum[i] + primitive.magnetic[i] * scale;
            total_energy[i] += scale * state.velocities[i].dot(primitive.magnetic[i]);
            magnetic_volume[i] = magnetic_volume[i] + state.velocities[i] * scale;
        }
        if controls.dedner {
            let uncorrected_fourth = dedner_uncorrected_fourth(
                magnetic_volume[i],
                state.magnetic_volume[i],
                maximum_signal_speed[i],
                particle_size,
            );
            let correction_fourth = dedner_jump[i].squared_norm().powi(2);
            let scale = if correction_fourth > 100.0 * uncorrected_fourth
                && correction_fourth > 0.0
                && uncorrected_fourth > 0.0
            {
                100.0 * uncorrected_fourth / correction_fourth
            } else {
                1.0
            };
            let correction = dedner_jump[i] * scale;
            magnetic_volume[i] = magnetic_volume[i] + correction;
            total_energy[i] += primitive.magnetic[i].dot(correction);
            let phi = primitive.cleaning_scalar[i];
            let clipped_divergence = clip_normalized_magnetic_divergence(
                magnetic_divergence[i],
                primitive.magnetic[i],
                state.smoothing_lengths[i],
            );
            cleaning_mass[i] += state.masses[i]
                * (dedner_hyperbolic_source(
                    clipped_divergence,
                    0.5 * cleaning_speed,
                    controls.hyperbolic_sigma,
                )? + dedner_parabolic_source(
                    phi,
                    2.0 * global_fastest_wave_speed,
                    particle_size,
                    controls.parabolic_sigma,
                )?);
        }
    }
    let mut acceleration = Vec::with_capacity(count);
    let mut specific_internal_energy = Vec::with_capacity(count);
    for i in 0..count {
        let mass = state.masses[i];
        let volume = mass / primitive.density[i];
        let a = momentum[i] / mass;
        let du = (total_energy[i]
            - state.velocities[i].dot(momentum[i])
            - primitive.magnetic[i].dot(magnetic_volume[i])
            + 0.5 * primitive.magnetic[i].squared_norm() * volume * velocity_divergence[i])
            / mass;
        if !a.is_finite() || !du.is_finite() {
            return Err(invalid(Some(i), "primitive_rate", f64::NAN));
        }
        acceleration.push(a);
        specific_internal_energy.push(du);
    }
    Ok(MhdMfmRates2d {
        momentum,
        total_energy,
        magnetic_volume,
        cleaning_mass,
        acceleration,
        specific_internal_energy,
        maximum_signal_speed,
        velocity_divergence,
        magnetic_divergence,
        pair_count: pairs.len(),
    })
}

#[allow(clippy::too_many_arguments)]
fn solve_hlld_with_public_retries_2d(
    reconstructed_left: IdealMhdPrimitive1d,
    reconstructed_right: IdealMhdPrimitive1d,
    centered_left: IdealMhdPrimitive1d,
    centered_right: IdealMhdPrimitive1d,
    normal: Vector2,
    gamma: f64,
    controls: DivergenceControl2d,
    pressure_cap: f64,
) -> Result<(HlldResult2d, RiemannRetry2d), MhdEvolution2dError> {
    let solve = |left, right, pressure_multiplier| {
        hlld_riemann_2d(
            left,
            right,
            [normal.x, normal.y],
            gamma,
            HlldOptions {
                frame: FluxFrame1d::Contact,
                dedner: controls.dedner.then_some(DednerOptions {
                    implicit_limiter: controls.implicit_limiter,
                }),
                maximum_star_total_pressure: Some(pressure_multiplier * pressure_cap),
            },
        )
    };
    match solve(reconstructed_left, reconstructed_right, 1.0) {
        Ok(result) => Ok((result, RiemannRetry2d::Reconstructed)),
        Err(Mhd2dError::Riemann(MhdError::NoAdmissibleContactFlux)) => {
            match solve(centered_left, centered_right, 1.4) {
                Ok(result) => Ok((result, RiemannRetry2d::Centered)),
                Err(Mhd2dError::Riemann(MhdError::NoAdmissibleContactFlux)) => {
                    let quiet_left = IdealMhdPrimitive1d {
                        velocity: Vector3::ZERO,
                        ..centered_left
                    };
                    let quiet_right = IdealMhdPrimitive1d {
                        velocity: Vector3::ZERO,
                        ..centered_right
                    };
                    Ok((
                        solve(quiet_left, quiet_right, 2.0)?,
                        RiemannRetry2d::QuietCentered,
                    ))
                }
                Err(error) => Err(error.into()),
            }
        }
        Err(error) => Err(error.into()),
    }
}

fn dedner_uncorrected_fourth(
    raw_magnetic_rate: Vector3,
    extensive_magnetic: Vector3,
    maximum_signal_speed: f64,
    particle_size: f64,
) -> f64 {
    let regularizer = 0.05 * maximum_signal_speed / particle_size;
    (raw_magnetic_rate.squared_norm() + extensive_magnetic.squared_norm() * regularizer.powi(2))
        .powi(2)
}

fn clip_normalized_magnetic_divergence(
    divergence: f64,
    magnetic: Vector3,
    smoothing_length: f64,
) -> f64 {
    let maximum = 100.0 * magnetic.squared_norm().sqrt() / smoothing_length;
    divergence.clamp(-maximum, maximum)
}

/// Return the global fast-wave CFL bound.
///
/// # Errors
///
/// Returns an error for invalid state, rates, or Courant factor.
pub fn global_mhd_courant_timestep_2d(
    state: &MhdMfmState2d,
    rates: &MhdMfmRates2d,
    courant_factor: f64,
) -> Result<f64, MhdEvolution2dError> {
    state.validate()?;
    if !courant_factor.is_finite() || courant_factor <= 0.0 || courant_factor > 0.5 {
        return Err(invalid(None, "courant_factor", courant_factor));
    }
    validate_lengths(
        state.positions.len(),
        &[("maximum_signal_speed", rates.maximum_signal_speed.len())],
    )?;
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
    Ok(timestep)
}

/// Advance a synchronized KDK step while retaining caller-supplied H.
///
/// # Errors
///
/// Returns an error for invalid inputs or loss of primitive positivity.
pub fn advance_mhd_kdk_2d(
    state: &MhdMfmState2d,
    old_rates: &MhdMfmRates2d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    controls: DivergenceControl2d,
) -> Result<MhdKdkResult2d, MhdEvolution2dError> {
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
    let half_state = recover_state(
        positions,
        state.masses.clone(),
        state.smoothing_lengths.clone(),
        state.domain,
        state.gamma,
        &half,
        minimum_specific_internal_energy,
    )?;
    let rates = mhd_mfm_spatial_rates_2d(&half_state, controls)?;
    let final_conserved = kick_extensive(&half, &rates, 0.5 * timestep);
    let final_state = recover_state(
        half_state.positions,
        half_state.masses,
        half_state.smoothing_lengths,
        half_state.domain,
        half_state.gamma,
        &final_conserved,
        minimum_specific_internal_energy,
    )?;
    Ok(MhdKdkResult2d {
        state: final_state,
        rates,
    })
}

/// Advance a synchronized KDK step and repeat the public-C adaptive-H solve
/// after the drift, before evaluating the new force.
///
/// The old rates must have been evaluated from `state` with its current,
/// already-converged smoothing lengths. This mirrors the density/force order
/// of a synchronized public-C step; it does not emulate hierarchical time bins.
///
/// # Errors
///
/// Returns an error for invalid smoothing controls, a failed adaptive-H solve,
/// invalid inputs, or loss of primitive positivity.
#[allow(clippy::too_many_arguments)]
pub fn advance_mhd_kdk_adaptive_2d(
    state: &MhdMfmState2d,
    old_rates: &MhdMfmRates2d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    controls: DivergenceControl2d,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
) -> Result<MhdKdkResult2d, MhdEvolution2dError> {
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
    let smoothing_lengths = solve_public_c_smoothing_lengths_from_seeds_2d(
        &positions,
        &state.masses,
        &state.smoothing_lengths,
        state.domain,
        desired_neighbors,
        neighbor_tolerance,
    )?
    .into_iter()
    .map(|particle| particle.smoothing_length)
    .collect();
    let half_state = recover_state(
        positions,
        state.masses.clone(),
        smoothing_lengths,
        state.domain,
        state.gamma,
        &half,
        minimum_specific_internal_energy,
    )?;
    let rates = mhd_mfm_spatial_rates_2d(&half_state, controls)?;
    let final_conserved = kick_extensive(&half, &rates, 0.5 * timestep);
    let final_state = recover_state(
        half_state.positions,
        half_state.masses,
        half_state.smoothing_lengths,
        half_state.domain,
        half_state.gamma,
        &final_conserved,
        minimum_specific_internal_energy,
    )?;
    Ok(MhdKdkResult2d {
        state: final_state,
        rates,
    })
}

#[derive(Clone)]
struct ExtensiveColumns {
    momentum: Vec<Vector3>,
    energy: Vec<f64>,
    magnetic: Vec<Vector3>,
    cleaning: Vec<f64>,
}

fn extensive_columns(state: &MhdMfmState2d, primitive: &MhdPrimitiveColumns2d) -> ExtensiveColumns {
    let mut momentum = Vec::with_capacity(state.positions.len());
    let mut energy = Vec::with_capacity(state.positions.len());
    for i in 0..state.positions.len() {
        let volume = state.masses[i] / primitive.density[i];
        momentum.push(state.velocities[i] * state.masses[i]);
        energy.push(
            state.masses[i] * state.specific_internal_energy[i]
                + 0.5 * state.masses[i] * state.velocities[i].squared_norm()
                + 0.5 * volume * primitive.magnetic[i].squared_norm(),
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
    conserved: &ExtensiveColumns,
    rates: &MhdMfmRates2d,
    duration: f64,
) -> ExtensiveColumns {
    ExtensiveColumns {
        momentum: conserved
            .momentum
            .iter()
            .zip(&rates.momentum)
            .map(|(&value, &rate)| value + rate * duration)
            .collect(),
        energy: conserved
            .energy
            .iter()
            .zip(&rates.total_energy)
            .map(|(&value, &rate)| value + rate * duration)
            .collect(),
        magnetic: conserved
            .magnetic
            .iter()
            .zip(&rates.magnetic_volume)
            .map(|(&value, &rate)| value + rate * duration)
            .collect(),
        cleaning: conserved
            .cleaning
            .iter()
            .zip(&rates.cleaning_mass)
            .map(|(&value, &rate)| value + rate * duration)
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn recover_state(
    positions: Vec<Vector2>,
    masses: Vec<f64>,
    smoothing_lengths: Vec<f64>,
    domain: Box2d,
    gamma: f64,
    conserved: &ExtensiveColumns,
    minimum_specific_internal_energy: f64,
) -> Result<MhdMfmState2d, MhdEvolution2dError> {
    let density = density_values(&positions, &masses, &smoothing_lengths, domain)?;
    let mut velocities = Vec::with_capacity(positions.len());
    let mut internal = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        let volume = masses[i] / density[i];
        let velocity = conserved.momentum[i] / masses[i];
        let field = conserved.magnetic[i] / volume;
        let specific = (conserved.energy[i]
            - 0.5 * masses[i] * velocity.squared_norm()
            - 0.5 * volume * field.squared_norm())
            / masses[i];
        if !specific.is_finite() || specific < minimum_specific_internal_energy {
            return Err(invalid(
                Some(i),
                "recovered_specific_internal_energy",
                specific,
            ));
        }
        velocities.push(velocity);
        internal.push(specific);
    }
    let state = MhdMfmState2d {
        positions,
        masses,
        velocities,
        specific_internal_energy: internal,
        smoothing_lengths,
        magnetic_volume: conserved.magnetic.clone(),
        cleaning_mass: conserved.cleaning.clone(),
        domain,
        gamma,
    };
    state.validate()?;
    Ok(state)
}

fn primitive_gradients(
    state: &MhdMfmState2d,
    primitive: &MhdPrimitiveColumns2d,
) -> Result<Gradients2d, MhdEvolution2dError> {
    let velocity = vector_columns(&state.velocities);
    let magnetic = vector_columns(&primitive.magnetic);
    let limiter = gradient_limiter_geometry(state)?;
    Ok(Gradients2d {
        density: gradient(
            state,
            &primitive.density,
            &limiter,
            GradientConstraint2d::Positive,
        )?,
        pressure: gradient(
            state,
            &primitive.pressure,
            &limiter,
            GradientConstraint2d::Positive,
        )?,
        velocity: [
            gradient(state, &velocity[0], &limiter, GradientConstraint2d::Signed)?,
            gradient(state, &velocity[1], &limiter, GradientConstraint2d::Signed)?,
            gradient(state, &velocity[2], &limiter, GradientConstraint2d::Signed)?,
        ],
        magnetic: [
            gradient(state, &magnetic[0], &limiter, GradientConstraint2d::Signed)?,
            gradient(state, &magnetic[1], &limiter, GradientConstraint2d::Signed)?,
            gradient(state, &magnetic[2], &limiter, GradientConstraint2d::Signed)?,
        ],
        cleaning: gradient(
            state,
            &primitive.cleaning_scalar,
            &limiter,
            GradientConstraint2d::Signed,
        )?,
    })
}

fn gradient_limiter_geometry(
    state: &MhdMfmState2d,
) -> Result<GradientLimiterGeometry2d, MhdEvolution2dError> {
    let pairs = interacting_pairs_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    let mut maximum_neighbor_distance = vec![0.0_f64; state.positions.len()];
    for pair in &pairs {
        maximum_neighbor_distance[pair.i] = maximum_neighbor_distance[pair.i].max(pair.distance);
        maximum_neighbor_distance[pair.j] = maximum_neighbor_distance[pair.j].max(pair.distance);
    }
    let condition_number =
        inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain)?
            .into_iter()
            .map(|moment| moment.condition_number)
            .collect();
    Ok(GradientLimiterGeometry2d {
        pairs,
        maximum_neighbor_distance,
        condition_number,
    })
}

fn gradient(
    state: &MhdMfmState2d,
    values: &[f64],
    limiter: &GradientLimiterGeometry2d,
    constraint: GradientConstraint2d,
) -> Result<Vec<Vector2>, MhdEvolution2dError> {
    let mut gradients = scalar_gradients_at_hsml_2d(
        &state.positions,
        values,
        &state.smoothing_lengths,
        state.domain,
    )?;
    let mut minima = vec![0.0_f64; state.positions.len()];
    let mut maxima = vec![0.0_f64; state.positions.len()];
    for pair in &limiter.pairs {
        let delta = values[pair.j] - values[pair.i];
        minima[pair.i] = minima[pair.i].min(delta);
        maxima[pair.i] = maxima[pair.i].max(delta);
        minima[pair.j] = minima[pair.j].min(-delta);
        maxima[pair.j] = maxima[pair.j].max(-delta);
    }
    for i in 0..gradients.len() {
        let condition = limiter.condition_number[i];
        let distance_fraction = if condition > 100.0 {
            (LOCAL_GRADIENT_LIMITER_DISTANCE_FRACTION + 0.25 * (condition - 100.0) / 100.0).min(0.5)
        } else {
            LOCAL_GRADIENT_LIMITER_DISTANCE_FRACTION
        };
        let maximum_distance = state.smoothing_lengths[i].max(limiter.maximum_neighbor_distance[i]);
        gradients[i] = local_slope_limiter(
            gradients[i],
            maxima[i],
            minima[i],
            distance_fraction,
            maximum_distance,
            0.0,
            matches!(constraint, GradientConstraint2d::Positive),
            maximum_distance,
            values[i],
        );
    }
    Ok(gradients)
}

// `hydro/gradients.c::local_slopelimiter`, specialized to two dimensions and
// the default non-cosmological MHD limiter settings.
#[allow(clippy::too_many_arguments)]
fn local_slope_limiter(
    gradient: Vector2,
    maximum_delta: f64,
    minimum_delta: f64,
    distance_fraction: f64,
    limiting_length: f64,
    overshoot_tolerance: f64,
    preserve_positivity: bool,
    maximum_distance: f64,
    central_value: f64,
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
            correction.min((central_value - minimum_value) / (maximum_distance * magnitude));
    }
    if correction < 1.0 {
        gradient * correction
    } else {
        gradient
    }
}

fn reconstruct_pair(
    state: &MhdMfmState2d,
    primitive: &MhdPrimitiveColumns2d,
    gradients: &Gradients2d,
    left: usize,
    right: usize,
    offset_left: Vector2,
    offset_right: Vector2,
) -> (IdealMhdPrimitive1d, IdealMhdPrimitive1d) {
    let reconstruct =
        |value_left, slope_left, value_right, slope_right, mode: FaceLimiterMode2d| {
            reconstruct_face_states(
                value_left,
                slope_left,
                value_right,
                slope_right,
                offset_left,
                offset_right,
                mode,
            )
        };
    let (rho_l, rho_r) = reconstruct(
        primitive.density[left],
        gradients.density[left],
        primitive.density[right],
        gradients.density[right],
        FaceLimiterMode2d::Standard,
    );
    let (p_l, p_r) = reconstruct(
        primitive.pressure[left],
        gradients.pressure[left],
        primitive.pressure[right],
        gradients.pressure[right],
        FaceLimiterMode2d::Standard,
    );
    let vx = reconstruct(
        state.velocities[left].x,
        gradients.velocity[0][left],
        state.velocities[right].x,
        gradients.velocity[0][right],
        FaceLimiterMode2d::Standard,
    );
    let vy = reconstruct(
        state.velocities[left].y,
        gradients.velocity[1][left],
        state.velocities[right].y,
        gradients.velocity[1][right],
        FaceLimiterMode2d::Standard,
    );
    let vz = reconstruct(
        state.velocities[left].z,
        gradients.velocity[2][left],
        state.velocities[right].z,
        gradients.velocity[2][right],
        FaceLimiterMode2d::Standard,
    );
    let bx = reconstruct(
        primitive.magnetic[left].x,
        gradients.magnetic[0][left],
        primitive.magnetic[right].x,
        gradients.magnetic[0][right],
        FaceLimiterMode2d::Standard,
    );
    let by = reconstruct(
        primitive.magnetic[left].y,
        gradients.magnetic[1][left],
        primitive.magnetic[right].y,
        gradients.magnetic[1][right],
        FaceLimiterMode2d::Standard,
    );
    let bz = reconstruct(
        primitive.magnetic[left].z,
        gradients.magnetic[2][left],
        primitive.magnetic[right].z,
        gradients.magnetic[2][right],
        FaceLimiterMode2d::Standard,
    );
    let phi = reconstruct(
        primitive.cleaning_scalar[left],
        gradients.cleaning[left],
        primitive.cleaning_scalar[right],
        gradients.cleaning[right],
        FaceLimiterMode2d::Cleaning,
    );
    (
        IdealMhdPrimitive1d {
            density: rho_l,
            velocity: Vector3::new(vx.0, vy.0, vz.0),
            gas_pressure: p_l,
            magnetic: Vector3::new(bx.0, by.0, bz.0),
            cleaning_scalar: phi.0,
        },
        IdealMhdPrimitive1d {
            density: rho_r,
            velocity: Vector3::new(vx.1, vy.1, vz.1),
            gas_pressure: p_r,
            magnetic: Vector3::new(bx.1, by.1, bz.1),
            cleaning_scalar: phi.1,
        },
    )
}

// `hydro/reimann.h::reconstruct_face_states`, with the public MHD coefficients.
#[allow(clippy::float_cmp, clippy::too_many_arguments)]
fn reconstruct_face_states(
    value_i: f64,
    gradient_i: Vector2,
    value_j: f64,
    gradient_j: Vector2,
    offset_i: Vector2,
    offset_j: Vector2,
    mode: FaceLimiterMode2d,
) -> (f64, f64) {
    if value_i == value_j {
        return (value_i, value_i);
    }
    let mut face_i = value_i + gradient_i.dot(offset_i);
    let mut face_j = value_j + gradient_j.dot(offset_j);
    let (minimum, maximum) = if value_i < value_j {
        (value_i, value_j)
    } else {
        (value_j, value_i)
    };
    let midpoint = 0.5 * (value_i + value_j);
    let difference = maximum - minimum;
    let (minimum_maximum_fraction, midpoint_deviation_fraction) = match mode {
        FaceLimiterMode2d::Standard => (0.5, 0.375),
        FaceLimiterMode2d::Cleaning => (0.0, 0.25),
    };
    let minmax_tolerance = minimum_maximum_fraction * difference;
    let mut effective_maximum = maximum + minmax_tolerance;
    let mut effective_minimum = minimum - minmax_tolerance;
    if maximum < 0.0 && effective_maximum > 0.0 {
        effective_maximum = maximum * maximum / (maximum - (effective_maximum - maximum));
    }
    if minimum > 0.0 && effective_minimum < 0.0 {
        effective_minimum = minimum * minimum / (minimum + (minimum - effective_minimum));
    }
    let midpoint_tolerance = midpoint_deviation_fraction * difference;
    let midpoint_maximum = (midpoint + midpoint_tolerance).min(effective_maximum);
    let midpoint_minimum = (midpoint - midpoint_tolerance).max(effective_minimum);
    if value_i < value_j {
        face_i = face_i.clamp(effective_minimum, midpoint_maximum);
        face_j = face_j.clamp(midpoint_minimum, effective_maximum);
    } else {
        face_i = face_i.clamp(midpoint_minimum, effective_maximum);
        face_j = face_j.clamp(effective_minimum, midpoint_maximum);
    }
    (face_i, face_j)
}

fn velocity_divergence(state: &MhdMfmState2d) -> Result<Vec<f64>, MhdEvolution2dError> {
    let velocity = vector_columns(&state.velocities);
    let grad_x = scalar_gradients_at_hsml_2d(
        &state.positions,
        &velocity[0],
        &state.smoothing_lengths,
        state.domain,
    )?;
    let grad_y = scalar_gradients_at_hsml_2d(
        &state.positions,
        &velocity[1],
        &state.smoothing_lengths,
        state.domain,
    )?;
    Ok(grad_x.iter().zip(&grad_y).map(|(x, y)| x.x + y.y).collect())
}

fn primitive_at(
    state: &MhdMfmState2d,
    primitive: &MhdPrimitiveColumns2d,
    i: usize,
) -> IdealMhdPrimitive1d {
    IdealMhdPrimitive1d {
        density: primitive.density[i],
        velocity: state.velocities[i],
        gas_pressure: primitive.pressure[i],
        magnetic: primitive.magnetic[i],
        cleaning_scalar: primitive.cleaning_scalar[i],
    }
}

fn directional_fast_speed(state: IdealMhdPrimitive1d, direction: Vector2, gamma: f64) -> f64 {
    let sound_squared = gamma * state.gas_pressure / state.density;
    let magnetic_squared_over_density = state.magnetic.squared_norm() / state.density;
    let normal_magnetic = state
        .magnetic
        .x
        .mul_add(direction.x, state.magnetic.y * direction.y);
    let normal_alfven_squared = normal_magnetic * normal_magnetic / state.density;
    let sum = sound_squared + magnetic_squared_over_density;
    let discriminant = (sum * sum - 4.0 * sound_squared * normal_alfven_squared).max(0.0);
    (0.5 * (sum + discriminant.sqrt())).sqrt()
}

fn vector_columns(values: &[Vector3]) -> [Vec<f64>; 3] {
    [
        values.iter().map(|value| value.x).collect(),
        values.iter().map(|value| value.y).collect(),
        values.iter().map(|value| value.z).collect(),
    ]
}

fn density_values(
    positions: &[Vector2],
    masses: &[f64],
    smoothing_lengths: &[f64],
    domain: Box2d,
) -> Result<Vec<f64>, MhdEvolution2dError> {
    Ok(
        density_at_hsml_2d(positions, masses, smoothing_lengths, domain)?
            .into_iter()
            .map(|estimate| estimate.density)
            .collect(),
    )
}

fn validate_controls(controls: DivergenceControl2d) -> Result<(), MhdEvolution2dError> {
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
) -> Result<(), MhdEvolution2dError> {
    for &(field, actual) in columns {
        if actual != expected {
            return Err(MhdEvolution2dError::MismatchedLength {
                field,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn validate_rate_lengths(
    rates: &MhdMfmRates2d,
    expected: usize,
) -> Result<(), MhdEvolution2dError> {
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

fn invalid(index: Option<usize>, field: &'static str, value: f64) -> MhdEvolution2dError {
    MhdEvolution2dError::InvalidState {
        index,
        field,
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::cast_precision_loss)]
    fn sheet(nx: usize, ny: usize, jump: bool) -> MhdMfmState2d {
        let domain = Box2d::new(1.0, 0.25).unwrap();
        let dx = 1.0 / nx as f64;
        let dy = 0.25 / ny as f64;
        let mut positions = Vec::new();
        let mut masses = Vec::new();
        let mut velocities = Vec::new();
        let mut internal = Vec::new();
        let mut magnetic = Vec::new();
        for iy in 0..ny {
            for ix in 0..nx {
                let x = (ix as f64 + 0.5) * dx;
                positions.push(Vector2::new(x, (iy as f64 + 0.5) * dy));
                let left = !jump || x < 0.5;
                let rho = if left { 1.0 } else { 0.125 };
                let pressure = if left { 1.0 } else { 0.1 };
                masses.push(rho * dx * dy);
                velocities.push(Vector3::ZERO);
                internal.push(pressure / rho);
                magnetic.push(Vector3::new(0.75, if left { 1.0 } else { -1.0 }, 0.0));
            }
        }
        MhdMfmState2d::from_primitive(
            positions,
            masses,
            velocities,
            internal,
            vec![3.1 * dx.max(dy); nx * ny],
            &magnetic,
            &vec![0.0; nx * ny],
            domain,
            2.0,
        )
        .unwrap()
    }

    fn no_sources() -> DivergenceControl2d {
        DivergenceControl2d {
            powell: false,
            dedner: false,
            ..Default::default()
        }
    }

    fn sum_vectors(values: &[Vector3]) -> Vector3 {
        values
            .iter()
            .copied()
            .fold(Vector3::ZERO, |sum, value| sum + value)
    }

    fn max_abs(value: Vector3) -> f64 {
        value.x.abs().max(value.y.abs()).max(value.z.abs())
    }

    #[test]
    fn local_limiter_and_face_reconstruction_preserve_a_smooth_linear_field() {
        let mut state = sheet(8, 8, false);
        state.domain = Box2d::new(1.0, 1.0).unwrap();
        for position in &mut state.positions {
            position.y *= 4.0;
        }
        state.smoothing_lengths.fill(0.39);
        let values: Vec<_> = state
            .positions
            .iter()
            .map(|position| 2.0 + 0.7 * position.x - 0.4 * position.y)
            .collect();
        let limiter = gradient_limiter_geometry(&state).unwrap();
        let gradients =
            gradient(&state, &values, &limiter, GradientConstraint2d::Positive).unwrap();
        let i = 3 + 3 * 8;
        let j = i + 1;
        for index in [i, j] {
            assert!((gradients[index].x - 0.7).abs() < 2.0e-14);
            assert!((gradients[index].y + 0.4).abs() < 2.0e-14);
        }
        let half_separation = Vector2::new(1.0 / 16.0, 0.0);
        let (face_i, face_j) = reconstruct_face_states(
            values[i],
            gradients[i],
            values[j],
            gradients[j],
            half_separation,
            -half_separation,
            FaceLimiterMode2d::Standard,
        );
        let exact_midpoint = 0.5 * (values[i] + values[j]);
        assert!((face_i - exact_midpoint).abs() < 2.0e-14);
        assert!((face_j - exact_midpoint).abs() < 2.0e-14);
    }

    #[test]
    fn pair_face_limiters_bound_a_discontinuity() {
        let (standard_i, standard_j) = reconstruct_face_states(
            1.0,
            Vector2::new(20.0, 0.0),
            0.125,
            Vector2::new(20.0, 0.0),
            Vector2::new(0.5, 0.0),
            Vector2::new(-0.5, 0.0),
            FaceLimiterMode2d::Standard,
        );
        assert!((standard_i - 1.4375).abs() < 1.0e-15);
        assert!((standard_j - (1.0 / 36.0)).abs() < 1.0e-15);
        assert!(standard_i > 0.0 && standard_j > 0.0);

        let (cleaning_i, cleaning_j) = reconstruct_face_states(
            1.0,
            Vector2::new(20.0, 0.0),
            -1.0,
            Vector2::new(20.0, 0.0),
            Vector2::new(0.5, 0.0),
            Vector2::new(-0.5, 0.0),
            FaceLimiterMode2d::Cleaning,
        );
        assert!((cleaning_i - 1.0).abs() < 1.0e-15);
        assert!((cleaning_j + 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn riemann_retry_ladder_reaches_centered_and_quiet_states() {
        let brio_left = IdealMhdPrimitive1d {
            density: 1.0,
            velocity: Vector3::ZERO,
            gas_pressure: 1.0,
            magnetic: Vector3::new(0.75, 1.0, 0.0),
            cleaning_scalar: 0.0,
        };
        let brio_right = IdealMhdPrimitive1d {
            density: 0.125,
            velocity: Vector3::ZERO,
            gas_pressure: 0.1,
            magnetic: Vector3::new(0.75, -1.0, 0.0),
            cleaning_scalar: 0.0,
        };
        let hostile_left = IdealMhdPrimitive1d {
            density: 1.0e-5,
            velocity: Vector3::new(-20.0, 4.0, 0.0),
            gas_pressure: 1.0e-8,
            magnetic: Vector3::ZERO,
            cleaning_scalar: 0.0,
        };
        let hostile_right = IdealMhdPrimitive1d {
            density: 20.0,
            velocity: Vector3::new(15.0, -3.0, 1.0),
            gas_pressure: 100.0,
            magnetic: Vector3::ZERO,
            cleaning_scalar: 0.0,
        };
        let diagonal = hlld_riemann_2d(
            brio_left,
            brio_right,
            [1.0, -1.0],
            2.0,
            HlldOptions {
                frame: FluxFrame1d::Contact,
                dedner: Some(DednerOptions::default()),
                maximum_star_total_pressure: Some(6.0),
            },
        )
        .expect("the public oblique discontinuity has a finite contact flux");
        assert_eq!(diagonal.method, MhdRiemannMethod::Hlld);
        assert!(diagonal.flux.mass.abs() < 1.0e-14);

        let (_, centered_path) = solve_hlld_with_public_retries_2d(
            hostile_left,
            hostile_right,
            brio_left,
            brio_right,
            Vector2::new(1.0, 0.0),
            2.0,
            no_sources(),
            1.0e6,
        )
        .unwrap();
        assert_eq!(centered_path, RiemannRetry2d::Centered);

        let (_, quiet_path) = solve_hlld_with_public_retries_2d(
            hostile_left,
            hostile_right,
            hostile_left,
            hostile_right,
            Vector2::new(1.0, 0.0),
            2.0,
            no_sources(),
            1.0e6,
        )
        .unwrap();
        assert_eq!(quiet_path, RiemannRetry2d::QuietCentered);
    }

    #[test]
    fn dedner_guards_use_normalized_divergence_and_extensive_magnetic_units() {
        let field = Vector3::new(2.0, 0.0, 0.0);
        assert!((clip_normalized_magnetic_divergence(1.0e6, field, 0.5) - 400.0).abs() < 1.0e-12);
        assert!((clip_normalized_magnetic_divergence(-1.0e6, field, 0.5) + 400.0).abs() < 1.0e-12);

        let extensive = Vector3::new(0.02, 0.0, 0.0);
        let actual = dedner_uncorrected_fourth(Vector3::ZERO, extensive, 4.0, 0.1);
        let regularized_squared = extensive.squared_norm() * 2.0_f64.powi(2);
        assert!((actual - regularized_squared.powi(2)).abs() < 1.0e-20);
    }

    #[test]
    fn uniform_state_has_zero_induction_and_closed_extensive_rates() {
        let state = sheet(16, 4, false);
        let rates = mhd_mfm_spatial_rates_2d(&state, no_sources()).unwrap();
        assert!(max_abs(sum_vectors(&rates.momentum)) < 1.0e-10);
        assert!(rates.total_energy.iter().sum::<f64>().abs() < 1.0e-10);
        assert!(max_abs(sum_vectors(&rates.magnetic_volume)) < 1.0e-10);
        assert!(
            rates
                .magnetic_volume
                .iter()
                .all(|rate| max_abs(*rate) < 1.0e-10)
        );
    }

    #[test]
    fn unordered_pairs_are_exactly_antisymmetric() {
        let state = sheet(16, 4, true);
        let rates = mhd_mfm_spatial_rates_2d(&state, no_sources()).unwrap();
        assert!(rates.pair_count > 0);
        assert!(max_abs(sum_vectors(&rates.momentum)) < 1.0e-10);
        assert!(rates.total_energy.iter().sum::<f64>().abs() < 1.0e-10);
        assert!(max_abs(sum_vectors(&rates.magnetic_volume)) < 1.0e-10);
    }

    #[test]
    fn courant_bound_uses_two_dimensional_volume_length() {
        let state = sheet(16, 4, true);
        let rates = mhd_mfm_spatial_rates_2d(&state, no_sources()).unwrap();
        let primitive = state.primitive_columns().unwrap();
        let expected = primitive
            .density
            .iter()
            .enumerate()
            .map(|(i, &density)| {
                0.2 * (state.masses[i] / density).sqrt() / (0.5 * rates.maximum_signal_speed[i])
            })
            .fold(f64::INFINITY, f64::min);
        let actual = global_mhd_courant_timestep_2d(&state, &rates, 0.2).unwrap();
        assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected);
    }

    #[test]
    fn adaptive_kdk_resolves_smoothing_lengths_after_drift() {
        let state = sheet(16, 4, false);
        let rates = mhd_mfm_spatial_rates_2d(&state, no_sources()).unwrap();
        let result =
            advance_mhd_kdk_adaptive_2d(&state, &rates, 0.001, 0.0, no_sources(), 20.0, 0.05)
                .unwrap();
        assert!(
            result
                .state
                .smoothing_lengths
                .iter()
                .zip(&state.smoothing_lengths)
                .any(|(adapted, seed)| adapted.to_bits() != seed.to_bits())
        );
        let density = density_at_hsml_2d(
            &result.state.positions,
            &result.state.masses,
            &result.state.smoothing_lengths,
            result.state.domain,
        )
        .unwrap();
        assert!(
            density
                .iter()
                .all(|estimate| (estimate.effective_neighbors - 20.0).abs() <= 0.05)
        );
    }

    #[test]
    fn periodic_y_drift_wraps_and_brio_sheet_evolves() {
        let mut state = sheet(16, 4, true);
        for velocity in &mut state.velocities {
            velocity.y = 0.4;
        }
        let old_x_velocity = state.velocities[7].x;
        let rates = mhd_mfm_spatial_rates_2d(&state, no_sources()).unwrap();
        let result = advance_mhd_kdk_2d(&state, &rates, 0.01, 1.0e-10, no_sources()).unwrap();
        assert!(
            result
                .state
                .positions
                .iter()
                .all(|position| result.state.domain.contains(*position))
        );
        assert!(
            result
                .state
                .velocities
                .iter()
                .all(|velocity| velocity.is_finite())
        );
        assert!(
            result
                .state
                .specific_internal_energy
                .iter()
                .all(|&energy| energy > 0.0)
        );
        assert!((result.state.velocities[7].x - old_x_velocity).abs() > 1.0e-8);
    }

    #[test]
    fn quarter_turn_rotates_planar_rates() {
        let square = Box2d::new(1.0, 1.0).unwrap();
        let mut state = sheet(8, 8, true);
        state.domain = square;
        for position in &mut state.positions {
            position.y *= 4.0;
        }
        state.smoothing_lengths.fill(0.39);
        let base = mhd_mfm_spatial_rates_2d(&state, no_sources()).unwrap();

        let rotate = |vector: Vector3| Vector3::new(-vector.y, vector.x, vector.z);
        let mut rotated = state.clone();
        for position in &mut rotated.positions {
            *position = Vector2::new((-position.y).rem_euclid(1.0), position.x);
        }
        for velocity in &mut rotated.velocities {
            *velocity = rotate(*velocity);
        }
        for magnetic_volume in &mut rotated.magnetic_volume {
            *magnetic_volume = rotate(*magnetic_volume);
        }
        let turned = mhd_mfm_spatial_rates_2d(&rotated, no_sources()).unwrap();
        for i in 0..state.positions.len() {
            assert!(max_abs(turned.momentum[i] - rotate(base.momentum[i])) < 2.0e-10);
            assert!(max_abs(turned.magnetic_volume[i] - rotate(base.magnetic_volume[i])) < 2.0e-10);
            assert!((turned.total_energy[i] - base.total_energy[i]).abs() < 2.0e-10);
        }
    }
}
