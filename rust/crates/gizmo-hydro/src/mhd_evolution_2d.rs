//! Two-dimensional rectangular-periodic meshless finite-mass ideal-MHD evolution.
//!
//! This is a genuine planar operator: particle geometry, gradients, face
//! vectors, drift, and Riemann normals all retain both spatial coordinates.
//! Magnetic flux is stored as the extensive `V B` variable used by the public
//! MFM equations. Both fixed-H and public-C adaptive-H synchronized KDK entry
//! points are available. The hierarchical entry point reproduces the
//! independently verifiable initial per-particle kick and first drift event;
//! active-target endpoint force evolution remains under construction.

use std::error::Error;
use std::fmt;

use crate::individual_timeline::{IndividualParticleTimeline, IndividualTimelineError};
use crate::legacy_float_equal;
use crate::meshless_2d::{
    Box2d, FaceClosure2d, GeometryError, InteractionPair2d, InverseMoment2d, MeshlessFace2d,
    MeshlessPoint2d, Vector2, cubic_kernel_2d, density_at_hsml_2d, face_closure_diagnostics_2d,
    interacting_pairs_2d, inverse_moments_2d, meshless_face_geometry_2d,
    particle_divergence_at_hsml_2d, scalar_gradients_batch_with_moments_2d,
    solve_public_c_smoothing_lengths_from_seeds_2d,
};
use crate::mhd::{
    DednerOptions, FluxFrame1d, HlldOptions, IdealMhdPrimitive1d, MhdError, MhdRiemannMethod,
    Vector3, dedner_hyperbolic_source, fast_magnetosonic_speed,
};
use crate::mhd_2d::{HlldResult2d, Mhd2dError, hlld_riemann_2d};

const LOCAL_GRADIENT_LIMITER_DISTANCE_FRACTION: f64 = 0.25;
const MIN_REAL_NUMBER: f64 = 1.0e-56;
const EPSILON_ENTROPIC_BIG: f64 = 0.5;
const EPSILON_ENTROPIC_SMALL: f64 = 1.0e-3;
// allvars.h raises CONDITION_NUMBER_DANGER to 1e7 for the non-cooling MHD
// build used by Brio-Wu.
const CONDITION_NUMBER_DANGER_SQUARED: f64 = 1.0e14;

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
    pub dhsml_factor: Vec<f64>,
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
    /// Exponential Dedner damping inverse time retained for kick/drift
    /// operator splitting.
    pub cleaning_damping_rate: Vec<f64>,
    pub acceleration: Vec<Vector3>,
    pub specific_internal_energy: Vec<f64>,
    pub maximum_signal_speed: Vec<f64>,
    pub global_fastest_wave_speed: f64,
    pub velocity_divergence: Vec<f64>,
    pub magnetic_divergence: Vec<f64>,
    /// Integrated, clipped `divB` retained by the public force loop and used
    /// to relax the magnetic slope limiter on the next force call.
    pub stored_magnetic_divergence: Vec<f64>,
    pub pair_count: usize,
    pub entropic_pair_count: usize,
}

/// Per-particle physical timestep criteria enabled by the public Brio-Wu
/// configuration, before the maximum-step cap and integer quantization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublicMhdTimestepBounds2d {
    pub acceleration: f64,
    pub courant: f64,
    pub dedner: f64,
    pub velocity_divergence: f64,
    pub selected: f64,
}

/// Initial per-particle time bin selected by public C's `find_timesteps`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublicMhdInitialTimebin2d {
    /// Physical criterion after applying `MaxSizeTimestep`.
    pub bounded_timestep: f64,
    /// Truncated integer request returned by `get_timestep`, before rounding
    /// down to a power of two.
    pub raw_ticks: u64,
    pub ticks: u64,
    pub time_bin: u32,
    pub duration: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MhdRateContext2d<'a> {
    /// Stored integrated divergence from the preceding force evaluation.
    /// `None` is the restart-zero initialization state.
    pub previous_stored_magnetic_divergence: Option<&'a [f64]>,
    /// Physical step used by `Get_DtB_FaceArea_Limiter`.
    pub timestep: Option<f64>,
    /// Courant factor paired with `timestep`.
    pub courant_factor: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MhdKdkResult2d {
    pub state: MhdMfmState2d,
    pub rates: MhdMfmRates2d,
}

/// Mixed actual/predicted state serialized by public C during a drift.
#[derive(Clone, Debug, PartialEq)]
pub struct PublicMhdDriftState2d {
    pub positions: Vec<Vector2>,
    /// Actual velocity after the first half-kick.
    pub actual_velocities: Vec<Vector3>,
    pub predicted_velocities: Vec<Vector3>,
    pub predicted_specific_internal_energy: Vec<f64>,
    pub predicted_density: Vec<f64>,
    pub predicted_smoothing_lengths: Vec<f64>,
    /// Predicted extensive `V B`.
    pub predicted_magnetic_volume: Vec<Vector3>,
    /// Predicted extensive `m phi`.
    pub predicted_cleaning_mass: Vec<f64>,
}

/// Prepared first kick and drift predictor for a public synchronized MHD step.
#[derive(Clone, Debug, PartialEq)]
pub struct PublicMhdKdkStep2d {
    start: MhdMfmState2d,
    old_rates: MhdMfmRates2d,
    half_internal: Vec<f64>,
    half_magnetic: Vec<Vector3>,
    half_cleaning: Vec<f64>,
    drift: PublicMhdDriftState2d,
    elapsed: f64,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    controls: DivergenceControl2d,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
    courant_factor: f64,
}

/// Prepared initial kick and first drift for public C's hierarchical MHD KDK.
///
/// This first event needs no unavailable endpoint oracle: every particle is
/// initially active, but its half-kick uses its own assigned time bin.
#[derive(Clone, Debug, PartialEq)]
pub struct PublicMhdInitialHierarchy2d {
    timeline: IndividualParticleTimeline,
    start: MhdMfmState2d,
    old_rates: MhdMfmRates2d,
    half_internal: Vec<f64>,
    half_magnetic: Vec<Vector3>,
    half_cleaning: Vec<f64>,
    drift: PublicMhdDriftState2d,
    predictor_ticks: Vec<u64>,
    primitive_cache: MhdPrimitiveColumns2d,
    gradient_cache: MhdPrimitiveGradients2d,
    minimum_specific_internal_energy: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublicMhdHierarchySync2d {
    pub tick: u64,
    pub time: f64,
    pub active: Vec<bool>,
    /// Stored mixed-epoch state: arriving active particles are current;
    /// inactive particles remain at their preceding predictor epoch.
    pub drift: PublicMhdDriftState2d,
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
    Timeline(IndividualTimelineError),
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
            Self::Timeline(error) => write!(formatter, "{error}"),
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

impl From<IndividualTimelineError> for MhdEvolution2dError {
    fn from(value: IndividualTimelineError) -> Self {
        Self::Timeline(value)
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
        let density_estimates = density_at_hsml_2d(
            &self.positions,
            &self.masses,
            &self.smoothing_lengths,
            self.domain,
        )?;
        let density: Vec<f64> = density_estimates
            .iter()
            .map(|estimate| estimate.density)
            .collect();
        let dhsml_factor = density_estimates
            .iter()
            .map(|estimate| estimate.dhsml_factor)
            .collect();
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
            dhsml_factor,
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

#[derive(Clone, Debug, PartialEq)]
pub struct MhdPrimitiveGradients2d {
    pub density: Vec<Vector2>,
    pub pressure: Vec<Vector2>,
    pub velocity: [Vec<Vector2>; 3],
    pub magnetic: [Vec<Vector2>; 3],
    pub cleaning: Vec<Vector2>,
}

struct GradientLimiterGeometry2d {
    pairs: Vec<InteractionPair2d>,
    maximum_neighbor_distance: Vec<f64>,
    moments: Vec<InverseMoment2d>,
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
    mhd_mfm_spatial_rates_with_context_2d(state, controls, MhdRateContext2d::default())
}

/// Evaluate the 2-D MFM operator with the force-history and current-step
/// inputs used by the public magnetic limiters.
///
/// # Errors
///
/// Returns an error for invalid state, history, timestep, geometry, or
/// Riemann arithmetic.
#[allow(clippy::too_many_lines)]
pub fn mhd_mfm_spatial_rates_with_context_2d(
    state: &MhdMfmState2d,
    controls: DivergenceControl2d,
    context: MhdRateContext2d<'_>,
) -> Result<MhdMfmRates2d, MhdEvolution2dError> {
    state.validate()?;
    validate_controls(controls)?;
    validate_rate_context(state.positions.len(), context)?;
    let primitive = state.primitive_columns()?;
    let gradients = primitive_gradients(
        state,
        &primitive,
        context.previous_stored_magnetic_divergence,
    )?;
    let planar_velocities: Vec<_> = state
        .velocities
        .iter()
        .map(|velocity| Vector2::new(velocity.x, velocity.y))
        .collect();
    let velocity_divergence = particle_divergence_at_hsml_2d(
        &state.positions,
        &planar_velocities,
        &state.smoothing_lengths,
        &primitive.dhsml_factor,
        state.domain,
    )?;
    let moments = inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    let face_closure = face_closure_diagnostics_2d(
        &state.positions,
        &state.masses,
        &state.smoothing_lengths,
        state.domain,
    )?;
    let count = state.positions.len();
    let mut momentum = vec![Vector3::ZERO; count];
    let mut total_energy = vec![0.0; count];
    let mut magnetic_volume = vec![Vector3::ZERO; count];
    let mut cleaning_mass = vec![0.0; count];
    let mut cleaning_damping_rate = vec![0.0; count];
    let mut magnetic_divergence_volume = vec![0.0; count];
    let mut dedner_jump = vec![Vector3::ZERO; count];
    let mut entropic_pair_count = 0_usize;
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
        let (pair_energy, selected_entropic) = apply_entropic_pdv_energy_2d(
            result.flux.total_energy * face.area,
            result,
            face,
            displacement,
            state,
            &primitive,
            &moments,
            &face_closure,
            pair.i,
            pair.j,
        )?;
        entropic_pair_count += usize::from(selected_entropic);
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
    let magnetic_divergence: Vec<f64> = magnetic_divergence_volume
        .iter()
        .enumerate()
        .map(|(i, &integrated)| integrated / (state.masses[i] / primitive.density[i]))
        .collect();
    let mut stored_magnetic_divergence = magnetic_divergence_volume.clone();
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
        let magnetic_rate_scale = magnetic_face_closure_rate_scale(
            state,
            &primitive,
            &face_closure,
            i,
            magnetic_volume[i],
            context,
        );
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
            let clipped_divergence = clip_normalized_magnetic_divergence(
                magnetic_divergence[i],
                primitive.magnetic[i],
                state.smoothing_lengths[i],
            );
            stored_magnetic_divergence[i] = volume * clipped_divergence;
            cleaning_mass[i] += state.masses[i]
                * dedner_hyperbolic_source(
                    clipped_divergence,
                    0.5 * cleaning_speed,
                    controls.hyperbolic_sigma,
                )?;
            cleaning_damping_rate[i] =
                controls.parabolic_sigma * global_fastest_wave_speed / particle_size;
        }
        if magnetic_rate_scale < 1.0 {
            total_energy[i] +=
                primitive.magnetic[i].dot(magnetic_volume[i] * (magnetic_rate_scale - 1.0));
            magnetic_volume[i] = magnetic_volume[i] * magnetic_rate_scale;
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
        cleaning_damping_rate,
        acceleration,
        specific_internal_energy,
        maximum_signal_speed,
        global_fastest_wave_speed,
        velocity_divergence,
        magnetic_divergence,
        stored_magnetic_divergence,
        pair_count: pairs.len(),
        entropic_pair_count,
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

#[allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn apply_entropic_pdv_energy_2d(
    raw_energy: f64,
    result: HlldResult2d,
    face: MeshlessFace2d,
    displacement: Vector2,
    state: &MhdMfmState2d,
    primitive: &MhdPrimitiveColumns2d,
    moments: &[InverseMoment2d],
    closure: &[FaceClosure2d],
    i: usize,
    j: usize,
) -> Result<(f64, bool), MhdEvolution2dError> {
    let face_velocity_i = state.velocities[i].x.mul_add(
        face.unit_normal.x,
        state.velocities[i].y * face.unit_normal.y,
    );
    let face_velocity_j = state.velocities[j].x.mul_add(
        face.unit_normal.x,
        state.velocities[j].y * face.unit_normal.y,
    );
    let face_velocity = 0.5 * (face_velocity_i + face_velocity_j);
    // The Rust HLLD boundary receives lab-frame states.  The C solver first
    // removes the midpoint frame, so its S_M is this relative contact speed.
    let contact_speed_in_face_frame = result.contact_speed - face_velocity;
    let sound_i = (state.gamma * primitive.pressure[i] / primitive.density[i]).sqrt();
    let sound_j = (state.gamma * primitive.pressure[j] / primitive.density[j]).sqrt();
    let speed_ratio = contact_speed_in_face_frame.abs() / sound_i.min(sound_j);
    let closure_leak =
        0.5 * (closure[i].legacy_dimensionless_leak + closure[j].legacy_dimensionless_leak);
    if speed_ratio >= EPSILON_ENTROPIC_BIG && closure_leak <= 1.0 {
        return Ok((raw_energy, false));
    }

    let star_gas_pressure = result.star_total_pressure - 0.5 * result.face_magnetic.squared_norm();
    if !star_gas_pressure.is_finite() {
        return Err(invalid(Some(i), "star_gas_pressure", star_gas_pressure));
    }
    let distance = displacement.norm();
    let radial = displacement / distance;
    let relative_radial_velocity = (state.velocities[i] - state.velocities[j]).x.mul_add(
        radial.x,
        (state.velocities[i].y - state.velocities[j].y) * radial.y,
    );
    let kernel_i = cubic_kernel_2d(distance, state.smoothing_lengths[i])?;
    let kernel_j = cubic_kernel_2d(distance, state.smoothing_lengths[j])?;
    let volume_i = state.masses[i] / primitive.density[i];
    let volume_j = state.masses[j] / primitive.density[j];
    let pressure_area = star_gas_pressure * face.area;
    let pdv_factor = star_gas_pressure * relative_radial_velocity;
    let pdv_i =
        kernel_i.radial_derivative * volume_i * volume_i * primitive.dhsml_factor[i] * pdv_factor;
    let pdv_j =
        kernel_j.radial_derivative * volume_j * volume_j * primitive.dhsml_factor[j] * pdv_factor;
    let old_energy = pressure_area * (contact_speed_in_face_frame + face_velocity);
    let new_energy = 0.5 * (pdv_i - pdv_j + pressure_area * (face_velocity_i + face_velocity_j));

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

fn magnetic_face_closure_rate_scale(
    state: &MhdMfmState2d,
    primitive: &MhdPrimitiveColumns2d,
    closure: &[FaceClosure2d],
    i: usize,
    pre_dedner_jump_rate: Vector3,
    context: MhdRateContext2d<'_>,
) -> f64 {
    let (Some(timestep), Some(courant_factor)) = (context.timestep, context.courant_factor) else {
        return 1.0;
    };
    let area_sum = closure[i].net_area_vector.x.abs() + closure[i].net_area_vector.y.abs();
    let expected_area = 2.0 * std::f64::consts::PI * state.smoothing_lengths[i];
    if area_sum / expected_area <= 0.001 {
        return 1.0;
    }
    let volume = state.masses[i] / primitive.density[i];
    let magnetic_volume_norm = state.magnetic_volume[i].squared_norm().sqrt();
    let pressure_allowance = (2.0 * primitive.pressure[i]).sqrt() * volume;
    magnetic_face_closure_scale(
        area_sum,
        expected_area,
        magnetic_volume_norm,
        pressure_allowance,
        pre_dedner_jump_rate.squared_norm().sqrt(),
        timestep,
        courant_factor,
    )
}

fn magnetic_face_closure_scale(
    area_sum: f64,
    expected_area: f64,
    magnetic_volume_norm: f64,
    pressure_allowance: f64,
    pre_dedner_jump_rate_norm: f64,
    timestep: f64,
    courant_factor: f64,
) -> f64 {
    if area_sum / expected_area <= 0.001 {
        return 1.0;
    }
    let maximum_magnetic_volume =
        magnetic_volume_norm.max(pressure_allowance.min(10.0 * magnetic_volume_norm));
    let tolerance = (courant_factor / 0.2)
        * 0.01_f64.max(expected_area / (200.0 * area_sum))
        * maximum_magnetic_volume;
    let predicted_change = pre_dedner_jump_rate_norm * timestep;
    if predicted_change > tolerance {
        tolerance / predicted_change
    } else {
        1.0
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

/// Return every per-particle non-cosmological timestep criterion enabled by
/// the public Brio-Wu build, before `MaxSizeTimestep` and integer quantization.
///
/// # Errors
///
/// Returns an error for invalid state, rates, Courant factor, or integration
/// accuracy.
pub fn public_mhd_particle_timestep_bounds_2d(
    state: &MhdMfmState2d,
    rates: &MhdMfmRates2d,
    courant_factor: f64,
    integration_accuracy: f64,
) -> Result<Vec<PublicMhdTimestepBounds2d>, MhdEvolution2dError> {
    state.validate()?;
    validate_rate_lengths(rates, state.positions.len())?;
    if !courant_factor.is_finite() || courant_factor <= 0.0 || courant_factor > 0.5 {
        return Err(invalid(None, "courant_factor", courant_factor));
    }
    if !integration_accuracy.is_finite() || integration_accuracy <= 0.0 {
        return Err(invalid(None, "integration_accuracy", integration_accuracy));
    }
    let primitive = state.primitive_columns()?;
    let mut bounds = Vec::with_capacity(state.positions.len());
    for i in 0..state.positions.len() {
        let acceleration = rates.acceleration[i].squared_norm().sqrt();
        let acceleration = if acceleration == 0.0 {
            1.0e-30
        } else {
            acceleration
        };
        // 2 * ErrTolIntAccuracy * KERNEL_CORE_SIZE with the cubic kernel's
        // KERNEL_CORE_SIZE=1/2 reduces to ErrTolIntAccuracy.
        let acceleration_bound =
            (integration_accuracy * state.smoothing_lengths[i] / acceleration).sqrt();
        // `Get_Particle_Size` uses the literal sqrt(pi) approximation 1.77245
        // and the square root of the 2-D effective neighbor number.
        let effective_neighbor_root =
            (std::f64::consts::PI * state.smoothing_lengths[i].powi(2) * primitive.density[i]
                / state.masses[i])
                .sqrt();
        let particle_size = 1.77245 * state.smoothing_lengths[i] / effective_neighbor_root;
        let courant_bound = courant_factor * particle_size / (0.5 * rates.maximum_signal_speed[i]);
        let sound_squared = state.gamma * primitive.pressure[i] / primitive.density[i];
        let phi_over_signal = primitive.cleaning_scalar[i] / rates.maximum_signal_speed[i];
        let dedner_speed = (sound_squared
            + (primitive.magnetic[i].squared_norm() + phi_over_signal * phi_over_signal)
                / primitive.density[i])
            .sqrt();
        let dedner_bound = 0.8 * courant_factor * particle_size / dedner_speed;
        let divergence_bound = if rates.velocity_divergence[i] == 0.0 {
            f64::INFINITY
        } else {
            1.5 / rates.velocity_divergence[i].abs()
        };
        let candidate = acceleration_bound
            .min(courant_bound)
            .min(dedner_bound)
            .min(divergence_bound);
        for (field, value) in [
            ("acceleration_timestep", acceleration_bound),
            ("courant_timestep", courant_bound),
            ("dedner_timestep", dedner_bound),
            ("velocity_divergence_timestep", divergence_bound),
        ] {
            if value.is_nan() || value <= 0.0 {
                return Err(invalid(Some(i), field, value));
            }
        }
        if !candidate.is_finite() || candidate <= 0.0 {
            return Err(invalid(Some(i), "public_mhd_timestep", candidate));
        }
        bounds.push(PublicMhdTimestepBounds2d {
            acceleration: acceleration_bound,
            courant: courant_bound,
            dedner: dedner_bound,
            velocity_divergence: divergence_bound,
            selected: candidate,
        });
    }
    Ok(bounds)
}

/// Quantize initial public-C particle steps on its default `2^60` timebase.
///
/// At `Ti_Current == 0`, every time bin is synchronized, so
/// `find_timesteps` independently truncates each physical request to integer
/// ticks and rounds it down to a power of two.
///
/// # Errors
///
/// Returns an error for invalid times, timestep bounds, or a request outside
/// the public integer timeline.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn quantize_public_mhd_initial_timebins_2d(
    bounds: &[PublicMhdTimestepBounds2d],
    time_begin: f64,
    time_max: f64,
    maximum_timestep: f64,
) -> Result<Vec<PublicMhdInitialTimebin2d>, MhdEvolution2dError> {
    if !time_begin.is_finite()
        || !time_max.is_finite()
        || time_max <= time_begin
        || !maximum_timestep.is_finite()
        || maximum_timestep <= 0.0
    {
        return Err(invalid(None, "public_mhd_timeline", time_max));
    }
    let tick_duration = (time_max - time_begin) / crate::LEGACY_TIMEBASE_TICKS as f64;
    let mut timebins = Vec::with_capacity(bounds.len());
    for (i, bound) in bounds.iter().enumerate() {
        if !bound.selected.is_finite() || bound.selected <= 0.0 {
            return Err(invalid(Some(i), "public_mhd_timestep", bound.selected));
        }
        let bounded_timestep = bound.selected.min(maximum_timestep);
        let requested_ticks = (bounded_timestep / tick_duration).trunc();
        if !requested_ticks.is_finite() || requested_ticks < 0.0 {
            return Err(invalid(
                Some(i),
                "public_mhd_timestep_ticks",
                requested_ticks,
            ));
        }
        // Unless STOP_WHEN_BELOW_MINTIMESTEP is enabled, public C promotes
        // integer requests 0 and 1 to the smallest legal step of two ticks.
        let raw_ticks = (requested_ticks as u64).max(2);
        if raw_ticks >= crate::LEGACY_TIMEBASE_TICKS {
            return Err(invalid(
                Some(i),
                "public_mhd_timestep_ticks",
                requested_ticks,
            ));
        }
        let time_bin = raw_ticks.ilog2();
        let ticks = 1_u64 << time_bin;
        timebins.push(PublicMhdInitialTimebin2d {
            bounded_timestep,
            raw_ticks,
            ticks,
            time_bin,
            duration: ticks as f64 * tick_duration,
        });
    }
    Ok(timebins)
}

/// Return the minimum non-cosmological timestep bound enabled by the public
/// Brio-Wu build before integer power-of-two quantization.
///
/// This synchronized-runner adapter preserves the previous global interface;
/// public C itself assigns the corresponding bounds to individual particles.
///
/// # Errors
///
/// Returns an error for invalid state, rates, Courant factor, or integration
/// accuracy.
pub fn global_public_mhd_timestep_bound_2d(
    state: &MhdMfmState2d,
    rates: &MhdMfmRates2d,
    courant_factor: f64,
    integration_accuracy: f64,
) -> Result<f64, MhdEvolution2dError> {
    let bounds =
        public_mhd_particle_timestep_bounds_2d(state, rates, courant_factor, integration_accuracy)?;
    bounds
        .iter()
        .map(|bound| bound.selected)
        .reduce(f64::min)
        .ok_or_else(|| invalid(None, "public_mhd_timestep", f64::INFINITY))
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
    advance_mhd_kdk_adaptive_impl_2d(
        state,
        old_rates,
        timestep,
        minimum_specific_internal_energy,
        controls,
        desired_neighbors,
        neighbor_tolerance,
        None,
    )
}

/// Advance the adaptive synchronized KDK path while carrying the public
/// force-history and timestep-dependent magnetic limiters into the endpoint
/// force evaluation.
///
/// # Errors
///
/// Returns the same errors as [`advance_mhd_kdk_adaptive_2d`], plus invalid
/// Courant context.
#[allow(clippy::too_many_arguments)]
pub fn advance_public_mhd_kdk_adaptive_2d(
    state: &MhdMfmState2d,
    old_rates: &MhdMfmRates2d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    controls: DivergenceControl2d,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
    courant_factor: f64,
) -> Result<MhdKdkResult2d, MhdEvolution2dError> {
    let step = begin_public_mhd_kdk_adaptive_2d(
        state,
        old_rates,
        timestep,
        minimum_specific_internal_energy,
        controls,
        desired_neighbors,
        neighbor_tolerance,
        courant_factor,
    )?;
    finish_public_mhd_kdk_adaptive_2d(step)
}

/// Apply the public initial half-kicks using each particle's independently
/// assigned timestep and prepare the mixed drift snapshot state.
///
/// # Errors
///
/// Returns an error for invalid state, rates, time bins, or kick arithmetic.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub fn begin_public_mhd_initial_hierarchy_2d(
    state: &MhdMfmState2d,
    old_rates: &MhdMfmRates2d,
    timebins: &[PublicMhdInitialTimebin2d],
    time_begin: f64,
    time_max: f64,
    minimum_specific_internal_energy: f64,
) -> Result<PublicMhdInitialHierarchy2d, MhdEvolution2dError> {
    state.validate()?;
    validate_rate_lengths(old_rates, state.positions.len())?;
    if timebins.len() != state.positions.len() {
        return Err(MhdEvolution2dError::MismatchedLength {
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
    let steps: Vec<_> = timebins.iter().map(|timebin| timebin.ticks).collect();
    let timeline = IndividualParticleTimeline::from_initial_steps(time_begin, time_max, &steps)?;
    let tick_duration = timeline.duration_for_ticks(1);
    for (i, timebin) in timebins.iter().enumerate() {
        let expected_duration = timeline.duration_for_ticks(timebin.ticks);
        let requested_ticks = (timebin.bounded_timestep / tick_duration).trunc();
        let expected_raw_ticks = if requested_ticks.is_finite() && requested_ticks >= 0.0 {
            (requested_ticks as u64).max(2)
        } else {
            0
        };
        if timebin.time_bin != timebin.ticks.ilog2()
            || !timebin.bounded_timestep.is_finite()
            || timebin.bounded_timestep <= 0.0
            || timebin.raw_ticks != expected_raw_ticks
            || timebin.raw_ticks < timebin.ticks
            || timebin.raw_ticks >= 2 * timebin.ticks
            || !legacy_float_equal(timebin.duration, expected_duration)
        {
            return Err(invalid(Some(i), "initial_timebin", timebin.duration));
        }
    }

    let primitive = state.primitive_columns()?;
    let gradient_cache = primitive_gradients(state, &primitive, None)?;
    let mut primitive_cache = primitive.clone();
    let count = state.positions.len();
    let mut effective_old_rates = old_rates.clone();
    let mut half_velocity = Vec::with_capacity(count);
    let mut half_internal = Vec::with_capacity(count);
    let mut half_magnetic = Vec::with_capacity(count);
    let mut half_cleaning = Vec::with_capacity(count);
    let mut predicted_cleaning = state.cleaning_mass.clone();
    for (i, timebin) in timebins.iter().enumerate() {
        let half_timestep = 0.5 * timeline.duration_for_ticks(timebin.ticks);
        half_velocity.push(state.velocities[i] + old_rates.acceleration[i] * half_timestep);
        half_internal.push(limited_internal_energy_update_2d(
            state.specific_internal_energy[i],
            old_rates.specific_internal_energy[i],
            half_timestep,
            minimum_specific_internal_energy,
        )?);
        half_magnetic.push(state.magnetic_volume[i] + old_rates.magnetic_volume[i] * half_timestep);
        let cleaning_kick = kick_cleaning_mass_public_2d(
            state.cleaning_mass[i],
            state.cleaning_mass[i],
            old_rates.cleaning_mass[i],
            old_rates.cleaning_damping_rate[i],
            half_timestep,
            state.masses[i],
            primitive.density[i],
            primitive.pressure[i],
            primitive.magnetic[i],
            state.gamma,
            old_rates.maximum_signal_speed[i],
            old_rates.global_fastest_wave_speed,
        );
        half_cleaning.push(cleaning_kick.value);
        effective_old_rates.cleaning_mass[i] = cleaning_kick.effective_rate;
        if cleaning_kick.reset_predicted {
            predicted_cleaning[i] = 0.0;
        }
    }
    for (i, &predicted) in predicted_cleaning.iter().enumerate() {
        primitive_cache.cleaning_scalar[i] = predicted / state.masses[i];
    }
    let drift = PublicMhdDriftState2d {
        positions: state.positions.clone(),
        actual_velocities: half_velocity,
        predicted_velocities: state.velocities.clone(),
        predicted_specific_internal_energy: state.specific_internal_energy.clone(),
        predicted_density: primitive.density,
        predicted_smoothing_lengths: state.smoothing_lengths.clone(),
        predicted_magnetic_volume: state.magnetic_volume.clone(),
        predicted_cleaning_mass: predicted_cleaning,
    };
    Ok(PublicMhdInitialHierarchy2d {
        timeline,
        start: state.clone(),
        old_rates: effective_old_rates,
        half_internal,
        half_magnetic,
        half_cleaning,
        drift,
        predictor_ticks: vec![0; count],
        primitive_cache,
        gradient_cache,
        minimum_specific_internal_energy,
    })
}

impl PublicMhdInitialHierarchy2d {
    /// Drift active particles to the earliest occupied bin.
    ///
    /// Inactive columns retain their preceding predictor epoch until
    /// [`Self::drift_neighbor_to_current`] is called by a neighbor traversal.
    ///
    /// # Errors
    ///
    /// Returns an error after the first event or for invalid predictor math.
    pub fn drift_to_first_sync(&mut self) -> Result<PublicMhdHierarchySync2d, MhdEvolution2dError> {
        if self.timeline.current_tick() != 0 {
            return Err(invalid(
                None,
                "initial_hierarchy_tick",
                self.timeline.current_time(),
            ));
        }
        let active = self.timeline.advance_to_next_sync()?;
        for (i, &is_active) in active.iter().enumerate() {
            if is_active {
                self.drift_particle_to_current(i)?;
            }
        }
        Ok(PublicMhdHierarchySync2d {
            tick: self.timeline.current_tick(),
            time: self.timeline.current_time(),
            active,
            drift: self.drift.clone(),
        })
    }

    /// Lazily bring an encountered inactive neighbor to the current sync.
    ///
    /// The update is performed once from that particle's own retained epoch,
    /// preserving the non-semigroup density and internal-energy limiters.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid index or predictor arithmetic.
    pub fn drift_neighbor_to_current(&mut self, index: usize) -> Result<(), MhdEvolution2dError> {
        self.drift_particle_to_current(index)
    }

    #[must_use]
    pub fn predictor_ticks(&self) -> &[u64] {
        &self.predictor_ticks
    }

    #[must_use]
    pub fn predicted_primitive_cache(&self) -> &MhdPrimitiveColumns2d {
        &self.primitive_cache
    }

    #[must_use]
    pub fn retained_gradient_cache(&self) -> &MhdPrimitiveGradients2d {
        &self.gradient_cache
    }

    #[must_use]
    pub fn actual_specific_internal_energy(&self) -> &[f64] {
        &self.half_internal
    }

    #[must_use]
    pub fn actual_magnetic_volume(&self) -> &[Vector3] {
        &self.half_magnetic
    }

    #[must_use]
    pub fn actual_cleaning_mass(&self) -> &[f64] {
        &self.half_cleaning
    }

    fn drift_particle_to_current(&mut self, i: usize) -> Result<(), MhdEvolution2dError> {
        if i >= self.start.positions.len() {
            return Err(invalid(Some(i), "hierarchy_particle_index", f64::NAN));
        }
        let target_tick = self.timeline.current_tick();
        let source_tick = self.predictor_ticks[i];
        if source_tick > target_tick {
            return Err(invalid(
                Some(i),
                "hierarchy_predictor_tick",
                self.timeline.current_time(),
            ));
        }
        let segment = self.timeline.duration_for_ticks(target_tick - source_tick);
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
        let divergence_increment =
            (self.old_rates.velocity_divergence[i] * segment).clamp(-0.3, 0.3);
        self.drift.predicted_density[i] *= (-divergence_increment).exp();
        self.drift.predicted_smoothing_lengths[i] *= (0.5 * divergence_increment).exp();
        self.drift.predicted_magnetic_volume[i] =
            self.drift.predicted_magnetic_volume[i] + self.old_rates.magnetic_volume[i] * segment;
        self.drift.predicted_cleaning_mass[i] = predict_cleaning_mass_2d(
            self.drift.predicted_cleaning_mass[i],
            self.old_rates.cleaning_mass[i],
            self.old_rates.cleaning_damping_rate[i],
            segment,
        );
        self.primitive_cache.density[i] = self.drift.predicted_density[i];
        self.primitive_cache.pressure[i] = (self.start.gamma - 1.0)
            * self.drift.predicted_density[i]
            * self.drift.predicted_specific_internal_energy[i];
        let volume = self.start.masses[i] / self.drift.predicted_density[i];
        self.primitive_cache.magnetic[i] = self.drift.predicted_magnetic_volume[i] / volume;
        self.primitive_cache.cleaning_scalar[i] =
            self.drift.predicted_cleaning_mass[i] / self.start.masses[i];
        self.predictor_ticks[i] = target_tick;
        Ok(())
    }
}

impl PublicMhdKdkStep2d {
    /// Advance the predictor monotonically to an elapsed time in this step.
    ///
    /// The returned fields match the mixed actual/predicted view used by the
    /// public snapshot writer before endpoint density and force evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error for a backward/out-of-step cursor or invalid physics.
    pub fn drift_state(
        &mut self,
        elapsed: f64,
    ) -> Result<PublicMhdDriftState2d, MhdEvolution2dError> {
        if !elapsed.is_finite() || elapsed < self.elapsed || elapsed > self.timestep {
            return Err(invalid(None, "kdk_drift_elapsed", elapsed));
        }
        let segment = elapsed - self.elapsed;
        if segment == 0.0 {
            return Ok(self.drift.clone());
        }
        for i in 0..self.start.positions.len() {
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
            let divergence_increment =
                (self.old_rates.velocity_divergence[i] * segment).clamp(-0.3, 0.3);
            self.drift.predicted_density[i] *= (-divergence_increment).exp();
            self.drift.predicted_smoothing_lengths[i] *= (0.5 * divergence_increment).exp();
            self.drift.predicted_magnetic_volume[i] = self.drift.predicted_magnetic_volume[i]
                + self.old_rates.magnetic_volume[i] * segment;
            self.drift.predicted_cleaning_mass[i] = predict_cleaning_mass_2d(
                self.drift.predicted_cleaning_mass[i],
                self.old_rates.cleaning_mass[i],
                self.old_rates.cleaning_damping_rate[i],
                segment,
            );
        }
        self.elapsed = elapsed;
        Ok(self.drift.clone())
    }

    #[must_use]
    pub fn timestep(&self) -> f64 {
        self.timestep
    }

    #[must_use]
    pub fn elapsed(&self) -> f64 {
        self.elapsed
    }
}

/// Apply the public first half-kick and expose the drift-phase predictor.
///
/// # Errors
///
/// Returns an error for invalid state, rates, integration inputs, or kick
/// arithmetic.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn begin_public_mhd_kdk_adaptive_2d(
    state: &MhdMfmState2d,
    old_rates: &MhdMfmRates2d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    controls: DivergenceControl2d,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
    courant_factor: f64,
) -> Result<PublicMhdKdkStep2d, MhdEvolution2dError> {
    state.validate()?;
    validate_rate_lengths(old_rates, state.positions.len())?;
    validate_rate_context(
        state.positions.len(),
        MhdRateContext2d {
            previous_stored_magnetic_divergence: Some(&old_rates.stored_magnetic_divergence),
            timestep: Some(timestep),
            courant_factor: Some(courant_factor),
        },
    )?;
    if !minimum_specific_internal_energy.is_finite() || minimum_specific_internal_energy < 0.0 {
        return Err(invalid(
            None,
            "minimum_specific_internal_energy",
            minimum_specific_internal_energy,
        ));
    }
    let primitive = state.primitive_columns()?;
    let half_timestep = 0.5 * timestep;
    let count = state.positions.len();
    let mut effective_old_rates = old_rates.clone();
    let mut half_velocity = Vec::with_capacity(count);
    let mut half_internal = Vec::with_capacity(count);
    let mut half_magnetic = Vec::with_capacity(count);
    let mut half_cleaning = Vec::with_capacity(count);
    for i in 0..count {
        half_velocity.push(state.velocities[i] + old_rates.acceleration[i] * half_timestep);
        half_internal.push(limited_internal_energy_update_2d(
            state.specific_internal_energy[i],
            old_rates.specific_internal_energy[i],
            half_timestep,
            minimum_specific_internal_energy,
        )?);
        half_magnetic.push(state.magnetic_volume[i] + old_rates.magnetic_volume[i] * half_timestep);
        let cleaning_kick = kick_cleaning_mass_public_2d(
            state.cleaning_mass[i],
            state.cleaning_mass[i],
            old_rates.cleaning_mass[i],
            old_rates.cleaning_damping_rate[i],
            half_timestep,
            state.masses[i],
            primitive.density[i],
            primitive.pressure[i],
            primitive.magnetic[i],
            state.gamma,
            old_rates.maximum_signal_speed[i],
            old_rates.global_fastest_wave_speed,
        );
        half_cleaning.push(cleaning_kick.value);
        effective_old_rates.cleaning_mass[i] = cleaning_kick.effective_rate;
    }
    let drift = PublicMhdDriftState2d {
        positions: state.positions.clone(),
        actual_velocities: half_velocity,
        predicted_velocities: state.velocities.clone(),
        predicted_specific_internal_energy: state.specific_internal_energy.clone(),
        predicted_density: primitive.density,
        predicted_smoothing_lengths: state.smoothing_lengths.clone(),
        predicted_magnetic_volume: state.magnetic_volume.clone(),
        predicted_cleaning_mass: state.cleaning_mass.clone(),
    };
    Ok(PublicMhdKdkStep2d {
        start: state.clone(),
        old_rates: effective_old_rates,
        half_internal,
        half_magnetic,
        half_cleaning,
        drift,
        elapsed: 0.0,
        timestep,
        minimum_specific_internal_energy,
        controls,
        desired_neighbors,
        neighbor_tolerance,
        courant_factor,
    })
}

/// Finish endpoint density/force evaluation and the second public half-kick.
///
/// # Errors
///
/// Returns an error for invalid endpoint density, force, or kick arithmetic.
#[allow(clippy::too_many_lines)]
pub fn finish_public_mhd_kdk_adaptive_2d(
    mut step: PublicMhdKdkStep2d,
) -> Result<MhdKdkResult2d, MhdEvolution2dError> {
    let endpoint = step.drift_state(step.timestep)?;
    let smoothing_lengths: Vec<f64> = solve_public_c_smoothing_lengths_from_seeds_2d(
        &endpoint.positions,
        &step.start.masses,
        &endpoint.predicted_smoothing_lengths,
        step.start.domain,
        step.desired_neighbors,
        step.neighbor_tolerance,
    )?
    .into_iter()
    .map(|particle| particle.smoothing_length)
    .collect();
    let predicted_state = MhdMfmState2d {
        positions: endpoint.positions.clone(),
        masses: step.start.masses.clone(),
        velocities: endpoint.predicted_velocities,
        specific_internal_energy: endpoint.predicted_specific_internal_energy,
        smoothing_lengths: smoothing_lengths.clone(),
        magnetic_volume: endpoint.predicted_magnetic_volume,
        cleaning_mass: endpoint.predicted_cleaning_mass,
        domain: step.start.domain,
        gamma: step.start.gamma,
    };
    predicted_state.validate()?;
    let mut rates = mhd_mfm_spatial_rates_with_context_2d(
        &predicted_state,
        step.controls,
        MhdRateContext2d {
            previous_stored_magnetic_divergence: Some(&step.old_rates.stored_magnetic_divergence),
            timestep: Some(step.timestep),
            courant_factor: Some(step.courant_factor),
        },
    )?;
    let predicted_primitive = predicted_state.primitive_columns()?;
    let half_timestep = 0.5 * step.timestep;
    let count = step.start.positions.len();
    let mut final_velocity = Vec::with_capacity(count);
    let mut final_internal = Vec::with_capacity(count);
    let mut final_magnetic = Vec::with_capacity(count);
    let mut final_cleaning = Vec::with_capacity(count);
    for i in 0..count {
        final_velocity.push(endpoint.actual_velocities[i] + rates.acceleration[i] * half_timestep);
        final_internal.push(limited_internal_energy_update_2d(
            step.half_internal[i],
            rates.specific_internal_energy[i],
            half_timestep,
            step.minimum_specific_internal_energy,
        )?);
        final_magnetic.push(step.half_magnetic[i] + rates.magnetic_volume[i] * half_timestep);
        let particle_size = (step.start.masses[i] / predicted_primitive.density[i]).sqrt();
        let damping_rate = step.controls.parabolic_sigma * step.old_rates.global_fastest_wave_speed
            / particle_size;
        let cleaning_kick = kick_cleaning_mass_public_2d(
            step.half_cleaning[i],
            predicted_state.cleaning_mass[i],
            rates.cleaning_mass[i],
            damping_rate,
            half_timestep,
            step.start.masses[i],
            predicted_primitive.density[i],
            predicted_primitive.pressure[i],
            predicted_primitive.magnetic[i],
            step.start.gamma,
            rates.maximum_signal_speed[i],
            step.old_rates.global_fastest_wave_speed,
        );
        final_cleaning.push(cleaning_kick.value);
        rates.cleaning_mass[i] = cleaning_kick.effective_rate;
        rates.cleaning_damping_rate[i] = damping_rate;
    }
    let state = MhdMfmState2d {
        positions: endpoint.positions,
        masses: step.start.masses,
        velocities: final_velocity,
        specific_internal_energy: final_internal,
        smoothing_lengths,
        magnetic_volume: final_magnetic,
        cleaning_mass: final_cleaning,
        domain: step.start.domain,
        gamma: step.start.gamma,
    };
    state.validate()?;
    Ok(MhdKdkResult2d { state, rates })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn advance_public_mhd_kdk_legacy_reference_2d(
    state: &MhdMfmState2d,
    old_rates: &MhdMfmRates2d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    controls: DivergenceControl2d,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
    courant_factor: f64,
) -> Result<MhdKdkResult2d, MhdEvolution2dError> {
    state.validate()?;
    validate_rate_lengths(old_rates, state.positions.len())?;
    validate_rate_context(
        state.positions.len(),
        MhdRateContext2d {
            previous_stored_magnetic_divergence: Some(&old_rates.stored_magnetic_divergence),
            timestep: Some(timestep),
            courant_factor: Some(courant_factor),
        },
    )?;
    if !minimum_specific_internal_energy.is_finite() || minimum_specific_internal_energy < 0.0 {
        return Err(invalid(
            None,
            "minimum_specific_internal_energy",
            minimum_specific_internal_energy,
        ));
    }
    let primitive = state.primitive_columns()?;
    let half_timestep = 0.5 * timestep;
    let count = state.positions.len();
    let mut half_velocity = Vec::with_capacity(count);
    let mut half_internal = Vec::with_capacity(count);
    let mut half_magnetic = Vec::with_capacity(count);
    let mut half_cleaning = Vec::with_capacity(count);
    let mut predicted_velocity = Vec::with_capacity(count);
    let mut predicted_internal = Vec::with_capacity(count);
    let mut predicted_magnetic = Vec::with_capacity(count);
    let mut predicted_cleaning = Vec::with_capacity(count);
    let mut predicted_hsml = Vec::with_capacity(count);
    for i in 0..count {
        half_velocity.push(state.velocities[i] + old_rates.acceleration[i] * half_timestep);
        half_internal.push(limited_internal_energy_update_2d(
            state.specific_internal_energy[i],
            old_rates.specific_internal_energy[i],
            half_timestep,
            minimum_specific_internal_energy,
        )?);
        half_magnetic.push(state.magnetic_volume[i] + old_rates.magnetic_volume[i] * half_timestep);
        let cleaning_kick = kick_cleaning_mass_public_2d(
            state.cleaning_mass[i],
            state.cleaning_mass[i],
            old_rates.cleaning_mass[i],
            old_rates.cleaning_damping_rate[i],
            half_timestep,
            state.masses[i],
            primitive.density[i],
            primitive.pressure[i],
            primitive.magnetic[i],
            state.gamma,
            old_rates.maximum_signal_speed[i],
            old_rates.global_fastest_wave_speed,
        );
        half_cleaning.push(cleaning_kick.value);
        predicted_velocity.push(state.velocities[i] + old_rates.acceleration[i] * timestep);
        predicted_internal.push(limited_internal_energy_update_2d(
            state.specific_internal_energy[i],
            old_rates.specific_internal_energy[i],
            timestep,
            minimum_specific_internal_energy,
        )?);
        predicted_magnetic.push(state.magnetic_volume[i] + old_rates.magnetic_volume[i] * timestep);
        predicted_cleaning.push(predict_cleaning_mass_2d(
            if cleaning_kick.reset_predicted {
                0.0
            } else {
                state.cleaning_mass[i]
            },
            cleaning_kick.effective_rate,
            old_rates.cleaning_damping_rate[i],
            timestep,
        ));
        let divergence_increment = (old_rates.velocity_divergence[i] * timestep).clamp(-0.3, 0.3);
        predicted_hsml.push(state.smoothing_lengths[i] * (0.5 * divergence_increment).exp());
    }
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
    let smoothing_lengths: Vec<f64> = solve_public_c_smoothing_lengths_from_seeds_2d(
        &positions,
        &state.masses,
        &predicted_hsml,
        state.domain,
        desired_neighbors,
        neighbor_tolerance,
    )?
    .into_iter()
    .map(|particle| particle.smoothing_length)
    .collect();
    let predicted_state = MhdMfmState2d {
        positions: positions.clone(),
        masses: state.masses.clone(),
        velocities: predicted_velocity,
        specific_internal_energy: predicted_internal,
        smoothing_lengths: smoothing_lengths.clone(),
        magnetic_volume: predicted_magnetic,
        cleaning_mass: predicted_cleaning,
        domain: state.domain,
        gamma: state.gamma,
    };
    predicted_state.validate()?;
    let mut rates = mhd_mfm_spatial_rates_with_context_2d(
        &predicted_state,
        controls,
        MhdRateContext2d {
            previous_stored_magnetic_divergence: Some(&old_rates.stored_magnetic_divergence),
            timestep: Some(timestep),
            courant_factor: Some(courant_factor),
        },
    )?;
    let predicted_primitive = predicted_state.primitive_columns()?;
    let mut final_velocity = Vec::with_capacity(count);
    let mut final_internal = Vec::with_capacity(count);
    let mut final_magnetic = Vec::with_capacity(count);
    let mut final_cleaning = Vec::with_capacity(count);
    for i in 0..count {
        final_velocity.push(half_velocity[i] + rates.acceleration[i] * half_timestep);
        final_internal.push(limited_internal_energy_update_2d(
            half_internal[i],
            rates.specific_internal_energy[i],
            half_timestep,
            minimum_specific_internal_energy,
        )?);
        final_magnetic.push(half_magnetic[i] + rates.magnetic_volume[i] * half_timestep);
        let endpoint_particle_size = (state.masses[i] / predicted_primitive.density[i]).sqrt();
        let endpoint_damping_rate =
            controls.parabolic_sigma * old_rates.global_fastest_wave_speed / endpoint_particle_size;
        let cleaning_kick = kick_cleaning_mass_public_2d(
            half_cleaning[i],
            predicted_state.cleaning_mass[i],
            rates.cleaning_mass[i],
            endpoint_damping_rate,
            half_timestep,
            state.masses[i],
            predicted_primitive.density[i],
            predicted_primitive.pressure[i],
            predicted_primitive.magnetic[i],
            state.gamma,
            rates.maximum_signal_speed[i],
            old_rates.global_fastest_wave_speed,
        );
        final_cleaning.push(cleaning_kick.value);
        rates.cleaning_mass[i] = cleaning_kick.effective_rate;
        rates.cleaning_damping_rate[i] = endpoint_damping_rate;
    }
    let final_state = MhdMfmState2d {
        positions,
        masses: state.masses.clone(),
        velocities: final_velocity,
        specific_internal_energy: final_internal,
        smoothing_lengths,
        magnetic_volume: final_magnetic,
        cleaning_mass: final_cleaning,
        domain: state.domain,
        gamma: state.gamma,
    };
    final_state.validate()?;
    Ok(MhdKdkResult2d {
        state: final_state,
        rates,
    })
}

#[allow(clippy::too_many_arguments)]
fn advance_mhd_kdk_adaptive_impl_2d(
    state: &MhdMfmState2d,
    old_rates: &MhdMfmRates2d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    controls: DivergenceControl2d,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
    courant_factor: Option<f64>,
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
    let rates = mhd_mfm_spatial_rates_with_context_2d(
        &half_state,
        controls,
        MhdRateContext2d {
            previous_stored_magnetic_divergence: Some(&old_rates.stored_magnetic_divergence),
            timestep: courant_factor.map(|_| timestep),
            courant_factor,
        },
    )?;
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

fn limited_internal_energy_update_2d(
    previous: f64,
    rate: f64,
    timestep: f64,
    floor: f64,
) -> Result<f64, MhdEvolution2dError> {
    let candidate = previous + timestep * rate;
    if !previous.is_finite()
        || previous <= 0.0
        || !rate.is_finite()
        || !timestep.is_finite()
        || timestep < 0.0
        || !floor.is_finite()
        || floor < 0.0
        || !candidate.is_finite()
    {
        return Err(invalid(None, "internal_energy_update", candidate));
    }
    Ok(if candidate < 0.5 * previous {
        0.5 * previous
    } else {
        candidate
    }
    .max(floor))
}

fn predict_cleaning_mass_2d(current: f64, rate: f64, damping_rate: f64, timestep: f64) -> f64 {
    (current + timestep * rate) * (-timestep * damping_rate).exp()
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CleaningKick2d {
    value: f64,
    effective_rate: f64,
    reset_predicted: bool,
}

#[allow(clippy::too_many_arguments)]
fn kick_cleaning_mass_public_2d(
    current: f64,
    predicted_for_guard: f64,
    rate: f64,
    damping_rate: f64,
    timestep: f64,
    mass: f64,
    density: f64,
    pressure: f64,
    magnetic: Vector3,
    gamma: f64,
    maximum_signal_speed: f64,
    global_fastest_wave_speed: f64,
) -> CleaningKick2d {
    let phi_abs = (predicted_for_guard / mass).abs();
    let magnetic_norm = magnetic.squared_norm().sqrt();
    let local_wave_speed = (gamma * pressure / density + magnetic.squared_norm() / density).sqrt();
    let fastest = local_wave_speed
        .max(0.5 * maximum_signal_speed.abs())
        .max(global_fastest_wave_speed);
    let magnetic_wave_scale = fastest * magnetic_norm;
    let mut updated = current;
    let mut effective_rate = rate;
    let mut reset_predicted = false;
    if phi_abs > 0.0 && magnetic_wave_scale > 0.0 {
        if phi_abs > 10_000.0 * magnetic_wave_scale {
            updated = 0.0;
            effective_rate = 0.0;
            reset_predicted = true;
        } else if phi_abs > 10.0 * magnetic_wave_scale {
            effective_rate = if current > 0.0 {
                rate.min(0.0)
            } else {
                rate.max(0.0)
            };
            if current != 0.0 {
                updated *= (-((timestep * effective_rate).abs() / current.abs())).exp();
            }
        } else {
            updated += timestep * rate;
        }
    }
    CleaningKick2d {
        value: updated * (-timestep * damping_rate).exp(),
        effective_rate,
        reset_predicted,
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

#[allow(clippy::too_many_lines)]
fn primitive_gradients(
    state: &MhdMfmState2d,
    primitive: &MhdPrimitiveColumns2d,
    previous_stored_magnetic_divergence: Option<&[f64]>,
) -> Result<MhdPrimitiveGradients2d, MhdEvolution2dError> {
    let velocity = vector_columns(&state.velocities);
    let magnetic = vector_columns(&primitive.magnetic);
    let limiter = gradient_limiter_geometry(state)?;
    let fields: [&[f64]; 9] = [
        &primitive.density,
        &primitive.pressure,
        &velocity[0],
        &velocity[1],
        &velocity[2],
        &magnetic[0],
        &magnetic[1],
        &magnetic[2],
        &primitive.cleaning_scalar,
    ];
    let mut raw = scalar_gradients_batch_with_moments_2d(
        &state.positions,
        &fields,
        &state.smoothing_lengths,
        state.domain,
        &limiter.moments,
    )?;
    let magnetic_distance_fraction: Vec<f64> = (0..state.positions.len())
        .map(|i| {
            let base = base_gradient_limiter_fraction(limiter.moments[i].condition_number);
            let integrated_divergence =
                previous_stored_magnetic_divergence.map_or(0.0, |values| values[i]);
            let volume = state.masses[i] / primitive.density[i];
            let normalization = (1.0e-37
                + 2.0 * primitive.pressure[i] * volume * volume
                + state.magnetic_volume[i].squared_norm())
            .sqrt();
            let q = integrated_divergence.abs() * state.smoothing_lengths[i] / normalization;
            relaxed_magnetic_gradient_fraction(base, q)
        })
        .collect();
    let cleaning = raw.pop().expect("nine gradient fields");
    let magnetic_z = raw.pop().expect("nine gradient fields");
    let magnetic_y = raw.pop().expect("nine gradient fields");
    let magnetic_x = raw.pop().expect("nine gradient fields");
    let velocity_z = raw.pop().expect("nine gradient fields");
    let velocity_y = raw.pop().expect("nine gradient fields");
    let velocity_x = raw.pop().expect("nine gradient fields");
    let pressure = raw.pop().expect("nine gradient fields");
    let density = raw.pop().expect("nine gradient fields");
    let gradients = MhdPrimitiveGradients2d {
        density: limit_precomputed_gradient(
            state,
            &primitive.density,
            &limiter,
            GradientConstraint2d::Positive,
            density,
            None,
        ),
        pressure: limit_precomputed_gradient(
            state,
            &primitive.pressure,
            &limiter,
            GradientConstraint2d::Positive,
            pressure,
            None,
        ),
        velocity: [
            limit_precomputed_gradient(
                state,
                &velocity[0],
                &limiter,
                GradientConstraint2d::Signed,
                velocity_x,
                None,
            ),
            limit_precomputed_gradient(
                state,
                &velocity[1],
                &limiter,
                GradientConstraint2d::Signed,
                velocity_y,
                None,
            ),
            limit_precomputed_gradient(
                state,
                &velocity[2],
                &limiter,
                GradientConstraint2d::Signed,
                velocity_z,
                None,
            ),
        ],
        magnetic: [
            limit_precomputed_gradient(
                state,
                &magnetic[0],
                &limiter,
                GradientConstraint2d::Signed,
                magnetic_x,
                Some(&magnetic_distance_fraction),
            ),
            limit_precomputed_gradient(
                state,
                &magnetic[1],
                &limiter,
                GradientConstraint2d::Signed,
                magnetic_y,
                Some(&magnetic_distance_fraction),
            ),
            limit_precomputed_gradient(
                state,
                &magnetic[2],
                &limiter,
                GradientConstraint2d::Signed,
                magnetic_z,
                Some(&magnetic_distance_fraction),
            ),
        ],
        cleaning: limit_precomputed_gradient(
            state,
            &primitive.cleaning_scalar,
            &limiter,
            GradientConstraint2d::Signed,
            cleaning,
            None,
        ),
    };
    Ok(gradients)
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
    let moments = inverse_moments_2d(&state.positions, &state.smoothing_lengths, state.domain)?;
    Ok(GradientLimiterGeometry2d {
        pairs,
        maximum_neighbor_distance,
        moments,
    })
}

fn limit_precomputed_gradient(
    state: &MhdMfmState2d,
    values: &[f64],
    limiter: &GradientLimiterGeometry2d,
    constraint: GradientConstraint2d,
    mut gradients: Vec<Vector2>,
    distance_fractions: Option<&[f64]>,
) -> Vec<Vector2> {
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
        let distance_fraction = distance_fractions.map_or_else(
            || base_gradient_limiter_fraction(limiter.moments[i].condition_number),
            |values| values[i],
        );
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
    gradients
}

fn base_gradient_limiter_fraction(condition_number: f64) -> f64 {
    if condition_number > 100.0 {
        (LOCAL_GRADIENT_LIMITER_DISTANCE_FRACTION + 0.25 * (condition_number - 100.0) / 100.0)
            .min(0.5)
    } else {
        LOCAL_GRADIENT_LIMITER_DISTANCE_FRACTION
    }
}

fn relaxed_magnetic_gradient_fraction(base: f64, normalized_divergence: f64) -> f64 {
    (base * normalized_divergence.mul_add(normalized_divergence, 1.0)).min(0.5)
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
    gradients: &MhdPrimitiveGradients2d,
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

fn validate_rate_context(
    particle_count: usize,
    context: MhdRateContext2d<'_>,
) -> Result<(), MhdEvolution2dError> {
    if let Some(previous) = context.previous_stored_magnetic_divergence {
        validate_lengths(
            particle_count,
            &[("previous_stored_magnetic_divergence", previous.len())],
        )?;
        if previous.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                None,
                "previous_stored_magnetic_divergence",
                f64::NAN,
            ));
        }
    }
    match (context.timestep, context.courant_factor) {
        (None, None) => Ok(()),
        (Some(timestep), Some(courant_factor))
            if timestep.is_finite()
                && timestep > 0.0
                && courant_factor.is_finite()
                && courant_factor > 0.0
                && courant_factor <= 0.5 =>
        {
            Ok(())
        }
        (timestep, courant_factor) => Err(invalid(
            None,
            "magnetic_limiter_context",
            timestep.or(courant_factor).unwrap_or(f64::NAN),
        )),
    }
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
            ("cleaning_damping_rate", rates.cleaning_damping_rate.len()),
            ("acceleration", rates.acceleration.len()),
            (
                "specific_internal_energy_rates",
                rates.specific_internal_energy.len(),
            ),
            ("maximum_signal_speed", rates.maximum_signal_speed.len()),
            ("velocity_divergence", rates.velocity_divergence.len()),
            ("magnetic_divergence", rates.magnetic_divergence.len()),
            (
                "stored_magnetic_divergence",
                rates.stored_magnetic_divergence.len(),
            ),
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
        let raw = scalar_gradients_batch_with_moments_2d(
            &state.positions,
            &[&values],
            &state.smoothing_lengths,
            state.domain,
            &limiter.moments,
        )
        .unwrap()
        .pop()
        .unwrap();
        let gradients = limit_precomputed_gradient(
            &state,
            &values,
            &limiter,
            GradientConstraint2d::Positive,
            raw,
            None,
        );
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

        let predicted = predict_cleaning_mass_2d(1.0, 0.2, 2.0, 0.5);
        assert!((predicted - 1.1 / std::f64::consts::E).abs() < 1.0e-15);
        let restart_zero = kick_cleaning_mass_public_2d(
            0.0,
            0.0,
            10.0,
            2.0,
            0.5,
            1.0,
            1.0,
            1.0,
            Vector3::new(100.0, 0.0, 0.0),
            2.0,
            1.0,
            1.0,
        );
        assert!(restart_zero.value.abs() < 1.0e-15);
        let normal = kick_cleaning_mass_public_2d(
            0.1,
            0.1,
            0.2,
            2.0,
            0.5,
            1.0,
            1.0,
            1.0,
            Vector3::new(100.0, 0.0, 0.0),
            2.0,
            1.0,
            1.0,
        );
        assert!((normal.value - 0.2 / std::f64::consts::E).abs() < 1.0e-15);

        let predicted_guard = kick_cleaning_mass_public_2d(
            0.0,
            0.1,
            0.2,
            0.0,
            0.5,
            1.0,
            1.0,
            1.0,
            Vector3::new(1.0, 0.0, 0.0),
            2.0,
            1.0,
            1.0,
        );
        assert!((predicted_guard.value - 0.1).abs() < 1.0e-15);

        let decay_only = kick_cleaning_mass_public_2d(
            20.0,
            20.0,
            3.0,
            0.0,
            0.5,
            1.0,
            1.0,
            1.0,
            Vector3::new(1.0, 0.0, 0.0),
            2.0,
            1.0,
            1.0,
        );
        assert!(decay_only.effective_rate.abs() < f64::EPSILON);
        assert!((decay_only.value - 20.0).abs() < f64::EPSILON);

        let catastrophic = kick_cleaning_mass_public_2d(
            1.0,
            20_000.0,
            3.0,
            0.0,
            0.5,
            1.0,
            1.0,
            1.0,
            Vector3::new(1.0, 0.0, 0.0),
            2.0,
            1.0,
            1.0,
        );
        assert!(catastrophic.value.abs() < f64::EPSILON);
        assert!(catastrophic.effective_rate.abs() < f64::EPSILON);
        assert!(catastrophic.reset_predicted);
    }

    #[test]
    fn magnetic_gradient_limiter_relaxes_with_stored_divergence() {
        assert!((relaxed_magnetic_gradient_fraction(0.25, 0.0) - 0.25).abs() < 1.0e-15);
        assert!((relaxed_magnetic_gradient_fraction(0.25, 0.5) - 0.3125).abs() < 1.0e-15);
        assert!((relaxed_magnetic_gradient_fraction(0.25, 1.0) - 0.5).abs() < 1.0e-15);
        assert!((relaxed_magnetic_gradient_fraction(0.25, 100.0) - 0.5).abs() < 1.0e-15);
    }

    #[test]
    fn face_closure_magnetic_rate_limiter_matches_public_2d_formula() {
        let expected_area = 2.0 * std::f64::consts::PI;
        let scale = magnetic_face_closure_scale(0.1, expected_area, 1.0, 0.5, 10.0, 0.1, 0.2);
        assert!((scale - std::f64::consts::PI / 10.0).abs() < 1.0e-15);
        assert!(
            (magnetic_face_closure_scale(
                0.001 * expected_area,
                expected_area,
                1.0,
                0.5,
                1.0e6,
                1.0,
                0.2,
            ) - 1.0)
                .abs()
                < 1.0e-15
        );
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
        assert_eq!(rates.entropic_pair_count, rates.pair_count);
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
    fn public_particle_bounds_use_the_literal_c_particle_size() {
        let state = sheet(16, 4, true);
        let rates = mhd_mfm_spatial_rates_2d(&state, no_sources()).unwrap();
        let primitive = state.primitive_columns().unwrap();
        let bounds = public_mhd_particle_timestep_bounds_2d(&state, &rates, 0.2, 0.01).unwrap();
        for (i, bound) in bounds.iter().enumerate() {
            let effective_neighbor_root =
                (std::f64::consts::PI * state.smoothing_lengths[i].powi(2) * primitive.density[i]
                    / state.masses[i])
                    .sqrt();
            let particle_size = 1.77245 * state.smoothing_lengths[i] / effective_neighbor_root;
            let expected = 0.2 * particle_size / (0.5 * rates.maximum_signal_speed[i]);
            assert!((bound.courant - expected).abs() <= 4.0 * f64::EPSILON * expected);
        }
        let expected_global = bounds
            .iter()
            .map(|bound| bound.selected)
            .fold(f64::INFINITY, f64::min);
        let actual_global = global_public_mhd_timestep_bound_2d(&state, &rates, 0.2, 0.01).unwrap();
        assert_eq!(actual_global.to_bits(), expected_global.to_bits());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn initial_public_timebins_match_c_truncation_and_power_of_two_rounding() {
        let selected_ticks = [(1_u64 << 49) + 123, (1_u64 << 50) - 1, 1_u64 << 50, 1];
        let bounds: Vec<_> = selected_ticks
            .iter()
            .map(|&ticks| PublicMhdTimestepBounds2d {
                acceleration: f64::INFINITY,
                courant: ticks as f64 / crate::LEGACY_TIMEBASE_TICKS as f64,
                dedner: f64::INFINITY,
                velocity_divergence: f64::INFINITY,
                selected: ticks as f64 / crate::LEGACY_TIMEBASE_TICKS as f64,
            })
            .collect();
        let timebins = quantize_public_mhd_initial_timebins_2d(&bounds, 0.0, 1.0, 0.25).unwrap();
        assert_eq!(
            timebins
                .iter()
                .map(|timebin| timebin.raw_ticks)
                .collect::<Vec<_>>(),
            vec![(1_u64 << 49) + 123, (1_u64 << 50) - 1, 1_u64 << 50, 2,]
        );
        assert_eq!(
            timebins
                .iter()
                .map(|timebin| (timebin.time_bin, timebin.ticks))
                .collect::<Vec<_>>(),
            vec![
                (49, 1_u64 << 49),
                (49, 1_u64 << 49),
                (50, 1_u64 << 50),
                (1, 2),
            ]
        );
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
    fn public_kdk_split_exposes_c_snapshot_phase_and_matches_reference_step() {
        let state = sheet(16, 4, true);
        let controls = no_sources();
        let rates = mhd_mfm_spatial_rates_2d(&state, controls).unwrap();
        let mut step =
            begin_public_mhd_kdk_adaptive_2d(&state, &rates, 0.001, 0.0, controls, 20.0, 0.05, 0.2)
                .unwrap();
        let at_zero = step.drift_state(0.0).unwrap();
        assert_eq!(at_zero.positions, state.positions);
        assert_eq!(at_zero.predicted_velocities, state.velocities);
        assert!(
            at_zero
                .actual_velocities
                .iter()
                .zip(&state.velocities)
                .any(|(actual, initial)| *actual != *initial)
        );
        let split = finish_public_mhd_kdk_adaptive_2d(step).unwrap();
        let reference = advance_public_mhd_kdk_legacy_reference_2d(
            &state, &rates, 0.001, 0.0, controls, 20.0, 0.05, 0.2,
        )
        .unwrap();
        assert_eq!(split, reference);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn initial_hierarchy_uses_particle_half_steps_and_minimum_bin_sync() {
        let state = sheet(16, 4, true);
        let rates = mhd_mfm_spatial_rates_2d(&state, no_sources()).unwrap();
        let tick_duration = 1.0 / crate::LEGACY_TIMEBASE_TICKS as f64;
        let short_ticks = 1_u64 << 49;
        let long_ticks = 1_u64 << 50;
        let timebins: Vec<_> = (0..state.positions.len())
            .map(|i| {
                let ticks = if i % 2 == 0 { long_ticks } else { short_ticks };
                PublicMhdInitialTimebin2d {
                    bounded_timestep: ticks as f64 * tick_duration,
                    raw_ticks: ticks,
                    ticks,
                    time_bin: ticks.ilog2(),
                    duration: ticks as f64 * tick_duration,
                }
            })
            .collect();
        let mut hierarchy =
            begin_public_mhd_initial_hierarchy_2d(&state, &rates, &timebins, 0.0, 1.0, 0.0)
                .unwrap();
        let initial_primitive_cache = hierarchy.predicted_primitive_cache().clone();
        let initial_gradient_cache = hierarchy.retained_gradient_cache().clone();
        let sync = hierarchy.drift_to_first_sync().unwrap();
        assert_eq!(sync.tick, short_ticks);
        assert_eq!(
            sync.active,
            (0..state.positions.len())
                .map(|i| i % 2 != 0)
                .collect::<Vec<_>>()
        );
        let drift_duration = short_ticks as f64 * tick_duration;
        for (i, timebin) in timebins.iter().enumerate() {
            let particle_duration = timebin.duration;
            assert_eq!(
                sync.drift.actual_velocities[i],
                state.velocities[i] + rates.acceleration[i] * (0.5 * particle_duration)
            );
            let expected_predicted = if sync.active[i] {
                state.velocities[i] + rates.acceleration[i] * drift_duration
            } else {
                state.velocities[i]
            };
            assert_eq!(sync.drift.predicted_velocities[i], expected_predicted);
            assert_eq!(
                hierarchy.predictor_ticks()[i],
                if sync.active[i] { short_ticks } else { 0 }
            );
        }
        assert_eq!(
            hierarchy.predicted_primitive_cache().density[0].to_bits(),
            initial_primitive_cache.density[0].to_bits()
        );
        assert_eq!(hierarchy.retained_gradient_cache(), &initial_gradient_cache);
        hierarchy.drift_neighbor_to_current(0).unwrap();
        assert_eq!(hierarchy.predictor_ticks()[0], short_ticks);
        assert_eq!(
            hierarchy.drift.predicted_velocities[0],
            state.velocities[0] + rates.acceleration[0] * drift_duration
        );
        assert_eq!(hierarchy.retained_gradient_cache(), &initial_gradient_cache);
        assert!(hierarchy.drift_to_first_sync().is_err());

        let mut malformed_timebins = timebins.clone();
        malformed_timebins[0].raw_ticks += 1;
        assert!(
            begin_public_mhd_initial_hierarchy_2d(
                &state,
                &rates,
                &malformed_timebins,
                0.0,
                1.0,
                0.0,
            )
            .is_err()
        );

        let mut catastrophic_cleaning = state.clone();
        catastrophic_cleaning.cleaning_mass.fill(1.0e30);
        let guarded = begin_public_mhd_initial_hierarchy_2d(
            &catastrophic_cleaning,
            &rates,
            &timebins,
            0.0,
            1.0,
            0.0,
        )
        .unwrap();
        assert!(
            guarded
                .actual_cleaning_mass()
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            guarded
                .drift
                .predicted_cleaning_mass
                .iter()
                .all(|&value| value == 0.0)
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
