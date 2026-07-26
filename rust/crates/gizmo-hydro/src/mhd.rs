//! One-dimensional ideal-MHD primitives and an HLLD Riemann solver.
//!
//! The normal direction is the local `x` direction and magnetic permeability
//! is normalized to one. The Dedner scalar is auxiliary: it is transported and
//! damped by the caller and is not included in the ideal-MHD total energy.

use std::error::Error;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

const DEGENERACY_TOLERANCE: f64 = 1.0e-11;

/// A Cartesian three-vector used by the local one-dimensional MHD solver.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x
            .mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))
    }

    #[must_use]
    pub fn squared_norm(self) -> f64 {
        self.dot(self)
    }

    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Vector3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vector3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for Vector3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f64> for Vector3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Neg for Vector3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Primitive ideal-MHD state in a local coordinate system whose normal is `x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdealMhdPrimitive1d {
    pub density: f64,
    pub velocity: Vector3,
    pub gas_pressure: f64,
    pub magnetic: Vector3,
    /// Dedner cleaning scalar, with dimensions of magnetic field times speed.
    pub cleaning_scalar: f64,
}

impl IdealMhdPrimitive1d {
    /// Convert to conservative variables for an ideal gas.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-physical primitive state or invalid `gamma`.
    pub fn to_conserved(self, gamma: f64) -> Result<IdealMhdConserved1d, MhdError> {
        validate_gamma(gamma)?;
        validate_primitive("state", self)?;
        let total_energy = self.gas_pressure / (gamma - 1.0)
            + 0.5 * self.density * self.velocity.squared_norm()
            + 0.5 * self.magnetic.squared_norm();
        let conserved = IdealMhdConserved1d {
            density: self.density,
            momentum: self.velocity * self.density,
            total_energy,
            magnetic: self.magnetic,
            cleaning_scalar: self.cleaning_scalar,
        };
        if !conserved.is_finite() {
            return Err(MhdError::NonFiniteResult("primitive-to-conserved"));
        }
        Ok(conserved)
    }

    #[must_use]
    pub fn total_pressure(self) -> f64 {
        self.gas_pressure + 0.5 * self.magnetic.squared_norm()
    }
}

/// Conservative ideal-MHD state with magnetic permeability normalized to one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IdealMhdConserved1d {
    pub density: f64,
    pub momentum: Vector3,
    pub total_energy: f64,
    pub magnetic: Vector3,
    /// Auxiliary Dedner scalar; it is not part of `total_energy`.
    pub cleaning_scalar: f64,
}

impl IdealMhdConserved1d {
    /// Convert to primitive variables for an ideal gas.
    ///
    /// # Errors
    ///
    /// Returns an error when density or recovered gas pressure is not positive,
    /// or when any input or result is non-finite.
    pub fn to_primitive(self, gamma: f64) -> Result<IdealMhdPrimitive1d, MhdError> {
        validate_gamma(gamma)?;
        if !self.is_finite() || self.density <= 0.0 {
            return Err(MhdError::InvalidConservedState);
        }
        let velocity = self.momentum / self.density;
        let gas_pressure = (gamma - 1.0)
            * (self.total_energy
                - 0.5 * self.density * velocity.squared_norm()
                - 0.5 * self.magnetic.squared_norm());
        let primitive = IdealMhdPrimitive1d {
            density: self.density,
            velocity,
            gas_pressure,
            magnetic: self.magnetic,
            cleaning_scalar: self.cleaning_scalar,
        };
        validate_primitive("recovered", primitive)?;
        Ok(primitive)
    }

    #[must_use]
    fn is_finite(self) -> bool {
        self.density.is_finite()
            && self.momentum.is_finite()
            && self.total_energy.is_finite()
            && self.magnetic.is_finite()
            && self.cleaning_scalar.is_finite()
    }
}

impl Add for IdealMhdConserved1d {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            density: self.density + rhs.density,
            momentum: self.momentum + rhs.momentum,
            total_energy: self.total_energy + rhs.total_energy,
            magnetic: self.magnetic + rhs.magnetic,
            cleaning_scalar: self.cleaning_scalar + rhs.cleaning_scalar,
        }
    }
}

impl Sub for IdealMhdConserved1d {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            density: self.density - rhs.density,
            momentum: self.momentum - rhs.momentum,
            total_energy: self.total_energy - rhs.total_energy,
            magnetic: self.magnetic - rhs.magnetic,
            cleaning_scalar: self.cleaning_scalar - rhs.cleaning_scalar,
        }
    }
}

impl Mul<f64> for IdealMhdConserved1d {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            density: self.density * rhs,
            momentum: self.momentum * rhs,
            total_energy: self.total_energy * rhs,
            magnetic: self.magnetic * rhs,
            cleaning_scalar: self.cleaning_scalar * rhs,
        }
    }
}

/// Ideal-MHD conservative flux through a face normal to the local `x` axis.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IdealMhdFlux1d {
    pub mass: f64,
    pub momentum: Vector3,
    pub total_energy: f64,
    pub magnetic: Vector3,
}

impl IdealMhdFlux1d {
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.mass.is_finite()
            && self.momentum.is_finite()
            && self.total_energy.is_finite()
            && self.magnetic.is_finite()
    }

    #[must_use]
    fn from_conserved(state: IdealMhdConserved1d) -> Self {
        Self {
            mass: state.density,
            momentum: state.momentum,
            total_energy: state.total_energy,
            magnetic: state.magnetic,
        }
    }
}

impl Add for IdealMhdFlux1d {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            mass: self.mass + rhs.mass,
            momentum: self.momentum + rhs.momentum,
            total_energy: self.total_energy + rhs.total_energy,
            magnetic: self.magnetic + rhs.magnetic,
        }
    }
}

impl Sub for IdealMhdFlux1d {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            mass: self.mass - rhs.mass,
            momentum: self.momentum - rhs.momentum,
            total_energy: self.total_energy - rhs.total_energy,
            magnetic: self.magnetic - rhs.magnetic,
        }
    }
}

impl Mul<f64> for IdealMhdFlux1d {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            mass: self.mass * rhs,
            momentum: self.momentum * rhs,
            total_energy: self.total_energy * rhs,
            magnetic: self.magnetic * rhs,
        }
    }
}

/// Frame in which a Riemann flux is returned.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FluxFrame1d {
    /// Fixed Eulerian face (`v_face = 0`).
    #[default]
    Eulerian,
    /// Face moving at the HLLD contact speed, yielding zero mass flux.
    Contact,
}

/// Optional Dedner interface correction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DednerOptions {
    /// Maximum cleaning-scalar correction as a multiple of `|B_normal|`.
    pub implicit_limiter: f64,
}

impl Default for DednerOptions {
    fn default() -> Self {
        Self {
            // reimann.h uses 0.75 unless MHD_CONSTRAINED_GRADIENT is enabled.
            // The public linear-wave profile does not enable that option.
            implicit_limiter: 0.75,
        }
    }
}

/// Configuration for the HLLD solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HlldOptions {
    pub frame: FluxFrame1d,
    pub dedner: Option<DednerOptions>,
    /// Pair-level reconstruction guard. Values above this total pressure
    /// trigger the legacy Roe/symmetric wave-speed retries.
    pub maximum_star_total_pressure: Option<f64>,
}

impl Default for HlldOptions {
    fn default() -> Self {
        Self {
            frame: FluxFrame1d::Eulerian,
            // GIZMO enables Dedner cleaning automatically under MAGNETIC.
            dedner: Some(DednerOptions::default()),
            maximum_star_total_pressure: None,
        }
    }
}

/// Solver branch used for the returned flux.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MhdRiemannMethod {
    Hlld,
    /// Positivity/degeneracy fallback enclosing all waves.
    Hlle,
}

/// HLLD result and interface quantities needed by meshless MHD updates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HlldResult {
    pub flux: IdealMhdFlux1d,
    pub method: MhdRiemannMethod,
    pub contact_speed: f64,
    pub face_velocity: f64,
    pub star_total_pressure: f64,
    pub corrected_normal_b: f64,
    pub face_magnetic: Vector3,
    pub fast_speed_left: f64,
    pub fast_speed_right: f64,
    pub phi_mean: f64,
    pub phi_db: f64,
}

/// Dedner two-wave solution for the normal magnetic field and cleaning scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DednerInterface {
    pub corrected_normal_b: f64,
    pub phi_mean: f64,
    pub phi_db: f64,
    pub fast_speed_left: f64,
    pub fast_speed_right: f64,
}

/// Errors produced by the ideal-MHD mathematical core.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MhdError {
    InvalidGamma(f64),
    InvalidPrimitiveState {
        side: &'static str,
        field: &'static str,
        value: f64,
    },
    InvalidConservedState,
    /// HLLD could not construct a physical contact fan. A fixed-mass MFM
    /// caller must retry reconstruction; accepting HLLE here would transport
    /// mass through a face whose particles keep fixed masses.
    NoAdmissibleContactFlux,
    InvalidDednerParameter {
        field: &'static str,
        value: f64,
    },
    NonFiniteResult(&'static str),
}

impl fmt::Display for MhdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidGamma(gamma) => write!(formatter, "invalid ideal-gas gamma {gamma}"),
            Self::InvalidPrimitiveState { side, field, value } => {
                write!(formatter, "invalid MHD {side} {field}={value}")
            }
            Self::InvalidConservedState => write!(formatter, "invalid conservative MHD state"),
            Self::NoAdmissibleContactFlux => {
                write!(formatter, "HLLD has no admissible fixed-mass contact flux")
            }
            Self::InvalidDednerParameter { field, value } => {
                write!(formatter, "invalid Dedner parameter {field}={value}")
            }
            Self::NonFiniteResult(stage) => {
                write!(formatter, "non-finite MHD result in {stage}")
            }
        }
    }
}

impl Error for MhdError {}

/// Fast magnetosonic speed in the local `x` direction.
///
/// # Errors
///
/// Returns an error for a non-physical state or invalid `gamma`.
pub fn fast_magnetosonic_speed(state: IdealMhdPrimitive1d, gamma: f64) -> Result<f64, MhdError> {
    validate_gamma(gamma)?;
    validate_primitive("state", state)?;
    let speed = fast_speed_unchecked(state, gamma);
    speed
        .is_finite()
        .then_some(speed)
        .ok_or(MhdError::NonFiniteResult("fast magnetosonic speed"))
}

/// Apply the two-speed Dedner interface solve used by the legacy MHD path.
///
/// The returned `phi_db` is the hyperbolic correction that multiplies the
/// oriented face area in the magnetic update.
///
/// # Errors
///
/// Returns an error for invalid states, `gamma`, or limiter.
pub fn dedner_interface(
    left: IdealMhdPrimitive1d,
    right: IdealMhdPrimitive1d,
    gamma: f64,
    options: DednerOptions,
) -> Result<DednerInterface, MhdError> {
    validate_gamma(gamma)?;
    validate_primitive("left", left)?;
    validate_primitive("right", right)?;
    if !options.implicit_limiter.is_finite() || options.implicit_limiter < 0.0 {
        return Err(MhdError::InvalidDednerParameter {
            field: "implicit_limiter",
            value: options.implicit_limiter,
        });
    }
    let fast_speed_left = fast_speed_unchecked(left, gamma);
    let fast_speed_right = fast_speed_unchecked(right, gamma);
    let speed_sum = fast_speed_left + fast_speed_right;
    if !speed_sum.is_finite() || speed_sum <= 0.0 {
        return Err(MhdError::NonFiniteResult("Dedner wave speeds"));
    }
    let inverse_sum = speed_sum.recip();
    let weighted_normal_b =
        (left.magnetic.x * fast_speed_left + right.magnetic.x * fast_speed_right) * inverse_sum;
    let correction = (left.cleaning_scalar - right.cleaning_scalar) * inverse_sum;
    let correction_limit = options.implicit_limiter * weighted_normal_b.abs();
    let correction_scale = if correction.abs() > correction_limit && correction.abs() > 0.0 {
        correction_limit / correction.abs()
    } else {
        1.0
    };
    let interface = DednerInterface {
        corrected_normal_b: weighted_normal_b + correction_scale * correction,
        phi_mean: correction_scale
            * (fast_speed_left * right.cleaning_scalar + fast_speed_right * left.cleaning_scalar)
            * inverse_sum,
        phi_db: correction_scale
            * fast_speed_left
            * fast_speed_right
            * (left.magnetic.x - right.magnetic.x)
            * inverse_sum,
        fast_speed_left,
        fast_speed_right,
    };
    if [
        interface.corrected_normal_b,
        interface.phi_mean,
        interface.phi_db,
        interface.fast_speed_left,
        interface.fast_speed_right,
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        Ok(interface)
    } else {
        Err(MhdError::NonFiniteResult("Dedner interface"))
    }
}

/// Hyperbolic Dedner source, `d(phi)/dt = -sigma * c_h^2 * div(B)`.
///
/// # Errors
///
/// Returns an error unless all arguments are finite and the speed and sigma
/// are non-negative.
pub fn dedner_hyperbolic_source(
    divergence_b: f64,
    cleaning_speed: f64,
    hyperbolic_sigma: f64,
) -> Result<f64, MhdError> {
    validate_nonnegative_dedner("cleaning_speed", cleaning_speed)?;
    validate_nonnegative_dedner("hyperbolic_sigma", hyperbolic_sigma)?;
    if !divergence_b.is_finite() {
        return Err(MhdError::InvalidDednerParameter {
            field: "divergence_b",
            value: divergence_b,
        });
    }
    let rate = -hyperbolic_sigma * cleaning_speed * cleaning_speed * divergence_b;
    rate.is_finite()
        .then_some(rate)
        .ok_or(MhdError::NonFiniteResult("Dedner hyperbolic source"))
}

/// Local parabolic Dedner damping, `d(phi)/dt = -phi/tau`.
///
/// This uses the legacy local decay rate
/// `1/tau = 0.5 * sigma * signal_speed / length`.
/// `signal_speed` is GIZMO's two-sided `MaxSignalVel` (approximately
/// `c_fast,left + c_fast,right`), not a one-sided physical wave speed.
///
/// # Errors
///
/// Returns an error unless inputs are finite, length is positive, and speed
/// and sigma are non-negative.
pub fn dedner_parabolic_source(
    cleaning_scalar: f64,
    signal_speed: f64,
    length: f64,
    parabolic_sigma: f64,
) -> Result<f64, MhdError> {
    if !cleaning_scalar.is_finite() {
        return Err(MhdError::InvalidDednerParameter {
            field: "cleaning_scalar",
            value: cleaning_scalar,
        });
    }
    validate_nonnegative_dedner("signal_speed", signal_speed)?;
    validate_nonnegative_dedner("parabolic_sigma", parabolic_sigma)?;
    if !length.is_finite() || length <= 0.0 {
        return Err(MhdError::InvalidDednerParameter {
            field: "length",
            value: length,
        });
    }
    let rate = -0.5 * parabolic_sigma * signal_speed * cleaning_scalar / length;
    rate.is_finite()
        .then_some(rate)
        .ok_or(MhdError::NonFiniteResult("Dedner parabolic source"))
}

/// Solve a local one-dimensional ideal-MHD Riemann problem with HLLD.
///
/// Invalid or non-positive HLLD intermediate states fall back to the
/// positivity-preserving two-wave HLLE flux. Both paths first enforce a single
/// normal magnetic field; Dedner correction is optional.
///
/// # Errors
///
/// Returns an error for invalid input states or non-finite final arithmetic.
#[allow(clippy::too_many_lines)]
pub fn hlld_riemann(
    mut left: IdealMhdPrimitive1d,
    mut right: IdealMhdPrimitive1d,
    gamma: f64,
    options: HlldOptions,
) -> Result<HlldResult, MhdError> {
    validate_gamma(gamma)?;
    validate_primitive("left", left)?;
    validate_primitive("right", right)?;
    if let Some(limit) = options.maximum_star_total_pressure
        && (!limit.is_finite() || limit <= 0.0)
    {
        return Err(MhdError::InvalidPrimitiveState {
            side: "interface",
            field: "maximum_star_total_pressure",
            value: limit,
        });
    }

    let (normal_b, phi_mean, phi_db, initial_fast_left, initial_fast_right) =
        if let Some(dedner) = options.dedner {
            let interface = dedner_interface(left, right, gamma, dedner)?;
            (
                interface.corrected_normal_b,
                interface.phi_mean,
                interface.phi_db,
                interface.fast_speed_left,
                interface.fast_speed_right,
            )
        } else {
            (
                0.5 * (left.magnetic.x + right.magnetic.x),
                0.0,
                0.0,
                fast_speed_unchecked(left, gamma),
                fast_speed_unchecked(right, gamma),
            )
        };
    left.magnetic.x = normal_b;
    right.magnetic.x = normal_b;

    let fast_left = fast_speed_unchecked(left, gamma);
    let fast_right = fast_speed_unchecked(right, gamma);
    if ![
        normal_b,
        initial_fast_left,
        initial_fast_right,
        fast_left,
        fast_right,
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        return Err(MhdError::NonFiniteResult("corrected wave speeds"));
    }

    let maximum_fast = fast_left.max(fast_right);
    let mut wave_left = left.velocity.x.min(right.velocity.x) - maximum_fast;
    let mut wave_right = left.velocity.x.max(right.velocity.x) + maximum_fast;
    let solve_contact = |speed_left: f64, speed_right: f64| {
        let weighted_left = left.density * (speed_left - left.velocity.x);
        let weighted_right = right.density * (speed_right - right.velocity.x);
        let denominator = weighted_left - weighted_right;
        let contact = ((right.total_pressure() - left.total_pressure())
            + weighted_left * left.velocity.x
            - weighted_right * right.velocity.x)
            / denominator;
        let pressure = left.total_pressure() + weighted_left * (contact - left.velocity.x);
        (denominator, contact, pressure)
    };
    let pressure_is_bad = |pressure: f64| {
        !pressure.is_finite()
            || pressure <= 0.0
            || options
                .maximum_star_total_pressure
                .is_some_and(|limit| pressure > limit)
    };
    let (mut denominator, mut contact_speed, mut star_total_pressure) =
        solve_contact(wave_left, wave_right);
    if pressure_is_bad(star_total_pressure) {
        let sqrt_left = left.density.sqrt();
        let sqrt_right = right.density.sqrt();
        let inverse_sum = (sqrt_left + sqrt_right).recip();
        let roe_velocity =
            (sqrt_left * left.velocity.x + sqrt_right * right.velocity.x) * inverse_sum;
        let roe_fast = (sqrt_left * fast_left + sqrt_right * fast_right) * inverse_sum;
        wave_right = (right.velocity.x + fast_right).max(roe_velocity + roe_fast);
        wave_left = (left.velocity.x - fast_left).min(roe_velocity - roe_fast);
        (denominator, contact_speed, star_total_pressure) = solve_contact(wave_left, wave_right);
    }
    if pressure_is_bad(star_total_pressure) {
        let symmetric = left.velocity.x.abs().max(right.velocity.x.abs()) + maximum_fast;
        wave_left = -symmetric;
        wave_right = symmetric;
        (denominator, contact_speed, star_total_pressure) = solve_contact(wave_left, wave_right);
    }
    let fallback_contact_speed = if contact_speed.is_finite() {
        contact_speed.clamp(wave_left, wave_right)
    } else {
        0.5 * (wave_left + wave_right)
    };
    let requested_face_velocity = match options.frame {
        FluxFrame1d::Eulerian => 0.0,
        FluxFrame1d::Contact => fallback_contact_speed,
    };

    let common = ResultCommon {
        contact_speed: fallback_contact_speed,
        face_velocity: requested_face_velocity,
        star_total_pressure: if star_total_pressure.is_finite() {
            star_total_pressure.max(0.0)
        } else {
            0.0
        },
        corrected_normal_b: normal_b,
        fast_speed_left: fast_left,
        fast_speed_right: fast_right,
        phi_mean,
        phi_db,
    };
    if !denominator.is_finite()
        || denominator.abs() <= f64::MIN_POSITIVE
        || !contact_speed.is_finite()
        || pressure_is_bad(star_total_pressure)
        || contact_speed <= wave_left
        || contact_speed >= wave_right
    {
        if options.frame == FluxFrame1d::Contact {
            return Err(MhdError::NoAdmissibleContactFlux);
        }
        return hlle_result(left, right, gamma, wave_left, wave_right, common);
    }

    match build_hlld_fan(left, right, gamma, wave_left, wave_right, common) {
        Some(result) if result.flux.is_finite() => Ok(result),
        _ if options.frame == FluxFrame1d::Contact => Err(MhdError::NoAdmissibleContactFlux),
        _ => hlle_result(left, right, gamma, wave_left, wave_right, common),
    }
}

#[derive(Clone, Copy)]
struct ResultCommon {
    contact_speed: f64,
    face_velocity: f64,
    star_total_pressure: f64,
    corrected_normal_b: f64,
    fast_speed_left: f64,
    fast_speed_right: f64,
    phi_mean: f64,
    phi_db: f64,
}

#[derive(Clone, Copy)]
struct SampledState {
    conserved: IdealMhdConserved1d,
    flux: IdealMhdFlux1d,
}

#[allow(clippy::too_many_lines)]
fn build_hlld_fan(
    left: IdealMhdPrimitive1d,
    right: IdealMhdPrimitive1d,
    gamma: f64,
    wave_left: f64,
    wave_right: f64,
    common: ResultCommon,
) -> Option<HlldResult> {
    let conserved_left = left.to_conserved(gamma).ok()?;
    let conserved_right = right.to_conserved(gamma).ok()?;
    let flux_left = physical_flux(left, gamma).ok()?;
    let flux_right = physical_flux(right, gamma).ok()?;
    let star_left = star_state(
        left,
        conserved_left,
        wave_left,
        common.contact_speed,
        common.star_total_pressure,
        common.corrected_normal_b,
    )?;
    let star_right = star_state(
        right,
        conserved_right,
        wave_right,
        common.contact_speed,
        common.star_total_pressure,
        common.corrected_normal_b,
    )?;
    if !star_is_physical(star_left, gamma) || !star_is_physical(star_right, gamma) {
        return None;
    }
    // When the normal field is dynamically negligible, the Alfvén and
    // contact waves coalesce and the star-star state is just the ordinary
    // star state. Constructing it anyway introduces irrelevant divisions and
    // can spuriously reject the hydrodynamic/HLLC limit. This is the
    // reimann.h SMALL_NUMBER branch.
    if 0.5 * common.corrected_normal_b * common.corrected_normal_b
        < DEGENERACY_TOLERANCE * common.star_total_pressure
    {
        let flux_star_left =
            flux_left + IdealMhdFlux1d::from_conserved(star_left - conserved_left) * wave_left;
        let flux_star_right =
            flux_right + IdealMhdFlux1d::from_conserved(star_right - conserved_right) * wave_right;
        let sampled = if common.face_velocity <= common.contact_speed {
            SampledState {
                conserved: star_left,
                flux: flux_star_left,
            }
        } else {
            SampledState {
                conserved: star_right,
                flux: flux_star_right,
            }
        };
        let moving_flux =
            sampled.flux - IdealMhdFlux1d::from_conserved(sampled.conserved) * common.face_velocity;
        return Some(HlldResult {
            flux: moving_flux,
            method: MhdRiemannMethod::Hlld,
            contact_speed: common.contact_speed,
            face_velocity: common.face_velocity,
            star_total_pressure: common.star_total_pressure,
            corrected_normal_b: common.corrected_normal_b,
            face_magnetic: sampled.conserved.magnetic,
            fast_speed_left: common.fast_speed_left,
            fast_speed_right: common.fast_speed_right,
            phi_mean: common.phi_mean,
            phi_db: common.phi_db,
        });
    }
    let sqrt_density_left = star_left.density.sqrt();
    let sqrt_density_right = star_right.density.sqrt();
    if !sqrt_density_left.is_finite()
        || !sqrt_density_right.is_finite()
        || sqrt_density_left <= 0.0
        || sqrt_density_right <= 0.0
    {
        return None;
    }
    let alfven_left = common.contact_speed - common.corrected_normal_b.abs() / sqrt_density_left;
    let alfven_right = common.contact_speed + common.corrected_normal_b.abs() / sqrt_density_right;
    if !alfven_left.is_finite()
        || !alfven_right.is_finite()
        || alfven_left < wave_left
        || alfven_left > common.contact_speed
        || alfven_right < common.contact_speed
        || alfven_right > wave_right
    {
        return None;
    }

    let (double_left, double_right) = double_star_states(
        star_left,
        star_right,
        common.contact_speed,
        common.corrected_normal_b,
    )?;
    if !star_is_physical(double_left, gamma) || !star_is_physical(double_right, gamma) {
        return None;
    }

    let flux_star_left =
        flux_left + IdealMhdFlux1d::from_conserved(star_left - conserved_left) * wave_left;
    let flux_star_right =
        flux_right + IdealMhdFlux1d::from_conserved(star_right - conserved_right) * wave_right;
    let flux_double_left =
        flux_star_left + IdealMhdFlux1d::from_conserved(double_left - star_left) * alfven_left;
    let flux_double_right =
        flux_star_right + IdealMhdFlux1d::from_conserved(double_right - star_right) * alfven_right;

    let sampled = if common.face_velocity <= wave_left {
        SampledState {
            conserved: conserved_left,
            flux: flux_left,
        }
    } else if common.face_velocity <= alfven_left {
        SampledState {
            conserved: star_left,
            flux: flux_star_left,
        }
    } else if common.face_velocity <= common.contact_speed {
        SampledState {
            conserved: double_left,
            flux: flux_double_left,
        }
    } else if common.face_velocity <= alfven_right {
        SampledState {
            conserved: double_right,
            flux: flux_double_right,
        }
    } else if common.face_velocity <= wave_right {
        SampledState {
            conserved: star_right,
            flux: flux_star_right,
        }
    } else {
        SampledState {
            conserved: conserved_right,
            flux: flux_right,
        }
    };
    let moving_flux =
        sampled.flux - IdealMhdFlux1d::from_conserved(sampled.conserved) * common.face_velocity;
    Some(HlldResult {
        flux: moving_flux,
        method: MhdRiemannMethod::Hlld,
        contact_speed: common.contact_speed,
        face_velocity: common.face_velocity,
        star_total_pressure: common.star_total_pressure,
        corrected_normal_b: common.corrected_normal_b,
        face_magnetic: sampled.conserved.magnetic,
        fast_speed_left: common.fast_speed_left,
        fast_speed_right: common.fast_speed_right,
        phi_mean: common.phi_mean,
        phi_db: common.phi_db,
    })
}

fn star_state(
    primitive: IdealMhdPrimitive1d,
    conserved: IdealMhdConserved1d,
    wave: f64,
    contact: f64,
    star_total_pressure: f64,
    normal_b: f64,
) -> Option<IdealMhdConserved1d> {
    let relative_wave = wave - primitive.velocity.x;
    let star_density = primitive.density * relative_wave / (wave - contact);
    if !star_density.is_finite() || star_density <= 0.0 {
        return None;
    }
    let normal_b_squared = normal_b * normal_b;
    let denominator = primitive.density * relative_wave * (wave - contact) - normal_b_squared;
    let degeneracy_scale = DEGENERACY_TOLERANCE * star_total_pressure.max(f64::MIN_POSITIVE);
    let (velocity_y, velocity_z, magnetic_y, magnetic_z) = if denominator.abs() < degeneracy_scale {
        (
            primitive.velocity.y,
            primitive.velocity.z,
            primitive.magnetic.y,
            primitive.magnetic.z,
        )
    } else {
        let velocity_factor = normal_b * (contact - primitive.velocity.x) / denominator;
        let magnetic_factor =
            (primitive.density * relative_wave * relative_wave - normal_b_squared) / denominator;
        (
            primitive.velocity.y - primitive.magnetic.y * velocity_factor,
            primitive.velocity.z - primitive.magnetic.z * velocity_factor,
            primitive.magnetic.y * magnetic_factor,
            primitive.magnetic.z * magnetic_factor,
        )
    };
    let star_velocity = Vector3::new(contact, velocity_y, velocity_z);
    let star_magnetic = Vector3::new(normal_b, magnetic_y, magnetic_z);
    let velocity_dot_b = primitive.velocity.dot(primitive.magnetic);
    let star_velocity_dot_b = star_velocity.dot(star_magnetic);
    let star_energy = (conserved.total_energy * relative_wave
        - primitive.total_pressure() * primitive.velocity.x
        + star_total_pressure * contact
        + normal_b * (velocity_dot_b - star_velocity_dot_b))
        / (wave - contact);
    let state = IdealMhdConserved1d {
        density: star_density,
        momentum: star_velocity * star_density,
        total_energy: star_energy,
        magnetic: star_magnetic,
        cleaning_scalar: primitive.cleaning_scalar,
    };
    state.is_finite().then_some(state)
}

fn double_star_states(
    left: IdealMhdConserved1d,
    right: IdealMhdConserved1d,
    contact: f64,
    normal_b: f64,
) -> Option<(IdealMhdConserved1d, IdealMhdConserved1d)> {
    let sqrt_left = left.density.sqrt();
    let sqrt_right = right.density.sqrt();
    let denominator = sqrt_left + sqrt_right;
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let inverse_sum = denominator.recip();
    let sign_b = if normal_b < 0.0 { -1.0 } else { 1.0 };
    let velocity_left = left.momentum / left.density;
    let velocity_right = right.momentum / right.density;
    let velocity_y = (sqrt_left * velocity_left.y
        + sqrt_right * velocity_right.y
        + sign_b * (right.magnetic.y - left.magnetic.y))
        * inverse_sum;
    let velocity_z = (sqrt_left * velocity_left.z
        + sqrt_right * velocity_right.z
        + sign_b * (right.magnetic.z - left.magnetic.z))
        * inverse_sum;
    let magnetic_y = (sqrt_right * left.magnetic.y
        + sqrt_left * right.magnetic.y
        + sign_b * sqrt_left * sqrt_right * (velocity_right.y - velocity_left.y))
        * inverse_sum;
    let magnetic_z = (sqrt_right * left.magnetic.z
        + sqrt_left * right.magnetic.z
        + sign_b * sqrt_left * sqrt_right * (velocity_right.z - velocity_left.z))
        * inverse_sum;
    let velocity = Vector3::new(contact, velocity_y, velocity_z);
    let magnetic = Vector3::new(normal_b, magnetic_y, magnetic_z);
    let middle_dot = velocity.dot(magnetic);
    let left_dot = velocity_left.dot(left.magnetic);
    let right_dot = velocity_right.dot(right.magnetic);
    let double_left = IdealMhdConserved1d {
        density: left.density,
        momentum: velocity * left.density,
        total_energy: left.total_energy - sign_b * sqrt_left * (left_dot - middle_dot),
        magnetic,
        cleaning_scalar: left.cleaning_scalar,
    };
    let double_right = IdealMhdConserved1d {
        density: right.density,
        momentum: velocity * right.density,
        total_energy: right.total_energy + sign_b * sqrt_right * (right_dot - middle_dot),
        magnetic,
        cleaning_scalar: right.cleaning_scalar,
    };
    (double_left.is_finite() && double_right.is_finite()).then_some((double_left, double_right))
}

fn hlle_result(
    left: IdealMhdPrimitive1d,
    right: IdealMhdPrimitive1d,
    gamma: f64,
    wave_left: f64,
    wave_right: f64,
    common: ResultCommon,
) -> Result<HlldResult, MhdError> {
    let conserved_left = left.to_conserved(gamma)?;
    let conserved_right = right.to_conserved(gamma)?;
    let flux_left = physical_flux(left, gamma)?;
    let flux_right = physical_flux(right, gamma)?;
    let sampled = if common.face_velocity <= wave_left {
        SampledState {
            conserved: conserved_left,
            flux: flux_left,
        }
    } else if common.face_velocity >= wave_right {
        SampledState {
            conserved: conserved_right,
            flux: flux_right,
        }
    } else {
        let inverse_span = (wave_right - wave_left).recip();
        SampledState {
            conserved: (conserved_right * wave_right - conserved_left * wave_left
                + IdealMhdConserved1d {
                    density: flux_left.mass - flux_right.mass,
                    momentum: flux_left.momentum - flux_right.momentum,
                    total_energy: flux_left.total_energy - flux_right.total_energy,
                    magnetic: flux_left.magnetic - flux_right.magnetic,
                    cleaning_scalar: 0.0,
                })
                * inverse_span,
            flux: (flux_left * wave_right - flux_right * wave_left
                + IdealMhdFlux1d::from_conserved(conserved_right - conserved_left)
                    * (wave_left * wave_right))
                * inverse_span,
        }
    };
    let flux =
        sampled.flux - IdealMhdFlux1d::from_conserved(sampled.conserved) * common.face_velocity;
    if !flux.is_finite() || !sampled.conserved.is_finite() {
        return Err(MhdError::NonFiniteResult("HLLE fallback"));
    }
    Ok(HlldResult {
        flux,
        method: MhdRiemannMethod::Hlle,
        contact_speed: common.contact_speed,
        face_velocity: common.face_velocity,
        star_total_pressure: common.star_total_pressure,
        corrected_normal_b: common.corrected_normal_b,
        face_magnetic: sampled.conserved.magnetic,
        fast_speed_left: common.fast_speed_left,
        fast_speed_right: common.fast_speed_right,
        phi_mean: common.phi_mean,
        phi_db: common.phi_db,
    })
}

fn physical_flux(state: IdealMhdPrimitive1d, gamma: f64) -> Result<IdealMhdFlux1d, MhdError> {
    let conserved = state.to_conserved(gamma)?;
    let normal_velocity = state.velocity.x;
    let normal_b = state.magnetic.x;
    let total_pressure = state.total_pressure();
    let mass = state.density * normal_velocity;
    let flux = IdealMhdFlux1d {
        mass,
        momentum: state.velocity * mass + Vector3::new(total_pressure, 0.0, 0.0)
            - state.magnetic * normal_b,
        total_energy: (conserved.total_energy + total_pressure) * normal_velocity
            - normal_b * state.velocity.dot(state.magnetic),
        magnetic: Vector3::new(
            0.0,
            normal_velocity * state.magnetic.y - normal_b * state.velocity.y,
            normal_velocity * state.magnetic.z - normal_b * state.velocity.z,
        ),
    };
    flux.is_finite()
        .then_some(flux)
        .ok_or(MhdError::NonFiniteResult("physical flux"))
}

fn fast_speed_unchecked(state: IdealMhdPrimitive1d, gamma: f64) -> f64 {
    let sound_squared = gamma * state.gas_pressure / state.density;
    let magnetic_squared_over_density = state.magnetic.squared_norm() / state.density;
    let normal_alfven_squared = state.magnetic.x * state.magnetic.x / state.density;
    let sum = sound_squared + magnetic_squared_over_density;
    let discriminant = (sum * sum - 4.0 * sound_squared * normal_alfven_squared).max(0.0);
    (0.5 * (sum + discriminant.sqrt())).sqrt()
}

fn star_is_physical(state: IdealMhdConserved1d, gamma: f64) -> bool {
    if !state.is_finite() || state.density <= 0.0 {
        return false;
    }
    let velocity = state.momentum / state.density;
    let gas_internal_energy = state.total_energy
        - 0.5 * state.density * velocity.squared_norm()
        - 0.5 * state.magnetic.squared_norm();
    gas_internal_energy.is_finite() && (gamma - 1.0) * gas_internal_energy > 0.0
}

fn validate_gamma(gamma: f64) -> Result<(), MhdError> {
    if gamma.is_finite() && gamma > 1.0 {
        Ok(())
    } else {
        Err(MhdError::InvalidGamma(gamma))
    }
}

fn validate_primitive(side: &'static str, state: IdealMhdPrimitive1d) -> Result<(), MhdError> {
    for (field, value) in [
        ("density", state.density),
        ("gas_pressure", state.gas_pressure),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(MhdError::InvalidPrimitiveState { side, field, value });
        }
    }
    if !state.velocity.is_finite() {
        return Err(MhdError::InvalidPrimitiveState {
            side,
            field: "velocity",
            value: f64::NAN,
        });
    }
    if !state.magnetic.is_finite() {
        return Err(MhdError::InvalidPrimitiveState {
            side,
            field: "magnetic",
            value: f64::NAN,
        });
    }
    if !state.cleaning_scalar.is_finite() {
        return Err(MhdError::InvalidPrimitiveState {
            side,
            field: "cleaning_scalar",
            value: state.cleaning_scalar,
        });
    }
    Ok(())
}

fn validate_nonnegative_dedner(field: &'static str, value: f64) -> Result<(), MhdError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(MhdError::InvalidDednerParameter { field, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PrimitiveState1d, ideal_gas_mfm_flux_1d};

    const GAMMA: f64 = 5.0 / 3.0;

    fn fixture() -> IdealMhdPrimitive1d {
        IdealMhdPrimitive1d {
            density: 1.0,
            velocity: Vector3::ZERO,
            gas_pressure: 0.6,
            magnetic: Vector3::new(1.0, std::f64::consts::SQRT_2, 0.5),
            cleaning_scalar: 0.0,
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    fn assert_vector_close(actual: Vector3, expected: Vector3, tolerance: f64) {
        assert_close(actual.x, expected.x, tolerance);
        assert_close(actual.y, expected.y, tolerance);
        assert_close(actual.z, expected.z, tolerance);
    }

    #[test]
    fn public_mhd_wave_fixture_has_fast_speed_two() {
        assert_close(
            fast_magnetosonic_speed(fixture(), GAMMA).unwrap(),
            2.0,
            8.0 * f64::EPSILON,
        );
    }

    #[test]
    fn primitive_conservative_roundtrip_preserves_all_fields() {
        let primitive = IdealMhdPrimitive1d {
            density: 1.7,
            velocity: Vector3::new(-0.4, 1.2, -0.7),
            gas_pressure: 2.3,
            magnetic: Vector3::new(0.9, -1.1, 0.2),
            cleaning_scalar: -0.08,
        };
        let recovered = primitive
            .to_conserved(1.4)
            .unwrap()
            .to_primitive(1.4)
            .unwrap();
        assert_close(recovered.density, primitive.density, 8.0 * f64::EPSILON);
        assert_vector_close(recovered.velocity, primitive.velocity, 8.0 * f64::EPSILON);
        assert_close(
            recovered.gas_pressure,
            primitive.gas_pressure,
            8.0 * f64::EPSILON,
        );
        assert_vector_close(recovered.magnetic, primitive.magnetic, 8.0 * f64::EPSILON);
        assert_close(
            recovered.cleaning_scalar,
            primitive.cleaning_scalar,
            f64::EPSILON,
        );
    }

    #[test]
    fn uniform_state_returns_exact_physical_flux() {
        let state = IdealMhdPrimitive1d {
            density: 1.0,
            velocity: Vector3::new(0.3, -0.2, 0.1),
            ..fixture()
        };
        let expected = physical_flux(state, GAMMA).unwrap();
        let result = hlld_riemann(state, state, GAMMA, HlldOptions::default()).unwrap();
        assert_eq!(result.method, MhdRiemannMethod::Hlld);
        assert_close(result.flux.mass, expected.mass, 32.0 * f64::EPSILON);
        assert_vector_close(result.flux.momentum, expected.momentum, 32.0 * f64::EPSILON);
        assert_close(
            result.flux.total_energy,
            expected.total_energy,
            32.0 * f64::EPSILON,
        );
        assert_vector_close(result.flux.magnetic, expected.magnetic, 32.0 * f64::EPSILON);
    }

    #[test]
    fn contact_frame_has_zero_mass_flux_and_advects_normal_b_with_face() {
        let left = IdealMhdPrimitive1d {
            velocity: Vector3::new(0.2, 0.1, -0.2),
            ..fixture()
        };
        let right = IdealMhdPrimitive1d {
            density: 0.8,
            velocity: Vector3::new(-0.1, -0.05, 0.3),
            gas_pressure: 0.4,
            magnetic: Vector3::new(1.0, 0.8, -0.2),
            cleaning_scalar: 0.0,
        };
        let result = hlld_riemann(
            left,
            right,
            GAMMA,
            HlldOptions {
                frame: FluxFrame1d::Contact,
                dedner: None,
                maximum_star_total_pressure: None,
            },
        )
        .unwrap();
        assert_close(result.flux.mass, 0.0, 2.0e-14);
        assert_close(
            result.flux.magnetic.x,
            -result.face_velocity * result.corrected_normal_b,
            2.0e-14,
        );
        assert_close(result.face_magnetic.x, 1.0, 2.0e-14);
    }

    #[test]
    fn zero_magnetic_contact_flux_reduces_to_existing_hllc_path() {
        let left_hydro = PrimitiveState1d {
            density: 1.1,
            velocity: 0.35,
            pressure: 1.2,
        };
        let right_hydro = PrimitiveState1d {
            density: 0.7,
            velocity: -0.15,
            pressure: 0.8,
        };
        let hydro = ideal_gas_mfm_flux_1d(left_hydro, right_hydro, 1.4, 1.0e20).unwrap();
        let to_mhd = |state: PrimitiveState1d| IdealMhdPrimitive1d {
            density: state.density,
            velocity: Vector3::new(state.velocity, 0.0, 0.0),
            gas_pressure: state.pressure,
            magnetic: Vector3::ZERO,
            cleaning_scalar: 0.0,
        };
        let mhd = hlld_riemann(
            to_mhd(left_hydro),
            to_mhd(right_hydro),
            1.4,
            HlldOptions {
                frame: FluxFrame1d::Contact,
                dedner: None,
                maximum_star_total_pressure: None,
            },
        )
        .unwrap();
        assert_eq!(mhd.method, MhdRiemannMethod::Hlld);
        assert_close(mhd.flux.mass, 0.0, 2.0e-14);
        assert_close(mhd.flux.momentum.x, hydro.momentum, 2.0e-14);
        assert_close(mhd.flux.total_energy, hydro.energy, 2.0e-14);
        assert_vector_close(mhd.flux.magnetic, Vector3::ZERO, 0.0);
    }

    #[test]
    fn reversing_normal_and_states_has_covariant_flux() {
        let left = IdealMhdPrimitive1d {
            density: 1.3,
            velocity: Vector3::new(0.4, -0.3, 0.2),
            gas_pressure: 0.9,
            magnetic: Vector3::new(0.7, 1.1, -0.6),
            cleaning_scalar: 0.0,
        };
        let right = IdealMhdPrimitive1d {
            density: 0.6,
            velocity: Vector3::new(-0.2, 0.5, -0.1),
            gas_pressure: 0.35,
            magnetic: Vector3::new(0.7, -0.4, 0.8),
            cleaning_scalar: 0.0,
        };
        let reverse = |state: IdealMhdPrimitive1d| IdealMhdPrimitive1d {
            velocity: Vector3::new(-state.velocity.x, state.velocity.y, state.velocity.z),
            magnetic: Vector3::new(-state.magnetic.x, state.magnetic.y, state.magnetic.z),
            ..state
        };
        let forward = hlld_riemann(left, right, GAMMA, HlldOptions::default()).unwrap();
        let backward =
            hlld_riemann(reverse(right), reverse(left), GAMMA, HlldOptions::default()).unwrap();
        assert_eq!(forward.method, backward.method);
        assert_close(backward.flux.mass, -forward.flux.mass, 5.0e-13);
        assert_close(backward.flux.momentum.x, forward.flux.momentum.x, 5.0e-13);
        assert_close(backward.flux.momentum.y, -forward.flux.momentum.y, 5.0e-13);
        assert_close(backward.flux.momentum.z, -forward.flux.momentum.z, 5.0e-13);
        assert_close(
            backward.flux.total_energy,
            -forward.flux.total_energy,
            5.0e-13,
        );
        assert_close(backward.flux.magnetic.x, forward.flux.magnetic.x, 5.0e-13);
        assert_close(backward.flux.magnetic.y, -forward.flux.magnetic.y, 5.0e-13);
        assert_close(backward.flux.magnetic.z, -forward.flux.magnetic.z, 5.0e-13);
    }

    #[test]
    fn magnetic_and_cleaning_sign_flip_is_covariant() {
        let left = IdealMhdPrimitive1d {
            cleaning_scalar: 0.03,
            velocity: Vector3::new(0.2, -0.1, 0.05),
            ..fixture()
        };
        let right = IdealMhdPrimitive1d {
            density: 0.85,
            velocity: Vector3::new(-0.15, 0.08, -0.04),
            gas_pressure: 0.55,
            magnetic: Vector3::new(0.92, 0.7, -0.2),
            cleaning_scalar: -0.02,
        };
        let flip = |state: IdealMhdPrimitive1d| IdealMhdPrimitive1d {
            magnetic: -state.magnetic,
            cleaning_scalar: -state.cleaning_scalar,
            ..state
        };
        let options = HlldOptions {
            frame: FluxFrame1d::Eulerian,
            dedner: Some(DednerOptions::default()),
            maximum_star_total_pressure: None,
        };
        let original = hlld_riemann(left, right, GAMMA, options).unwrap();
        let flipped = hlld_riemann(flip(left), flip(right), GAMMA, options).unwrap();
        assert_eq!(original.method, flipped.method);
        assert_close(flipped.flux.mass, original.flux.mass, 5.0e-13);
        assert_vector_close(flipped.flux.momentum, original.flux.momentum, 5.0e-13);
        assert_close(
            flipped.flux.total_energy,
            original.flux.total_energy,
            5.0e-13,
        );
        assert_vector_close(flipped.flux.magnetic, -original.flux.magnetic, 5.0e-13);
        assert_close(
            flipped.corrected_normal_b,
            -original.corrected_normal_b,
            5.0e-13,
        );
        assert_close(flipped.phi_mean, -original.phi_mean, 5.0e-13);
        assert_close(flipped.phi_db, -original.phi_db, 5.0e-13);
    }

    #[test]
    fn strong_and_degenerate_eulerian_states_return_finite_fluxes() {
        let cases = [
            (
                IdealMhdPrimitive1d {
                    density: 1.0,
                    velocity: Vector3::new(0.0, 0.0, 0.0),
                    gas_pressure: 1.0,
                    magnetic: Vector3::new(0.75, 1.0, 0.0),
                    cleaning_scalar: 0.0,
                },
                IdealMhdPrimitive1d {
                    density: 0.125,
                    velocity: Vector3::new(0.0, 0.0, 0.0),
                    gas_pressure: 0.1,
                    magnetic: Vector3::new(0.75, -1.0, 0.0),
                    cleaning_scalar: 0.0,
                },
            ),
            (
                IdealMhdPrimitive1d {
                    density: 1.0e-5,
                    velocity: Vector3::new(-20.0, 4.0, 0.0),
                    gas_pressure: 1.0e-8,
                    magnetic: Vector3::ZERO,
                    cleaning_scalar: 0.0,
                },
                IdealMhdPrimitive1d {
                    density: 20.0,
                    velocity: Vector3::new(15.0, -3.0, 1.0),
                    gas_pressure: 100.0,
                    magnetic: Vector3::ZERO,
                    cleaning_scalar: 0.0,
                },
            ),
            (
                IdealMhdPrimitive1d {
                    magnetic: Vector3::new(1.0e-20, 1.0e-20, -1.0e-20),
                    ..fixture()
                },
                IdealMhdPrimitive1d {
                    density: 0.99,
                    velocity: Vector3::new(1.0e-8, -1.0e-8, 2.0e-8),
                    gas_pressure: 0.61,
                    magnetic: Vector3::new(-1.0e-20, -1.0e-20, 1.0e-20),
                    cleaning_scalar: 0.0,
                },
            ),
        ];
        for (left, right) in cases {
            let result = hlld_riemann(
                left,
                right,
                GAMMA,
                HlldOptions {
                    frame: FluxFrame1d::Eulerian,
                    dedner: None,
                    maximum_star_total_pressure: None,
                },
            )
            .unwrap();
            assert!(result.flux.is_finite());
            assert!(result.face_magnetic.is_finite());
            assert!(result.contact_speed.is_finite());
            assert!(result.star_total_pressure.is_finite());
        }
    }

    #[test]
    fn contact_frame_rejects_hlle_mass_transport() {
        let left = IdealMhdPrimitive1d {
            density: 1.0e-5,
            velocity: Vector3::new(-20.0, 4.0, 0.0),
            gas_pressure: 1.0e-8,
            magnetic: Vector3::ZERO,
            cleaning_scalar: 0.0,
        };
        let right = IdealMhdPrimitive1d {
            density: 20.0,
            velocity: Vector3::new(15.0, -3.0, 1.0),
            gas_pressure: 100.0,
            magnetic: Vector3::ZERO,
            cleaning_scalar: 0.0,
        };
        assert!(matches!(
            hlld_riemann(
                left,
                right,
                GAMMA,
                HlldOptions {
                    frame: FluxFrame1d::Contact,
                    dedner: None,
                    maximum_star_total_pressure: None,
                },
            ),
            Err(MhdError::NoAdmissibleContactFlux)
        ));
        let eulerian = hlld_riemann(left, right, GAMMA, HlldOptions::default()).unwrap();
        assert_eq!(eulerian.method, MhdRiemannMethod::Hlle);
        assert!(eulerian.flux.is_finite());
    }

    #[test]
    fn star_pressure_guard_rejects_reconstruction_overshoot() {
        assert!(matches!(
            hlld_riemann(
                fixture(),
                fixture(),
                GAMMA,
                HlldOptions {
                    frame: FluxFrame1d::Contact,
                    dedner: Some(DednerOptions::default()),
                    maximum_star_total_pressure: Some(1.0e-6),
                },
            ),
            Err(MhdError::NoAdmissibleContactFlux)
        ));
    }

    #[test]
    fn dedner_interface_and_sources_have_expected_signs() {
        let left = IdealMhdPrimitive1d {
            magnetic: Vector3::new(1.1, 0.0, 0.0),
            cleaning_scalar: 0.04,
            ..fixture()
        };
        let right = IdealMhdPrimitive1d {
            magnetic: Vector3::new(0.9, 0.0, 0.0),
            cleaning_scalar: -0.02,
            ..fixture()
        };
        let interface = dedner_interface(left, right, GAMMA, DednerOptions::default()).unwrap();
        assert!(interface.corrected_normal_b > 1.0);
        assert!(interface.phi_db > 0.0);
        assert_close(
            dedner_hyperbolic_source(0.25, 2.0, 0.2).unwrap(),
            -0.2,
            f64::EPSILON,
        );
        assert_close(
            dedner_parabolic_source(0.5, 2.0, 0.25, 1.0).unwrap(),
            -2.0,
            f64::EPSILON,
        );
    }
}
