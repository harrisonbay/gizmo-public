#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// Normalization of GIZMO's default cubic spline in one dimension.
pub const CUBIC_1D_NORMALIZATION: f64 = 4.0 / 3.0;

/// Value and radial derivative of the default one-dimensional cubic kernel.
///
/// This is the `KERNEL_FUNCTION == 3`, `NUMDIMS == 1` specialization of
/// `kernel_main` in the pinned C baseline. `radius` is non-negative and `hsml`
/// is the full compact-support radius.
///
/// # Errors
///
/// Returns an error for non-finite inputs, negative radius, or non-positive
/// smoothing length.
pub fn cubic_kernel_1d(radius: f64, hsml: f64) -> Result<KernelValue, HydroError> {
    if !radius.is_finite() || !hsml.is_finite() || radius < 0.0 || hsml <= 0.0 {
        return Err(HydroError::InvalidKernelInput { radius, hsml });
    }
    let u = radius / hsml;
    if u >= 1.0 {
        return Ok(KernelValue {
            weight: 0.0,
            radial_derivative: 0.0,
        });
    }

    let (shape, derivative_shape) = if u < 0.5 {
        (1.0 + 6.0 * (u - 1.0) * u * u, u * (18.0 * u - 12.0))
    } else {
        let one_minus_u = 1.0 - u;
        let squared = one_minus_u * one_minus_u;
        (2.0 * squared * one_minus_u, -6.0 * squared)
    };
    let result = KernelValue {
        weight: shape * CUBIC_1D_NORMALIZATION / hsml,
        radial_derivative: derivative_shape * CUBIC_1D_NORMALIZATION / (hsml * hsml),
    };
    if !result.weight.is_finite() || !result.radial_derivative.is_finite() {
        return Err(HydroError::NonFiniteKernelResult { radius, hsml });
    }
    Ok(result)
}

/// Signed legacy periodic displacement `a - b` in a one-dimensional box.
///
/// Positions must already be wrapped into `[0, box_size)`. Exactly half-box
/// separations retain their sign, matching GIZMO's strict `>`/`<` macros.
///
/// # Errors
///
/// Returns an error unless both positions are wrapped and finite and
/// `box_size` is finite and positive.
pub fn periodic_displacement_1d(a: f64, b: f64, box_size: f64) -> Result<f64, HydroError> {
    if !a.is_finite()
        || !b.is_finite()
        || !box_size.is_finite()
        || box_size <= 0.0
        || a < 0.0
        || a >= box_size
        || b < 0.0
        || b >= box_size
    {
        return Err(HydroError::InvalidPeriodicInput { a, b, box_size });
    }
    let mut displacement = a - b;
    if displacement > 0.5 * box_size {
        displacement -= box_size;
    }
    if displacement < -0.5 * box_size {
        displacement += box_size;
    }
    Ok(displacement)
}

/// Recompute the MFM density and effective neighbor number at supplied `Hsml`.
///
/// This ports the density summation and one-dimensional neighbor-number
/// normalization from `hydro/density.c`. It intentionally does not yet solve
/// the adaptive `Hsml` constraint; callers must supply the smoothing lengths
/// whose semantics they want to validate.
///
/// # Errors
///
/// Returns an error for mismatched columns, invalid particle fields, or an
/// invalid periodic box.
pub fn density_at_hsml_1d(
    positions: &[f64],
    masses: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) -> Result<Vec<DensityEstimate>, HydroError> {
    validate_particle_columns(positions, masses, smoothing_lengths, box_size)?;
    let mut output = Vec::with_capacity(positions.len());
    for (index, &hsml) in smoothing_lengths.iter().enumerate() {
        output.push(estimate_particle(index, positions, masses, hsml, box_size)?);
    }
    Ok(output)
}

/// Solve the one-dimensional effective-neighbor constraint and density.
///
/// This uses a conservative geometric bracket around the same normalized
/// neighbor sum used by the legacy density loop. It intentionally does not
/// reproduce that loop's performance-oriented Newton jump schedule; the
/// converged physical constraint is identical and every failure is explicit.
///
/// # Errors
///
/// Returns an error for invalid particle state or constraints, or when a
/// particle cannot bracket and converge on the requested neighbor count.
pub fn solve_smoothing_lengths_1d(
    positions: &[f64],
    masses: &[f64],
    initial_smoothing_lengths: &[f64],
    box_size: f64,
    desired_neighbors: f64,
    tolerance: f64,
) -> Result<Vec<AdaptiveDensityEstimate>, HydroError> {
    validate_particle_columns(positions, masses, initial_smoothing_lengths, box_size)?;
    if !desired_neighbors.is_finite()
        || desired_neighbors <= 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || tolerance >= desired_neighbors
    {
        return Err(HydroError::InvalidNeighborConstraint {
            desired: desired_neighbors,
            tolerance,
        });
    }

    let mut output = Vec::with_capacity(positions.len());
    for (index, &initial_hsml) in initial_smoothing_lengths.iter().enumerate() {
        let mut hsml = initial_hsml;
        let mut lower: Option<f64> = None;
        let mut upper: Option<f64> = None;
        let mut last_estimate = estimate_particle(index, positions, masses, hsml, box_size)?;
        let mut converged = false;

        for _ in 0..128 {
            if (last_estimate.effective_neighbors - desired_neighbors).abs() <= tolerance {
                converged = true;
                break;
            }
            if last_estimate.effective_neighbors < desired_neighbors {
                lower = Some(lower.map_or(hsml, |bound| bound.max(hsml)));
            } else {
                upper = Some(upper.map_or(hsml, |bound| bound.min(hsml)));
            }
            hsml = match (lower, upper) {
                (Some(left), Some(right)) => (left * right).sqrt(),
                (Some(left), None) => left * 2.0,
                (None, Some(right)) => right * 0.5,
                (None, None) => unreachable!("the current estimate always sets one bound"),
            };
            if !hsml.is_finite() || hsml <= 0.0 {
                break;
            }
            last_estimate = estimate_particle(index, positions, masses, hsml, box_size)?;
        }
        if !converged {
            return Err(HydroError::SmoothingLengthDidNotConverge {
                index,
                lower,
                upper,
                effective_neighbors: last_estimate.effective_neighbors,
            });
        }
        output.push(AdaptiveDensityEstimate {
            smoothing_length: hsml,
            estimate: last_estimate,
        });
    }
    Ok(output)
}

/// Build the inverse one-dimensional MLS moment for each particle.
///
/// This is the `NV_T` geometry produced by the legacy density loop and consumed
/// by both gradient construction and meshless face geometry.
///
/// # Errors
///
/// Returns an error for invalid columns, a singular local moment, or non-finite
/// arithmetic.
pub fn inverse_moments_1d(
    positions: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) -> Result<Vec<f64>, HydroError> {
    let unit_masses = vec![1.0; positions.len()];
    validate_particle_columns(positions, &unit_masses, smoothing_lengths, box_size)?;
    let mut output = Vec::with_capacity(positions.len());
    for (index, (&position, &hsml)) in positions.iter().zip(smoothing_lengths).enumerate() {
        let mut moment = 0.0;
        for &neighbor_position in positions {
            let displacement = periodic_displacement_1d(position, neighbor_position, box_size)?;
            let distance = displacement.abs();
            if distance <= 0.0 || distance >= hsml {
                continue;
            }
            moment += cubic_kernel_1d(distance, hsml)?.weight * displacement * displacement;
        }
        if !moment.is_finite() || moment <= 0.0 {
            return Err(HydroError::SingularGradientMoment { index, moment });
        }
        let inverse = moment.recip();
        if !inverse.is_finite() {
            return Err(HydroError::NonFiniteGradient {
                index,
                field: "inverse_moment",
                value: inverse,
            });
        }
        output.push(inverse);
    }
    Ok(output)
}

/// Reconstruct a slope-limited moving-least-squares gradient in one dimension.
///
/// This is the one-dimensional specialization of the default meshless
/// `hydro_gradient_calc` path: target-kernel moment matrix, neighbor extrema,
/// matrix gradient construction, and the pure-hydro local slope limiter.
///
/// # Errors
///
/// Returns an error for invalid particle columns, non-finite values, singular
/// local moment matrices, or non-finite intermediate results.
pub fn gradients_at_hsml_1d(
    positions: &[f64],
    values: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
    shoot_tolerance: f64,
    positivity_preserving: bool,
) -> Result<Vec<GradientEstimate>, HydroError> {
    let unit_masses = vec![1.0; positions.len()];
    validate_particle_columns(positions, &unit_masses, smoothing_lengths, box_size)?;
    if values.len() != positions.len() {
        return Err(HydroError::MismatchedLength {
            field: "gradient_values",
            expected: positions.len(),
            actual: values.len(),
        });
    }
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() || (positivity_preserving && value <= 0.0) {
            return Err(HydroError::InvalidParticle {
                index,
                field: "gradient_value",
                value,
            });
        }
    }
    if !shoot_tolerance.is_finite() || shoot_tolerance < 0.0 {
        return Err(HydroError::InvalidGradientTolerance(shoot_tolerance));
    }
    let inverse_moments = inverse_moments_1d(positions, smoothing_lengths, box_size)?;

    let mut output = Vec::with_capacity(positions.len());
    for (index, ((&position, &center), &hsml)) in positions
        .iter()
        .zip(values)
        .zip(smoothing_lengths)
        .enumerate()
    {
        let mut numerator = 0.0;
        let mut minimum_delta = 0.0_f64;
        let mut maximum_delta = 0.0_f64;
        let mut max_distance = 0.0_f64;
        for ((&neighbor_position, &neighbor_value), &neighbor_hsml) in
            positions.iter().zip(values).zip(smoothing_lengths)
        {
            let displacement = periodic_displacement_1d(position, neighbor_position, box_size)?;
            let distance = displacement.abs();
            if distance <= 0.0 || (distance >= hsml && distance >= neighbor_hsml) {
                continue;
            }
            let delta = neighbor_value - center;
            minimum_delta = minimum_delta.min(delta);
            maximum_delta = maximum_delta.max(delta);
            max_distance = max_distance.max(distance);
            if distance < hsml {
                let weight = cubic_kernel_1d(distance, hsml)?.weight;
                numerator += -weight * displacement * delta;
            }
        }
        if !numerator.is_finite() {
            return Err(HydroError::NonFiniteGradient {
                index,
                field: "numerator",
                value: numerator,
            });
        }
        let unlimited = numerator * inverse_moments[index];
        if !unlimited.is_finite() {
            return Err(HydroError::NonFiniteGradient {
                index,
                field: "unlimited",
                value: unlimited,
            });
        }
        let limited = limit_gradient_1d(
            unlimited,
            minimum_delta,
            maximum_delta,
            GradientLimiter {
                center,
                hsml,
                max_distance,
                shoot_tolerance,
                positivity_preserving,
            },
        );
        if !limited.is_finite() {
            return Err(HydroError::NonFiniteGradient {
                index,
                field: "limited",
                value: limited,
            });
        }
        output.push(GradientEstimate {
            unlimited,
            limited,
            minimum_delta,
            maximum_delta,
            max_distance,
        });
    }
    Ok(output)
}

/// Construct the default non-cosmological 1-D MFM face between two particles.
///
/// The signed area follows the legacy orientation from particle `j` toward
/// particle `i`. The reconstruction offsets point from each particle center to
/// the default midpoint face.
///
/// # Errors
///
/// Returns an error for invalid particle geometry, coincident or
/// non-interacting particles, or non-finite face arithmetic.
pub fn meshless_face_geometry_1d(
    i: MeshlessPoint1d,
    j: MeshlessPoint1d,
    box_size: f64,
) -> Result<MeshlessFace1d, HydroError> {
    validate_meshless_point("i", i, box_size)?;
    validate_meshless_point("j", j, box_size)?;
    let displacement = periodic_displacement_1d(i.position, j.position, box_size)?;
    let distance = displacement.abs();
    if distance <= 0.0 || (distance >= i.smoothing_length && distance >= j.smoothing_length) {
        return Err(HydroError::InvalidFacePair {
            distance,
            hsml_i: i.smoothing_length,
            hsml_j: j.smoothing_length,
        });
    }
    let kernel_i = cubic_kernel_1d(distance, i.smoothing_length)?;
    let kernel_j = cubic_kernel_1d(distance, j.smoothing_length)?;
    let volume_i = i.mass / i.density;
    let volume_j = j.mass / j.density;
    let relative_volume_jump = (volume_i - volume_j).abs() / volume_i.min(volume_j);
    let (weight_i, weight_j) = if relative_volume_jump > 1.25 {
        let denominator = volume_i * kernel_i.weight + volume_j * kernel_j.weight;
        let centered = volume_i * volume_j * (kernel_i.weight + kernel_j.weight) / denominator;
        (centered, centered)
    } else {
        (volume_i, volume_j)
    };
    let signed_area = displacement
        * (kernel_i.weight * weight_i * i.inverse_moment
            + kernel_j.weight * weight_j * j.inverse_moment);
    let area = signed_area.abs();
    if !volume_i.is_finite()
        || !volume_j.is_finite()
        || !weight_i.is_finite()
        || !weight_j.is_finite()
        || !signed_area.is_finite()
        || area <= 0.0
    {
        return Err(HydroError::NonFiniteFaceGeometry {
            signed_area,
            volume_i,
            volume_j,
        });
    }
    Ok(MeshlessFace1d {
        signed_area,
        area,
        distance_from_i: -0.5 * displacement,
        distance_from_j: 0.5 * displacement,
    })
}

/// Reconstruct a scalar pair state using the default pure-hydro face limiter.
///
/// `right` is the state originating at particle `i`; `left` originates at
/// particle `j`, matching the legacy face-normal convention.
///
/// # Errors
///
/// Returns an error unless every input and reconstructed output is finite.
pub fn reconstruct_face_states_1d(
    value_i: f64,
    gradient_i: f64,
    value_j: f64,
    gradient_j: f64,
    face: MeshlessFace1d,
    order: ReconstructionOrder,
) -> Result<FaceStates1d, HydroError> {
    for (field, value) in [
        ("value_i", value_i),
        ("gradient_i", gradient_i),
        ("value_j", value_j),
        ("gradient_j", gradient_j),
        ("signed_area", face.signed_area),
        ("area", face.area),
        ("distance_from_i", face.distance_from_i),
        ("distance_from_j", face.distance_from_j),
    ] {
        if !value.is_finite() {
            return Err(HydroError::InvalidReconstructionInput { field, value });
        }
    }
    if face.area <= 0.0 {
        return Err(HydroError::InvalidReconstructionInput {
            field: "area",
            value: face.area,
        });
    }
    if order == ReconstructionOrder::Zeroth || legacy_float_equal(value_i, value_j) {
        return Ok(FaceStates1d {
            left: value_j,
            right: value_i,
        });
    }

    let mut right = value_i + gradient_i * face.distance_from_i;
    let mut left = value_j + gradient_j * face.distance_from_j;
    let midpoint = 0.5 * (value_i + value_j);
    let minimum = value_i.min(value_j);
    let maximum = value_i.max(value_j);
    let spread = maximum - minimum;
    let mut effective_maximum = maximum + 0.5 * spread;
    let mut effective_minimum = minimum - 0.5 * spread;
    if maximum < 0.0 && effective_maximum > 0.0 {
        effective_maximum = maximum * maximum / (maximum - (effective_maximum - maximum));
    }
    if minimum > 0.0 && effective_minimum < 0.0 {
        effective_minimum = minimum * minimum / (minimum + (minimum - effective_minimum));
    }
    let midpoint_tolerance = 0.375 * spread;
    let midpoint_maximum = (midpoint + midpoint_tolerance).min(effective_maximum);
    let midpoint_minimum = (midpoint - midpoint_tolerance).max(effective_minimum);
    if [
        right,
        left,
        midpoint,
        spread,
        effective_minimum,
        effective_maximum,
        midpoint_minimum,
        midpoint_maximum,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(HydroError::NonFiniteReconstruction { left, right });
    }
    if value_i < value_j {
        right = right.clamp(effective_minimum, midpoint_maximum);
        left = left.clamp(midpoint_minimum, effective_maximum);
    } else {
        right = right.clamp(midpoint_minimum, effective_maximum);
        left = left.clamp(effective_minimum, midpoint_maximum);
    }
    if !left.is_finite() || !right.is_finite() {
        return Err(HydroError::NonFiniteReconstruction { left, right });
    }
    Ok(FaceStates1d { left, right })
}

#[allow(clippy::float_cmp)]
fn legacy_float_equal(left: f64, right: f64) -> bool {
    left == right
}

fn limit_gradient_1d(
    gradient: f64,
    minimum_delta: f64,
    maximum_delta: f64,
    limiter: GradientLimiter,
) -> f64 {
    let magnitude = gradient.abs();
    if magnitude == 0.0 {
        return gradient;
    }
    let (absolute_minimum, absolute_maximum) = {
        let left = minimum_delta.abs();
        let right = maximum_delta.abs();
        (left.min(right), left.max(right))
    };
    let corrected_overshoot =
        (absolute_minimum + limiter.shoot_tolerance * absolute_maximum).min(absolute_maximum);
    let mut factor = corrected_overshoot / (0.25 * limiter.hsml * magnitude);
    if limiter.positivity_preserving {
        let positivity_distance = limiter.hsml.max(limiter.max_distance);
        let minimum_value = limiter
            .center
            .min(0.0_f64.max((1.0e-56 * limiter.center).max(
                (0.5 * (limiter.center + minimum_delta)).min(limiter.center - corrected_overshoot),
            )));
        factor = factor.min((limiter.center - minimum_value) / (positivity_distance * magnitude));
    }
    if factor < 1.0 {
        gradient * factor.max(0.0)
    } else {
        gradient
    }
}

#[derive(Clone, Copy)]
struct GradientLimiter {
    center: f64,
    hsml: f64,
    max_distance: f64,
    shoot_tolerance: f64,
    positivity_preserving: bool,
}

fn validate_meshless_point(
    side: &'static str,
    point: MeshlessPoint1d,
    box_size: f64,
) -> Result<(), HydroError> {
    for (field, value, positive) in [
        ("position", point.position, false),
        ("mass", point.mass, true),
        ("density", point.density, true),
        ("smoothing_length", point.smoothing_length, true),
        ("inverse_moment", point.inverse_moment, true),
    ] {
        if !value.is_finite()
            || (positive && value <= 0.0)
            || (field == "position" && (value < 0.0 || value >= box_size))
        {
            return Err(HydroError::InvalidFaceInput { side, field, value });
        }
    }
    Ok(())
}

fn validate_particle_columns(
    positions: &[f64],
    masses: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) -> Result<(), HydroError> {
    let count = positions.len();
    if masses.len() != count {
        return Err(HydroError::MismatchedLength {
            field: "masses",
            expected: count,
            actual: masses.len(),
        });
    }
    if smoothing_lengths.len() != count {
        return Err(HydroError::MismatchedLength {
            field: "smoothing_lengths",
            expected: count,
            actual: smoothing_lengths.len(),
        });
    }
    if !box_size.is_finite() || box_size <= 0.0 {
        return Err(HydroError::InvalidBoxSize(box_size));
    }
    for (index, ((&position, &mass), &hsml)) in positions
        .iter()
        .zip(masses)
        .zip(smoothing_lengths)
        .enumerate()
    {
        for (field, value, positive) in [
            ("position", position, false),
            ("mass", mass, true),
            ("smoothing_length", hsml, true),
        ] {
            if !value.is_finite()
                || (positive && value <= 0.0)
                || (field == "position" && (value < 0.0 || value >= box_size))
            {
                return Err(HydroError::InvalidParticle {
                    index,
                    field,
                    value,
                });
            }
        }
    }
    Ok(())
}

fn estimate_particle(
    index: usize,
    positions: &[f64],
    masses: &[f64],
    hsml: f64,
    box_size: f64,
) -> Result<DensityEstimate, HydroError> {
    let position = positions[index];
    let mut kernel_sum = 0.0;
    let mut mass_weighted_sum = 0.0;
    let mut derivative_sum = 0.0;
    for (&neighbor_position, &neighbor_mass) in positions.iter().zip(masses) {
        let radius = periodic_displacement_1d(position, neighbor_position, box_size)?.abs();
        let kernel = cubic_kernel_1d(radius, hsml)?;
        kernel_sum += kernel.weight;
        mass_weighted_sum += neighbor_mass * kernel.weight;
        if radius < hsml {
            derivative_sum += -(kernel.weight / hsml + (radius / hsml) * kernel.radial_derivative);
        }
    }
    for (field, value) in [
        ("kernel_sum", kernel_sum),
        ("mass_weighted_sum", mass_weighted_sum),
        ("derivative_sum", derivative_sum),
    ] {
        if !value.is_finite() {
            return Err(HydroError::NonFiniteDensityEstimate {
                index,
                field,
                value,
            });
        }
    }
    Ok(DensityEstimate {
        density: mass_weighted_sum,
        effective_neighbors: kernel_sum * 2.0 * hsml,
        dhsml_factor: if kernel_sum > 0.0 {
            let raw = derivative_sum * hsml / kernel_sum;
            if raw > -0.9 { 1.0 / (1.0 + raw) } else { 1.0 }
        } else {
            0.0
        },
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelValue {
    pub weight: f64,
    pub radial_derivative: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityEstimate {
    pub density: f64,
    pub effective_neighbors: f64,
    pub dhsml_factor: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveDensityEstimate {
    pub smoothing_length: f64,
    pub estimate: DensityEstimate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientEstimate {
    pub unlimited: f64,
    pub limited: f64,
    pub minimum_delta: f64,
    pub maximum_delta: f64,
    pub max_distance: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshlessPoint1d {
    pub position: f64,
    pub mass: f64,
    pub density: f64,
    pub smoothing_length: f64,
    pub inverse_moment: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshlessFace1d {
    pub signed_area: f64,
    pub area: f64,
    pub distance_from_i: f64,
    pub distance_from_j: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReconstructionOrder {
    Zeroth,
    First,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceStates1d {
    pub left: f64,
    pub right: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HydroError {
    InvalidKernelInput {
        radius: f64,
        hsml: f64,
    },
    NonFiniteKernelResult {
        radius: f64,
        hsml: f64,
    },
    InvalidPeriodicInput {
        a: f64,
        b: f64,
        box_size: f64,
    },
    InvalidBoxSize(f64),
    MismatchedLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidParticle {
        index: usize,
        field: &'static str,
        value: f64,
    },
    InvalidNeighborConstraint {
        desired: f64,
        tolerance: f64,
    },
    InvalidGradientTolerance(f64),
    SmoothingLengthDidNotConverge {
        index: usize,
        lower: Option<f64>,
        upper: Option<f64>,
        effective_neighbors: f64,
    },
    NonFiniteDensityEstimate {
        index: usize,
        field: &'static str,
        value: f64,
    },
    SingularGradientMoment {
        index: usize,
        moment: f64,
    },
    NonFiniteGradient {
        index: usize,
        field: &'static str,
        value: f64,
    },
    InvalidFaceInput {
        side: &'static str,
        field: &'static str,
        value: f64,
    },
    InvalidFacePair {
        distance: f64,
        hsml_i: f64,
        hsml_j: f64,
    },
    NonFiniteFaceGeometry {
        signed_area: f64,
        volume_i: f64,
        volume_j: f64,
    },
    InvalidReconstructionInput {
        field: &'static str,
        value: f64,
    },
    NonFiniteReconstruction {
        left: f64,
        right: f64,
    },
}

impl fmt::Display for HydroError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKernelInput { radius, hsml } => {
                write!(
                    formatter,
                    "invalid kernel inputs radius={radius}, hsml={hsml}"
                )
            }
            Self::NonFiniteKernelResult { radius, hsml } => write!(
                formatter,
                "kernel result is non-finite for radius={radius}, hsml={hsml}"
            ),
            Self::InvalidPeriodicInput { a, b, box_size } => {
                write!(
                    formatter,
                    "invalid periodic inputs a={a}, b={b}, box={box_size}"
                )
            }
            Self::InvalidBoxSize(value) => write!(formatter, "invalid periodic box size {value}"),
            Self::MismatchedLength {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} has length {actual}, expected {expected}"
            ),
            Self::InvalidParticle {
                index,
                field,
                value,
            } => write!(formatter, "particle {index} has invalid {field} {value}"),
            Self::InvalidNeighborConstraint { desired, tolerance } => write!(
                formatter,
                "invalid neighbor constraint desired={desired}, tolerance={tolerance}"
            ),
            Self::InvalidGradientTolerance(tolerance) => {
                write!(formatter, "invalid gradient shoot tolerance {tolerance}")
            }
            Self::SmoothingLengthDidNotConverge {
                index,
                lower,
                upper,
                effective_neighbors,
            } => write!(
                formatter,
                "particle {index} did not converge on smoothing length: \
                 bounds={lower:?}..{upper:?}, effective neighbors={effective_neighbors}"
            ),
            Self::NonFiniteDensityEstimate {
                index,
                field,
                value,
            } => write!(
                formatter,
                "particle {index} produced non-finite density accumulator `{field}`={value}"
            ),
            Self::SingularGradientMoment { index, moment } => write!(
                formatter,
                "particle {index} has singular gradient moment {moment}"
            ),
            Self::NonFiniteGradient {
                index,
                field,
                value,
            } => write!(
                formatter,
                "particle {index} produced non-finite gradient `{field}`={value}"
            ),
            Self::InvalidFaceInput { side, field, value } => {
                write!(
                    formatter,
                    "face particle {side} has invalid {field} {value}"
                )
            }
            Self::InvalidFacePair {
                distance,
                hsml_i,
                hsml_j,
            } => write!(
                formatter,
                "invalid face pair distance={distance}, hsml_i={hsml_i}, hsml_j={hsml_j}"
            ),
            Self::NonFiniteFaceGeometry {
                signed_area,
                volume_i,
                volume_j,
            } => write!(
                formatter,
                "invalid face geometry area={signed_area}, volumes={volume_i}/{volume_j}"
            ),
            Self::InvalidReconstructionInput { field, value } => {
                write!(formatter, "invalid reconstruction input {field}={value}")
            }
            Self::NonFiniteReconstruction { left, right } => {
                write!(
                    formatter,
                    "non-finite reconstructed states left={left}, right={right}"
                )
            }
        }
    }
}

impl Error for HydroError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= f64::EPSILON * expected.abs().max(1.0) * 4.0,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn cubic_kernel_matches_piecewise_boundaries() {
        let at_zero = cubic_kernel_1d(0.0, 2.0).unwrap();
        assert_close(at_zero.weight, 2.0 / 3.0);
        assert_close(at_zero.radial_derivative, 0.0);

        let at_half = cubic_kernel_1d(1.0, 2.0).unwrap();
        assert_close(at_half.weight, 1.0 / 6.0);
        assert_close(at_half.radial_derivative, -0.5);

        let at_edge = cubic_kernel_1d(2.0, 2.0).unwrap();
        assert_close(at_edge.weight, 0.0);
        assert_close(at_edge.radial_derivative, 0.0);
    }

    #[test]
    fn periodic_displacement_uses_minimum_image() {
        assert!((periodic_displacement_1d(0.99, 0.01, 1.0).unwrap() + 0.02).abs() < 1e-15);
        assert!((periodic_displacement_1d(0.01, 0.99, 1.0).unwrap() - 0.02).abs() < 1e-15);
        assert_close(periodic_displacement_1d(0.5, 0.0, 1.0).unwrap(), 0.5);
        assert_close(periodic_displacement_1d(0.0, 0.5, 1.0).unwrap(), -0.5);
        assert!(periodic_displacement_1d(1.0, 0.5, 1.0).is_err());
    }

    #[test]
    fn uniform_periodic_lattice_has_uniform_density() {
        let positions = [0.125, 0.375, 0.625, 0.875];
        let masses = [0.25; 4];
        let hsml = [0.5; 4];
        let estimates = density_at_hsml_1d(&positions, &masses, &hsml, 1.0).unwrap();
        for estimate in estimates {
            assert!((estimate.density - 1.0).abs() < 1e-15);
            assert!((estimate.effective_neighbors - 4.0).abs() < 1e-15);
            assert!(estimate.dhsml_factor.is_finite());
        }
    }

    #[test]
    fn malformed_columns_fail_closed() {
        assert!(density_at_hsml_1d(&[0.0], &[], &[0.5], 1.0).is_err());
        assert!(density_at_hsml_1d(&[0.0], &[1.0], &[0.0], 1.0).is_err());
        assert!(density_at_hsml_1d(&[0.0], &[1.0], &[0.5], -1.0).is_err());
        assert!(density_at_hsml_1d(&[0.0, 0.25], &[1.0e308; 2], &[0.5; 2], 1.0).is_err());
        assert!(cubic_kernel_1d(0.0, f64::MIN_POSITIVE).is_err());
    }

    #[test]
    fn adaptive_solver_recovers_uniform_neighbor_constraint() {
        let positions = [0.125, 0.375, 0.625, 0.875];
        let masses = [0.25; 4];
        let solved =
            solve_smoothing_lengths_1d(&positions, &masses, &[0.2; 4], 1.0, 4.0, 1.0e-12).unwrap();
        for particle in solved {
            assert!((particle.smoothing_length - 0.5).abs() < 1.0e-12);
            assert!((particle.estimate.density - 1.0).abs() < 1.0e-12);
            assert!((particle.estimate.effective_neighbors - 4.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn adaptive_solver_rejects_invalid_constraint() {
        assert!(solve_smoothing_lengths_1d(&[0.0], &[1.0], &[0.5], 1.0, 0.0, 0.1).is_err());
        assert!(solve_smoothing_lengths_1d(&[0.0], &[1.0], &[0.5], 1.0, 4.0, 4.0).is_err());
    }

    #[test]
    fn moving_least_squares_gradient_tracks_periodic_sine() {
        let count = 128_u32;
        let positions: Vec<f64> = (0..count)
            .map(|index| (f64::from(index) + 0.5) / f64::from(count))
            .collect();
        let values: Vec<f64> = positions
            .iter()
            .map(|position| 1.0 + 0.01 * (2.0 * std::f64::consts::PI * position).sin())
            .collect();
        let hsml = vec![2.0 / f64::from(count); positions.len()];
        let gradients = gradients_at_hsml_1d(&positions, &values, &hsml, 1.0, 0.0, true).unwrap();
        let normalized_mean_error = gradients
            .iter()
            .zip(&positions)
            .map(|(estimate, position)| {
                let expected =
                    0.02 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * position).cos();
                (estimate.limited - expected).abs()
            })
            .sum::<f64>()
            / (f64::from(count) * 0.02 * std::f64::consts::PI);
        assert!(
            normalized_mean_error < 1.1e-3,
            "normalized mean error {normalized_mean_error}"
        );
    }

    #[test]
    fn gradient_limiter_matches_field_specific_legacy_overshoot() {
        let limiter = |shoot_tolerance| GradientLimiter {
            center: 20.0,
            hsml: 1.0,
            max_distance: 1.0,
            shoot_tolerance,
            positivity_preserving: false,
        };
        let density_limited = limit_gradient_1d(8.0, -1.0, 10.0, limiter(0.0));
        let pressure_limited = limit_gradient_1d(8.0, -1.0, 10.0, limiter(0.1));
        assert!((density_limited - 4.0).abs() < f64::EPSILON);
        assert!((pressure_limited - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn limiter_extrema_include_neighbor_only_support() {
        let estimates = gradients_at_hsml_1d(
            &[0.5, 0.6, 0.8],
            &[10.0, 11.0, 0.0],
            &[0.15, 0.15, 0.35],
            1.0,
            0.0,
            false,
        )
        .unwrap();
        assert!((estimates[0].minimum_delta + 10.0).abs() < f64::EPSILON);
        assert!((estimates[0].max_distance - 0.3).abs() < 1.0e-15);
    }

    #[test]
    fn uniform_lattice_has_unit_antisymmetric_faces() {
        let positions = [0.125, 0.375, 0.625, 0.875];
        let hsml = [0.5; 4];
        let moments = inverse_moments_1d(&positions, &hsml, 1.0).unwrap();
        let point = |index| MeshlessPoint1d {
            position: positions[index],
            mass: 0.25,
            density: 1.0,
            smoothing_length: hsml[index],
            inverse_moment: moments[index],
        };
        let forward = meshless_face_geometry_1d(point(1), point(0), 1.0).unwrap();
        let reverse = meshless_face_geometry_1d(point(0), point(1), 1.0).unwrap();
        assert_close(forward.signed_area, 1.0);
        assert_close(forward.area, 1.0);
        assert_close(reverse.signed_area, -1.0);
        assert_close(reverse.area, 1.0);
        assert_close(forward.distance_from_i, -0.125);
        assert_close(forward.distance_from_j, 0.125);
        let across_seam = meshless_face_geometry_1d(point(0), point(3), 1.0).unwrap();
        assert_close(across_seam.signed_area, 1.0);
        assert_close(across_seam.area, 1.0);
    }

    #[test]
    fn face_geometry_matches_centered_and_single_kernel_branches() {
        let i = MeshlessPoint1d {
            position: 0.6,
            mass: 1.0,
            density: 1.0,
            smoothing_length: 0.5,
            inverse_moment: 0.5,
        };
        let j = MeshlessPoint1d {
            position: 0.4,
            mass: 3.0,
            density: 1.0,
            smoothing_length: 0.4,
            inverse_moment: 0.25,
        };
        let kernel_i = cubic_kernel_1d(0.2, i.smoothing_length).unwrap().weight;
        let kernel_j = cubic_kernel_1d(0.2, j.smoothing_length).unwrap().weight;
        let centered_weight = 3.0 * (kernel_i + kernel_j) / (kernel_i + 3.0 * kernel_j);
        let expected_centered =
            0.2 * centered_weight * (kernel_i * i.inverse_moment + kernel_j * j.inverse_moment);
        let centered = meshless_face_geometry_1d(i, j, 1.0).unwrap();
        assert_close(centered.signed_area, expected_centered);

        let strict_boundary = MeshlessPoint1d { mass: 2.25, ..j };
        let expected_unmodified =
            0.2 * (kernel_i * i.inverse_moment + kernel_j * 2.25 * strict_boundary.inverse_moment);
        let unmodified = meshless_face_geometry_1d(i, strict_boundary, 1.0).unwrap();
        assert_close(unmodified.signed_area, expected_unmodified);

        let outside_j = MeshlessPoint1d {
            mass: 1.0,
            smoothing_length: 0.1,
            ..j
        };
        let single_kernel = meshless_face_geometry_1d(i, outside_j, 1.0).unwrap();
        assert_close(single_kernel.signed_area, 0.2 * kernel_i * i.inverse_moment);
    }

    #[test]
    fn face_reconstruction_matches_legacy_orientation_and_limits() {
        let face = MeshlessFace1d {
            signed_area: 1.0,
            area: 1.0,
            distance_from_i: -1.0,
            distance_from_j: 1.0,
        };
        let linear =
            reconstruct_face_states_1d(2.0, 1.0, 0.0, 1.0, face, ReconstructionOrder::First)
                .unwrap();
        assert_close(linear.left, 1.0);
        assert_close(linear.right, 1.0);

        let saturated =
            reconstruct_face_states_1d(1.0, 11.0, 2.0, 8.0, face, ReconstructionOrder::First)
                .unwrap();
        assert_close(saturated.right, 0.5);
        assert_close(saturated.left, 2.5);

        let zeroth =
            reconstruct_face_states_1d(1.0, 11.0, 2.0, 8.0, face, ReconstructionOrder::Zeroth)
                .unwrap();
        assert_close(zeroth.right, 1.0);
        assert_close(zeroth.left, 2.0);
    }

    #[test]
    fn face_geometry_and_reconstruction_fail_closed() {
        let point = MeshlessPoint1d {
            position: 0.25,
            mass: 1.0,
            density: 1.0,
            smoothing_length: 0.1,
            inverse_moment: 1.0,
        };
        assert!(meshless_face_geometry_1d(point, point, 1.0).is_err());
        let distant = MeshlessPoint1d {
            position: 0.75,
            ..point
        };
        assert!(meshless_face_geometry_1d(point, distant, 1.0).is_err());
        let invalid = MeshlessPoint1d {
            inverse_moment: f64::INFINITY,
            ..point
        };
        assert!(meshless_face_geometry_1d(invalid, distant, 1.0).is_err());
        let face = MeshlessFace1d {
            signed_area: 1.0,
            area: 1.0,
            distance_from_i: -0.1,
            distance_from_j: 0.1,
        };
        assert!(
            reconstruct_face_states_1d(1.0, f64::NAN, 2.0, 0.0, face, ReconstructionOrder::First)
                .is_err()
        );
        assert!(
            reconstruct_face_states_1d(
                f64::MAX,
                0.0,
                0.5 * f64::MAX,
                0.0,
                face,
                ReconstructionOrder::First
            )
            .is_err()
        );
    }

    #[test]
    fn gradient_reconstruction_fails_closed() {
        assert!(gradients_at_hsml_1d(&[0.5], &[1.0], &[0.25], 1.0, 0.0, true).is_err());
        assert!(
            gradients_at_hsml_1d(&[0.25, 0.75], &[1.0, f64::NAN], &[0.6; 2], 1.0, 0.0, true)
                .is_err()
        );
        assert!(
            gradients_at_hsml_1d(&[0.25, 0.75], &[0.0, 1.0], &[0.6; 2], 1.0, 0.0, true).is_err()
        );
        assert!(
            gradients_at_hsml_1d(&[0.25, 0.75], &[1.0, 2.0], &[0.6; 2], 1.0, -0.1, true).is_err()
        );
    }
}
