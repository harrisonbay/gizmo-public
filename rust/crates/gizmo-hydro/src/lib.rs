#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// Normalization of GIZMO's default cubic spline in one dimension.
pub const CUBIC_1D_NORMALIZATION: f64 = 4.0 / 3.0;
const EPSILON_ENTROPIC_BIG: f64 = 0.5;
const EPSILON_ENTROPIC_SMALL: f64 = 1.0e-3;
const CONDITION_NUMBER_DANGER_SQUARED: f64 = 1.0e6;

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

/// Compute the one-dimensional meshless face-closure diagnostic.
///
/// This is the scalar specialization of the legacy density-loop
/// `FaceClosureError`: `|(1 / sum W) * inverse_moment * sum(W dx)|`.
///
/// # Errors
///
/// Returns an error for invalid columns, singular moments, or non-finite
/// kernel arithmetic.
pub fn face_closure_errors_1d(
    positions: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) -> Result<Vec<f64>, HydroError> {
    let unit_masses = vec![1.0; positions.len()];
    validate_particle_columns(positions, &unit_masses, smoothing_lengths, box_size)?;
    let inverse_moments = inverse_moments_1d(positions, smoothing_lengths, box_size)?;
    let mut output = Vec::with_capacity(positions.len());
    for (index, (&position, &hsml)) in positions.iter().zip(smoothing_lengths).enumerate() {
        let mut kernel_sum = 0.0;
        let mut first_moment = 0.0;
        for &neighbor_position in positions {
            let displacement = periodic_displacement_1d(position, neighbor_position, box_size)?;
            let kernel = cubic_kernel_1d(displacement.abs(), hsml)?;
            kernel_sum += kernel.weight;
            if !legacy_float_equal(displacement, 0.0) {
                first_moment += kernel.weight * displacement;
            }
        }
        let closure_error = (first_moment * inverse_moments[index] / kernel_sum).abs();
        if !kernel_sum.is_finite()
            || kernel_sum <= 0.0
            || !first_moment.is_finite()
            || !closure_error.is_finite()
        {
            return Err(HydroError::NonFiniteGradient {
                index,
                field: "face_closure_error",
                value: closure_error,
            });
        }
        output.push(closure_error);
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

/// Solve the corrected ideal-gas, nonmagnetic 1-D MFM Riemann problem.
///
/// Inputs are primitive states in the rest frame of the interface, with
/// velocities projected along the face normal. The fast HLLC pressure-star
/// estimates match the legacy default path; failed estimates use its KT
/// fallback. A true two-rarefaction vacuum is the only path that emits the
/// positive minimum-pressure sentinel.
///
/// # Errors
///
/// Returns an error for invalid primitives, non-finite arithmetic, or failure
/// of the finite-checked exact-solver fallback to converge.
pub fn ideal_gas_mfm_flux_1d(
    left: PrimitiveState1d,
    right: PrimitiveState1d,
    gamma: f64,
    pressure_limit: f64,
) -> Result<MfmFlux1d, HydroError> {
    validate_riemann_state("left", left)?;
    validate_riemann_state("right", right)?;
    if !gamma.is_finite() || gamma <= 1.0 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "gamma",
            value: gamma,
        });
    }
    if !pressure_limit.is_finite() || pressure_limit <= 0.0 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "pressure_limit",
            value: pressure_limit,
        });
    }
    let sound_left = (gamma * left.pressure / left.density).sqrt();
    let sound_right = (gamma * right.pressure / right.density).sqrt();
    let enthalpy_left = specific_enthalpy(left, gamma);
    let enthalpy_right = specific_enthalpy(right, gamma);
    if [sound_left, sound_right, enthalpy_left, enthalpy_right]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "thermodynamic_state",
            value: f64::NAN,
        });
    }

    let velocity_jump = right.velocity - left.velocity;
    let vacuum_threshold = 2.0 * (sound_left + sound_right) / (gamma - 1.0);
    if velocity_jump > vacuum_threshold {
        let vacuum_pressure = 1.0e-56;
        if vacuum_pressure > pressure_limit {
            return exact_mfm_flux(left, right, gamma, sound_left, sound_right);
        }
        return Ok(MfmFlux1d {
            mass: 0.0,
            momentum: vacuum_pressure,
            energy: 0.0,
            star_pressure: vacuum_pressure,
            solver_speed: 0.0,
            method: RiemannMethod::Vacuum,
        });
    }

    match hllc_star_state(
        left,
        right,
        gamma,
        sound_left,
        sound_right,
        enthalpy_left,
        enthalpy_right,
        pressure_limit,
    ) {
        HllcOutcome::Flux {
            star_pressure,
            contact_speed,
        } => {
            let energy = star_pressure * contact_speed;
            if !energy.is_finite() {
                return Err(HydroError::NonFiniteRiemannResult {
                    field: "hllc_energy_flux",
                    value: energy,
                });
            }
            return Ok(MfmFlux1d {
                mass: 0.0,
                momentum: star_pressure,
                energy,
                star_pressure,
                solver_speed: contact_speed,
                method: RiemannMethod::Hllc,
            });
        }
        HllcOutcome::NeedsExact { star_pressure } => {
            debug_assert!(star_pressure > pressure_limit);
            return exact_mfm_flux(left, right, gamma, sound_left, sound_right);
        }
        HllcOutcome::NeedsKt => {}
    }

    match kt_mfm_flux(
        left,
        right,
        sound_left,
        sound_right,
        enthalpy_left,
        enthalpy_right,
        pressure_limit,
    ) {
        Ok(flux) => Ok(flux),
        Err(HydroError::ExactRiemannSolverRequired { .. }) => {
            exact_mfm_flux(left, right, gamma, sound_left, sound_right)
        }
        Err(error) => Err(error),
    }
}

/// Reconstruct and solve the non-cosmological 1-D MFM pair Riemann flux.
///
/// The returned momentum and energy fluxes are area-integrated and oriented
/// from particle `j` toward particle `i`, matching `face.signed_area`. They are
/// de-boosted from the midpoint interface frame back to the simulation frame.
/// The legacy pair-level reconstruction retries are included. Its subsequent
/// low-contact-speed entropic/PdV energy replacement is not.
///
/// # Errors
///
/// Returns an error for an inconsistent face, invalid primitive state,
/// non-finite reconstruction, or Riemann-solver failure.
#[allow(clippy::too_many_lines)]
pub fn mfm_pair_flux_1d(
    i: ReconstructedPoint1d,
    j: ReconstructedPoint1d,
    face: MeshlessFace1d,
    gamma: f64,
) -> Result<PairFlux1d, HydroError> {
    if !face.signed_area.is_finite()
        || face.signed_area == 0.0
        || !legacy_float_equal(face.area, face.signed_area.abs())
        || !face.distance_from_i.is_finite()
        || !face.distance_from_j.is_finite()
        || legacy_float_equal(face.distance_from_i, face.distance_from_j)
        || face.distance_from_i.signum() != -face.signed_area.signum()
        || face.distance_from_j.signum() != face.signed_area.signum()
        || (face.distance_from_i + face.distance_from_j).abs()
            > f64::EPSILON * (face.distance_from_i.abs() + face.distance_from_j.abs())
    {
        return Err(HydroError::InvalidFaceInput {
            side: "pair",
            field: "signed_area",
            value: face.signed_area,
        });
    }
    validate_riemann_state("i", i.primitive)?;
    validate_riemann_state("j", j.primitive)?;
    if !i.face_closure_error.is_finite() || i.face_closure_error < 0.0 {
        return Err(HydroError::InvalidReconstructionInput {
            field: "i_face_closure_error",
            value: i.face_closure_error,
        });
    }
    if !j.face_closure_error.is_finite() || j.face_closure_error < 0.0 {
        return Err(HydroError::InvalidReconstructionInput {
            field: "j_face_closure_error",
            value: j.face_closure_error,
        });
    }
    let closure_leak = 0.5 * (i.face_closure_error + j.face_closure_error);
    if !closure_leak.is_finite() {
        return Err(HydroError::InvalidReconstructionInput {
            field: "closure_leak",
            value: closure_leak,
        });
    }

    let density = reconstruct_face_states_1d(
        i.primitive.density,
        i.density_gradient,
        j.primitive.density,
        j.density_gradient,
        face,
        ReconstructionOrder::First,
    );
    let velocity = reconstruct_face_states_1d(
        i.primitive.velocity,
        i.velocity_gradient,
        j.primitive.velocity,
        j.velocity_gradient,
        face,
        ReconstructionOrder::First,
    );
    let pressure = reconstruct_face_states_1d(
        i.primitive.pressure,
        i.pressure_gradient,
        j.primitive.pressure,
        j.pressure_gradient,
        face,
        ReconstructionOrder::First,
    );

    let normal = face.signed_area.signum();
    let interface_velocity = 0.5 * (i.primitive.velocity + j.primitive.velocity);
    let normal_velocity_i = i.primitive.velocity * normal;
    let normal_velocity_j = j.primitive.velocity * normal;
    let approach_velocity = (normal_velocity_i - normal_velocity_j).min(0.0);
    let approach_speed_squared = approach_velocity * approach_velocity;
    let pressure_limit = 1.1
        * (i.primitive.pressure + i.primitive.density * approach_speed_squared)
            .max(j.primitive.pressure + j.primitive.density * approach_speed_squared);
    if !pressure_limit.is_finite() || pressure_limit <= 0.0 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "pair_pressure_limit",
            value: pressure_limit,
        });
    }

    let solve = |left: PrimitiveState1d, right: PrimitiveState1d, limit: f64| {
        ideal_gas_mfm_flux_1d(left, right, gamma, limit)
    };
    let first = (closure_leak <= 1.0).then(|| match (density, velocity, pressure) {
        (Ok(density), Ok(velocity), Ok(pressure)) => solve(
            PrimitiveState1d {
                density: density.left,
                velocity: (velocity.left - interface_velocity) * normal,
                pressure: pressure.left,
            },
            PrimitiveState1d {
                density: density.right,
                velocity: (velocity.right - interface_velocity) * normal,
                pressure: pressure.right,
            },
            pressure_limit,
        ),
        (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => Err(error),
    });
    let (interface_flux, solve_path) = match first {
        Some(Ok(flux)) if flux.star_pressure <= 1.4 * pressure_limit => {
            (flux, PairSolvePath::Reconstructed)
        }
        Some(Ok(_) | Err(_)) | None => {
            let centered_left = PrimitiveState1d {
                density: j.primitive.density,
                velocity: (j.primitive.velocity - interface_velocity) * normal,
                pressure: j.primitive.pressure,
            };
            let centered_right = PrimitiveState1d {
                density: i.primitive.density,
                velocity: (i.primitive.velocity - interface_velocity) * normal,
                pressure: i.primitive.pressure,
            };
            match solve(centered_left, centered_right, 1.4 * pressure_limit) {
                Ok(flux) => (flux, PairSolvePath::Centered),
                Err(_) => (
                    solve(
                        PrimitiveState1d {
                            velocity: 0.0,
                            ..centered_left
                        },
                        PrimitiveState1d {
                            velocity: 0.0,
                            ..centered_right
                        },
                        2.0 * pressure_limit,
                    )?,
                    PairSolvePath::ZeroRelativeVelocity,
                ),
            }
        }
    };
    let momentum = face.area * interface_flux.momentum * normal;
    let energy =
        face.area * (interface_flux.energy + interface_velocity * interface_flux.momentum * normal);
    if !momentum.is_finite() || !energy.is_finite() {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "pair_flux",
            value: f64::NAN,
        });
    }
    Ok(PairFlux1d {
        mass: 0.0,
        momentum,
        energy,
        star_pressure: interface_flux.star_pressure,
        interface_velocity,
        solver_speed: interface_flux.solver_speed,
        method: interface_flux.method,
        solve_path,
        closure_leak,
    })
}

/// Inputs to GIZMO's low-contact-speed MFM entropic/PdV energy correction.
///
/// `kernel_radial_derivative` is `dW(r,h)/dr` evaluated with this particle's
/// smoothing length. `volume` is the particle mass divided by its density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EntropicPoint1d {
    pub velocity: f64,
    pub density: f64,
    pub pressure: f64,
    pub sound_speed: f64,
    pub volume: f64,
    pub dhsml_factor: f64,
    pub kernel_radial_derivative: f64,
    pub condition_number: f64,
    pub face_closure_error: f64,
}

/// Apply the legacy non-cosmological 1-D MFM entropic/PdV energy correction.
///
/// The input must be the raw, area-integrated lab-frame result from
/// [`mfm_pair_flux_1d`]. The returned boolean reports whether the entropic
/// energy equation was selected. Momentum and mass fluxes are unchanged.
///
/// # Errors
///
/// Returns an error for invalid thermodynamic, geometric, kernel, or flux
/// inputs, or if the correction produces a non-finite energy flux.
#[allow(clippy::float_cmp, clippy::too_many_lines)]
pub fn apply_entropic_pdv_1d(
    mut flux: PairFlux1d,
    face: MeshlessFace1d,
    i: EntropicPoint1d,
    j: EntropicPoint1d,
) -> Result<(PairFlux1d, bool), HydroError> {
    if !face.signed_area.is_finite()
        || face.signed_area == 0.0
        || !legacy_float_equal(face.area, face.signed_area.abs())
        || !face.distance_from_i.is_finite()
        || !face.distance_from_j.is_finite()
        || legacy_float_equal(face.distance_from_i, face.distance_from_j)
        || face.distance_from_i.signum() != -face.signed_area.signum()
        || face.distance_from_j.signum() != face.signed_area.signum()
        || (face.distance_from_i + face.distance_from_j).abs()
            > f64::EPSILON * (face.distance_from_i.abs() + face.distance_from_j.abs())
    {
        return Err(HydroError::InvalidFaceInput {
            side: "pair",
            field: "signed_area",
            value: face.signed_area,
        });
    }
    validate_entropic_point("i", i)?;
    validate_entropic_point("j", j)?;
    for (field, value) in [
        ("mass", flux.mass),
        ("momentum", flux.momentum),
        ("energy", flux.energy),
        ("star_pressure", flux.star_pressure),
        ("interface_velocity", flux.interface_velocity),
        ("solver_speed", flux.solver_speed),
        ("closure_leak", flux.closure_leak),
    ] {
        if !value.is_finite() {
            return Err(HydroError::NonFiniteRiemannResult { field, value });
        }
    }
    if flux.star_pressure < 0.0 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "star_pressure",
            value: flux.star_pressure,
        });
    }
    let midpoint_velocity = 0.5 * (i.velocity + j.velocity);
    let expected_momentum = flux.star_pressure * face.area * face.signed_area.signum();
    if !legacy_float_equal(flux.mass, 0.0)
        || !legacy_float_equal(flux.interface_velocity, midpoint_velocity)
        || !legacy_float_equal(flux.momentum, expected_momentum)
    {
        return Err(HydroError::InvalidRiemannParameter {
            field: "entropic_source_flux",
            value: flux.momentum,
        });
    }

    let normal = face.signed_area / face.area;
    let face_velocity_i = i.velocity * normal;
    let face_velocity_j = j.velocity * normal;
    let face_velocity = 0.5 * (face_velocity_i + face_velocity_j);
    let relative_velocity = face_velocity_i - face_velocity_j;
    let sound_speed = i.sound_speed.min(j.sound_speed);
    let speed_ratio = flux.solver_speed.abs() / sound_speed;
    let closure_leak = 0.5 * (i.face_closure_error + j.face_closure_error);
    if !legacy_float_equal(flux.closure_leak, closure_leak) {
        return Err(HydroError::InvalidRiemannParameter {
            field: "closure_leak",
            value: flux.closure_leak,
        });
    }
    if !speed_ratio.is_finite() || !closure_leak.is_finite() {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "entropic_gate",
            value: f64::NAN,
        });
    }
    if speed_ratio >= EPSILON_ENTROPIC_BIG && closure_leak <= 1.0 {
        return Ok((flux, false));
    }

    let pressure_area = flux.star_pressure * face.area;
    let pdv_factor = flux.star_pressure * relative_velocity;
    let pdv_i = i.kernel_radial_derivative * i.volume * i.volume * i.dhsml_factor * pdv_factor;
    let pdv_j = j.kernel_radial_derivative * j.volume * j.volume * j.dhsml_factor * pdv_factor;
    let old_energy = pressure_area * (flux.solver_speed + face_velocity);
    let new_energy = 0.5 * (pdv_i - pdv_j + pressure_area * (face_velocity_i + face_velocity_j));
    if !pressure_area.is_finite()
        || !pdv_i.is_finite()
        || !pdv_j.is_finite()
        || !old_energy.is_finite()
        || !new_energy.is_finite()
    {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "entropic_energy",
            value: f64::NAN,
        });
    }

    let condition_i_squared = i.condition_number * i.condition_number;
    let neighbor_condition_squared = j.condition_number * j.condition_number;
    if !condition_i_squared.is_finite() || !neighbor_condition_squared.is_finite() {
        return Err(HydroError::InvalidRiemannParameter {
            field: "condition_number",
            value: i.condition_number.max(j.condition_number),
        });
    }
    let condition_threshold = CONDITION_NUMBER_DANGER_SQUARED - condition_i_squared;
    let mut use_entropic_energy = true;
    if speed_ratio > EPSILON_ENTROPIC_SMALL
        && neighbor_condition_squared < condition_threshold
        && i.pressure / i.density != j.pressure / j.density
    {
        if i.pressure / i.density > j.pressure / j.density {
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
    if neighbor_condition_squared >= condition_threshold {
        use_entropic_energy = true;
    }
    if use_entropic_energy {
        flux.energy += new_energy - old_energy;
        if !flux.energy.is_finite() {
            return Err(HydroError::NonFiniteRiemannResult {
                field: "entropic_corrected_energy",
                value: flux.energy,
            });
        }
    }
    Ok((flux, use_entropic_energy))
}

fn validate_entropic_point(side: &'static str, point: EntropicPoint1d) -> Result<(), HydroError> {
    for (field, value) in [
        ("density", point.density),
        ("pressure", point.pressure),
        ("sound_speed", point.sound_speed),
        ("volume", point.volume),
        ("dhsml_factor", point.dhsml_factor),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(HydroError::InvalidRiemannState { side, field, value });
        }
    }
    for (field, value) in [
        ("velocity", point.velocity),
        ("kernel_radial_derivative", point.kernel_radial_derivative),
    ] {
        if !value.is_finite() || (field == "kernel_radial_derivative" && value > 0.0) {
            return Err(HydroError::InvalidRiemannState { side, field, value });
        }
    }
    for (field, value) in [
        ("condition_number", point.condition_number),
        ("face_closure_error", point.face_closure_error),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(HydroError::InvalidRiemannState { side, field, value });
        }
    }
    Ok(())
}

enum HllcOutcome {
    Flux {
        star_pressure: f64,
        contact_speed: f64,
    },
    NeedsKt,
    NeedsExact {
        star_pressure: f64,
    },
}

fn accepted_hllc_flux(
    star_pressure: f64,
    contact_speed: f64,
    pressure_limit: f64,
) -> Option<HllcOutcome> {
    (star_pressure.is_finite()
        && contact_speed.is_finite()
        && star_pressure > 0.0
        && star_pressure <= pressure_limit)
        .then_some(HllcOutcome::Flux {
            star_pressure,
            contact_speed,
        })
}

#[allow(clippy::too_many_arguments)]
fn hllc_star_state(
    left: PrimitiveState1d,
    right: PrimitiveState1d,
    gamma: f64,
    sound_left: f64,
    sound_right: f64,
    enthalpy_left: f64,
    enthalpy_right: f64,
    pressure_limit: f64,
) -> HllcOutcome {
    let sound_maximum = sound_left.max(sound_right);
    let mut wave_left = left.velocity.min(right.velocity) - sound_maximum;
    let mut wave_right = left.velocity.max(right.velocity) + sound_maximum;
    let mut density_weight_left = left.density * (wave_left - left.velocity);
    let mut density_weight_right = right.density * (wave_right - right.velocity);
    let mut contact_speed = ((right.pressure - left.pressure)
        + density_weight_left * left.velocity
        - density_weight_right * right.velocity)
        / (density_weight_left - density_weight_right);
    let mut star_pressure = (left.pressure * density_weight_right
        - right.pressure * density_weight_left
        + density_weight_left * density_weight_right * (right.velocity - left.velocity))
        / (density_weight_right - density_weight_left);
    if let Some(outcome) = accepted_hllc_flux(star_pressure, contact_speed, pressure_limit) {
        return outcome;
    }

    let sqrt_density_left = left.density.sqrt();
    let sqrt_density_right = right.density.sqrt();
    let inverse_sum = (sqrt_density_left + sqrt_density_right).recip();
    let roe_velocity =
        (sqrt_density_left * left.velocity + sqrt_density_right * right.velocity) * inverse_sum;
    let roe_enthalpy =
        (sqrt_density_left * enthalpy_left + sqrt_density_right * enthalpy_right) * inverse_sum;
    let roe_sound = ((gamma - 1.0) * (roe_enthalpy - 0.5 * roe_velocity * roe_velocity))
        .max(1.0e-30)
        .sqrt();
    wave_right = (right.velocity + sound_right).max(roe_velocity + roe_sound);
    wave_left = (left.velocity - sound_left).min(roe_velocity - roe_sound);
    density_weight_right = right.density * (wave_right - right.velocity);
    density_weight_left = -left.density * (wave_left - left.velocity);
    contact_speed = (density_weight_right * right.velocity
        + density_weight_left * left.velocity
        + left.pressure
        - right.pressure)
        / (density_weight_right + density_weight_left);
    star_pressure = left.density * (left.velocity - wave_left) * (left.velocity - contact_speed)
        + left.pressure;
    if let Some(outcome) = accepted_hllc_flux(star_pressure, contact_speed, pressure_limit) {
        return outcome;
    }

    star_pressure = 0.5
        * (left.pressure
            + right.pressure
            + (left.velocity - right.velocity)
                * 0.25
                * (left.density + right.density)
                * (sound_left + sound_right));
    contact_speed = 0.5 * (right.velocity + left.velocity)
        + 2.0 * (left.pressure - right.pressure)
            / ((left.density + right.density) * (sound_left + sound_right));
    let signal_speed = [
        (left.velocity - sound_left).abs(),
        (right.velocity - sound_right).abs(),
        (left.velocity + sound_left).abs(),
        (right.velocity + sound_right).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max);
    contact_speed = contact_speed.clamp(-signal_speed, signal_speed);
    if !star_pressure.is_finite() || !contact_speed.is_finite() || star_pressure < 0.0 {
        HllcOutcome::NeedsKt
    } else if star_pressure > pressure_limit {
        HllcOutcome::NeedsExact { star_pressure }
    } else {
        HllcOutcome::Flux {
            star_pressure,
            contact_speed,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn kt_mfm_flux(
    left: PrimitiveState1d,
    right: PrimitiveState1d,
    sound_left: f64,
    sound_right: f64,
    enthalpy_left: f64,
    enthalpy_right: f64,
    pressure_limit: f64,
) -> Result<MfmFlux1d, HydroError> {
    let momentum_difference = right.density * right.velocity - left.density * left.velocity;
    let threshold = 0.001 * 0.5 * (left.density + right.density) * 0.5 * (sound_left + sound_right);
    let alpha = momentum_difference.abs()
        / (threshold * threshold + momentum_difference * momentum_difference).sqrt();
    let wave_left = alpha * sound_left + left.velocity.abs();
    let wave_right = alpha * sound_right + right.velocity.abs();
    let diffusion_speed = wave_left.max(wave_right);
    let signal_speed = (sound_left + left.velocity.abs()).max(sound_right + right.velocity.abs());
    let weighted_left = left.density * (left.velocity + diffusion_speed);
    let weighted_right = right.density * (right.velocity - diffusion_speed);
    let denominator_base = left.density * left.velocity - right.density * right.velocity
        + diffusion_speed * (left.density + right.density);
    if !denominator_base.is_finite() || denominator_base == 0.0 {
        return Err(HydroError::ExactRiemannSolverRequired {
            star_pressure: f64::NAN,
            pressure_limit,
        });
    }
    let denominator = denominator_base.recip();
    let weighted_product = weighted_left * weighted_right;
    let star_pressure =
        (weighted_left * right.pressure - weighted_right * left.pressure) * denominator;
    if !star_pressure.is_finite() {
        return Err(HydroError::ExactRiemannSolverRequired {
            star_pressure,
            pressure_limit,
        });
    }
    if star_pressure <= 0.0 || star_pressure > pressure_limit {
        return Err(HydroError::ExactRiemannSolverRequired {
            star_pressure,
            pressure_limit,
        });
    }
    let momentum =
        (right.velocity - left.velocity) * weighted_product * denominator + star_pressure;
    let energy = (diffusion_speed
        * (weighted_left * right.pressure + weighted_right * left.pressure)
        + (enthalpy_right - enthalpy_left) * weighted_product)
        * denominator;
    if [
        alpha,
        diffusion_speed,
        signal_speed,
        denominator,
        momentum,
        energy,
        star_pressure,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "kt_flux",
            value: f64::NAN,
        });
    }
    Ok(MfmFlux1d {
        mass: 0.0,
        momentum,
        energy,
        star_pressure,
        solver_speed: signal_speed,
        method: RiemannMethod::KurganovTadmor,
    })
}

fn exact_mfm_flux(
    left: PrimitiveState1d,
    right: PrimitiveState1d,
    gamma: f64,
    sound_left: f64,
    sound_right: f64,
) -> Result<MfmFlux1d, HydroError> {
    let rarefaction_factor = 2.0 / (gamma - 1.0);
    let vacuum_threshold = rarefaction_factor * (sound_left + sound_right);
    if right.velocity - left.velocity >= vacuum_threshold {
        let left_fan_edge = left.velocity + rarefaction_factor * sound_left;
        let right_fan_edge = right.velocity - rarefaction_factor * sound_right;
        let contact_speed = if 0.0 <= left_fan_edge {
            left_fan_edge
        } else if 0.0 >= right_fan_edge {
            right_fan_edge
        } else {
            0.0
        };
        return Ok(MfmFlux1d {
            mass: 0.0,
            momentum: 0.0,
            energy: 0.0,
            star_pressure: 0.0,
            solver_speed: contact_speed,
            method: RiemannMethod::Exact,
        });
    }

    let mut star_pressure = exact_pressure_guess(left, right, gamma, sound_left, sound_right);
    if !star_pressure.is_finite() || star_pressure <= 0.0 {
        return Err(HydroError::ExactRiemannSolverDidNotConverge {
            iterations: 0,
            pressure: star_pressure,
        });
    }

    let mut converged = false;
    let mut iterations = 0_u32;
    while iterations < 1_000 {
        let previous = star_pressure;
        let wave_left = exact_wave_curve(previous, left, gamma, sound_left);
        let wave_right = exact_wave_curve(previous, right, gamma, sound_right);
        if !wave_left.value.is_finite()
            || !wave_left.derivative.is_finite()
            || !wave_right.value.is_finite()
            || !wave_right.derivative.is_finite()
        {
            break;
        }
        let derivative = wave_left.derivative + wave_right.derivative;
        if !derivative.is_finite() || derivative <= 0.0 {
            break;
        }
        star_pressure -=
            (wave_left.value + wave_right.value + right.velocity - left.velocity) / derivative;
        if star_pressure < 0.1 * previous {
            star_pressure = 0.1 * previous;
        }
        iterations += 1;
        if !star_pressure.is_finite() || star_pressure <= 0.0 {
            break;
        }
        let tolerance = 2.0 * ((star_pressure - previous) / (star_pressure + previous)).abs();
        if !tolerance.is_finite() {
            break;
        }
        if tolerance <= 1.0e-6 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(HydroError::ExactRiemannSolverDidNotConverge {
            iterations,
            pressure: star_pressure,
        });
    }

    let wave_left = exact_wave_curve(star_pressure, left, gamma, sound_left);
    let wave_right = exact_wave_curve(star_pressure, right, gamma, sound_right);
    let residual = wave_left.value + wave_right.value + right.velocity - left.velocity;
    let residual_scale =
        (sound_left + sound_right + (right.velocity - left.velocity).abs()).max(1.0);
    if !residual.is_finite() || residual.abs() > 2.0e-6 * residual_scale {
        return Err(HydroError::ExactRiemannSolverDidNotConverge {
            iterations,
            pressure: star_pressure,
        });
    }
    let contact_speed =
        0.5 * (left.velocity + right.velocity) + 0.5 * (wave_right.value - wave_left.value);
    let energy = star_pressure * contact_speed;
    if [star_pressure, contact_speed, energy]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "exact_flux",
            value: f64::NAN,
        });
    }
    Ok(MfmFlux1d {
        mass: 0.0,
        momentum: star_pressure,
        energy,
        star_pressure,
        solver_speed: contact_speed,
        method: RiemannMethod::Exact,
    })
}

struct ExactWaveCurve {
    value: f64,
    derivative: f64,
}

fn exact_wave_curve(
    pressure: f64,
    state: PrimitiveState1d,
    gamma: f64,
    sound_speed: f64,
) -> ExactWaveCurve {
    if pressure > state.pressure {
        let coefficient = 2.0 / ((gamma + 1.0) * state.density);
        let offset = (gamma - 1.0) * state.pressure / (gamma + 1.0);
        let root = (coefficient / (pressure + offset)).sqrt();
        ExactWaveCurve {
            value: (pressure - state.pressure) * root,
            derivative: root * (1.0 - 0.5 * (pressure - state.pressure) / (pressure + offset)),
        }
    } else {
        let pressure_ratio = pressure / state.pressure;
        let pressure_exponent = (gamma - 1.0) / (2.0 * gamma);
        ExactWaveCurve {
            value: 2.0 * sound_speed / (gamma - 1.0)
                * (pressure_ratio.powf(pressure_exponent) - 1.0),
            derivative: pressure_ratio.powf(-(gamma + 1.0) / (2.0 * gamma))
                / (state.density * sound_speed),
        }
    }
}

fn exact_pressure_guess(
    left: PrimitiveState1d,
    right: PrimitiveState1d,
    gamma: f64,
    sound_left: f64,
    sound_right: f64,
) -> f64 {
    let minimum = left.pressure.min(right.pressure);
    let maximum = left.pressure.max(right.pressure);
    let primitive = 0.5 * (left.pressure + right.pressure)
        - 0.125
            * (right.velocity - left.velocity)
            * (left.density + right.density)
            * (sound_left + sound_right);
    if maximum / minimum <= 2.0 && (minimum..=maximum).contains(&primitive) {
        return primitive;
    }
    if primitive < minimum {
        let exponent = (gamma - 1.0) / (2.0 * gamma);
        let numerator =
            sound_left + sound_right - 0.5 * (gamma - 1.0) * (right.velocity - left.velocity);
        let denominator =
            sound_left / left.pressure.powf(exponent) + sound_right / right.pressure.powf(exponent);
        return (numerator / denominator).powf(2.0 * gamma / (gamma - 1.0));
    }
    let left_weight = (2.0
        / ((gamma + 1.0)
            * left.density
            * ((gamma - 1.0) * left.pressure / (gamma + 1.0) + primitive)))
        .sqrt();
    let right_weight = (2.0
        / ((gamma + 1.0)
            * right.density
            * ((gamma - 1.0) * right.pressure / (gamma + 1.0) + primitive)))
        .sqrt();
    let two_shock = (left_weight * left.pressure + right_weight * right.pressure
        - (right.velocity - left.velocity))
        / (left_weight + right_weight);
    if two_shock < minimum || two_shock > maximum {
        minimum
    } else {
        two_shock
    }
}

fn specific_enthalpy(state: PrimitiveState1d, gamma: f64) -> f64 {
    state.pressure / state.density
        + state.pressure / ((gamma - 1.0) * state.density)
        + 0.5 * state.velocity * state.velocity
}

fn validate_riemann_state(side: &'static str, state: PrimitiveState1d) -> Result<(), HydroError> {
    for (field, value, positive) in [
        ("density", state.density, true),
        ("velocity", state.velocity, false),
        ("pressure", state.pressure, true),
    ] {
        if !value.is_finite() || (positive && value <= 0.0) {
            return Err(HydroError::InvalidRiemannState { side, field, value });
        }
    }
    Ok(())
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimitiveState1d {
    pub density: f64,
    pub velocity: f64,
    pub pressure: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructedPoint1d {
    pub primitive: PrimitiveState1d,
    pub density_gradient: f64,
    pub velocity_gradient: f64,
    pub pressure_gradient: f64,
    pub face_closure_error: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RiemannMethod {
    Hllc,
    KurganovTadmor,
    Exact,
    Vacuum,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MfmFlux1d {
    pub mass: f64,
    pub momentum: f64,
    pub energy: f64,
    pub star_pressure: f64,
    /// HLLC contact speed or the fallback solver's signal speed.
    pub solver_speed: f64,
    pub method: RiemannMethod,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairFlux1d {
    pub mass: f64,
    pub momentum: f64,
    pub energy: f64,
    pub star_pressure: f64,
    pub interface_velocity: f64,
    pub solver_speed: f64,
    pub method: RiemannMethod,
    pub solve_path: PairSolvePath,
    pub closure_leak: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairSolvePath {
    Reconstructed,
    Centered,
    ZeroRelativeVelocity,
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
    InvalidRiemannState {
        side: &'static str,
        field: &'static str,
        value: f64,
    },
    InvalidRiemannParameter {
        field: &'static str,
        value: f64,
    },
    NonFiniteRiemannResult {
        field: &'static str,
        value: f64,
    },
    ExactRiemannSolverRequired {
        star_pressure: f64,
        pressure_limit: f64,
    },
    ExactRiemannSolverDidNotConverge {
        iterations: u32,
        pressure: f64,
    },
}

impl fmt::Display for HydroError {
    #[allow(clippy::too_many_lines)]
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
            Self::InvalidRiemannState { side, field, value } => {
                write!(
                    formatter,
                    "Riemann {side} state has invalid {field} {value}"
                )
            }
            Self::InvalidRiemannParameter { field, value } => {
                write!(formatter, "invalid Riemann parameter {field}={value}")
            }
            Self::NonFiniteRiemannResult { field, value } => {
                write!(
                    formatter,
                    "Riemann solver produced non-finite {field}={value}"
                )
            }
            Self::ExactRiemannSolverRequired {
                star_pressure,
                pressure_limit,
            } => write!(
                formatter,
                "Riemann star pressure {star_pressure} is outside the accepted range \
                 (0, {pressure_limit}]; \
                 exact Riemann solver is required"
            ),
            Self::ExactRiemannSolverDidNotConverge {
                iterations,
                pressure,
            } => write!(
                formatter,
                "exact Riemann solver did not converge after {iterations} iterations; \
                 last pressure={pressure}"
            ),
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
        for closure_error in face_closure_errors_1d(&positions, &hsml, 1.0).unwrap() {
            assert_close(closure_error, 0.0);
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

    fn soundwave_state(velocity: f64) -> PrimitiveState1d {
        PrimitiveState1d {
            density: 1.0,
            velocity,
            pressure: 0.6,
        }
    }

    #[test]
    fn mfm_hllc_flux_preserves_a_uniform_state() {
        let flux =
            ideal_gas_mfm_flux_1d(soundwave_state(0.0), soundwave_state(0.0), 5.0 / 3.0, 1.0)
                .unwrap();
        assert_eq!(flux.method, RiemannMethod::Hllc);
        assert_close(flux.mass, 0.0);
        assert_close(flux.momentum, 0.6);
        assert_close(flux.energy, 0.0);
        assert_close(flux.star_pressure, 0.6);
        assert_close(flux.solver_speed, 0.0);
    }

    #[test]
    fn mfm_hllc_flux_resolves_a_weak_symmetric_expansion() {
        let flux =
            ideal_gas_mfm_flux_1d(soundwave_state(-0.1), soundwave_state(0.1), 5.0 / 3.0, 1.0)
                .unwrap();
        assert_eq!(flux.method, RiemannMethod::Hllc);
        assert_close(flux.momentum, 0.5);
        assert_close(flux.energy, 0.0);
        assert_close(flux.star_pressure, 0.5);
        assert_close(flux.solver_speed, 0.0);
    }

    #[test]
    fn failed_hllc_estimates_fall_back_instead_of_declaring_false_vacuum() {
        let flux =
            ideal_gas_mfm_flux_1d(soundwave_state(-1.0), soundwave_state(1.0), 5.0 / 3.0, 1.0)
                .unwrap();
        assert_eq!(flux.method, RiemannMethod::KurganovTadmor);
        assert_close(flux.star_pressure, 0.6);

        let below_threshold = ideal_gas_mfm_flux_1d(
            soundwave_state(-2.999),
            soundwave_state(2.999),
            5.0 / 3.0,
            1.0,
        )
        .unwrap();
        assert_ne!(below_threshold.method, RiemannMethod::Vacuum);

        let vacuum =
            ideal_gas_mfm_flux_1d(soundwave_state(-3.1), soundwave_state(3.1), 5.0 / 3.0, 1.0)
                .unwrap();
        assert_eq!(vacuum.method, RiemannMethod::Vacuum);
        assert_close(vacuum.momentum, 1.0e-56);
        let exact_vacuum = ideal_gas_mfm_flux_1d(
            soundwave_state(-3.1),
            soundwave_state(3.1),
            5.0 / 3.0,
            1.0e-57,
        )
        .unwrap();
        assert_eq!(exact_vacuum.method, RiemannMethod::Exact);
        assert_close(exact_vacuum.star_pressure, 0.0);
        assert_close(exact_vacuum.momentum, 0.0);
    }

    #[test]
    fn kt_fallback_uses_the_lagrangian_mfm_flux_not_the_mfv_flux() {
        let left = PrimitiveState1d {
            density: 1.0,
            velocity: -1.0,
            pressure: 0.6,
        };
        let right = PrimitiveState1d {
            density: 2.0,
            velocity: 1.0,
            pressure: 0.8,
        };
        let flux = ideal_gas_mfm_flux_1d(left, right, 5.0 / 3.0, 1.0).unwrap();
        assert_eq!(flux.method, RiemannMethod::KurganovTadmor);
        assert!((flux.star_pressure - 2.0 / 3.0).abs() < 1.0e-12);
        assert!((flux.momentum - (-0.666_666_529_18)).abs() < 1.0e-10);
        assert!((flux.energy - 0.066_666_646_04).abs() < 1.0e-10);

        // The accidental MFV/MFM hybrid returned about -0.8 here.
        assert!((flux.momentum - (-0.8)).abs() > 0.1);
    }

    #[test]
    fn exact_mfm_solver_matches_reference_star_states() {
        let sod = ideal_gas_mfm_flux_1d(
            PrimitiveState1d {
                density: 1.0,
                velocity: 0.0,
                pressure: 1.0,
            },
            PrimitiveState1d {
                density: 0.125,
                velocity: 0.0,
                pressure: 0.1,
            },
            5.0 / 3.0,
            0.01,
        )
        .unwrap();
        assert_eq!(sod.method, RiemannMethod::Exact);
        assert!((sod.star_pressure - 0.293_945_187_666_017_85).abs() < 1.0e-12);
        assert!((sod.solver_speed - 0.841_194_852_168_808_3).abs() < 1.0e-12);
        assert!((sod.energy - 0.247_265_178_684_448_5).abs() < 1.0e-12);

        let expansion =
            ideal_gas_mfm_flux_1d(soundwave_state(-1.0), soundwave_state(1.0), 5.0 / 3.0, 0.05)
                .unwrap();
        assert_eq!(expansion.method, RiemannMethod::Exact);
        assert!((expansion.star_pressure - 0.079_012_345_679_012_3).abs() < 1.0e-12);
        assert_close(expansion.solver_speed, 0.0);
        assert_close(expansion.energy, 0.0);

        let compression =
            ideal_gas_mfm_flux_1d(soundwave_state(1.0), soundwave_state(-1.0), 5.0 / 3.0, 0.01)
                .unwrap();
        assert_eq!(compression.method, RiemannMethod::Exact);
        assert!((compression.star_pressure - 2.468_517_091_821_330_4).abs() < 1.0e-12);
        assert_close(compression.solver_speed, 0.0);
        assert_close(compression.energy, 0.0);

        let exact_boundary = ideal_gas_mfm_flux_1d(
            soundwave_state(-3.0),
            soundwave_state(3.0),
            5.0 / 3.0,
            1.0e-57,
        )
        .unwrap();
        assert_eq!(exact_boundary.method, RiemannMethod::Exact);
        assert_close(exact_boundary.star_pressure, 0.0);
    }

    fn reconstructed_point(state: PrimitiveState1d) -> ReconstructedPoint1d {
        ReconstructedPoint1d {
            primitive: state,
            density_gradient: 0.0,
            velocity_gradient: 0.0,
            pressure_gradient: 0.0,
            face_closure_error: 0.0,
        }
    }

    fn unit_pair_face(normal: f64) -> MeshlessFace1d {
        MeshlessFace1d {
            signed_area: normal,
            area: 1.0,
            distance_from_i: -0.5 * normal,
            distance_from_j: 0.5 * normal,
        }
    }

    #[test]
    fn pair_flux_is_oriented_conservative_and_galilean_deboosted() {
        let state = PrimitiveState1d {
            density: 1.0,
            velocity: 2.0,
            pressure: 0.6,
        };
        let point = reconstructed_point(state);
        let forward = mfm_pair_flux_1d(point, point, unit_pair_face(1.0), 5.0 / 3.0).unwrap();
        let reverse = mfm_pair_flux_1d(point, point, unit_pair_face(-1.0), 5.0 / 3.0).unwrap();

        assert_close(forward.mass, 0.0);
        assert_close(forward.momentum, 0.6);
        assert_close(forward.energy, 1.2);
        assert_close(forward.interface_velocity, 2.0);
        assert_eq!(forward.solve_path, PairSolvePath::Reconstructed);
        assert_close(reverse.momentum, -forward.momentum);
        assert_close(reverse.energy, -forward.energy);

        let double_area = mfm_pair_flux_1d(
            point,
            point,
            MeshlessFace1d {
                signed_area: 2.0,
                area: 2.0,
                ..unit_pair_face(1.0)
            },
            5.0 / 3.0,
        )
        .unwrap();
        assert_close(double_area.momentum, 2.0 * forward.momentum);
        assert_close(double_area.energy, 2.0 * forward.energy);

        let i = reconstructed_point(PrimitiveState1d {
            density: 1.0,
            velocity: 0.2,
            pressure: 0.7,
        });
        let j = reconstructed_point(PrimitiveState1d {
            density: 0.8,
            velocity: -0.1,
            pressure: 0.5,
        });
        let asymmetric = mfm_pair_flux_1d(i, j, unit_pair_face(1.0), 5.0 / 3.0).unwrap();
        let swapped = mfm_pair_flux_1d(j, i, unit_pair_face(-1.0), 5.0 / 3.0).unwrap();
        assert!((asymmetric.momentum + swapped.momentum).abs() < 1.0e-14);
        assert!((asymmetric.energy + swapped.energy).abs() < 1.0e-14);
        assert_close(asymmetric.star_pressure, swapped.star_pressure);

        let gradient_i = ReconstructedPoint1d {
            density_gradient: 0.2,
            velocity_gradient: -0.3,
            pressure_gradient: 0.1,
            ..i
        };
        let gradient_j = ReconstructedPoint1d {
            density_gradient: -0.1,
            velocity_gradient: 0.2,
            pressure_gradient: -0.15,
            ..j
        };
        let gradient_flux =
            mfm_pair_flux_1d(gradient_i, gradient_j, unit_pair_face(1.0), 5.0 / 3.0).unwrap();
        let gradient_swap =
            mfm_pair_flux_1d(gradient_j, gradient_i, unit_pair_face(-1.0), 5.0 / 3.0).unwrap();
        assert!((gradient_flux.momentum + gradient_swap.momentum).abs() < 1.0e-14);
        assert!((gradient_flux.energy + gradient_swap.energy).abs() < 1.0e-14);

        let boost = |mut point: ReconstructedPoint1d| {
            point.primitive.velocity += 3.0;
            point
        };
        let boosted = mfm_pair_flux_1d(
            boost(gradient_i),
            boost(gradient_j),
            unit_pair_face(1.0),
            5.0 / 3.0,
        )
        .unwrap();
        assert_close(boosted.momentum, gradient_flux.momentum);
        assert!(
            (boosted.energy - (gradient_flux.energy + 3.0 * gradient_flux.momentum)).abs()
                < 1.0e-14
        );
    }

    #[test]
    fn pair_flux_rejects_inconsistent_face_geometry() {
        let point = reconstructed_point(soundwave_state(0.0));
        let invalid_face = MeshlessFace1d {
            signed_area: 1.0,
            area: 2.0,
            distance_from_i: -0.5,
            distance_from_j: 0.5,
        };
        assert!(mfm_pair_flux_1d(point, point, invalid_face, 5.0 / 3.0).is_err());

        let off_center = MeshlessFace1d {
            distance_from_i: -0.25,
            distance_from_j: 0.5,
            ..unit_pair_face(1.0)
        };
        assert!(mfm_pair_flux_1d(point, point, off_center, 5.0 / 3.0).is_err());
    }

    #[test]
    fn pair_flux_retries_centered_states_after_bad_reconstruction() {
        let i = ReconstructedPoint1d {
            pressure_gradient: f64::MAX,
            ..reconstructed_point(PrimitiveState1d {
                density: 1.0,
                velocity: 0.0,
                pressure: 0.6,
            })
        };
        let j = ReconstructedPoint1d {
            pressure_gradient: -f64::MAX,
            ..reconstructed_point(PrimitiveState1d {
                density: 1.0,
                velocity: 0.0,
                pressure: 0.5,
            })
        };
        let retry_face = MeshlessFace1d {
            distance_from_i: -2.0,
            distance_from_j: 2.0,
            ..unit_pair_face(1.0)
        };
        let flux = mfm_pair_flux_1d(i, j, retry_face, 5.0 / 3.0).unwrap();
        assert_eq!(flux.solve_path, PairSolvePath::Centered);
        assert!(flux.star_pressure.is_finite());
    }

    #[test]
    fn pair_flux_disables_reconstruction_for_excessive_face_leak() {
        let i = ReconstructedPoint1d {
            pressure_gradient: 0.2,
            face_closure_error: 2.1,
            ..reconstructed_point(PrimitiveState1d {
                density: 1.0,
                velocity: 0.1,
                pressure: 0.6,
            })
        };
        let j = ReconstructedPoint1d {
            pressure_gradient: -0.2,
            face_closure_error: 0.1,
            ..reconstructed_point(PrimitiveState1d {
                density: 0.9,
                velocity: -0.1,
                pressure: 0.5,
            })
        };
        let flux = mfm_pair_flux_1d(i, j, unit_pair_face(1.0), 5.0 / 3.0).unwrap();
        assert_eq!(flux.solve_path, PairSolvePath::Centered);
        assert_close(flux.closure_leak, 1.1);

        let mut entropic_i = entropic_point(i.primitive.velocity);
        entropic_i.face_closure_error = i.face_closure_error;
        let mut entropic_j = entropic_point(j.primitive.velocity);
        entropic_j.face_closure_error = j.face_closure_error;
        let (_, selected) =
            apply_entropic_pdv_1d(flux, unit_pair_face(1.0), entropic_i, entropic_j).unwrap();
        assert!(selected);
    }

    fn entropic_point(velocity: f64) -> EntropicPoint1d {
        EntropicPoint1d {
            velocity,
            density: 1.0,
            pressure: 0.6,
            sound_speed: 1.0,
            volume: 1.0,
            dhsml_factor: 1.0,
            kernel_radial_derivative: -1.0,
            condition_number: 1.0,
            face_closure_error: 0.0,
        }
    }

    fn entropic_flux(solver_speed: f64, energy: f64) -> PairFlux1d {
        PairFlux1d {
            mass: 0.0,
            momentum: 0.6,
            energy,
            star_pressure: 0.6,
            interface_velocity: 0.0,
            solver_speed,
            method: RiemannMethod::Hllc,
            solve_path: PairSolvePath::Reconstructed,
            closure_leak: 0.0,
        }
    }

    #[test]
    fn entropic_pdv_uniform_state_is_conservative_and_swap_antisymmetric() {
        let i = entropic_point(0.2);
        let j = EntropicPoint1d {
            velocity: -0.1,
            pressure: 0.5,
            kernel_radial_derivative: -2.0,
            ..entropic_point(-0.1)
        };
        let raw = PairFlux1d {
            interface_velocity: 0.05,
            ..entropic_flux(0.0, 0.03)
        };
        let (corrected, selected) = apply_entropic_pdv_1d(raw, unit_pair_face(1.0), i, j).unwrap();
        assert!(selected);

        let reverse_raw = PairFlux1d {
            momentum: -raw.momentum,
            energy: -raw.energy,
            solver_speed: -raw.solver_speed,
            ..raw
        };
        let (reverse, reverse_selected) =
            apply_entropic_pdv_1d(reverse_raw, unit_pair_face(-1.0), j, i).unwrap();
        assert!(reverse_selected);
        assert_close(reverse.momentum, -corrected.momentum);
        assert_close(reverse.energy, -corrected.energy);
        assert_close(corrected.energy + reverse.energy, 0.0);

        let boost = 3.0;
        let boosted_i = EntropicPoint1d {
            velocity: i.velocity + boost,
            ..i
        };
        let boosted_j = EntropicPoint1d {
            velocity: j.velocity + boost,
            ..j
        };
        let boosted_raw = PairFlux1d {
            energy: raw.energy + boost * raw.momentum,
            interface_velocity: raw.interface_velocity + boost,
            ..raw
        };
        let (boosted, boosted_selected) =
            apply_entropic_pdv_1d(boosted_raw, unit_pair_face(1.0), boosted_i, boosted_j).unwrap();
        assert!(boosted_selected);
        assert_close(
            boosted.energy,
            corrected.energy + boost * corrected.momentum,
        );

        let uniform = entropic_point(2.0);
        let uniform_raw = PairFlux1d {
            interface_velocity: 2.0,
            ..entropic_flux(0.0, 1.2)
        };
        let (uniform_corrected, uniform_selected) =
            apply_entropic_pdv_1d(uniform_raw, unit_pair_face(1.0), uniform, uniform).unwrap();
        assert!(uniform_selected);
        assert_close(uniform_corrected.energy, uniform_raw.energy);
    }

    #[test]
    fn entropic_pdv_preserves_strict_legacy_thresholds() {
        let mut i = entropic_point(0.1);
        i.pressure = 0.8;
        i.kernel_radial_derivative = -1.0;
        let mut j = entropic_point(0.0);
        j.pressure = 0.4;
        j.kernel_radial_derivative = -2.0;
        let raw = PairFlux1d {
            interface_velocity: 0.05,
            ..entropic_flux(0.5, 7.0)
        };

        let (_, at_big_threshold) = apply_entropic_pdv_1d(raw, unit_pair_face(1.0), i, j).unwrap();
        assert!(!at_big_threshold);
        let just_below_half = f64::from_bits(0.5_f64.to_bits() - 1);
        let (_, below_big_threshold) = apply_entropic_pdv_1d(
            PairFlux1d {
                interface_velocity: 0.05,
                ..entropic_flux(just_below_half, 7.0)
            },
            unit_pair_face(1.0),
            i,
            j,
        )
        .unwrap();
        assert!(below_big_threshold);

        let (_, at_small_threshold) = apply_entropic_pdv_1d(
            PairFlux1d {
                interface_velocity: 0.05,
                ..entropic_flux(1.0e-3, 7.0)
            },
            unit_pair_face(1.0),
            i,
            j,
        )
        .unwrap();
        assert!(at_small_threshold);
        let just_above_small = f64::from_bits(1.0e-3_f64.to_bits() + 1);
        let (_, above_small_threshold) = apply_entropic_pdv_1d(
            PairFlux1d {
                interface_velocity: 0.05,
                ..entropic_flux(just_above_small, 7.0)
            },
            unit_pair_face(1.0),
            i,
            j,
        )
        .unwrap();
        assert!(!above_small_threshold);
    }

    #[test]
    fn entropic_pdv_condition_boundary_forces_selection_and_kt_uses_delta() {
        let mut i = entropic_point(0.1);
        i.pressure = 0.8;
        i.condition_number = 0.0;
        let mut j = entropic_point(0.0);
        j.pressure = 0.4;
        j.kernel_radial_derivative = -2.0;
        j.condition_number = 1000.0;
        let mut raw = PairFlux1d {
            interface_velocity: 0.05,
            ..entropic_flux(0.01, 42.0)
        };
        raw.method = RiemannMethod::KurganovTadmor;

        let (corrected, selected) = apply_entropic_pdv_1d(raw, unit_pair_face(1.0), i, j).unwrap();
        assert!(selected);
        let pressure_area = raw.star_pressure;
        let relative_velocity = i.velocity - j.velocity;
        let pdv_i = i.kernel_radial_derivative * relative_velocity * raw.star_pressure;
        let pdv_j = j.kernel_radial_derivative * relative_velocity * raw.star_pressure;
        let old_energy = pressure_area * (raw.solver_speed + 0.5 * (i.velocity + j.velocity));
        let new_energy = 0.5 * (pdv_i - pdv_j + pressure_area * (i.velocity + j.velocity));
        assert_close(corrected.energy, raw.energy + new_energy - old_energy);
        assert!((corrected.energy - new_energy).abs() > 1.0);
    }

    #[test]
    fn entropic_pdv_rejects_invalid_inputs() {
        let invalid = EntropicPoint1d {
            sound_speed: 0.0,
            ..entropic_point(0.0)
        };
        assert!(
            apply_entropic_pdv_1d(
                entropic_flux(0.0, 0.0),
                unit_pair_face(1.0),
                invalid,
                entropic_point(0.0)
            )
            .is_err()
        );
    }

    #[test]
    fn mfm_riemann_solver_fails_closed_for_invalid_or_nonfinite_cases() {
        let invalid_density = PrimitiveState1d {
            density: 0.0,
            ..soundwave_state(0.0)
        };
        assert!(
            ideal_gas_mfm_flux_1d(invalid_density, soundwave_state(0.0), 5.0 / 3.0, 1.0).is_err()
        );
        assert!(
            ideal_gas_mfm_flux_1d(soundwave_state(0.0), soundwave_state(0.0), 1.0, 1.0).is_err()
        );
        let exact_uniform =
            ideal_gas_mfm_flux_1d(soundwave_state(0.0), soundwave_state(0.0), 5.0 / 3.0, 0.5)
                .unwrap();
        assert_eq!(exact_uniform.method, RiemannMethod::Exact);
        assert_close(exact_uniform.star_pressure, 0.6);
        assert!(
            ideal_gas_mfm_flux_1d(
                PrimitiveState1d {
                    density: f64::MAX,
                    velocity: f64::MAX,
                    pressure: f64::MAX,
                },
                soundwave_state(0.0),
                5.0 / 3.0,
                f64::MAX
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
