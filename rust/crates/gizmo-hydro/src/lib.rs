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
}
