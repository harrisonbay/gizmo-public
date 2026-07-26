#![forbid(unsafe_code)]

pub mod grain;
pub mod hydro_evolution_2d;
pub mod individual_timeline;
pub mod meshless_2d;
pub mod mhd;
pub mod mhd_2d;
pub mod mhd_evolution;
pub mod mhd_evolution_2d;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// Normalization of GIZMO's default cubic spline in one dimension.
pub const CUBIC_1D_NORMALIZATION: f64 = 4.0 / 3.0;
/// Core-radius factor of GIZMO's default cubic spline kernel.
pub const CUBIC_KERNEL_CORE_SIZE: f64 = 0.5;
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

/// Boundary topology used by the one-dimensional hydro operators.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BoundaryMode1d {
    #[default]
    Periodic,
    Reflective,
}

/// Signed displacement `a - b` for the selected boundary topology.
///
/// # Errors
///
/// Returns an error for non-finite, out-of-domain, or invalid-box inputs.
pub fn displacement_1d(
    a: f64,
    b: f64,
    box_size: f64,
    boundary: BoundaryMode1d,
) -> Result<f64, HydroError> {
    let upper_inclusive = boundary == BoundaryMode1d::Reflective;
    if !a.is_finite()
        || !b.is_finite()
        || !box_size.is_finite()
        || box_size <= 0.0
        || a < 0.0
        || b < 0.0
        || if upper_inclusive {
            a > box_size || b > box_size
        } else {
            a >= box_size || b >= box_size
        }
    {
        return Err(HydroError::InvalidPeriodicInput { a, b, box_size });
    }
    let mut displacement = a - b;
    if boundary == BoundaryMode1d::Periodic {
        if displacement > 0.5 * box_size {
            displacement -= box_size;
        }
        if displacement < -0.5 * box_size {
            displacement += box_size;
        }
    }
    Ok(displacement)
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
    displacement_1d(a, b, box_size, BoundaryMode1d::Periodic)
}

/// Reusable exact spatial index for periodic target-kernel queries.
///
/// Query results are sorted by original particle index, not position. Every
/// kernel accumulator therefore retains the arithmetic order of the legacy
/// all-particles scan while omitting compact-support zeroes.
struct NeighborIndex1d {
    sorted: Vec<(f64, usize)>,
    box_size: f64,
    boundary: BoundaryMode1d,
}

impl NeighborIndex1d {
    fn new(positions: &[f64], box_size: f64, boundary: BoundaryMode1d) -> Self {
        let mut sorted: Vec<(f64, usize)> = positions
            .iter()
            .copied()
            .enumerate()
            .map(|(index, position)| (position, index))
            .collect();
        sorted.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        Self {
            sorted,
            box_size,
            boundary,
        }
    }

    fn query(&self, position: f64, support: f64, output: &mut Vec<usize>) {
        output.clear();
        if self.boundary == BoundaryMode1d::Reflective {
            append_sorted_position_range(
                &self.sorted,
                (position - support).max(0.0),
                (position + support).min(self.box_size),
                output,
            );
        } else if support >= 0.5 * self.box_size {
            output.extend(0..self.sorted.len());
        } else {
            let lower = position - support;
            let upper = position + support;
            if lower < 0.0 {
                append_sorted_position_range(&self.sorted, 0.0, upper, output);
                append_sorted_position_range(
                    &self.sorted,
                    lower + self.box_size,
                    self.box_size,
                    output,
                );
            } else if upper >= self.box_size {
                append_sorted_position_range(&self.sorted, lower, self.box_size, output);
                append_sorted_position_range(&self.sorted, 0.0, upper - self.box_size, output);
            } else {
                append_sorted_position_range(&self.sorted, lower, upper, output);
            }
        }
        output.sort_unstable();
        output.dedup();
    }
}

/// Enumerate exact unordered interactions in legacy particle-index order.
///
/// The sorted position index queries each particle's own compact support.
/// Canonicalizing those directed neighborhoods applies the meshless
/// union-support rule, `r < H_i || r < H_j`, without using a global search
/// radius. The returned order is identical to the former nested `for i`/`for
/// j` scan, preserving floating-point force accumulation.
///
/// This costs `O(N log N + sum(k_i log k_i) + P log P)`, where `k_i` is the
/// number of particles inside `H_i` and `P` is the interacting-pair count,
/// instead of `O(N²)` for locally bounded support.
fn interacting_pairs_1d(
    positions: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
    boundary: BoundaryMode1d,
) -> Result<Vec<(usize, usize, f64)>, HydroError> {
    let neighbor_index = NeighborIndex1d::new(positions, box_size, boundary);
    let mut pair_indices = Vec::new();
    let mut candidates = Vec::new();
    for (source, (&position, &support)) in positions.iter().zip(smoothing_lengths).enumerate() {
        neighbor_index.query(position, support, &mut candidates);
        for &neighbor in &candidates {
            if neighbor == source {
                continue;
            }
            let distance =
                displacement_1d(position, positions[neighbor], box_size, boundary)?.abs();
            if distance > 0.0 && distance < support {
                pair_indices.push(if source < neighbor {
                    (source, neighbor)
                } else {
                    (neighbor, source)
                });
            }
        }
    }
    pair_indices.sort_unstable();
    pair_indices.dedup();
    pair_indices
        .into_iter()
        .map(|(i, j)| {
            displacement_1d(positions[i], positions[j], box_size, boundary)
                .map(|displacement| (i, j, displacement))
        })
        .collect()
}

#[cfg(test)]
fn interacting_pairs_periodic_1d(
    positions: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) -> Result<Vec<(usize, usize, f64)>, HydroError> {
    interacting_pairs_1d(
        positions,
        smoothing_lengths,
        box_size,
        BoundaryMode1d::Periodic,
    )
}

fn append_sorted_position_range(
    sorted: &[(f64, usize)],
    lower: f64,
    upper: f64,
    output: &mut Vec<usize>,
) {
    let start = sorted.partition_point(|&(position, _)| position < lower);
    let end = sorted.partition_point(|&(position, _)| position <= upper);
    output.extend(sorted[start..end].iter().map(|&(_, index)| index));
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
    density_at_hsml_1d_with_boundary(
        positions,
        masses,
        smoothing_lengths,
        box_size,
        BoundaryMode1d::Periodic,
    )
}

/// Boundary-aware density evaluation.
///
/// # Errors
///
/// Returns an error under the same conditions as [`density_at_hsml_1d`].
pub fn density_at_hsml_1d_with_boundary(
    positions: &[f64],
    masses: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
    boundary: BoundaryMode1d,
) -> Result<Vec<DensityEstimate>, HydroError> {
    validate_particle_columns_with_boundary(
        positions,
        masses,
        smoothing_lengths,
        box_size,
        boundary,
    )?;
    let neighbor_index = NeighborIndex1d::new(positions, box_size, boundary);
    let mut output = Vec::with_capacity(positions.len());
    for (index, &hsml) in smoothing_lengths.iter().enumerate() {
        output.push(estimate_particle(
            index,
            positions,
            masses,
            hsml,
            box_size,
            &neighbor_index,
        )?);
    }
    Ok(output)
}

/// Estimate the particle-trajectory velocity divergence used to predict `Hsml`.
///
/// This is the one-dimensional specialization of the density-loop
/// `Particle_DivVel` estimator. Unlike the MLS velocity gradient, this
/// kernel-gradient estimate controls smoothing-length drift between force
/// evaluations.
///
/// # Errors
///
/// Returns an error for mismatched or invalid columns, invalid kernel
/// arithmetic, or a non-finite divergence.
pub fn particle_divergence_at_hsml_1d(
    positions: &[f64],
    velocities: &[f64],
    masses: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) -> Result<Vec<f64>, HydroError> {
    particle_divergence_at_hsml_1d_with_boundary(
        positions,
        velocities,
        masses,
        smoothing_lengths,
        box_size,
        BoundaryMode1d::Periodic,
    )
}

/// Boundary-aware particle-divergence evaluation.
///
/// # Errors
///
/// Returns an error under the same conditions as [`particle_divergence_at_hsml_1d`].
pub fn particle_divergence_at_hsml_1d_with_boundary(
    positions: &[f64],
    velocities: &[f64],
    masses: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
    boundary: BoundaryMode1d,
) -> Result<Vec<f64>, HydroError> {
    validate_particle_columns_with_boundary(
        positions,
        masses,
        smoothing_lengths,
        box_size,
        boundary,
    )?;
    if velocities.len() != positions.len() {
        return Err(HydroError::MismatchedLength {
            field: "velocities",
            expected: positions.len(),
            actual: velocities.len(),
        });
    }
    for (index, &velocity) in velocities.iter().enumerate() {
        if !velocity.is_finite() {
            return Err(HydroError::InvalidParticle {
                index,
                field: "velocity",
                value: velocity,
            });
        }
    }

    let density =
        density_at_hsml_1d_with_boundary(positions, masses, smoothing_lengths, box_size, boundary)?;
    let neighbor_index = NeighborIndex1d::new(positions, box_size, boundary);
    let mut neighbors = Vec::new();
    let mut output = Vec::with_capacity(positions.len());
    for (index, ((&position, &velocity), &hsml)) in positions
        .iter()
        .zip(velocities)
        .zip(smoothing_lengths)
        .enumerate()
    {
        let mut kernel_sum = 0.0;
        let mut divergence_numerator = 0.0;
        neighbor_index.query(position, hsml, &mut neighbors);
        for &neighbor in &neighbors {
            let neighbor_position = positions[neighbor];
            let neighbor_velocity = velocities[neighbor];
            let displacement = displacement_1d(position, neighbor_position, box_size, boundary)?;
            let distance = displacement.abs();
            let kernel = cubic_kernel_1d(distance, hsml)?;
            kernel_sum += kernel.weight;
            if distance > 0.0 && distance < hsml {
                let velocity_difference = velocity - neighbor_velocity;
                divergence_numerator -=
                    kernel.radial_derivative * displacement * velocity_difference / distance;
            }
        }
        let divergence = divergence_numerator * density[index].dhsml_factor / kernel_sum;
        if !kernel_sum.is_finite()
            || kernel_sum <= 0.0
            || !divergence_numerator.is_finite()
            || !divergence.is_finite()
        {
            return Err(HydroError::NonFiniteDensityEstimate {
                index,
                field: "particle_divergence",
                value: divergence,
            });
        }
        output.push(divergence);
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
    solve_smoothing_lengths_1d_with_boundary(
        positions,
        masses,
        initial_smoothing_lengths,
        box_size,
        desired_neighbors,
        tolerance,
        BoundaryMode1d::Periodic,
    )
}

#[allow(clippy::too_many_arguments)]
/// Boundary-aware adaptive smoothing-length solve.
///
/// # Errors
///
/// Returns an error under the same conditions as [`solve_smoothing_lengths_1d`].
pub fn solve_smoothing_lengths_1d_with_boundary(
    positions: &[f64],
    masses: &[f64],
    initial_smoothing_lengths: &[f64],
    box_size: f64,
    desired_neighbors: f64,
    tolerance: f64,
    boundary: BoundaryMode1d,
) -> Result<Vec<AdaptiveDensityEstimate>, HydroError> {
    validate_particle_columns_with_boundary(
        positions,
        masses,
        initial_smoothing_lengths,
        box_size,
        boundary,
    )?;
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

    let neighbor_index = NeighborIndex1d::new(positions, box_size, boundary);
    let mut output = Vec::with_capacity(positions.len());
    for (index, &initial_hsml) in initial_smoothing_lengths.iter().enumerate() {
        let mut hsml = initial_hsml;
        let mut lower: Option<f64> = None;
        let mut upper: Option<f64> = None;
        let mut last_estimate =
            estimate_particle(index, positions, masses, hsml, box_size, &neighbor_index)?;
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
            last_estimate =
                estimate_particle(index, positions, masses, hsml, box_size, &neighbor_index)?;
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

/// Reproduce the public C soundwave's gravitational-tree smoothing-length seeds.
///
/// `setup_smoothinglengths()` does not start its density iteration from a
/// uniform kernel size. It ascends the 42-bit gravitational oct-tree from each
/// particle until the enclosing node contains at least
/// `2 * desired_neighbors * particle_mass`, then scales that node's length by
/// the particle's share of its mass. In the pinned one-dimensional profile all
/// particles have the same transverse coordinates, so the oct-tree reduces
/// exactly to this dyadic mass tree.
///
/// The domain construction and bottom-up child accumulation deliberately
/// mirror `domain_findExtent()`, `domain_double_to_int()`, and
/// `force_update_node_recursive()` rather than using position-space searches.
/// This preserves the floating-point branches selected by the public C
/// initializer.
///
/// # Errors
///
/// Returns an error for invalid or mismatched particle columns, a degenerate
/// domain, or a non-finite seed.
pub fn public_c_tree_smoothing_length_seeds_1d(
    positions: &[f64],
    masses: &[f64],
    box_size: f64,
    desired_neighbors: f64,
) -> Result<Vec<f64>, HydroError> {
    public_c_tree_smoothing_length_seeds_1d_with_boundary(
        positions,
        masses,
        box_size,
        desired_neighbors,
        BoundaryMode1d::Periodic,
    )
}

/// Boundary-aware public-C tree seed construction.
///
/// # Errors
///
/// Returns an error under the same conditions as [`public_c_tree_smoothing_length_seeds_1d`].
pub fn public_c_tree_smoothing_length_seeds_1d_with_boundary(
    positions: &[f64],
    masses: &[f64],
    box_size: f64,
    desired_neighbors: f64,
    boundary: BoundaryMode1d,
) -> Result<Vec<f64>, HydroError> {
    const TREE_BITS: u32 = 42;

    if positions.len() != masses.len() {
        return Err(HydroError::MismatchedLength {
            field: "masses",
            expected: positions.len(),
            actual: masses.len(),
        });
    }
    if !box_size.is_finite() || box_size <= 0.0 {
        return Err(HydroError::InvalidBoxSize(box_size));
    }
    if !desired_neighbors.is_finite() || desired_neighbors <= 0.0 {
        return Err(HydroError::InvalidNeighborConstraint {
            desired: desired_neighbors,
            tolerance: 0.0,
        });
    }
    for (index, (&position, &mass)) in positions.iter().zip(masses).enumerate() {
        for (field, value, positive) in [("position", position, false), ("mass", mass, true)] {
            if !value.is_finite()
                || (positive && value <= 0.0)
                || (field == "position"
                    && (value < 0.0
                        || if boundary == BoundaryMode1d::Reflective {
                            value > box_size
                        } else {
                            value >= box_size
                        }))
            {
                return Err(HydroError::InvalidParticle {
                    index,
                    field,
                    value,
                });
            }
        }
    }
    let (&minimum, &maximum) = positions
        .iter()
        .min_by(|left, right| left.total_cmp(right))
        .zip(positions.iter().max_by(|left, right| left.total_cmp(right)))
        .ok_or(HydroError::DegenerateTreeDomain {
            minimum: f64::NAN,
            maximum: f64::NAN,
        })?;
    let domain_length = 1.001 * (maximum - minimum);
    if !domain_length.is_finite() || domain_length <= 0.0 {
        return Err(HydroError::DegenerateTreeDomain { minimum, maximum });
    }
    let domain_corner = 0.5 * (minimum + maximum) - 0.5 * domain_length;
    let keys: Vec<u64> = positions
        .iter()
        .map(|&position| legacy_tree_coordinate(position, domain_corner, domain_length, TREE_BITS))
        .collect();

    let mut levels = vec![BTreeMap::<u64, f64>::new(); (TREE_BITS + 1) as usize];
    for (&key, &mass) in keys.iter().zip(masses) {
        *levels[TREE_BITS as usize].entry(key).or_insert(0.0) += mass;
    }
    for depth in (0..TREE_BITS).rev() {
        let (parents, children) = levels.split_at_mut((depth + 1) as usize);
        let parent_level = &mut parents[depth as usize];
        for (&child, &mass) in &children[0] {
            *parent_level.entry(child >> 1).or_insert(0.0) += mass;
        }
    }

    keys.iter()
        .zip(masses)
        .enumerate()
        .map(|(index, (&key, &particle_mass))| {
            let threshold = 2.0 * desired_neighbors * particle_mass;
            let mut selected_depth = 0;
            let mut selected_mass = levels[0][&0];
            for depth in (0..=TREE_BITS).rev() {
                let prefix = key >> (TREE_BITS - depth);
                let node_mass = levels[depth as usize][&prefix];
                selected_depth = depth;
                selected_mass = node_mass;
                if node_mass >= threshold {
                    break;
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let node_length = domain_length / (1_u64 << selected_depth) as f64;
            let seed = desired_neighbors * (particle_mass / selected_mass) * node_length;
            if !seed.is_finite() || seed <= 0.0 {
                return Err(HydroError::NonFiniteDensityEstimate {
                    index,
                    field: "tree_smoothing_length_seed",
                    value: seed,
                });
            }
            Ok(seed)
        })
        .collect()
}

/// Run the public C restart-0 smoothing-length iteration from its tree seeds.
///
/// This is the `NUMDIMS == 1` specialization of the bracket and Newton-jump
/// schedule in `density()`, including its gradually relaxed tolerance and
/// narrow-bracket acceptance rule. Unlike [`solve_smoothing_lengths_1d`],
/// which computes a canonical tightly converged root, this routine preserves
/// the accepted floating-point branch used by public-C restart-0 snapshots.
///
/// # Errors
///
/// Returns an error for invalid state or constraints, or if the legacy
/// iteration does not converge.
pub fn solve_public_c_initial_smoothing_lengths_1d(
    positions: &[f64],
    masses: &[f64],
    box_size: f64,
    desired_neighbors: f64,
    tolerance: f64,
) -> Result<Vec<AdaptiveDensityEstimate>, HydroError> {
    solve_public_c_initial_smoothing_lengths_1d_with_boundary(
        positions,
        masses,
        box_size,
        desired_neighbors,
        tolerance,
        BoundaryMode1d::Periodic,
    )
}

/// Boundary-aware public-C initial smoothing-length solve.
///
/// # Errors
///
/// Returns an error under the same conditions as
/// [`solve_public_c_initial_smoothing_lengths_1d`].
pub fn solve_public_c_initial_smoothing_lengths_1d_with_boundary(
    positions: &[f64],
    masses: &[f64],
    box_size: f64,
    desired_neighbors: f64,
    tolerance: f64,
    boundary: BoundaryMode1d,
) -> Result<Vec<AdaptiveDensityEstimate>, HydroError> {
    let mut seeds = public_c_tree_smoothing_length_seeds_1d_with_boundary(
        positions,
        masses,
        box_size,
        desired_neighbors,
        boundary,
    )?;
    let mut solved = Vec::new();
    // Restart-0 evaluates density once inside `setup_smoothinglengths()`, once
    // again while completing `init()`, and once in the initial force
    // evaluation before the visible t=0 drift snapshot. Each call resets its
    // brackets but retains the previously accepted Hsml.
    for _ in 0..3 {
        solved = solve_public_c_smoothing_lengths_from_seeds_1d_with_boundary(
            positions,
            masses,
            &seeds,
            box_size,
            desired_neighbors,
            tolerance,
            boundary,
        )?;
        seeds = solved
            .iter()
            .map(|particle| particle.smoothing_length)
            .collect();
    }
    Ok(solved)
}

/// Run one public-C density/Hsml iteration pass from caller-supplied seeds.
///
/// This is the pass used at force endpoints after the restart-0 initialization
/// sequence. Bounds are reset for each pass, while accepted input Hsml values
/// are retained exactly when they already satisfy the corrected constraint.
///
/// # Errors
///
/// Returns an error for invalid state or constraints, or if the legacy
/// iteration does not converge.
pub fn solve_public_c_smoothing_lengths_from_seeds_1d(
    positions: &[f64],
    masses: &[f64],
    seeds: &[f64],
    box_size: f64,
    desired_neighbors: f64,
    tolerance: f64,
) -> Result<Vec<AdaptiveDensityEstimate>, HydroError> {
    solve_public_c_smoothing_lengths_from_seeds_1d_with_boundary(
        positions,
        masses,
        seeds,
        box_size,
        desired_neighbors,
        tolerance,
        BoundaryMode1d::Periodic,
    )
}

#[allow(clippy::too_many_arguments)]
/// Boundary-aware public-C smoothing-length pass.
///
/// # Errors
///
/// Returns an error under the same conditions as
/// [`solve_public_c_smoothing_lengths_from_seeds_1d`].
pub fn solve_public_c_smoothing_lengths_from_seeds_1d_with_boundary(
    positions: &[f64],
    masses: &[f64],
    seeds: &[f64],
    box_size: f64,
    desired_neighbors: f64,
    tolerance: f64,
    boundary: BoundaryMode1d,
) -> Result<Vec<AdaptiveDensityEstimate>, HydroError> {
    validate_particle_columns_with_boundary(positions, masses, seeds, box_size, boundary)?;
    if !tolerance.is_finite() || tolerance <= 0.0 || tolerance >= desired_neighbors {
        return Err(HydroError::InvalidNeighborConstraint {
            desired: desired_neighbors,
            tolerance,
        });
    }
    let neighbor_index = NeighborIndex1d::new(positions, box_size, boundary);
    seeds
        .iter()
        .copied()
        .enumerate()
        .map(|(index, seed)| {
            solve_public_c_particle_from_seed(
                index,
                positions,
                masses,
                box_size,
                desired_neighbors,
                tolerance,
                seed,
                &neighbor_index,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn solve_public_c_particle_from_seed(
    index: usize,
    positions: &[f64],
    masses: &[f64],
    box_size: f64,
    desired_neighbors: f64,
    base_tolerance: f64,
    seed: f64,
    neighbor_index: &NeighborIndex1d,
) -> Result<AdaptiveDensityEstimate, HydroError> {
    let mut hsml = seed;
    let mut lower = 0.0_f64;
    let mut upper = 0.0_f64;
    let mut last = estimate_public_c_density_geometry(
        index,
        positions,
        masses,
        hsml,
        box_size,
        neighbor_index,
    )?;

    for iteration in 0..=128 {
        let neighbor_correction = (last.face_closure_error / 0.35).clamp(1.0, 2.0);
        let corrected_desired_neighbors = desired_neighbors * neighbor_correction;
        let corrected_base_tolerance = base_tolerance * neighbor_correction;
        let tolerance = if iteration > 1 {
            let growth = (0.1
                * (corrected_desired_neighbors / (16.0 * corrected_base_tolerance)).ln()
                * f64::from(iteration))
            .exp();
            (corrected_base_tolerance * growth).min(0.25 * corrected_desired_neighbors)
        } else {
            corrected_base_tolerance
        };
        if (last.estimate.effective_neighbors - corrected_desired_neighbors).abs() <= tolerance {
            return Ok(AdaptiveDensityEstimate {
                smoothing_length: hsml,
                estimate: last.estimate,
            });
        }

        if lower > 0.0 && upper > 0.0 && upper - lower < 1.0e-3 * lower {
            return Ok(AdaptiveDensityEstimate {
                smoothing_length: hsml,
                estimate: last.estimate,
            });
        }
        if iteration == 128 {
            break;
        }

        if last.estimate.effective_neighbors < corrected_desired_neighbors - tolerance {
            lower = lower.max(hsml);
        } else if upper == 0.0 || hsml < upper {
            upper = hsml;
        }

        hsml = public_c_hsml_jump(
            hsml,
            last.estimate,
            corrected_desired_neighbors,
            iteration,
            lower,
            upper,
        );
        last = estimate_public_c_density_geometry(
            index,
            positions,
            masses,
            hsml,
            box_size,
            neighbor_index,
        )?;
    }

    Err(HydroError::SmoothingLengthDidNotConverge {
        index,
        lower: (lower > 0.0).then_some(lower),
        upper: (upper > 0.0).then_some(upper),
        effective_neighbors: last.estimate.effective_neighbors,
    })
}

struct PublicCDensityGeometry {
    estimate: DensityEstimate,
    face_closure_error: f64,
}

fn estimate_public_c_density_geometry(
    index: usize,
    positions: &[f64],
    masses: &[f64],
    hsml: f64,
    box_size: f64,
    neighbor_index: &NeighborIndex1d,
) -> Result<PublicCDensityGeometry, HydroError> {
    let position = positions[index];
    let mut kernel_sum = 0.0;
    let mut derivative_sum = 0.0;
    let mut second_moment = 0.0;
    let mut first_moment = 0.0;
    let mut neighbors = Vec::new();
    neighbor_index.query(position, hsml, &mut neighbors);
    for neighbor in neighbors {
        let displacement = displacement_1d(
            position,
            positions[neighbor],
            box_size,
            neighbor_index.boundary,
        )?;
        let radius = displacement.abs();
        let kernel = cubic_kernel_1d(radius, hsml)?;
        kernel_sum += kernel.weight;
        if radius < hsml {
            derivative_sum += -(kernel.weight / hsml + (radius / hsml) * kernel.radial_derivative);
        }
        if radius > 0.0 && radius < hsml {
            second_moment += kernel.weight * displacement * displacement;
            first_moment += kernel.weight * displacement;
        }
    }
    // Public MFM overwrites the raw neighbor-mass accumulator after the Hsml
    // loop with the particle-volume density `m_i * sum_j W_ij`.
    let particle_density = masses[index] * kernel_sum;
    let effective_neighbors = kernel_sum * 2.0 * hsml;
    let raw_derivative = derivative_sum * hsml / kernel_sum;
    let dhsml_factor = if raw_derivative > -0.9 {
        1.0 / (1.0 + raw_derivative)
    } else {
        1.0
    };
    let face_closure_error = (first_moment / second_moment / kernel_sum).abs();
    for (field, value) in [
        ("density", particle_density),
        ("effective_neighbors", effective_neighbors),
        ("dhsml_factor", dhsml_factor),
        ("face_closure_error", face_closure_error),
    ] {
        if !value.is_finite() {
            return Err(HydroError::NonFiniteDensityEstimate {
                index,
                field,
                value,
            });
        }
    }
    Ok(PublicCDensityGeometry {
        estimate: DensityEstimate {
            density: particle_density,
            effective_neighbors,
            dhsml_factor,
        },
        face_closure_error,
    })
}

#[allow(clippy::float_cmp)]
fn public_c_hsml_jump(
    mut hsml: f64,
    estimate: DensityEstimate,
    desired_neighbors: f64,
    iteration: u32,
    lower: f64,
    upper: f64,
) -> f64 {
    let neighbors = estimate.effective_neighbors;
    if lower > 0.0 && upper > 0.0 {
        let max_jump = if iteration > 1 {
            0.2 * (upper / lower).ln()
        } else {
            0.0
        };
        if neighbors > 1.0 {
            let mut jump = estimate.dhsml_factor * (desired_neighbors / neighbors).ln();
            if iteration > 1 && jump.abs() < max_jump {
                jump = max_jump.copysign(jump);
            }
            hsml *= jump.exp();
        } else {
            hsml *= 2.0;
        }
        if hsml < upper && hsml > lower {
            if iteration > 1 {
                let jump_factor = max_jump.exp();
                if hsml > upper / jump_factor {
                    hsml = upper / jump_factor;
                }
                if hsml < lower * jump_factor {
                    hsml = lower * jump_factor;
                }
            }
        } else {
            hsml = hsml.clamp(lower, upper);
            hsml = (hsml * lower * upper).powf(1.0 / 3.0);
        }
        return hsml;
    }

    let mut limited_log_jump = if neighbors > 1.0 {
        (desired_neighbors / neighbors).ln()
    } else {
        1.4
    };
    if upper == 0.0 {
        if neighbors < 2.0 * desired_neighbors && neighbors > 0.1 * desired_neighbors {
            let mut slope = estimate.dhsml_factor;
            if iteration > 2 && slope < 1.0 {
                slope = 0.5 * (slope + 1.0);
            }
            let mut jump = limited_log_jump * slope;
            if iteration >= 4 && estimate.dhsml_factor == 1.0 {
                jump *= 10.0;
            }
            jump = jump.min(limited_log_jump + 0.231);
            hsml *= jump.exp();
        } else {
            hsml *= limited_log_jump.exp();
        }
    } else {
        limited_log_jump = limited_log_jump.max(-1.535);
        if neighbors < 2.0 * desired_neighbors && neighbors > 0.1 * desired_neighbors {
            let mut slope = estimate.dhsml_factor;
            if iteration > 2 && slope < 1.0 {
                slope = 0.5 * (slope + 1.0);
            }
            let mut jump = limited_log_jump * slope;
            if iteration >= 4 && estimate.dhsml_factor == 1.0 {
                jump *= 10.0;
            }
            jump = jump.max(limited_log_jump - 0.231);
            hsml *= jump.exp();
        } else {
            hsml *= limited_log_jump.exp();
        }
    }
    hsml
}

fn legacy_tree_coordinate(position: f64, domain_corner: f64, domain_length: f64, bits: u32) -> u64 {
    const MANTISSA_MASK: u64 = (1_u64 << 52) - 1;
    let normalized = (position - domain_corner) / domain_length + 1.0;
    (normalized.to_bits() & MANTISSA_MASK) >> (52 - bits)
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
    inverse_moments_1d_with_boundary(
        positions,
        smoothing_lengths,
        box_size,
        BoundaryMode1d::Periodic,
    )
}

/// Boundary-aware inverse-moment construction.
///
/// # Errors
///
/// Returns an error under the same conditions as [`inverse_moments_1d`].
pub fn inverse_moments_1d_with_boundary(
    positions: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
    boundary: BoundaryMode1d,
) -> Result<Vec<f64>, HydroError> {
    let unit_masses = vec![1.0; positions.len()];
    validate_particle_columns_with_boundary(
        positions,
        &unit_masses,
        smoothing_lengths,
        box_size,
        boundary,
    )?;
    let neighbor_index = NeighborIndex1d::new(positions, box_size, boundary);
    let mut neighbors = Vec::new();
    let mut output = Vec::with_capacity(positions.len());
    for (index, (&position, &hsml)) in positions.iter().zip(smoothing_lengths).enumerate() {
        let mut moment = 0.0;
        neighbor_index.query(position, hsml, &mut neighbors);
        for &neighbor in &neighbors {
            let neighbor_position = positions[neighbor];
            let displacement = displacement_1d(position, neighbor_position, box_size, boundary)?;
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
    face_closure_errors_1d_with_boundary(
        positions,
        smoothing_lengths,
        box_size,
        BoundaryMode1d::Periodic,
    )
}

/// Boundary-aware face-closure diagnostics.
///
/// # Errors
///
/// Returns an error under the same conditions as [`face_closure_errors_1d`].
pub fn face_closure_errors_1d_with_boundary(
    positions: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
    boundary: BoundaryMode1d,
) -> Result<Vec<f64>, HydroError> {
    let unit_masses = vec![1.0; positions.len()];
    validate_particle_columns_with_boundary(
        positions,
        &unit_masses,
        smoothing_lengths,
        box_size,
        boundary,
    )?;
    let inverse_moments =
        inverse_moments_1d_with_boundary(positions, smoothing_lengths, box_size, boundary)?;
    let neighbor_index = NeighborIndex1d::new(positions, box_size, boundary);
    let mut neighbors = Vec::new();
    let mut output = Vec::with_capacity(positions.len());
    for (index, (&position, &hsml)) in positions.iter().zip(smoothing_lengths).enumerate() {
        let mut kernel_sum = 0.0;
        let mut first_moment = 0.0;
        neighbor_index.query(position, hsml, &mut neighbors);
        for &neighbor in &neighbors {
            let neighbor_position = positions[neighbor];
            let displacement = displacement_1d(position, neighbor_position, box_size, boundary)?;
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
    gradients_at_hsml_1d_with_boundary(
        positions,
        values,
        smoothing_lengths,
        box_size,
        shoot_tolerance,
        positivity_preserving,
        BoundaryMode1d::Periodic,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
/// Boundary-aware MLS gradient construction.
///
/// # Errors
///
/// Returns an error under the same conditions as [`gradients_at_hsml_1d`].
pub fn gradients_at_hsml_1d_with_boundary(
    positions: &[f64],
    values: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
    shoot_tolerance: f64,
    positivity_preserving: bool,
    boundary: BoundaryMode1d,
) -> Result<Vec<GradientEstimate>, HydroError> {
    let unit_masses = vec![1.0; positions.len()];
    validate_particle_columns_with_boundary(
        positions,
        &unit_masses,
        smoothing_lengths,
        box_size,
        boundary,
    )?;
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
    let inverse_moments =
        inverse_moments_1d_with_boundary(positions, smoothing_lengths, box_size, boundary)?;
    let mut neighbor_lists = vec![Vec::new(); positions.len()];
    for (i, j, _) in interacting_pairs_1d(positions, smoothing_lengths, box_size, boundary)? {
        neighbor_lists[i].push(j);
        neighbor_lists[j].push(i);
    }
    for neighbors in &mut neighbor_lists {
        neighbors.sort_unstable();
    }

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
        for &neighbor in &neighbor_lists[index] {
            let neighbor_position = positions[neighbor];
            let neighbor_value = values[neighbor];
            let displacement = displacement_1d(position, neighbor_position, box_size, boundary)?;
            let distance = displacement.abs();
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
    meshless_face_geometry_1d_with_boundary(i, j, box_size, BoundaryMode1d::Periodic)
}

/// Boundary-aware meshless face construction.
///
/// # Errors
///
/// Returns an error under the same conditions as [`meshless_face_geometry_1d`].
pub fn meshless_face_geometry_1d_with_boundary(
    i: MeshlessPoint1d,
    j: MeshlessPoint1d,
    box_size: f64,
    boundary: BoundaryMode1d,
) -> Result<MeshlessFace1d, HydroError> {
    validate_meshless_point("i", i, box_size, boundary)?;
    validate_meshless_point("j", j, box_size, boundary)?;
    let displacement = displacement_1d(i.position, j.position, box_size, boundary)?;
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
    let momentum_matches_source = flux.method == RiemannMethod::KurganovTadmor
        || legacy_float_equal(flux.momentum, expected_momentum);
    if !legacy_float_equal(flux.mass, 0.0)
        || !legacy_float_equal(flux.interface_velocity, midpoint_velocity)
        || !momentum_matches_source
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

/// Evaluate the semidiscrete default 1-D MFM hydro operator.
///
/// Every interacting unordered pair is solved once. The returned momentum and
/// total-energy rates are extensive and exactly antisymmetric by construction;
/// acceleration and specific-internal-energy rates are the corresponding
/// primitive-variable derivatives used by the legacy predictor and kicks.
///
/// # Errors
///
/// Returns an error for mismatched or invalid state columns, invalid meshless
/// geometry, reconstruction failure, or a non-finite rate.
#[allow(clippy::too_many_lines)]
pub fn mfm_spatial_rates_1d(state: MfmState1d<'_>) -> Result<MfmRates1d, HydroError> {
    mfm_spatial_rates_1d_with_boundary(state, BoundaryMode1d::Periodic)
}

#[allow(clippy::too_many_lines)]
/// Boundary-aware semidiscrete MFM hydro operator.
///
/// # Errors
///
/// Returns an error under the same conditions as [`mfm_spatial_rates_1d`].
pub fn mfm_spatial_rates_1d_with_boundary(
    state: MfmState1d<'_>,
    boundary: BoundaryMode1d,
) -> Result<MfmRates1d, HydroError> {
    let particle_count = state.positions.len();
    for (field, actual) in [
        ("masses", state.masses.len()),
        ("velocities", state.velocities.len()),
        (
            "specific_internal_energy",
            state.specific_internal_energy.len(),
        ),
        ("smoothing_lengths", state.smoothing_lengths.len()),
    ] {
        if actual != particle_count {
            return Err(HydroError::MismatchedLength {
                field,
                expected: particle_count,
                actual,
            });
        }
    }
    validate_particle_columns_with_boundary(
        state.positions,
        state.masses,
        state.smoothing_lengths,
        state.box_size,
        boundary,
    )?;
    if !state.gamma.is_finite() || state.gamma <= 1.0 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "gamma",
            value: state.gamma,
        });
    }
    for (index, (&velocity, &internal_energy)) in state
        .velocities
        .iter()
        .zip(state.specific_internal_energy)
        .enumerate()
    {
        if !velocity.is_finite() {
            return Err(HydroError::InvalidParticle {
                index,
                field: "velocity",
                value: velocity,
            });
        }
        if !internal_energy.is_finite() || internal_energy <= 0.0 {
            return Err(HydroError::InvalidParticle {
                index,
                field: "specific_internal_energy",
                value: internal_energy,
            });
        }
    }

    let density = density_at_hsml_1d_with_boundary(
        state.positions,
        state.masses,
        state.smoothing_lengths,
        state.box_size,
        boundary,
    )?;
    let density_values: Vec<f64> = density.iter().map(|value| value.density).collect();
    let pressure: Vec<f64> = density_values
        .iter()
        .zip(state.specific_internal_energy)
        .map(|(&rho, &internal_energy)| (state.gamma - 1.0) * rho * internal_energy)
        .collect();
    let density_gradients = gradients_at_hsml_1d_with_boundary(
        state.positions,
        &density_values,
        state.smoothing_lengths,
        state.box_size,
        0.0,
        true,
        boundary,
    )?;
    let velocity_gradients = gradients_at_hsml_1d_with_boundary(
        state.positions,
        state.velocities,
        state.smoothing_lengths,
        state.box_size,
        0.1,
        false,
        boundary,
    )?;
    let pressure_gradients = gradients_at_hsml_1d_with_boundary(
        state.positions,
        &pressure,
        state.smoothing_lengths,
        state.box_size,
        0.1,
        true,
        boundary,
    )?;
    let inverse_moments = inverse_moments_1d_with_boundary(
        state.positions,
        state.smoothing_lengths,
        state.box_size,
        boundary,
    )?;
    let closure_errors = face_closure_errors_1d_with_boundary(
        state.positions,
        state.smoothing_lengths,
        state.box_size,
        boundary,
    )?;
    let particle_divergence = particle_divergence_at_hsml_1d_with_boundary(
        state.positions,
        state.velocities,
        state.masses,
        state.smoothing_lengths,
        state.box_size,
        boundary,
    )?;

    let mut momentum = vec![0.0; particle_count];
    let mut total_energy = vec![0.0; particle_count];
    let mut pair_count = 0_usize;
    let mut entropic_pair_count = 0_usize;
    let mut maximum_signal_speed: Vec<f64> = pressure
        .iter()
        .zip(&density_values)
        .map(|(&particle_pressure, &rho)| (state.gamma * particle_pressure / rho).sqrt())
        .collect();
    for (i, j, displacement) in interacting_pairs_1d(
        state.positions,
        state.smoothing_lengths,
        state.box_size,
        boundary,
    )? {
        let distance = displacement.abs();
        let point_geometry = |index: usize| MeshlessPoint1d {
            position: state.positions[index],
            mass: state.masses[index],
            density: density_values[index],
            smoothing_length: state.smoothing_lengths[index],
            inverse_moment: inverse_moments[index],
        };
        let face = meshless_face_geometry_1d_with_boundary(
            point_geometry(i),
            point_geometry(j),
            state.box_size,
            boundary,
        )?;
        let reconstructed = |index: usize| ReconstructedPoint1d {
            primitive: PrimitiveState1d {
                density: density_values[index],
                velocity: state.velocities[index],
                pressure: pressure[index],
            },
            density_gradient: density_gradients[index].limited,
            velocity_gradient: velocity_gradients[index].limited,
            pressure_gradient: pressure_gradients[index].limited,
            face_closure_error: closure_errors[index],
        };
        let raw_flux = mfm_pair_flux_1d(reconstructed(i), reconstructed(j), face, state.gamma)?;
        let entropic = |index: usize| {
            Ok::<EntropicPoint1d, HydroError>(EntropicPoint1d {
                velocity: state.velocities[index],
                density: density_values[index],
                pressure: pressure[index],
                sound_speed: (state.gamma * pressure[index] / density_values[index]).sqrt(),
                volume: state.masses[index] / density_values[index],
                dhsml_factor: density[index].dhsml_factor,
                kernel_radial_derivative: cubic_kernel_1d(
                    distance,
                    state.smoothing_lengths[index],
                )?
                .radial_derivative,
                condition_number: 1.0,
                face_closure_error: closure_errors[index],
            })
        };
        let (flux, selected_entropic) =
            apply_entropic_pdv_1d(raw_flux, face, entropic(i)?, entropic(j)?)?;
        momentum[i] += flux.momentum;
        momentum[j] -= flux.momentum;
        total_energy[i] += flux.energy;
        total_energy[j] -= flux.energy;
        pair_count += 1;
        entropic_pair_count += usize::from(selected_entropic);
        let signal_speed = pair_signal_speed(
            reconstructed(i),
            reconstructed(j),
            displacement.signum(),
            state.gamma,
        )?;
        maximum_signal_speed[i] = maximum_signal_speed[i].max(signal_speed);
        maximum_signal_speed[j] = maximum_signal_speed[j].max(signal_speed);
    }

    let mut acceleration = Vec::with_capacity(particle_count);
    let mut specific_internal_energy = Vec::with_capacity(particle_count);
    for index in 0..particle_count {
        let particle_acceleration = momentum[index] / state.masses[index];
        let internal_energy_rate =
            (total_energy[index] - state.velocities[index] * momentum[index]) / state.masses[index];
        if !momentum[index].is_finite()
            || !total_energy[index].is_finite()
            || !particle_acceleration.is_finite()
            || !internal_energy_rate.is_finite()
        {
            return Err(HydroError::NonFiniteRiemannResult {
                field: "spatial_rate",
                value: f64::NAN,
            });
        }
        acceleration.push(particle_acceleration);
        specific_internal_energy.push(internal_energy_rate);
    }
    Ok(MfmRates1d {
        momentum,
        total_energy,
        acceleration,
        specific_internal_energy,
        pair_count,
        entropic_pair_count,
        maximum_signal_speed,
        particle_divergence,
    })
}

/// Select the synchronized non-cosmological Courant step used by the 1-D MFM
/// sound-wave profile.
///
/// The effective particle size is `2 Hsml / N_eff`; the legacy denominator is
/// one half of the particle's maximum signal speed.
///
/// # Errors
///
/// Returns an error for invalid state/rate columns, Courant factor, neighbor
/// estimate, or signal speed.
pub fn global_courant_timestep_1d(
    state: MfmState1d<'_>,
    rates: &MfmRates1d,
    courant_factor: f64,
) -> Result<f64, HydroError> {
    if !courant_factor.is_finite() || courant_factor <= 0.0 || courant_factor > 0.5 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "courant_factor",
            value: courant_factor,
        });
    }
    if rates.maximum_signal_speed.len() != state.positions.len() {
        return Err(HydroError::MismatchedLength {
            field: "maximum_signal_speed",
            expected: state.positions.len(),
            actual: rates.maximum_signal_speed.len(),
        });
    }
    let density = density_at_hsml_1d(
        state.positions,
        state.masses,
        state.smoothing_lengths,
        state.box_size,
    )?;
    let mut timestep = f64::INFINITY;
    for (index, estimate) in density.iter().enumerate() {
        let signal_speed = rates.maximum_signal_speed[index];
        let particle_size = 2.0 * state.smoothing_lengths[index] / estimate.effective_neighbors;
        let candidate = courant_factor * particle_size / (0.5 * signal_speed);
        if !particle_size.is_finite()
            || particle_size <= 0.0
            || !signal_speed.is_finite()
            || signal_speed <= 0.0
            || !candidate.is_finite()
            || candidate <= 0.0
        {
            return Err(HydroError::NonFiniteRiemannResult {
                field: "courant_timestep",
                value: candidate,
            });
        }
        timestep = timestep.min(candidate);
    }
    if !timestep.is_finite() {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "global_timestep",
            value: timestep,
        });
    }
    Ok(timestep)
}

/// Provenance for the bound selected by [`select_public_soundwave_timestep_1d`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimestepBound1d {
    MaximumSize,
    Courant { particle_index: usize },
    Acceleration { particle_index: usize },
    GasDivergence { particle_index: usize },
}

/// A continuous timestep and the bound that selected it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestepSelection1d {
    pub duration: f64,
    pub bound: TimestepBound1d,
}

fn validate_public_soundwave_timestep_inputs(
    state: MfmState1d<'_>,
    rates: &MfmRates1d,
    maximum_timestep: f64,
    courant_factor: f64,
    integration_accuracy: f64,
) -> Result<(), HydroError> {
    for (field, value) in [
        ("maximum_timestep", maximum_timestep),
        ("integration_accuracy", integration_accuracy),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(HydroError::InvalidRiemannParameter { field, value });
        }
    }
    if !courant_factor.is_finite() || courant_factor <= 0.0 || courant_factor > 0.5 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "courant_factor",
            value: courant_factor,
        });
    }
    if !state.gamma.is_finite() || state.gamma <= 1.0 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "gamma",
            value: state.gamma,
        });
    }

    let particle_count = state.positions.len();
    for (field, actual) in [
        ("masses", state.masses.len()),
        ("velocities", state.velocities.len()),
        (
            "specific_internal_energy",
            state.specific_internal_energy.len(),
        ),
        ("smoothing_lengths", state.smoothing_lengths.len()),
        ("maximum_signal_speed", rates.maximum_signal_speed.len()),
        ("acceleration", rates.acceleration.len()),
        ("particle_divergence", rates.particle_divergence.len()),
    ] {
        if actual != particle_count {
            return Err(HydroError::MismatchedLength {
                field,
                expected: particle_count,
                actual,
            });
        }
    }
    for (index, (&velocity, &internal_energy)) in state
        .velocities
        .iter()
        .zip(state.specific_internal_energy)
        .enumerate()
    {
        if !velocity.is_finite() {
            return Err(HydroError::InvalidParticle {
                index,
                field: "velocity",
                value: velocity,
            });
        }
        if !internal_energy.is_finite() || internal_energy <= 0.0 {
            return Err(HydroError::InvalidParticle {
                index,
                field: "specific_internal_energy",
                value: internal_energy,
            });
        }
    }
    Ok(())
}

/// Select the strict non-cosmological timestep bound for the public 1-D
/// sound-wave configuration.
///
/// This specializes `get_timestep` in `timestep.c` to the enabled public
/// sound-wave physics. The result is the minimum of `MaxSizeTimestep`, the gas
/// Courant bound, the acceleration bound
/// `sqrt(2 ErrTolIntAccuracy KERNEL_CORE_SIZE Hsml / |a|)`, and the gas
/// divergence bound `1.5 / |Particle_DivVel|`. The public configuration uses
/// the cubic kernel, hence [`CUBIC_KERNEL_CORE_SIZE`] is one half.
///
/// Zero acceleration and divergence impose no bound. Exact ties retain the
/// earlier criterion in the order documented above, matching the C code's
/// strict `candidate < dt` comparisons.
///
/// # Errors
///
/// Returns an error for invalid state/rate columns, selector parameters, or a
/// non-finite/non-positive candidate.
pub fn select_public_soundwave_timestep_1d(
    state: MfmState1d<'_>,
    rates: &MfmRates1d,
    maximum_timestep: f64,
    courant_factor: f64,
    integration_accuracy: f64,
) -> Result<TimestepSelection1d, HydroError> {
    validate_public_soundwave_timestep_inputs(
        state,
        rates,
        maximum_timestep,
        courant_factor,
        integration_accuracy,
    )?;

    let density = density_at_hsml_1d(
        state.positions,
        state.masses,
        state.smoothing_lengths,
        state.box_size,
    )?;
    let mut selection = TimestepSelection1d {
        duration: maximum_timestep,
        bound: TimestepBound1d::MaximumSize,
    };

    for (index, estimate) in density.iter().enumerate() {
        let signal_speed = rates.maximum_signal_speed[index];
        let particle_size = 2.0 * state.smoothing_lengths[index] / estimate.effective_neighbors;
        let courant = courant_factor * particle_size / (0.5 * signal_speed);
        if !particle_size.is_finite()
            || particle_size <= 0.0
            || !signal_speed.is_finite()
            || signal_speed <= 0.0
            || !courant.is_finite()
            || courant <= 0.0
        {
            return Err(HydroError::NonFiniteRiemannResult {
                field: "courant_timestep",
                value: courant,
            });
        }
        if courant < selection.duration {
            selection = TimestepSelection1d {
                duration: courant,
                bound: TimestepBound1d::Courant {
                    particle_index: index,
                },
            };
        }

        let acceleration = rates.acceleration[index];
        if !acceleration.is_finite() {
            return Err(HydroError::InvalidParticle {
                index,
                field: "acceleration",
                value: acceleration,
            });
        }
        if acceleration != 0.0 {
            let acceleration_bound = (2.0
                * integration_accuracy
                * CUBIC_KERNEL_CORE_SIZE
                * state.smoothing_lengths[index]
                / acceleration.abs())
            .sqrt();
            if !acceleration_bound.is_finite() || acceleration_bound <= 0.0 {
                return Err(HydroError::NonFiniteRiemannResult {
                    field: "acceleration_timestep",
                    value: acceleration_bound,
                });
            }
            if acceleration_bound < selection.duration {
                selection = TimestepSelection1d {
                    duration: acceleration_bound,
                    bound: TimestepBound1d::Acceleration {
                        particle_index: index,
                    },
                };
            }
        }

        let divergence = rates.particle_divergence[index];
        if !divergence.is_finite() {
            return Err(HydroError::InvalidParticle {
                index,
                field: "particle_divergence",
                value: divergence,
            });
        }
        if divergence != 0.0 {
            let divergence_bound = 1.5 / divergence.abs();
            if !divergence_bound.is_finite() || divergence_bound <= 0.0 {
                return Err(HydroError::NonFiniteRiemannResult {
                    field: "divergence_timestep",
                    value: divergence_bound,
                });
            }
            if divergence_bound < selection.duration {
                selection = TimestepSelection1d {
                    duration: divergence_bound,
                    bound: TimestepBound1d::GasDivergence {
                        particle_index: index,
                    },
                };
            }
        }
    }

    Ok(selection)
}

pub const LEGACY_TIMEBASE_TICKS: u64 = 1_u64 << 60;

/// Integer power-of-two timeline for GIZMO's default `LONG_INTEGER_TIME`
/// synchronized integration mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynchronizedTimeline1d {
    time_begin: f64,
    time_max: f64,
    current_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynchronizedStep1d {
    pub ticks: u64,
    pub duration: f64,
    pub end_time: f64,
}

// The public C configuration uses a 2^60 tick timebase. Conversion to f64 is
// intentional here: it mirrors C's double-precision timeline arithmetic, while
// power-of-two synchronized steps remain exactly representable.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
impl SynchronizedTimeline1d {
    /// Construct an integer timeline at its initial time.
    ///
    /// # Errors
    ///
    /// Returns an error unless both times are finite and `time_max` is later
    /// than `time_begin`.
    pub fn new(time_begin: f64, time_max: f64) -> Result<Self, HydroError> {
        if !time_begin.is_finite() || !time_max.is_finite() || time_max <= time_begin {
            return Err(HydroError::InvalidRiemannParameter {
                field: "timeline",
                value: time_max,
            });
        }
        Ok(Self {
            time_begin,
            time_max,
            current_tick: 0,
        })
    }

    #[must_use]
    pub fn current_tick(self) -> u64 {
        self.current_tick
    }

    #[must_use]
    pub fn current_time(self) -> f64 {
        self.time_begin + self.tick_duration() * self.current_tick as f64
    }

    #[must_use]
    pub fn is_finished(self) -> bool {
        self.current_tick >= LEGACY_TIMEBASE_TICKS
    }

    /// Quantize a physical bound to the largest synchronized power-of-two
    /// timestep that does not cross the end of the timeline.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid desired/cap timestep or a finished
    /// timeline.
    pub fn select_step(
        self,
        desired_timestep: f64,
        maximum_timestep: f64,
    ) -> Result<SynchronizedStep1d, HydroError> {
        if self.is_finished()
            || !desired_timestep.is_finite()
            || desired_timestep <= 0.0
            || !maximum_timestep.is_finite()
            || maximum_timestep <= 0.0
        {
            return Err(HydroError::InvalidRiemannParameter {
                field: "timeline_timestep",
                value: desired_timestep,
            });
        }
        let bounded = desired_timestep.min(maximum_timestep);
        let requested_ticks = (bounded / self.tick_duration()).floor();
        if !requested_ticks.is_finite() || requested_ticks < 2.0 {
            return Err(HydroError::InvalidRiemannParameter {
                field: "timeline_ticks",
                value: requested_ticks,
            });
        }
        let remaining = LEGACY_TIMEBASE_TICKS - self.current_tick;
        let raw_ticks = requested_ticks.min(remaining as f64);
        let integer_ticks = raw_ticks as u64;
        let next_power = integer_ticks.next_power_of_two();
        let mut ticks = if next_power > integer_ticks {
            next_power >> 1
        } else {
            next_power
        };
        while self.current_tick % ticks != 0 {
            ticks >>= 1;
        }
        while ticks > remaining {
            ticks >>= 1;
        }
        if ticks == 0 {
            return Err(HydroError::InvalidRiemannParameter {
                field: "timeline_remaining",
                value: remaining as f64,
            });
        }
        let duration = ticks as f64 * self.tick_duration();
        let end_time = self.current_time() + duration;
        Ok(SynchronizedStep1d {
            ticks,
            duration,
            end_time,
        })
    }

    /// Commit a previously selected step.
    ///
    /// # Errors
    ///
    /// Returns an error if the step is zero, unsynchronized, or would cross
    /// the end of the timeline.
    pub fn advance(&mut self, step: SynchronizedStep1d) -> Result<(), HydroError> {
        let remaining = LEGACY_TIMEBASE_TICKS - self.current_tick;
        if step.ticks == 0
            || !step.ticks.is_power_of_two()
            || self.current_tick % step.ticks != 0
            || step.ticks > remaining
            || !legacy_float_equal(step.duration, step.ticks as f64 * self.tick_duration())
            || !legacy_float_equal(step.end_time, self.current_time() + step.duration)
        {
            return Err(HydroError::InvalidRiemannParameter {
                field: "timeline_step",
                value: step.duration,
            });
        }
        self.current_tick += step.ticks;
        Ok(())
    }

    fn tick_duration(self) -> f64 {
        (self.time_max - self.time_begin) / LEGACY_TIMEBASE_TICKS as f64
    }
}

/// State during the drift portion of a synchronized MFM KDK step.
///
/// Legacy snapshots write the conserved half-kicked velocity, but predicted
/// internal energy, density, and smoothing length. Both velocity phases are
/// retained explicitly so callers cannot silently serialize the wrong one.
#[derive(Clone, Debug, PartialEq)]
pub struct MfmDriftState1d {
    pub positions: Vec<f64>,
    pub conserved_velocities: Vec<f64>,
    pub predicted_velocities: Vec<f64>,
    pub predicted_specific_internal_energy: Vec<f64>,
    pub predicted_density: Vec<f64>,
    pub predicted_smoothing_lengths: Vec<f64>,
}

/// Prepared first kick and drift predictor for one synchronized MFM step.
#[derive(Clone, Debug, PartialEq)]
pub struct MfmKdkStep1d {
    start: MfmEvolvingState1d,
    half_velocity: Vec<f64>,
    half_internal_energy: Vec<f64>,
    acceleration: Vec<f64>,
    specific_internal_energy_rate: Vec<f64>,
    particle_divergence: Vec<f64>,
    drift: MfmDriftState1d,
    elapsed: f64,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    boundary: BoundaryMode1d,
    reflective_wall_offsets: Option<Vec<f64>>,
}

impl MfmKdkStep1d {
    /// Advance the legacy predictor state to an elapsed drift time.
    ///
    /// Calls must be monotonic. Each call mutates the predictor from its
    /// current cursor, matching the legacy `move_particles` calls made for
    /// scheduled outputs inside a force step. This distinction is observable
    /// because the smoothing-length clamp and internal-energy limiter apply to
    /// every drift segment, not once to the total elapsed interval.
    ///
    /// # Errors
    ///
    /// Returns an error when the elapsed time precedes the current cursor, lies
    /// outside this step, or any predicted field becomes invalid. The cursor
    /// and predictor remain unchanged on error.
    pub fn drift_state(&mut self, elapsed: f64) -> Result<MfmDriftState1d, HydroError> {
        if !elapsed.is_finite() || elapsed < self.elapsed || elapsed > self.timestep {
            return Err(HydroError::InvalidRiemannParameter {
                field: "kdk_drift_elapsed",
                value: elapsed,
            });
        }
        let segment = elapsed - self.elapsed;
        let particle_count = self.start.positions.len();
        let mut positions = Vec::with_capacity(particle_count);
        let mut predicted_velocities = Vec::with_capacity(particle_count);
        let mut predicted_specific_internal_energy = Vec::with_capacity(particle_count);
        let mut predicted_density = Vec::with_capacity(particle_count);
        let mut predicted_smoothing_lengths = Vec::with_capacity(particle_count);
        let mut half_velocity = self.half_velocity.clone();
        let mut acceleration = self.acceleration.clone();
        for index in 0..particle_count {
            let crossed_position = self.drift.positions[index] + segment * half_velocity[index];
            let mut position = match self.boundary {
                BoundaryMode1d::Periodic => crossed_position.rem_euclid(self.start.box_size),
                BoundaryMode1d::Reflective => crossed_position.clamp(0.0, self.start.box_size),
            };
            let mut predicted_velocity =
                self.drift.predicted_velocities[index] + segment * acceleration[index];
            if self.boundary == BoundaryMode1d::Reflective {
                let Some(offset) = self
                    .reflective_wall_offsets
                    .as_ref()
                    .and_then(|offsets| offsets.get(index))
                    .copied()
                else {
                    return Err(HydroError::InvalidParticle {
                        index,
                        field: "reflective_wall_offset",
                        value: f64::NAN,
                    });
                };
                if crossed_position <= 0.0 {
                    if half_velocity[index] < 0.0 {
                        half_velocity[index] = -half_velocity[index];
                        predicted_velocity = half_velocity[index];
                        acceleration[index] = 0.0;
                    }
                    position = (0.1 * crossed_position).max(offset * self.start.box_size);
                } else if crossed_position >= self.start.box_size {
                    if half_velocity[index] > 0.0 {
                        half_velocity[index] = -half_velocity[index];
                        predicted_velocity = half_velocity[index];
                        acceleration[index] = 0.0;
                    }
                    position = self.start.box_size * (1.0 - offset);
                }
            }
            let predicted_internal_energy = limited_internal_energy_update(
                self.drift.predicted_specific_internal_energy[index],
                self.specific_internal_energy_rate[index],
                segment,
                self.minimum_specific_internal_energy,
            )?;
            let divergence_increment = (self.particle_divergence[index] * segment).clamp(-0.3, 0.3);
            let density = self.drift.predicted_density[index] * (-divergence_increment).exp();
            let smoothing_length =
                self.drift.predicted_smoothing_lengths[index] * divergence_increment.exp();
            if !position.is_finite()
                || !predicted_velocity.is_finite()
                || !density.is_finite()
                || density <= 0.0
                || !smoothing_length.is_finite()
                || smoothing_length <= 0.0
            {
                return Err(HydroError::NonFiniteRiemannResult {
                    field: "kdk_drift_predictor",
                    value: f64::NAN,
                });
            }
            positions.push(position);
            predicted_velocities.push(predicted_velocity);
            predicted_specific_internal_energy.push(predicted_internal_energy);
            predicted_density.push(density);
            predicted_smoothing_lengths.push(smoothing_length);
        }
        let drift = MfmDriftState1d {
            positions,
            conserved_velocities: half_velocity.clone(),
            predicted_velocities,
            predicted_specific_internal_energy,
            predicted_density,
            predicted_smoothing_lengths,
        };
        self.half_velocity = half_velocity;
        self.acceleration = acceleration;
        self.drift = drift.clone();
        self.elapsed = elapsed;
        Ok(drift)
    }

    #[must_use]
    pub fn timestep(&self) -> f64 {
        self.timestep
    }

    /// Elapsed drift time already committed to this predictor.
    #[must_use]
    pub fn elapsed(&self) -> f64 {
        self.elapsed
    }
}

/// Apply the first half-kick and prepare a synchronized MFM drift.
///
/// # Errors
///
/// Returns an error for invalid state, rates, timestep, or energy floor.
pub fn begin_mfm_kdk_1d(
    state: &MfmEvolvingState1d,
    old_rates: &MfmRates1d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
) -> Result<MfmKdkStep1d, HydroError> {
    begin_mfm_kdk_1d_with_boundary_data(
        state,
        old_rates,
        timestep,
        minimum_specific_internal_energy,
        BoundaryMode1d::Periodic,
        None,
    )
}

/// Prepare a reflective public-C KDK step using explicit particle IDs.
///
/// The IDs are required because the legacy wall nudge is `ID * 2e-8` of the
/// box length. This API fails closed instead of inferring IDs from row order.
///
/// # Errors
///
/// Returns an error for invalid state/rates, mismatched or invalid IDs, or
/// invalid timestep and energy-floor inputs.
pub fn begin_mfm_reflective_kdk_1d(
    state: &MfmEvolvingState1d,
    old_rates: &MfmRates1d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    particle_ids: &[u64],
) -> Result<MfmKdkStep1d, HydroError> {
    if particle_ids.len() != state.positions.len() {
        return Err(HydroError::MismatchedLength {
            field: "particle_ids",
            expected: state.positions.len(),
            actual: particle_ids.len(),
        });
    }
    let mut offsets = Vec::with_capacity(particle_ids.len());
    for (index, &id) in particle_ids.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let offset = id as f64 * 2.0e-8;
        if id == 0 || !offset.is_finite() || offset >= 1.0 {
            return Err(HydroError::InvalidParticle {
                index,
                field: "reflective_particle_id",
                value: offset,
            });
        }
        offsets.push(offset);
    }
    begin_mfm_kdk_1d_with_boundary_data(
        state,
        old_rates,
        timestep,
        minimum_specific_internal_energy,
        BoundaryMode1d::Reflective,
        Some(offsets),
    )
}

fn begin_mfm_kdk_1d_with_boundary_data(
    state: &MfmEvolvingState1d,
    old_rates: &MfmRates1d,
    timestep: f64,
    minimum_specific_internal_energy: f64,
    boundary: BoundaryMode1d,
    reflective_wall_offsets: Option<Vec<f64>>,
) -> Result<MfmKdkStep1d, HydroError> {
    let particle_count = state.positions.len();
    validate_rate_columns(old_rates, particle_count)?;
    validate_evolving_state(state, boundary)?;
    if !timestep.is_finite() || timestep <= 0.0 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "timestep",
            value: timestep,
        });
    }
    if !minimum_specific_internal_energy.is_finite() || minimum_specific_internal_energy < 0.0 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "minimum_specific_internal_energy",
            value: minimum_specific_internal_energy,
        });
    }

    let half_timestep = 0.5 * timestep;
    let mut half_velocity = Vec::with_capacity(particle_count);
    let mut half_internal_energy = Vec::with_capacity(particle_count);
    for index in 0..particle_count {
        let velocity = state.velocities[index] + half_timestep * old_rates.acceleration[index];
        let internal_energy = limited_internal_energy_update(
            state.specific_internal_energy[index],
            old_rates.specific_internal_energy[index],
            half_timestep,
            minimum_specific_internal_energy,
        )?;
        if !velocity.is_finite() {
            return Err(HydroError::NonFiniteRiemannResult {
                field: "kdk_first_kick",
                value: velocity,
            });
        }
        half_velocity.push(velocity);
        half_internal_energy.push(internal_energy);
    }
    let start_density = density_at_hsml_1d_with_boundary(
        &state.positions,
        &state.masses,
        &state.smoothing_lengths,
        state.box_size,
        boundary,
    )?
    .into_iter()
    .map(|estimate| estimate.density)
    .collect();
    let drift = MfmDriftState1d {
        positions: state.positions.clone(),
        conserved_velocities: half_velocity.clone(),
        predicted_velocities: state.velocities.clone(),
        predicted_specific_internal_energy: state.specific_internal_energy.clone(),
        predicted_density: start_density,
        predicted_smoothing_lengths: state.smoothing_lengths.clone(),
    };
    Ok(MfmKdkStep1d {
        start: state.clone(),
        half_velocity,
        half_internal_energy,
        acceleration: old_rates.acceleration.clone(),
        specific_internal_energy_rate: old_rates.specific_internal_energy.clone(),
        particle_divergence: old_rates.particle_divergence.clone(),
        drift,
        elapsed: 0.0,
        timestep,
        minimum_specific_internal_energy,
        boundary,
        reflective_wall_offsets,
    })
}

/// Finish a prepared MFM step with endpoint density, force, and second kick.
///
/// Any drift segments already committed for scheduled outputs are retained;
/// only the interval from the current cursor to the endpoint is drifted here.
///
/// # Errors
///
/// Returns an error for smoothing-length failure or invalid endpoint physics.
/// No caller-owned state is mutated on failure.
pub fn finish_mfm_kdk_1d(
    mut step: MfmKdkStep1d,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
) -> Result<(MfmEvolvingState1d, MfmRates1d), HydroError> {
    let endpoint = step.drift_state(step.timestep)?;
    let solved = solve_public_c_smoothing_lengths_from_seeds_1d_with_boundary(
        &endpoint.positions,
        &step.start.masses,
        &endpoint.predicted_smoothing_lengths,
        step.start.box_size,
        desired_neighbors,
        neighbor_tolerance,
        step.boundary,
    )?;
    let endpoint_smoothing_lengths: Vec<f64> = solved
        .iter()
        .map(|particle| particle.smoothing_length)
        .collect();
    let new_rates = mfm_spatial_rates_1d_with_boundary(
        MfmState1d {
            positions: &endpoint.positions,
            masses: &step.start.masses,
            velocities: &endpoint.predicted_velocities,
            specific_internal_energy: &endpoint.predicted_specific_internal_energy,
            smoothing_lengths: &endpoint_smoothing_lengths,
            box_size: step.start.box_size,
            gamma: step.start.gamma,
        },
        step.boundary,
    )?;

    let half_timestep = 0.5 * step.timestep;
    let mut endpoint_velocity = Vec::with_capacity(step.start.positions.len());
    let mut endpoint_internal_energy = Vec::with_capacity(step.start.positions.len());
    for index in 0..step.start.positions.len() {
        let velocity = step.half_velocity[index] + half_timestep * new_rates.acceleration[index];
        let internal_energy = limited_internal_energy_update(
            step.half_internal_energy[index],
            new_rates.specific_internal_energy[index],
            half_timestep,
            step.minimum_specific_internal_energy,
        )?;
        if !velocity.is_finite() {
            return Err(HydroError::NonFiniteRiemannResult {
                field: "kdk_second_kick",
                value: velocity,
            });
        }
        endpoint_velocity.push(velocity);
        endpoint_internal_energy.push(internal_energy);
    }
    Ok((
        MfmEvolvingState1d {
            positions: endpoint.positions,
            masses: step.start.masses,
            velocities: endpoint_velocity,
            specific_internal_energy: endpoint_internal_energy,
            smoothing_lengths: endpoint_smoothing_lengths,
            box_size: step.start.box_size,
            gamma: step.start.gamma,
        },
        new_rates,
    ))
}

/// Advance one synchronized kick-drift-kick step of the default 1-D MFM path.
///
/// Endpoint forces use full-step old-RHS predictions for velocity and internal
/// energy, while positions drift with the first-half-kicked actual velocity,
/// matching the legacy predictor ordering. Smoothing lengths are re-solved at
/// the endpoint before the new RHS evaluation.
///
/// # Errors
///
/// Returns an error for invalid state/rate columns, timestep or energy floor,
/// smoothing-length failure, or endpoint hydro failure. The state is only
/// replaced after the complete step succeeds.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn advance_mfm_kdk_1d(
    state: &mut MfmEvolvingState1d,
    old_rates: &MfmRates1d,
    timestep: f64,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
    minimum_specific_internal_energy: f64,
) -> Result<MfmRates1d, HydroError> {
    let step = begin_mfm_kdk_1d(state, old_rates, timestep, minimum_specific_internal_energy)?;
    let (endpoint, new_rates) = finish_mfm_kdk_1d(step, desired_neighbors, neighbor_tolerance)?;
    *state = endpoint;
    Ok(new_rates)
}

#[allow(clippy::too_many_arguments)]
/// Advance one synchronized reflective public-C KDK step.
///
/// # Errors
///
/// Returns an error under the same conditions as
/// [`begin_mfm_reflective_kdk_1d`] or [`finish_mfm_kdk_1d`].
pub fn advance_mfm_reflective_kdk_1d(
    state: &mut MfmEvolvingState1d,
    old_rates: &MfmRates1d,
    timestep: f64,
    desired_neighbors: f64,
    neighbor_tolerance: f64,
    minimum_specific_internal_energy: f64,
    particle_ids: &[u64],
) -> Result<MfmRates1d, HydroError> {
    let step = begin_mfm_reflective_kdk_1d(
        state,
        old_rates,
        timestep,
        minimum_specific_internal_energy,
        particle_ids,
    )?;
    let (endpoint, new_rates) = finish_mfm_kdk_1d(step, desired_neighbors, neighbor_tolerance)?;
    *state = endpoint;
    Ok(new_rates)
}

fn validate_rate_columns(rates: &MfmRates1d, expected: usize) -> Result<(), HydroError> {
    for (field, actual) in [
        ("momentum_rate", rates.momentum.len()),
        ("total_energy_rate", rates.total_energy.len()),
        ("acceleration", rates.acceleration.len()),
        (
            "specific_internal_energy_rate",
            rates.specific_internal_energy.len(),
        ),
        ("maximum_signal_speed", rates.maximum_signal_speed.len()),
        ("particle_divergence", rates.particle_divergence.len()),
    ] {
        if actual != expected {
            return Err(HydroError::MismatchedLength {
                field,
                expected,
                actual,
            });
        }
    }
    for value in rates
        .momentum
        .iter()
        .chain(&rates.total_energy)
        .chain(&rates.acceleration)
        .chain(&rates.specific_internal_energy)
        .chain(&rates.maximum_signal_speed)
        .chain(&rates.particle_divergence)
    {
        if !value.is_finite() {
            return Err(HydroError::NonFiniteRiemannResult {
                field: "input_rate",
                value: *value,
            });
        }
    }
    Ok(())
}

fn validate_evolving_state(
    state: &MfmEvolvingState1d,
    boundary: BoundaryMode1d,
) -> Result<(), HydroError> {
    let view = state.as_view();
    let expected = view.positions.len();
    for (field, actual) in [
        ("masses", view.masses.len()),
        ("velocities", view.velocities.len()),
        (
            "specific_internal_energy",
            view.specific_internal_energy.len(),
        ),
        ("smoothing_lengths", view.smoothing_lengths.len()),
    ] {
        if actual != expected {
            return Err(HydroError::MismatchedLength {
                field,
                expected,
                actual,
            });
        }
    }
    validate_particle_columns_with_boundary(
        view.positions,
        view.masses,
        view.smoothing_lengths,
        view.box_size,
        boundary,
    )?;
    for (index, (&velocity, &internal_energy)) in view
        .velocities
        .iter()
        .zip(view.specific_internal_energy)
        .enumerate()
    {
        if !velocity.is_finite() || !internal_energy.is_finite() || internal_energy <= 0.0 {
            return Err(HydroError::InvalidParticle {
                index,
                field: "evolving_primitive",
                value: internal_energy,
            });
        }
    }
    if !view.gamma.is_finite() || view.gamma <= 1.0 {
        return Err(HydroError::InvalidRiemannParameter {
            field: "gamma",
            value: view.gamma,
        });
    }
    Ok(())
}

fn limited_internal_energy_update(
    previous: f64,
    rate: f64,
    timestep: f64,
    floor: f64,
) -> Result<f64, HydroError> {
    if !previous.is_finite()
        || previous <= 0.0
        || !rate.is_finite()
        || !timestep.is_finite()
        || timestep < 0.0
        || !floor.is_finite()
        || floor < 0.0
    {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "internal_energy_update",
            value: f64::NAN,
        });
    }
    let candidate = previous + timestep * rate;
    if !candidate.is_finite() {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "internal_energy_candidate",
            value: candidate,
        });
    }
    Ok(if candidate < 0.5 * previous {
        0.5 * previous
    } else {
        candidate
    }
    .max(floor))
}

fn pair_signal_speed(
    i: ReconstructedPoint1d,
    j: ReconstructedPoint1d,
    radial_normal: f64,
    gamma: f64,
) -> Result<f64, HydroError> {
    let sound_i = (gamma * i.primitive.pressure / i.primitive.density).sqrt();
    let sound_j = (gamma * j.primitive.pressure / j.primitive.density).sqrt();
    let radial_relative_velocity = (i.primitive.velocity - j.primitive.velocity) * radial_normal;
    let signal_speed = sound_i + sound_j - radial_relative_velocity.min(0.0);
    if !signal_speed.is_finite() || signal_speed <= 0.0 {
        return Err(HydroError::NonFiniteRiemannResult {
            field: "signal_speed",
            value: signal_speed,
        });
    }
    Ok(signal_speed)
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
    boundary: BoundaryMode1d,
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
            || (field == "position"
                && (value < 0.0
                    || if boundary == BoundaryMode1d::Reflective {
                        value > box_size
                    } else {
                        value >= box_size
                    }))
        {
            return Err(HydroError::InvalidFaceInput { side, field, value });
        }
    }
    Ok(())
}

fn validate_particle_columns_with_boundary(
    positions: &[f64],
    masses: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
    boundary: BoundaryMode1d,
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
                || (field == "position"
                    && (value < 0.0
                        || if boundary == BoundaryMode1d::Reflective {
                            value > box_size
                        } else {
                            value >= box_size
                        }))
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
    neighbor_index: &NeighborIndex1d,
) -> Result<DensityEstimate, HydroError> {
    let position = positions[index];
    let mut kernel_sum = 0.0;
    let mut derivative_sum = 0.0;
    let mut neighbors = Vec::new();
    neighbor_index.query(position, hsml, &mut neighbors);
    for neighbor in neighbors {
        let neighbor_position = positions[neighbor];
        let radius = displacement_1d(
            position,
            neighbor_position,
            box_size,
            neighbor_index.boundary,
        )?
        .abs();
        let kernel = cubic_kernel_1d(radius, hsml)?;
        kernel_sum += kernel.weight;
        if radius < hsml {
            derivative_sum += -(kernel.weight / hsml + (radius / hsml) * kernel.radial_derivative);
        }
    }
    for (field, value) in [
        ("kernel_sum", kernel_sum),
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
    let particle_density = masses[index] * kernel_sum;
    if !particle_density.is_finite() {
        return Err(HydroError::NonFiniteDensityEstimate {
            index,
            field: "density",
            value: particle_density,
        });
    }
    Ok(DensityEstimate {
        density: particle_density,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MfmState1d<'a> {
    pub positions: &'a [f64],
    pub masses: &'a [f64],
    pub velocities: &'a [f64],
    pub specific_internal_energy: &'a [f64],
    pub smoothing_lengths: &'a [f64],
    pub box_size: f64,
    pub gamma: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MfmEvolvingState1d {
    pub positions: Vec<f64>,
    pub masses: Vec<f64>,
    pub velocities: Vec<f64>,
    pub specific_internal_energy: Vec<f64>,
    pub smoothing_lengths: Vec<f64>,
    pub box_size: f64,
    pub gamma: f64,
}

impl MfmEvolvingState1d {
    #[must_use]
    pub fn as_view(&self) -> MfmState1d<'_> {
        MfmState1d {
            positions: &self.positions,
            masses: &self.masses,
            velocities: &self.velocities,
            specific_internal_energy: &self.specific_internal_energy,
            smoothing_lengths: &self.smoothing_lengths,
            box_size: self.box_size,
            gamma: self.gamma,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MfmRates1d {
    pub momentum: Vec<f64>,
    pub total_energy: Vec<f64>,
    pub acceleration: Vec<f64>,
    pub specific_internal_energy: Vec<f64>,
    pub pair_count: usize,
    pub entropic_pair_count: usize,
    pub maximum_signal_speed: Vec<f64>,
    pub particle_divergence: Vec<f64>,
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
    DegenerateTreeDomain {
        minimum: f64,
        maximum: f64,
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
            Self::DegenerateTreeDomain { minimum, maximum } => write!(
                formatter,
                "cannot build a tree for degenerate position extent {minimum}..{maximum}"
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

    fn single_particle_rates(signal_speed: f64, acceleration: f64, divergence: f64) -> MfmRates1d {
        MfmRates1d {
            momentum: vec![acceleration],
            total_energy: vec![0.0],
            acceleration: vec![acceleration],
            specific_internal_energy: vec![0.0],
            pair_count: 0,
            entropic_pair_count: 0,
            maximum_signal_speed: vec![signal_speed],
            particle_divergence: vec![divergence],
        }
    }

    fn single_particle_state() -> MfmState1d<'static> {
        MfmState1d {
            positions: &[0.5],
            masses: &[1.0],
            velocities: &[0.0],
            specific_internal_energy: &[0.9],
            smoothing_lengths: &[0.2],
            box_size: 1.0,
            gamma: 5.0 / 3.0,
        }
    }

    #[test]
    fn reflective_topology_does_not_interact_across_periodic_seam() {
        let positions = [0.01, 0.99];
        let masses = [1.0, 1.0];
        let smoothing_lengths = [0.05, 0.05];
        let periodic = density_at_hsml_1d(&positions, &masses, &smoothing_lengths, 1.0).unwrap();
        let reflective = density_at_hsml_1d_with_boundary(
            &positions,
            &masses,
            &smoothing_lengths,
            1.0,
            BoundaryMode1d::Reflective,
        )
        .unwrap();

        assert!(periodic[0].density > reflective[0].density);
        assert!(periodic[1].density > reflective[1].density);
        assert_eq!(
            interacting_pairs_1d(
                &positions,
                &smoothing_lengths,
                1.0,
                BoundaryMode1d::Reflective
            )
            .unwrap(),
            Vec::new()
        );
        assert_close(
            displacement_1d(0.99, 0.01, 1.0, BoundaryMode1d::Reflective).unwrap(),
            0.98,
        );
    }

    #[test]
    fn reflective_kdk_applies_public_c_id_nudge_and_resets_predictor_acceleration() {
        let state = MfmEvolvingState1d {
            positions: vec![0.99],
            masses: vec![1.0],
            velocities: vec![1.0],
            specific_internal_energy: vec![1.0],
            smoothing_lengths: vec![0.2],
            box_size: 1.0,
            gamma: 5.0 / 3.0,
        };
        let rates = single_particle_rates(1.0, 0.4, 0.0);
        let mut step = begin_mfm_reflective_kdk_1d(&state, &rates, 0.04, 0.0, &[7]).unwrap();

        let reflected = step.drift_state(0.02).unwrap();
        assert_close(reflected.positions[0], 1.0 - 7.0 * 2.0e-8);
        assert_close(reflected.conserved_velocities[0], -1.008);
        assert_close(reflected.predicted_velocities[0], -1.008);

        let continued = step.drift_state(0.04).unwrap();
        assert_close(continued.positions[0], 1.0 - 7.0 * 2.0e-8 - 0.02 * 1.008);
        assert_close(continued.predicted_velocities[0], -1.008);
        assert!(begin_mfm_reflective_kdk_1d(&state, &rates, 0.04, 0.0, &[0]).is_err());

        let lower_state = MfmEvolvingState1d {
            positions: vec![0.01],
            velocities: vec![-1.0],
            ..state
        };
        let zero_acceleration = single_particle_rates(1.0, 0.0, 0.0);
        let mut lower =
            begin_mfm_reflective_kdk_1d(&lower_state, &zero_acceleration, 0.02, 0.0, &[3]).unwrap();
        let lower_reflected = lower.drift_state(0.02).unwrap();
        assert_close(lower_reflected.positions[0], 3.0 * 2.0e-8);
        assert_close(lower_reflected.conserved_velocities[0], 1.0);
        assert_close(lower_reflected.predicted_velocities[0], 1.0);

        let boundary_state = MfmEvolvingState1d {
            positions: vec![0.0],
            velocities: vec![1.0],
            ..lower_state
        };
        let mut inward =
            begin_mfm_reflective_kdk_1d(&boundary_state, &zero_acceleration, 0.02, 0.0, &[5])
                .unwrap();
        let nudged = inward.drift_state(0.0).unwrap();
        assert_close(nudged.positions[0], 5.0 * 2.0e-8);
        assert_close(nudged.conserved_velocities[0], 1.0);
        assert_close(nudged.predicted_velocities[0], 1.0);
    }

    fn brute_force_interacting_pairs(
        positions: &[f64],
        smoothing_lengths: &[f64],
        box_size: f64,
    ) -> Vec<(usize, usize, f64)> {
        let mut pairs = Vec::new();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let displacement =
                    periodic_displacement_1d(positions[i], positions[j], box_size).unwrap();
                let distance = displacement.abs();
                if distance > 0.0
                    && (distance < smoothing_lengths[i] || distance < smoothing_lengths[j])
                {
                    pairs.push((i, j, displacement));
                }
            }
        }
        pairs
    }

    #[test]
    fn public_c_tree_seeds_are_particle_order_invariant() {
        let positions = [0.07, 0.19, 0.31, 0.44, 0.58, 0.69, 0.83, 0.94];
        let masses = [0.8, 1.1, 0.9, 1.2, 0.7, 1.3, 1.05, 0.95];
        let expected =
            public_c_tree_smoothing_length_seeds_1d(&positions, &masses, 1.0, 2.0).unwrap();

        let permutation = [5, 1, 7, 3, 0, 6, 2, 4];
        let permuted_positions: Vec<f64> =
            permutation.iter().map(|&index| positions[index]).collect();
        let permuted_masses: Vec<f64> = permutation.iter().map(|&index| masses[index]).collect();
        let permuted = public_c_tree_smoothing_length_seeds_1d(
            &permuted_positions,
            &permuted_masses,
            1.0,
            2.0,
        )
        .unwrap();

        for (permuted_index, &original_index) in permutation.iter().enumerate() {
            assert_eq!(
                permuted[permuted_index].to_bits(),
                expected[original_index].to_bits()
            );
        }
    }

    #[test]
    fn public_c_tree_seed_rejects_degenerate_extent() {
        assert!(matches!(
            public_c_tree_smoothing_length_seeds_1d(&[0.5, 0.5], &[1.0, 1.0], 1.0, 4.0),
            Err(HydroError::DegenerateTreeDomain {
                minimum: 0.5,
                maximum: 0.5
            })
        ));
    }

    #[test]
    fn sparse_periodic_pairs_match_brute_force_on_irregular_states() {
        let box_size = 1.0;
        let mut seed = 0x5eed_f00d_cafe_babe_u64;
        let mut random_unit = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bytes = seed.to_le_bytes();
            f64::from(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]))
                / f64::from(u32::MAX)
        };
        let mut positions = vec![0.0, 1.0e-12, 0.999_999_999_999];
        let mut smoothing_lengths = vec![0.031, 0.077, 0.043];
        for _ in positions.len()..96 {
            positions.push(random_unit());
            smoothing_lengths.push(0.008 + 0.082 * random_unit());
        }

        let sparse =
            interacting_pairs_periodic_1d(&positions, &smoothing_lengths, box_size).unwrap();
        let brute = brute_force_interacting_pairs(&positions, &smoothing_lengths, box_size);
        assert_eq!(sparse, brute);
        assert!(sparse.windows(2).all(|pair| {
            let (left_i, left_j, _) = pair[0];
            let (right_i, right_j, _) = pair[1];
            (left_i, left_j) < (right_i, right_j)
        }));
    }

    #[test]
    fn sorted_periodic_target_queries_match_brute_force_index_order() {
        let box_size = 1.0;
        let mut seed = 0xd1ff_3a71_a15e_5eed_u64;
        let mut random_unit = || {
            seed = seed
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            let bytes = seed.to_le_bytes();
            f64::from(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]))
                / f64::from(u32::MAX)
        };
        let mut positions = vec![0.0, 1.0e-13, 0.5, 0.999_999_999_999_9];
        for _ in positions.len()..128 {
            positions.push(random_unit());
        }
        let index = NeighborIndex1d::new(&positions, box_size, BoundaryMode1d::Periodic);
        let mut actual = Vec::new();
        for source in 0..positions.len() {
            for support in [0.001, 0.017 + 0.08 * random_unit(), 0.5, 0.73] {
                index.query(positions[source], support, &mut actual);
                let expected: Vec<usize> = positions
                    .iter()
                    .enumerate()
                    .filter_map(|(neighbor, &position)| {
                        let distance =
                            periodic_displacement_1d(positions[source], position, box_size)
                                .unwrap()
                                .abs();
                        (distance <= support).then_some(neighbor)
                    })
                    .collect();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn sparse_periodic_pairs_handle_seam_duplicates_and_global_support() {
        let positions = [0.92, 0.08, 0.51, 0.49, 0.08, 0.75, 0.25];
        for smoothing_lengths in [
            [0.04, 0.17, 0.03, 0.08, 0.11, 0.02, 0.26],
            [0.5, 0.03, 0.04, 0.02, 0.01, 0.6, 0.05],
        ] {
            let sparse =
                interacting_pairs_periodic_1d(&positions, &smoothing_lengths, 1.0).unwrap();
            let brute = brute_force_interacting_pairs(&positions, &smoothing_lengths, 1.0);
            assert_eq!(sparse, brute);
        }
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
    fn mfm_density_uses_particle_mass_times_kernel_number_density() {
        let estimates = density_at_hsml_1d(&[0.0, 0.25], &[1.0, 3.0], &[0.5; 2], 1.0).unwrap();
        assert_close(estimates[0].density, estimates[1].density / 3.0);
        assert_close(
            estimates[0].density,
            estimates[0].effective_neighbors / (2.0 * 0.5),
        );
    }

    #[test]
    fn particle_divergence_recovers_linear_expansion_and_uniform_translation() {
        let positions = [0.25, 0.5, 0.75];
        let masses = [1.0 / 3.0; 3];
        let hsml = [0.4; 3];
        let velocities = [-0.5, 0.0, 0.5];
        let divergence =
            particle_divergence_at_hsml_1d(&positions, &velocities, &masses, &hsml, 1.0).unwrap();
        assert_close(divergence[1], 2.0);

        let translated: Vec<f64> = velocities.iter().map(|velocity| velocity + 7.0).collect();
        let translated_divergence =
            particle_divergence_at_hsml_1d(&positions, &translated, &masses, &hsml, 1.0).unwrap();
        assert_close(translated_divergence[1], divergence[1]);
    }

    #[test]
    fn malformed_columns_fail_closed() {
        assert!(density_at_hsml_1d(&[0.0], &[], &[0.5], 1.0).is_err());
        assert!(density_at_hsml_1d(&[0.0], &[1.0], &[0.0], 1.0).is_err());
        assert!(density_at_hsml_1d(&[0.0], &[1.0], &[0.5], -1.0).is_err());
        assert!(density_at_hsml_1d(&[0.0, 0.25], &[f64::MAX; 2], &[0.5; 2], 1.0).is_err());
        assert!(cubic_kernel_1d(0.0, f64::MIN_POSITIVE).is_err());
        assert!(particle_divergence_at_hsml_1d(&[0.0], &[], &[1.0], &[0.5], 1.0).is_err());
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
    fn legitimate_kt_pair_flows_through_entropic_correction() {
        let i = reconstructed_point(PrimitiveState1d {
            density: 2.0,
            velocity: 1.0,
            pressure: 0.8,
        });
        let j = reconstructed_point(PrimitiveState1d {
            density: 1.0,
            velocity: -1.0,
            pressure: 0.6,
        });
        let raw = mfm_pair_flux_1d(i, j, unit_pair_face(1.0), 5.0 / 3.0).unwrap();
        assert_eq!(raw.method, RiemannMethod::KurganovTadmor);
        assert!((raw.momentum - raw.star_pressure).abs() > 1.0);
        let entropic = |point: ReconstructedPoint1d| EntropicPoint1d {
            velocity: point.primitive.velocity,
            density: point.primitive.density,
            pressure: point.primitive.pressure,
            sound_speed: ((5.0 / 3.0) * point.primitive.pressure / point.primitive.density).sqrt(),
            volume: point.primitive.density.recip(),
            dhsml_factor: 1.0,
            kernel_radial_derivative: -1.0,
            condition_number: 1.0,
            face_closure_error: 0.0,
        };
        let (corrected, _) =
            apply_entropic_pdv_1d(raw, unit_pair_face(1.0), entropic(i), entropic(j)).unwrap();
        assert_eq!(corrected.method, RiemannMethod::KurganovTadmor);
        assert!(corrected.energy.is_finite());
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
    fn spatial_operator_is_conservative_uniform_and_galilean_covariant() {
        let count = 8_u32;
        let positions: Vec<f64> = (0..count)
            .map(|index| (f64::from(index) + 0.5) / f64::from(count))
            .collect();
        let masses = vec![1.0 / f64::from(count); positions.len()];
        let smoothing_lengths = vec![0.25; positions.len()];
        let uniform_velocity = vec![2.0; positions.len()];
        let uniform_energy = vec![0.9; positions.len()];
        let uniform = mfm_spatial_rates_1d(MfmState1d {
            positions: &positions,
            masses: &masses,
            velocities: &uniform_velocity,
            specific_internal_energy: &uniform_energy,
            smoothing_lengths: &smoothing_lengths,
            box_size: 1.0,
            gamma: 5.0 / 3.0,
        })
        .unwrap();
        assert_eq!(uniform.pair_count, positions.len());
        for value in uniform
            .momentum
            .iter()
            .chain(&uniform.total_energy)
            .chain(&uniform.acceleration)
            .chain(&uniform.specific_internal_energy)
        {
            assert!(value.abs() < 1.0e-14);
        }

        let velocity: Vec<f64> = positions
            .iter()
            .map(|position| 0.01 * (std::f64::consts::TAU * position).sin())
            .collect();
        let internal_energy: Vec<f64> = positions
            .iter()
            .map(|position| 0.9 + 0.01 * (std::f64::consts::TAU * position).sin())
            .collect();
        let state = MfmState1d {
            positions: &positions,
            masses: &masses,
            velocities: &velocity,
            specific_internal_energy: &internal_energy,
            smoothing_lengths: &smoothing_lengths,
            box_size: 1.0,
            gamma: 5.0 / 3.0,
        };
        let rates = mfm_spatial_rates_1d(state).unwrap();
        assert_close(rates.momentum.iter().sum(), 0.0);
        assert_close(rates.total_energy.iter().sum(), 0.0);
        for index in 0..positions.len() {
            assert_close(
                rates.acceleration[index],
                rates.momentum[index] / masses[index],
            );
            assert_close(
                rates.specific_internal_energy[index],
                (rates.total_energy[index] - velocity[index] * rates.momentum[index])
                    / masses[index],
            );
        }

        let boost = 3.0;
        let boosted_velocity: Vec<f64> = velocity.iter().map(|value| value + boost).collect();
        let boosted = mfm_spatial_rates_1d(MfmState1d {
            velocities: &boosted_velocity,
            ..state
        })
        .unwrap();
        for index in 0..positions.len() {
            assert!((boosted.acceleration[index] - rates.acceleration[index]).abs() < 1.0e-13);
            assert!(
                (boosted.specific_internal_energy[index] - rates.specific_internal_energy[index])
                    .abs()
                    < 1.0e-13
            );
        }
    }

    #[test]
    fn synchronized_courant_and_kdk_preserve_uniform_translation() {
        let count = 8_u32;
        let positions: Vec<f64> = (0..count)
            .map(|index| (f64::from(index) + 0.5) / f64::from(count))
            .collect();
        let mut state = MfmEvolvingState1d {
            positions,
            masses: vec![1.0 / f64::from(count); usize::try_from(count).unwrap()],
            velocities: vec![2.0; usize::try_from(count).unwrap()],
            specific_internal_energy: vec![0.9; usize::try_from(count).unwrap()],
            smoothing_lengths: vec![0.25; usize::try_from(count).unwrap()],
            box_size: 1.0,
            gamma: 5.0 / 3.0,
        };
        let rates = mfm_spatial_rates_1d(state.as_view()).unwrap();
        let timestep = global_courant_timestep_1d(state.as_view(), &rates, 0.05).unwrap();
        assert_close(timestep, 0.00625);
        let initial_positions = state.positions.clone();
        let mut phase = begin_mfm_kdk_1d(&state, &rates, timestep, 0.0).unwrap();
        let at_start = phase.drift_state(0.0).unwrap();
        let at_midpoint = phase.drift_state(0.5 * timestep).unwrap();
        for (index, &initial_position) in initial_positions.iter().enumerate() {
            assert_close(at_start.positions[index], initial_positions[index]);
            assert_close(at_start.conserved_velocities[index], 2.0);
            assert_close(at_start.predicted_velocities[index], 2.0);
            assert_close(at_start.predicted_specific_internal_energy[index], 0.9);
            assert_close(at_start.predicted_density[index], 1.0);
            assert_close(at_start.predicted_smoothing_lengths[index], 0.25);
            assert_close(
                at_midpoint.positions[index],
                (initial_position + timestep).rem_euclid(1.0),
            );
        }
        let (split_state, split_rates) = finish_mfm_kdk_1d(phase, 4.0, 1.0e-12).unwrap();
        let new_rates =
            advance_mfm_kdk_1d(&mut state, &rates, timestep, 4.0, 1.0e-12, 0.0).unwrap();
        assert_eq!(state.masses, split_state.masses);
        for (atomic, segmented) in state
            .positions
            .iter()
            .chain(&state.velocities)
            .chain(&state.specific_internal_energy)
            .chain(&state.smoothing_lengths)
            .zip(
                split_state
                    .positions
                    .iter()
                    .chain(&split_state.velocities)
                    .chain(&split_state.specific_internal_energy)
                    .chain(&split_state.smoothing_lengths),
            )
        {
            assert!(
                (*atomic - *segmented).abs() < 1.0e-13,
                "atomic state {atomic} differs from segmented state {segmented}"
            );
        }
        assert_eq!(new_rates.pair_count, split_rates.pair_count);
        assert_eq!(
            new_rates.entropic_pair_count,
            split_rates.entropic_pair_count
        );
        for (atomic, segmented) in new_rates
            .momentum
            .iter()
            .chain(&new_rates.total_energy)
            .chain(&new_rates.acceleration)
            .chain(&new_rates.specific_internal_energy)
            .chain(&new_rates.maximum_signal_speed)
            .chain(&new_rates.particle_divergence)
            .zip(
                split_rates
                    .momentum
                    .iter()
                    .chain(&split_rates.total_energy)
                    .chain(&split_rates.acceleration)
                    .chain(&split_rates.specific_internal_energy)
                    .chain(&split_rates.maximum_signal_speed)
                    .chain(&split_rates.particle_divergence),
            )
        {
            assert!(
                (*atomic - *segmented).abs() < 1.0e-13,
                "atomic rate {atomic} differs from segmented rate {segmented}"
            );
        }
        for (index, &position) in state.positions.iter().enumerate() {
            assert_close(
                position,
                (initial_positions[index] + 2.0 * timestep).rem_euclid(1.0),
            );
            assert_close(state.velocities[index], 2.0);
            assert_close(state.specific_internal_energy[index], 0.9);
            assert_close(state.smoothing_lengths[index], 0.25);
            assert!(new_rates.acceleration[index].abs() < 1.0e-14);
            assert!(new_rates.specific_internal_energy[index].abs() < 1.0e-14);
        }
    }

    #[test]
    fn scheduled_drift_crossings_mutate_predictor_limiters_and_endpoint_cursor() {
        let state = MfmEvolvingState1d {
            positions: vec![0.25],
            masses: vec![1.0],
            velocities: vec![0.1],
            specific_internal_energy: vec![1.0],
            smoothing_lengths: vec![0.2],
            box_size: 1.0,
            gamma: 5.0 / 3.0,
        };
        let rates = MfmRates1d {
            momentum: vec![0.0],
            total_energy: vec![0.0],
            acceleration: vec![0.4],
            specific_internal_energy: vec![-12.0],
            pair_count: 0,
            entropic_pair_count: 0,
            maximum_signal_speed: vec![1.0],
            particle_divergence: vec![10.0],
        };
        let mut segmented = begin_mfm_kdk_1d(&state, &rates, 0.1, 0.0).unwrap();
        let first_output = segmented.drift_state(0.05).unwrap();
        let endpoint_output = segmented.drift_state(0.1).unwrap();
        assert_close(first_output.predicted_specific_internal_energy[0], 0.5);
        assert_close(endpoint_output.positions[0], 0.262);
        assert_close(endpoint_output.predicted_velocities[0], 0.14);
        assert_close(endpoint_output.predicted_specific_internal_energy[0], 0.25);
        assert_close(
            endpoint_output.predicted_smoothing_lengths[0],
            0.2 * 0.6_f64.exp(),
        );
        assert_close(
            endpoint_output.predicted_density[0],
            first_output.predicted_density[0] * (-0.3_f64).exp(),
        );
        assert_close(segmented.elapsed(), 0.1);

        let committed = endpoint_output.clone();
        assert!(segmented.drift_state(0.09).is_err());
        assert_close(segmented.elapsed(), 0.1);
        assert_eq!(segmented.drift_state(0.1).unwrap(), committed);

        let mut unsegmented = begin_mfm_kdk_1d(&state, &rates, 0.1, 0.0).unwrap();
        let direct_endpoint = unsegmented.drift_state(0.1).unwrap();
        assert_close(direct_endpoint.predicted_specific_internal_energy[0], 0.5);
        assert_close(
            direct_endpoint.predicted_smoothing_lengths[0],
            0.2 * 0.3_f64.exp(),
        );
        assert_ne!(direct_endpoint, endpoint_output);

        let count = 8_u32;
        let particle_count = usize::try_from(count).unwrap();
        let finish_state = MfmEvolvingState1d {
            positions: (0..count)
                .map(|index| (f64::from(index) + 0.5) / f64::from(count))
                .collect(),
            masses: vec![1.0 / f64::from(count); particle_count],
            velocities: vec![0.0; particle_count],
            specific_internal_energy: vec![1.0; particle_count],
            smoothing_lengths: vec![0.25; particle_count],
            box_size: 1.0,
            gamma: 5.0 / 3.0,
        };
        let mut finish_rates = single_particle_rates(1.0, 0.0, 10.0);
        for column in [
            &mut finish_rates.momentum,
            &mut finish_rates.total_energy,
            &mut finish_rates.acceleration,
            &mut finish_rates.specific_internal_energy,
            &mut finish_rates.maximum_signal_speed,
            &mut finish_rates.particle_divergence,
        ] {
            let fill = column[0];
            column.resize(particle_count, fill);
        }
        finish_rates.specific_internal_energy[0] = -12.0;
        let mut finish_from_crossing =
            begin_mfm_kdk_1d(&finish_state, &finish_rates, 0.1, 0.0).unwrap();
        finish_from_crossing.drift_state(0.05).unwrap();
        let mut explicitly_at_endpoint = finish_from_crossing.clone();
        explicitly_at_endpoint.drift_state(0.1).unwrap();
        let finished_from_cursor = finish_mfm_kdk_1d(finish_from_crossing, 4.0, 1.0e-12).unwrap();
        let finished_from_endpoint =
            finish_mfm_kdk_1d(explicitly_at_endpoint, 4.0, 1.0e-12).unwrap();
        assert_eq!(finished_from_cursor, finished_from_endpoint);
    }

    #[test]
    fn public_soundwave_timestep_reports_each_limiting_bound() {
        let state = single_particle_state();

        let maximum = select_public_soundwave_timestep_1d(
            state,
            &single_particle_rates(1.0, 0.0, 0.0),
            0.01,
            0.1,
            0.01,
        )
        .unwrap();
        assert_close(maximum.duration, 0.01);
        assert_eq!(maximum.bound, TimestepBound1d::MaximumSize);

        let courant = select_public_soundwave_timestep_1d(
            state,
            &single_particle_rates(1.0, 0.0, 0.0),
            1.0,
            0.1,
            0.01,
        )
        .unwrap();
        assert_close(courant.duration, 0.03);
        assert_eq!(
            courant.bound,
            TimestepBound1d::Courant { particle_index: 0 }
        );

        let acceleration = select_public_soundwave_timestep_1d(
            state,
            &single_particle_rates(1.0, -5.0, 0.0),
            1.0,
            0.1,
            0.01,
        )
        .unwrap();
        assert_close(acceleration.duration, 0.02);
        assert_eq!(
            acceleration.bound,
            TimestepBound1d::Acceleration { particle_index: 0 }
        );

        let divergence = select_public_soundwave_timestep_1d(
            state,
            &single_particle_rates(1.0, -5.0, -100.0),
            1.0,
            0.1,
            0.01,
        )
        .unwrap();
        assert_close(divergence.duration, 0.015);
        assert_eq!(
            divergence.bound,
            TimestepBound1d::GasDivergence { particle_index: 0 }
        );
    }

    #[test]
    fn public_soundwave_timestep_matches_c_formula_and_strict_tie_order() {
        let state = single_particle_state();
        let rates = single_particle_rates(1.0, 5.0, 75.0);
        let expected = (2.0_f64 * 0.01 * CUBIC_KERNEL_CORE_SIZE * 0.2 / 5.0).sqrt();
        assert_close(expected, 0.02);
        assert_close(1.5 / 75.0, expected);

        let selection = select_public_soundwave_timestep_1d(state, &rates, 1.0, 0.1, 0.01).unwrap();
        assert_close(selection.duration, expected);
        assert_eq!(
            selection.bound,
            TimestepBound1d::Acceleration { particle_index: 0 }
        );

        let capped =
            select_public_soundwave_timestep_1d(state, &rates, expected, 0.1, 0.01).unwrap();
        assert_close(capped.duration, expected);
        assert_eq!(capped.bound, TimestepBound1d::MaximumSize);
    }

    #[test]
    fn public_soundwave_timestep_rejects_incomplete_or_nonfinite_inputs() {
        let state = single_particle_state();
        let mut rates = single_particle_rates(1.0, 0.0, 0.0);
        rates.particle_divergence.clear();
        assert!(select_public_soundwave_timestep_1d(state, &rates, 1.0, 0.1, 0.01).is_err());

        let nonfinite = single_particle_rates(1.0, f64::NAN, 0.0);
        assert!(select_public_soundwave_timestep_1d(state, &nonfinite, 1.0, 0.1, 0.01).is_err());
        assert!(
            select_public_soundwave_timestep_1d(
                state,
                &single_particle_rates(1.0, 0.0, 0.0),
                1.0,
                0.1,
                0.0,
            )
            .is_err()
        );
    }

    #[test]
    fn synchronized_timeline_matches_legacy_power_of_two_bins() {
        let shared_step_ticks = 1_u64 << 44;
        let mut timeline = SynchronizedTimeline1d::new(0.0, 1.5).unwrap();
        let initial = timeline.select_step(3.0e-5, 1.0e-3).unwrap();
        assert_eq!(initial.ticks, shared_step_ticks);
        assert_close(initial.duration, 2.288_818_359_375e-5);
        timeline.advance(initial).unwrap();

        let blocked_increase = timeline.select_step(5.0e-5, 1.0e-3).unwrap();
        assert_eq!(blocked_increase.ticks, shared_step_ticks);
        timeline.advance(blocked_increase).unwrap();
        let synchronized_increase = timeline.select_step(5.0e-5, 1.0e-3).unwrap();
        assert_eq!(synchronized_increase.ticks, shared_step_ticks << 1);

        let mut ending = SynchronizedTimeline1d::new(0.0, 1.5).unwrap();
        ending.current_tick = LEGACY_TIMEBASE_TICKS - shared_step_ticks;
        let final_step = ending.select_step(1.0e-3, 1.0e-3).unwrap();
        assert_eq!(final_step.ticks, shared_step_ticks);
        ending.advance(final_step).unwrap();
        assert!(ending.is_finished());
        assert_close(ending.current_time(), 1.5);
    }

    #[test]
    fn synchronized_timeline_clamps_a_large_bound_to_its_full_remaining_span() {
        let time_max = 0.000_610_351_562_5;
        let mut timeline = SynchronizedTimeline1d::new(0.0, time_max).unwrap();
        let full_span = timeline.select_step(0.001, 0.001).unwrap();
        assert_eq!(full_span.ticks, LEGACY_TIMEBASE_TICKS);
        assert_eq!(full_span.duration.to_bits(), time_max.to_bits());
        timeline.advance(full_span).unwrap();
        assert!(timeline.is_finished());
        assert_eq!(timeline.current_time().to_bits(), time_max.to_bits());
    }

    #[test]
    fn internal_energy_update_matches_legacy_half_loss_limiter_and_floor() {
        assert_close(
            limited_internal_energy_update(1.0, -6.0, 0.1, 0.0).unwrap(),
            0.5,
        );
        assert_close(
            limited_internal_energy_update(1.0, -4.0, 0.1, 0.0).unwrap(),
            0.6,
        );
        assert_close(
            limited_internal_energy_update(1.0, -6.0, 0.1, 0.75).unwrap(),
            0.75,
        );
        assert!(limited_internal_energy_update(1.0, f64::NAN, 0.1, 0.0).is_err());
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
