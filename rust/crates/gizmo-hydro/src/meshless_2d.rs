//! Two-dimensional rectangular-periodic meshless geometry.
//!
//! The formulas in this module are the `NUMDIMS == 2`,
//! `KERNEL_FUNCTION == 3` specializations of the pinned public C baseline:
//! `kernel.h` supplies the compact cubic kernel, `hydro/density.c` supplies
//! density and the MLS moment matrix, and
//! `hydro/compute_finitevol_faces.h` supplies the MFM face vector.  Smoothing
//! lengths are full compact-support radii, as in the C code.

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use crate::{AdaptiveDensityEstimate, DensityEstimate, KernelValue};

/// Exact normalization of GIZMO's default cubic spline in two dimensions.
pub const CUBIC_2D_NORMALIZATION: f64 = 40.0 / (7.0 * PI);
const MOMENT_CONDITION_LIMIT: f64 = 1.0e4;

/// A finite Cartesian vector in the simulated plane.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector2 {
    pub x: f64,
    pub y: f64,
}

impl Vector2 {
    pub const ZERO: Self = Self::new(0.0, 0.0);

    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x.mul_add(other.x, self.y * other.y)
    }

    #[must_use]
    pub fn squared_norm(self) -> f64 {
        self.dot(self)
    }

    #[must_use]
    pub fn norm(self) -> f64 {
        self.squared_norm().sqrt()
    }
}

impl Add for Vector2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Vector2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vector2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign for Vector2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f64> for Vector2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Div<f64> for Vector2 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl Neg for Vector2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

/// Row-major two-dimensional matrix.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix2 {
    pub xx: f64,
    pub xy: f64,
    pub yx: f64,
    pub yy: f64,
}

impl Matrix2 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0, 0.0);

    #[must_use]
    pub const fn new(xx: f64, xy: f64, yx: f64, yy: f64) -> Self {
        Self { xx, xy, yx, yy }
    }

    #[must_use]
    pub fn is_finite(self) -> bool {
        self.xx.is_finite() && self.xy.is_finite() && self.yx.is_finite() && self.yy.is_finite()
    }

    #[must_use]
    pub fn mul_vector(self, vector: Vector2) -> Vector2 {
        Vector2::new(
            self.xx.mul_add(vector.x, self.xy * vector.y),
            self.yx.mul_add(vector.x, self.yy * vector.y),
        )
    }

    #[must_use]
    pub fn transpose(self) -> Self {
        Self::new(self.xx, self.yx, self.xy, self.yy)
    }
}

/// Validated side lengths of a rectangular periodic domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box2d {
    lengths: Vector2,
}

impl Box2d {
    /// Validate and construct a box.
    ///
    /// # Errors
    ///
    /// Returns an error unless both lengths are finite and positive.
    pub fn new(length_x: f64, length_y: f64) -> Result<Self, GeometryError> {
        if !length_x.is_finite() || length_x <= 0.0 {
            return Err(GeometryError::InvalidBoxLength {
                axis: "x",
                value: length_x,
            });
        }
        if !length_y.is_finite() || length_y <= 0.0 {
            return Err(GeometryError::InvalidBoxLength {
                axis: "y",
                value: length_y,
            });
        }
        Ok(Self {
            lengths: Vector2::new(length_x, length_y),
        })
    }

    #[must_use]
    pub const fn lengths(self) -> Vector2 {
        self.lengths
    }

    #[must_use]
    pub fn contains(self, point: Vector2) -> bool {
        point.is_finite()
            && (0.0..self.lengths.x).contains(&point.x)
            && (0.0..self.lengths.y).contains(&point.y)
    }

    /// Wrap any finite point into the half-open periodic domain.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite coordinate.
    pub fn wrap(self, point: Vector2) -> Result<Vector2, GeometryError> {
        if !point.is_finite() {
            return Err(GeometryError::InvalidPoint {
                index: None,
                field: "position",
                value: f64::NAN,
            });
        }
        Ok(Vector2::new(
            wrap_component(point.x, self.lengths.x),
            wrap_component(point.y, self.lengths.y),
        ))
    }

    /// Minimum-image signed displacement `a - b`.
    ///
    /// Exactly half-box separations retain their input sign, matching the
    /// strict `>`/`<` periodic macros in the public C implementation.
    ///
    /// # Errors
    ///
    /// Returns an error unless both points are already wrapped.
    pub fn displacement(self, a: Vector2, b: Vector2) -> Result<Vector2, GeometryError> {
        self.validate_wrapped(a, None)?;
        self.validate_wrapped(b, None)?;
        let mut displacement = a - b;
        displacement.x = minimum_image(displacement.x, self.lengths.x);
        displacement.y = minimum_image(displacement.y, self.lengths.y);
        Ok(displacement)
    }

    fn validate_wrapped(self, point: Vector2, index: Option<usize>) -> Result<(), GeometryError> {
        for (field, value, length) in [
            ("position.x", point.x, self.lengths.x),
            ("position.y", point.y, self.lengths.y),
        ] {
            if !value.is_finite() || !(0.0..length).contains(&value) {
                return Err(GeometryError::InvalidPoint {
                    index,
                    field,
                    value,
                });
            }
        }
        Ok(())
    }
}

fn wrap_component(value: f64, length: f64) -> f64 {
    value.rem_euclid(length)
}

fn minimum_image(mut displacement: f64, length: f64) -> f64 {
    if displacement > 0.5 * length {
        displacement -= length;
    }
    if displacement < -0.5 * length {
        displacement += length;
    }
    displacement
}

/// Value and radial derivative of the default two-dimensional cubic kernel.
///
/// The returned derivative is `dW/dr`, not the derivative of the dimensionless
/// shape. Both values are zero at and outside the full support radius.
///
/// # Errors
///
/// Returns an error for non-finite inputs, negative radius, or non-positive
/// smoothing length.
pub fn cubic_kernel_2d(radius: f64, hsml: f64) -> Result<KernelValue, GeometryError> {
    if !radius.is_finite() || !hsml.is_finite() || radius < 0.0 || hsml <= 0.0 {
        return Err(GeometryError::InvalidKernelInput { radius, hsml });
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
    let inverse_h = hsml.recip();
    let result = KernelValue {
        weight: shape * CUBIC_2D_NORMALIZATION * inverse_h * inverse_h,
        radial_derivative: derivative_shape
            * CUBIC_2D_NORMALIZATION
            * inverse_h
            * inverse_h
            * inverse_h,
    };
    if !result.weight.is_finite() || !result.radial_derivative.is_finite() {
        return Err(GeometryError::NonFiniteKernelResult { radius, hsml });
    }
    Ok(result)
}

/// Exact cell-list index for periodic radial queries.
///
/// The grid is sized from `search_scale`, capped by the particle count so a
/// tiny smoothing length cannot cause an unbounded empty allocation. Queries
/// are filtered by exact minimum-image distance and returned in particle-index
/// order, preserving deterministic accumulator order.
#[derive(Clone, Debug)]
pub struct CellList2d {
    positions: Vec<Vector2>,
    domain: Box2d,
    cell_counts: [usize; 2],
    cell_widths: Vector2,
    cells: Vec<Vec<usize>>,
}

impl CellList2d {
    /// Build a reusable periodic index.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid position or search scale.
    pub fn new(
        positions: &[Vector2],
        domain: Box2d,
        search_scale: f64,
    ) -> Result<Self, GeometryError> {
        if !search_scale.is_finite() || search_scale <= 0.0 {
            return Err(GeometryError::InvalidSearchRadius(search_scale));
        }
        validate_positions(positions, domain)?;
        let allocation_cap = positions.len().max(1);
        let lengths = domain.lengths();
        let [nx, ny] = cell_grid_shape(lengths, search_scale, allocation_cap);
        let cell_widths = Vector2::new(
            lengths.x / exactly_representable_usize(nx),
            lengths.y / exactly_representable_usize(ny),
        );
        let mut cells = vec![Vec::new(); nx * ny];
        for (index, &position) in positions.iter().enumerate() {
            let [x, y] = cell_coordinates(position, cell_widths, [nx, ny]);
            cells[y * nx + x].push(index);
        }
        Ok(Self {
            positions: positions.to_vec(),
            domain,
            cell_counts: [nx, ny],
            cell_widths,
            cells,
        })
    }

    #[must_use]
    pub const fn cell_counts(&self) -> [usize; 2] {
        self.cell_counts
    }

    /// Return all particles at minimum-image distance strictly below `radius`.
    ///
    /// A zero-radius query is valid and returns no particles.
    ///
    /// # Errors
    ///
    /// Returns an error unless the target is wrapped and the radius is finite
    /// and non-negative.
    pub fn neighbors_within(
        &self,
        target: Vector2,
        radius: f64,
    ) -> Result<Vec<usize>, GeometryError> {
        self.domain.validate_wrapped(target, None)?;
        if !radius.is_finite() || radius < 0.0 {
            return Err(GeometryError::InvalidSearchRadius(radius));
        }
        if radius == 0.0 || self.positions.is_empty() {
            return Ok(Vec::new());
        }
        let [center_x, center_y] = cell_coordinates(target, self.cell_widths, self.cell_counts);
        let x_cells = candidate_axis_cells(
            center_x,
            radius,
            self.cell_widths.x,
            self.cell_counts[0],
            self.domain.lengths().x,
        );
        let y_cells = candidate_axis_cells(
            center_y,
            radius,
            self.cell_widths.y,
            self.cell_counts[1],
            self.domain.lengths().y,
        );
        let radius_squared = radius * radius;
        let mut output = Vec::new();
        for &y in &y_cells {
            for &x in &x_cells {
                for &index in &self.cells[y * self.cell_counts[0] + x] {
                    let displacement = self.domain.displacement(target, self.positions[index])?;
                    if displacement.squared_norm() < radius_squared {
                        output.push(index);
                    }
                }
            }
        }
        output.sort_unstable();
        output.dedup();
        Ok(output)
    }
}

fn cell_grid_shape(lengths: Vector2, scale: f64, allocation_cap: usize) -> [usize; 2] {
    let cap = exactly_representable_usize(allocation_cap);
    let desired_x = (lengths.x / scale).floor().max(1.0).min(cap);
    let desired_y = (lengths.y / scale).floor().max(1.0).min(cap);
    let downscale = (cap / (desired_x * desired_y)).min(1.0).sqrt();
    let mut nx = nonnegative_f64_to_usize((desired_x * downscale).floor()).max(1);
    let mut ny = nonnegative_f64_to_usize((desired_y * downscale).floor()).max(1);
    // Rounding can only overshoot by a small amount, but retain a checked
    // integer postcondition before allocating the flattened grid.
    while nx.saturating_mul(ny) > allocation_cap {
        if nx >= ny && nx > 1 {
            nx -= 1;
        } else if ny > 1 {
            ny -= 1;
        } else {
            break;
        }
    }
    [nx, ny]
}

fn cell_coordinates(position: Vector2, widths: Vector2, counts: [usize; 2]) -> [usize; 2] {
    [
        nonnegative_f64_to_usize((position.x / widths.x).floor()).min(counts[0] - 1),
        nonnegative_f64_to_usize((position.y / widths.y).floor()).min(counts[1] - 1),
    ]
}

fn candidate_axis_cells(
    center: usize,
    radius: f64,
    width: f64,
    count: usize,
    domain_length: f64,
) -> Vec<usize> {
    if count == 1 || radius >= 0.5 * domain_length {
        return (0..count).collect();
    }
    let span = nonnegative_f64_to_usize((radius / width).ceil());
    if span.saturating_mul(2).saturating_add(1) >= count {
        return (0..count).collect();
    }
    let mut output = Vec::with_capacity(2 * span + 1);
    output.push(center);
    for offset in 1..=span {
        output.push((center + offset) % count);
        output.push((center + count - offset) % count);
    }
    output.sort_unstable();
    output
}

#[allow(clippy::cast_precision_loss)]
fn exactly_representable_usize(value: usize) -> f64 {
    // Allocation-sized values are far below f64's exact-integer limit on
    // supported targets. Keeping this cast centralized documents that bound.
    value as f64
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn nonnegative_f64_to_usize(value: f64) -> usize {
    debug_assert!(value.is_finite() && value >= 0.0);
    value as usize
}

/// One unordered union-support interaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractionPair2d {
    pub i: usize,
    pub j: usize,
    /// Minimum-image `position[i] - position[j]`.
    pub displacement: Vector2,
    pub distance: f64,
}

/// Enumerate unordered pairs satisfying `r < H_i || r < H_j`.
///
/// The result is sorted lexicographically by `(i, j)`. The directed query from
/// every particle followed by canonicalization is equivalent to the C
/// union-support neighbor rule without using one global physical search radius.
///
/// # Errors
///
/// Returns an error for invalid particle geometry.
pub fn interacting_pairs_2d(
    positions: &[Vector2],
    smoothing_lengths: &[f64],
    domain: Box2d,
) -> Result<Vec<InteractionPair2d>, GeometryError> {
    validate_geometry_columns(positions, smoothing_lengths, domain)?;
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    let maximum_support = smoothing_lengths.iter().copied().fold(0.0_f64, f64::max);
    let index = CellList2d::new(positions, domain, maximum_support)?;
    let mut pair_indices = Vec::new();
    for (source, (&position, &support)) in positions.iter().zip(smoothing_lengths).enumerate() {
        for neighbor in index.neighbors_within(position, support)? {
            if neighbor != source {
                let distance = domain.displacement(position, positions[neighbor])?.norm();
                if distance > 0.0 {
                    pair_indices.push(if source < neighbor {
                        (source, neighbor)
                    } else {
                        (neighbor, source)
                    });
                }
            }
        }
    }
    pair_indices.sort_unstable();
    pair_indices.dedup();
    pair_indices
        .into_iter()
        .map(|(i, j)| {
            let displacement = domain.displacement(positions[i], positions[j])?;
            Ok(InteractionPair2d {
                i,
                j,
                distance: displacement.norm(),
                displacement,
            })
        })
        .collect()
}

/// Recompute two-dimensional MFM density at supplied smoothing lengths.
///
/// For MFM, density is `m_i sum_j W_ij(H_i)`, not the usual SPH
/// `sum_j m_j W_ij`; this is the explicit `#ifndef HYDRO_SPH` overwrite at the
/// end of `hydro/density.c`. Effective neighbor number is
/// `pi H_i^2 sum_j W_ij`, and the smoothing-length response is its
/// `NUMDIMS == 2` specialization.
///
/// # Errors
///
/// Returns an error for invalid or mismatched particle columns.
pub fn density_at_hsml_2d(
    positions: &[Vector2],
    masses: &[f64],
    smoothing_lengths: &[f64],
    domain: Box2d,
) -> Result<Vec<DensityEstimate>, GeometryError> {
    validate_particle_columns(positions, masses, smoothing_lengths, domain)?;
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    let maximum_support = smoothing_lengths.iter().copied().fold(0.0_f64, f64::max);
    let index = CellList2d::new(positions, domain, maximum_support)?;
    let mut output = Vec::with_capacity(positions.len());
    for (particle, (&position, &hsml)) in positions.iter().zip(smoothing_lengths).enumerate() {
        let mut kernel_sum = 0.0;
        let mut derivative_sum = 0.0;
        for neighbor in index.neighbors_within(position, hsml)? {
            let radius = domain.displacement(position, positions[neighbor])?.norm();
            let kernel = cubic_kernel_2d(radius, hsml)?;
            kernel_sum += kernel.weight;
            derivative_sum +=
                -(2.0 * kernel.weight / hsml + (radius / hsml) * kernel.radial_derivative);
        }
        let raw_response = if kernel_sum > 0.0 {
            derivative_sum * hsml / (2.0 * kernel_sum)
        } else {
            0.0
        };
        let estimate = DensityEstimate {
            density: masses[particle] * kernel_sum,
            effective_neighbors: PI * hsml * hsml * kernel_sum,
            dhsml_factor: if kernel_sum > 0.0 {
                if raw_response > -0.9 {
                    1.0 / (1.0 + raw_response)
                } else {
                    1.0
                }
            } else {
                0.0
            },
        };
        if !estimate.density.is_finite()
            || !estimate.effective_neighbors.is_finite()
            || !estimate.dhsml_factor.is_finite()
        {
            return Err(GeometryError::NonFiniteResult {
                index: particle,
                field: "density estimate",
                value: f64::NAN,
            });
        }
        output.push(estimate);
    }
    Ok(output)
}

/// Evaluate the public density-loop particle velocity-divergence estimator.
///
/// This is distinct from the trace of the MLS velocity gradient. It controls
/// smoothing-length prediction and the public divergence timestep bound.
///
/// # Errors
///
/// Returns an error for invalid columns, geometry, or non-finite arithmetic.
pub fn particle_divergence_at_hsml_2d(
    positions: &[Vector2],
    velocities: &[Vector2],
    smoothing_lengths: &[f64],
    dhsml_factors: &[f64],
    domain: Box2d,
) -> Result<Vec<f64>, GeometryError> {
    validate_geometry_columns(positions, smoothing_lengths, domain)?;
    if velocities.len() != positions.len() {
        return Err(GeometryError::MismatchedLength {
            field: "velocities",
            expected: positions.len(),
            actual: velocities.len(),
        });
    }
    if dhsml_factors.len() != positions.len() {
        return Err(GeometryError::MismatchedLength {
            field: "dhsml_factors",
            expected: positions.len(),
            actual: dhsml_factors.len(),
        });
    }
    for (index, velocity) in velocities.iter().enumerate() {
        if !velocity.is_finite() {
            return Err(GeometryError::NonFiniteResult {
                index,
                field: "velocity",
                value: f64::NAN,
            });
        }
    }
    let maximum_support = smoothing_lengths.iter().copied().fold(0.0_f64, f64::max);
    let neighbors = CellList2d::new(positions, domain, maximum_support)?;
    let mut output = Vec::with_capacity(positions.len());
    for (particle, (&position, &hsml)) in positions.iter().zip(smoothing_lengths).enumerate() {
        let mut kernel_sum = 0.0;
        let mut divergence_sum = 0.0;
        for neighbor in neighbors.neighbors_within(position, hsml)? {
            let displacement = domain.displacement(position, positions[neighbor])?;
            let radius = displacement.norm();
            let kernel = cubic_kernel_2d(radius, hsml)?;
            kernel_sum += kernel.weight;
            if radius > 0.0 {
                let velocity_difference = velocities[particle] - velocities[neighbor];
                divergence_sum -=
                    kernel.radial_derivative * displacement.dot(velocity_difference) / radius;
            }
        }
        let divergence = if kernel_sum > 0.0 {
            divergence_sum / kernel_sum * dhsml_factors[particle]
        } else {
            0.0
        };
        if !divergence.is_finite() {
            return Err(GeometryError::NonFiniteResult {
                index: particle,
                field: "particle velocity divergence",
                value: divergence,
            });
        }
        output.push(divergence);
    }
    Ok(output)
}

/// Run one public-C two-dimensional density/smoothing-length pass.
///
/// This is the `NUMDIMS == 2` specialization of the bracketed Newton-like
/// iteration in `hydro/density.c`. The supplied values are the smoothing
/// lengths retained from the preceding density pass; this function
/// deliberately does not reproduce the gravitational-tree seed generation.
/// As in [`density_at_hsml_2d`], the returned MFM density is
/// `m_i * sum_j W_ij(H_i)`.
///
/// # Errors
///
/// Returns an error for invalid particle columns or neighbor constraints, a
/// singular target neighborhood, or failure to converge within the public
/// iteration limit.
pub fn solve_public_c_smoothing_lengths_from_seeds_2d(
    positions: &[Vector2],
    masses: &[f64],
    seeds: &[f64],
    domain: Box2d,
    desired_neighbors: f64,
    tolerance: f64,
) -> Result<Vec<AdaptiveDensityEstimate>, GeometryError> {
    validate_particle_columns(positions, masses, seeds, domain)?;
    if !desired_neighbors.is_finite()
        || desired_neighbors <= 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || tolerance >= desired_neighbors
    {
        return Err(GeometryError::InvalidNeighborConstraint {
            desired: desired_neighbors,
            tolerance,
        });
    }
    if positions.is_empty() {
        return Ok(Vec::new());
    }

    // The cell layout is only an acceleration structure: queries remain exact
    // for radii larger than this construction scale as H grows during solving.
    let maximum_seed = seeds.iter().copied().fold(0.0_f64, f64::max);
    let neighbor_index = CellList2d::new(positions, domain, maximum_seed)?;
    seeds
        .iter()
        .copied()
        .enumerate()
        .map(|(index, seed)| {
            solve_public_c_particle_from_seed_2d(
                index,
                positions,
                masses,
                domain,
                desired_neighbors,
                tolerance,
                seed,
                &neighbor_index,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn solve_public_c_particle_from_seed_2d(
    index: usize,
    positions: &[Vector2],
    masses: &[f64],
    domain: Box2d,
    desired_neighbors: f64,
    base_tolerance: f64,
    seed: f64,
    neighbor_index: &CellList2d,
) -> Result<AdaptiveDensityEstimate, GeometryError> {
    let mut hsml = seed;
    let mut lower = 0.0_f64;
    let mut upper = 0.0_f64;
    let mut last = estimate_public_c_density_geometry_2d(
        index,
        positions,
        masses,
        hsml,
        domain,
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
        if (last.estimate.effective_neighbors - corrected_desired_neighbors).abs() <= tolerance
            || (lower > 0.0 && upper > 0.0 && upper - lower < 1.0e-3 * lower)
        {
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
        hsml = public_c_hsml_jump_2d(
            hsml,
            last.estimate,
            corrected_desired_neighbors,
            iteration,
            lower,
            upper,
        );
        last = estimate_public_c_density_geometry_2d(
            index,
            positions,
            masses,
            hsml,
            domain,
            neighbor_index,
        )?;
    }

    Err(GeometryError::SmoothingLengthDidNotConverge {
        index,
        lower: (lower > 0.0).then_some(lower),
        upper: (upper > 0.0).then_some(upper),
        effective_neighbors: last.estimate.effective_neighbors,
    })
}

struct PublicCDensityGeometry2d {
    estimate: DensityEstimate,
    face_closure_error: f64,
}

#[allow(clippy::similar_names)]
fn estimate_public_c_density_geometry_2d(
    index: usize,
    positions: &[Vector2],
    masses: &[f64],
    hsml: f64,
    domain: Box2d,
    neighbor_index: &CellList2d,
) -> Result<PublicCDensityGeometry2d, GeometryError> {
    let position = positions[index];
    let mut kernel_sum = 0.0;
    let mut derivative_sum = 0.0;
    let mut moment_xx = 0.0;
    let mut moment_xy = 0.0;
    let mut moment_yy = 0.0;
    let mut first_moment = Vector2::ZERO;
    for neighbor in neighbor_index.neighbors_within(position, hsml)? {
        let displacement = domain.displacement(position, positions[neighbor])?;
        let radius = displacement.norm();
        let kernel = cubic_kernel_2d(radius, hsml)?;
        kernel_sum += kernel.weight;
        derivative_sum +=
            -(2.0 * kernel.weight / hsml + (radius / hsml) * kernel.radial_derivative);
        if radius > 0.0 {
            moment_xx += kernel.weight * displacement.x * displacement.x;
            moment_xy += kernel.weight * displacement.x * displacement.y;
            moment_yy += kernel.weight * displacement.y * displacement.y;
            first_moment += displacement * kernel.weight;
        }
    }
    if kernel_sum <= 0.0 {
        return Err(GeometryError::NonFiniteResult {
            index,
            field: "kernel sum",
            value: kernel_sum,
        });
    }

    let raw_response = derivative_sum * hsml / (2.0 * kernel_sum);
    let estimate = DensityEstimate {
        density: masses[index] * kernel_sum,
        effective_neighbors: PI * hsml * hsml * kernel_sum,
        dhsml_factor: if raw_response > -0.9 {
            1.0 / (1.0 + raw_response)
        } else {
            1.0
        },
    };

    // This reproduces the target-local FaceClosureError used by density.c to
    // increase the requested neighbor count near an asymmetric neighborhood.
    let inverse = invert_symmetric_moment(index, moment_xx, moment_xy, moment_yy)?.matrix;
    let face_closure_error =
        public_c_face_closure_error_2d(kernel_sum, moment_xx + moment_yy, inverse, first_moment);

    for (field, value) in [
        ("density", estimate.density),
        ("effective neighbors", estimate.effective_neighbors),
        ("smoothing-length factor", estimate.dhsml_factor),
        ("face closure error", face_closure_error),
    ] {
        if !value.is_finite() {
            return Err(GeometryError::NonFiniteResult {
                index,
                field,
                value,
            });
        }
    }
    Ok(PublicCDensityGeometry2d {
        estimate,
        face_closure_error,
    })
}

fn public_c_face_closure_error_2d(
    kernel_sum: f64,
    moment_trace: f64,
    inverse_moment: Matrix2,
    first_moment: Vector2,
) -> f64 {
    let volume = kernel_sum.recip();
    let characteristic_length = (volume * moment_trace).sqrt();
    let one_sided_area = inverse_moment.mul_vector(first_moment) * (2.0 * volume);
    // density.c divides the component L1 norm by NUMDIMS, then divides by
    // 2*NUMDIMS*dx^(NUMDIMS-1): for NUMDIMS=2 this is 8*dx.
    (one_sided_area.x.abs() + one_sided_area.y.abs()) / (8.0 * characteristic_length)
}

#[allow(clippy::float_cmp)]
fn public_c_hsml_jump_2d(
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
            let mut jump = estimate.dhsml_factor * (desired_neighbors / neighbors).ln() / 2.0;
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
                hsml = hsml.min(upper / jump_factor).max(lower * jump_factor);
            }
        } else {
            hsml = hsml.clamp(lower, upper);
            hsml = (hsml * lower * upper).powf(1.0 / 3.0);
        }
        return hsml;
    }

    let mut limited_log_jump = if neighbors > 1.0 {
        (desired_neighbors / neighbors).ln() / 2.0
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
            hsml *= jump.min(limited_log_jump + 0.231).exp();
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
            hsml *= jump.max(limited_log_jump - 0.231).exp();
        } else {
            hsml *= limited_log_jump.exp();
        }
    }
    hsml
}

/// Inverted MLS moment and its conditioning metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseMoment2d {
    pub matrix: Matrix2,
    /// Spectral condition number after any regularization.
    pub condition_number: f64,
    /// Diagonal value added to the raw moment before inversion.
    pub diagonal_regularization: f64,
}

/// Build and invert each target-kernel MLS moment.
///
/// The raw moment is `sum_j W_ij dx_ij outer dx_ij`. Following
/// `hydro/density.c`, matrices above condition number `1e4` receive a growing
/// isotropic diagonal term before inversion. The returned metadata makes that
/// intervention observable to callers.
///
/// # Errors
///
/// Returns an error for invalid geometry, a neighborhood with zero moment
/// trace, or non-finite arithmetic.
pub fn inverse_moments_2d(
    positions: &[Vector2],
    smoothing_lengths: &[f64],
    domain: Box2d,
) -> Result<Vec<InverseMoment2d>, GeometryError> {
    validate_geometry_columns(positions, smoothing_lengths, domain)?;
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    let maximum_support = smoothing_lengths.iter().copied().fold(0.0_f64, f64::max);
    let index = CellList2d::new(positions, domain, maximum_support)?;
    let mut output = Vec::with_capacity(positions.len());
    for (particle, (&position, &hsml)) in positions.iter().zip(smoothing_lengths).enumerate() {
        let mut xx = 0.0;
        let mut xy = 0.0;
        let mut yy = 0.0;
        for neighbor in index.neighbors_within(position, hsml)? {
            let displacement = domain.displacement(position, positions[neighbor])?;
            let radius = displacement.norm();
            if radius == 0.0 {
                continue;
            }
            let weight = cubic_kernel_2d(radius, hsml)?.weight;
            xx += weight * displacement.x * displacement.x;
            xy += weight * displacement.x * displacement.y;
            yy += weight * displacement.y * displacement.y;
        }
        output.push(invert_symmetric_moment(particle, xx, xy, yy)?);
    }
    Ok(output)
}

fn invert_symmetric_moment(
    index: usize,
    mut xx: f64,
    xy: f64,
    mut yy: f64,
) -> Result<InverseMoment2d, GeometryError> {
    let trace = xx + yy;
    if !trace.is_finite() || trace <= 0.0 || !xy.is_finite() {
        return Err(GeometryError::SingularMoment { index, xx, xy, yy });
    }
    let initial_increment = 1.05 * (trace / 2.0) / MOMENT_CONDITION_LIMIT;
    let mut increment = initial_increment;
    let mut total_regularization = 0.0;
    let mut condition = symmetric_condition_number(xx, xy, yy);
    for _ in 0..256 {
        if condition.is_finite() && condition < MOMENT_CONDITION_LIMIT {
            let determinant = xx.mul_add(yy, -(xy * xy));
            let matrix = Matrix2::new(
                yy / determinant,
                -xy / determinant,
                -xy / determinant,
                xx / determinant,
            );
            if matrix.is_finite() {
                return Ok(InverseMoment2d {
                    matrix,
                    condition_number: condition,
                    diagonal_regularization: total_regularization,
                });
            }
        }
        xx += increment;
        yy += increment;
        total_regularization += increment;
        increment *= 1.2;
        condition = symmetric_condition_number(xx, xy, yy);
    }
    Err(GeometryError::SingularMoment { index, xx, xy, yy })
}

fn symmetric_condition_number(xx: f64, xy: f64, yy: f64) -> f64 {
    let trace = xx + yy;
    let discriminant = (xx - yy).hypot(2.0 * xy);
    let largest = 0.5 * (trace + discriminant);
    let smallest = 0.5 * (trace - discriminant);
    if smallest > 0.0 {
        largest / smallest
    } else {
        f64::INFINITY
    }
}

/// Unlimited scalar MLS gradients.
///
/// Only neighbors inside the target particle's support contribute, exactly as
/// in the C target-kernel gradient numerator.
///
/// # Errors
///
/// Returns an error for invalid values or geometry.
pub fn scalar_gradients_at_hsml_2d(
    positions: &[Vector2],
    values: &[f64],
    smoothing_lengths: &[f64],
    domain: Box2d,
) -> Result<Vec<Vector2>, GeometryError> {
    validate_values(values, positions.len(), "scalar_values")?;
    let moments = inverse_moments_2d(positions, smoothing_lengths, domain)?;
    gradient_numerators(positions, smoothing_lengths, domain, |center, neighbor| {
        Vector2::new(values[neighbor] - values[center], 0.0)
    })?
    .into_iter()
    .zip(moments)
    .enumerate()
    .map(|(index, (numerator, moment))| {
        let gradient = moment
            .matrix
            .mul_vector(Vector2::new(numerator.xx, numerator.xy));
        if gradient.is_finite() {
            Ok(gradient)
        } else {
            Err(GeometryError::NonFiniteResult {
                index,
                field: "scalar gradient",
                value: f64::NAN,
            })
        }
    })
    .collect()
}

/// Evaluate several scalar MLS gradients in one target-neighbor traversal
/// while reusing caller-supplied inverse moments.
///
/// # Errors
///
/// Returns an error for mismatched/non-finite fields, moments, or geometry.
pub fn scalar_gradients_batch_with_moments_2d(
    positions: &[Vector2],
    fields: &[&[f64]],
    smoothing_lengths: &[f64],
    domain: Box2d,
    moments: &[InverseMoment2d],
) -> Result<Vec<Vec<Vector2>>, GeometryError> {
    validate_geometry_columns(positions, smoothing_lengths, domain)?;
    if moments.len() != positions.len() {
        return Err(GeometryError::MismatchedLength {
            field: "inverse_moments",
            expected: positions.len(),
            actual: moments.len(),
        });
    }
    for field in fields {
        validate_values(field, positions.len(), "scalar_values")?;
    }
    if positions.is_empty() {
        return Ok(vec![Vec::new(); fields.len()]);
    }
    let maximum_support = smoothing_lengths.iter().copied().fold(0.0_f64, f64::max);
    let index = CellList2d::new(positions, domain, maximum_support)?;
    let mut numerators = vec![vec![Vector2::ZERO; positions.len()]; fields.len()];
    for (center, (&position, &hsml)) in positions.iter().zip(smoothing_lengths).enumerate() {
        for neighbor in index.neighbors_within(position, hsml)? {
            if neighbor == center {
                continue;
            }
            let displacement = domain.displacement(position, positions[neighbor])?;
            let weight = cubic_kernel_2d(displacement.norm(), hsml)?.weight;
            for (field_index, values) in fields.iter().enumerate() {
                let delta = values[neighbor] - values[center];
                numerators[field_index][center].x -= weight * displacement.x * delta;
                numerators[field_index][center].y -= weight * displacement.y * delta;
            }
        }
    }
    numerators
        .into_iter()
        .map(|field| {
            field
                .into_iter()
                .zip(moments)
                .enumerate()
                .map(|(index, (numerator, moment))| {
                    let gradient = moment.matrix.mul_vector(numerator);
                    if gradient.is_finite() {
                        Ok(gradient)
                    } else {
                        Err(GeometryError::NonFiniteResult {
                            index,
                            field: "scalar gradient",
                            value: f64::NAN,
                        })
                    }
                })
                .collect()
        })
        .collect()
}

/// Unlimited vector MLS gradients.
///
/// Matrix row one is `grad(value.x)` and row two is `grad(value.y)`.
///
/// # Errors
///
/// Returns an error for invalid values or geometry.
pub fn vector_gradients_at_hsml_2d(
    positions: &[Vector2],
    values: &[Vector2],
    smoothing_lengths: &[f64],
    domain: Box2d,
) -> Result<Vec<Matrix2>, GeometryError> {
    if values.len() != positions.len() {
        return Err(GeometryError::MismatchedLength {
            field: "vector_values",
            expected: positions.len(),
            actual: values.len(),
        });
    }
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GeometryError::InvalidPoint {
                index: Some(index),
                field: "vector_value",
                value: f64::NAN,
            });
        }
    }
    let moments = inverse_moments_2d(positions, smoothing_lengths, domain)?;
    gradient_numerators(positions, smoothing_lengths, domain, |center, neighbor| {
        values[neighbor] - values[center]
    })?
    .into_iter()
    .zip(moments)
    .enumerate()
    .map(|(index, (numerator, moment))| {
        let gradient_x = moment
            .matrix
            .mul_vector(Vector2::new(numerator.xx, numerator.xy));
        let gradient_y = moment
            .matrix
            .mul_vector(Vector2::new(numerator.yx, numerator.yy));
        let gradient = Matrix2::new(gradient_x.x, gradient_x.y, gradient_y.x, gradient_y.y);
        if gradient.is_finite() {
            Ok(gradient)
        } else {
            Err(GeometryError::NonFiniteResult {
                index,
                field: "vector gradient",
                value: f64::NAN,
            })
        }
    })
    .collect()
}

fn gradient_numerators(
    positions: &[Vector2],
    smoothing_lengths: &[f64],
    domain: Box2d,
    delta: impl Fn(usize, usize) -> Vector2,
) -> Result<Vec<Matrix2>, GeometryError> {
    validate_geometry_columns(positions, smoothing_lengths, domain)?;
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    let maximum_support = smoothing_lengths.iter().copied().fold(0.0_f64, f64::max);
    let index = CellList2d::new(positions, domain, maximum_support)?;
    let mut output = Vec::with_capacity(positions.len());
    for (center, (&position, &hsml)) in positions.iter().zip(smoothing_lengths).enumerate() {
        let mut numerator = Matrix2::ZERO;
        for neighbor in index.neighbors_within(position, hsml)? {
            if neighbor == center {
                continue;
            }
            let displacement = domain.displacement(position, positions[neighbor])?;
            let weight = cubic_kernel_2d(displacement.norm(), hsml)?.weight;
            let value_delta = delta(center, neighbor);
            // C uses (x_j-x_i) * (f_j-f_i); displacement is x_i-x_j.
            numerator.xx -= weight * displacement.x * value_delta.x;
            numerator.xy -= weight * displacement.y * value_delta.x;
            numerator.yx -= weight * displacement.x * value_delta.y;
            numerator.yy -= weight * displacement.y * value_delta.y;
        }
        if !numerator.is_finite() {
            return Err(GeometryError::NonFiniteResult {
                index: center,
                field: "gradient numerator",
                value: f64::NAN,
            });
        }
        output.push(numerator);
    }
    Ok(output)
}

/// Particle geometry needed for an MFM face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshlessPoint2d {
    pub position: Vector2,
    pub mass: f64,
    pub density: f64,
    pub smoothing_length: f64,
    pub inverse_moment: Matrix2,
    /// Spectral condition number associated with `inverse_moment`.
    pub condition_number: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaceGeometryPath2d {
    Mls,
    /// Radial-SPH fallback used by `compute_finitevol_faces.h`.
    RsphFallback,
}

/// Default non-cosmological two-dimensional MFM face geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshlessFace2d {
    /// Oriented from particle `j` toward particle `i`.
    pub area_vector: Vector2,
    pub area: f64,
    pub unit_normal: Vector2,
    /// Vector from particle `i` to the default midpoint face.
    pub offset_from_i: Vector2,
    /// Vector from particle `j` to the default midpoint face.
    pub offset_from_j: Vector2,
    /// Whether the consistent MLS face or the public C stability fallback won.
    pub path: FaceGeometryPath2d,
}

/// Construct the public baseline's vector MFM face.
///
/// # Errors
///
/// Returns an error for invalid points, coincident points, a pair outside both
/// supports, or non-finite face arithmetic.
pub fn meshless_face_geometry_2d(
    i: MeshlessPoint2d,
    j: MeshlessPoint2d,
    domain: Box2d,
) -> Result<MeshlessFace2d, GeometryError> {
    validate_meshless_point("i", i, domain)?;
    validate_meshless_point("j", j, domain)?;
    let displacement = domain.displacement(i.position, j.position)?;
    let distance = displacement.norm();
    if distance <= 0.0 || (distance >= i.smoothing_length && distance >= j.smoothing_length) {
        return Err(GeometryError::InvalidFacePair {
            distance,
            hsml_i: i.smoothing_length,
            hsml_j: j.smoothing_length,
        });
    }
    let kernel_i = cubic_kernel_2d(distance, i.smoothing_length)?;
    let kernel_j = cubic_kernel_2d(distance, j.smoothing_length)?;
    let volume_i = i.mass / i.density;
    let volume_j = j.mass / j.density;
    let relative_volume_jump_per_dimension =
        (volume_i - volume_j).abs() / volume_i.min(volume_j) / 2.0;
    let (weight_i, weight_j) = if relative_volume_jump_per_dimension > 1.25 {
        let denominator = volume_i * kernel_i.weight + volume_j * kernel_j.weight;
        let centered = volume_i * volume_j * (kernel_i.weight + kernel_j.weight) / denominator;
        (centered, centered)
    } else {
        (volume_i, volume_j)
    };
    let mut area_vector = i.inverse_moment.mul_vector(displacement) * (kernel_i.weight * weight_i)
        + j.inverse_moment.mul_vector(displacement) * (kernel_j.weight * weight_j);
    let condition_fallback = i
        .condition_number
        .mul_add(i.condition_number, j.condition_number * j.condition_number)
        > 1.0e12 + 1.0e6;
    let mut path = FaceGeometryPath2d::Mls;
    if condition_fallback || area_vector.dot(displacement) < 0.0 {
        let radial_factor = -(weight_i * volume_i * kernel_i.radial_derivative
            + weight_j * volume_j * kernel_j.radial_derivative)
            / distance;
        area_vector = displacement * radial_factor;
        path = FaceGeometryPath2d::RsphFallback;
    }
    let area = area_vector.norm();
    if !volume_i.is_finite()
        || !volume_j.is_finite()
        || !weight_i.is_finite()
        || !weight_j.is_finite()
        || !area_vector.is_finite()
        || !area.is_finite()
        || area <= 0.0
    {
        return Err(GeometryError::NonFiniteFaceGeometry {
            area_vector,
            volume_i,
            volume_j,
        });
    }
    Ok(MeshlessFace2d {
        area_vector,
        area,
        unit_normal: area_vector / area,
        offset_from_i: displacement * -0.5,
        offset_from_j: displacement * 0.5,
        path,
    })
}

/// Per-particle face closure measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FaceClosure2d {
    /// Sum of oriented pair-face vectors incident on this particle.
    pub net_area_vector: Vector2,
    /// Sum of incident face magnitudes.
    pub summed_face_area: f64,
    /// `|sum A| / sum |A|`, or zero when no valid faces exist.
    pub relative_net_area: f64,
    /// The density-loop `FaceClosureError` diagnostic from the public C code.
    pub legacy_dimensionless_leak: f64,
}

/// Compute both actual pair-face closure and the C density-loop leak estimate.
///
/// # Errors
///
/// Returns an error for invalid geometry, moments, density, or a face.
pub fn face_closure_diagnostics_2d(
    positions: &[Vector2],
    masses: &[f64],
    smoothing_lengths: &[f64],
    domain: Box2d,
) -> Result<Vec<FaceClosure2d>, GeometryError> {
    validate_particle_columns(positions, masses, smoothing_lengths, domain)?;
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    let density = density_at_hsml_2d(positions, masses, smoothing_lengths, domain)?;
    let moments = inverse_moments_2d(positions, smoothing_lengths, domain)?;
    let mut diagnostics = vec![FaceClosure2d::default(); positions.len()];
    for pair in interacting_pairs_2d(positions, smoothing_lengths, domain)? {
        let point = |index: usize| MeshlessPoint2d {
            position: positions[index],
            mass: masses[index],
            density: density[index].density,
            smoothing_length: smoothing_lengths[index],
            inverse_moment: moments[index].matrix,
            condition_number: moments[index].condition_number,
        };
        let face = meshless_face_geometry_2d(point(pair.i), point(pair.j), domain)?;
        diagnostics[pair.i].net_area_vector += face.area_vector;
        diagnostics[pair.j].net_area_vector -= face.area_vector;
        diagnostics[pair.i].summed_face_area += face.area;
        diagnostics[pair.j].summed_face_area += face.area;
    }

    let maximum_support = smoothing_lengths.iter().copied().fold(0.0_f64, f64::max);
    let index = CellList2d::new(positions, domain, maximum_support)?;
    for particle in 0..positions.len() {
        let mut kernel_sum = 0.0;
        let mut first_moment = Vector2::ZERO;
        let mut second_moment_trace = 0.0;
        for neighbor in index.neighbors_within(positions[particle], smoothing_lengths[particle])? {
            let displacement = domain.displacement(positions[particle], positions[neighbor])?;
            let kernel = cubic_kernel_2d(displacement.norm(), smoothing_lengths[particle])?;
            kernel_sum += kernel.weight;
            if displacement.squared_norm() > 0.0 {
                first_moment += displacement * kernel.weight;
                second_moment_trace += kernel.weight * displacement.squared_norm();
            }
        }
        if kernel_sum <= 0.0 || second_moment_trace <= 0.0 {
            return Err(GeometryError::NonFiniteResult {
                index: particle,
                field: "closure moments",
                value: kernel_sum.min(second_moment_trace),
            });
        }
        let volume = kernel_sum.recip();
        let characteristic_length = (volume * second_moment_trace).sqrt();
        let one_sided = moments[particle].matrix.mul_vector(first_moment) * (2.0 * volume);
        diagnostics[particle].legacy_dimensionless_leak =
            (one_sided.x.abs() + one_sided.y.abs()) / (8.0 * characteristic_length);
        diagnostics[particle].relative_net_area = if diagnostics[particle].summed_face_area > 0.0 {
            diagnostics[particle].net_area_vector.norm() / diagnostics[particle].summed_face_area
        } else {
            0.0
        };
    }
    Ok(diagnostics)
}

fn validate_positions(positions: &[Vector2], domain: Box2d) -> Result<(), GeometryError> {
    for (index, &position) in positions.iter().enumerate() {
        domain.validate_wrapped(position, Some(index))?;
    }
    Ok(())
}

fn validate_geometry_columns(
    positions: &[Vector2],
    smoothing_lengths: &[f64],
    domain: Box2d,
) -> Result<(), GeometryError> {
    validate_positions(positions, domain)?;
    if smoothing_lengths.len() != positions.len() {
        return Err(GeometryError::MismatchedLength {
            field: "smoothing_lengths",
            expected: positions.len(),
            actual: smoothing_lengths.len(),
        });
    }
    for (index, &hsml) in smoothing_lengths.iter().enumerate() {
        if !hsml.is_finite() || hsml <= 0.0 {
            return Err(GeometryError::InvalidPoint {
                index: Some(index),
                field: "smoothing_length",
                value: hsml,
            });
        }
    }
    Ok(())
}

fn validate_particle_columns(
    positions: &[Vector2],
    masses: &[f64],
    smoothing_lengths: &[f64],
    domain: Box2d,
) -> Result<(), GeometryError> {
    validate_geometry_columns(positions, smoothing_lengths, domain)?;
    if masses.len() != positions.len() {
        return Err(GeometryError::MismatchedLength {
            field: "masses",
            expected: positions.len(),
            actual: masses.len(),
        });
    }
    for (index, &mass) in masses.iter().enumerate() {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(GeometryError::InvalidPoint {
                index: Some(index),
                field: "mass",
                value: mass,
            });
        }
    }
    Ok(())
}

fn validate_values(
    values: &[f64],
    expected: usize,
    field: &'static str,
) -> Result<(), GeometryError> {
    if values.len() != expected {
        return Err(GeometryError::MismatchedLength {
            field,
            expected,
            actual: values.len(),
        });
    }
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(GeometryError::InvalidPoint {
                index: Some(index),
                field,
                value,
            });
        }
    }
    Ok(())
}

fn validate_meshless_point(
    side: &'static str,
    point: MeshlessPoint2d,
    domain: Box2d,
) -> Result<(), GeometryError> {
    domain.validate_wrapped(point.position, None)?;
    for (field, value) in [
        ("mass", point.mass),
        ("density", point.density),
        ("smoothing_length", point.smoothing_length),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(GeometryError::InvalidFaceInput { side, field, value });
        }
    }
    if !point.inverse_moment.is_finite() {
        return Err(GeometryError::InvalidFaceInput {
            side,
            field: "inverse_moment",
            value: f64::NAN,
        });
    }
    if !point.condition_number.is_finite() || point.condition_number < 1.0 {
        return Err(GeometryError::InvalidFaceInput {
            side,
            field: "condition_number",
            value: point.condition_number,
        });
    }
    Ok(())
}

/// Errors from two-dimensional meshless geometry.
#[derive(Clone, Debug, PartialEq)]
pub enum GeometryError {
    InvalidBoxLength {
        axis: &'static str,
        value: f64,
    },
    InvalidPoint {
        index: Option<usize>,
        field: &'static str,
        value: f64,
    },
    InvalidKernelInput {
        radius: f64,
        hsml: f64,
    },
    NonFiniteKernelResult {
        radius: f64,
        hsml: f64,
    },
    InvalidSearchRadius(f64),
    MismatchedLength {
        field: &'static str,
        expected: usize,
        actual: usize,
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
    SingularMoment {
        index: usize,
        xx: f64,
        xy: f64,
        yy: f64,
    },
    NonFiniteResult {
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
        area_vector: Vector2,
        volume_i: f64,
        volume_j: f64,
    },
}

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBoxLength { axis, value } => {
                write!(formatter, "invalid 2-D box {axis} length {value}")
            }
            Self::InvalidPoint {
                index,
                field,
                value,
            } => write!(formatter, "invalid {field}={value} at particle {index:?}"),
            Self::InvalidKernelInput { radius, hsml } => {
                write!(formatter, "invalid 2-D kernel radius={radius}, hsml={hsml}")
            }
            Self::NonFiniteKernelResult { radius, hsml } => write!(
                formatter,
                "non-finite 2-D kernel result for radius={radius}, hsml={hsml}"
            ),
            Self::InvalidSearchRadius(value) => {
                write!(formatter, "invalid 2-D neighbor search radius {value}")
            }
            Self::MismatchedLength {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} has length {actual}, expected particle count {expected}"
            ),
            Self::InvalidNeighborConstraint { desired, tolerance } => write!(
                formatter,
                "invalid 2-D neighbor constraint desired={desired}, tolerance={tolerance}"
            ),
            Self::SmoothingLengthDidNotConverge {
                index,
                lower,
                upper,
                effective_neighbors,
            } => write!(
                formatter,
                "particle {index} did not converge on 2-D smoothing length: \
                 bounds={lower:?}..{upper:?}, effective neighbors={effective_neighbors}"
            ),
            Self::SingularMoment { index, xx, xy, yy } => write!(
                formatter,
                "singular 2-D MLS moment at particle {index}: [{xx}, {xy}; {xy}, {yy}]"
            ),
            Self::NonFiniteResult {
                index,
                field,
                value,
            } => write!(
                formatter,
                "non-finite 2-D geometry result {field}={value} at particle {index}"
            ),
            Self::InvalidFaceInput { side, field, value } => {
                write!(formatter, "invalid 2-D face input {side}.{field}={value}")
            }
            Self::InvalidFacePair {
                distance,
                hsml_i,
                hsml_j,
            } => write!(
                formatter,
                "invalid 2-D face pair r={distance}, H_i={hsml_i}, H_j={hsml_j}"
            ),
            Self::NonFiniteFaceGeometry {
                area_vector,
                volume_i,
                volume_j,
            } => write!(
                formatter,
                "invalid 2-D face A={area_vector:?}, V_i={volume_i}, V_j={volume_j}"
            ),
        }
    }
}

impl Error for GeometryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::cast_precision_loss)]
    fn square_lattice(side: usize, domain: Box2d) -> (Vec<Vector2>, Vec<f64>, Vec<f64>) {
        let lengths = domain.lengths();
        let dx = lengths.x / side as f64;
        let dy = lengths.y / side as f64;
        let mut positions = Vec::with_capacity(side * side);
        for y in 0..side {
            for x in 0..side {
                positions.push(Vector2::new((x as f64 + 0.5) * dx, (y as f64 + 0.5) * dy));
            }
        }
        let masses = vec![dx * dy; positions.len()];
        let smoothing_lengths = vec![2.2 * dx.max(dy); positions.len()];
        (positions, masses, smoothing_lengths)
    }

    #[test]
    fn cubic_kernel_has_exact_2d_normalization() {
        let hsml = 2.7;
        let bins = 200_000;
        let dr = hsml / f64::from(bins);
        let mut integral = 0.0;
        for bin in 0..bins {
            let radius = (f64::from(bin) + 0.5) * dr;
            integral += 2.0 * PI * radius * cubic_kernel_2d(radius, hsml).unwrap().weight * dr;
        }
        assert!((integral - 1.0).abs() < 2.0e-10, "{integral}");
        assert!(cubic_kernel_2d(hsml, hsml).unwrap().weight.abs() < f64::EPSILON);
    }

    #[test]
    fn rectangular_wrap_and_minimum_image_preserve_half_box_sign() {
        let domain = Box2d::new(4.0, 0.25).unwrap();
        assert_eq!(
            domain.wrap(Vector2::new(-0.1, 0.3)).unwrap(),
            Vector2::new(3.9, 0.049_999_999_999_999_99)
        );
        assert_eq!(
            domain
                .displacement(Vector2::new(0.1, 0.24), Vector2::new(3.9, 0.01))
                .unwrap(),
            Vector2::new(0.200_000_000_000_000_18, -0.020_000_000_000_000_018)
        );
        assert!(
            (domain
                .displacement(Vector2::new(2.0, 0.0), Vector2::ZERO)
                .unwrap()
                .x
                - 2.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn cell_list_matches_brute_force_queries_and_union_pairs() {
        let domain = Box2d::new(4.0, 0.25).unwrap();
        let positions = vec![
            Vector2::new(0.01, 0.01),
            Vector2::new(3.99, 0.24),
            Vector2::new(2.0, 0.125),
            Vector2::new(2.08, 0.02),
            Vector2::new(1.0, 0.2),
        ];
        let hsml = vec![0.04, 0.02, 0.07, 0.14, 0.3];
        let index = CellList2d::new(&positions, domain, 0.3).unwrap();
        for (&target, &radius) in positions.iter().zip(&hsml) {
            let mut expected: Vec<_> = positions
                .iter()
                .enumerate()
                .filter_map(|(neighbor, &position)| {
                    (domain.displacement(target, position).unwrap().norm() < radius)
                        .then_some(neighbor)
                })
                .collect();
            expected.sort_unstable();
            assert_eq!(index.neighbors_within(target, radius).unwrap(), expected);
        }
        let pairs = interacting_pairs_2d(&positions, &hsml, domain).unwrap();
        let mut expected = Vec::new();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let distance = domain
                    .displacement(positions[i], positions[j])
                    .unwrap()
                    .norm();
                if distance > 0.0 && (distance < hsml[i] || distance < hsml[j]) {
                    expected.push((i, j));
                }
            }
        }
        assert_eq!(
            pairs
                .iter()
                .map(|pair| (pair.i, pair.j))
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn tiny_search_scale_does_not_allocate_a_quadratic_grid() {
        let domain = Box2d::new(1.0e6, 1.0e-3).unwrap();
        let positions = vec![
            Vector2::new(0.1, 0.000_1),
            Vector2::new(0.2, 0.000_2),
            Vector2::new(0.3, 0.000_3),
        ];
        let index = CellList2d::new(&positions, domain, 1.0e-30).unwrap();
        let [nx, ny] = index.cell_counts();
        assert!(nx * ny <= positions.len());
        assert_eq!(
            index.neighbors_within(positions[0], 0.5).unwrap().len(),
            positions.len()
        );
    }

    #[test]
    fn unequal_mass_mfm_density_uses_target_mass_times_number_density() {
        let domain = Box2d::new(1.0, 1.0).unwrap();
        let positions = vec![Vector2::new(0.25, 0.5), Vector2::new(0.35, 0.5)];
        let masses = vec![1.0, 3.0];
        let hsml = vec![0.4, 0.4];
        let density = density_at_hsml_2d(&positions, &masses, &hsml, domain).unwrap();
        let self_weight = cubic_kernel_2d(0.0, 0.4).unwrap().weight;
        let neighbor_weight = cubic_kernel_2d(0.1, 0.4).unwrap().weight;
        let number_density = self_weight + neighbor_weight;
        assert!((density[0].density - number_density).abs() < 1.0e-12);
        assert!((density[1].density - 3.0 * number_density).abs() < 1.0e-12);
    }

    #[test]
    fn public_c_solver_hits_2d_neighbor_constraint_and_preserves_mfm_density() {
        let domain = Box2d::new(1.0, 1.0).unwrap();
        let (positions, masses, seeds) = square_lattice(16, domain);
        let initial = density_at_hsml_2d(&positions, &masses, &seeds, domain).unwrap();
        let desired = 20.0;
        let tolerance = 0.05;
        assert!(
            initial
                .iter()
                .any(|estimate| (estimate.effective_neighbors - desired).abs() > tolerance)
        );

        let solved = solve_public_c_smoothing_lengths_from_seeds_2d(
            &positions, &masses, &seeds, domain, desired, tolerance,
        )
        .unwrap();
        for (particle, &mass) in solved.iter().zip(&masses) {
            assert!(
                (particle.estimate.effective_neighbors - desired).abs() <= tolerance,
                "{particle:?}"
            );
            let kernel_sum =
                particle.estimate.effective_neighbors / (PI * particle.smoothing_length.powi(2));
            assert!((particle.estimate.density - mass * kernel_sum).abs() < 1.0e-12);
        }
    }

    #[test]
    fn public_c_solver_rejects_invalid_neighbor_constraint() {
        let domain = Box2d::new(1.0, 1.0).unwrap();
        let error = solve_public_c_smoothing_lengths_from_seeds_2d(
            &[Vector2::new(0.5, 0.5)],
            &[1.0],
            &[0.1],
            domain,
            0.0,
            0.05,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            GeometryError::InvalidNeighborConstraint { .. }
        ));
    }

    #[test]
    fn public_c_face_closure_keeps_both_numdims_divisors() {
        let error = public_c_face_closure_error_2d(
            1.0,
            2.0,
            Matrix2::new(1.0, 0.0, 0.0, 1.0),
            Vector2::new(0.1, -0.2),
        );
        let expected = 0.6 / (8.0 * 2.0_f64.sqrt());
        assert!((error - expected).abs() < 1.0e-15, "{error}");
    }

    #[test]
    fn mls_exactly_recovers_linear_scalar_and_vector_fields() {
        let domain = Box2d::new(10.0, 10.0).unwrap();
        let positions = vec![
            Vector2::new(4.5, 4.5),
            Vector2::new(5.0, 4.5),
            Vector2::new(5.5, 4.5),
            Vector2::new(4.5, 5.0),
            Vector2::new(5.0, 5.0),
            Vector2::new(5.5, 5.0),
            Vector2::new(4.5, 5.5),
            Vector2::new(5.0, 5.5),
            Vector2::new(5.5, 5.5),
        ];
        let hsml = vec![1.1; positions.len()];
        let scalar: Vec<_> = positions
            .iter()
            .map(|p| 3.0 * p.x - 2.0 * p.y + 7.0)
            .collect();
        let vector: Vec<_> = positions
            .iter()
            .map(|p| Vector2::new(3.0 * p.x - 2.0 * p.y, -p.x + 4.0 * p.y))
            .collect();
        let scalar_gradient =
            scalar_gradients_at_hsml_2d(&positions, &scalar, &hsml, domain).unwrap();
        let second_scalar: Vec<_> = positions.iter().map(|p| -p.x + 4.0 * p.y).collect();
        let moments = inverse_moments_2d(&positions, &hsml, domain).unwrap();
        let batch = scalar_gradients_batch_with_moments_2d(
            &positions,
            &[&scalar, &second_scalar],
            &hsml,
            domain,
            &moments,
        )
        .unwrap();
        assert_eq!(batch[0], scalar_gradient);
        let vector_gradient =
            vector_gradients_at_hsml_2d(&positions, &vector, &hsml, domain).unwrap();
        for gradient in scalar_gradient {
            assert!((gradient.x - 3.0).abs() < 2.0e-13, "{gradient:?}");
            assert!((gradient.y + 2.0).abs() < 2.0e-13, "{gradient:?}");
        }
        for gradient in vector_gradient {
            assert!((gradient.xx - 3.0).abs() < 2.0e-13, "{gradient:?}");
            assert!((gradient.xy + 2.0).abs() < 2.0e-13, "{gradient:?}");
            assert!((gradient.yx + 1.0).abs() < 2.0e-13, "{gradient:?}");
            assert!((gradient.yy - 4.0).abs() < 2.0e-13, "{gradient:?}");
        }
        for gradient in &batch[1] {
            assert!((gradient.x + 1.0).abs() < 2.0e-13, "{gradient:?}");
            assert!((gradient.y - 4.0).abs() < 2.0e-13, "{gradient:?}");
        }
    }

    #[test]
    fn collinear_moment_regularization_is_explicit() {
        let domain = Box2d::new(4.0, 1.0).unwrap();
        let positions = vec![
            Vector2::new(1.0, 0.5),
            Vector2::new(1.2, 0.5),
            Vector2::new(0.8, 0.5),
        ];
        let moments = inverse_moments_2d(&positions, &[0.5; 3], domain).unwrap();
        assert!(moments.iter().all(|moment| {
            moment.diagonal_regularization > 0.0
                && moment.condition_number < MOMENT_CONDITION_LIMIT
                && moment.matrix.is_finite()
        }));
    }

    #[test]
    fn particle_divergence_matches_literal_density_loop_sum() {
        let domain = Box2d::new(2.0, 2.0).unwrap();
        let positions = vec![
            Vector2::new(0.8, 0.9),
            Vector2::new(1.1, 0.82),
            Vector2::new(0.93, 1.18),
            Vector2::new(1.24, 1.13),
        ];
        let velocities = vec![
            Vector2::new(-0.2, 0.4),
            Vector2::new(0.7, -0.1),
            Vector2::new(0.15, 0.9),
            Vector2::new(-0.6, 0.2),
        ];
        let hsml = vec![0.72, 0.64, 0.68, 0.75];
        let dhsml = vec![0.83, 1.07, 0.91, 1.14];
        let measured =
            particle_divergence_at_hsml_2d(&positions, &velocities, &hsml, &dhsml, domain).unwrap();
        for i in 0..positions.len() {
            let mut kernel_sum = 0.0;
            let mut numerator = 0.0;
            for j in 0..positions.len() {
                let displacement = domain.displacement(positions[i], positions[j]).unwrap();
                let radius = displacement.norm();
                if radius < hsml[i] {
                    let kernel = cubic_kernel_2d(radius, hsml[i]).unwrap();
                    kernel_sum += kernel.weight;
                    if radius > 0.0 {
                        numerator -= kernel.radial_derivative
                            * displacement.dot(velocities[i] - velocities[j])
                            / radius;
                    }
                }
            }
            let expected = numerator / kernel_sum * dhsml[i];
            assert!((measured[i] - expected).abs() < 2.0e-15);
        }
    }

    #[test]
    fn face_is_antisymmetric_under_particle_swap() {
        let domain = Box2d::new(1.0, 1.0).unwrap();
        let inverse = Matrix2::new(2.0, 0.1, 0.1, 3.0);
        let i = MeshlessPoint2d {
            position: Vector2::new(0.95, 0.4),
            mass: 1.0,
            density: 2.0,
            smoothing_length: 0.4,
            inverse_moment: inverse,
            condition_number: 1.5,
        };
        let j = MeshlessPoint2d {
            position: Vector2::new(0.05, 0.46),
            mass: 0.8,
            density: 1.5,
            smoothing_length: 0.3,
            inverse_moment: Matrix2::new(2.5, -0.2, -0.2, 2.2),
            condition_number: 1.5,
        };
        let forward = meshless_face_geometry_2d(i, j, domain).unwrap();
        let reverse = meshless_face_geometry_2d(j, i, domain).unwrap();
        assert!((forward.area_vector.x + reverse.area_vector.x).abs() < 1.0e-14);
        assert!((forward.area_vector.y + reverse.area_vector.y).abs() < 1.0e-14);
        assert!((forward.area - reverse.area).abs() < 1.0e-14);
        assert_eq!(forward.offset_from_i, reverse.offset_from_j);
        assert_eq!(forward.path, FaceGeometryPath2d::Mls);
    }

    #[test]
    fn non_positive_mls_face_uses_public_rsph_fallback() {
        let domain = Box2d::new(1.0, 1.0).unwrap();
        let point = |x, inverse_moment| MeshlessPoint2d {
            position: Vector2::new(x, 0.5),
            mass: 1.0,
            density: 1.0,
            smoothing_length: 0.5,
            inverse_moment,
            condition_number: 1.0,
        };
        let face = meshless_face_geometry_2d(
            point(0.6, Matrix2::new(-1.0, 0.0, 0.0, -1.0)),
            point(0.4, Matrix2::new(-1.0, 0.0, 0.0, -1.0)),
            domain,
        )
        .unwrap();
        assert_eq!(face.path, FaceGeometryPath2d::RsphFallback);
        assert!(face.area_vector.x > 0.0);
        assert!(face.area_vector.y.abs() < f64::EPSILON);
    }

    #[test]
    fn periodic_lattice_has_near_machine_face_closure() {
        let domain = Box2d::new(1.0, 1.0).unwrap();
        let (positions, masses, hsml) = square_lattice(8, domain);
        let diagnostics = face_closure_diagnostics_2d(&positions, &masses, &hsml, domain).unwrap();
        for closure in diagnostics {
            assert!(closure.relative_net_area < 2.0e-15, "{closure:?}");
            assert!(closure.legacy_dimensionless_leak < 2.0e-15, "{closure:?}");
        }
    }
}
