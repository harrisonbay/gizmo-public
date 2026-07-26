//! Coordinate-covariant two-dimensional interface around the local HLLD solver.
//!
//! [`crate::mhd::hlld_riemann`] solves a one-dimensional Riemann problem whose
//! normal is the local x axis. Meshless faces in two dimensions have arbitrary
//! normals, so both primitive vectors and returned flux vectors must be rotated
//! through the same orthonormal basis. Keeping this transformation in one
//! typed boundary prevents callers from accidentally rotating velocity but not
//! magnetic field, momentum, or the induction flux.

use std::error::Error;
use std::fmt;

use crate::mhd::{
    HlldOptions, HlldResult, IdealMhdFlux1d, IdealMhdPrimitive1d, MhdError, MhdRiemannMethod,
    Vector3, hlld_riemann,
};

/// Right-handed planar basis `(normal, tangent, z)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarBasis {
    normal_x: f64,
    normal_y: f64,
}

impl PlanarBasis {
    /// Construct a basis from any finite nonzero planar direction.
    ///
    /// The input is normalized deliberately: a meshless face-area vector can
    /// be supplied directly without leaking its magnitude into the Riemann
    /// state. The caller remains responsible for multiplying the returned flux
    /// by the face area exactly once.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero or non-finite direction.
    pub fn from_normal(normal_x: f64, normal_y: f64) -> Result<Self, Mhd2dError> {
        let norm = normal_x.hypot(normal_y);
        if !normal_x.is_finite() || !normal_y.is_finite() || !norm.is_finite() || norm == 0.0 {
            return Err(Mhd2dError::InvalidNormal { normal_x, normal_y });
        }
        Ok(Self {
            normal_x: normal_x / norm,
            normal_y: normal_y / norm,
        })
    }

    /// Unit normal in global Cartesian coordinates.
    #[must_use]
    pub const fn normal(self) -> [f64; 2] {
        [self.normal_x, self.normal_y]
    }

    /// Unit tangent `(-n_y, n_x)` in global Cartesian coordinates.
    #[must_use]
    pub const fn tangent(self) -> [f64; 2] {
        [-self.normal_y, self.normal_x]
    }

    /// Rotate a global vector into local `(normal, tangent, z)` components.
    #[must_use]
    pub fn to_local(self, vector: Vector3) -> Vector3 {
        Vector3::new(
            self.normal_x.mul_add(vector.x, self.normal_y * vector.y),
            (-self.normal_y).mul_add(vector.x, self.normal_x * vector.y),
            vector.z,
        )
    }

    /// Rotate local `(normal, tangent, z)` components into global coordinates.
    #[must_use]
    pub fn to_global(self, vector: Vector3) -> Vector3 {
        Vector3::new(
            self.normal_x.mul_add(vector.x, -self.normal_y * vector.y),
            self.normal_y.mul_add(vector.x, self.normal_x * vector.y),
            vector.z,
        )
    }

    #[must_use]
    fn primitive_to_local(self, primitive: IdealMhdPrimitive1d) -> IdealMhdPrimitive1d {
        IdealMhdPrimitive1d {
            density: primitive.density,
            velocity: self.to_local(primitive.velocity),
            gas_pressure: primitive.gas_pressure,
            magnetic: self.to_local(primitive.magnetic),
            cleaning_scalar: primitive.cleaning_scalar,
        }
    }
}

/// HLLD result with every vector expressed in the global Cartesian frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HlldResult2d {
    pub flux: IdealMhdFlux1d,
    pub method: MhdRiemannMethod,
    pub contact_speed: f64,
    /// Scalar normal speed of the moving face.
    pub face_normal_velocity: f64,
    pub star_total_pressure: f64,
    pub corrected_normal_b: f64,
    pub face_magnetic: Vector3,
    pub fast_speed_left: f64,
    pub fast_speed_right: f64,
    pub phi_mean: f64,
    pub phi_db: f64,
}

impl HlldResult2d {
    fn from_local(local: HlldResult, basis: PlanarBasis) -> Self {
        Self {
            flux: IdealMhdFlux1d {
                mass: local.flux.mass,
                momentum: basis.to_global(local.flux.momentum),
                total_energy: local.flux.total_energy,
                magnetic: basis.to_global(local.flux.magnetic),
            },
            method: local.method,
            contact_speed: local.contact_speed,
            face_normal_velocity: local.face_velocity,
            star_total_pressure: local.star_total_pressure,
            corrected_normal_b: local.corrected_normal_b,
            face_magnetic: basis.to_global(local.face_magnetic),
            fast_speed_left: local.fast_speed_left,
            fast_speed_right: local.fast_speed_right,
            phi_mean: local.phi_mean,
            phi_db: local.phi_db,
        }
    }
}

/// Errors from the planar HLLD coordinate boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mhd2dError {
    InvalidNormal { normal_x: f64, normal_y: f64 },
    Riemann(MhdError),
}

impl fmt::Display for Mhd2dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidNormal { normal_x, normal_y } => {
                write!(
                    formatter,
                    "invalid planar face normal ({normal_x}, {normal_y})"
                )
            }
            Self::Riemann(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for Mhd2dError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidNormal { .. } => None,
            Self::Riemann(error) => Some(error),
        }
    }
}

impl From<MhdError> for Mhd2dError {
    fn from(value: MhdError) -> Self {
        Self::Riemann(value)
    }
}

/// Solve an ideal-MHD interface with an arbitrary planar face normal.
///
/// Input velocity and magnetic vectors are global Cartesian components. All
/// returned vectors are rotated back to that same frame; scalar wave speeds
/// remain signed along the supplied normal.
///
/// # Errors
///
/// Returns an error for an invalid normal or any error from the local HLLD
/// solver.
pub fn hlld_riemann_2d(
    left: IdealMhdPrimitive1d,
    right: IdealMhdPrimitive1d,
    face_normal: [f64; 2],
    gamma: f64,
    options: HlldOptions,
) -> Result<HlldResult2d, Mhd2dError> {
    let basis = PlanarBasis::from_normal(face_normal[0], face_normal[1])?;
    let local = hlld_riemann(
        basis.primitive_to_local(left),
        basis.primitive_to_local(right),
        gamma,
        options,
    )?;
    Ok(HlldResult2d::from_local(local, basis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mhd::{DednerOptions, FluxFrame1d};

    fn state(rho: f64, pressure: f64, velocity: Vector3, magnetic: Vector3) -> IdealMhdPrimitive1d {
        IdealMhdPrimitive1d {
            density: rho,
            velocity,
            gas_pressure: pressure,
            magnetic,
            cleaning_scalar: 0.0,
        }
    }

    fn assert_close(left: f64, right: f64) {
        let tolerance = 2.0e-13 * left.abs().max(right.abs()).max(1.0);
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    fn assert_vector_close(left: Vector3, right: Vector3) {
        assert_close(left.x, right.x);
        assert_close(left.y, right.y);
        assert_close(left.z, right.z);
    }

    #[test]
    fn basis_round_trip_is_identity() {
        let basis = PlanarBasis::from_normal(3.0, 4.0).unwrap();
        let vector = Vector3::new(1.25, -8.5, 0.75);
        assert_vector_close(basis.to_global(basis.to_local(vector)), vector);
        assert_close(basis.normal()[0], 0.6);
        assert_close(basis.normal()[1], 0.8);
        assert_close(basis.tangent()[0], -0.8);
        assert_close(basis.tangent()[1], 0.6);
    }

    #[test]
    fn rejects_invalid_normals() {
        for normal in [[0.0, 0.0], [f64::NAN, 1.0], [1.0, f64::INFINITY]] {
            assert!(matches!(
                PlanarBasis::from_normal(normal[0], normal[1]),
                Err(Mhd2dError::InvalidNormal { .. })
            ));
        }
    }

    #[test]
    fn x_normal_is_exact_local_solver_boundary() {
        let left = state(
            1.0,
            1.0,
            Vector3::new(0.1, -0.2, 0.3),
            Vector3::new(0.75, 1.0, -0.1),
        );
        let right = state(
            0.125,
            0.1,
            Vector3::new(-0.4, 0.5, -0.6),
            Vector3::new(0.75, -1.0, 0.2),
        );
        let options = HlldOptions {
            frame: FluxFrame1d::Eulerian,
            dedner: Some(DednerOptions::default()),
            maximum_star_total_pressure: None,
        };
        let expected = hlld_riemann(left, right, 2.0, options).unwrap();
        let actual = hlld_riemann_2d(left, right, [2.0, 0.0], 2.0, options).unwrap();
        assert_close(actual.flux.mass, expected.flux.mass);
        assert_vector_close(actual.flux.momentum, expected.flux.momentum);
        assert_close(actual.flux.total_energy, expected.flux.total_energy);
        assert_vector_close(actual.flux.magnetic, expected.flux.magnetic);
        assert_vector_close(actual.face_magnetic, expected.face_magnetic);
        assert_close(actual.corrected_normal_b, expected.corrected_normal_b);
    }

    #[test]
    fn solve_is_covariant_under_planar_rotation() {
        let basis = PlanarBasis::from_normal(0.6, 0.8).unwrap();
        let local_left = state(
            1.0,
            1.0,
            Vector3::new(0.2, -0.7, 0.4),
            Vector3::new(0.75, 1.0, -0.2),
        );
        let local_right = state(
            0.125,
            0.1,
            Vector3::new(-0.1, 0.3, -0.5),
            Vector3::new(0.75, -1.0, 0.6),
        );
        let global_left = IdealMhdPrimitive1d {
            velocity: basis.to_global(local_left.velocity),
            magnetic: basis.to_global(local_left.magnetic),
            ..local_left
        };
        let global_right = IdealMhdPrimitive1d {
            velocity: basis.to_global(local_right.velocity),
            magnetic: basis.to_global(local_right.magnetic),
            ..local_right
        };
        let options = HlldOptions::default();
        let local = hlld_riemann(local_left, local_right, 2.0, options).unwrap();
        let global =
            hlld_riemann_2d(global_left, global_right, basis.normal(), 2.0, options).unwrap();
        assert_close(global.flux.mass, local.flux.mass);
        assert_vector_close(global.flux.momentum, basis.to_global(local.flux.momentum));
        assert_close(global.flux.total_energy, local.flux.total_energy);
        assert_vector_close(global.flux.magnetic, basis.to_global(local.flux.magnetic));
        assert_vector_close(global.face_magnetic, basis.to_global(local.face_magnetic));
        assert_close(global.contact_speed, local.contact_speed);
        assert_close(global.corrected_normal_b, local.corrected_normal_b);
    }
}
