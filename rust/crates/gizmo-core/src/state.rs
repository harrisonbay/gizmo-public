use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;

use crate::{CodeMass, Vector3};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParticleId(NonZeroU64);

impl ParticleId {
    #[must_use]
    pub const fn new(value: NonZeroU64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParticleIndex(usize);

impl ParticleIndex {
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub id: ParticleId,
    pub position: Vector3,
    pub velocity: Vector3,
    pub mass: CodeMass,
}

/// Core particle fields stored as structure-of-arrays.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParticleState {
    ids: Vec<ParticleId>,
    positions: Vec<Vector3>,
    velocities: Vec<Vector3>,
    masses: Vec<CodeMass>,
}

impl ParticleState {
    /// Construct checked structure-of-arrays storage from complete columns.
    ///
    /// # Errors
    ///
    /// Returns an error if column lengths differ or any physical field is
    /// non-finite; negative mass is also rejected.
    pub fn from_columns(
        ids: Vec<ParticleId>,
        positions: Vec<Vector3>,
        velocities: Vec<Vector3>,
        masses: Vec<CodeMass>,
    ) -> Result<Self, StateError> {
        let expected = ids.len();
        let lengths = [
            ("positions", positions.len()),
            ("velocities", velocities.len()),
            ("masses", masses.len()),
        ];
        if let Some(&(field, actual)) = lengths.iter().find(|(_, length)| *length != expected) {
            return Err(StateError::MismatchedColumnLength {
                field,
                expected,
                actual,
            });
        }

        let state = Self {
            ids,
            positions,
            velocities,
            masses,
        };
        for raw_index in 0..state.len() {
            let index = ParticleIndex(raw_index);
            state.validate_particle(index)?;
        }
        Ok(state)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[must_use]
    pub fn index(&self, raw: usize) -> Option<ParticleIndex> {
        (raw < self.len()).then_some(ParticleIndex(raw))
    }

    #[must_use]
    pub fn particle(&self, index: ParticleIndex) -> Particle {
        let raw = index.0;
        Particle {
            id: self.ids[raw],
            position: self.positions[raw],
            velocity: self.velocities[raw],
            mass: self.masses[raw],
        }
    }

    fn validate_particle(&self, index: ParticleIndex) -> Result<(), StateError> {
        let particle = self.particle(index);
        if !particle.position.is_finite() {
            return Err(StateError::NonFinite {
                index,
                field: "position",
            });
        }
        if !particle.velocity.is_finite() {
            return Err(StateError::NonFinite {
                index,
                field: "velocity",
            });
        }
        if !particle.mass.is_finite() || particle.mass.value() < 0.0 {
            return Err(StateError::InvalidMass {
                index,
                value: particle.mass.value(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum StateError {
    MismatchedColumnLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        index: ParticleIndex,
        field: &'static str,
    },
    InvalidMass {
        index: ParticleIndex,
        value: f64,
    },
}

impl fmt::Display for StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedColumnLength {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "particle column `{field}` has length {actual}, expected {expected}"
            ),
            Self::NonFinite { index, field } => write!(
                formatter,
                "particle {} has a non-finite {field}",
                index.get()
            ),
            Self::InvalidMass { index, value } => {
                write!(
                    formatter,
                    "particle {} has invalid mass {value}",
                    index.get()
                )
            }
        }
    }
}

impl Error for StateError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: u64) -> ParticleId {
        ParticleId::new(NonZeroU64::new(value).unwrap())
    }

    #[test]
    fn columns_are_checked_before_state_is_created() {
        let error = ParticleState::from_columns(
            vec![id(1)],
            vec![],
            vec![Vector3::ZERO],
            vec![CodeMass::new(1.0)],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            StateError::MismatchedColumnLength {
                field: "positions",
                ..
            }
        ));
    }

    #[test]
    fn only_state_owned_indices_can_be_constructed() {
        let state = ParticleState::from_columns(
            vec![id(7)],
            vec![Vector3::new(1.0, 2.0, 3.0)],
            vec![Vector3::ZERO],
            vec![CodeMass::new(4.0)],
        )
        .unwrap();
        assert_eq!(state.particle(state.index(0).unwrap()).id.get(), 7);
        assert!(state.index(1).is_none());
    }

    #[test]
    fn rejects_negative_and_nonfinite_physical_fields() {
        for mass in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                ParticleState::from_columns(
                    vec![id(1)],
                    vec![Vector3::ZERO],
                    vec![Vector3::ZERO],
                    vec![CodeMass::new(mass)],
                )
                .is_err()
            );
        }
    }
}
