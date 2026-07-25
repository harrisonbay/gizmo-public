#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;
use std::path::Path;

const PARTICLE_TYPES: usize = 6;
const VECTOR_COMPONENTS: usize = 3;

#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotHeader {
    pub time: f64,
    pub box_size: f64,
    pub num_part_total: [u64; PARTICLE_TYPES],
    pub double_precision: bool,
}

impl SnapshotHeader {
    /// Validate header values needed by the sound-wave oracle.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite or physically invalid scalar values.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if !self.time.is_finite() || self.time < 0.0 {
            return Err(ValidationError::InvalidHeaderScalar {
                field: "Time",
                value: self.time,
            });
        }
        if !self.box_size.is_finite() || self.box_size <= 0.0 {
            return Err(ValidationError::InvalidHeaderScalar {
                field: "BoxSize",
                value: self.box_size,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GasParticles {
    pub coordinates: Vec<[f64; VECTOR_COMPONENTS]>,
    pub velocities: Vec<[f64; VECTOR_COMPONENTS]>,
    pub ids: Vec<u64>,
    pub masses: Vec<f64>,
    pub internal_energy: Vec<f64>,
    pub density: Option<Vec<f64>>,
    pub smoothing_length: Option<Vec<f64>>,
}

impl GasParticles {
    /// Validate complete particle columns and sort them by `ParticleIDs`.
    ///
    /// Sorting makes semantic comparisons independent of HDF5 row order. Every
    /// field, including optional fields, is reordered with its particle ID.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched columns, duplicate IDs, non-finite
    /// vectors, or non-positive thermodynamic and size fields. Particle ID zero
    /// is valid because the pinned public fixture uses it.
    pub fn validate_and_sort(&mut self) -> Result<(), ValidationError> {
        let expected = self.ids.len();
        for (field, actual) in [
            ("Coordinates", self.coordinates.len()),
            ("Velocities", self.velocities.len()),
            ("Masses", self.masses.len()),
            ("InternalEnergy", self.internal_energy.len()),
        ] {
            validate_column_length(field, expected, actual)?;
        }
        if let Some(values) = &self.density {
            validate_column_length("Density", expected, values.len())?;
        }
        if let Some(values) = &self.smoothing_length {
            validate_column_length("SmoothingLength", expected, values.len())?;
        }

        for index in 0..expected {
            validate_vector("Coordinates", index, self.coordinates[index])?;
            validate_vector("Velocities", index, self.velocities[index])?;
            validate_positive("Masses", index, self.masses[index])?;
            validate_positive("InternalEnergy", index, self.internal_energy[index])?;
            if let Some(values) = &self.density {
                validate_positive("Density", index, values[index])?;
            }
            if let Some(values) = &self.smoothing_length {
                validate_positive("SmoothingLength", index, values[index])?;
            }
        }

        let mut order: Vec<usize> = (0..expected).collect();
        order.sort_unstable_by_key(|&index| self.ids[index]);
        for pair in order.windows(2) {
            let left = self.ids[pair[0]];
            let right = self.ids[pair[1]];
            if left == right {
                return Err(ValidationError::DuplicateParticleId(left));
            }
        }

        self.coordinates = reorder(&self.coordinates, &order);
        self.velocities = reorder(&self.velocities, &order);
        self.ids = reorder(&self.ids, &order);
        self.masses = reorder(&self.masses, &order);
        self.internal_energy = reorder(&self.internal_energy, &order);
        self.density = self.density.as_ref().map(|values| reorder(values, &order));
        self.smoothing_length = self
            .smoothing_length
            .as_ref()
            .map(|values| reorder(values, &order));
        Ok(())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoundWaveSnapshot {
    pub header: SnapshotHeader,
    pub gas: GasParticles,
}

impl SoundWaveSnapshot {
    /// Validate the snapshot as the gas-only public sound-wave fixture.
    ///
    /// # Errors
    ///
    /// Returns an error if header values or gas columns are invalid, the gas
    /// count disagrees with `NumPart_Total`, or another particle type is present.
    pub fn validate_and_sort(&mut self) -> Result<(), ValidationError> {
        if self.gas.is_empty() {
            return Err(ValidationError::EmptyGasState);
        }
        self.header.validate()?;
        let header_gas_count = usize::try_from(self.header.num_part_total[0])
            .map_err(|_| ValidationError::ParticleCountOverflow(self.header.num_part_total[0]))?;
        if header_gas_count != self.gas.len() {
            return Err(ValidationError::ParticleCountMismatch {
                header: header_gas_count,
                dataset: self.gas.len(),
            });
        }
        if let Some((particle_type, count)) = self
            .header
            .num_part_total
            .iter()
            .copied()
            .enumerate()
            .skip(1)
            .find(|(_, count)| *count != 0)
        {
            return Err(ValidationError::UnexpectedParticleType {
                particle_type,
                count,
            });
        }
        self.gas.validate_and_sort()
    }
}

/// Read and validate the public gas-only sound-wave initial condition.
///
/// Numeric HDF5 values are converted through HDF5's checked conversion layer
/// to the canonical in-memory `f64` and `u64` representation.
///
/// # Errors
///
/// Returns an HDF5 error for missing or unreadable objects, or a validation
/// error for malformed shapes and physically invalid data.
pub fn read_soundwave(path: impl AsRef<Path>) -> Result<SoundWaveSnapshot, InputError> {
    let file = hdf5::File::open(path)?;
    let header_group = file.group("Header")?;
    let num_part_total = read_particle_counts(&header_group, "NumPart_Total")?;
    let double_precision_raw: i32 = header_group.attr("Flag_DoublePrecision")?.read_scalar()?;
    let double_precision = match double_precision_raw {
        0 => false,
        1 => true,
        value => {
            return Err(ValidationError::InvalidPrecisionFlag(value).into());
        }
    };
    let header = SnapshotHeader {
        time: header_group.attr("Time")?.read_scalar()?,
        box_size: header_group.attr("BoxSize")?.read_scalar()?,
        num_part_total,
        double_precision,
    };

    let gas_group = file.group("PartType0")?;
    let coordinates = read_vectors(&gas_group, "Coordinates")?;
    let velocities = read_vectors(&gas_group, "Velocities")?;
    let ids = read_scalar_dataset::<u64>(&gas_group, "ParticleIDs")?;
    let masses = read_scalar_dataset::<f64>(&gas_group, "Masses")?;
    let internal_energy = read_scalar_dataset::<f64>(&gas_group, "InternalEnergy")?;
    let density = read_optional_scalars(&gas_group, "Density")?;
    let smoothing_length = read_optional_scalars(&gas_group, "SmoothingLength")?;

    let mut snapshot = SoundWaveSnapshot {
        header,
        gas: GasParticles {
            coordinates,
            velocities,
            ids,
            masses,
            internal_energy,
            density,
            smoothing_length,
        },
    };
    snapshot.validate_and_sort()?;
    Ok(snapshot)
}

fn read_particle_counts(
    group: &hdf5::Group,
    name: &'static str,
) -> Result<[u64; PARTICLE_TYPES], InputError> {
    let values = group.attr(name)?.read_raw::<u64>()?;
    values
        .try_into()
        .map_err(|values: Vec<u64>| ValidationError::InvalidShape {
            field: name,
            expected: vec![PARTICLE_TYPES],
            actual: vec![values.len()],
        })
        .map_err(Into::into)
}

fn read_vectors(
    group: &hdf5::Group,
    name: &'static str,
) -> Result<Vec<[f64; VECTOR_COMPONENTS]>, InputError> {
    let dataset = group.dataset(name)?;
    let shape = dataset.shape();
    let values = dataset.read_raw::<f64>()?;
    reshape_vectors(&values, &shape, name).map_err(Into::into)
}

fn read_scalar_dataset<T>(group: &hdf5::Group, name: &'static str) -> Result<Vec<T>, InputError>
where
    T: hdf5::H5Type,
{
    let dataset = group.dataset(name)?;
    let shape = dataset.shape();
    if shape.len() != 1 {
        return Err(ValidationError::InvalidShape {
            field: name,
            expected: vec![shape.iter().product()],
            actual: shape,
        }
        .into());
    }
    Ok(dataset.read_raw::<T>()?)
}

fn read_optional_scalars(
    group: &hdf5::Group,
    name: &'static str,
) -> Result<Option<Vec<f64>>, InputError> {
    if group.link_exists(name) {
        read_scalar_dataset(group, name).map(Some)
    } else {
        Ok(None)
    }
}

fn reshape_vectors(
    values: &[f64],
    shape: &[usize],
    field: &'static str,
) -> Result<Vec<[f64; VECTOR_COMPONENTS]>, ValidationError> {
    if shape.len() != 2 || shape[1] != VECTOR_COMPONENTS {
        return Err(ValidationError::InvalidShape {
            field,
            expected: vec![shape.first().copied().unwrap_or(0), VECTOR_COMPONENTS],
            actual: shape.to_vec(),
        });
    }
    if values.len() != shape[0].saturating_mul(VECTOR_COMPONENTS) {
        return Err(ValidationError::InvalidShape {
            field,
            expected: vec![shape[0], VECTOR_COMPONENTS],
            actual: vec![values.len()],
        });
    }
    Ok(values
        .chunks_exact(VECTOR_COMPONENTS)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect())
}

fn validate_column_length(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), ValidationError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ValidationError::ColumnLength {
            field,
            expected,
            actual,
        })
    }
}

fn validate_vector(
    field: &'static str,
    index: usize,
    value: [f64; VECTOR_COMPONENTS],
) -> Result<(), ValidationError> {
    if value.into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(ValidationError::NonFiniteVector {
            field,
            index,
            value,
        })
    }
}

fn validate_positive(field: &'static str, index: usize, value: f64) -> Result<(), ValidationError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ValidationError::NonPositiveValue {
            field,
            index,
            value,
        })
    }
}

fn reorder<T: Copy>(values: &[T], order: &[usize]) -> Vec<T> {
    order.iter().map(|&index| values[index]).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum ValidationError {
    EmptyGasState,
    InvalidHeaderScalar {
        field: &'static str,
        value: f64,
    },
    InvalidPrecisionFlag(i32),
    InvalidShape {
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    ColumnLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    ParticleCountOverflow(u64),
    ParticleCountMismatch {
        header: usize,
        dataset: usize,
    },
    UnexpectedParticleType {
        particle_type: usize,
        count: u64,
    },
    DuplicateParticleId(u64),
    NonFiniteVector {
        field: &'static str,
        index: usize,
        value: [f64; VECTOR_COMPONENTS],
    },
    NonPositiveValue {
        field: &'static str,
        index: usize,
        value: f64,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGasState => formatter.write_str("sound-wave gas state is empty"),
            Self::InvalidHeaderScalar { field, value } => {
                write!(formatter, "header `{field}` has invalid value {value}")
            }
            Self::InvalidPrecisionFlag(value) => write!(
                formatter,
                "header `Flag_DoublePrecision` is {value}, expected 0 or 1"
            ),
            Self::InvalidShape {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "HDF5 field `{field}` has shape {actual:?}, expected {expected:?}"
            ),
            Self::ColumnLength {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "particle column `{field}` has length {actual}, expected {expected}"
            ),
            Self::ParticleCountOverflow(count) => {
                write!(
                    formatter,
                    "gas particle count {count} does not fit in usize"
                )
            }
            Self::ParticleCountMismatch { header, dataset } => write!(
                formatter,
                "header declares {header} gas particles, datasets contain {dataset}"
            ),
            Self::UnexpectedParticleType {
                particle_type,
                count,
            } => write!(
                formatter,
                "sound-wave fixture unexpectedly contains {count} particles of type {particle_type}"
            ),
            Self::DuplicateParticleId(id) => write!(formatter, "particle ID {id} is duplicated"),
            Self::NonFiniteVector {
                field,
                index,
                value,
            } => write!(
                formatter,
                "particle row {index} has non-finite `{field}` value {value:?}"
            ),
            Self::NonPositiveValue {
                field,
                index,
                value,
            } => write!(
                formatter,
                "particle row {index} has non-positive or non-finite `{field}` value {value}"
            ),
        }
    }
}

impl Error for ValidationError {}

#[derive(Debug)]
pub enum InputError {
    Hdf5(hdf5::Error),
    Validation(ValidationError),
}

impl fmt::Display for InputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hdf5(error) => write!(formatter, "HDF5 input error: {error}"),
            Self::Validation(error) => error.fmt(formatter),
        }
    }
}

impl Error for InputError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Hdf5(error) => Some(error),
            Self::Validation(error) => Some(error),
        }
    }
}

impl From<hdf5::Error> for InputError {
    fn from(error: hdf5::Error) -> Self {
        Self::Hdf5(error)
    }
}

impl From<ValidationError> for InputError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float_slice_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    fn valid_snapshot() -> SoundWaveSnapshot {
        SoundWaveSnapshot {
            header: SnapshotHeader {
                time: 0.0,
                box_size: 1.0,
                num_part_total: [3, 0, 0, 0, 0, 0],
                double_precision: true,
            },
            gas: GasParticles {
                coordinates: vec![[0.3, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
                velocities: vec![[30.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                ids: vec![3, 1, 2],
                masses: vec![3.0, 1.0, 2.0],
                internal_energy: vec![300.0, 100.0, 200.0],
                density: Some(vec![30.0, 10.0, 20.0]),
                smoothing_length: Some(vec![0.03, 0.01, 0.02]),
            },
        }
    }

    #[test]
    fn validation_sorts_every_column_by_particle_id() {
        let mut snapshot = valid_snapshot();
        snapshot.validate_and_sort().unwrap();
        assert_eq!(snapshot.gas.ids, [1, 2, 3]);
        assert_float_slice_eq(&snapshot.gas.coordinates[0], &[0.1, 0.0, 0.0]);
        assert_float_slice_eq(&snapshot.gas.velocities[1], &[20.0, 0.0, 0.0]);
        assert_float_slice_eq(&snapshot.gas.masses, &[1.0, 2.0, 3.0]);
        assert_float_slice_eq(&snapshot.gas.internal_energy, &[100.0, 200.0, 300.0]);
        assert_float_slice_eq(
            snapshot.gas.density.as_deref().unwrap(),
            &[10.0, 20.0, 30.0],
        );
        assert_float_slice_eq(
            snapshot.gas.smoothing_length.as_deref().unwrap(),
            &[0.01, 0.02, 0.03],
        );
    }

    #[test]
    fn validation_accepts_zero_but_rejects_duplicate_ids() {
        let mut zero = valid_snapshot();
        zero.gas.ids[1] = 0;
        zero.validate_and_sort().unwrap();
        assert_eq!(zero.gas.ids, [0, 2, 3]);

        let mut duplicate = valid_snapshot();
        duplicate.gas.ids[2] = 3;
        assert_eq!(
            duplicate.validate_and_sort(),
            Err(ValidationError::DuplicateParticleId(3))
        );
    }

    #[test]
    fn validation_rejects_bad_counts_and_non_gas_particles() {
        let mut wrong_count = valid_snapshot();
        wrong_count.header.num_part_total[0] = 4;
        assert!(matches!(
            wrong_count.validate_and_sort(),
            Err(ValidationError::ParticleCountMismatch {
                header: 4,
                dataset: 3
            })
        ));

        let mut stars = valid_snapshot();
        stars.header.num_part_total[4] = 1;
        assert!(matches!(
            stars.validate_and_sort(),
            Err(ValidationError::UnexpectedParticleType {
                particle_type: 4,
                count: 1
            })
        ));
    }

    #[test]
    fn validation_rejects_empty_gas_state() {
        let mut empty = valid_snapshot();
        empty.header.num_part_total[0] = 0;
        empty.gas.coordinates.clear();
        empty.gas.velocities.clear();
        empty.gas.ids.clear();
        empty.gas.masses.clear();
        empty.gas.internal_energy.clear();
        empty.gas.density = Some(Vec::new());
        empty.gas.smoothing_length = Some(Vec::new());
        assert_eq!(
            empty.validate_and_sort(),
            Err(ValidationError::EmptyGasState)
        );
    }

    #[test]
    fn validation_rejects_invalid_physical_fields() {
        let mut non_finite = valid_snapshot();
        non_finite.gas.coordinates[0][0] = f64::NAN;
        assert!(matches!(
            non_finite.validate_and_sort(),
            Err(ValidationError::NonFiniteVector {
                field: "Coordinates",
                ..
            })
        ));

        for value in [0.0, -1.0, f64::INFINITY] {
            let mut invalid = valid_snapshot();
            invalid.gas.internal_energy[0] = value;
            assert!(matches!(
                invalid.validate_and_sort(),
                Err(ValidationError::NonPositiveValue {
                    field: "InternalEnergy",
                    ..
                })
            ));
        }
    }

    #[test]
    fn vector_shape_must_be_n_by_three() {
        assert!(reshape_vectors(&[0.0; 6], &[2, 3], "Coordinates").is_ok());
        assert!(matches!(
            reshape_vectors(&[0.0; 4], &[2, 2], "Coordinates"),
            Err(ValidationError::InvalidShape { .. })
        ));
        assert!(matches!(
            reshape_vectors(&[0.0; 6], &[6], "Coordinates"),
            Err(ValidationError::InvalidShape { .. })
        ));
    }

    #[test]
    #[ignore = "requires GIZMO_SOUNDWAVE_IC; run via validation oracle script"]
    fn reads_external_soundwave_fixture_when_configured() {
        let path = std::env::var_os("GIZMO_SOUNDWAVE_IC")
            .expect("GIZMO_SOUNDWAVE_IC must identify the pinned fixture");
        let snapshot = read_soundwave(path).unwrap();
        assert!(!snapshot.gas.is_empty());
        assert!(snapshot.gas.ids.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            snapshot.header.num_part_total[0],
            u64::try_from(snapshot.gas.len()).unwrap()
        );
    }
}
