#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;
use std::path::Path;

const PARTICLE_TYPES: usize = 6;
const VECTOR_COMPONENTS: usize = 3;
const GIZMO_RUST_PORT_VERSION: i32 = 1;
const NON_UPSTREAM_GIZMO_VERSION: i32 = -1;
const LEGACY_KERNEL_FUNCTION_ID: i32 = 3;
const LEGACY_GRAVITATIONAL_CONSTANT: f64 = 6.672e-8;

#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotHeader {
    pub time: f64,
    pub box_size: f64,
    pub num_part_total: [u64; PARTICLE_TYPES],
    pub double_precision: bool,
    pub effective_kernel_neighbors: Option<f64>,
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
        if let Some(value) = self.effective_kernel_neighbors
            && (!value.is_finite() || value <= 0.0)
        {
            return Err(ValidationError::InvalidHeaderScalar {
                field: "Effective_Kernel_NeighborNumber",
                value,
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
pub struct GrainParticles {
    pub coordinates: Vec<[f64; VECTOR_COMPONENTS]>,
    pub velocities: Vec<[f64; VECTOR_COMPONENTS]>,
    pub ids: Vec<u64>,
    pub masses: Vec<f64>,
    pub grain_size: Vec<f64>,
    pub smoothing_length: Option<Vec<f64>>,
}

impl GrainParticles {
    /// Validate complete grain columns and sort them by `ParticleIDs`.
    ///
    /// # Errors
    ///
    /// Returns an error for mismatched columns, duplicate IDs, non-finite
    /// vectors, or non-positive masses and grain sizes.
    pub fn validate_and_sort(&mut self) -> Result<(), ValidationError> {
        let expected = self.ids.len();
        for (field, actual) in [
            ("Coordinates", self.coordinates.len()),
            ("Velocities", self.velocities.len()),
            ("Masses", self.masses.len()),
            ("GrainSize", self.grain_size.len()),
        ] {
            validate_column_length(field, expected, actual)?;
        }
        if let Some(values) = &self.smoothing_length {
            validate_column_length("SmoothingLength", expected, values.len())?;
        }
        for index in 0..expected {
            validate_vector("Coordinates", index, self.coordinates[index])?;
            validate_vector("Velocities", index, self.velocities[index])?;
            validate_positive("Masses", index, self.masses[index])?;
            validate_positive("GrainSize", index, self.grain_size[index])?;
            if let Some(values) = &self.smoothing_length {
                validate_positive("SmoothingLength", index, values[index])?;
            }
        }

        let order = particle_id_order(&self.ids)?;
        self.coordinates = reorder(&self.coordinates, &order);
        self.velocities = reorder(&self.velocities, &order);
        self.ids = reorder(&self.ids, &order);
        self.masses = reorder(&self.masses, &order);
        self.grain_size = reorder(&self.grain_size, &order);
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

#[derive(Clone, Debug, PartialEq)]
pub struct DustyWaveSnapshot {
    pub header: SnapshotHeader,
    pub gas: GasParticles,
    pub grains: GrainParticles,
}

/// Borrowed, complete gas state to serialize as a sound-wave snapshot.
///
/// Unlike [`GasParticles`], density and smoothing length are required here:
/// evolved snapshots must be self-contained rather than relying on a reader to
/// reconstruct hydrodynamic state. The vector fields retain all three
/// components, so a one-dimensional evolution can update x while preserving y
/// and z from its input snapshot.
#[derive(Clone, Copy, Debug)]
pub struct SoundWaveWriteView<'a> {
    pub header: &'a SnapshotHeader,
    pub coordinates: &'a [[f64; VECTOR_COMPONENTS]],
    pub velocities: &'a [[f64; VECTOR_COMPONENTS]],
    pub ids: &'a [u64],
    pub masses: &'a [f64],
    pub internal_energy: &'a [f64],
    pub density: &'a [f64],
    pub smoothing_length: &'a [f64],
}

/// Borrowed, complete gas columns used by a multi-species snapshot writer.
#[derive(Clone, Copy, Debug)]
pub struct GasWriteView<'a> {
    pub coordinates: &'a [[f64; VECTOR_COMPONENTS]],
    pub velocities: &'a [[f64; VECTOR_COMPONENTS]],
    pub ids: &'a [u64],
    pub masses: &'a [f64],
    pub internal_energy: &'a [f64],
    pub density: &'a [f64],
    pub smoothing_length: &'a [f64],
}

/// Borrowed, complete type-3 grain columns.
#[derive(Clone, Copy, Debug)]
pub struct GrainWriteView<'a> {
    pub coordinates: &'a [[f64; VECTOR_COMPONENTS]],
    pub velocities: &'a [[f64; VECTOR_COMPONENTS]],
    pub ids: &'a [u64],
    pub masses: &'a [f64],
    pub grain_size: &'a [f64],
    pub smoothing_length: &'a [f64],
}

/// Borrowed gas-and-grain state to serialize as a dusty-wave snapshot.
#[derive(Clone, Copy, Debug)]
pub struct DustyWaveWriteView<'a> {
    pub header: &'a SnapshotHeader,
    pub gas: GasWriteView<'a>,
    pub grains: GrainWriteView<'a>,
}

impl<'a> TryFrom<&'a SoundWaveSnapshot> for SoundWaveWriteView<'a> {
    type Error = ValidationError;

    fn try_from(snapshot: &'a SoundWaveSnapshot) -> Result<Self, Self::Error> {
        Ok(Self {
            header: &snapshot.header,
            coordinates: &snapshot.gas.coordinates,
            velocities: &snapshot.gas.velocities,
            ids: &snapshot.gas.ids,
            masses: &snapshot.gas.masses,
            internal_energy: &snapshot.gas.internal_energy,
            density: snapshot
                .gas
                .density
                .as_deref()
                .ok_or(ValidationError::MissingRequiredField("Density"))?,
            smoothing_length: snapshot
                .gas
                .smoothing_length
                .as_deref()
                .ok_or(ValidationError::MissingRequiredField("SmoothingLength"))?,
        })
    }
}

impl<'a> TryFrom<&'a DustyWaveSnapshot> for DustyWaveWriteView<'a> {
    type Error = ValidationError;

    fn try_from(snapshot: &'a DustyWaveSnapshot) -> Result<Self, Self::Error> {
        Ok(Self {
            header: &snapshot.header,
            gas: GasWriteView {
                coordinates: &snapshot.gas.coordinates,
                velocities: &snapshot.gas.velocities,
                ids: &snapshot.gas.ids,
                masses: &snapshot.gas.masses,
                internal_energy: &snapshot.gas.internal_energy,
                density: snapshot
                    .gas
                    .density
                    .as_deref()
                    .ok_or(ValidationError::MissingRequiredField("Density"))?,
                smoothing_length: snapshot
                    .gas
                    .smoothing_length
                    .as_deref()
                    .ok_or(ValidationError::MissingRequiredField("SmoothingLength"))?,
            },
            grains: GrainWriteView {
                coordinates: &snapshot.grains.coordinates,
                velocities: &snapshot.grains.velocities,
                ids: &snapshot.grains.ids,
                masses: &snapshot.grains.masses,
                grain_size: &snapshot.grains.grain_size,
                smoothing_length: snapshot.grains.smoothing_length.as_deref().ok_or(
                    ValidationError::MissingRequiredField("PartType3/SmoothingLength"),
                )?,
            },
        })
    }
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

impl DustyWaveSnapshot {
    /// Validate a gas-and-type-3-grain snapshot and sort each particle type.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid header counts, unexpected particle types,
    /// malformed particle columns, or an ID duplicated within or across types.
    pub fn validate_and_sort(&mut self) -> Result<(), ValidationError> {
        if self.gas.is_empty() {
            return Err(ValidationError::EmptyGasState);
        }
        if self.grains.is_empty() {
            return Err(ValidationError::EmptyGrainState);
        }
        self.header.validate()?;
        validate_particle_type_count(&self.header, 0, self.gas.len())?;
        validate_particle_type_count(&self.header, 3, self.grains.len())?;
        if let Some((particle_type, count)) = self
            .header
            .num_part_total
            .iter()
            .copied()
            .enumerate()
            .find(|(particle_type, count)| {
                *count != 0 && *particle_type != 0 && *particle_type != 3
            })
        {
            return Err(ValidationError::UnexpectedParticleType {
                particle_type,
                count,
            });
        }
        self.gas.validate_and_sort()?;
        self.grains.validate_and_sort()?;
        validate_disjoint_particle_ids(&self.gas.ids, &self.grains.ids)
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
    let mut snapshot = SoundWaveSnapshot {
        header: read_snapshot_header(&file)?,
        gas: read_gas_particles(&file)?,
    };
    snapshot.validate_and_sort()?;
    Ok(snapshot)
}

/// Read and validate a gas-and-type-3-grain dusty-wave snapshot.
///
/// Numeric HDF5 values are converted to the canonical in-memory `f64` and
/// `u64` representation. Dataset rows are sorted independently within each
/// particle type, and IDs must remain globally unique.
///
/// # Errors
///
/// Returns an HDF5 error for missing or unreadable objects, or a validation
/// error for malformed shapes, invalid counts, or invalid physical data.
pub fn read_dustywave(path: impl AsRef<Path>) -> Result<DustyWaveSnapshot, InputError> {
    let file = hdf5::File::open(path)?;
    let grain_group = file.group("PartType3")?;
    let mut snapshot = DustyWaveSnapshot {
        header: read_snapshot_header(&file)?,
        gas: read_gas_particles(&file)?,
        grains: GrainParticles {
            coordinates: read_vectors(&grain_group, "Coordinates")?,
            velocities: read_vectors(&grain_group, "Velocities")?,
            ids: read_scalar_dataset(&grain_group, "ParticleIDs")?,
            masses: read_scalar_dataset(&grain_group, "Masses")?,
            grain_size: read_scalar_dataset(&grain_group, "GrainSize")?,
            smoothing_length: read_optional_scalars(&grain_group, "SmoothingLength")?,
        },
    };
    snapshot.validate_and_sort()?;
    Ok(snapshot)
}

fn read_snapshot_header(file: &hdf5::File) -> Result<SnapshotHeader, InputError> {
    let header = file.group("Header")?;
    let double_precision_raw: i32 = header.attr("Flag_DoublePrecision")?.read_scalar()?;
    let double_precision = match double_precision_raw {
        0 => false,
        1 => true,
        value => return Err(ValidationError::InvalidPrecisionFlag(value).into()),
    };
    Ok(SnapshotHeader {
        time: header.attr("Time")?.read_scalar()?,
        box_size: header.attr("BoxSize")?.read_scalar()?,
        num_part_total: read_particle_counts(&header, "NumPart_Total")?,
        double_precision,
        effective_kernel_neighbors: read_optional_scalar_attribute(
            &header,
            "Effective_Kernel_NeighborNumber",
        )?,
    })
}

fn read_gas_particles(file: &hdf5::File) -> Result<GasParticles, InputError> {
    let gas = file.group("PartType0")?;
    Ok(GasParticles {
        coordinates: read_vectors(&gas, "Coordinates")?,
        velocities: read_vectors(&gas, "Velocities")?,
        ids: read_scalar_dataset(&gas, "ParticleIDs")?,
        masses: read_scalar_dataset(&gas, "Masses")?,
        internal_energy: read_scalar_dataset(&gas, "InternalEnergy")?,
        density: read_optional_scalars(&gas, "Density")?,
        smoothing_length: read_optional_scalars(&gas, "SmoothingLength")?,
    })
}

/// Write a complete, validated gas-only sound-wave snapshot.
///
/// Dataset rows are emitted in the supplied order. No implicit sorting or
/// scalar-to-vector expansion occurs, which makes particle identity and the
/// transverse coordinate and velocity components explicit at the call site.
///
/// # Errors
///
/// Returns a validation error before creating the file if the view is
/// inconsistent, or an HDF5 error if the destination cannot be written.
pub fn write_soundwave(
    path: impl AsRef<Path>,
    snapshot: SoundWaveWriteView<'_>,
) -> Result<(), OutputError> {
    validate_write_view(snapshot)?;

    let gas_count = i32::try_from(snapshot.ids.len())
        .map_err(|_| ValidationError::LegacyFileParticleCountOverflow(snapshot.ids.len()))?;
    let mut num_part_this_file = [0_i32; PARTICLE_TYPES];
    num_part_this_file[0] = gas_count;
    let num_part_total_low = snapshot
        .header
        .num_part_total
        .map(legacy_particle_count_low_word);
    let num_part_total_high = snapshot
        .header
        .num_part_total
        .map(legacy_particle_count_high_word);
    let mass_table = [0.0_f64; PARTICLE_TYPES];
    let fixed_force_softening = [0.0_f64; PARTICLE_TYPES];
    let minimum_mass_for_merge = 0.49
        * snapshot
            .masses
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
    let maximum_mass_for_split = 3.01
        * snapshot
            .masses
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
    let legacy_ids: Vec<u32> = snapshot
        .ids
        .iter()
        .copied()
        .map(|id| u32::try_from(id).map_err(|_| ValidationError::LegacyParticleIdOverflow(id)))
        .collect::<Result<_, _>>()?;
    let effective_kernel_neighbors =
        snapshot
            .header
            .effective_kernel_neighbors
            .ok_or(ValidationError::MissingRequiredField(
                "Effective_Kernel_NeighborNumber",
            ))?;

    let file = hdf5::File::create(path)?;
    let header = file.create_group("Header")?;
    write_scalar_attribute(&header, "Time", &snapshot.header.time)?;
    write_scalar_attribute(&header, "BoxSize", &snapshot.header.box_size)?;
    write_array_attribute(&header, "NumPart_ThisFile", &num_part_this_file)?;
    write_array_attribute(&header, "NumPart_Total", &num_part_total_low)?;
    write_array_attribute(&header, "NumPart_Total_HighWord", &num_part_total_high)?;
    write_array_attribute(&header, "MassTable", &mass_table)?;
    write_scalar_attribute(&header, "NumFilesPerSnapshot", &1_i32)?;
    let precision_flag = i32::from(snapshot.header.double_precision);
    write_scalar_attribute(&header, "Flag_DoublePrecision", &precision_flag)?;
    write_legacy_compatibility_attributes(
        &header,
        effective_kernel_neighbors,
        minimum_mass_for_merge,
        maximum_mass_for_split,
        &fixed_force_softening,
    )?;

    let gas = file.create_group("PartType0")?;
    write_vectors(&gas, "Coordinates", snapshot.coordinates)?;
    write_vectors(&gas, "Velocities", snapshot.velocities)?;
    write_scalars(&gas, "ParticleIDs", &legacy_ids)?;
    write_scalars(&gas, "Masses", snapshot.masses)?;
    write_scalars(&gas, "InternalEnergy", snapshot.internal_energy)?;
    write_scalars(&gas, "Density", snapshot.density)?;
    write_scalars(&gas, "SmoothingLength", snapshot.smoothing_length)?;
    Ok(())
}

/// Write a complete, validated gas-and-type-3-grain dusty-wave snapshot.
///
/// Both particle groups are emitted in the supplied order. IDs are stored in
/// the public snapshot format's `u32` representation, while header totals
/// retain their complete low/high-word encoding.
///
/// # Errors
///
/// Returns a validation error before creating the file if either particle
/// group is inconsistent, or an HDF5 error if the destination cannot be
/// written.
pub fn write_dustywave(
    path: impl AsRef<Path>,
    snapshot: DustyWaveWriteView<'_>,
) -> Result<(), OutputError> {
    validate_dustywave_write_view(snapshot)?;

    let gas_count = legacy_file_particle_count(snapshot.gas.ids.len())?;
    let grain_count = legacy_file_particle_count(snapshot.grains.ids.len())?;
    let mut num_part_this_file = [0_i32; PARTICLE_TYPES];
    num_part_this_file[0] = gas_count;
    num_part_this_file[3] = grain_count;
    let num_part_total_low = snapshot
        .header
        .num_part_total
        .map(legacy_particle_count_low_word);
    let num_part_total_high = snapshot
        .header
        .num_part_total
        .map(legacy_particle_count_high_word);
    let legacy_gas_ids = legacy_particle_ids(snapshot.gas.ids)?;
    let legacy_grain_ids = legacy_particle_ids(snapshot.grains.ids)?;
    let effective_kernel_neighbors =
        snapshot
            .header
            .effective_kernel_neighbors
            .ok_or(ValidationError::MissingRequiredField(
                "Effective_Kernel_NeighborNumber",
            ))?;
    let minimum_mass_for_merge = 0.49
        * snapshot
            .gas
            .masses
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
    let maximum_mass_for_split = 3.01
        * snapshot
            .gas
            .masses
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
    let mass_table = [0.0_f64; PARTICLE_TYPES];
    let fixed_force_softening = [0.0_f64; PARTICLE_TYPES];

    let file = hdf5::File::create(path)?;
    let header = file.create_group("Header")?;
    write_scalar_attribute(&header, "Time", &snapshot.header.time)?;
    write_scalar_attribute(&header, "BoxSize", &snapshot.header.box_size)?;
    write_array_attribute(&header, "NumPart_ThisFile", &num_part_this_file)?;
    write_array_attribute(&header, "NumPart_Total", &num_part_total_low)?;
    write_array_attribute(&header, "NumPart_Total_HighWord", &num_part_total_high)?;
    write_array_attribute(&header, "MassTable", &mass_table)?;
    write_scalar_attribute(&header, "NumFilesPerSnapshot", &1_i32)?;
    write_scalar_attribute(
        &header,
        "Flag_DoublePrecision",
        &i32::from(snapshot.header.double_precision),
    )?;
    write_legacy_compatibility_attributes(
        &header,
        effective_kernel_neighbors,
        minimum_mass_for_merge,
        maximum_mass_for_split,
        &fixed_force_softening,
    )?;

    let gas = file.create_group("PartType0")?;
    write_vectors(&gas, "Coordinates", snapshot.gas.coordinates)?;
    write_vectors(&gas, "Velocities", snapshot.gas.velocities)?;
    write_scalars(&gas, "ParticleIDs", &legacy_gas_ids)?;
    write_scalars(&gas, "Masses", snapshot.gas.masses)?;
    write_scalars(&gas, "InternalEnergy", snapshot.gas.internal_energy)?;
    write_scalars(&gas, "Density", snapshot.gas.density)?;
    write_scalars(&gas, "SmoothingLength", snapshot.gas.smoothing_length)?;

    let grains = file.create_group("PartType3")?;
    write_vectors(&grains, "Coordinates", snapshot.grains.coordinates)?;
    write_vectors(&grains, "Velocities", snapshot.grains.velocities)?;
    write_scalars(&grains, "ParticleIDs", &legacy_grain_ids)?;
    write_scalars(&grains, "Masses", snapshot.grains.masses)?;
    write_scalars(&grains, "GrainSize", snapshot.grains.grain_size)?;
    write_scalars(&grains, "SmoothingLength", snapshot.grains.smoothing_length)?;
    Ok(())
}

fn write_legacy_compatibility_attributes(
    header: &hdf5::Group,
    effective_kernel_neighbors: f64,
    minimum_mass_for_merge: f64,
    maximum_mass_for_split: f64,
    fixed_force_softening: &[f64; PARTICLE_TYPES],
) -> Result<(), hdf5::Error> {
    write_scalar_attribute(header, "ComovingIntegrationOn", &0_i32)?;
    write_scalar_attribute(
        header,
        "Effective_Kernel_NeighborNumber",
        &effective_kernel_neighbors,
    )?;
    write_array_attribute(
        header,
        "Fixed_ForceSoftening_Keplerian_Kernel_Extent",
        fixed_force_softening,
    )?;
    for name in [
        "Flag_Cooling",
        "Flag_Feedback",
        "Flag_IC_Info",
        "Flag_Metals",
        "Flag_Sfr",
        "Flag_StellarAge",
    ] {
        write_scalar_attribute(header, name, &0_i32)?;
    }
    // Do not claim that a Rust-port snapshot was produced by upstream GIZMO
    // 2022. The legacy-typed sentinel keeps readers that require this attribute
    // working, while the explicit port schema version records true provenance.
    write_scalar_attribute(header, "GIZMO_version", &NON_UPSTREAM_GIZMO_VERSION)?;
    write_scalar_attribute(header, "GIZMO_RustPort_version", &GIZMO_RUST_PORT_VERSION)?;
    write_scalar_attribute(
        header,
        "Gravitational_Constant_In_Code_Inits",
        &LEGACY_GRAVITATIONAL_CONSTANT,
    )?;
    write_scalar_attribute(header, "HubbleParam", &1.0_f64)?;
    write_scalar_attribute(header, "Kernel_Function_ID", &LEGACY_KERNEL_FUNCTION_ID)?;
    write_scalar_attribute(
        header,
        "Maximum_Mass_For_Cell_Split",
        &maximum_mass_for_split,
    )?;
    write_scalar_attribute(
        header,
        "Minimum_Mass_For_Cell_Merge",
        &minimum_mass_for_merge,
    )?;
    write_scalar_attribute(header, "Redshift", &0.0_f64)?;
    for name in [
        "UnitLength_In_CGS",
        "UnitMass_In_CGS",
        "UnitVelocity_In_CGS",
    ] {
        write_scalar_attribute(header, name, &1.0_f64)?;
    }
    Ok(())
}

fn validate_write_view(snapshot: SoundWaveWriteView<'_>) -> Result<(), ValidationError> {
    snapshot.header.validate()?;
    let expected = snapshot.ids.len();
    if expected == 0 {
        return Err(ValidationError::EmptyGasState);
    }
    let header_gas_count = usize::try_from(snapshot.header.num_part_total[0])
        .map_err(|_| ValidationError::ParticleCountOverflow(snapshot.header.num_part_total[0]))?;
    if header_gas_count != expected {
        return Err(ValidationError::ParticleCountMismatch {
            header: header_gas_count,
            dataset: expected,
        });
    }
    if let Some((particle_type, count)) = snapshot
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

    for (field, actual) in [
        ("Coordinates", snapshot.coordinates.len()),
        ("Velocities", snapshot.velocities.len()),
        ("Masses", snapshot.masses.len()),
        ("InternalEnergy", snapshot.internal_energy.len()),
        ("Density", snapshot.density.len()),
        ("SmoothingLength", snapshot.smoothing_length.len()),
    ] {
        validate_column_length(field, expected, actual)?;
    }
    for index in 0..expected {
        validate_vector("Coordinates", index, snapshot.coordinates[index])?;
        validate_vector("Velocities", index, snapshot.velocities[index])?;
        validate_positive("Masses", index, snapshot.masses[index])?;
        validate_positive("InternalEnergy", index, snapshot.internal_energy[index])?;
        validate_positive("Density", index, snapshot.density[index])?;
        validate_positive("SmoothingLength", index, snapshot.smoothing_length[index])?;
    }
    let mut ids = snapshot.ids.to_vec();
    ids.sort_unstable();
    if let Some(id) = ids.windows(2).find_map(|pair| {
        if pair[0] == pair[1] {
            Some(pair[0])
        } else {
            None
        }
    }) {
        return Err(ValidationError::DuplicateParticleId(id));
    }
    Ok(())
}

fn validate_dustywave_write_view(snapshot: DustyWaveWriteView<'_>) -> Result<(), ValidationError> {
    snapshot.header.validate()?;
    if snapshot.gas.ids.is_empty() {
        return Err(ValidationError::EmptyGasState);
    }
    if snapshot.grains.ids.is_empty() {
        return Err(ValidationError::EmptyGrainState);
    }
    validate_particle_type_count(snapshot.header, 0, snapshot.gas.ids.len())?;
    validate_particle_type_count(snapshot.header, 3, snapshot.grains.ids.len())?;
    if let Some((particle_type, count)) = snapshot
        .header
        .num_part_total
        .iter()
        .copied()
        .enumerate()
        .find(|(particle_type, count)| *count != 0 && *particle_type != 0 && *particle_type != 3)
    {
        return Err(ValidationError::UnexpectedParticleType {
            particle_type,
            count,
        });
    }

    let gas_count = snapshot.gas.ids.len();
    for (field, actual) in [
        ("Coordinates", snapshot.gas.coordinates.len()),
        ("Velocities", snapshot.gas.velocities.len()),
        ("Masses", snapshot.gas.masses.len()),
        ("InternalEnergy", snapshot.gas.internal_energy.len()),
        ("Density", snapshot.gas.density.len()),
        ("SmoothingLength", snapshot.gas.smoothing_length.len()),
    ] {
        validate_column_length(field, gas_count, actual)?;
    }
    for index in 0..gas_count {
        validate_vector("Coordinates", index, snapshot.gas.coordinates[index])?;
        validate_vector("Velocities", index, snapshot.gas.velocities[index])?;
        validate_positive("Masses", index, snapshot.gas.masses[index])?;
        validate_positive("InternalEnergy", index, snapshot.gas.internal_energy[index])?;
        validate_positive("Density", index, snapshot.gas.density[index])?;
        validate_positive(
            "SmoothingLength",
            index,
            snapshot.gas.smoothing_length[index],
        )?;
    }

    let grain_count = snapshot.grains.ids.len();
    for (field, actual) in [
        ("Coordinates", snapshot.grains.coordinates.len()),
        ("Velocities", snapshot.grains.velocities.len()),
        ("Masses", snapshot.grains.masses.len()),
        ("GrainSize", snapshot.grains.grain_size.len()),
        ("SmoothingLength", snapshot.grains.smoothing_length.len()),
    ] {
        validate_column_length(field, grain_count, actual)?;
    }
    for index in 0..grain_count {
        validate_vector("Coordinates", index, snapshot.grains.coordinates[index])?;
        validate_vector("Velocities", index, snapshot.grains.velocities[index])?;
        validate_positive("Masses", index, snapshot.grains.masses[index])?;
        validate_positive("GrainSize", index, snapshot.grains.grain_size[index])?;
        validate_positive(
            "SmoothingLength",
            index,
            snapshot.grains.smoothing_length[index],
        )?;
    }

    particle_id_order(snapshot.gas.ids)?;
    particle_id_order(snapshot.grains.ids)?;
    validate_disjoint_particle_ids(snapshot.gas.ids, snapshot.grains.ids)
}

fn validate_particle_type_count(
    header: &SnapshotHeader,
    particle_type: usize,
    dataset: usize,
) -> Result<(), ValidationError> {
    let declared = header.num_part_total[particle_type];
    let header_count =
        usize::try_from(declared).map_err(|_| ValidationError::ParticleCountOverflow(declared))?;
    if header_count == dataset {
        Ok(())
    } else {
        Err(ValidationError::ParticleTypeCountMismatch {
            particle_type,
            header: header_count,
            dataset,
        })
    }
}

fn particle_id_order(ids: &[u64]) -> Result<Vec<usize>, ValidationError> {
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_unstable_by_key(|&index| ids[index]);
    if let Some(id) = order.windows(2).find_map(|pair| {
        let left = ids[pair[0]];
        (left == ids[pair[1]]).then_some(left)
    }) {
        return Err(ValidationError::DuplicateParticleId(id));
    }
    Ok(order)
}

fn validate_disjoint_particle_ids(left: &[u64], right: &[u64]) -> Result<(), ValidationError> {
    let mut ids = Vec::with_capacity(left.len() + right.len());
    ids.extend_from_slice(left);
    ids.extend_from_slice(right);
    ids.sort_unstable();
    if let Some(id) = ids
        .windows(2)
        .find_map(|pair| (pair[0] == pair[1]).then_some(pair[0]))
    {
        Err(ValidationError::DuplicateParticleId(id))
    } else {
        Ok(())
    }
}

fn legacy_file_particle_count(count: usize) -> Result<i32, ValidationError> {
    i32::try_from(count).map_err(|_| ValidationError::LegacyFileParticleCountOverflow(count))
}

fn legacy_particle_ids(ids: &[u64]) -> Result<Vec<u32>, ValidationError> {
    ids.iter()
        .copied()
        .map(|id| u32::try_from(id).map_err(|_| ValidationError::LegacyParticleIdOverflow(id)))
        .collect()
}

fn write_scalar_attribute<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    value: &T,
) -> Result<(), hdf5::Error> {
    group
        .new_attr::<T>()
        .shape(())
        .create(name)?
        .write_scalar(value)
}

#[allow(clippy::cast_possible_truncation)]
const fn legacy_particle_count_low_word(count: u64) -> u32 {
    count as u32
}

#[allow(clippy::cast_possible_truncation)]
const fn legacy_particle_count_high_word(count: u64) -> u32 {
    (count >> u32::BITS) as u32
}

fn write_array_attribute<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    values: &[T],
) -> Result<(), hdf5::Error> {
    group
        .new_attr::<T>()
        .shape([values.len()])
        .create(name)?
        .write_raw(values)
}

fn write_scalars<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    values: &[T],
) -> Result<(), hdf5::Error> {
    group
        .new_dataset::<T>()
        .shape([values.len()])
        .create(name)?
        .write_raw(values)
}

fn write_vectors(
    group: &hdf5::Group,
    name: &str,
    values: &[[f64; VECTOR_COMPONENTS]],
) -> Result<(), hdf5::Error> {
    let flattened: Vec<f64> = values.iter().flatten().copied().collect();
    group
        .new_dataset::<f64>()
        .shape([values.len(), VECTOR_COMPONENTS])
        .create(name)?
        .write_raw(&flattened)
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

fn read_optional_scalar_attribute<T>(
    group: &hdf5::Group,
    name: &str,
) -> Result<Option<T>, hdf5::Error>
where
    T: hdf5::H5Type,
{
    if group
        .attr_names()?
        .iter()
        .any(|candidate| candidate == name)
    {
        group.attr(name)?.read_scalar().map(Some)
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
    EmptyGrainState,
    MissingRequiredField(&'static str),
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
    LegacyFileParticleCountOverflow(usize),
    LegacyParticleIdOverflow(u64),
    ParticleCountMismatch {
        header: usize,
        dataset: usize,
    },
    ParticleTypeCountMismatch {
        particle_type: usize,
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
            Self::EmptyGrainState => formatter.write_str("dusty-wave grain state is empty"),
            Self::MissingRequiredField(field) => {
                write!(formatter, "snapshot output requires `{field}`")
            }
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
            Self::LegacyFileParticleCountOverflow(count) => write!(
                formatter,
                "particle count {count} does not fit in GIZMO's per-file header count"
            ),
            Self::LegacyParticleIdOverflow(id) => {
                write!(
                    formatter,
                    "particle ID {id} does not fit in GIZMO's public u32 ID field"
                )
            }
            Self::ParticleCountMismatch { header, dataset } => write!(
                formatter,
                "header declares {header} gas particles, datasets contain {dataset}"
            ),
            Self::ParticleTypeCountMismatch {
                particle_type,
                header,
                dataset,
            } => write!(
                formatter,
                "header declares {header} particles of type {particle_type}, \
                 datasets contain {dataset}"
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

#[derive(Debug)]
pub enum OutputError {
    Hdf5(hdf5::Error),
    Validation(ValidationError),
}

impl fmt::Display for OutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hdf5(error) => write!(formatter, "HDF5 output error: {error}"),
            Self::Validation(error) => error.fmt(formatter),
        }
    }
}

impl Error for OutputError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Hdf5(error) => Some(error),
            Self::Validation(error) => Some(error),
        }
    }
}

impl From<hdf5::Error> for OutputError {
    fn from(error: hdf5::Error) -> Self {
        Self::Hdf5(error)
    }
}

impl From<ValidationError> for OutputError {
    fn from(error: ValidationError) -> Self {
        Self::Validation(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
                effective_kernel_neighbors: Some(4.0),
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

    fn valid_dustywave_snapshot() -> DustyWaveSnapshot {
        let soundwave = valid_snapshot();
        DustyWaveSnapshot {
            header: SnapshotHeader {
                num_part_total: [3, 0, 0, 2, 0, 0],
                ..soundwave.header
            },
            gas: soundwave.gas,
            grains: GrainParticles {
                coordinates: vec![[0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
                velocities: vec![[50.0, 0.0, 0.0], [40.0, 0.0, 0.0]],
                ids: vec![5, 4],
                masses: vec![0.5, 0.4],
                grain_size: vec![0.05, 0.04],
                smoothing_length: Some(vec![0.005, 0.004]),
            },
        }
    }

    fn temporary_hdf5_path(test_name: &str) -> std::path::PathBuf {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "gizmo-io-{test_name}-{}-{sequence}.hdf5",
            std::process::id()
        ))
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
    fn dustywave_validation_sorts_both_types_and_requires_global_id_uniqueness() {
        let mut snapshot = valid_dustywave_snapshot();
        snapshot.validate_and_sort().unwrap();
        assert_eq!(snapshot.gas.ids, [1, 2, 3]);
        assert_eq!(snapshot.grains.ids, [4, 5]);
        assert_float_slice_eq(&snapshot.grains.masses, &[0.4, 0.5]);
        assert_float_slice_eq(&snapshot.grains.grain_size, &[0.04, 0.05]);
        assert_float_slice_eq(
            snapshot.grains.smoothing_length.as_deref().unwrap(),
            &[0.004, 0.005],
        );
        assert_float_slice_eq(&snapshot.grains.coordinates[0], &[0.4, 0.0, 0.0]);

        let mut duplicate = valid_dustywave_snapshot();
        duplicate.grains.ids[0] = 2;
        assert_eq!(
            duplicate.validate_and_sort(),
            Err(ValidationError::DuplicateParticleId(2))
        );
    }

    #[test]
    fn dustywave_validation_rejects_bad_grain_count_and_fields() {
        let mut wrong_count = valid_dustywave_snapshot();
        wrong_count.header.num_part_total[3] = 3;
        assert_eq!(
            wrong_count.validate_and_sort(),
            Err(ValidationError::ParticleTypeCountMismatch {
                particle_type: 3,
                header: 3,
                dataset: 2,
            })
        );

        let mut invalid_size = valid_dustywave_snapshot();
        invalid_size.grains.grain_size[0] = 0.0;
        assert!(matches!(
            invalid_size.validate_and_sort(),
            Err(ValidationError::NonPositiveValue {
                field: "GrainSize",
                ..
            })
        ));

        let mut empty = valid_dustywave_snapshot();
        empty.header.num_part_total[3] = 0;
        empty.grains.coordinates.clear();
        empty.grains.velocities.clear();
        empty.grains.ids.clear();
        empty.grains.masses.clear();
        empty.grains.grain_size.clear();
        empty.grains.smoothing_length = Some(Vec::new());
        assert_eq!(
            empty.validate_and_sort(),
            Err(ValidationError::EmptyGrainState)
        );
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
    #[allow(clippy::too_many_lines)]
    fn soundwave_writer_roundtrips_complete_state_exactly() {
        let mut expected = valid_snapshot();
        expected.validate_and_sort().unwrap();
        expected.header.time = 0.125;
        expected.header.box_size = 2.5;
        expected.gas.coordinates[1] = [0.2, -4.0, 8.0];
        expected.gas.velocities[1] = [20.0, 1.25, -2.5];

        let path = temporary_hdf5_path("roundtrip");
        write_soundwave(&path, SoundWaveWriteView::try_from(&expected).unwrap()).unwrap();

        let actual = read_soundwave(&path).unwrap();
        assert_eq!(actual, expected);

        let file = hdf5::File::open(&path).unwrap();
        let header = file.group("Header").unwrap();
        let mut attribute_names = header.attr_names().unwrap();
        attribute_names.sort();
        assert_eq!(
            attribute_names,
            [
                "BoxSize",
                "ComovingIntegrationOn",
                "Effective_Kernel_NeighborNumber",
                "Fixed_ForceSoftening_Keplerian_Kernel_Extent",
                "Flag_Cooling",
                "Flag_DoublePrecision",
                "Flag_Feedback",
                "Flag_IC_Info",
                "Flag_Metals",
                "Flag_Sfr",
                "Flag_StellarAge",
                "GIZMO_RustPort_version",
                "GIZMO_version",
                "Gravitational_Constant_In_Code_Inits",
                "HubbleParam",
                "Kernel_Function_ID",
                "MassTable",
                "Maximum_Mass_For_Cell_Split",
                "Minimum_Mass_For_Cell_Merge",
                "NumFilesPerSnapshot",
                "NumPart_ThisFile",
                "NumPart_Total",
                "NumPart_Total_HighWord",
                "Redshift",
                "Time",
                "UnitLength_In_CGS",
                "UnitMass_In_CGS",
                "UnitVelocity_In_CGS",
            ]
        );
        assert_eq!(
            header
                .attr("Flag_DoublePrecision")
                .unwrap()
                .read_scalar::<i32>()
                .unwrap(),
            1
        );
        assert_eq!(
            header
                .attr("NumPart_Total")
                .unwrap()
                .read_raw::<u32>()
                .unwrap(),
            vec![3, 0, 0, 0, 0, 0]
        );
        for (name, expected_values) in [
            ("NumPart_ThisFile", vec![3_i32, 0, 0, 0, 0, 0]),
            ("NumFilesPerSnapshot", vec![1_i32]),
        ] {
            let attribute = header.attr(name).unwrap();
            assert!(attribute.dtype().unwrap().is::<i32>());
            assert_eq!(attribute.read_raw::<i32>().unwrap(), expected_values);
        }
        for name in ["NumPart_Total", "NumPart_Total_HighWord"] {
            assert!(header.attr(name).unwrap().dtype().unwrap().is::<u32>());
        }
        assert_eq!(
            header
                .attr("NumPart_Total_HighWord")
                .unwrap()
                .read_raw::<u32>()
                .unwrap(),
            vec![0; PARTICLE_TYPES]
        );
        let mass_table = header.attr("MassTable").unwrap();
        assert!(mass_table.dtype().unwrap().is::<f64>());
        assert_eq!(
            mass_table.read_raw::<f64>().unwrap(),
            vec![0.0; PARTICLE_TYPES]
        );
        let gas = file.group("PartType0").unwrap();
        let ids = gas.dataset("ParticleIDs").unwrap();
        assert!(ids.dtype().unwrap().is::<u32>());
        assert_eq!(ids.read_raw::<u32>().unwrap(), [1, 2, 3]);
        assert_eq!(
            gas.dataset("Coordinates").unwrap().shape(),
            [expected.gas.len(), VECTOR_COMPONENTS]
        );
        assert_eq!(
            gas.dataset("Velocities").unwrap().shape(),
            [expected.gas.len(), VECTOR_COMPONENTS]
        );
        for (name, expected_value) in [
            ("ComovingIntegrationOn", 0),
            ("Flag_Cooling", 0),
            ("Flag_Feedback", 0),
            ("Flag_IC_Info", 0),
            ("Flag_Metals", 0),
            ("Flag_Sfr", 0),
            ("Flag_StellarAge", 0),
            ("GIZMO_version", NON_UPSTREAM_GIZMO_VERSION),
            ("GIZMO_RustPort_version", GIZMO_RUST_PORT_VERSION),
            ("Kernel_Function_ID", LEGACY_KERNEL_FUNCTION_ID),
        ] {
            let attribute = header.attr(name).unwrap();
            assert!(attribute.dtype().unwrap().is::<i32>());
            assert_eq!(attribute.read_scalar::<i32>().unwrap(), expected_value);
        }
        for (name, expected_value) in [
            ("Effective_Kernel_NeighborNumber", 4.0),
            (
                "Gravitational_Constant_In_Code_Inits",
                LEGACY_GRAVITATIONAL_CONSTANT,
            ),
            ("HubbleParam", 1.0),
            ("Maximum_Mass_For_Cell_Split", 3.01 * 3.0),
            ("Minimum_Mass_For_Cell_Merge", 0.49),
            ("Redshift", 0.0),
            ("UnitLength_In_CGS", 1.0),
            ("UnitMass_In_CGS", 1.0),
            ("UnitVelocity_In_CGS", 1.0),
        ] {
            let attribute = header.attr(name).unwrap();
            assert!(attribute.dtype().unwrap().is::<f64>());
            assert_eq!(
                attribute.read_scalar::<f64>().unwrap().to_bits(),
                expected_value.to_bits()
            );
        }
        let softening = header
            .attr("Fixed_ForceSoftening_Keplerian_Kernel_Extent")
            .unwrap();
        assert!(softening.dtype().unwrap().is::<f64>());
        assert_eq!(
            softening.read_raw::<f64>().unwrap(),
            vec![0.0; PARTICLE_TYPES]
        );
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn soundwave_writer_checks_required_fields_before_creating_file() {
        let mut snapshot = valid_snapshot();
        snapshot.gas.density = None;
        assert_eq!(
            SoundWaveWriteView::try_from(&snapshot).unwrap_err(),
            ValidationError::MissingRequiredField("Density")
        );

        let path = temporary_hdf5_path("invalid");
        let mut complete = valid_snapshot();
        complete.gas.smoothing_length.as_mut().unwrap().pop();
        let view = SoundWaveWriteView::try_from(&complete).unwrap();
        assert!(matches!(
            write_soundwave(&path, view),
            Err(OutputError::Validation(ValidationError::ColumnLength {
                field: "SmoothingLength",
                ..
            }))
        ));
        assert!(!path.exists());

        let path = temporary_hdf5_path("missing-header-metadata");
        let mut missing_metadata = valid_snapshot();
        missing_metadata.header.effective_kernel_neighbors = None;
        assert!(matches!(
            write_soundwave(
                &path,
                SoundWaveWriteView::try_from(&missing_metadata).unwrap()
            ),
            Err(OutputError::Validation(
                ValidationError::MissingRequiredField("Effective_Kernel_NeighborNumber")
            ))
        ));
        assert!(!path.exists());

        let path = temporary_hdf5_path("wide-id");
        let mut wide_id = valid_snapshot();
        wide_id.gas.ids[0] = u64::from(u32::MAX) + 1;
        assert!(matches!(
            write_soundwave(&path, SoundWaveWriteView::try_from(&wide_id).unwrap()),
            Err(OutputError::Validation(
                ValidationError::LegacyParticleIdOverflow(_)
            ))
        ));
        assert!(!path.exists());
    }

    #[test]
    fn dustywave_writer_roundtrips_both_particle_types() {
        let mut expected = valid_dustywave_snapshot();
        expected.validate_and_sort().unwrap();
        expected.header.time = 1.2;

        let path = temporary_hdf5_path("dustywave-roundtrip");
        write_dustywave(&path, DustyWaveWriteView::try_from(&expected).unwrap()).unwrap();
        let actual = read_dustywave(&path).unwrap();
        assert_eq!(actual, expected);

        let file = hdf5::File::open(&path).unwrap();
        let header = file.group("Header").unwrap();
        assert_eq!(
            header
                .attr("NumPart_ThisFile")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [3, 0, 0, 2, 0, 0]
        );
        assert_eq!(
            header
                .attr("NumPart_Total")
                .unwrap()
                .read_raw::<u32>()
                .unwrap(),
            [3, 0, 0, 2, 0, 0]
        );
        let grains = file.group("PartType3").unwrap();
        let mut dataset_names = grains.member_names().unwrap();
        dataset_names.sort();
        assert_eq!(
            dataset_names,
            [
                "Coordinates",
                "GrainSize",
                "Masses",
                "ParticleIDs",
                "SmoothingLength",
                "Velocities",
            ]
        );
        assert!(
            grains
                .dataset("ParticleIDs")
                .unwrap()
                .dtype()
                .unwrap()
                .is::<u32>()
        );
        assert_eq!(
            grains
                .dataset("ParticleIDs")
                .unwrap()
                .read_raw::<u32>()
                .unwrap(),
            [4, 5]
        );
        assert_eq!(
            grains.dataset("Coordinates").unwrap().shape(),
            [expected.grains.len(), VECTOR_COMPONENTS]
        );
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dustywave_writer_rejects_invalid_state_before_creating_file() {
        let mut missing_hsml = valid_dustywave_snapshot();
        missing_hsml.grains.smoothing_length = None;
        assert_eq!(
            DustyWaveWriteView::try_from(&missing_hsml).unwrap_err(),
            ValidationError::MissingRequiredField("PartType3/SmoothingLength")
        );

        let path = temporary_hdf5_path("dustywave-invalid");
        let mut snapshot = valid_dustywave_snapshot();
        snapshot.grains.ids[0] = 2;
        assert!(matches!(
            write_dustywave(&path, DustyWaveWriteView::try_from(&snapshot).unwrap()),
            Err(OutputError::Validation(
                ValidationError::DuplicateParticleId(2)
            ))
        ));
        assert!(!path.exists());

        let path = temporary_hdf5_path("dustywave-wide-grain-id");
        let mut wide_id = valid_dustywave_snapshot();
        wide_id.grains.ids[0] = u64::from(u32::MAX) + 1;
        assert!(matches!(
            write_dustywave(&path, DustyWaveWriteView::try_from(&wide_id).unwrap()),
            Err(OutputError::Validation(
                ValidationError::LegacyParticleIdOverflow(_)
            ))
        ));
        assert!(!path.exists());
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

    #[test]
    #[ignore = "requires GIZMO_DUSTYWAVE_IC; run via validation oracle script"]
    fn reads_external_dustywave_fixture_when_configured() {
        let path = std::env::var_os("GIZMO_DUSTYWAVE_IC")
            .expect("GIZMO_DUSTYWAVE_IC must identify the pinned fixture");
        let snapshot = read_dustywave(path).unwrap();
        assert_eq!(snapshot.gas.len(), 64);
        assert_eq!(snapshot.grains.len(), 64);
        assert_eq!(snapshot.header.num_part_total, [64, 0, 0, 64, 0, 0]);
        assert!(snapshot.gas.ids.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(snapshot.grains.ids.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            snapshot
                .gas
                .ids
                .iter()
                .all(|id| snapshot.grains.ids.binary_search(id).is_err())
        );
    }
}
