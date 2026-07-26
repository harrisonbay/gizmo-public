#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gizmo_cli::{CliError, Invocation, RestartFlag, USAGE};
use gizmo_config::ConfigManifest;
use gizmo_hydro::{
    BoundaryMode1d, GradientEstimate, LEGACY_TIMEBASE_TICKS, MeshlessPoint1d, MfmDriftState1d,
    MfmEvolvingState1d, SynchronizedTimeline1d, begin_mfm_kdk_1d, begin_mfm_reflective_kdk_1d,
    density_at_hsml_1d, density_at_hsml_1d_with_boundary, finish_mfm_kdk_1d, gradients_at_hsml_1d,
    inverse_moments_1d, meshless_face_geometry_1d, mfm_spatial_rates_1d_with_boundary,
    select_public_soundwave_timestep_1d, solve_public_c_initial_smoothing_lengths_1d_with_boundary,
};
use gizmo_io::{SnapshotHeader, SoundWaveWriteView, read_soundwave, write_soundwave};
use gizmo_params::SoundwaveParameters;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(ApplicationError::Cli(CliError::HelpRequested)) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("gizmo: {error}");
            if matches!(error, ApplicationError::Cli(_)) {
                eprintln!("\n{USAGE}");
            }
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), ApplicationError> {
    let invocation = Invocation::parse(env::args_os().skip(1)).map_err(ApplicationError::Cli)?;
    reject_unsupported_restart(invocation.restart)?;
    let manifest =
        ConfigManifest::from_path(&invocation.config_file).map_err(ApplicationError::Config)?;

    eprintln!(
        "validated {} enabled options; config sha256={}",
        manifest.len(),
        manifest.sha256()
    );
    eprintln!(
        "parameter file: {}; restart flag: {}",
        invocation.parameter_file.display(),
        invocation.restart as u8
    );

    let profile = validate_strict_config(&manifest)?;
    let initialized = initialize_profile(&invocation.parameter_file, profile)?;
    initialized.validate_owned_state()?;
    if invocation.initialize_only {
        initialized.print_initialization(&manifest.sha256());
        Ok(())
    } else {
        evolve_profile(initialized)
    }
}

fn reject_unsupported_restart(restart: RestartFlag) -> Result<(), ApplicationError> {
    if restart == RestartFlag::InitialConditions {
        Ok(())
    } else {
        Err(ApplicationError::UnsupportedRestart(restart))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StrictProfile {
    Soundwave,
    EqualMassShocktube,
    InteractingBlast,
}

impl StrictProfile {
    const fn gamma(self) -> f64 {
        match self {
            Self::Soundwave => 5.0 / 3.0,
            Self::EqualMassShocktube | Self::InteractingBlast => 1.4,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Soundwave => "soundwave",
            Self::EqualMassShocktube => "equal-mass shocktube",
            Self::InteractingBlast => "interacting blastwave",
        }
    }

    const fn boundary(self) -> BoundaryMode1d {
        match self {
            Self::Soundwave | Self::EqualMassShocktube => BoundaryMode1d::Periodic,
            Self::InteractingBlast => BoundaryMode1d::Reflective,
        }
    }
}

fn validate_strict_config(manifest: &ConfigManifest) -> Result<StrictProfile, ApplicationError> {
    const REQUIRED_FLAGS: [&str; 5] = [
        "DEVELOPER_MODE",
        "HYDRO_MESHLESS_FINITE_MASS",
        "INPUT_IN_DOUBLEPRECISION",
        "OUTPUT_IN_DOUBLEPRECISION",
        "SELFGRAVITY_OFF",
    ];
    const ALLOWED: [&str; 11] = [
        "BOX_BND_PARTICLES",
        "BOX_PERIODIC",
        "BOX_REFLECT_X",
        "BOX_SPATIAL_DIMENSION",
        "DEVELOPER_MODE",
        "EOS_GAMMA",
        "FORCE_EQUAL_TIMESTEPS",
        "HYDRO_MESHLESS_FINITE_MASS",
        "INPUT_IN_DOUBLEPRECISION",
        "OUTPUT_IN_DOUBLEPRECISION",
        "SELFGRAVITY_OFF",
    ];
    for option in manifest.iter() {
        if !ALLOWED.contains(&option.name.as_str()) {
            return Err(ApplicationError::UnsupportedConfig(format!(
                "option `{}` is outside the strict one-dimensional hydro profiles",
                option.name
            )));
        }
    }
    for required in REQUIRED_FLAGS {
        require_config_flag(manifest, required)?;
    }
    for topology_flag in [
        "BOX_BND_PARTICLES",
        "BOX_PERIODIC",
        "BOX_REFLECT_X",
        "FORCE_EQUAL_TIMESTEPS",
    ] {
        if manifest.get(topology_flag).is_some() {
            require_config_flag(manifest, topology_flag)?;
        }
    }
    require_config_value(manifest, "BOX_SPATIAL_DIMENSION", "1")?;
    let gamma = manifest
        .get("EOS_GAMMA")
        .and_then(|option| option.value.as_deref());
    let periodic = manifest.get("BOX_PERIODIC").is_some();
    let equal_timesteps = manifest.get("FORCE_EQUAL_TIMESTEPS").is_some();
    let reflective_x = manifest.get("BOX_REFLECT_X").is_some();
    let boundary_particles = manifest.get("BOX_BND_PARTICLES").is_some();

    match (
        gamma,
        periodic,
        equal_timesteps,
        reflective_x,
        boundary_particles,
    ) {
        (Some("(5.0/3.0)"), true, true, false, false) => Ok(StrictProfile::Soundwave),
        (Some("(1.4)"), true, true, false, false) => Ok(StrictProfile::EqualMassShocktube),
        (Some("(1.4)"), false, false, true, true) => Ok(StrictProfile::InteractingBlast),
        _ => Err(ApplicationError::UnsupportedConfig(format!(
            "configuration does not exactly match a ported profile: \
             EOS_GAMMA={gamma:?}, BOX_PERIODIC={periodic}, \
             FORCE_EQUAL_TIMESTEPS={equal_timesteps}, BOX_REFLECT_X={reflective_x}, \
             BOX_BND_PARTICLES={boundary_particles}"
        ))),
    }
}

fn require_config_flag(manifest: &ConfigManifest, name: &str) -> Result<(), ApplicationError> {
    let actual = manifest.get(name).map(|option| option.value.as_deref());
    match actual {
        Some(None) => Ok(()),
        None => Err(ApplicationError::UnsupportedConfig(format!(
            "required option `{name}` is missing"
        ))),
        Some(Some(value)) => Err(ApplicationError::UnsupportedConfig(format!(
            "`{name}` must be a bare enabled flag, found value `{value}`"
        ))),
    }
}

fn require_config_value(
    manifest: &ConfigManifest,
    name: &str,
    expected: &str,
) -> Result<(), ApplicationError> {
    let actual = manifest
        .get(name)
        .and_then(|option| option.value.as_deref());
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(ApplicationError::UnsupportedConfig(format!(
            "`{name}` must equal `{expected}`, found {actual:?}"
        )))
    }
}

fn initialize_profile(
    parameter_file: &Path,
    profile: StrictProfile,
) -> Result<InitializedSoundwave, ApplicationError> {
    let parameters = read_profile_parameters(parameter_file, profile)?;
    let fixture_path = resolve_initial_conditions(&parameters.init_cond_file);
    let snapshot = read_soundwave(&fixture_path).map_err(ApplicationError::Input)?;
    if snapshot.header.box_size.to_bits() != parameters.box_size.to_bits() {
        return Err(ApplicationError::StateMismatch(format!(
            "parameter BoxSize={} differs from HDF5 BoxSize={}",
            parameters.box_size, snapshot.header.box_size
        )));
    }
    let positions: Vec<f64> = snapshot
        .gas
        .coordinates
        .iter()
        .map(|coordinate| coordinate[0])
        .collect();
    let particle_count = u32::try_from(snapshot.gas.len())
        .map_err(|_| ApplicationError::StateMismatch("particle count exceeds u32".to_owned()))?;
    let solved = solve_public_c_initial_smoothing_lengths_1d_with_boundary(
        &positions,
        &snapshot.gas.masses,
        snapshot.header.box_size,
        parameters.desired_num_neighbors,
        parameters.max_neighbor_deviation,
        profile.boundary(),
    )
    .map_err(ApplicationError::Hydro)?;

    let summary = if profile == StrictProfile::Soundwave {
        let expected_density = snapshot
            .gas
            .density
            .as_deref()
            .ok_or(ApplicationError::MissingDataset("Density"))?;
        let legacy_hsml = snapshot
            .gas
            .smoothing_length
            .as_deref()
            .ok_or(ApplicationError::MissingDataset("SmoothingLength"))?;
        let at_legacy_hsml = density_at_hsml_1d(
            &positions,
            &snapshot.gas.masses,
            legacy_hsml,
            snapshot.header.box_size,
        )
        .map_err(ApplicationError::Hydro)?;
        let summary = summarize_initialization(
            &snapshot,
            &positions,
            expected_density,
            legacy_hsml,
            &at_legacy_hsml,
            &solved,
            (particle_count, parameters.desired_num_neighbors),
        )?;
        summary.validate(parameters.max_neighbor_deviation)?;
        Some(summary)
    } else {
        None
    };
    let particle_ids = snapshot.gas.ids;
    let transverse_vectors = snapshot
        .gas
        .coordinates
        .into_iter()
        .zip(&snapshot.gas.velocities)
        .map(|(position, velocity)| TransverseVectorShell {
            position: [position[1], position[2]],
            velocity: [velocity[1], velocity[2]],
        })
        .collect();
    let state = MfmEvolvingState1d {
        positions,
        masses: snapshot.gas.masses,
        velocities: snapshot
            .gas
            .velocities
            .iter()
            .map(|velocity| velocity[0])
            .collect(),
        specific_internal_energy: snapshot.gas.internal_energy,
        smoothing_lengths: solved
            .into_iter()
            .map(|particle| particle.smoothing_length)
            .collect(),
        box_size: snapshot.header.box_size,
        gamma: profile.gamma(),
    };
    Ok(InitializedSoundwave {
        profile,
        parameters,
        particle_ids,
        transverse_vectors,
        state,
        summary,
    })
}

fn read_profile_parameters(
    parameter_file: &Path,
    profile: StrictProfile,
) -> Result<SoundwaveParameters, ApplicationError> {
    if profile == StrictProfile::Soundwave {
        return SoundwaveParameters::from_path(parameter_file)
            .map_err(ApplicationError::Parameters);
    }
    let input = fs::read_to_string(parameter_file).map_err(ApplicationError::ParameterFile)?;
    let mut retained = Vec::new();
    let mut profile_tags = BTreeSet::new();
    for (line_index, raw_line) in input.lines().enumerate() {
        let definition = raw_line
            .split_once('%')
            .map_or(raw_line, |(before, _)| before)
            .trim();
        if definition.is_empty() {
            retained.push(raw_line);
            continue;
        }
        let mut tokens = definition.split_whitespace();
        let tag = tokens.next().unwrap_or_default();
        let expected = match tag {
            "TimeBegin" => Some("0"),
            "ICFormat" | "SnapFormat" => Some("3"),
            "BufferSize" => Some("8"),
            _ => None,
        };
        if let Some(expected) = expected {
            let value = tokens.next();
            if value != Some(expected) || tokens.next().is_some() {
                return Err(ApplicationError::UnsupportedParameters(format!(
                    "line {}: `{tag}` must equal `{expected}` for the {}",
                    line_index + 1,
                    profile.name()
                )));
            }
            if !profile_tags.insert(tag) {
                return Err(ApplicationError::UnsupportedParameters(format!(
                    "line {}: duplicate profile parameter `{tag}`",
                    line_index + 1
                )));
            }
        } else {
            retained.push(raw_line);
        }
    }
    for required in ["TimeBegin", "ICFormat", "SnapFormat", "BufferSize"] {
        if !profile_tags.contains(required) {
            return Err(ApplicationError::UnsupportedParameters(format!(
                "required {} parameter `{required}` is missing",
                profile.name()
            )));
        }
    }
    let parameters = SoundwaveParameters::parse(&retained.join("\n"))
        .map_err(|error| ApplicationError::UnsupportedParameters(error.to_string()))?;
    if profile == StrictProfile::InteractingBlast
        && parameters.min_timestep != Some(parameters.max_timestep)
    {
        return Err(ApplicationError::UnsupportedParameters(
            "interacting blastwave requires equal explicit MinSizeTimestep and \
             MaxSizeTimestep values"
                .to_owned(),
        ));
    }
    if profile == StrictProfile::InteractingBlast {
        let required_scalars: [(&str, f64, f64); 10] = [
            ("TimeMax", parameters.time_max, 0.038),
            ("BoxSize", parameters.box_size, 1.0),
            ("TimeBetSnapshot", parameters.time_between_snapshots, 0.0038),
            ("MaxSizeTimestep", parameters.max_timestep, 2.0e-7),
            ("DesNumNgb", parameters.desired_num_neighbors, 4.0),
            ("ErrTolIntAccuracy", parameters.integration_accuracy, 0.002),
            ("CourantFac", parameters.courant_factor, 0.01),
            (
                "MaxRMSDisplacementFac",
                parameters.max_rms_displacement_factor,
                0.05,
            ),
            ("ErrTolForceAcc", parameters.force_accuracy, 0.001),
            (
                "MaxNumNgbDeviation",
                parameters.max_neighbor_deviation,
                0.05,
            ),
        ];
        for (field, actual, expected) in required_scalars {
            if actual.to_bits() != expected.to_bits() {
                return Err(ApplicationError::UnsupportedParameters(format!(
                    "interacting blastwave requires `{field} {expected}`, found `{actual}`"
                )));
            }
        }
        if parameters.init_cond_file != "interactblast_ics" {
            return Err(ApplicationError::UnsupportedParameters(format!(
                "interacting blastwave requires `InitCondFile interactblast_ics`, found `{}`",
                parameters.init_cond_file
            )));
        }
    }
    Ok(parameters)
}

fn summarize_initialization(
    snapshot: &gizmo_io::SoundWaveSnapshot,
    positions: &[f64],
    expected_density: &[f64],
    legacy_hsml: &[f64],
    at_legacy_hsml: &[gizmo_hydro::DensityEstimate],
    solved: &[gizmo_hydro::AdaptiveDensityEstimate],
    neighbor_constraint: (u32, f64),
) -> Result<InitializationSummary, ApplicationError> {
    let (particle_count, desired_num_neighbors) = neighbor_constraint;
    let max_density_relative_error = at_legacy_hsml
        .iter()
        .zip(expected_density)
        .map(|(estimate, expected)| (estimate.density - expected).abs() / expected)
        .fold(0.0, f64::max);
    let max_hsml_relative_difference = solved
        .iter()
        .zip(legacy_hsml)
        .map(|(particle, legacy)| (particle.smoothing_length - legacy).abs() / legacy)
        .fold(0.0, f64::max);
    let max_neighbor_deviation = solved
        .iter()
        .map(|particle| (particle.estimate.effective_neighbors - desired_num_neighbors).abs())
        .fold(0.0, f64::max);
    let [
        density_gradient_error,
        velocity_gradient_error,
        pressure_gradient_error,
    ] = soundwave_gradient_errors(snapshot, positions, expected_density, legacy_hsml)?;
    let max_face_area_deviation = soundwave_face_area_deviation(
        positions,
        &snapshot.gas.masses,
        expected_density,
        legacy_hsml,
        snapshot.header.box_size,
    )?;
    Ok(InitializationSummary {
        particle_count,
        box_size: snapshot.header.box_size,
        max_density_relative_error,
        max_hsml_relative_difference,
        max_neighbor_deviation,
        density_gradient_error,
        velocity_gradient_error,
        pressure_gradient_error,
        max_face_area_deviation,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TransverseVectorShell {
    position: [f64; 2],
    velocity: [f64; 2],
}

#[derive(Clone, Debug, PartialEq)]
struct InitializedSoundwave {
    profile: StrictProfile,
    parameters: SoundwaveParameters,
    particle_ids: Vec<u64>,
    transverse_vectors: Vec<TransverseVectorShell>,
    state: MfmEvolvingState1d,
    summary: Option<InitializationSummary>,
}

impl InitializedSoundwave {
    fn validate_owned_state(&self) -> Result<(), ApplicationError> {
        let particle_count = self.state.positions.len();
        for (field, actual) in [
            ("ParticleIDs", self.particle_ids.len()),
            ("transverse vectors", self.transverse_vectors.len()),
            ("positions", self.state.positions.len()),
            ("masses", self.state.masses.len()),
            ("velocities", self.state.velocities.len()),
            (
                "specific internal energy",
                self.state.specific_internal_energy.len(),
            ),
            ("smoothing lengths", self.state.smoothing_lengths.len()),
        ] {
            if actual != particle_count {
                return Err(ApplicationError::StateMismatch(format!(
                    "{field} has {actual} entries, expected {particle_count}"
                )));
            }
        }
        if self.particle_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ApplicationError::StateMismatch(
                "ParticleIDs are not strictly sorted".to_owned(),
            ));
        }
        if self.profile == StrictProfile::InteractingBlast
            && (self.particle_ids.len() != 512
                || self.particle_ids.iter().copied().ne(1_u64..=512_u64))
        {
            return Err(ApplicationError::StateMismatch(
                "interacting-blast ParticleIDs must be exactly 1 through 512; \
                 ID zero would activate unported BOX_BND_PARTICLES semantics"
                    .to_owned(),
            ));
        }
        if self.parameters.box_size.to_bits() != self.state.box_size.to_bits()
            || self
                .summary
                .is_some_and(|summary| summary.box_size.to_bits() != self.state.box_size.to_bits())
        {
            return Err(ApplicationError::StateMismatch(
                "runtime bundle has inconsistent box sizes".to_owned(),
            ));
        }
        if self.state.gamma.to_bits() != self.profile.gamma().to_bits() {
            return Err(ApplicationError::StateMismatch(
                "runtime bundle has inconsistent EOS gamma".to_owned(),
            ));
        }
        if self
            .transverse_vectors
            .iter()
            .flat_map(|shell| shell.position.into_iter().chain(shell.velocity))
            .any(|component| !component.is_finite())
        {
            return Err(ApplicationError::StateMismatch(
                "runtime bundle has non-finite transverse vectors".to_owned(),
            ));
        }
        Ok(())
    }

    fn print_initialization(&self, config_sha256: &str) {
        if let Some(summary) = self.summary {
            summary.print(config_sha256);
            return;
        }
        println!("{{");
        println!("  \"config_sha256\": \"{config_sha256}\",");
        println!("  \"profile\": \"{}\",", self.profile.name());
        println!("  \"particles\": {},", self.state.positions.len());
        println!("  \"box_size\": {},", self.state.box_size);
        println!("  \"gamma\": {}", self.state.gamma);
        println!("}}");
    }
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn evolve_profile(mut initialized: InitializedSoundwave) -> Result<(), ApplicationError> {
    let output_dir = PathBuf::from(&initialized.parameters.output_dir);
    fs::create_dir_all(&output_dir).map_err(ApplicationError::OutputDirectory)?;
    let mut timeline = SynchronizedTimeline1d::new(0.0, initialized.parameters.time_max)
        .map_err(ApplicationError::Hydro)?;
    let tick_duration = initialized.parameters.time_max / LEGACY_TIMEBASE_TICKS as f64;
    let mut rates = mfm_spatial_rates_1d_with_boundary(
        initialized.state.as_view(),
        initialized.profile.boundary(),
    )
    .map_err(ApplicationError::Hydro)?;
    let mut next_output_time = 0.0_f64;
    let mut next_output_tick = Some(0_u64);
    let mut snapshot_number = 0_u32;
    let mut last_output_tick = None;
    let mut step_count = 0_u64;

    while !timeline.is_finished() {
        let desired_timestep = if initialized.profile == StrictProfile::InteractingBlast {
            // Public C computes the physical criteria, then clamps them between
            // equal explicit MinSizeTimestep and MaxSizeTimestep values.
            initialized.parameters.max_timestep
        } else {
            select_public_soundwave_timestep_1d(
                initialized.state.as_view(),
                &rates,
                initialized.parameters.max_timestep,
                initialized.parameters.courant_factor,
                initialized.parameters.integration_accuracy,
            )
            .map_err(ApplicationError::Hydro)?
            .duration
        };
        let synchronized = timeline
            .select_step(desired_timestep, initialized.parameters.max_timestep)
            .map_err(ApplicationError::Hydro)?;
        let start_tick = timeline.current_tick();
        let end_tick = start_tick + synchronized.ticks;
        let mut prepared = if initialized.profile == StrictProfile::InteractingBlast {
            begin_mfm_reflective_kdk_1d(
                &initialized.state,
                &rates,
                synchronized.duration,
                0.0,
                &initialized.particle_ids,
            )
        } else {
            begin_mfm_kdk_1d(&initialized.state, &rates, synchronized.duration, 0.0)
        }
        .map_err(ApplicationError::Hydro)?;

        while next_output_tick.is_some_and(|tick| tick <= end_tick) {
            let output_tick = next_output_tick.expect("checked above");
            if output_tick < start_tick {
                return Err(ApplicationError::StateMismatch(format!(
                    "snapshot tick {output_tick} precedes current tick {start_tick}"
                )));
            }
            let elapsed = (output_tick - start_tick) as f64 * tick_duration;
            let drift = prepared
                .drift_state(elapsed)
                .map_err(ApplicationError::Hydro)?;
            write_drift_snapshot(
                &initialized,
                &drift,
                output_dir.join(format!("snapshot_{snapshot_number:03}.hdf5")),
                output_tick as f64 * tick_duration,
            )?;
            last_output_tick = Some(output_tick);
            snapshot_number = snapshot_number.checked_add(1).ok_or_else(|| {
                ApplicationError::StateMismatch("snapshot number overflow".to_owned())
            })?;

            next_output_time += initialized.parameters.time_between_snapshots;
            next_output_tick = (next_output_time <= initialized.parameters.time_max)
                .then(|| legacy_output_tick(next_output_time, tick_duration));
            if next_output_tick.is_some_and(|tick| tick <= output_tick) {
                return Err(ApplicationError::StateMismatch(
                    "snapshot cadence does not advance on the integer timeline".to_owned(),
                ));
            }
        }

        let (endpoint, new_rates) = finish_mfm_kdk_1d(
            prepared,
            initialized.parameters.desired_num_neighbors,
            initialized.parameters.max_neighbor_deviation,
        )
        .map_err(ApplicationError::Hydro)?;
        initialized.state = endpoint;
        rates = new_rates;
        timeline
            .advance(synchronized)
            .map_err(ApplicationError::Hydro)?;
        step_count = step_count
            .checked_add(1)
            .ok_or_else(|| ApplicationError::StateMismatch("step count overflow".to_owned()))?;
    }

    if last_output_tick != Some(LEGACY_TIMEBASE_TICKS) {
        write_completed_snapshot(
            &initialized,
            output_dir.join(format!("snapshot_{snapshot_number:03}.hdf5")),
            initialized.parameters.time_max,
        )?;
        snapshot_number = snapshot_number.checked_add(1).ok_or_else(|| {
            ApplicationError::StateMismatch("snapshot number overflow".to_owned())
        })?;
    }
    eprintln!(
        "completed {step_count} synchronized steps to t={:.17e}; wrote {snapshot_number} snapshots",
        timeline.current_time()
    );
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn legacy_output_tick(output_time: f64, tick_duration: f64) -> u64 {
    (output_time / tick_duration) as u64
}

fn write_drift_snapshot(
    initialized: &InitializedSoundwave,
    drift: &MfmDriftState1d,
    path: PathBuf,
    time: f64,
) -> Result<(), ApplicationError> {
    write_snapshot_columns(
        initialized,
        &drift.positions,
        &drift.conserved_velocities,
        &drift.predicted_specific_internal_energy,
        &drift.predicted_density,
        &drift.predicted_smoothing_lengths,
        path,
        time,
    )
}

fn write_completed_snapshot(
    initialized: &InitializedSoundwave,
    path: PathBuf,
    time: f64,
) -> Result<(), ApplicationError> {
    let density: Vec<f64> = density_at_hsml_1d_with_boundary(
        &initialized.state.positions,
        &initialized.state.masses,
        &initialized.state.smoothing_lengths,
        initialized.state.box_size,
        initialized.profile.boundary(),
    )
    .map_err(ApplicationError::Hydro)?
    .into_iter()
    .map(|estimate| estimate.density)
    .collect();
    write_snapshot_columns(
        initialized,
        &initialized.state.positions,
        &initialized.state.velocities,
        &initialized.state.specific_internal_energy,
        &density,
        &initialized.state.smoothing_lengths,
        path,
        time,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_snapshot_columns(
    initialized: &InitializedSoundwave,
    positions: &[f64],
    velocities: &[f64],
    internal_energy: &[f64],
    density: &[f64],
    smoothing_lengths: &[f64],
    path: PathBuf,
    time: f64,
) -> Result<(), ApplicationError> {
    let coordinates: Vec<[f64; 3]> = positions
        .iter()
        .zip(&initialized.transverse_vectors)
        .map(|(&x, shell)| [x, shell.position[0], shell.position[1]])
        .collect();
    let velocity_vectors: Vec<[f64; 3]> = velocities
        .iter()
        .zip(&initialized.transverse_vectors)
        .map(|(&x, shell)| [x, shell.velocity[0], shell.velocity[1]])
        .collect();
    let gas_count = u64::try_from(initialized.particle_ids.len())
        .map_err(|_| ApplicationError::StateMismatch("particle count exceeds u64".to_owned()))?;
    let header = SnapshotHeader {
        time,
        box_size: initialized.state.box_size,
        num_part_total: [gas_count, 0, 0, 0, 0, 0],
        double_precision: true,
        effective_kernel_neighbors: Some(initialized.parameters.desired_num_neighbors),
    };
    write_soundwave(
        path,
        SoundWaveWriteView {
            header: &header,
            coordinates: &coordinates,
            velocities: &velocity_vectors,
            ids: &initialized.particle_ids,
            masses: &initialized.state.masses,
            internal_energy,
            density,
            smoothing_length: smoothing_lengths,
        },
    )
    .map_err(ApplicationError::Output)
}

fn soundwave_gradient_errors(
    snapshot: &gizmo_io::SoundWaveSnapshot,
    positions: &[f64],
    density: &[f64],
    smoothing_lengths: &[f64],
) -> Result<[f64; 3], ApplicationError> {
    let velocity: Vec<f64> = snapshot
        .gas
        .velocities
        .iter()
        .map(|components| components[0])
        .collect();
    let pressure: Vec<f64> = density
        .iter()
        .zip(&snapshot.gas.internal_energy)
        .map(|(density, internal_energy)| (2.0 / 3.0) * density * internal_energy)
        .collect();
    let density_gradient_error = soundwave_gradient_error(
        positions,
        density,
        smoothing_lengths,
        snapshot.header.box_size,
        0.0,
        true,
    )?;
    let velocity_gradient_error = soundwave_gradient_error(
        positions,
        &velocity,
        smoothing_lengths,
        snapshot.header.box_size,
        0.1,
        false,
    )?;
    let pressure_gradient_error = soundwave_gradient_error(
        positions,
        &pressure,
        smoothing_lengths,
        snapshot.header.box_size,
        0.1,
        true,
    )?;
    Ok([
        density_gradient_error,
        velocity_gradient_error,
        pressure_gradient_error,
    ])
}

fn soundwave_face_area_deviation(
    positions: &[f64],
    masses: &[f64],
    density: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
) -> Result<f64, ApplicationError> {
    let inverse_moments = inverse_moments_1d(positions, smoothing_lengths, box_size)
        .map_err(ApplicationError::Hydro)?;
    let point = |index| MeshlessPoint1d {
        position: positions[index],
        mass: masses[index],
        density: density[index],
        smoothing_length: smoothing_lengths[index],
        inverse_moment: inverse_moments[index],
    };
    let mut spatial_order: Vec<usize> = (0..positions.len()).collect();
    spatial_order.sort_unstable_by(|left, right| positions[*left].total_cmp(&positions[*right]));
    spatial_order
        .iter()
        .enumerate()
        .map(|(order_index, &index)| {
            let neighbor = spatial_order[(order_index + 1) % spatial_order.len()];
            meshless_face_geometry_1d(point(index), point(neighbor), box_size)
                .map(|face| (face.area - 1.0).abs())
                .map_err(ApplicationError::Hydro)
        })
        .try_fold(0.0, |maximum, deviation| {
            deviation.map(|value| f64::max(maximum, value))
        })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct InitializationSummary {
    particle_count: u32,
    box_size: f64,
    max_density_relative_error: f64,
    max_hsml_relative_difference: f64,
    max_neighbor_deviation: f64,
    density_gradient_error: f64,
    velocity_gradient_error: f64,
    pressure_gradient_error: f64,
    max_face_area_deviation: f64,
}

impl InitializationSummary {
    fn validate(self, neighbor_tolerance: f64) -> Result<(), ApplicationError> {
        for (field, value, limit) in [
            ("density parity", self.max_density_relative_error, 1.0e-10),
            ("Hsml parity", self.max_hsml_relative_difference, 1.0e-2),
            (
                "neighbor constraint",
                self.max_neighbor_deviation,
                neighbor_tolerance,
            ),
            ("density gradient", self.density_gradient_error, 1.0e-4),
            ("velocity gradient", self.velocity_gradient_error, 1.0e-4),
            ("pressure gradient", self.pressure_gradient_error, 1.0e-4),
            ("face area", self.max_face_area_deviation, 1.0e-6),
        ] {
            if !value.is_finite() || value > limit {
                return Err(ApplicationError::StateMismatch(format!(
                    "{field} exceeded {limit}: {value}"
                )));
            }
        }
        Ok(())
    }

    fn print(self, config_sha256: &str) {
        println!("{{");
        println!("  \"config_sha256\": \"{config_sha256}\",");
        println!("  \"particles\": {},", self.particle_count);
        println!("  \"box_size\": {},", self.box_size);
        println!(
            "  \"max_density_relative_error\": {:.17e},",
            self.max_density_relative_error
        );
        println!(
            "  \"max_hsml_relative_difference\": {:.17e},",
            self.max_hsml_relative_difference
        );
        println!(
            "  \"max_neighbor_deviation\": {:.17e},",
            self.max_neighbor_deviation
        );
        println!(
            "  \"density_gradient_error\": {:.17e},",
            self.density_gradient_error
        );
        println!(
            "  \"velocity_gradient_error\": {:.17e},",
            self.velocity_gradient_error
        );
        println!(
            "  \"pressure_gradient_error\": {:.17e},",
            self.pressure_gradient_error
        );
        println!(
            "  \"max_face_area_deviation\": {:.17e}",
            self.max_face_area_deviation
        );
        println!("}}");
    }
}

fn soundwave_gradient_error(
    positions: &[f64],
    values: &[f64],
    smoothing_lengths: &[f64],
    box_size: f64,
    shoot_tolerance: f64,
    positivity_preserving: bool,
) -> Result<f64, ApplicationError> {
    let gradients = gradients_at_hsml_1d(
        positions,
        values,
        smoothing_lengths,
        box_size,
        shoot_tolerance,
        positivity_preserving,
    )
    .map_err(ApplicationError::Hydro)?;
    Ok(fundamental_mode_gradient_error(
        positions, values, &gradients, box_size,
    ))
}

fn fundamental_mode_gradient_error(
    positions: &[f64],
    values: &[f64],
    gradients: &[GradientEstimate],
    box_size: f64,
) -> f64 {
    let count = u32::try_from(positions.len()).expect("validated particle count fits in u32");
    let count_float = f64::from(count);
    let sine = 2.0
        * positions
            .iter()
            .zip(values)
            .map(|(position, value)| value * (std::f64::consts::TAU * position / box_size).sin())
            .sum::<f64>()
        / count_float;
    let cosine = 2.0
        * positions
            .iter()
            .zip(values)
            .map(|(position, value)| value * (std::f64::consts::TAU * position / box_size).cos())
            .sum::<f64>()
        / count_float;
    let amplitude = sine.hypot(cosine);
    positions
        .iter()
        .zip(gradients)
        .map(|(position, estimate)| {
            let wave_number = std::f64::consts::TAU / box_size;
            let phase = wave_number * position;
            let expected = wave_number * (sine * phase.cos() - cosine * phase.sin());
            (estimate.limited - expected).abs()
        })
        .sum::<f64>()
        / (count_float * (std::f64::consts::TAU / box_size) * amplitude)
}

fn resolve_initial_conditions(configured: &str) -> PathBuf {
    PathBuf::from(format!("{configured}.hdf5"))
}

#[derive(Debug)]
enum ApplicationError {
    Cli(CliError),
    Config(gizmo_config::ReadConfigError),
    Parameters(gizmo_params::ReadParameterError),
    ParameterFile(std::io::Error),
    Input(gizmo_io::InputError),
    Output(gizmo_io::OutputError),
    OutputDirectory(std::io::Error),
    Hydro(gizmo_hydro::HydroError),
    UnsupportedConfig(String),
    UnsupportedParameters(String),
    UnsupportedRestart(RestartFlag),
    MissingDataset(&'static str),
    StateMismatch(String),
}

impl std::fmt::Display for ApplicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cli(error) => error.fmt(formatter),
            Self::Config(error) => error.fmt(formatter),
            Self::Parameters(error) => error.fmt(formatter),
            Self::ParameterFile(error) => {
                write!(formatter, "failed to read parameter file: {error}")
            }
            Self::Input(error) => error.fmt(formatter),
            Self::Output(error) => error.fmt(formatter),
            Self::OutputDirectory(error) => {
                write!(
                    formatter,
                    "failed to create snapshot output directory: {error}"
                )
            }
            Self::Hydro(error) => error.fmt(formatter),
            Self::UnsupportedConfig(error) => {
                write!(formatter, "unsupported initialization config: {error}")
            }
            Self::UnsupportedParameters(error) => {
                write!(formatter, "unsupported runtime parameters: {error}")
            }
            Self::UnsupportedRestart(restart) => write!(
                formatter,
                "restart flag {} is not ported; only restart flag 0 can initialize a simulation",
                *restart as u8
            ),
            Self::MissingDataset(name) => {
                write!(formatter, "initial condition is missing required `{name}`")
            }
            Self::StateMismatch(error) => write!(formatter, "initial state mismatch: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRICT_CONFIG: &str = "\
HYDRO_MESHLESS_FINITE_MASS
BOX_SPATIAL_DIMENSION=1
BOX_PERIODIC
SELFGRAVITY_OFF
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
EOS_GAMMA=(5.0/3.0)
FORCE_EQUAL_TIMESTEPS
DEVELOPER_MODE
";
    const SHOCKTUBE_CONFIG: &str = "\
HYDRO_MESHLESS_FINITE_MASS
BOX_SPATIAL_DIMENSION=1
BOX_PERIODIC
SELFGRAVITY_OFF
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
EOS_GAMMA=(1.4)
FORCE_EQUAL_TIMESTEPS
DEVELOPER_MODE
";
    const INTERACTBLAST_CONFIG: &str =
        include_str!("../../../../validation/oracles/interactblast/legacy-config.sh");
    const SHOCKTUBE_PARAMETERS: &str = "\
InitCondFile shocktube_ics_emass
OutputDir output/
ICFormat 3
SnapFormat 3
TimeBegin 0
TimeMax 5
BoxSize 80
TimeBetSnapshot 0.5
MaxSizeTimestep 0.001
DesNumNgb 4
BufferSize 8
ErrTolIntAccuracy 0.0025
CourantFac 0.05
MaxRMSDisplacementFac 0.125
TimeBetStatistics 0.5
ErrTolForceAcc 0.0025
ErrTolTheta 0.5
MaxNumNgbDeviation 0.05
ResubmitOn 0
ResubmitCommand none
";

    #[test]
    fn hdf5_suffix_is_appended_to_legacy_basename_even_when_it_contains_dots() {
        assert_eq!(
            resolve_initial_conditions("run.v1"),
            PathBuf::from("run.v1.hdf5")
        );
    }

    #[test]
    fn exact_soundwave_config_profile_is_required() {
        let manifest = ConfigManifest::parse(STRICT_CONFIG).unwrap();
        assert_eq!(
            validate_strict_config(&manifest).unwrap(),
            StrictProfile::Soundwave
        );

        for required in ["INPUT_IN_DOUBLEPRECISION", "OUTPUT_IN_DOUBLEPRECISION"] {
            let incomplete = STRICT_CONFIG
                .lines()
                .filter(|line| *line != required)
                .collect::<Vec<_>>()
                .join("\n");
            let manifest = ConfigManifest::parse(&incomplete).unwrap();
            assert!(
                matches!(
                    validate_strict_config(&manifest),
                    Err(ApplicationError::UnsupportedConfig(message))
                        if message.contains(required) && message.contains("missing")
                ),
                "unexpectedly accepted config without {required}"
            );
        }
        let no_equal_steps = STRICT_CONFIG.replace("FORCE_EQUAL_TIMESTEPS\n", "");
        assert!(matches!(
            validate_strict_config(&ConfigManifest::parse(&no_equal_steps).unwrap()),
            Err(ApplicationError::UnsupportedConfig(message))
                if message.contains("FORCE_EQUAL_TIMESTEPS=false")
        ));
    }

    #[test]
    fn required_flags_must_not_have_values() {
        let config =
            STRICT_CONFIG.replace("INPUT_IN_DOUBLEPRECISION\n", "INPUT_IN_DOUBLEPRECISION=1\n");
        let manifest = ConfigManifest::parse(&config).unwrap();
        assert!(matches!(
            validate_strict_config(&manifest),
            Err(ApplicationError::UnsupportedConfig(message))
                if message.contains("INPUT_IN_DOUBLEPRECISION")
                    && message.contains("bare enabled flag")
        ));
    }

    #[test]
    fn exact_equal_mass_shocktube_config_profile_is_accepted() {
        let manifest = ConfigManifest::parse(SHOCKTUBE_CONFIG).unwrap();
        assert_eq!(
            validate_strict_config(&manifest).unwrap(),
            StrictProfile::EqualMassShocktube
        );
        for gamma in ["1.4000000001", "(7.0/5.0)", "1.4"] {
            let config = SHOCKTUBE_CONFIG.replace("EOS_GAMMA=(1.4)", &format!("EOS_GAMMA={gamma}"));
            assert!(matches!(
                validate_strict_config(&ConfigManifest::parse(&config).unwrap()),
                Err(ApplicationError::UnsupportedConfig(message))
                    if message.contains("EOS_GAMMA")
            ));
        }
    }

    #[test]
    fn exact_interacting_blast_config_profile_is_accepted() {
        let manifest = ConfigManifest::parse(INTERACTBLAST_CONFIG).unwrap();
        assert_eq!(
            validate_strict_config(&manifest).unwrap(),
            StrictProfile::InteractingBlast
        );
        for forbidden in ["BOX_PERIODIC", "FORCE_EQUAL_TIMESTEPS"] {
            let config = format!("{INTERACTBLAST_CONFIG}{forbidden}\n");
            assert!(matches!(
                validate_strict_config(&ConfigManifest::parse(&config).unwrap()),
                Err(ApplicationError::UnsupportedConfig(message))
                    if message.contains(forbidden)
            ));
        }
    }

    #[test]
    fn interacting_blast_requires_the_public_fixed_timestep() {
        let path = std::env::temp_dir().join(format!(
            "gizmo-interactblast-params-{}.txt",
            std::process::id()
        ));
        let parameters = include_str!("../../../../validation/oracles/interactblast/legacy.params");
        fs::write(&path, parameters).unwrap();
        let parsed = read_profile_parameters(&path, StrictProfile::InteractingBlast).unwrap();
        assert_eq!(parsed.min_timestep, Some(2.0e-7));
        fs::write(
            &path,
            parameters.replace("MinSizeTimestep                    2e-07\n", ""),
        )
        .unwrap();
        assert!(matches!(
            read_profile_parameters(&path, StrictProfile::InteractingBlast),
            Err(ApplicationError::UnsupportedParameters(message))
                if message.contains("equal explicit")
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn shocktube_parameters_are_strict() {
        let path =
            std::env::temp_dir().join(format!("gizmo-shocktube-params-{}.txt", std::process::id()));
        fs::write(&path, SHOCKTUBE_PARAMETERS).unwrap();
        let parameters = read_profile_parameters(&path, StrictProfile::EqualMassShocktube).unwrap();
        assert_eq!(parameters.box_size.to_bits(), 80.0_f64.to_bits());
        assert_eq!(parameters.time_max.to_bits(), 5.0_f64.to_bits());
        fs::write(
            &path,
            SHOCKTUBE_PARAMETERS.replace("BufferSize 8", "BufferSize 16"),
        )
        .unwrap();
        assert!(matches!(
            read_profile_parameters(&path, StrictProfile::EqualMassShocktube),
            Err(ApplicationError::UnsupportedParameters(message))
                if message.contains("BufferSize") && message.contains("must equal `8`")
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn every_non_initial_restart_mode_is_rejected_explicitly() {
        reject_unsupported_restart(RestartFlag::InitialConditions).unwrap();
        for value in 1..=6 {
            let restart = RestartFlag::try_from(value).unwrap();
            assert!(matches!(
                reject_unsupported_restart(restart),
                Err(ApplicationError::UnsupportedRestart(actual)) if actual == restart
            ));
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn default_output_schedule_matches_long_integer_c_timeline() {
        let tick_duration = 1.5 / LEGACY_TIMEBASE_TICKS as f64;
        assert_eq!(
            legacy_output_tick(0.1, tick_duration),
            76_861_433_640_456_464
        );

        let mut time = 0.0;
        let mut ticks = Vec::new();
        while time <= 1.5 {
            ticks.push(legacy_output_tick(time, tick_duration));
            time += 0.1;
        }
        assert_eq!(ticks.len(), 15);
        assert_eq!(ticks[0], 0);
        assert!(ticks[14] < LEGACY_TIMEBASE_TICKS);
        assert!(time > 1.5);
    }
}
