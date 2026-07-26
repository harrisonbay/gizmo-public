#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gizmo_cli::{CliError, Invocation, RestartFlag, USAGE};
use gizmo_config::ConfigManifest;
use gizmo_hydro::grain::{
    EpsteinDragParameters, GrainGasPoint1d, GrainPoint1d, compute_epstein_drag_batch_1d,
};
use gizmo_hydro::meshless_2d::{Box2d, Vector2, solve_public_c_smoothing_lengths_from_seeds_2d};
use gizmo_hydro::mhd::Vector3;
use gizmo_hydro::mhd_evolution::{
    DivergenceControl1d, MhdMfmRates1d, MhdMfmState1d, advance_mhd_kdk_1d,
    global_mhd_courant_timestep_1d, mhd_mfm_spatial_rates_1d,
};
use gizmo_hydro::mhd_evolution_2d::{
    DivergenceControl2d, MhdMfmRates2d, MhdMfmState2d, PublicMhdDriftState2d,
    begin_public_mhd_initial_hierarchy_2d, mhd_mfm_spatial_rates_2d,
    public_mhd_particle_timestep_bounds_2d, public_mhd_particle_timestep_bounds_from_primitive_2d,
    quantize_public_mhd_initial_timebins_2d,
};
use gizmo_hydro::{
    BoundaryMode1d, GradientEstimate, LEGACY_TIMEBASE_TICKS, MeshlessPoint1d, MfmDriftState1d,
    MfmEvolvingState1d, SynchronizedTimeline1d, begin_mfm_kdk_1d, begin_mfm_reflective_kdk_1d,
    cubic_kernel_1d, density_at_hsml_1d, density_at_hsml_1d_with_boundary, finish_mfm_kdk_1d,
    gradients_at_hsml_1d, inverse_moments_1d, meshless_face_geometry_1d,
    mfm_spatial_rates_1d_with_boundary, periodic_displacement_1d,
    select_public_soundwave_timestep_1d, solve_public_c_initial_smoothing_lengths_1d_with_boundary,
};
use gizmo_io::{
    DustyWaveWriteView, GasWriteView, GrainWriteView, MhdWaveWriteView, SnapshotHeader,
    SoundWaveWriteView, read_dustywave, read_mhd_wave, read_soundwave, write_dustywave,
    write_mhd_wave, write_soundwave,
};
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
    MhdWave,
    BrioWu,
    EqualMassShocktube,
    InteractingBlast,
    Dustywave,
}

impl StrictProfile {
    const fn gamma(self) -> f64 {
        match self {
            Self::Soundwave | Self::MhdWave | Self::Dustywave => 5.0 / 3.0,
            Self::BrioWu => 2.0,
            Self::EqualMassShocktube | Self::InteractingBlast => 1.4,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Soundwave => "soundwave",
            Self::MhdWave => "MHD wave",
            Self::BrioWu => "Brio-Wu",
            Self::EqualMassShocktube => "equal-mass shocktube",
            Self::InteractingBlast => "interacting blastwave",
            Self::Dustywave => "dusty wave",
        }
    }

    const fn boundary(self) -> BoundaryMode1d {
        match self {
            Self::Soundwave
            | Self::MhdWave
            | Self::BrioWu
            | Self::EqualMassShocktube
            | Self::Dustywave => BoundaryMode1d::Periodic,
            Self::InteractingBlast => BoundaryMode1d::Reflective,
        }
    }
}

#[allow(clippy::too_many_lines)]
fn validate_strict_config(manifest: &ConfigManifest) -> Result<StrictProfile, ApplicationError> {
    const REQUIRED_FLAGS: [&str; 4] = [
        "DEVELOPER_MODE",
        "HYDRO_MESHLESS_FINITE_MASS",
        "OUTPUT_IN_DOUBLEPRECISION",
        "SELFGRAVITY_OFF",
    ];
    const ALLOWED: [&str; 15] = [
        "BOX_BND_PARTICLES",
        "BOX_PERIODIC",
        "BOX_REFLECT_X",
        "BOX_SPATIAL_DIMENSION",
        "DEVELOPER_MODE",
        "EOS_ENFORCE_ADIABAT",
        "EOS_GAMMA",
        "FORCE_EQUAL_TIMESTEPS",
        "GRAIN_BACKREACTION",
        "GRAIN_FLUID",
        "HYDRO_MESHLESS_FINITE_MASS",
        "INPUT_IN_DOUBLEPRECISION",
        "MAGNETIC",
        "OUTPUT_IN_DOUBLEPRECISION",
        "SELFGRAVITY_OFF",
    ];
    if manifest
        .get("BOX_SPATIAL_DIMENSION")
        .is_some_and(|option| option.value.as_deref() == Some("2"))
    {
        return validate_briowu_config(manifest);
    }
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
        "GRAIN_BACKREACTION",
        "GRAIN_FLUID",
        "MAGNETIC",
    ] {
        if manifest.get(topology_flag).is_some() {
            require_config_flag(manifest, topology_flag)?;
        }
    }
    if manifest.get("INPUT_IN_DOUBLEPRECISION").is_some() {
        require_config_flag(manifest, "INPUT_IN_DOUBLEPRECISION")?;
    }
    require_config_value(manifest, "BOX_SPATIAL_DIMENSION", "1")?;
    let gamma = manifest
        .get("EOS_GAMMA")
        .and_then(|option| option.value.as_deref());
    let periodic = manifest.get("BOX_PERIODIC").is_some();
    let equal_timesteps = manifest.get("FORCE_EQUAL_TIMESTEPS").is_some();
    let reflective_x = manifest.get("BOX_REFLECT_X").is_some();
    let boundary_particles = manifest.get("BOX_BND_PARTICLES").is_some();
    let grain_fluid = manifest.get("GRAIN_FLUID").is_some();
    let grain_backreaction = manifest.get("GRAIN_BACKREACTION").is_some();
    let enforce_adiabat = manifest
        .get("EOS_ENFORCE_ADIABAT")
        .and_then(|option| option.value.as_deref());
    let input_double = manifest.get("INPUT_IN_DOUBLEPRECISION").is_some();
    let magnetic = manifest.get("MAGNETIC").is_some();
    if !grain_fluid && !input_double {
        return Err(ApplicationError::UnsupportedConfig(
            "required option `INPUT_IN_DOUBLEPRECISION` is missing".to_owned(),
        ));
    }

    match (
        gamma,
        periodic,
        equal_timesteps,
        reflective_x,
        boundary_particles,
        grain_fluid,
        grain_backreaction,
        enforce_adiabat,
        input_double,
        magnetic,
    ) {
        (Some("(5.0/3.0)"), true, true, false, false, false, false, None, true, false) => {
            Ok(StrictProfile::Soundwave)
        }
        (Some("(5.0/3.0)"), true, false, false, false, false, false, None, true, true) => {
            Ok(StrictProfile::MhdWave)
        }
        (Some("(1.4)"), true, true, false, false, false, false, None, true, false) => {
            Ok(StrictProfile::EqualMassShocktube)
        }
        (Some("(1.4)"), false, false, true, true, false, false, None, true, false) => {
            Ok(StrictProfile::InteractingBlast)
        }
        (Some("(5./3.)"), true, false, false, false, true, true, Some("(3./5.)"), false, false) => {
            Ok(StrictProfile::Dustywave)
        }
        _ => Err(ApplicationError::UnsupportedConfig(format!(
            "configuration does not exactly match a ported profile: \
             EOS_GAMMA={gamma:?}, BOX_PERIODIC={periodic}, \
             FORCE_EQUAL_TIMESTEPS={equal_timesteps}, BOX_REFLECT_X={reflective_x}, \
             BOX_BND_PARTICLES={boundary_particles}, GRAIN_FLUID={grain_fluid}, \
             GRAIN_BACKREACTION={grain_backreaction}, \
             EOS_ENFORCE_ADIABAT={enforce_adiabat:?}, \
             INPUT_IN_DOUBLEPRECISION={input_double}, MAGNETIC={magnetic}"
        ))),
    }
}

fn validate_briowu_config(manifest: &ConfigManifest) -> Result<StrictProfile, ApplicationError> {
    const REQUIRED_FLAGS: [&str; 4] = [
        "HYDRO_MESHLESS_FINITE_MASS",
        "BOX_PERIODIC",
        "MAGNETIC",
        "SELFGRAVITY_OFF",
    ];
    const REQUIRED_VALUES: [(&str, &str); 5] = [
        ("BOX_LONG_X", "16"),
        ("BOX_LONG_Y", "1"),
        ("BOX_LONG_Z", "1"),
        ("BOX_SPATIAL_DIMENSION", "2"),
        ("EOS_GAMMA", "(2.0)"),
    ];
    const ALLOWED: [&str; 11] = [
        "HYDRO_MESHLESS_FINITE_MASS",
        "BOX_PERIODIC",
        "BOX_LONG_X",
        "BOX_LONG_Y",
        "BOX_LONG_Z",
        "BOX_SPATIAL_DIMENSION",
        "EOS_GAMMA",
        "MAGNETIC",
        "SELFGRAVITY_OFF",
        "OUTPUT_IN_DOUBLEPRECISION",
        "DEVELOPER_MODE",
    ];
    for option in manifest.iter() {
        if !ALLOWED.contains(&option.name.as_str()) {
            return Err(ApplicationError::UnsupportedConfig(format!(
                "option `{}` is outside the exact public Brio-Wu profile",
                option.name
            )));
        }
    }
    for required in REQUIRED_FLAGS {
        require_config_flag(manifest, required)?;
    }
    if manifest.get("OUTPUT_IN_DOUBLEPRECISION").is_some() {
        require_config_flag(manifest, "OUTPUT_IN_DOUBLEPRECISION")?;
    }
    if manifest.get("DEVELOPER_MODE").is_some() {
        require_config_flag(manifest, "DEVELOPER_MODE")?;
    }
    for (name, value) in REQUIRED_VALUES {
        require_config_value(manifest, name, value)?;
    }
    Ok(StrictProfile::BrioWu)
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

#[allow(clippy::too_many_lines)]
fn initialize_profile(
    parameter_file: &Path,
    profile: StrictProfile,
) -> Result<InitializedProfile, ApplicationError> {
    let parameters = read_profile_parameters(parameter_file, profile)?;
    let fixture_path = resolve_initial_conditions(&parameters.init_cond_file);
    if profile == StrictProfile::BrioWu {
        return initialize_briowu(&fixture_path, parameters).map(InitializedProfile::BrioWu);
    }
    if profile == StrictProfile::MhdWave {
        return initialize_mhd_wave(&fixture_path, parameters).map(InitializedProfile::Mhd);
    }
    if profile == StrictProfile::Dustywave {
        return initialize_dustywave(&fixture_path, parameters, profile)
            .map(InitializedProfile::Hydro);
    }
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
    Ok(InitializedProfile::Hydro(InitializedSoundwave {
        profile,
        parameters,
        particle_ids,
        transverse_vectors,
        state,
        grains: None,
        summary,
    }))
}

fn initialize_briowu(
    fixture_path: &Path,
    parameters: SoundwaveParameters,
) -> Result<InitializedBrioWu, ApplicationError> {
    let snapshot = read_mhd_wave(fixture_path).map_err(ApplicationError::Input)?;
    if snapshot.header.box_size.to_bits() != parameters.box_size.to_bits() {
        return Err(ApplicationError::StateMismatch(format!(
            "parameter BoxSize={} differs from HDF5 BoxSize={}",
            parameters.box_size, snapshot.header.box_size
        )));
    }
    if snapshot.header.double_precision {
        return Err(ApplicationError::StateMismatch(
            "the pinned public Brio-Wu initial condition must use float32 HDF5 fields".to_owned(),
        ));
    }
    let particle_count = snapshot.gas.len();
    Ok(InitializedBrioWu {
        parameters,
        particle_ids: snapshot.gas.ids,
        positions: snapshot.gas.coordinates,
        masses: snapshot.gas.masses,
        velocities: snapshot.gas.velocities,
        specific_internal_energy: snapshot.gas.internal_energy,
        density: snapshot.gas.density,
        smoothing_lengths: snapshot.gas.smoothing_length,
        magnetic_field: snapshot.gas.magnetic_field,
        // Restart flag zero treats these as derived/restart-only fields. In
        // particular, the public IC misspells the GradPhi dataset name; no
        // stored cleaning diagnostic is allowed to seed a fresh run.
        cleaning_phi: vec![0.0; particle_count],
        cleaning_grad_phi: vec![[0.0; 3]; particle_count],
        divergence_of_magnetic_field: vec![0.0; particle_count],
        box_lengths: [
            16.0 * snapshot.header.box_size,
            snapshot.header.box_size,
            snapshot.header.box_size,
        ],
    })
}

fn initialize_mhd_wave(
    fixture_path: &Path,
    parameters: SoundwaveParameters,
) -> Result<InitializedMhdWave, ApplicationError> {
    let snapshot = read_mhd_wave(fixture_path).map_err(ApplicationError::Input)?;
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
    let solved = solve_public_c_initial_smoothing_lengths_1d_with_boundary(
        &positions,
        &snapshot.gas.masses,
        snapshot.header.box_size,
        parameters.desired_num_neighbors,
        parameters.max_neighbor_deviation,
        BoundaryMode1d::Periodic,
    )
    .map_err(ApplicationError::Hydro)?;
    let smoothing_lengths = solved
        .into_iter()
        .map(|particle| particle.smoothing_length)
        .collect();
    let velocities = snapshot
        .gas
        .velocities
        .iter()
        .copied()
        .map(|value| Vector3::new(value[0], value[1], value[2]))
        .collect();
    let magnetic: Vec<Vector3> = snapshot
        .gas
        .magnetic_field
        .iter()
        .copied()
        .map(|value| Vector3::new(value[0], value[1], value[2]))
        .collect();
    // Restart mode zero follows the C initial-condition path: the magnetic
    // field is physical input, while stored cleaning diagnostics are stale
    // derived state and must not seed a new evolution.
    let cleaning_scalar = vec![0.0; positions.len()];
    let state = MhdMfmState1d::from_primitive(
        positions,
        snapshot.gas.masses,
        velocities,
        snapshot.gas.internal_energy,
        smoothing_lengths,
        &magnetic,
        &cleaning_scalar,
        snapshot.header.box_size,
        StrictProfile::MhdWave.gamma(),
    )
    .map_err(ApplicationError::MhdEvolution)?;
    let controls = DivergenceControl1d {
        hyperbolic_sigma: parameters
            .divb_cleaning_hyperbolic_sigma
            .expect("strict MHD parameters require hyperbolic cleaning"),
        parabolic_sigma: parameters
            .divb_cleaning_parabolic_sigma
            .expect("strict MHD parameters require parabolic cleaning"),
        ..DivergenceControl1d::default()
    };
    Ok(InitializedMhdWave {
        parameters,
        particle_ids: snapshot.gas.ids,
        transverse_positions: snapshot
            .gas
            .coordinates
            .into_iter()
            .map(|position| [position[1], position[2]])
            .collect(),
        state,
        controls,
    })
}

#[allow(clippy::too_many_lines)]
fn initialize_dustywave(
    fixture_path: &Path,
    parameters: SoundwaveParameters,
    profile: StrictProfile,
) -> Result<InitializedSoundwave, ApplicationError> {
    let snapshot = read_dustywave(fixture_path).map_err(ApplicationError::Input)?;
    if snapshot.header.box_size.to_bits() != parameters.box_size.to_bits() {
        return Err(ApplicationError::StateMismatch(format!(
            "parameter BoxSize={} differs from HDF5 BoxSize={}",
            parameters.box_size, snapshot.header.box_size
        )));
    }
    let gas_positions: Vec<f64> = snapshot
        .gas
        .coordinates
        .iter()
        .map(|coordinate| coordinate[0])
        .collect();
    let solved_gas = solve_public_c_initial_smoothing_lengths_1d_with_boundary(
        &gas_positions,
        &snapshot.gas.masses,
        snapshot.header.box_size,
        parameters.desired_num_neighbors,
        parameters.max_neighbor_deviation,
        BoundaryMode1d::Periodic,
    )
    .map_err(ApplicationError::Hydro)?;
    let gas_smoothing_lengths: Vec<f64> = solved_gas
        .iter()
        .map(|particle| particle.smoothing_length)
        .collect();
    let gas_density: Vec<f64> = solved_gas
        .iter()
        .map(|particle| particle.estimate.density)
        .collect();
    let gas_internal_energy: Vec<f64> = gas_density
        .iter()
        .map(|density| 0.9 * density.powf(2.0 / 3.0))
        .collect();
    let grain_positions: Vec<f64> = snapshot
        .grains
        .coordinates
        .iter()
        .map(|coordinate| coordinate[0])
        .collect();
    let grain_smoothing_lengths = grain_smoothing_lengths_1d(
        &grain_positions,
        &gas_positions,
        snapshot.header.box_size,
        parameters.desired_num_neighbors,
        parameters.max_neighbor_deviation,
    )?;
    let gas_transverse_vectors = snapshot
        .gas
        .coordinates
        .into_iter()
        .zip(&snapshot.gas.velocities)
        .map(|(position, velocity)| TransverseVectorShell {
            position: [position[1], position[2]],
            velocity: [velocity[1], velocity[2]],
        })
        .collect();
    let grain_count = snapshot.grains.len();
    let grain_transverse_vectors = snapshot
        .grains
        .coordinates
        .into_iter()
        .zip(&snapshot.grains.velocities)
        .map(|(position, velocity)| TransverseVectorShell {
            position: [position[1], position[2]],
            velocity: [velocity[1], velocity[2]],
        })
        .collect();
    let configured_grain_size = parameters
        .grain_size_max
        .expect("strict dusty-wave parameters");
    Ok(InitializedSoundwave {
        profile,
        parameters,
        particle_ids: snapshot.gas.ids,
        transverse_vectors: gas_transverse_vectors,
        state: MfmEvolvingState1d {
            positions: gas_positions,
            masses: snapshot.gas.masses,
            velocities: snapshot
                .gas
                .velocities
                .iter()
                .map(|velocity| velocity[0])
                .collect(),
            specific_internal_energy: gas_internal_energy,
            smoothing_lengths: gas_smoothing_lengths,
            box_size: snapshot.header.box_size,
            gamma: profile.gamma(),
        },
        grains: Some(GrainRuntimeState1d {
            particle_ids: snapshot.grains.ids,
            transverse_vectors: grain_transverse_vectors,
            positions: grain_positions,
            masses: snapshot.grains.masses,
            velocities: snapshot
                .grains
                .velocities
                .iter()
                .map(|velocity| velocity[0])
                .collect(),
            smoothing_lengths: grain_smoothing_lengths,
            grain_sizes: vec![configured_grain_size; grain_count],
        }),
        summary: None,
    })
}

fn grain_smoothing_lengths_1d(
    grain_positions: &[f64],
    gas_positions: &[f64],
    box_size: f64,
    desired_neighbors: f64,
    tolerance: f64,
) -> Result<Vec<f64>, ApplicationError> {
    let mut output = Vec::with_capacity(grain_positions.len());
    for &grain_position in grain_positions {
        let mut lower = box_size * 2.0_f64.powi(-40);
        let mut upper = box_size;
        let mut best = upper;
        let mut best_error = f64::INFINITY;
        for _ in 0..160 {
            let smoothing_length = 0.5 * (lower + upper);
            let mut kernel_sum = 0.0;
            for &gas_position in gas_positions {
                let radius = periodic_displacement_1d(grain_position, gas_position, box_size)
                    .map_err(ApplicationError::Hydro)?
                    .abs();
                if radius < smoothing_length {
                    kernel_sum += cubic_kernel_1d(radius, smoothing_length)
                        .map_err(ApplicationError::Hydro)?
                        .weight;
                }
            }
            let effective_neighbors = 2.0 * smoothing_length * kernel_sum;
            let error = (effective_neighbors - desired_neighbors).abs();
            if error < best_error {
                best = smoothing_length;
                best_error = error;
            }
            if error <= tolerance {
                break;
            }
            if effective_neighbors < desired_neighbors {
                lower = smoothing_length;
            } else {
                upper = smoothing_length;
            }
        }
        if best_error > tolerance {
            return Err(ApplicationError::StateMismatch(format!(
                "grain smoothing-length solve missed neighbor target: error={best_error}, \
                 tolerance={tolerance}"
            )));
        }
        output.push(best);
    }
    Ok(output)
}

#[allow(clippy::too_many_lines)]
fn read_profile_parameters(
    parameter_file: &Path,
    profile: StrictProfile,
) -> Result<SoundwaveParameters, ApplicationError> {
    if profile == StrictProfile::Soundwave {
        return SoundwaveParameters::from_path(parameter_file)
            .map_err(ApplicationError::Parameters);
    }
    let input = fs::read_to_string(parameter_file).map_err(ApplicationError::ParameterFile)?;
    if profile == StrictProfile::BrioWu {
        return read_briowu_parameters(&input);
    }
    if profile == StrictProfile::MhdWave {
        return read_mhd_wave_parameters(&input);
    }
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
    if profile == StrictProfile::Dustywave {
        let required_scalars: [(&str, f64, f64); 11] = [
            ("TimeMax", parameters.time_max, 2.5),
            ("BoxSize", parameters.box_size, 1.0),
            ("TimeBetSnapshot", parameters.time_between_snapshots, 0.01),
            ("MaxSizeTimestep", parameters.max_timestep, 0.0001),
            ("DesNumNgb", parameters.desired_num_neighbors, 4.0),
            ("ErrTolIntAccuracy", parameters.integration_accuracy, 0.01),
            ("CourantFac", parameters.courant_factor, 0.1),
            (
                "MaxRMSDisplacementFac",
                parameters.max_rms_displacement_factor,
                0.125,
            ),
            ("ErrTolForceAcc", parameters.force_accuracy, 0.0025),
            (
                "MaxNumNgbDeviation",
                parameters.max_neighbor_deviation,
                0.05,
            ),
            (
                "Softening_Type3",
                parameters.type3_softening.unwrap_or(f64::NAN),
                0.001,
            ),
        ];
        for (field, actual, expected) in required_scalars {
            if actual.to_bits() != expected.to_bits() {
                return Err(ApplicationError::UnsupportedParameters(format!(
                    "dusty wave requires `{field} {expected}`, found `{actual}`"
                )));
            }
        }
        let required_optional_scalars: [(&str, Option<f64>, f64); 4] = [
            (
                "Grain_Internal_Density",
                parameters.grain_internal_density,
                1.0,
            ),
            ("Grain_Size_Min", parameters.grain_size_min, 1.23608),
            ("Grain_Size_Max", parameters.grain_size_max, 1.23608),
            (
                "Grain_Size_Spectrum_Powerlaw",
                parameters.grain_size_spectrum_powerlaw,
                0.5,
            ),
        ];
        for (field, actual, expected) in required_optional_scalars {
            if actual.is_none_or(|value| value.to_bits() != expected.to_bits()) {
                return Err(ApplicationError::UnsupportedParameters(format!(
                    "dusty wave requires `{field} {expected}`, found {actual:?}"
                )));
            }
        }
        if !matches!(
            parameters.init_cond_file.as_str(),
            "dustywave_ics" | "dustybox_ics"
        ) {
            return Err(ApplicationError::UnsupportedParameters(format!(
                "dusty gas-grain profile requires `InitCondFile dustywave_ics` or \
                 `InitCondFile dustybox_ics`, found `{}`",
                parameters.init_cond_file
            )));
        }
    }
    Ok(parameters)
}

#[allow(clippy::too_many_lines)]
fn read_briowu_parameters(input: &str) -> Result<SoundwaveParameters, ApplicationError> {
    const PUBLIC_TAGS: [&str; 7] = [
        "InitCondFile",
        "OutputDir",
        "TimeMax",
        "BoxSize",
        "TimeBetSnapshot",
        "MaxSizeTimestep",
        "DesNumNgb",
    ];
    const FRONTIER_EXTRA_TAGS: [&str; 12] = [
        "MaxNumNgbDeviation",
        "MaxMemSize",
        "ErrTolIntAccuracy",
        "CourantFac",
        "MaxRMSDisplacementFac",
        "DivBcleaningParabolicSigma",
        "DivBcleaningHyperbolicSigma",
        "ResubmitOn",
        "ResubmitCommand",
        "ErrTolTheta",
        "ErrTolForceAcc",
        "TimeBetStatistics",
    ];
    const LEGACY_SOFTENINGS: [&str; 6] = [
        "SofteningGas",
        "SofteningHalo",
        "SofteningDisk",
        "SofteningBulge",
        "SofteningStars",
        "SofteningBndry",
    ];
    let mut retained = Vec::new();
    let mut actual_tags = BTreeSet::new();
    let mut softening_tags = BTreeSet::new();
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
        if !actual_tags.insert(tag) {
            return Err(ApplicationError::UnsupportedParameters(format!(
                "line {}: duplicate Brio-Wu parameter `{tag}`",
                line_index + 1
            )));
        }
        if LEGACY_SOFTENINGS.contains(&tag) {
            let value = tokens.next();
            if value != Some("0.001") || tokens.next().is_some() {
                return Err(ApplicationError::UnsupportedParameters(format!(
                    "line {}: `{tag}` must equal `0.001` in the corrected-C Brio-Wu profile",
                    line_index + 1
                )));
            }
            softening_tags.insert(tag);
        } else {
            retained.push(raw_line);
        }
    }
    let public: BTreeSet<&str> = PUBLIC_TAGS.into_iter().collect();
    let mut frontier = public.clone();
    frontier.extend(FRONTIER_EXTRA_TAGS);
    let actual_without_softening: BTreeSet<&str> =
        actual_tags.difference(&softening_tags).copied().collect();
    let valid_public = actual_without_softening == public && softening_tags.is_empty();
    let valid_frontier = actual_without_softening == frontier
        && (softening_tags.is_empty() || softening_tags.len() == LEGACY_SOFTENINGS.len());
    if !valid_public && !valid_frontier {
        return Err(ApplicationError::UnsupportedParameters(format!(
            "Brio-Wu requires either the exact 7-tag public profile or the exact \
             frontier profile (optionally with all six legacy softenings); found \
             tags={actual_tags:?}"
        )));
    }
    let parameters = SoundwaveParameters::parse(&retained.join("\n"))
        .map_err(|error| ApplicationError::UnsupportedParameters(error.to_string()))?;
    let required_scalars: [(&str, f64, f64); 5] = [
        ("TimeMax", parameters.time_max, 0.2),
        ("BoxSize", parameters.box_size, 0.25),
        ("TimeBetSnapshot", parameters.time_between_snapshots, 0.1),
        ("MaxSizeTimestep", parameters.max_timestep, 0.04),
        ("DesNumNgb", parameters.desired_num_neighbors, 20.0),
    ];
    for (field, actual, expected) in required_scalars {
        if actual.to_bits() != expected.to_bits() {
            return Err(ApplicationError::UnsupportedParameters(format!(
                "Brio-Wu requires `{field} {expected}`, found `{actual}`"
            )));
        }
    }
    let expected_neighbor_deviation: f64 = if valid_frontier { 0.1 } else { 0.05 };
    if parameters.max_neighbor_deviation.to_bits() != expected_neighbor_deviation.to_bits() {
        return Err(ApplicationError::UnsupportedParameters(format!(
            "Brio-Wu requires `MaxNumNgbDeviation {expected_neighbor_deviation}`, found `{}`",
            parameters.max_neighbor_deviation
        )));
    }
    if parameters.init_cond_file != "briowu_ics"
        || parameters.output_dir != "output"
        || parameters.min_timestep.is_some()
        || parameters.grain_internal_density.is_some()
        || parameters.grain_size_min.is_some()
        || parameters.grain_size_max.is_some()
        || parameters.grain_size_spectrum_powerlaw.is_some()
        || parameters.type3_softening.is_some()
    {
        return Err(ApplicationError::UnsupportedParameters(
            "Brio-Wu string, restart, or non-MHD parameters differ from the pinned profiles"
                .to_owned(),
        ));
    }
    if valid_frontier
        && (parameters.max_memory_mb != Some(2000)
            || parameters.integration_accuracy.to_bits() != 0.01_f64.to_bits()
            || parameters.courant_factor.to_bits() != 0.2_f64.to_bits()
            || parameters.max_rms_displacement_factor.to_bits() != 0.1_f64.to_bits()
            || parameters.divb_cleaning_parabolic_sigma != Some(1.0)
            || parameters.divb_cleaning_hyperbolic_sigma != Some(1.0)
            || parameters.resubmit
            || parameters.resubmit_command != "none"
            || parameters.tree_opening_angle.to_bits() != 0.7_f64.to_bits()
            || parameters.force_accuracy.to_bits() != 0.001_f64.to_bits()
            || parameters.time_between_statistics.to_bits() != 0.5_f64.to_bits())
    {
        return Err(ApplicationError::UnsupportedParameters(
            "Brio-Wu frontier controls differ from corrected-c.params".to_owned(),
        ));
    }
    Ok(parameters)
}

#[allow(clippy::too_many_lines)]
fn read_mhd_wave_parameters(input: &str) -> Result<SoundwaveParameters, ApplicationError> {
    const REQUIRED_TAGS: [&str; 19] = [
        "InitCondFile",
        "OutputDir",
        "TimeMax",
        "BoxSize",
        "TimeBetSnapshot",
        "MaxSizeTimestep",
        "DesNumNgb",
        "MaxMemSize",
        "ErrTolIntAccuracy",
        "CourantFac",
        "MaxRMSDisplacementFac",
        "ErrTolForceAcc",
        "TimeBetStatistics",
        "MaxNumNgbDeviation",
        "ErrTolTheta",
        "DivBcleaningParabolicSigma",
        "DivBcleaningHyperbolicSigma",
        "ResubmitOn",
        "ResubmitCommand",
    ];
    let actual_tags: BTreeSet<&str> = input
        .lines()
        .filter_map(|line| {
            let definition = line
                .split_once('%')
                .map_or(line, |(before, _)| before)
                .trim();
            (!definition.is_empty()).then(|| definition.split_whitespace().next().unwrap_or(""))
        })
        .collect();
    let required_tags: BTreeSet<&str> = REQUIRED_TAGS.iter().copied().collect();
    if actual_tags != required_tags {
        let missing: Vec<_> = required_tags.difference(&actual_tags).copied().collect();
        let unexpected: Vec<_> = actual_tags.difference(&required_tags).copied().collect();
        return Err(ApplicationError::UnsupportedParameters(format!(
            "MHD wave requires the exact frontier parameter vocabulary; \
             missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    let parameters = SoundwaveParameters::parse(input)
        .map_err(|error| ApplicationError::UnsupportedParameters(error.to_string()))?;
    let required_scalars: [(&str, f64, f64); 12] = [
        ("TimeMax", parameters.time_max, 0.5),
        ("BoxSize", parameters.box_size, 1.0),
        ("TimeBetSnapshot", parameters.time_between_snapshots, 0.05),
        ("MaxSizeTimestep", parameters.max_timestep, 0.1),
        ("DesNumNgb", parameters.desired_num_neighbors, 4.0),
        ("ErrTolIntAccuracy", parameters.integration_accuracy, 0.01),
        ("CourantFac", parameters.courant_factor, 0.2),
        (
            "MaxRMSDisplacementFac",
            parameters.max_rms_displacement_factor,
            0.1,
        ),
        ("ErrTolForceAcc", parameters.force_accuracy, 0.001),
        ("TimeBetStatistics", parameters.time_between_statistics, 0.5),
        (
            "MaxNumNgbDeviation",
            parameters.max_neighbor_deviation,
            1.0e-6,
        ),
        ("ErrTolTheta", parameters.tree_opening_angle, 0.7),
    ];
    for (field, actual, expected) in required_scalars {
        if actual.to_bits() != expected.to_bits() {
            return Err(ApplicationError::UnsupportedParameters(format!(
                "MHD wave requires `{field} {expected}`, found `{actual}`"
            )));
        }
    }
    let required_cleaning: [(&str, Option<f64>, f64); 2] = [
        (
            "DivBcleaningParabolicSigma",
            parameters.divb_cleaning_parabolic_sigma,
            0.2,
        ),
        (
            "DivBcleaningHyperbolicSigma",
            parameters.divb_cleaning_hyperbolic_sigma,
            1.0,
        ),
    ];
    for (field, actual, expected) in required_cleaning {
        if actual.is_none_or(|value| value.to_bits() != expected.to_bits()) {
            return Err(ApplicationError::UnsupportedParameters(format!(
                "MHD wave requires `{field} {expected}`, found {actual:?}"
            )));
        }
    }
    if parameters.init_cond_file != "mhd_wave_ics"
        || parameters.output_dir != "output"
        || parameters.max_memory_mb != Some(1000)
        || parameters.min_timestep.is_some()
        || parameters.resubmit
        || parameters.resubmit_command != "none"
        || parameters.grain_internal_density.is_some()
        || parameters.grain_size_min.is_some()
        || parameters.grain_size_max.is_some()
        || parameters.grain_size_spectrum_powerlaw.is_some()
        || parameters.type3_softening.is_some()
    {
        return Err(ApplicationError::UnsupportedParameters(
            "MHD wave string, memory, restart, or non-MHD parameters differ from frontier.params"
                .to_owned(),
        ));
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
struct GrainRuntimeState1d {
    particle_ids: Vec<u64>,
    transverse_vectors: Vec<TransverseVectorShell>,
    positions: Vec<f64>,
    masses: Vec<f64>,
    velocities: Vec<f64>,
    smoothing_lengths: Vec<f64>,
    grain_sizes: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct InitializedSoundwave {
    profile: StrictProfile,
    parameters: SoundwaveParameters,
    particle_ids: Vec<u64>,
    transverse_vectors: Vec<TransverseVectorShell>,
    state: MfmEvolvingState1d,
    grains: Option<GrainRuntimeState1d>,
    summary: Option<InitializationSummary>,
}

#[derive(Clone, Debug, PartialEq)]
struct InitializedMhdWave {
    parameters: SoundwaveParameters,
    particle_ids: Vec<u64>,
    transverse_positions: Vec<[f64; 2]>,
    state: MhdMfmState1d,
    controls: DivergenceControl1d,
}

#[derive(Clone, Debug, PartialEq)]
struct InitializedBrioWu {
    parameters: SoundwaveParameters,
    particle_ids: Vec<u64>,
    positions: Vec<[f64; 3]>,
    masses: Vec<f64>,
    velocities: Vec<[f64; 3]>,
    specific_internal_energy: Vec<f64>,
    density: Vec<f64>,
    smoothing_lengths: Vec<f64>,
    magnetic_field: Vec<[f64; 3]>,
    cleaning_phi: Vec<f64>,
    cleaning_grad_phi: Vec<[f64; 3]>,
    divergence_of_magnetic_field: Vec<f64>,
    box_lengths: [f64; 3],
}

#[derive(Clone, Debug, PartialEq)]
enum InitializedProfile {
    Hydro(InitializedSoundwave),
    Mhd(InitializedMhdWave),
    BrioWu(InitializedBrioWu),
}

impl InitializedProfile {
    fn validate_owned_state(&self) -> Result<(), ApplicationError> {
        match self {
            Self::Hydro(initialized) => initialized.validate_owned_state(),
            Self::Mhd(initialized) => initialized.validate_owned_state(),
            Self::BrioWu(initialized) => initialized.validate_owned_state(),
        }
    }

    fn print_initialization(&self, config_sha256: &str) {
        match self {
            Self::Hydro(initialized) => initialized.print_initialization(config_sha256),
            Self::Mhd(initialized) => initialized.print_initialization(config_sha256),
            Self::BrioWu(initialized) => initialized.print_initialization(config_sha256),
        }
    }
}

impl InitializedBrioWu {
    #[allow(clippy::too_many_lines)]
    fn validate_owned_state(&self) -> Result<(), ApplicationError> {
        let particle_count = self.positions.len();
        for (field, actual) in [
            ("ParticleIDs", self.particle_ids.len()),
            ("masses", self.masses.len()),
            ("velocities", self.velocities.len()),
            (
                "specific internal energy",
                self.specific_internal_energy.len(),
            ),
            ("density", self.density.len()),
            ("smoothing lengths", self.smoothing_lengths.len()),
            ("magnetic field", self.magnetic_field.len()),
            ("cleaning phi", self.cleaning_phi.len()),
            ("cleaning grad phi", self.cleaning_grad_phi.len()),
            (
                "divergence of magnetic field",
                self.divergence_of_magnetic_field.len(),
            ),
        ] {
            if actual != particle_count {
                return Err(ApplicationError::StateMismatch(format!(
                    "{field} has {actual} entries, expected {particle_count}"
                )));
            }
        }
        if particle_count != 50_176 {
            return Err(ApplicationError::StateMismatch(format!(
                "Brio-Wu fixture has {particle_count} particles, expected 50176"
            )));
        }
        if self.particle_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ApplicationError::StateMismatch(
                "Brio-Wu ParticleIDs are not strictly sorted".to_owned(),
            ));
        }
        if self.box_lengths.map(f64::to_bits) != [4.0_f64, 0.25_f64, 0.25_f64].map(f64::to_bits)
            || self.parameters.box_size.to_bits() != 0.25_f64.to_bits()
        {
            return Err(ApplicationError::StateMismatch(
                "Brio-Wu rectangular periodic box must be exactly 4 x 0.25 x 0.25".to_owned(),
            ));
        }
        if self.positions.iter().any(|position| {
            !position.iter().all(|value| value.is_finite())
                || !(0.0..self.box_lengths[0]).contains(&position[0])
                || !(0.0..self.box_lengths[1]).contains(&position[1])
                || position[2].to_bits() != 0.0_f64.to_bits()
        }) {
            return Err(ApplicationError::StateMismatch(
                "Brio-Wu coordinates do not lie in the exact two-dimensional rectangular box"
                    .to_owned(),
            ));
        }
        if self
            .masses
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || self
                .specific_internal_energy
                .iter()
                .chain(&self.density)
                .chain(&self.smoothing_lengths)
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self
                .velocities
                .iter()
                .chain(&self.magnetic_field)
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(ApplicationError::StateMismatch(
                "Brio-Wu fixture contains non-finite or non-positive physical fields".to_owned(),
            ));
        }
        let mut left_count = 0_usize;
        let mut right_count = 0_usize;
        for index in 0..particle_count {
            let left = self.positions[index][0] < 2.0;
            left_count += usize::from(left);
            right_count += usize::from(!left);
            let expected_density: f64 = if left { 1.0 } else { 0.125 };
            let expected_by: f64 = if left { 1.0 } else { -1.0 };
            let field = self.magnetic_field[index];
            if self.density[index].to_bits() != expected_density.to_bits()
                || field[0].to_bits() != 0.75_f64.to_bits()
                || field[1].to_bits() != expected_by.to_bits()
                || field[2].to_bits() != 0.0_f64.to_bits()
            {
                return Err(ApplicationError::StateMismatch(format!(
                    "Brio-Wu particle {} does not match its x-selected density/magnetic state",
                    self.particle_ids[index]
                )));
            }
        }
        if (left_count, right_count) != (25_088, 25_088) {
            return Err(ApplicationError::StateMismatch(format!(
                "Brio-Wu state partition is {left_count}/{right_count}, expected 25088/25088"
            )));
        }
        if self
            .cleaning_phi
            .iter()
            .chain(&self.divergence_of_magnetic_field)
            .any(|value| value.to_bits() != 0.0_f64.to_bits())
            || self
                .cleaning_grad_phi
                .iter()
                .flatten()
                .any(|value| value.to_bits() != 0.0_f64.to_bits())
        {
            return Err(ApplicationError::StateMismatch(
                "Brio-Wu restart-only cleaning diagnostics were not reset".to_owned(),
            ));
        }
        Ok(())
    }

    fn print_initialization(&self, config_sha256: &str) {
        let left = self
            .positions
            .iter()
            .filter(|position| position[0] < 2.0)
            .count();
        println!("{{");
        println!("  \"config_sha256\": \"{config_sha256}\",");
        println!("  \"profile\": \"Brio-Wu\",");
        println!("  \"particles\": {},", self.positions.len());
        println!("  \"box_lengths\": [4, 0.25, 0.25],");
        println!("  \"gamma\": 2,");
        println!("  \"left_particles\": {left},");
        println!("  \"cleaning_diagnostics_initialized_to_zero\": true");
        println!("}}");
    }
}

impl InitializedMhdWave {
    fn validate_owned_state(&self) -> Result<(), ApplicationError> {
        let particle_count = self.state.positions.len();
        for (field, actual) in [
            ("ParticleIDs", self.particle_ids.len()),
            ("transverse positions", self.transverse_positions.len()),
            ("masses", self.state.masses.len()),
            ("velocities", self.state.velocities.len()),
            (
                "specific internal energy",
                self.state.specific_internal_energy.len(),
            ),
            ("smoothing lengths", self.state.smoothing_lengths.len()),
            ("magnetic volume", self.state.magnetic_volume.len()),
            ("cleaning mass", self.state.cleaning_mass.len()),
        ] {
            if actual != particle_count {
                return Err(ApplicationError::StateMismatch(format!(
                    "{field} has {actual} entries, expected {particle_count}"
                )));
            }
        }
        if self.particle_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ApplicationError::StateMismatch(
                "MHD ParticleIDs are not strictly sorted".to_owned(),
            ));
        }
        if self.parameters.box_size.to_bits() != self.state.box_size.to_bits()
            || self.state.gamma.to_bits() != StrictProfile::MhdWave.gamma().to_bits()
        {
            return Err(ApplicationError::StateMismatch(
                "MHD runtime bundle has inconsistent box size or EOS gamma".to_owned(),
            ));
        }
        if self
            .transverse_positions
            .iter()
            .flatten()
            .any(|component| !component.is_finite())
        {
            return Err(ApplicationError::StateMismatch(
                "MHD runtime bundle has non-finite transverse coordinates".to_owned(),
            ));
        }
        self.state
            .primitive_columns()
            .map_err(ApplicationError::MhdEvolution)?;
        Ok(())
    }

    fn print_initialization(&self, config_sha256: &str) {
        println!("{{");
        println!("  \"config_sha256\": \"{config_sha256}\",");
        println!("  \"profile\": \"MHD wave\",");
        println!("  \"particles\": {},", self.state.positions.len());
        println!("  \"box_size\": {},", self.state.box_size);
        println!("  \"gamma\": {},", self.state.gamma);
        println!("  \"cleaning_phi_initialized_to_zero\": true");
        println!("}}");
    }
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
        if (self.profile == StrictProfile::Dustywave) != self.grains.is_some() {
            return Err(ApplicationError::StateMismatch(
                "grain state must exist exactly for the dusty-wave profile".to_owned(),
            ));
        }
        if let Some(grains) = &self.grains {
            let grain_count = grains.positions.len();
            for (field, actual) in [
                ("grain ParticleIDs", grains.particle_ids.len()),
                ("grain transverse vectors", grains.transverse_vectors.len()),
                ("grain masses", grains.masses.len()),
                ("grain velocities", grains.velocities.len()),
                ("grain smoothing lengths", grains.smoothing_lengths.len()),
                ("grain sizes", grains.grain_sizes.len()),
            ] {
                if actual != grain_count {
                    return Err(ApplicationError::StateMismatch(format!(
                        "{field} has {actual} entries, expected {grain_count}"
                    )));
                }
            }
            if grains
                .particle_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
                || self
                    .particle_ids
                    .iter()
                    .any(|id| grains.particle_ids.binary_search(id).is_ok())
            {
                return Err(ApplicationError::StateMismatch(
                    "gas and grain ParticleIDs must be globally unique and sorted".to_owned(),
                ));
            }
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
        if let Some(grains) = &self.grains {
            println!("  \"grains\": {},", grains.positions.len());
        }
        println!("  \"box_size\": {},", self.state.box_size);
        println!("  \"gamma\": {}", self.state.gamma);
        println!("}}");
    }
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn evolve_profile(initialized: InitializedProfile) -> Result<(), ApplicationError> {
    match initialized {
        InitializedProfile::Hydro(initialized) => evolve_hydro_profile(initialized),
        InitializedProfile::Mhd(initialized) => evolve_mhd_wave(initialized),
        InitializedProfile::BrioWu(initialized) => evolve_briowu(&initialized),
    }
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn evolve_hydro_profile(mut initialized: InitializedSoundwave) -> Result<(), ApplicationError> {
    if initialized.profile == StrictProfile::Dustywave {
        return evolve_dustywave(initialized);
    }
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
    let mut next_output_index = 0_u64;
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

            next_output_index = next_output_index.checked_add(1).ok_or_else(|| {
                ApplicationError::StateMismatch("output schedule index overflow".to_owned())
            })?;
            next_output_time = regular_output_time_with_terminal_snap(
                next_output_time + initialized.parameters.time_between_snapshots,
                0.0,
                initialized.parameters.time_between_snapshots,
                next_output_index,
                initialized.parameters.time_max,
            );
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

fn evolve_mhd_wave(mut initialized: InitializedMhdWave) -> Result<(), ApplicationError> {
    const OUTPUT_COUNT: u32 = 10;
    let output_dir = PathBuf::from(&initialized.parameters.output_dir);
    fs::create_dir_all(&output_dir).map_err(ApplicationError::OutputDirectory)?;
    let mut rates = mhd_mfm_spatial_rates_1d(&initialized.state, initialized.controls)
        .map_err(ApplicationError::MhdEvolution)?;
    let mut time = 0.0_f64;
    let mut snapshot_number = 0_u32;
    let mut step_count = 0_u64;
    write_mhd_snapshot(
        &initialized,
        &rates,
        output_dir.join(format!("snapshot_{snapshot_number:03}.hdf5")),
        time,
    )?;
    snapshot_number += 1;

    for output_index in 1..=OUTPUT_COUNT {
        let output_time = if output_index == OUTPUT_COUNT {
            initialized.parameters.time_max
        } else {
            f64::from(output_index) * initialized.parameters.time_between_snapshots
        };
        while time < output_time {
            let cfl = global_mhd_courant_timestep_1d(
                &initialized.state,
                &rates,
                initialized.parameters.courant_factor,
            )
            .map_err(ApplicationError::MhdEvolution)?;
            // Every particle advances on the same global step. Capping at the
            // next output boundary makes each of the eleven public snapshots a
            // completed KDK state, not a wave-specific interpolation.
            let timestep = cfl
                .min(initialized.parameters.max_timestep)
                .min(output_time - time);
            let result = advance_mhd_kdk_1d(
                &initialized.state,
                &rates,
                timestep,
                initialized.parameters.desired_num_neighbors,
                initialized.parameters.max_neighbor_deviation,
                0.0,
                initialized.controls,
            )
            .map_err(ApplicationError::MhdEvolution)?;
            initialized.state = result.state;
            rates = result.rates;
            time += timestep;
            let tolerance = 64.0 * f64::EPSILON * output_time.abs().max(1.0);
            if (time - output_time).abs() <= tolerance {
                time = output_time;
            }
            step_count = step_count.checked_add(1).ok_or_else(|| {
                ApplicationError::StateMismatch("MHD step count overflow".to_owned())
            })?;
        }
        write_mhd_snapshot(
            &initialized,
            &rates,
            output_dir.join(format!("snapshot_{snapshot_number:03}.hdf5")),
            output_time,
        )?;
        snapshot_number += 1;
    }
    eprintln!(
        "completed {step_count} synchronized MHD KDK steps to t={time:.17e}; \
         wrote {snapshot_number} snapshots"
    );
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn evolve_briowu(initialized: &InitializedBrioWu) -> Result<(), ApplicationError> {
    let output_dir = PathBuf::from(&initialized.parameters.output_dir);
    fs::create_dir_all(&output_dir).map_err(ApplicationError::OutputDirectory)?;
    let positions: Vec<_> = initialized
        .positions
        .iter()
        .map(|position| Vector2::new(position[0], position[1]))
        .collect();
    let domain = Box2d::new(initialized.box_lengths[0], initialized.box_lengths[1])
        .map_err(ApplicationError::Geometry2d)?;
    let mut smoothing_lengths = initialized.smoothing_lengths.clone();
    // Restart flag zero reaches density three times before the public C code
    // writes snapshot 000. Repeating independently bracketed passes preserves
    // that initialization schedule.
    for _ in 0..3 {
        smoothing_lengths = solve_public_c_smoothing_lengths_from_seeds_2d(
            &positions,
            &initialized.masses,
            &smoothing_lengths,
            domain,
            initialized.parameters.desired_num_neighbors,
            initialized.parameters.max_neighbor_deviation,
        )
        .map_err(ApplicationError::Geometry2d)?
        .into_iter()
        .map(|particle| particle.smoothing_length)
        .collect();
    }
    let velocities: Vec<_> = initialized
        .velocities
        .iter()
        .map(|value| Vector3::new(value[0], value[1], value[2]))
        .collect();
    let magnetic: Vec<_> = initialized
        .magnetic_field
        .iter()
        .map(|value| Vector3::new(value[0], value[1], value[2]))
        .collect();
    let controls = DivergenceControl2d {
        hyperbolic_sigma: initialized
            .parameters
            .divb_cleaning_hyperbolic_sigma
            .unwrap_or(1.0),
        parabolic_sigma: initialized
            .parameters
            .divb_cleaning_parabolic_sigma
            .unwrap_or(1.0),
        ..Default::default()
    };
    let state = MhdMfmState2d::from_primitive(
        positions,
        initialized.masses.clone(),
        velocities,
        initialized.specific_internal_energy.clone(),
        smoothing_lengths,
        &magnetic,
        &initialized.cleaning_phi,
        domain,
        2.0,
    )
    .map_err(ApplicationError::MhdEvolution2d)?;
    let rates =
        mhd_mfm_spatial_rates_2d(&state, controls).map_err(ApplicationError::MhdEvolution2d)?;
    let initial_bounds = public_mhd_particle_timestep_bounds_2d(
        &state,
        &rates,
        initialized.parameters.courant_factor,
        initialized.parameters.integration_accuracy,
    )
    .map_err(ApplicationError::MhdEvolution2d)?;
    let initial_timebins = quantize_public_mhd_initial_timebins_2d(
        &initial_bounds,
        0.0,
        initialized.parameters.time_max,
        initialized.parameters.max_timestep,
    )
    .map_err(ApplicationError::MhdEvolution2d)?;
    let mut hierarchy = begin_public_mhd_initial_hierarchy_2d(
        &state,
        &rates,
        &initial_timebins,
        0.0,
        initialized.parameters.time_max,
        0.0,
    )
    .map_err(ApplicationError::MhdEvolution2d)?;
    let mut snapshot_number = 0_u32;
    let mut step_count = 0_u64;
    write_briowu_drift_snapshot(
        initialized,
        &state.masses,
        hierarchy.current_drift_state(),
        hierarchy.retained_rates(),
        output_dir.join(format!("snapshot_{snapshot_number:03}.hdf5")),
        0.0,
    )?;
    snapshot_number += 1;
    let mut sync = hierarchy
        .drift_to_first_sync()
        .map_err(ApplicationError::MhdEvolution2d)?;
    loop {
        let next_output_time = if snapshot_number == 1 {
            initialized.parameters.time_between_snapshots
        } else {
            initialized.parameters.time_max
        };
        let tolerance = 64.0 * f64::EPSILON * next_output_time.abs().max(f64::MIN_POSITIVE);
        if sync.time > next_output_time + tolerance {
            return Err(ApplicationError::StateMismatch(format!(
                "hierarchical Brio-Wu schedule skipped output {next_output_time:.17e}; \
                 arrived at {:.17e}",
                sync.time
            )));
        }
        if (sync.time - next_output_time).abs() <= tolerance {
            hierarchy
                .drift_all_particles_to_current()
                .map_err(ApplicationError::MhdEvolution2d)?;
            write_briowu_drift_snapshot(
                initialized,
                &state.masses,
                hierarchy.current_drift_state(),
                hierarchy.retained_rates(),
                output_dir.join(format!("snapshot_{snapshot_number:03}.hdf5")),
                next_output_time,
            )?;
            snapshot_number += 1;
        }
        let terminal_sync = hierarchy.current_tick() >= LEGACY_TIMEBASE_TICKS;
        hierarchy
            .refresh_arriving_active_caches(
                initialized.parameters.desired_num_neighbors,
                initialized.parameters.max_neighbor_deviation,
            )
            .map_err(ApplicationError::MhdEvolution2d)?;
        let endpoint = hierarchy
            .evaluate_arriving_active_rates(controls, initialized.parameters.courant_factor)
            .map_err(ApplicationError::MhdEvolution2d)?;
        let kicked = hierarchy
            .finish_arriving_active_kicks(endpoint)
            .map_err(ApplicationError::MhdEvolution2d)?;
        step_count = step_count.checked_add(1).ok_or_else(|| {
            ApplicationError::StateMismatch("2-D MHD step count overflow".to_owned())
        })?;
        if step_count % 64 == 0 || terminal_sync {
            let active_count = hierarchy
                .active_mask()
                .iter()
                .filter(|&&is_active| is_active)
                .count();
            eprintln!(
                "Brio-Wu hierarchy progress: event={step_count} tick={} time={:.17e} active={active_count}",
                hierarchy.current_tick(),
                hierarchy.current_time()
            );
        }
        if terminal_sync {
            break;
        }
        let bounds = public_mhd_particle_timestep_bounds_from_primitive_2d(
            &kicked.state,
            hierarchy.predicted_primitive_cache(),
            &kicked.rates,
            initialized.parameters.courant_factor,
            initialized.parameters.integration_accuracy,
        )
        .map_err(ApplicationError::MhdEvolution2d)?;
        let active = hierarchy.active_mask();
        let active_bounds: Vec<_> = active
            .iter()
            .zip(&bounds)
            .map(|(&is_active, bound)| {
                is_active.then_some(bound.selected.min(initialized.parameters.max_timestep))
            })
            .collect();
        sync = hierarchy
            .begin_next_sync(&active_bounds)
            .map_err(ApplicationError::MhdEvolution2d)?;
    }
    if snapshot_number != 3 {
        return Err(ApplicationError::StateMismatch(format!(
            "hierarchical Brio-Wu wrote {snapshot_number} snapshots, expected 3"
        )));
    }
    let time = hierarchy.current_time();
    eprintln!(
        "completed {step_count} hierarchical 2-D MHD KDK events to t={time:.17e}; \
         wrote {snapshot_number} snapshots"
    );
    Ok(())
}

fn write_briowu_drift_snapshot(
    initialized: &InitializedBrioWu,
    masses: &[f64],
    drift: &PublicMhdDriftState2d,
    rates: &MhdMfmRates2d,
    path: PathBuf,
    time: f64,
) -> Result<(), ApplicationError> {
    let coordinates: Vec<[f64; 3]> = drift
        .positions
        .iter()
        .map(|position| [position.x, position.y, 0.0])
        .collect();
    let velocities: Vec<[f64; 3]> = drift
        .actual_velocities
        .iter()
        .map(|value| [value.x, value.y, value.z])
        .collect();
    let magnetic_field: Vec<[f64; 3]> = drift
        .predicted_magnetic_volume
        .iter()
        .zip(&drift.predicted_density)
        .zip(masses)
        .map(|((&value, &density), &mass)| value * (density / mass))
        .map(|value| [value.x, value.y, value.z])
        .collect();
    let cleaning_phi: Vec<f64> = drift
        .predicted_cleaning_mass
        .iter()
        .zip(masses)
        .map(|(&value, &mass)| value / mass)
        .collect();
    let gas_count = u64::try_from(initialized.particle_ids.len())
        .map_err(|_| ApplicationError::StateMismatch("particle count exceeds u64".to_owned()))?;
    let header = SnapshotHeader {
        time,
        box_size: initialized.parameters.box_size,
        num_part_total: [gas_count, 0, 0, 0, 0, 0],
        double_precision: true,
        effective_kernel_neighbors: Some(initialized.parameters.desired_num_neighbors),
    };
    write_mhd_wave(
        path,
        MhdWaveWriteView {
            header: &header,
            coordinates: &coordinates,
            velocities: &velocities,
            magnetic_field: &magnetic_field,
            ids: &initialized.particle_ids,
            masses,
            internal_energy: &drift.predicted_specific_internal_energy,
            density: &drift.predicted_density,
            smoothing_length: &drift.predicted_smoothing_lengths,
            cleaning_phi: Some(&cleaning_phi),
            cleaning_grad_phi: None,
            divergence_of_magnetic_field: Some(&rates.magnetic_divergence),
        },
    )
    .map_err(ApplicationError::Output)
}

fn write_mhd_snapshot(
    initialized: &InitializedMhdWave,
    rates: &MhdMfmRates1d,
    path: PathBuf,
    time: f64,
) -> Result<(), ApplicationError> {
    let primitive = initialized
        .state
        .primitive_columns()
        .map_err(ApplicationError::MhdEvolution)?;
    let coordinates: Vec<[f64; 3]> = initialized
        .state
        .positions
        .iter()
        .zip(&initialized.transverse_positions)
        .map(|(&x, transverse)| [x, transverse[0], transverse[1]])
        .collect();
    let velocities: Vec<[f64; 3]> = initialized
        .state
        .velocities
        .iter()
        .map(|value| [value.x, value.y, value.z])
        .collect();
    let magnetic_field: Vec<[f64; 3]> = primitive
        .magnetic
        .iter()
        .map(|value| [value.x, value.y, value.z])
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
    write_mhd_wave(
        path,
        MhdWaveWriteView {
            header: &header,
            coordinates: &coordinates,
            velocities: &velocities,
            magnetic_field: &magnetic_field,
            ids: &initialized.particle_ids,
            masses: &initialized.state.masses,
            internal_energy: &initialized.state.specific_internal_energy,
            density: &primitive.density,
            smoothing_length: &initialized.state.smoothing_lengths,
            cleaning_phi: Some(&primitive.cleaning_scalar),
            cleaning_grad_phi: None,
            divergence_of_magnetic_field: Some(&rates.magnetic_divergence),
        },
    )
    .map_err(ApplicationError::Output)
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn evolve_dustywave(mut initialized: InitializedSoundwave) -> Result<(), ApplicationError> {
    const MOMENTUM_RESIDUAL_LIMIT: f64 = 1.0e-18;

    let problem_name = if initialized.parameters.init_cond_file == "dustybox_ics" {
        "dusty-box"
    } else {
        "dusty-wave"
    };
    let output_dir = PathBuf::from(&initialized.parameters.output_dir);
    fs::create_dir_all(&output_dir).map_err(ApplicationError::OutputDirectory)?;
    let mut grains = initialized.grains.take().ok_or_else(|| {
        ApplicationError::StateMismatch("dusty-wave grain state is missing".to_owned())
    })?;
    let mut grain_acceleration = vec![0.0; grains.positions.len()];
    let mut timeline = SynchronizedTimeline1d::new(0.0, initialized.parameters.time_max)
        .map_err(ApplicationError::Hydro)?;
    let tick_duration = initialized.parameters.time_max / LEGACY_TIMEBASE_TICKS as f64;
    let mut rates =
        mfm_spatial_rates_1d_with_boundary(initialized.state.as_view(), BoundaryMode1d::Periodic)
            .map_err(ApplicationError::Hydro)?;
    let drag_parameters = EpsteinDragParameters {
        gamma: initialized.profile.gamma(),
        grain_internal_density: initialized
            .parameters
            .grain_internal_density
            .expect("strict dusty-wave parameters"),
    };
    let mut next_output_time = 0.0_f64;
    let mut next_output_index = 0_u64;
    let mut next_output_tick = Some(0_u64);
    let mut snapshot_number = 0_u32;
    let mut last_output_tick = None;
    let mut step_count = 0_u64;
    let mut maximum_momentum_residual = 0.0_f64;

    while !timeline.is_finished() {
        let synchronized = timeline
            .select_step(
                initialized.parameters.max_timestep,
                initialized.parameters.max_timestep,
            )
            .map_err(ApplicationError::Hydro)?;
        let start_tick = timeline.current_tick();
        let end_tick = start_tick + synchronized.ticks;
        let mut prepared = begin_mfm_kdk_1d(&initialized.state, &rates, synchronized.duration, 0.0)
            .map_err(ApplicationError::Hydro)?;
        let grain_start_positions = grains.positions.clone();
        let grain_half_velocities: Vec<f64> = grains
            .velocities
            .iter()
            .zip(&grain_acceleration)
            .map(|(&velocity, &acceleration)| velocity + 0.5 * synchronized.duration * acceleration)
            .collect();

        while next_output_tick.is_some_and(|tick| tick <= end_tick) {
            let output_tick = next_output_tick.expect("checked above");
            let elapsed = (output_tick - start_tick) as f64 * tick_duration;
            let gas_drift = prepared
                .drift_state(elapsed)
                .map_err(ApplicationError::Hydro)?;
            let grain_positions: Vec<f64> = grain_start_positions
                .iter()
                .zip(&grain_half_velocities)
                .map(|(&position, &velocity)| {
                    (position + elapsed * velocity).rem_euclid(initialized.state.box_size)
                })
                .collect();
            write_dusty_snapshot_columns(
                &initialized,
                &grains,
                &gas_drift.positions,
                &gas_drift.conserved_velocities,
                &gas_drift.predicted_specific_internal_energy,
                &gas_drift.predicted_density,
                &gas_drift.predicted_smoothing_lengths,
                &grain_positions,
                &grain_half_velocities,
                &grains.smoothing_lengths,
                output_dir.join(format!("snapshot_{snapshot_number:03}.hdf5")),
                output_tick as f64 * tick_duration,
            )?;
            last_output_tick = Some(output_tick);
            snapshot_number = snapshot_number.checked_add(1).ok_or_else(|| {
                ApplicationError::StateMismatch("snapshot number overflow".to_owned())
            })?;
            next_output_index = next_output_index.checked_add(1).ok_or_else(|| {
                ApplicationError::StateMismatch("output schedule index overflow".to_owned())
            })?;
            next_output_time = regular_output_time_with_terminal_snap(
                next_output_time + initialized.parameters.time_between_snapshots,
                0.0,
                initialized.parameters.time_between_snapshots,
                next_output_index,
                initialized.parameters.time_max,
            );
            let terminal_tolerance =
                64.0 * f64::EPSILON * initialized.parameters.time_max.abs().max(1.0);
            next_output_tick = (next_output_time
                <= initialized.parameters.time_max + terminal_tolerance)
                .then(|| {
                    terminal_snapped_output_tick(
                        next_output_time,
                        tick_duration,
                        initialized.parameters.time_max,
                    )
                });
        }

        let gas_drift = prepared
            .drift_state(synchronized.duration)
            .map_err(ApplicationError::Hydro)?;
        let endpoint_grain_positions: Vec<f64> = grain_start_positions
            .iter()
            .zip(&grain_half_velocities)
            .map(|(&position, &velocity)| {
                (position + synchronized.duration * velocity).rem_euclid(initialized.state.box_size)
            })
            .collect();
        let endpoint_grain_hsml = grain_smoothing_lengths_1d(
            &endpoint_grain_positions,
            &gas_drift.positions,
            initialized.state.box_size,
            initialized.parameters.desired_num_neighbors,
            initialized.parameters.max_neighbor_deviation,
        )?;
        let gas_points: Vec<GrainGasPoint1d> = gas_drift
            .positions
            .iter()
            .zip(&initialized.state.masses)
            .zip(&gas_drift.predicted_velocities)
            .zip(&gas_drift.predicted_specific_internal_energy)
            .map(
                |(((&position, &mass), &velocity), &specific_internal_energy)| GrainGasPoint1d {
                    position,
                    mass,
                    velocity,
                    specific_internal_energy,
                },
            )
            .collect();
        let grain_points: Vec<GrainPoint1d> = (0..grains.positions.len())
            .map(|index| GrainPoint1d {
                position: endpoint_grain_positions[index],
                mass: grains.masses[index],
                velocity: grain_half_velocities[index],
                smoothing_length: endpoint_grain_hsml[index],
                radius: grains.grain_sizes[index],
            })
            .collect();
        let drag_batch = compute_epstein_drag_batch_1d(
            &grain_points,
            &gas_points,
            initialized.state.box_size,
            synchronized.duration,
            drag_parameters,
        )
        .map_err(ApplicationError::Grain)?;
        let momentum_residual = drag_batch.momentum_residual.abs();
        if !momentum_residual.is_finite() || momentum_residual > MOMENTUM_RESIDUAL_LIMIT {
            return Err(ApplicationError::StateMismatch(format!(
                "drag batch momentum residual {momentum_residual:.17e} exceeds \
                 {MOMENTUM_RESIDUAL_LIMIT:.1e}"
            )));
        }
        maximum_momentum_residual = maximum_momentum_residual.max(momentum_residual);
        let grain_velocity_deltas: Vec<f64> = drag_batch
            .grains
            .iter()
            .map(|grain| grain.impulse.grain_velocity_delta)
            .collect();
        let gas_velocity_deltas = drag_batch.gas_velocity_deltas;

        let (mut endpoint, new_rates) = finish_mfm_kdk_1d(
            prepared,
            initialized.parameters.desired_num_neighbors,
            initialized.parameters.max_neighbor_deviation,
        )
        .map_err(ApplicationError::Hydro)?;
        for (velocity, delta) in endpoint.velocities.iter_mut().zip(&gas_velocity_deltas) {
            *velocity += delta;
        }
        let gas_density: Vec<f64> = density_at_hsml_1d(
            &endpoint.positions,
            &endpoint.masses,
            &endpoint.smoothing_lengths,
            endpoint.box_size,
        )
        .map_err(ApplicationError::Hydro)?
        .into_iter()
        .map(|estimate| estimate.density)
        .collect();
        endpoint.specific_internal_energy = gas_density
            .iter()
            .map(|density| 0.9 * density.powf(2.0 / 3.0))
            .collect();
        grains.positions = endpoint_grain_positions;
        grains.velocities = grain_half_velocities
            .iter()
            .zip(&grain_velocity_deltas)
            .map(|(&velocity, &delta)| velocity + 0.5 * delta)
            .collect();
        grain_acceleration = grain_velocity_deltas
            .iter()
            .map(|delta| delta / synchronized.duration)
            .collect();
        grains.smoothing_lengths = endpoint_grain_hsml;
        initialized.state = endpoint;
        rates = new_rates;
        timeline
            .advance(synchronized)
            .map_err(ApplicationError::Hydro)?;
        step_count += 1;
    }

    if last_output_tick != Some(LEGACY_TIMEBASE_TICKS) {
        let density: Vec<f64> = density_at_hsml_1d(
            &initialized.state.positions,
            &initialized.state.masses,
            &initialized.state.smoothing_lengths,
            initialized.state.box_size,
        )
        .map_err(ApplicationError::Hydro)?
        .into_iter()
        .map(|estimate| estimate.density)
        .collect();
        write_dusty_snapshot_columns(
            &initialized,
            &grains,
            &initialized.state.positions,
            &initialized.state.velocities,
            &initialized.state.specific_internal_energy,
            &density,
            &initialized.state.smoothing_lengths,
            &grains.positions,
            &grains.velocities,
            &grains.smoothing_lengths,
            output_dir.join(format!("snapshot_{snapshot_number:03}.hdf5")),
            initialized.parameters.time_max,
        )?;
        snapshot_number += 1;
    }
    eprintln!(
        "completed {step_count} synchronized {problem_name} steps to t={:.17e}; \
         wrote {snapshot_number} snapshots; maximum drag momentum residual={maximum_momentum_residual:.3e}",
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

fn terminal_snapped_output_tick(output_time: f64, tick_duration: f64, terminal_time: f64) -> u64 {
    let tolerance = 64.0 * f64::EPSILON * terminal_time.abs().max(1.0);
    if (output_time - terminal_time).abs() <= tolerance {
        LEGACY_TIMEBASE_TICKS
    } else {
        legacy_output_tick(output_time, tick_duration)
    }
}

#[allow(clippy::cast_precision_loss)]
fn regular_output_time_with_terminal_snap(
    accumulated_time: f64,
    first_output_time: f64,
    output_interval: f64,
    output_index: u64,
    terminal_time: f64,
) -> f64 {
    let indexed_time = (output_index as f64).mul_add(output_interval, first_output_time);
    let tolerance = 64.0 * f64::EPSILON * terminal_time.abs().max(1.0);
    if (indexed_time - terminal_time).abs() <= tolerance {
        terminal_time
    } else {
        accumulated_time
    }
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

#[allow(clippy::too_many_arguments)]
fn write_dusty_snapshot_columns(
    initialized: &InitializedSoundwave,
    grains: &GrainRuntimeState1d,
    gas_positions: &[f64],
    gas_velocities: &[f64],
    gas_internal_energy: &[f64],
    gas_density: &[f64],
    gas_smoothing_lengths: &[f64],
    grain_positions: &[f64],
    grain_velocities: &[f64],
    grain_smoothing_lengths: &[f64],
    path: PathBuf,
    time: f64,
) -> Result<(), ApplicationError> {
    let gas_coordinates: Vec<[f64; 3]> = gas_positions
        .iter()
        .zip(&initialized.transverse_vectors)
        .map(|(&x, shell)| [x, shell.position[0], shell.position[1]])
        .collect();
    let gas_velocity_vectors: Vec<[f64; 3]> = gas_velocities
        .iter()
        .zip(&initialized.transverse_vectors)
        .map(|(&x, shell)| [x, shell.velocity[0], shell.velocity[1]])
        .collect();
    let grain_coordinates: Vec<[f64; 3]> = grain_positions
        .iter()
        .zip(&grains.transverse_vectors)
        .map(|(&x, shell)| [x, shell.position[0], shell.position[1]])
        .collect();
    let grain_velocity_vectors: Vec<[f64; 3]> = grain_velocities
        .iter()
        .zip(&grains.transverse_vectors)
        .map(|(&x, shell)| [x, shell.velocity[0], shell.velocity[1]])
        .collect();
    let gas_count = u64::try_from(initialized.particle_ids.len())
        .map_err(|_| ApplicationError::StateMismatch("gas count exceeds u64".to_owned()))?;
    let grain_count = u64::try_from(grains.particle_ids.len())
        .map_err(|_| ApplicationError::StateMismatch("grain count exceeds u64".to_owned()))?;
    let header = SnapshotHeader {
        time,
        box_size: initialized.state.box_size,
        num_part_total: [gas_count, 0, 0, grain_count, 0, 0],
        double_precision: true,
        effective_kernel_neighbors: Some(initialized.parameters.desired_num_neighbors),
    };
    write_dustywave(
        path,
        DustyWaveWriteView {
            header: &header,
            gas: GasWriteView {
                coordinates: &gas_coordinates,
                velocities: &gas_velocity_vectors,
                ids: &initialized.particle_ids,
                masses: &initialized.state.masses,
                internal_energy: gas_internal_energy,
                density: gas_density,
                smoothing_length: gas_smoothing_lengths,
            },
            grains: GrainWriteView {
                coordinates: &grain_coordinates,
                velocities: &grain_velocity_vectors,
                ids: &grains.particle_ids,
                masses: &grains.masses,
                grain_size: &grains.grain_sizes,
                smoothing_length: grain_smoothing_lengths,
            },
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
    Geometry2d(gizmo_hydro::meshless_2d::GeometryError),
    MhdEvolution(gizmo_hydro::mhd_evolution::MhdEvolutionError),
    MhdEvolution2d(gizmo_hydro::mhd_evolution_2d::MhdEvolution2dError),
    Grain(gizmo_hydro::grain::GrainError),
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
            Self::Geometry2d(error) => error.fmt(formatter),
            Self::MhdEvolution(error) => error.fmt(formatter),
            Self::MhdEvolution2d(error) => error.fmt(formatter),
            Self::Grain(error) => error.fmt(formatter),
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
    const DUSTYWAVE_CONFIG: &str =
        include_str!("../../../../validation/oracles/dustywave/legacy-config.sh");
    const MHD_WAVE_CONFIG: &str =
        include_str!("../../../../validation/oracles/mhd_wave/frontier-config.sh");
    const MHD_WAVE_PARAMETERS: &str =
        include_str!("../../../../validation/oracles/mhd_wave/frontier.params");
    const BRIOWU_CONFIG: &str =
        include_str!("../../../../validation/oracles/briowu/corrected-c-config.sh");
    const BRIOWU_PUBLIC_PARAMETERS: &str =
        include_str!("../../../../validation/oracles/briowu/public.params");
    const BRIOWU_FRONTIER_PARAMETERS: &str =
        include_str!("../../../../validation/oracles/briowu/frontier.params");
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
    fn exact_mhd_wave_config_profile_is_required() {
        let manifest = ConfigManifest::parse(MHD_WAVE_CONFIG).unwrap();
        assert_eq!(
            validate_strict_config(&manifest).unwrap(),
            StrictProfile::MhdWave
        );
        for invalid in [
            MHD_WAVE_CONFIG.replace("MAGNETIC\n", ""),
            format!("{MHD_WAVE_CONFIG}FORCE_EQUAL_TIMESTEPS\n"),
            MHD_WAVE_CONFIG.replace("EOS_GAMMA=(5.0/3.0)", "EOS_GAMMA=(5./3.)"),
            MHD_WAVE_CONFIG.replace("MAGNETIC\n", "MAGNETIC=1\n"),
        ] {
            assert!(matches!(
                validate_strict_config(&ConfigManifest::parse(&invalid).unwrap()),
                Err(ApplicationError::UnsupportedConfig(_))
            ));
        }
    }

    #[test]
    fn exact_mhd_wave_parameters_and_cleaning_sigmas_are_required() {
        let parameters = read_mhd_wave_parameters(MHD_WAVE_PARAMETERS).unwrap();
        assert_eq!(parameters.divb_cleaning_parabolic_sigma, Some(0.2));
        assert_eq!(parameters.divb_cleaning_hyperbolic_sigma, Some(1.0));
        for invalid in [
            MHD_WAVE_PARAMETERS.replace("DivBcleaningParabolicSigma         0.2\n", ""),
            MHD_WAVE_PARAMETERS.replace(
                "DivBcleaningHyperbolicSigma        1.0",
                "DivBcleaningHyperbolicSigma        0.9",
            ),
            format!("{MHD_WAVE_PARAMETERS}MinSizeTimestep 1e-8\n"),
        ] {
            assert!(matches!(
                read_mhd_wave_parameters(&invalid),
                Err(ApplicationError::UnsupportedParameters(_))
            ));
        }
    }

    #[test]
    fn exact_two_dimensional_briowu_config_is_required() {
        let manifest = ConfigManifest::parse(BRIOWU_CONFIG).unwrap();
        assert_eq!(
            validate_strict_config(&manifest).unwrap(),
            StrictProfile::BrioWu
        );
        for invalid in [
            BRIOWU_CONFIG.replace("BOX_LONG_X=16", "BOX_LONG_X=1"),
            BRIOWU_CONFIG.replace("BOX_SPATIAL_DIMENSION=2", "BOX_SPATIAL_DIMENSION=1"),
            BRIOWU_CONFIG.replace("EOS_GAMMA=(2.0)", "EOS_GAMMA=(5.0/3.0)"),
            BRIOWU_CONFIG.replace("MAGNETIC\n", ""),
            format!("{BRIOWU_CONFIG}DEVELOPER_MODE=1\n"),
        ] {
            assert!(matches!(
                validate_strict_config(&ConfigManifest::parse(&invalid).unwrap()),
                Err(ApplicationError::UnsupportedConfig(_))
            ));
        }
        let exact_public = BRIOWU_CONFIG.replace("OUTPUT_IN_DOUBLEPRECISION\n", "");
        assert_eq!(
            validate_strict_config(&ConfigManifest::parse(&exact_public).unwrap()).unwrap(),
            StrictProfile::BrioWu
        );
        let frontier = include_str!("../../../../validation/oracles/briowu/frontier-config.sh");
        assert_eq!(
            validate_strict_config(&ConfigManifest::parse(frontier).unwrap()).unwrap(),
            StrictProfile::BrioWu
        );
    }

    #[test]
    fn exact_public_and_frontier_briowu_parameters_are_accepted() {
        let public = read_briowu_parameters(BRIOWU_PUBLIC_PARAMETERS).unwrap();
        assert_eq!(public.init_cond_file, "briowu_ics");
        assert_eq!(public.desired_num_neighbors.to_bits(), 20.0_f64.to_bits());
        assert_eq!(public.max_neighbor_deviation.to_bits(), 0.05_f64.to_bits());
        let frontier = read_briowu_parameters(BRIOWU_FRONTIER_PARAMETERS).unwrap();
        assert_eq!(frontier.max_memory_mb, Some(2000));
        assert_eq!(frontier.divb_cleaning_parabolic_sigma, Some(1.0));
        assert_eq!(frontier.max_neighbor_deviation.to_bits(), 0.1_f64.to_bits());
        for invalid in [
            BRIOWU_PUBLIC_PARAMETERS
                .replace("DesNumNgb                          20", "DesNumNgb 4"),
            format!("{BRIOWU_PUBLIC_PARAMETERS}CourantFac 0.2\n"),
            BRIOWU_FRONTIER_PARAMETERS
                .replace("DivBcleaningHyperbolicSigma", "UnexpectedHyperbolicSigma"),
        ] {
            assert!(matches!(
                read_briowu_parameters(&invalid),
                Err(ApplicationError::UnsupportedParameters(_))
            ));
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn tiny_briowu_cli_trajectory_uses_hierarchical_events() {
        let mut parameters = read_briowu_parameters(BRIOWU_FRONTIER_PARAMETERS).unwrap();
        parameters.time_max = 8.0 / LEGACY_TIMEBASE_TICKS as f64;
        parameters.max_timestep = 2.0 / LEGACY_TIMEBASE_TICKS as f64;
        parameters.time_between_snapshots = 4.0 / LEGACY_TIMEBASE_TICKS as f64;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let output_dir =
            std::env::temp_dir().join(format!("gizmo-tiny-briowu-{}-{nonce}", std::process::id()));
        parameters.output_dir = output_dir.to_string_lossy().into_owned();
        let nx = 16_usize;
        let ny = 4_usize;
        let count = nx * ny;
        let positions: Vec<_> = (0..ny)
            .flat_map(|iy| {
                (0..nx).map(move |ix| {
                    [
                        (ix as f64 + 0.5) * 4.0 / nx as f64,
                        (iy as f64 + 0.5) * 0.25 / ny as f64,
                        0.0,
                    ]
                })
            })
            .collect();
        let initialized = InitializedBrioWu {
            parameters,
            particle_ids: (1..=u64::try_from(count).unwrap()).collect(),
            positions,
            masses: vec![1.0 / count as f64; count],
            velocities: vec![[0.0; 3]; count],
            specific_internal_energy: vec![1.0; count],
            density: vec![1.0; count],
            smoothing_lengths: vec![0.18; count],
            magnetic_field: vec![[1.0, 0.0, 0.0]; count],
            cleaning_phi: vec![0.0; count],
            cleaning_grad_phi: vec![[0.0; 3]; count],
            divergence_of_magnetic_field: vec![0.0; count],
            box_lengths: [4.0, 0.25, 0.25],
        };
        evolve_briowu(&initialized).unwrap();
        for snapshot in 0..3 {
            assert!(
                output_dir
                    .join(format!("snapshot_{snapshot:03}.hdf5"))
                    .is_file()
            );
        }
        std::fs::remove_dir_all(output_dir).unwrap();
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
    fn exact_dustywave_profile_and_parameters_are_accepted() {
        assert_eq!(
            validate_strict_config(&ConfigManifest::parse(DUSTYWAVE_CONFIG).unwrap()).unwrap(),
            StrictProfile::Dustywave
        );
        let path =
            std::env::temp_dir().join(format!("gizmo-dustywave-params-{}.txt", std::process::id()));
        let parameters = include_str!("../../../../validation/oracles/dustywave/legacy.params");
        fs::write(&path, parameters).unwrap();
        let parsed = read_profile_parameters(&path, StrictProfile::Dustywave).unwrap();
        assert_eq!(parsed.type3_softening, Some(0.001));
        fs::write(
            &path,
            parameters.replace("Softening_Type3                    0.001", "% missing"),
        )
        .unwrap();
        assert!(matches!(
            read_profile_parameters(&path, StrictProfile::Dustywave),
            Err(ApplicationError::UnsupportedParameters(message))
                if message.contains("Softening_Type3")
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn dustybox_uses_the_same_strict_grain_profile() {
        let path =
            std::env::temp_dir().join(format!("gizmo-dustybox-params-{}.txt", std::process::id()));
        let parameters = include_str!("../../../../validation/oracles/dustywave/legacy.params")
            .replace("dustywave_ics", "dustybox_ics");
        fs::write(&path, parameters).unwrap();
        let parsed = read_profile_parameters(&path, StrictProfile::Dustywave).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(parsed.init_cond_file, "dustybox_ics");
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

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn terminal_output_time_roundoff_snaps_to_the_final_tick() {
        let terminal_time = 2.5;
        let tick_duration = terminal_time / LEGACY_TIMEBASE_TICKS as f64;
        let accumulated_terminal = 2.499_999_999_999_990_7;
        assert!(legacy_output_tick(accumulated_terminal, tick_duration) < LEGACY_TIMEBASE_TICKS);
        assert_eq!(
            terminal_snapped_output_tick(accumulated_terminal, tick_duration, terminal_time),
            LEGACY_TIMEBASE_TICKS
        );
        assert_eq!(
            terminal_snapped_output_tick(1.2, tick_duration, terminal_time),
            legacy_output_tick(1.2, tick_duration)
        );
    }

    #[test]
    fn indexed_schedule_detection_survives_large_accumulation_error() {
        assert_eq!(
            regular_output_time_with_terminal_snap(
                0.999_999_999_999_906_2,
                0.0,
                1.0e-4,
                10_000,
                1.0,
            )
            .to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            regular_output_time_with_terminal_snap(
                1.000_000_000_007_918,
                0.0,
                1.0e-6,
                1_000_000,
                1.0,
            )
            .to_bits(),
            1.0_f64.to_bits()
        );
    }
}
