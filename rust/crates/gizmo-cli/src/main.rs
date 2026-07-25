#![forbid(unsafe_code)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gizmo_cli::{CliError, Invocation, USAGE};
use gizmo_config::ConfigManifest;
use gizmo_hydro::{
    GradientEstimate, density_at_hsml_1d, gradients_at_hsml_1d, solve_smoothing_lengths_1d,
};
use gizmo_io::read_soundwave;
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

    if !invocation.initialize_only {
        return Err(ApplicationError::NotYetPorted);
    }
    validate_soundwave_config(&manifest)?;
    initialize_soundwave(&invocation.parameter_file, &manifest)
}

fn validate_soundwave_config(manifest: &ConfigManifest) -> Result<(), ApplicationError> {
    let allowed = [
        "BOX_PERIODIC",
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
        if !allowed.contains(&option.name.as_str()) {
            return Err(ApplicationError::UnsupportedConfig(format!(
                "option `{}` is outside the sound-wave initialization profile",
                option.name
            )));
        }
    }
    for required in [
        "BOX_PERIODIC",
        "HYDRO_MESHLESS_FINITE_MASS",
        "SELFGRAVITY_OFF",
    ] {
        if manifest.get(required).is_none() {
            return Err(ApplicationError::UnsupportedConfig(format!(
                "required option `{required}` is missing"
            )));
        }
    }
    require_config_value(manifest, "BOX_SPATIAL_DIMENSION", "1")?;
    require_config_value(manifest, "EOS_GAMMA", "(5.0/3.0)")?;
    Ok(())
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

fn initialize_soundwave(
    parameter_file: &Path,
    manifest: &ConfigManifest,
) -> Result<(), ApplicationError> {
    let parameters =
        SoundwaveParameters::from_path(parameter_file).map_err(ApplicationError::Parameters)?;
    let fixture_path = resolve_initial_conditions(&parameters.init_cond_file);
    let snapshot = read_soundwave(&fixture_path).map_err(ApplicationError::Input)?;
    if snapshot.header.box_size.to_bits() != parameters.box_size.to_bits() {
        return Err(ApplicationError::StateMismatch(format!(
            "parameter BoxSize={} differs from HDF5 BoxSize={}",
            parameters.box_size, snapshot.header.box_size
        )));
    }
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
    let positions: Vec<f64> = snapshot
        .gas
        .coordinates
        .iter()
        .map(|coordinate| coordinate[0])
        .collect();
    let at_legacy_hsml = density_at_hsml_1d(
        &positions,
        &snapshot.gas.masses,
        legacy_hsml,
        snapshot.header.box_size,
    )
    .map_err(ApplicationError::Hydro)?;
    let particle_count = u32::try_from(snapshot.gas.len())
        .map_err(|_| ApplicationError::StateMismatch("particle count exceeds u32".to_owned()))?;
    let initial_hsml =
        vec![2.0 * snapshot.header.box_size / f64::from(particle_count); snapshot.gas.len()];
    let solved = solve_smoothing_lengths_1d(
        &positions,
        &snapshot.gas.masses,
        &initial_hsml,
        snapshot.header.box_size,
        parameters.desired_num_neighbors,
        1.0e-8,
    )
    .map_err(ApplicationError::Hydro)?;

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
        .map(|particle| {
            (particle.estimate.effective_neighbors - parameters.desired_num_neighbors).abs()
        })
        .fold(0.0, f64::max);
    let [
        density_gradient_error,
        velocity_gradient_error,
        pressure_gradient_error,
    ] = soundwave_gradient_errors(&snapshot, &positions, expected_density, legacy_hsml)?;
    let summary = InitializationSummary {
        particle_count,
        box_size: snapshot.header.box_size,
        max_density_relative_error,
        max_hsml_relative_difference,
        max_neighbor_deviation,
        density_gradient_error,
        velocity_gradient_error,
        pressure_gradient_error,
    };
    summary.validate()?;
    summary.print(&manifest.sha256());
    Ok(())
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

#[derive(Clone, Copy, Debug)]
struct InitializationSummary {
    particle_count: u32,
    box_size: f64,
    max_density_relative_error: f64,
    max_hsml_relative_difference: f64,
    max_neighbor_deviation: f64,
    density_gradient_error: f64,
    velocity_gradient_error: f64,
    pressure_gradient_error: f64,
}

impl InitializationSummary {
    fn validate(self) -> Result<(), ApplicationError> {
        for (field, value, limit) in [
            ("density parity", self.max_density_relative_error, 1.0e-10),
            ("Hsml parity", self.max_hsml_relative_difference, 1.0e-3),
            ("neighbor constraint", self.max_neighbor_deviation, 1.0e-8),
            ("density gradient", self.density_gradient_error, 1.0e-4),
            ("velocity gradient", self.velocity_gradient_error, 1.0e-4),
            ("pressure gradient", self.pressure_gradient_error, 1.0e-4),
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
            "  \"pressure_gradient_error\": {:.17e}",
            self.pressure_gradient_error
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
    Input(gizmo_io::InputError),
    Hydro(gizmo_hydro::HydroError),
    UnsupportedConfig(String),
    MissingDataset(&'static str),
    StateMismatch(String),
    NotYetPorted,
}

impl std::fmt::Display for ApplicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cli(error) => error.fmt(formatter),
            Self::Config(error) => error.fmt(formatter),
            Self::Parameters(error) => error.fmt(formatter),
            Self::Input(error) => error.fmt(formatter),
            Self::Hydro(error) => error.fmt(formatter),
            Self::UnsupportedConfig(error) => {
                write!(formatter, "unsupported initialization config: {error}")
            }
            Self::MissingDataset(name) => {
                write!(formatter, "initial condition is missing required `{name}`")
            }
            Self::StateMismatch(error) => write!(formatter, "initial state mismatch: {error}"),
            Self::NotYetPorted => formatter.write_str(
                "simulation execution is not yet ported; no scientific computation was performed",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hdf5_suffix_is_appended_to_legacy_basename_even_when_it_contains_dots() {
        assert_eq!(
            resolve_initial_conditions("run.v1"),
            PathBuf::from("run.v1.hdf5")
        );
    }
}
