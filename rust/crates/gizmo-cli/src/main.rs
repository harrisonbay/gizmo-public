#![forbid(unsafe_code)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gizmo_cli::{CliError, Invocation, USAGE};
use gizmo_config::ConfigManifest;
use gizmo_hydro::{density_at_hsml_1d, solve_smoothing_lengths_1d};
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
    if !max_density_relative_error.is_finite() || max_density_relative_error > 1.0e-10 {
        return Err(ApplicationError::StateMismatch(format!(
            "density parity exceeded 1e-10: {max_density_relative_error}"
        )));
    }
    if !max_hsml_relative_difference.is_finite() || max_hsml_relative_difference > 1.0e-3 {
        return Err(ApplicationError::StateMismatch(format!(
            "Hsml parity exceeded 1e-3: {max_hsml_relative_difference}"
        )));
    }
    if !max_neighbor_deviation.is_finite() || max_neighbor_deviation > 1.0e-8 {
        return Err(ApplicationError::StateMismatch(format!(
            "neighbor constraint exceeded 1e-8: {max_neighbor_deviation}"
        )));
    }

    println!("{{");
    println!("  \"config_sha256\": \"{}\",", manifest.sha256());
    println!("  \"particles\": {particle_count},");
    println!("  \"box_size\": {},", snapshot.header.box_size);
    println!("  \"max_density_relative_error\": {max_density_relative_error:.17e},");
    println!("  \"max_hsml_relative_difference\": {max_hsml_relative_difference:.17e},");
    println!("  \"max_neighbor_deviation\": {max_neighbor_deviation:.17e}");
    println!("}}");
    Ok(())
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
