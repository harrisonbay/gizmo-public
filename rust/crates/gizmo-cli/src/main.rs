#![forbid(unsafe_code)]

use std::env;
use std::process::ExitCode;

use gizmo_cli::{CliError, Invocation, USAGE};
use gizmo_config::ConfigManifest;

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

    Err(ApplicationError::NotYetPorted)
}

#[derive(Debug)]
enum ApplicationError {
    Cli(CliError),
    Config(gizmo_config::ReadConfigError),
    NotYetPorted,
}

impl std::fmt::Display for ApplicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cli(error) => error.fmt(formatter),
            Self::Config(error) => error.fmt(formatter),
            Self::NotYetPorted => formatter.write_str(
                "simulation execution is not yet ported; no scientific computation was performed",
            ),
        }
    }
}
