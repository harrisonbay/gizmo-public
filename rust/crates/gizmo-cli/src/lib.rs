#![forbid(unsafe_code)]

use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Invocation {
    pub config_file: PathBuf,
    pub parameter_file: PathBuf,
    pub restart: RestartFlag,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum RestartFlag {
    #[default]
    InitialConditions = 0,
    RestartFiles = 1,
    Snapshot = 2,
    GroupFinding = 3,
    ConvertSnapshot = 4,
    PowerSpectrum = 5,
    GasVelocityPowerSpectrum = 6,
}

impl TryFrom<u8> for RestartFlag {
    type Error = CliError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::InitialConditions),
            1 => Ok(Self::RestartFiles),
            2 => Ok(Self::Snapshot),
            3 => Ok(Self::GroupFinding),
            4 => Ok(Self::ConvertSnapshot),
            5 => Ok(Self::PowerSpectrum),
            6 => Ok(Self::GasVelocityPowerSpectrum),
            _ => Err(CliError::InvalidRestartFlag(value.to_string())),
        }
    }
}

impl Invocation {
    /// Parse GIZMO-compatible positional arguments and the Rust config override.
    ///
    /// # Errors
    ///
    /// Returns an error for missing/extra arguments, unknown options, or a
    /// restart mode outside the legacy range 0 through 6.
    pub fn parse<I, S>(arguments: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        let mut arguments = arguments.into_iter().map(Into::into);
        let mut config_file = PathBuf::from("Config.sh");
        let mut positional = Vec::new();

        while let Some(argument) = arguments.next() {
            if argument == "--config" {
                config_file = arguments
                    .next()
                    .map(PathBuf::from)
                    .ok_or(CliError::MissingConfigPath)?;
            } else if argument == "--help" || argument == "-h" {
                return Err(CliError::HelpRequested);
            } else if argument
                .to_str()
                .is_some_and(|value| value.starts_with('-'))
            {
                return Err(CliError::UnknownOption(argument));
            } else {
                positional.push(argument);
            }
        }

        if positional.is_empty() {
            return Err(CliError::MissingParameterFile);
        }
        if positional.len() > 2 {
            return Err(CliError::TooManyArguments);
        }
        let parameter_file = PathBuf::from(positional.remove(0));
        let restart = positional
            .pop()
            .map(|value| parse_restart_flag(&value))
            .transpose()?
            .unwrap_or_default();

        Ok(Self {
            config_file,
            parameter_file,
            restart,
        })
    }
}

fn parse_restart_flag(value: &OsString) -> Result<RestartFlag, CliError> {
    let text = value
        .to_str()
        .ok_or_else(|| CliError::InvalidRestartFlag("<non-UTF-8>".to_owned()))?;
    let numeric = text
        .parse::<u8>()
        .map_err(|_| CliError::InvalidRestartFlag(text.to_owned()))?;
    RestartFlag::try_from(numeric)
}

#[derive(Debug, Eq, PartialEq)]
pub enum CliError {
    HelpRequested,
    MissingConfigPath,
    MissingParameterFile,
    TooManyArguments,
    UnknownOption(OsString),
    InvalidRestartFlag(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HelpRequested => formatter.write_str("help requested"),
            Self::MissingConfigPath => formatter.write_str("`--config` requires a file path"),
            Self::MissingParameterFile => formatter.write_str("parameter file is missing"),
            Self::TooManyArguments => {
                formatter.write_str("too many arguments; expected <ParameterFile> [<RestartFlag>]")
            }
            Self::UnknownOption(option) => write!(formatter, "unknown option {option:?}"),
            Self::InvalidRestartFlag(value) => write!(
                formatter,
                "invalid restart flag `{value}`; expected an integer from 0 through 6"
            ),
        }
    }
}

impl Error for CliError {}

pub const USAGE: &str = "\
Usage: gizmo [--config <Config.sh>] <ParameterFile> [<RestartFlag>]

RestartFlag:
  0  Read initial conditions and start simulation (default)
  1  Read restart files and resume simulation
  2  Restart from a snapshot
  3  Run FOF and optionally SUBFIND
  4  Convert a snapshot to a different format
  5  Calculate power spectrum and two-point function
  6  Calculate gas velocity power spectrum

The legacy restart-snapshot-number argument is not ported yet.
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_initial_conditions_and_config_sh() {
        let invocation = Invocation::parse(["params.txt"]).unwrap();
        assert_eq!(invocation.config_file, PathBuf::from("Config.sh"));
        assert_eq!(invocation.parameter_file, PathBuf::from("params.txt"));
        assert_eq!(invocation.restart, RestartFlag::InitialConditions);
    }

    #[test]
    fn parses_every_legacy_restart_mode() {
        for value in 0..=6_u8 {
            let invocation =
                Invocation::parse(["--config", "science.sh", "params.txt", &value.to_string()])
                    .unwrap();
            assert_eq!(invocation.config_file, PathBuf::from("science.sh"));
            assert_eq!(invocation.restart as u8, value);
        }
    }

    #[test]
    fn malformed_invocations_fail_explicitly() {
        let cases: &[&[&str]] = &[
            &[],
            &["--config"],
            &["--unknown", "params.txt"],
            &["params.txt", "-1"],
            &["params.txt", "7"],
            &["params.txt", "1", "12"],
        ];
        for arguments in cases {
            assert!(
                Invocation::parse(arguments.iter().copied()).is_err(),
                "unexpectedly accepted {arguments:?}"
            );
        }
    }
}
