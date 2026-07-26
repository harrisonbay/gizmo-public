#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

const SUPPORTED_TAGS: &[&str] = &[
    "InitCondFile",
    "OutputDir",
    "TimeMax",
    "MaxSizeTimestep",
    "MinSizeTimestep",
    "BoxSize",
    "TimeBetSnapshot",
    "DesNumNgb",
    "ErrTolIntAccuracy",
    "CourantFac",
    "MaxRMSDisplacementFac",
    "ErrTolForceAcc",
    "TimeBetStatistics",
    "MaxMemSize",
    "ErrTolTheta",
    "MaxNumNgbDeviation",
    "DivBcleaningParabolicSigma",
    "DivBcleaningHyperbolicSigma",
    "Grain_Internal_Density",
    "Grain_Size_Min",
    "Grain_Size_Max",
    "Grain_Size_Spectrum_Powerlaw",
    "Softening_Type3",
    "ResubmitOn",
    "ResubmitCommand",
];

/// Runtime parameters for the pinned, one-dimensional soundwave port.
///
/// This intentionally is not a generic GIZMO parameter bag. Unsupported
/// parameters are rejected so that the Rust executable cannot silently run a
/// physical model it has not ported.
#[derive(Clone, Debug, PartialEq)]
pub struct SoundwaveParameters {
    pub init_cond_file: String,
    pub output_dir: String,
    pub time_max: f64,
    pub max_timestep: f64,
    pub min_timestep: Option<f64>,
    pub box_size: f64,
    pub time_between_snapshots: f64,
    pub desired_num_neighbors: f64,
    pub integration_accuracy: f64,
    pub courant_factor: f64,
    pub max_rms_displacement_factor: f64,
    pub force_accuracy: f64,
    pub time_between_statistics: f64,
    pub max_memory_mb: Option<u64>,
    pub tree_opening_angle: f64,
    pub max_neighbor_deviation: f64,
    pub divb_cleaning_parabolic_sigma: Option<f64>,
    pub divb_cleaning_hyperbolic_sigma: Option<f64>,
    pub grain_internal_density: Option<f64>,
    pub grain_size_min: Option<f64>,
    pub grain_size_max: Option<f64>,
    pub grain_size_spectrum_powerlaw: Option<f64>,
    pub type3_softening: Option<f64>,
    pub resubmit: bool,
    pub resubmit_command: String,
}

impl SoundwaveParameters {
    /// Parse a legacy GIZMO runtime parameter document.
    ///
    /// Each enabled line is `Tag Value`; `%` begins a comment. The vocabulary
    /// is restricted to the public soundwave oracle profile.
    ///
    /// # Errors
    ///
    /// Returns a line-numbered error for malformed, unsupported, repeated, or
    /// invalid values, and reports required fields that are absent.
    #[allow(clippy::too_many_lines)]
    pub fn parse(input: &str) -> Result<Self, ParameterError> {
        let mut entries = BTreeMap::<String, Entry>::new();

        for (line_index, raw_line) in input.lines().enumerate() {
            let line = line_index + 1;
            let definition = raw_line
                .split_once('%')
                .map_or(raw_line, |pair| pair.0)
                .trim();
            if definition.is_empty() {
                continue;
            }

            let mut tokens = definition.split_whitespace();
            let tag = tokens.next().unwrap_or_default();
            let value = tokens
                .next()
                .ok_or_else(|| ParameterError::at(line, ErrorKind::MissingValue(tag.to_owned())))?;
            if tokens.next().is_some() {
                return Err(ParameterError::at(
                    line,
                    ErrorKind::TrailingTokens(tag.to_owned()),
                ));
            }
            if !SUPPORTED_TAGS.contains(&tag) {
                return Err(ParameterError::at(
                    line,
                    ErrorKind::UnsupportedTag(tag.to_owned()),
                ));
            }

            if let Some(previous) = entries.get(tag) {
                let kind = if previous.value == value {
                    ErrorKind::Duplicate {
                        tag: tag.to_owned(),
                        first_line: previous.line,
                    }
                } else {
                    ErrorKind::Conflict {
                        tag: tag.to_owned(),
                        first_line: previous.line,
                        first_value: previous.value.clone(),
                        conflicting_value: value.to_owned(),
                    }
                };
                return Err(ParameterError::at(line, kind));
            }

            entries.insert(
                tag.to_owned(),
                Entry {
                    value: value.to_owned(),
                    line,
                },
            );
        }

        let init_cond_file = required_string(&entries, "InitCondFile")?;
        let output_dir = required_string(&entries, "OutputDir")?;
        let time_max = required_f64(&entries, "TimeMax")?;
        let box_size = required_f64(&entries, "BoxSize")?;
        let time_between_snapshots = required_f64(&entries, "TimeBetSnapshot")?;
        let desired_num_neighbors = required_f64(&entries, "DesNumNgb")?;
        let result = Self {
            init_cond_file,
            output_dir,
            time_max,
            // This is the legacy non-cosmological default assigned in
            // begrun.c when MaxSizeTimestep is absent.
            max_timestep: optional_f64(
                &entries,
                "MaxSizeTimestep",
                (1.0e-3 * time_max).min(1.0e-2 * time_between_snapshots),
            )?,
            min_timestep: entries
                .get("MinSizeTimestep")
                .map(|entry| parse_f64(entry, "MinSizeTimestep"))
                .transpose()?,
            box_size,
            time_between_snapshots,
            desired_num_neighbors,
            // These are the explicit non-DEVELOPER_MODE defaults assigned in
            // begrun.c. They are deterministic and therefore safe to mirror.
            integration_accuracy: optional_f64(&entries, "ErrTolIntAccuracy", 0.02)?,
            courant_factor: optional_f64(&entries, "CourantFac", 0.4)?,
            max_rms_displacement_factor: optional_f64(&entries, "MaxRMSDisplacementFac", 0.25)?,
            force_accuracy: optional_f64(&entries, "ErrTolForceAcc", 0.0025)?,
            time_between_statistics: optional_f64(&entries, "TimeBetStatistics", 1.0e10)?,
            // The C executable uses this to size its monolithic allocator and
            // auto-detects it when absent. Rust allocations do not use it, but
            // preserve an explicit legacy value for provenance.
            max_memory_mb: optional_u64(&entries, "MaxMemSize")?,
            tree_opening_angle: optional_f64(&entries, "ErrTolTheta", 0.7)?,
            // For the non-GALSF soundwave profile the legacy expression is
            // max(DesNumNgb / 640, 0.05).
            max_neighbor_deviation: optional_f64(
                &entries,
                "MaxNumNgbDeviation",
                (desired_num_neighbors / 640.0).max(0.05),
            )?,
            divb_cleaning_parabolic_sigma: optional_present_f64(
                &entries,
                "DivBcleaningParabolicSigma",
            )?,
            divb_cleaning_hyperbolic_sigma: optional_present_f64(
                &entries,
                "DivBcleaningHyperbolicSigma",
            )?,
            grain_internal_density: optional_present_f64(&entries, "Grain_Internal_Density")?,
            grain_size_min: optional_present_f64(&entries, "Grain_Size_Min")?,
            grain_size_max: optional_present_f64(&entries, "Grain_Size_Max")?,
            grain_size_spectrum_powerlaw: optional_present_f64(
                &entries,
                "Grain_Size_Spectrum_Powerlaw",
            )?,
            type3_softening: optional_present_f64(&entries, "Softening_Type3")?,
            resubmit: optional_bool01(&entries, "ResubmitOn", false)?,
            resubmit_command: optional_string(&entries, "ResubmitCommand", "none"),
        };
        result.validate(&entries)?;
        Ok(result)
    }

    /// Read and parse a runtime parameter file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error or a structured parameter error.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ReadParameterError> {
        let input = fs::read_to_string(path).map_err(ReadParameterError::Io)?;
        Self::parse(&input).map_err(ReadParameterError::Parse)
    }

    fn validate(&self, entries: &BTreeMap<String, Entry>) -> Result<(), ParameterError> {
        positive(entries, "TimeMax", self.time_max)?;
        positive(entries, "MaxSizeTimestep", self.max_timestep)?;
        if let Some(min_timestep) = self.min_timestep {
            positive(entries, "MinSizeTimestep", min_timestep)?;
            if min_timestep > self.max_timestep {
                return Err(invalid_value(
                    entries,
                    "MinSizeTimestep",
                    "must not exceed MaxSizeTimestep",
                ));
            }
        }
        positive(entries, "BoxSize", self.box_size)?;
        positive(entries, "TimeBetSnapshot", self.time_between_snapshots)?;
        positive(entries, "DesNumNgb", self.desired_num_neighbors)?;
        positive(entries, "TimeBetStatistics", self.time_between_statistics)?;

        bounded(
            entries,
            "ErrTolIntAccuracy",
            self.integration_accuracy,
            0.0,
            0.05,
            true,
        )?;
        bounded(entries, "CourantFac", self.courant_factor, 0.0, 0.5, true)?;
        bounded(
            entries,
            "MaxRMSDisplacementFac",
            self.max_rms_displacement_factor,
            0.0,
            0.25,
            true,
        )?;
        bounded(
            entries,
            "ErrTolForceAcc",
            self.force_accuracy,
            0.0,
            0.01,
            false,
        )?;
        bounded(
            entries,
            "ErrTolTheta",
            self.tree_opening_angle,
            0.1,
            0.9,
            false,
        )?;
        if !self.max_neighbor_deviation.is_finite()
            || self.max_neighbor_deviation <= 0.0
            || self.max_neighbor_deviation > 0.1 * self.desired_num_neighbors
        {
            return Err(invalid_value(
                entries,
                "MaxNumNgbDeviation",
                "must be finite, positive, and at most 10% of DesNumNgb",
            ));
        }
        for (tag, value) in [
            (
                "DivBcleaningParabolicSigma",
                self.divb_cleaning_parabolic_sigma,
            ),
            (
                "DivBcleaningHyperbolicSigma",
                self.divb_cleaning_hyperbolic_sigma,
            ),
        ] {
            if let Some(value) = value {
                positive(entries, tag, value)?;
            }
        }
        for (tag, value) in [
            ("Grain_Internal_Density", self.grain_internal_density),
            ("Grain_Size_Min", self.grain_size_min),
            ("Grain_Size_Max", self.grain_size_max),
            ("Softening_Type3", self.type3_softening),
        ] {
            if let Some(value) = value {
                positive(entries, tag, value)?;
            }
        }
        if let Some(value) = self.grain_size_spectrum_powerlaw
            && !value.is_finite()
        {
            return Err(invalid_value(
                entries,
                "Grain_Size_Spectrum_Powerlaw",
                "must be finite",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Entry {
    value: String,
    line: usize,
}

fn required_entry<'a>(
    entries: &'a BTreeMap<String, Entry>,
    tag: &'static str,
) -> Result<&'a Entry, ParameterError> {
    entries
        .get(tag)
        .ok_or_else(|| ParameterError::global(ErrorKind::MissingRequired(tag)))
}

fn required_string(
    entries: &BTreeMap<String, Entry>,
    tag: &'static str,
) -> Result<String, ParameterError> {
    Ok(required_entry(entries, tag)?.value.clone())
}

fn optional_string(entries: &BTreeMap<String, Entry>, tag: &str, default: &str) -> String {
    entries
        .get(tag)
        .map_or_else(|| default.to_owned(), |entry| entry.value.clone())
}

fn parse_f64(entry: &Entry, tag: &'static str) -> Result<f64, ParameterError> {
    entry.value.parse::<f64>().map_err(|_| {
        ParameterError::at(
            entry.line,
            ErrorKind::InvalidValue {
                tag,
                value: entry.value.clone(),
                reason: "expected a floating-point number",
            },
        )
    })
}

fn required_f64(
    entries: &BTreeMap<String, Entry>,
    tag: &'static str,
) -> Result<f64, ParameterError> {
    parse_f64(required_entry(entries, tag)?, tag)
}

fn optional_f64(
    entries: &BTreeMap<String, Entry>,
    tag: &'static str,
    default: f64,
) -> Result<f64, ParameterError> {
    entries
        .get(tag)
        .map_or(Ok(default), |entry| parse_f64(entry, tag))
}

fn optional_present_f64(
    entries: &BTreeMap<String, Entry>,
    tag: &'static str,
) -> Result<Option<f64>, ParameterError> {
    entries
        .get(tag)
        .map(|entry| parse_f64(entry, tag))
        .transpose()
}

fn optional_u64(
    entries: &BTreeMap<String, Entry>,
    tag: &'static str,
) -> Result<Option<u64>, ParameterError> {
    let Some(entry) = entries.get(tag) else {
        return Ok(None);
    };
    let value = entry.value.parse::<u64>().map_err(|_| {
        ParameterError::at(
            entry.line,
            ErrorKind::InvalidValue {
                tag,
                value: entry.value.clone(),
                reason: "expected a positive integer",
            },
        )
    })?;
    if value == 0 {
        return Err(invalid_value(entries, tag, "must be positive"));
    }
    Ok(Some(value))
}

fn optional_bool01(
    entries: &BTreeMap<String, Entry>,
    tag: &'static str,
    default: bool,
) -> Result<bool, ParameterError> {
    let Some(entry) = entries.get(tag) else {
        return Ok(default);
    };
    match entry.value.as_str() {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(ParameterError::at(
            entry.line,
            ErrorKind::InvalidValue {
                tag,
                value: entry.value.clone(),
                reason: "expected 0 or 1",
            },
        )),
    }
}

fn positive(
    entries: &BTreeMap<String, Entry>,
    tag: &'static str,
    value: f64,
) -> Result<(), ParameterError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_value(entries, tag, "must be finite and positive"))
    }
}

fn bounded(
    entries: &BTreeMap<String, Entry>,
    tag: &'static str,
    value: f64,
    lower: f64,
    upper: f64,
    inclusive_upper: bool,
) -> Result<(), ParameterError> {
    let upper_ok = if inclusive_upper {
        value <= upper
    } else {
        value < upper
    };
    if value.is_finite() && value > lower && upper_ok {
        Ok(())
    } else {
        Err(invalid_value(
            entries,
            tag,
            if inclusive_upper {
                "outside the supported legacy range (lower, upper]"
            } else {
                "outside the supported legacy range (lower, upper)"
            },
        ))
    }
}

fn invalid_value(
    entries: &BTreeMap<String, Entry>,
    tag: &'static str,
    reason: &'static str,
) -> ParameterError {
    let entry = entries.get(tag);
    ParameterError {
        line: entry.map(|item| item.line),
        kind: ErrorKind::InvalidValue {
            tag,
            value: entry.map_or_else(|| "<default>".to_owned(), |item| item.value.clone()),
            reason,
        },
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterError {
    pub line: Option<usize>,
    pub kind: ErrorKind,
}

impl ParameterError {
    fn at(line: usize, kind: ErrorKind) -> Self {
        Self {
            line: Some(line),
            kind,
        }
    }

    fn global(kind: ErrorKind) -> Self {
        Self { line: None, kind }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    MissingValue(String),
    TrailingTokens(String),
    UnsupportedTag(String),
    Duplicate {
        tag: String,
        first_line: usize,
    },
    Conflict {
        tag: String,
        first_line: usize,
        first_value: String,
        conflicting_value: String,
    },
    MissingRequired(&'static str),
    InvalidValue {
        tag: &'static str,
        value: String,
        reason: &'static str,
    },
}

impl fmt::Display for ParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(line) = self.line {
            write!(formatter, "line {line}: ")?;
        }
        match &self.kind {
            ErrorKind::MissingValue(tag) => write!(formatter, "parameter `{tag}` has no value"),
            ErrorKind::TrailingTokens(tag) => {
                write!(
                    formatter,
                    "parameter `{tag}` has unexpected trailing tokens"
                )
            }
            ErrorKind::UnsupportedTag(tag) => write!(formatter, "unsupported parameter `{tag}`"),
            ErrorKind::Duplicate { tag, first_line } => {
                write!(
                    formatter,
                    "duplicate parameter `{tag}` (first set on line {first_line})"
                )
            }
            ErrorKind::Conflict {
                tag, first_line, ..
            } => write!(
                formatter,
                "conflicting parameter `{tag}` (first set on line {first_line})"
            ),
            ErrorKind::MissingRequired(tag) => {
                write!(formatter, "required parameter `{tag}` is missing")
            }
            ErrorKind::InvalidValue { tag, value, reason } => {
                write!(formatter, "invalid value `{value}` for `{tag}`: {reason}")
            }
        }
    }
}

impl Error for ParameterError {}

#[derive(Debug)]
pub enum ReadParameterError {
    Io(std::io::Error),
    Parse(ParameterError),
}

impl fmt::Display for ReadParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "failed to read parameter file: {error}"),
            Self::Parse(error) => error.fmt(formatter),
        }
    }
}

impl Error for ReadParameterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float_eq(actual: f64, expected: f64) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    const REQUIRED: &str = "\
InitCondFile soundwave_ics
OutputDir output
TimeMax 1.5
BoxSize 1
TimeBetSnapshot 0.1
DesNumNgb 4
";

    const PINNED: &str = include_str!("../../../../validation/oracles/soundwave/legacy.params");
    const INTERACTBLAST: &str =
        include_str!("../../../../validation/oracles/interactblast/legacy.params");
    const DUSTYWAVE: &str = include_str!("../../../../validation/oracles/dustywave/legacy.params");
    const MHD_WAVE: &str =
        include_str!("../../../../validation/oracles/mhd_wave/frontier.params");
    const UPSTREAM_PUBLIC: &str =
        include_str!("../../../../scripts/test_problems/soundwave.params");

    #[test]
    fn parses_the_pinned_soundwave_harness_parameters() {
        let parameters = SoundwaveParameters::parse(PINNED).unwrap();
        assert_eq!(parameters.init_cond_file, "soundwave_ics");
        assert_eq!(parameters.output_dir, "output");
        assert_float_eq(parameters.time_max, 1.5);
        assert_float_eq(parameters.max_timestep, 0.001);
        assert_eq!(parameters.min_timestep, None);
        assert_float_eq(parameters.box_size, 1.0);
        assert_float_eq(parameters.time_between_snapshots, 0.1);
        assert_float_eq(parameters.desired_num_neighbors, 4.0);
        assert_float_eq(parameters.integration_accuracy, 0.01);
        assert_float_eq(parameters.courant_factor, 0.05);
        assert_eq!(parameters.max_memory_mb, Some(1024));
        assert!(!parameters.resubmit);
        assert_eq!(parameters.resubmit_command, "none");
    }

    #[test]
    fn parses_the_fixed_timestep_interacting_blast_parameters() {
        let retained = INTERACTBLAST
            .lines()
            .filter(|line| {
                !matches!(
                    line.split_whitespace().next(),
                    Some("TimeBegin" | "ICFormat" | "SnapFormat" | "BufferSize")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let parameters = SoundwaveParameters::parse(&retained).unwrap();
        assert_float_eq(parameters.time_max, 0.038);
        assert_float_eq(parameters.max_timestep, 2.0e-7);
        assert_eq!(parameters.min_timestep, Some(2.0e-7));
        assert_float_eq(parameters.box_size, 1.0);
        assert_float_eq(parameters.courant_factor, 0.01);
    }

    #[test]
    fn parses_the_dustywave_grain_parameters() {
        let retained = DUSTYWAVE
            .lines()
            .filter(|line| {
                !matches!(
                    line.split_whitespace().next(),
                    Some("TimeBegin" | "ICFormat" | "SnapFormat" | "BufferSize")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let parameters = SoundwaveParameters::parse(&retained).unwrap();
        assert_eq!(parameters.grain_internal_density, Some(1.0));
        assert_eq!(parameters.grain_size_min, Some(1.23608));
        assert_eq!(parameters.grain_size_max, Some(1.23608));
        assert_eq!(parameters.grain_size_spectrum_powerlaw, Some(0.5));
        assert_eq!(parameters.type3_softening, Some(0.001));
    }

    #[test]
    fn parses_the_unmodified_upstream_public_parameters() {
        let parameters = SoundwaveParameters::parse(UPSTREAM_PUBLIC).unwrap();
        assert_float_eq(parameters.time_max, 1.5);
        assert_float_eq(parameters.max_timestep, 0.001);
        assert_float_eq(parameters.desired_num_neighbors, 4.0);
        assert_eq!(parameters.max_memory_mb, None);
    }

    #[test]
    fn parses_the_pinned_mhd_cleaning_parameters() {
        let parameters = SoundwaveParameters::parse(MHD_WAVE).unwrap();
        assert_float_eq(parameters.time_max, 0.5);
        assert_float_eq(parameters.courant_factor, 0.2);
        assert_eq!(parameters.divb_cleaning_parabolic_sigma, Some(0.2));
        assert_eq!(parameters.divb_cleaning_hyperbolic_sigma, Some(1.0));
    }

    #[test]
    fn percent_comments_and_whitespace_are_ignored() {
        let input = REQUIRED.replace(
            "BoxSize 1",
            "  BoxSize\t1   % unit periodic domain\n% full-line comment",
        );
        assert_float_eq(SoundwaveParameters::parse(&input).unwrap().box_size, 1.0);
    }

    #[test]
    fn uses_only_deterministic_legacy_defaults() {
        let parameters = SoundwaveParameters::parse(REQUIRED).unwrap();
        assert_float_eq(parameters.integration_accuracy, 0.02);
        assert_float_eq(parameters.max_timestep, 0.001);
        assert_float_eq(parameters.courant_factor, 0.4);
        assert_float_eq(parameters.max_rms_displacement_factor, 0.25);
        assert_float_eq(parameters.force_accuracy, 0.0025);
        assert_float_eq(parameters.time_between_statistics, 1.0e10);
        assert_float_eq(parameters.tree_opening_angle, 0.7);
        assert_float_eq(parameters.max_neighbor_deviation, 0.05);
        assert_eq!(parameters.max_memory_mb, None);
        assert!(!parameters.resubmit);
        assert_eq!(parameters.resubmit_command, "none");
    }

    #[test]
    fn rejects_unknown_tags_instead_of_ignoring_them() {
        let error = SoundwaveParameters::parse(&format!("{REQUIRED}CoolingOn 1\n")).unwrap_err();
        assert_eq!(
            error.kind,
            ErrorKind::UnsupportedTag("CoolingOn".to_owned())
        );
    }

    #[test]
    fn parses_an_explicit_maximum_timestep() {
        let parameters =
            SoundwaveParameters::parse(&format!("{REQUIRED}MaxSizeTimestep 2.5e-4\n")).unwrap();
        assert_float_eq(parameters.max_timestep, 2.5e-4);
    }

    #[test]
    fn default_maximum_timestep_uses_the_smaller_legacy_bound() {
        let time_max_bound =
            SoundwaveParameters::parse(&REQUIRED.replace("TimeMax 1.5", "TimeMax 0.25")).unwrap();
        assert_float_eq(time_max_bound.max_timestep, 2.5e-4);

        let snapshot_bound = SoundwaveParameters::parse(
            &REQUIRED.replace("TimeBetSnapshot 0.1", "TimeBetSnapshot 0.025"),
        )
        .unwrap();
        assert_float_eq(snapshot_bound.max_timestep, 2.5e-4);
    }

    #[test]
    fn distinguishes_duplicates_from_conflicts() {
        let duplicate = SoundwaveParameters::parse(&format!("{REQUIRED}BoxSize 1\n")).unwrap_err();
        assert!(matches!(duplicate.kind, ErrorKind::Duplicate { .. }));

        let conflict = SoundwaveParameters::parse(&format!("{REQUIRED}BoxSize 2\n")).unwrap_err();
        assert!(matches!(conflict.kind, ErrorKind::Conflict { .. }));
    }

    #[test]
    fn rejects_missing_values_and_trailing_tokens() {
        let missing = SoundwaveParameters::parse("TimeMax % absent\n").unwrap_err();
        assert!(matches!(missing.kind, ErrorKind::MissingValue(_)));

        let trailing =
            SoundwaveParameters::parse(&REQUIRED.replace("BoxSize 1", "BoxSize 1 extra"))
                .unwrap_err();
        assert!(matches!(trailing.kind, ErrorKind::TrailingTokens(_)));
    }

    #[test]
    fn reports_every_required_field() {
        for tag in [
            "InitCondFile",
            "OutputDir",
            "TimeMax",
            "BoxSize",
            "TimeBetSnapshot",
            "DesNumNgb",
        ] {
            let input = REQUIRED
                .lines()
                .filter(|line| !line.starts_with(tag))
                .collect::<Vec<_>>()
                .join("\n");
            let error = SoundwaveParameters::parse(&input).unwrap_err();
            assert_eq!(error.kind, ErrorKind::MissingRequired(tag));
        }
    }

    #[test]
    fn rejects_nonfinite_nonpositive_and_out_of_range_numbers() {
        let cases = [
            ("TimeMax 1.5", "TimeMax NaN"),
            ("BoxSize 1", "BoxSize 0"),
            ("TimeBetSnapshot 0.1", "TimeBetSnapshot inf"),
            ("DesNumNgb 4", "DesNumNgb -1"),
        ];
        for (from, to) in cases {
            let error = SoundwaveParameters::parse(&REQUIRED.replace(from, to)).unwrap_err();
            assert!(
                matches!(error.kind, ErrorKind::InvalidValue { .. }),
                "{to} produced {error:?}"
            );
        }

        for extra in [
            "MaxSizeTimestep 0",
            "MaxSizeTimestep -1e-3",
            "MaxSizeTimestep inf",
            "MaxMemSize 0",
            "ErrTolIntAccuracy 0.0501",
            "CourantFac 0.5001",
            "MaxRMSDisplacementFac 0.251",
            "ErrTolForceAcc 0.01",
            "ErrTolTheta 0.1",
            "MaxNumNgbDeviation 0.41",
            "DivBcleaningParabolicSigma 0",
            "DivBcleaningHyperbolicSigma -1",
            "ResubmitOn 2",
        ] {
            let error = SoundwaveParameters::parse(&format!("{REQUIRED}{extra}\n")).unwrap_err();
            assert!(
                matches!(error.kind, ErrorKind::InvalidValue { .. }),
                "{extra} produced {error:?}"
            );
        }
    }

    #[test]
    fn accepts_legacy_numeric_upper_endpoints() {
        let input = format!("{REQUIRED}ErrTolIntAccuracy 0.05\nCourantFac 0.5\n");
        let parameters = SoundwaveParameters::parse(&input).unwrap();
        assert_float_eq(parameters.integration_accuracy, 0.05);
        assert_float_eq(parameters.courant_factor, 0.5);
    }

    #[test]
    fn error_messages_include_source_lines() {
        let error = SoundwaveParameters::parse(&format!("{REQUIRED}Unsupported 1\n")).unwrap_err();
        assert_eq!(error.line, Some(7));
        assert!(error.to_string().starts_with("line 7:"));
    }
}
