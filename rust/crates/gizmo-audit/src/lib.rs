#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceClass {
    AnalyticalInvariant,
    Differential,
    RegressionStatistic,
    Smoke,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceLocation {
    pub path: PathBuf,
    pub line: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EquationReference {
    pub citation: String,
    pub equation: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FindingId(String);

impl FindingId {
    /// Parse the uppercase, portable identifier used by audit dossiers.
    ///
    /// # Errors
    ///
    /// Returns an error unless the identifier contains only ASCII uppercase
    /// letters, digits, and hyphens.
    pub fn parse(value: impl Into<String>) -> Result<Self, AuditError> {
        let value = value.into();
        let valid = !value.is_empty()
            && value.chars().all(|character| {
                character.is_ascii_uppercase() || character.is_ascii_digit() || character == '-'
            });
        if valid {
            Ok(Self(value))
        } else {
            Err(AuditError::InvalidFindingId(value))
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditFinding {
    pub id: FindingId,
    pub title: String,
    pub severity: Severity,
    pub source: SourceLocation,
    pub equation: Option<EquationReference>,
    pub evidence: EvidenceClass,
    pub regression_test: String,
}

impl AuditFinding {
    /// Check the fields required before a finding can be persisted.
    ///
    /// # Errors
    ///
    /// Returns an error for empty titles/tests or a zero source line.
    pub fn validate(&self) -> Result<(), AuditError> {
        if self.title.trim().is_empty() {
            return Err(AuditError::EmptyField("title"));
        }
        if self.source.line == 0 {
            return Err(AuditError::InvalidSourceLine);
        }
        if self.regression_test.trim().is_empty() {
            return Err(AuditError::EmptyField("regression_test"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildProvenance {
    pub config_sha256: String,
    pub source_revision: String,
}

impl BuildProvenance {
    /// Check canonical provenance fields.
    ///
    /// # Errors
    ///
    /// Returns an error unless the digest is 64 lowercase hexadecimal
    /// characters and the source revision is non-empty.
    pub fn validate(&self) -> Result<(), AuditError> {
        let valid_hash = self.config_sha256.len() == 64
            && self
                .config_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        if !valid_hash {
            return Err(AuditError::InvalidSha256);
        }
        if self.source_revision.trim().is_empty() {
            return Err(AuditError::EmptyField("source_revision"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuditError {
    InvalidFindingId(String),
    EmptyField(&'static str),
    InvalidSourceLine,
    InvalidSha256,
}

impl fmt::Display for AuditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFindingId(value) => write!(formatter, "invalid finding ID `{value}`"),
            Self::EmptyField(field) => write!(formatter, "audit field `{field}` must not be empty"),
            Self::InvalidSourceLine => formatter.write_str("source line numbers are one-based"),
            Self::InvalidSha256 => formatter.write_str("invalid lowercase SHA-256 digest"),
        }
    }
}

impl Error for AuditError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finding_ids_have_a_portable_machine_readable_form() {
        for valid in ["GIZMO-1", "GRAVITY-0042", "RT-MPI-7"] {
            assert_eq!(FindingId::parse(valid).unwrap().as_str(), valid);
        }
        for invalid in ["", "gizmo-1", "GIZMO_1", "GIZMO 1"] {
            assert!(FindingId::parse(invalid).is_err());
        }
    }

    #[test]
    fn provenance_requires_a_canonical_digest() {
        let provenance = BuildProvenance {
            config_sha256: "0".repeat(64),
            source_revision: "a828c4b".to_owned(),
        };
        assert_eq!(provenance.validate(), Ok(()));

        for digest in ["0", &"A".repeat(64), &"g".repeat(64)] {
            let invalid = BuildProvenance {
                config_sha256: digest.to_owned(),
                source_revision: "revision".to_owned(),
            };
            assert_eq!(invalid.validate(), Err(AuditError::InvalidSha256));
        }
    }
}
