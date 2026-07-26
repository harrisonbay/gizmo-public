#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Write as _};
use std::fs;
use std::path::Path;

/// One enabled option from a legacy GIZMO configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigOption {
    pub name: String,
    pub value: Option<String>,
    pub source_line: usize,
}

/// A validated configuration with deterministic ordering and identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigManifest {
    options: BTreeMap<String, ConfigOption>,
}

impl ConfigManifest {
    /// Parse a legacy `Config.sh` document.
    ///
    /// Blank lines, a shebang, and comments are ignored. Enabled lines must be
    /// either `FLAG` or `FLAG=value`. Values are intentionally opaque while the
    /// port still maps the complete option vocabulary, but are restricted to
    /// printable, single-line C-preprocessor-style expressions.
    ///
    /// # Errors
    ///
    /// Returns a line-numbered error for malformed syntax, invalid values, or
    /// any repeated option (including repeats with identical values).
    pub fn parse(input: &str) -> Result<Self, ConfigError> {
        let mut options: BTreeMap<String, ConfigOption> = BTreeMap::new();

        for (line_index, raw_line) in input.lines().enumerate() {
            let line_number = line_index + 1;
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let definition = strip_comment(trimmed).trim();
            if definition.is_empty() {
                continue;
            }

            let (name, value) = match definition.split_once('=') {
                Some((name, value)) => {
                    let name = name.trim();
                    let value = value.trim();
                    if value.is_empty() {
                        return Err(ConfigError::new(line_number, ConfigErrorKind::EmptyValue));
                    }
                    validate_value(value, line_number)?;
                    (name, Some(value.to_owned()))
                }
                None => (definition, None),
            };

            validate_name(name, line_number)?;
            if let Some(previous) = options.get(name) {
                let kind = if previous.value == value {
                    ConfigErrorKind::Duplicate {
                        name: name.to_owned(),
                        first_line: previous.source_line,
                    }
                } else {
                    ConfigErrorKind::Conflict {
                        name: name.to_owned(),
                        first_line: previous.source_line,
                        first_value: previous.value.clone(),
                        conflicting_value: value,
                    }
                };
                return Err(ConfigError::new(line_number, kind));
            }

            options.insert(
                name.to_owned(),
                ConfigOption {
                    name: name.to_owned(),
                    value,
                    source_line: line_number,
                },
            );
        }

        Ok(Self { options })
    }

    /// Read and parse a legacy configuration.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the file cannot be read or a parse error when
    /// an enabled definition is invalid.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ReadConfigError> {
        let contents = fs::read_to_string(path).map_err(ReadConfigError::Io)?;
        Self::parse(&contents).map_err(ReadConfigError::Parse)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.options.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ConfigOption> {
        self.options.get(name)
    }

    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ConfigOption> {
        self.options.values()
    }

    /// Canonical text used for provenance and hashing.
    ///
    /// Options are sorted lexicographically, comments and source formatting are
    /// discarded, and every option ends with one LF byte.
    #[must_use]
    pub fn normalized(&self) -> String {
        let mut output = String::new();
        for option in self.options.values() {
            output.push_str(&option.name);
            if let Some(value) = &option.value {
                output.push('=');
                output.push_str(value);
            }
            output.push('\n');
        }
        output
    }

    /// SHA-256 of [`Self::normalized`], encoded as lowercase hexadecimal.
    #[must_use]
    pub fn sha256(&self) -> String {
        sha256_hex(self.normalized().as_bytes())
    }
}

fn strip_comment(line: &str) -> &str {
    line.split_once('#').map_or(line, |(before, _)| before)
}

fn validate_name(name: &str, line: usize) -> Result<(), ConfigError> {
    let mut characters = name.chars();
    let valid_first = characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_uppercase());
    let valid_rest = characters.all(|character| {
        character == '_' || character.is_ascii_uppercase() || character.is_ascii_digit()
    });

    if valid_first && valid_rest {
        Ok(())
    } else {
        Err(ConfigError::new(
            line,
            ConfigErrorKind::InvalidName(name.to_owned()),
        ))
    }
}

fn validate_value(value: &str, line: usize) -> Result<(), ConfigError> {
    let forbidden = value.chars().find(|character| {
        !character.is_ascii_graphic()
            || matches!(character, '#' | '$' | '`' | ';' | '\\' | '\'' | '"')
    });
    if let Some(character) = forbidden {
        return Err(ConfigError::new(
            line,
            ConfigErrorKind::InvalidValueCharacter(character),
        ));
    }

    if value.contains('=') {
        return Err(ConfigError::new(line, ConfigErrorKind::MultipleAssignments));
    }

    let mut depth = 0_u32;
    for character in value.chars() {
        match character {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    ConfigError::new(line, ConfigErrorKind::UnbalancedParentheses)
                })?;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(ConfigError::new(
            line,
            ConfigErrorKind::UnbalancedParentheses,
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigError {
    pub line: usize,
    pub kind: ConfigErrorKind,
}

impl ConfigError {
    fn new(line: usize, kind: ConfigErrorKind) -> Self {
        Self { line, kind }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigErrorKind {
    InvalidName(String),
    EmptyValue,
    InvalidValueCharacter(char),
    MultipleAssignments,
    UnbalancedParentheses,
    Duplicate {
        name: String,
        first_line: usize,
    },
    Conflict {
        name: String,
        first_line: usize,
        first_value: Option<String>,
        conflicting_value: Option<String>,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Config.sh at line {}: ", self.line)?;
        match &self.kind {
            ConfigErrorKind::InvalidName(name) => {
                write!(formatter, "invalid option name `{name}`")
            }
            ConfigErrorKind::EmptyValue => formatter.write_str("assignment has an empty value"),
            ConfigErrorKind::InvalidValueCharacter(character) => {
                write!(
                    formatter,
                    "value contains forbidden character {character:?}"
                )
            }
            ConfigErrorKind::MultipleAssignments => {
                formatter.write_str("definition contains more than one `=`")
            }
            ConfigErrorKind::UnbalancedParentheses => {
                formatter.write_str("value has unbalanced parentheses")
            }
            ConfigErrorKind::Duplicate { name, first_line } => write!(
                formatter,
                "duplicate option `{name}` (first defined at line {first_line})"
            ),
            ConfigErrorKind::Conflict {
                name,
                first_line,
                first_value,
                conflicting_value,
            } => write!(
                formatter,
                "conflicting option `{name}`: line {first_line} set {}, this line sets {}",
                display_value(first_value.as_deref()),
                display_value(conflicting_value.as_deref())
            ),
        }
    }
}

fn display_value(value: Option<&str>) -> &str {
    value.unwrap_or("<enabled without a value>")
}

impl Error for ConfigError {}

#[derive(Debug)]
pub enum ReadConfigError {
    Io(std::io::Error),
    Parse(ConfigError),
}

impl fmt::Display for ReadConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "could not read Config.sh: {error}"),
            Self::Parse(error) => error.fmt(formatter),
        }
    }
}

impl Error for ReadConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse(error) => Some(error),
        }
    }
}

const SHA256_INITIAL: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];
const SHA256_ROUND: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

// Small, dependency-free SHA-256 implementation. Keeping this here makes the
// provenance format available in restricted HPC build environments.
fn sha256_hex(input: &[u8]) -> String {
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut hash = SHA256_INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (index, bytes) in chunk.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let mut state = hash;
        for index in 0..64 {
            let sum1 =
                state[4].rotate_right(6) ^ state[4].rotate_right(11) ^ state[4].rotate_right(25);
            let choose = (state[4] & state[5]) ^ (!state[4] & state[6]);
            let temporary1 = state[7]
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(SHA256_ROUND[index])
                .wrapping_add(words[index]);
            let sum0 =
                state[0].rotate_right(2) ^ state[0].rotate_right(13) ^ state[0].rotate_right(22);
            let majority = (state[0] & state[1]) ^ (state[0] & state[2]) ^ (state[1] & state[2]);
            let temporary2 = sum0.wrapping_add(majority);

            state = [
                temporary1.wrapping_add(temporary2),
                state[0],
                state[1],
                state[2],
                state[3].wrapping_add(temporary1),
                state[4],
                state[5],
                state[6],
            ];
        }
        for (accumulator, value) in hash.iter_mut().zip(state) {
            *accumulator = accumulator.wrapping_add(value);
        }
    }

    let mut output = String::with_capacity(64);
    for value in hash {
        write!(output, "{value:08x}").expect("writing to a String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_template_style_flags_and_expressions() {
        let manifest = ConfigManifest::parse(
            "#!/bin/bash\n\
             HYDRO_MESHLESS_FINITE_MASS # enabled\n\
             EOS_GAMMA=(5.0/3.0)\n\
             PM_PLACEHIGHRESREGION=1+2+16\n",
        )
        .unwrap();

        assert_eq!(manifest.len(), 3);
        assert_eq!(
            manifest.get("EOS_GAMMA").unwrap().value.as_deref(),
            Some("(5.0/3.0)")
        );
        assert_eq!(
            manifest.normalized(),
            "EOS_GAMMA=(5.0/3.0)\nHYDRO_MESHLESS_FINITE_MASS\nPM_PLACEHIGHRESREGION=1+2+16\n"
        );
    }

    #[test]
    fn normalization_is_invariant_to_order_comments_and_spacing() {
        let permutations = [
            "B=2\nA\nC=(1+2)\n",
            " C=(1+2) # c\n\nB = 2\nA # a\n",
            "# header\nA\nC=(1+2)\nB=2\n",
        ];
        let manifests: Vec<_> = permutations
            .iter()
            .map(|input| ConfigManifest::parse(input).unwrap())
            .collect();
        for manifest in &manifests[1..] {
            assert_eq!(manifest.normalized(), manifests[0].normalized());
            assert_eq!(manifest.sha256(), manifests[0].sha256());
        }
    }

    #[test]
    fn sha256_matches_published_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn rejects_duplicate_even_when_values_match() {
        let error = ConfigManifest::parse("OPENMP=2\nOPENMP=2\n").unwrap_err();
        assert_eq!(
            error,
            ConfigError {
                line: 2,
                kind: ConfigErrorKind::Duplicate {
                    name: "OPENMP".to_owned(),
                    first_line: 1,
                },
            }
        );
    }

    #[test]
    fn rejects_conflicting_duplicate() {
        let error = ConfigManifest::parse("OPENMP=1\nOPENMP=2\n").unwrap_err();
        assert_eq!(
            error.kind,
            ConfigErrorKind::Conflict {
                name: "OPENMP".to_owned(),
                first_line: 1,
                first_value: Some("1".to_owned()),
                conflicting_value: Some("2".to_owned()),
            }
        );
    }

    #[test]
    fn rejects_shell_and_malformed_syntax() {
        let invalid = [
            "export OPENMP=2",
            "lowercase=1",
            "OPENMP=",
            "OPENMP=2;echo",
            "OPENMP=$VALUE",
            "OPENMP=(2",
            "OPENMP=2)",
            "OPENMP=1=2",
            "OPTION=π",
        ];
        for input in invalid {
            assert!(
                ConfigManifest::parse(input).is_err(),
                "unexpectedly accepted {input:?}"
            );
        }
    }

    #[test]
    fn accepts_a_range_of_opaque_balanced_expressions() {
        for depth in 0..32 {
            let value = format!("{}1+2{}", "(".repeat(depth), ")".repeat(depth));
            let input = format!("OPTION={value}");
            assert_eq!(
                ConfigManifest::parse(&input)
                    .unwrap()
                    .get("OPTION")
                    .unwrap()
                    .value
                    .as_deref(),
                Some(value.as_str())
            );
        }
    }
}
