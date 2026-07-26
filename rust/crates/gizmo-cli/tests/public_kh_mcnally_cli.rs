use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
#[ignore = "requires the pinned 66,868-particle public McNally KH HDF5 fixture"]
fn strict_kh_mcnally_cli_initializes_the_exact_smooth_shear_layer() {
    let oracle =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../validation/oracles/kh_mcnally");
    let source_fixture = std::env::var_os("GIZMO_KH_MCNALLY_FIXTURE")
        .map_or_else(|| oracle.join("kh_mcnally_2d_ics.hdf5"), PathBuf::from);
    let temporary = TemporaryDirectory::new();
    std::fs::copy(
        source_fixture,
        temporary.path.join("kh_mcnally_2d_ics.hdf5"),
    )
    .expect("pinned McNally KH IC must copy into the isolated CLI directory");

    let initialization = Command::new(env!("CARGO_BIN_EXE_gizmo"))
        .current_dir(&temporary.path)
        .args(["--initialize-only", "--config"])
        .arg(oracle.join("public-config.sh"))
        .arg(oracle.join("public.params"))
        .arg("0")
        .output()
        .expect("strict McNally KH CLI must launch");
    assert!(
        initialization.status.success(),
        "CLI failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&initialization.stdout),
        String::from_utf8_lossy(&initialization.stderr)
    );
    let stdout = String::from_utf8(initialization.stdout).expect("initialization JSON is UTF-8");
    for required in [
        "\"profile\": \"McNally Kelvin-Helmholtz\"",
        "\"particles\": 66868",
        "\"box_lengths\": [1, 1, 1]",
        "\"gamma\": 1.6666666666666667",
        "\"density_range_after_public_initialization\"",
    ] {
        assert!(
            stdout.contains(required),
            "initialization summary omitted {required}\n{stdout}"
        );
    }
    assert!(!temporary.path.join("output").exists());
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock follows Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "gizmo-kh-mcnally-cli-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("isolated McNally KH CLI directory");
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
