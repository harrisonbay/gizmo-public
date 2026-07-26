use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
#[ignore = "requires the pinned 4,092-particle public Gresho HDF5 fixture"]
fn strict_gresho_cli_initializes_the_exact_two_dimensional_ring() {
    let oracle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../validation/oracles/gresho");
    let source_fixture = std::env::var_os("GIZMO_GRESHO_FIXTURE")
        .map_or_else(|| oracle.join("gresho_ics.hdf5"), PathBuf::from);
    let temporary = TemporaryDirectory::new();
    std::fs::copy(source_fixture, temporary.path.join("gresho_ics.hdf5"))
        .expect("pinned Gresho IC must copy into the isolated CLI directory");

    let initialization = Command::new(env!("CARGO_BIN_EXE_gizmo"))
        .current_dir(&temporary.path)
        .args(["--initialize-only", "--config"])
        .arg(oracle.join("public-config.sh"))
        .arg(oracle.join("public.params"))
        .arg("0")
        .output()
        .expect("strict Gresho CLI must launch");
    assert!(
        initialization.status.success(),
        "CLI failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&initialization.stdout),
        String::from_utf8_lossy(&initialization.stderr)
    );
    let stdout = String::from_utf8(initialization.stdout).expect("initialization JSON is UTF-8");
    for required in [
        "\"profile\": \"Gresho vortex\"",
        "\"particles\": 4092",
        "\"box_lengths\": [1, 1, 1]",
        "\"gamma\": 1.4",
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
        let path =
            std::env::temp_dir().join(format!("gizmo-gresho-cli-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path).expect("isolated Gresho CLI directory");
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
