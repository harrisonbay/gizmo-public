use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
#[ignore = "requires the pinned 50,176-particle public Brio-Wu HDF5 fixture"]
fn strict_briowu_cli_initializes_the_full_two_dimensional_fixture() {
    let oracle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../validation/oracles/briowu");
    let source_fixture = std::env::var_os("GIZMO_BRIOWU_FIXTURE")
        .map_or_else(|| oracle.join("briowu_ics.hdf5"), PathBuf::from);
    let temporary = TemporaryDirectory::new();
    std::fs::copy(source_fixture, temporary.path.join("briowu_ics.hdf5"))
        .expect("pinned Brio-Wu IC must copy into the isolated CLI directory");

    let initialization = Command::new(env!("CARGO_BIN_EXE_gizmo"))
        .current_dir(&temporary.path)
        .args(["--initialize-only", "--config"])
        .arg(oracle.join("frontier-config.sh"))
        .arg(oracle.join("frontier.params"))
        .arg("0")
        .output()
        .expect("strict Brio-Wu CLI must launch");
    assert!(
        initialization.status.success(),
        "CLI failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&initialization.stdout),
        String::from_utf8_lossy(&initialization.stderr)
    );
    let stdout = String::from_utf8(initialization.stdout).expect("initialization JSON is UTF-8");
    for required in [
        "\"profile\": \"Brio-Wu\"",
        "\"particles\": 50176",
        "\"box_lengths\": [4, 0.25, 0.25]",
        "\"gamma\": 2",
        "\"left_particles\": 25088",
        "\"cleaning_diagnostics_initialized_to_zero\": true",
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
            std::env::temp_dir().join(format!("gizmo-briowu-cli-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path).expect("isolated Brio-Wu CLI directory");
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
