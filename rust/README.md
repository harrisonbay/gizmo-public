# GIZMO Rust port

This workspace is the correctness-oriented Rust replacement for GIZMO. It is
currently a bootstrap: it can validate and fingerprint legacy `Config.sh`
files, model core particle data with checked typed interfaces, and parse the
legacy executable's primary invocation. It does not run simulations yet.

```console
cargo run -p gizmo-cli -- --config ../Config.sh ../params.txt 0
```

The command validates the build configuration and invocation, prints their
provenance, and then exits with an explicit `not yet ported` error.

Crates are layered so that scientific code does not depend on command-line or
configuration parsing:

- `gizmo-core`: typed quantities, vector math, particle indices, and checked
  structure-of-arrays state.
- `gizmo-config`: strict legacy `Config.sh` parsing and deterministic SHA-256
  manifests.
- `gizmo-audit`: finding and provenance records used by correctness tooling.
- `gizmo-cli`: the compatibility command-line boundary.

