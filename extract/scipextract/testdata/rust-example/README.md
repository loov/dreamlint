# rust-example SCIP fixture

A tiny Cargo workspace (lib + bin) indexed with `rust-analyzer scip`, used by the Rust golden test.

## Regenerating `index.scip`

Requires Docker. From this directory:

```sh
./generate.sh
```

The script builds a Docker image based on the pinned `rust:1-bookworm` toolchain, installs `rust-analyzer` as a `rustup` component, runs `cargo check` to populate the build graph, then invokes `rust-analyzer scip .` to emit `index.scip` alongside the sources.

`index.scip` is committed so the golden test runs without any external tooling. Keep it small — if it grows past ~100 KB, trim the fixture.

## Layout

```
Cargo.toml         # lib + bin manifest
src/lib.rs         # pub fn add, pub fn multiply
src/main.rs        # calls add + multiply; uses println!
Dockerfile         # image definition
entrypoint.sh      # cargo check + rust-analyzer scip, run inside the container
generate.sh        # build image, run it, copy index.scip out
index.scip         # committed output
```
