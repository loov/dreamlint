# cmake-example SCIP fixture

A tiny C++ project indexed with `scip-clang`, used by the SCIP extractor golden test.

## Regenerating `index.scip`

Requires Docker. From this directory:

```sh
./generate.sh
```

The script builds a Debian-based image with `cmake`, `clang`, and the pinned `scip-clang` release, runs CMake to produce `compile_commands.json`, invokes `scip-clang`, and writes `index.scip` alongside the sources.

`index.scip` is committed so the golden test runs without any external tooling. Keep it small — if it grows past ~100 KB, trim the fixture.

## Layout

```
CMakeLists.txt     # single-library + executable
src/math.{h,cpp}   # math::add, math::multiply
src/main.cpp       # calls math::add and math::multiply
Dockerfile         # image definition
entrypoint.sh      # cmake + scip-clang, run inside the container
generate.sh        # build image, run it, copy index.scip out
index.scip         # committed output
```
