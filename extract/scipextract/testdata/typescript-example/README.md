# typescript-example SCIP fixture

A tiny TypeScript project indexed with `@sourcegraph/scip-typescript`, used by the TypeScript golden test.

## Regenerating `index.scip`

Requires Docker. From this directory:

```sh
./generate.sh
```

The script builds a Node-based Docker image, installs `scip-typescript` (pinned in `package.json`), and runs `npx scip-typescript index` against the committed `tsconfig.json`. The resulting `index.scip` lands alongside the sources.

`index.scip` is committed so the golden test runs without any external tooling. Keep it small — if it grows past ~100 KB, trim the fixture.

## Layout

```
package.json       # pins typescript + @sourcegraph/scip-typescript
tsconfig.json      # minimal compiler options
src/math.ts        # exported add, multiply
src/main.ts        # calls add + multiply, uses console.log
Dockerfile         # image definition
entrypoint.sh      # npm install + scip-typescript index, run inside the container
generate.sh        # build image, run it, copy index.scip out
index.scip         # committed output
```
