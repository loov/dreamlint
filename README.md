# dreamlint

dreamlint is a staged Go code review tool that analyzes functions in dependency order using LLMs. It works around context limits by building a callgraph, grouping mutually recursive functions, and analyzing them bottom-up so that callee summaries inform caller analysis.

The name reflects the nature of LLM-based code review: the issues it finds might be real, might be hallucinated, or might be somewhere in between. Consider it a fever dream review.

This project was created with AI assistance using Claude.

## Quickstart

```sh
# 1. Install
go install github.com/loov/dreamlint@latest

# 2. Create dreamlint.cue in your project root
cat > dreamlint.cue << 'EOF'
package config

llm: {
    base_url:    string | *"https://api.anthropic.com/v1"
    model:       string | *"claude-sonnet-4-6"
    max_tokens:  int | *64000
    temperature: 0.1
}

pass: summary:         { prompt: "builtin:summary" }
pass: correctness:     { prompt: "builtin:correctness" }
pass: concurrency:     { prompt: "builtin:concurrency" }
pass: security:        { prompt: "builtin:security" }
pass: maintainability: { prompt: "builtin:maintainability" }
EOF

# 3. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
# For Gemini: export GEMINI_API_KEY="AIza..." and add -config dreamlint-gemini.cue

# 4. Run
dreamlint run -config dreamlint.cue ./...
```

Results are written to `dreamlint-report.{json,md,sarif}` in the working directory.

## Installation

```
go install github.com/loov/dreamlint@latest
```

## Usage

```
dreamlint run [flags] [packages...]
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-config` | `dreamlint.cue` | CUE config file. Repeatable; files are merged in order. |
| `-c` | — | Inline CUE config string. Repeatable; merged after `-config` files. |
| `-format` | `all` | Output format: `json`, `markdown`, `sarif`, or `all`. |
| `-resume` | `false` | Resume from a partial report left by an interrupted run. |
| `-prompts` | — | Directory of prompt files, overriding the builtin prompts. |
| `patterns` | `./...` | Packages to analyze (positional arguments). |

## Configuration

Create a `dreamlint.cue` in your project root. See [`config/schema.cue`](config/schema.cue) for the full schema.

A minimal config for Claude:

```cue
package config

llm: {
    base_url:    string | *"https://api.anthropic.com/v1"
    model:       string | *"claude-sonnet-4-6"
    max_tokens:  int | *64000
    temperature: 0.1
}

pass: summary:         { prompt: "builtin:summary" }
pass: correctness:     { prompt: "builtin:correctness" }
pass: concurrency:     { prompt: "builtin:concurrency" }
pass: security:        { prompt: "builtin:security" }
pass: maintainability: { prompt: "builtin:maintainability" }
```

**`base_url`, `model`, and `max_tokens` use CUE default syntax (`string | *"value"`).**
This is intentional: it lets a companion config file override just those fields without
editing the primary file. If you use plain concrete values instead (`model: "claude-sonnet-4-6"`),
a companion file that sets the same field will produce a CUE conflict error. Keep these
fields as defaults in the base config.

**`provider` is always `"openai"`** — the schema enforces this unconditionally. Omit it
from your config; there is no other valid value.

**`name:` in `pass:` map blocks is auto-derived from the map key** — the schema binds
`pass: security:` to `name: "security"` automatically. Omit `name:` when using the
`pass:` map syntax. When listing passes directly in `analyse:` as inline objects (without
the `pass:` map), `name:` must be supplied explicitly since there is no key to derive it from.

**The `summary` pass is required.** It runs first and produces a natural-language description
of each function. Every other pass receives that description as context. A config without
`pass: summary` will produce lower-quality results.

**`cache:` and `output:` blocks are optional.** The schema defaults match the values shown
in the repo's `dreamlint.cue`. Omit these blocks unless you need non-default values.

### Available builtin passes

| Pass | What it checks |
|------|----------------|
| `builtin:summary` | Function purpose, behavior, invariants (required; feeds other passes) |
| `builtin:baseline` | Low-context review with no callee summaries |
| `builtin:correctness` | Error handling, nil safety, resource management, SQL access patterns |
| `builtin:concurrency` | Race conditions, goroutine leaks, channel misuse, errgroup discipline |
| `builtin:security` | Injection, IDOR, credential exposure, auth bypass |
| `builtin:context-auth` | Context propagation, auth layer boundaries |
| `builtin:distributed` | Timeouts, retries, idempotency, dual writes, caching, backpressure |
| `builtin:domain-types` | CRUD conventions, validated constructors, type discipline |
| `builtin:functional-core` | Pure domain logic separated from I/O and side effects |
| `builtin:http` | Handler structure, middleware ordering, graceful shutdown |
| `builtin:layout` | `main`/`init` discipline, global state injection |
| `builtin:observability` | Logging, tracing, metrics, health endpoints |
| `builtin:testing` | Test helpers, parallelism, emulator lifecycle |
| `builtin:maintainability` | Function length, nesting, naming, API design, dead code |

**Disabling a pass**: set `enabled: false` on the pass block. To re-enable, edit the
field to `true` — do not use `-c` to override it, because `-c` would conflict with the
concrete `false` already set in the file.

```cue
pass: baseline: {
    prompt:  "builtin:baseline"
    enabled: false
}
```

**Controlling run order**: without `analyse:`, dreamlint runs all passes in definition
order. Add an explicit `analyse:` list to fix the order or to keep disabled passes in the
list so re-enabling them requires only changing `enabled:`, not editing the list:

```cue
analyse: [
    pass.summary,
    pass.baseline,    // enabled: false — skipped at runtime
    pass.correctness,
    pass.security,
]
```

## API key

The API key is never stored in a committed file. dreamlint resolves it in this order:

1. `-c 'llm: {api_key: "..."}'` or a `-config` file — highest priority
2. `DREAMLINT_API_KEY` — provider-agnostic; works with any `base_url`
3. Provider-specific variables, matched against `base_url`:
   - `ANTHROPIC_API_KEY` when `base_url` contains `anthropic.com`
   - `GEMINI_API_KEY` when `base_url` contains `googleapis.com`
   - `OPENAI_API_KEY` when `base_url` contains `openai.com`

If you prefer a file-based approach, write the key to an uncommitted CUE file and add it to `.gitignore`:

```cue
// api-key.cue — do not commit
package config
llm: { api_key: "sk-ant-..." }
```

```sh
dreamlint run -config dreamlint.cue -config api-key.cue ./...
```

## Switching providers

dreamlint uses an OpenAI-compatible HTTP client (`base_url + "/chat/completions"`). It works with any provider that exposes a compatible endpoint.

### Companion config files

Create a companion file that overrides only the `llm:` fields that differ from the base config. Pass it as a second `-config` argument — nothing in the primary `dreamlint.cue` needs to change. The repo includes [`dreamlint-gemini.cue`](dreamlint-gemini.cue) as a ready-made example:

```sh
dreamlint run -config dreamlint.cue -config dreamlint-gemini.cue ./...
```

**Companion files must use concrete values, not CUE defaults.** The base config uses
`string | *"value"` so those fields can be overridden. A companion file must supply
the overriding value as a plain string or integer. If it also uses `string | *"..."`,
CUE sees two competing defaults for the same field and refuses:

```
error: load config: decode config: #Config.llm.model: cannot convert non-concrete value
```

### Google Gemini

```cue
// dreamlint-gemini.cue
package config

llm: {
    base_url:   "https://generativelanguage.googleapis.com/v1beta/openai"
    model:      "gemini-2.5-pro"
    max_tokens: 65536
}
```

```sh
export GEMINI_API_KEY="AIza..."

# Run with Gemini 2.5 Pro
dreamlint run -config dreamlint.cue -config dreamlint-gemini.cue ./...

# Run with Gemini 2.5 Flash (create a separate companion file)
dreamlint run -config dreamlint.cue -config dreamlint-gemini-flash.cue ./...
```

Use separate companion files for different model tiers. You cannot use `-c` to override
the model after a companion file has already set it to a concrete value — the result
is a CUE conflict. Each companion file is its own complete `llm:` override.

Gemini 2.5 Flash is faster and cheaper than Pro. Use Flash for exploratory runs or
high-volume passes; use Pro when accuracy matters.

Obtain an API key at <https://aistudio.google.com/>. Keys begin with `AIza`.

### Any OpenAI-compatible provider

Set `base_url` to the provider's endpoint without a trailing slash (dreamlint appends
`/chat/completions` directly) and `model` to the identifier the provider expects.

The repo's [`dreamlint.cue`](dreamlint.cue) is itself a local-server config. If your
base config targets a cloud provider and you occasionally want to run against a local
server, create a companion file:

```cue
// dreamlint-local.cue — switches a cloud base config to a local server
package config

llm: {
    base_url:   "http://localhost:1234/v1"
    model:      "qwen/qwen3-coder-30b:8bit"
    max_tokens: 262144
}
```

For local model servers that require no authentication, set `api_key` to any non-empty
string (most servers require the header to be present but ignore the value):

```sh
dreamlint run -config dreamlint-claude.cue -config dreamlint-local.cue \
  -c 'llm: {api_key: "none"}' ./...
```

## Per-pass model override

Individual passes can use a different model via an `llm:` block. This lets you run a
cheaper model on routine passes while reserving the best model for security or correctness:

```cue
pass: security: {
    prompt:      "builtin:security"
    description: "Find security vulnerabilities"
    llm: {
        base_url:   "https://api.anthropic.com/v1"
        model:      "claude-opus-4-6"
        max_tokens: 64000
    }
}
```

**What a per-pass `llm:` block can override**: `model`, `max_tokens`, `temperature`.

**What it cannot override**: `base_url` and `api_key`. dreamlint creates one HTTP client
at startup from the global `llm.base_url` and `llm.api_key`. Per-pass `base_url` values
satisfy CUE schema validation (the field has no default) but are ignored at runtime.
Routing individual passes to a different provider is not supported.

**Required fields**: a per-pass `llm:` block must include both `base_url` and `model`
because `base_url: string` has no schema default. `max_tokens` and `temperature` fall
back to their schema defaults (32768 and 0.1) if omitted.

**Always set `max_tokens` explicitly** in per-pass `llm:` blocks. The schema default
(32768) is independent of your global `max_tokens` — a per-pass block that omits it
does not inherit the global value.

## Cache and provider switching

The cache key is a SHA-256 hash of function bodies and callee summaries. It does not
include the model name or provider, so switching providers reuses existing cached
summaries. The cache is also automatically invalidated when a callee changes — not only
when the function itself changes — so callers are always re-analyzed with up-to-date
context. For a fully clean run after switching providers or updating prompts, disable
the cache:

```sh
dreamlint run -config dreamlint.cue -config dreamlint-gemini.cue \
  -c 'cache: {enabled: false}' ./...
```

## Resuming an interrupted run

Long analyses save progress every ten units. If a run fails (rate limit, network error),
restart it with `-resume` and the same config flags:

```sh
dreamlint run -config dreamlint.cue -resume ./...
dreamlint run -config dreamlint.cue -config dreamlint-gemini.cue -resume ./...
```

## Structured output compatibility

dreamlint requests structured JSON output via `response_format.type = "json_schema"` with
`strict: true`. Claude enforces this strictly. Other providers may treat `strict: true`
as advisory. If a pass returns malformed JSON, the run fails with a parse error showing
the raw model response.

Try in order:

1. **Lower temperature to zero**: `-c 'llm: {temperature: 0}'`
2. **Use the provider's most capable model** — smaller models are less reliable with
   structured output constraints.

## Prompts

Prompts are Go templates stored in [`analyze/prompts`](analyze/prompts). Each pass
references a builtin prompt by name with the `builtin:` prefix (e.g. `builtin:security`).

To customize a prompt, copy the relevant file from `analyze/prompts`, edit it, and
point dreamlint at the directory with `-prompts`:

```sh
dreamlint run -config dreamlint.cue -prompts ./my-prompts ./...
```

dreamlint loads prompts from `-prompts` by name, falling back to builtins for any file
not present in the directory. You can override a single prompt without copying all of them.

## How It Works

dreamlint extracts all functions from the specified packages and builds a callgraph using
Class Hierarchy Analysis. It computes strongly connected components (Tarjan's algorithm)
to group mutually recursive functions, then sorts them in reverse topological order so
that callees are analyzed before their callers.

For each group, dreamlint runs the `summary` pass first, producing a natural-language
description that is cached and injected into every subsequent pass as context. The
analysis passes then run in the order listed, each receiving the function source, its
callee summaries, and a prompt enumerating specific violation patterns. The model reports
only actual violations with severity and line numbers.

Results are written as `dreamlint-report.json` (machine-readable), `dreamlint-report.md`
(human review), and `dreamlint-report.sarif` (GitHub Code Scanning and other SARIF tools).
SARIF severity mapping: `critical`→`error`, `high`→`warning`, `medium`/`low`/`info`→`note`.
