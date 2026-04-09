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
    api_key:     string | env.ANTHROPIC_API_KEY
    max_tokens:  int | *64000
    temperature: 0.1
}

output: {
    json:     "dreamlint-report.json"
    markdown: "dreamlint-report.md"
    sarif:    "dreamlint-report.sarif"
}

pass: summary:         { prompt: "builtin:summary" }
pass: correctness:     { prompt: "builtin:correctness" }
pass: concurrency:     { prompt: "builtin:concurrency" }
pass: security:        { prompt: "builtin:security" }
pass: maintainability: { prompt: "builtin:maintainability" }
EOF

# 3. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

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
    api_key:     string | env.ANTHROPIC_API_KEY
    max_tokens:  int | *64000
    temperature: 0.1
}

output: {
    json:     "dreamlint-report.json"
    markdown: "dreamlint-report.md"
    sarif:    "dreamlint-report.sarif"
}

pass: summary:         { prompt: "builtin:summary" }
pass: correctness:     { prompt: "builtin:correctness" }
pass: concurrency:     { prompt: "builtin:concurrency" }
pass: security:        { prompt: "builtin:security" }
pass: maintainability: { prompt: "builtin:maintainability" }
```

The example configuration uses Cue syntax to set default values:

```
base_url: string | *"https://api.anthropic.com/v1"
```

When multiple config files are loaded, they can override these values
as necessary.

**The `summary` pass is required.** It runs first and produces a natural-language description of each function. Every other pass receives that description as context. A config without `pass: summary` will produce lower-quality results.

**`cache:` and `output:` blocks are optional.** The schema defaults match the values shown
in the repo's `dreamlint.cue`. Omit these blocks unless you need non-default values.

See [./config/schema.cue](./config/schema.cue) for the full description of available fields.

### Available builtin passes

| Pass | What it checks |
|------|----------------|
| `builtin:summary` | Function purpose, behavior, invariants (required; feeds other passes) |
| `builtin:baseline` | Implementation vs. intent, logic errors, control flow |
| `builtin:correctness` | Error handling, nil safety, resource management |
| `builtin:concurrency` | Race conditions, goroutine leaks, channel misuse |
| `builtin:security` | Injection, IDOR, credential exposure, auth bypass |
| `builtin:maintainability` | Function length, nesting, naming, API design |

Additional example passes (context, distributed, HTTP, testing, etc.) are available in
[`examples/`](./examples/). Load them with `-config examples/web/*` or copy and customize for your project.

Builtin prompts are located in [`analyze/prompts`](analyze/prompts).

**Disabling a pass**: pass `-c 'pass: concurrency: { enabled: false }'` from the command-line.

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

## Environment variables

You can use environment variables from inside cue config files.

``` cue
llm: {
    api_key: string | env.ANTHROPIC_API_KEY
}
```

When resolving the config file all environment variables are passed in via "env." struct. You can use them to adjust things based on your own needs.

Instead of using environment variables you can also use separate config files. Write the key to an uncommitted CUE file and add it to `.gitignore`:

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

### Multiple config files

Cue by default allows using multiple config files, every file can
add new configurations as long as it does not conflict with any
other definition.

For example you could have separate configurations for Gemini and Anthropic and then call with:

``` sh
dreamlint run --config dreamlint.cue -config gemini.cue ./...
dreamlint run --config dreamlint.cue -config anthropic.cue ./...
```

### Google Gemini

```cue
// dreamlint-gemini.cue
package config

llm: {
    base_url:   "https://generativelanguage.googleapis.com/v1beta/openai"
    model:      "gemini-2.5-pro"
    api_key:    string | *env.GEMINI_API_KEY
    max_tokens: 65536
}
```

```sh
export GEMINI_API_KEY="AIza..."

# Run with Gemini 2.5 Pro
dreamlint run -config dreamlint.cue -config dreamlint-gemini.cue ./...

# Run with Gemini 2.5 Flash (create a separate companion file)
dreamlint run -config dreamlint.cue -config dreamlint-gemini-flash.cue ./...
# Alternatively to using companion files
dreamlint run -config dreamlint.cue -c 'llm: { model: "gemini-2.5-flash"}' ./...
```

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
back to their schema defaults when omitted.

**Always set `max_tokens` explicitly** in per-pass `llm:` blocks. The schema default is independent of your global `max_tokens` — a per-pass block that omits it does not inherit the global value.

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
