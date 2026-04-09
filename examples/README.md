# Example Rule Collections

These are **example configurations** showing how to structure custom analysis passes for dreamlint. They are not meant as universal guidelines — every project has different conventions and priorities.

Use these as a starting point. Copy a category, modify the prompts to match your team's conventions, and remove rules that don't apply.

## Usage

Load one or more categories with glob patterns:

```sh
dreamlint -config examples/core/* -config examples/concurrency/*
dreamlint -config examples/web/*
dreamlint -config examples/core/* -config examples/architecture/* -config examples/web/* -config examples/testing/*
```

Each `.cue` file defines a single analysis pass. Files within a category can be loaded independently — you don't need to load the entire category.

## Categories

### core/

General-purpose checks useful for most Go projects.

| Pass | Description |
|------|-------------|
| summary | Generates a behavioral summary used by other passes |
| baseline | Checks implementation against declared intent |
| correctness | Error handling, nil safety, resource management |
| security | Common vulnerability patterns (SQLi, XSS, SSRF, etc.) |

The `summary` pass is required by all other passes and should always be included.

### concurrency/

| Pass | Description |
|------|-------------|
| concurrency | Race conditions, goroutine leaks, mutex misuse, design issues |

### architecture/

Code structure and design patterns.

| Pass | Description |
|------|-------------|
| maintainability | Complexity, naming conventions, API design |
| layout | init/main patterns, global state, package structure |
| functional-core | I/O mixed into domain logic, mutation, shell absorbing logic |
| domain-types | CRUD conventions, filter structs, interface contracts |

### web/

Patterns for HTTP services and distributed systems.

| Pass | Description |
|------|-------------|
| http | Handler structure, wiring, JSON encoding, graceful shutdown |
| distributed | Timeouts, retries, idempotency, consistency, caching |
| context-propagation | Context value misuse, missing propagation |
| authorization | Auth enforcement boundaries between layers |
| observability | Structured logging, metrics, health endpoints, tracing |

### testing/

| Pass | Description |
|------|-------------|
| testing | Test helpers, assertions, naming, isolation, structure |

## Writing Your Own

Each pass is a `.cue` file in `package config` that defines a named pass with an inline prompt:

```cue
package config

pass: "my-check": {
	description: "What this check does"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for ...
{{template "function-context" .}}
{{- template "summary-context" .}}

Flag any of:
- ...
- ...

{{- template "issues-format" .}}
"""
```

Available templates: `function-context`, `summary-context`, `callees-context`, `external-funcs-context`, `issues-format`.

Prompts can also reference a file instead of inlining: use `prompt: "path/to/prompt.txt"` instead of `inline_prompt`.
