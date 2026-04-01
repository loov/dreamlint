// dreamlint.cue — development configuration for the dreamlint repo itself
//
// Default provider: local OpenAI-compatible server (LM Studio, Ollama, llama.cpp).
// qwen3-coder-30b:8bit is a good general choice. devstral-2 also works but produces
// imprecise line numbers. For local servers that require no authentication, add:
//   -c 'llm: {api_key: "none"}'
//
// For cloud providers, merge a companion file:
//   dreamlint run -config dreamlint.cue -config dreamlint-gemini.cue ./...
//
// ── Key resolution order ───────────────────────────────────────────────────────
//   1. -c 'llm: {api_key: "..."}' or a -config file  (highest priority)
//   2. DREAMLINT_API_KEY environment variable
//   3. ANTHROPIC_API_KEY  (when base_url contains anthropic.com)
//      GEMINI_API_KEY     (when base_url contains googleapis.com)
//      OPENAI_API_KEY     (when base_url contains openai.com)

package config

llm: {
	// base_url, model, and max_tokens are CUE defaults (string | *"..." / int | *N)
	// so companion files or inline -c flags can override them without a merge conflict.
	// A companion file must supply concrete values (model: "gemini-2.5-pro"), not another
	// default (model: string | *"..."), to avoid a "competing defaults" CUE error.
	base_url:    string | *"http://localhost:1234/v1"
	model:       string | *"qwen/qwen3-coder-30b:8bit"
	// Set at the local server ceiling; cloud companion files override this per-model.
	// Claude Sonnet 4.6: 64 000 tokens  |  Gemini 2.5 Pro/Flash: 65 536 tokens
	max_tokens:  int | *262144
	temperature: 0.1
}

// cache and output show the schema defaults — these blocks can be omitted
// entirely unless you need non-default values.
cache: {
	dir:     ".dreamlint/cache"
	enabled: true
}

output: {
	json:     "dreamlint-report.json"
	markdown: "dreamlint-report.md"
	sarif:    "dreamlint-report.sarif"
}

// ── Summary (required; feeds every other pass) ────────────────────────────────

pass: summary: {
	prompt:      "builtin:summary"
	description: "Summarize each function's purpose, behavior, and invariants for use by all other passes"
}

// ── Core correctness ──────────────────────────────────────────────────────────

// baseline runs with no callee or summary context. Disabled because all passes
// below run with full context and subsume its findings. To enable, set
// enabled: true here — do not attempt to override via -c, which would produce a
// CUE conflict against the concrete false value already set in this file.
pass: baseline: {
	prompt:      "builtin:baseline"
	description: "Low-context baseline review with no callee summaries"
	enabled:     false
}

pass: correctness: {
	prompt:      "builtin:correctness"
	description: "Find bugs in error handling, nil safety, and resource management"
}

pass: concurrency: {
	prompt:      "builtin:concurrency"
	description: "Find race conditions, goroutine leaks, and errgroup misuse"
}

pass: security: {
	prompt:      "builtin:security"
	description: "Find security vulnerabilities"
}

// ── Architecture passes ────────────────────────────────────────────────────────

pass: "context-auth": {
	prompt:      "builtin:context-auth"
	description: "Find context propagation bugs and auth-boundary violations"
}

pass: distributed: {
	prompt:      "builtin:distributed"
	description: "Find timeout, retry, idempotency, dual-write, and backpressure issues"
}

pass: "domain-types": {
	prompt:      "builtin:domain-types"
	description: "Find CRUD convention and type discipline violations"
}

pass: "functional-core": {
	prompt:      "builtin:functional-core"
	description: "Find domain logic mixed with I/O and side effects"
}

pass: http: {
	prompt:      "builtin:http"
	description: "Find handler structure, middleware ordering, and shutdown issues"
}

pass: layout: {
	prompt:      "builtin:layout"
	description: "Find main/init discipline and global state injection issues"
}

pass: observability: {
	prompt:      "builtin:observability"
	description: "Find logging, tracing, metrics, and health endpoint gaps"
}

// ── Code quality ───────────────────────────────────────────────────────────────

pass: testing: {
	prompt:      "builtin:testing"
	description: "Find test helper, parallelism, and emulator lifecycle issues"
}

pass: maintainability: {
	prompt:      "builtin:maintainability"
	description: "Find complexity and readability issues"
}

// ── Active passes ─────────────────────────────────────────────────────────────
// Passes marked enabled: false are included here so that flipping enabled to
// true is sufficient to activate them — no change to this list is needed.

analyse: [
	pass.summary,
	pass.baseline,          // enabled: false
	pass.correctness,
	pass.concurrency,
	pass.security,
	pass["context-auth"],
	pass.distributed,
	pass["domain-types"],
	pass["functional-core"],
	pass.http,
	pass.layout,
	pass.observability,
	pass.testing,
	pass.maintainability,
]
