// dreamlint.cue - Example configuration

package config

llm: {
	provider:    "openai"
	base_url:    "http://localhost:1234/v1"
	// qwen3-coder-30b and devstral-2 seem to be good choices,
	// however the "devstral-2" does not seem to provide good line numbers.
	model:       string | *"qwen/qwen3-coder-30b"
	max_tokens:  262144
	temperature: 0.1
}

cache: {
	dir:     ".dreamlint/cache"
	enabled: true
}

output: {
	json:     "dreamlint-report.json"
	markdown: "dreamlint-report.md"
	sarif:    "dreamlint-report.sarif"
}

pass: summary: {
	prompt:      "builtin:summary"
	description: "Summarize function behavior for use by other passes"
}
pass: baseline: {
	name:        "baseline"
	prompt:      "builtin:baseline"
	description: "Simple baseline analysis with very little context"
}
pass: security: {
	name:        "security"
	prompt:      "builtin:security"
	description: "Find security vulnerabilities"
}
pass: correctness: {
	name:        "correctness"
	prompt:      "builtin:correctness"
	description: "Find bugs in error handling, nil safety, and resource management"
}
pass: concurrency: {
	name:        "concurrency"
	prompt:      "builtin:concurrency"
	description: "Find race conditions and goroutine issues"
}
pass: maintainability: {
	name:        "maintainability"
	prompt:      "builtin:maintainability"
	description: "Find complexity and readability issues"
}
