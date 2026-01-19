// dreamlint.cue - Example configuration

llm: {
	provider:    "openai"
	base_url:    "http://localhost:1234/v1"
	model:       "qwen/qwen3-next-80b"
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

analyses: [
	{
		name:        "summary"
		prompt:      "builtin:summary"
		description: "Summarize function behavior for use by other passes"
	},
	{
		name:        "baseline"
		prompt:      "builtin:baseline"
		description: "Simple baseline analysis with very little context"
	},
	{
		name:        "security"
		prompt:      "builtin:security"
		description: "Find security vulnerabilities"
	},
	{
		name:        "correctness"
		prompt:      "builtin:correctness"
		description: "Find bugs in error handling, nil safety, and resource management"
	},
	{
		name:        "concurrency"
		prompt:      "builtin:concurrency"
		description: "Find race conditions and goroutine issues"
	},
	{
		name:        "maintainability"
		prompt:      "builtin:maintainability"
		description: "Find complexity and readability issues"
	},
]
