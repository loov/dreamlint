// reviewmod.cue - Example configuration

llm: {
	provider:    "openai"
	base_url:    "http://localhost:8080/v1"
	model:       "llama3"
	max_tokens:  4096
	temperature: 0.1
}

cache: {
	dir:     ".reviewmod/cache"
	enabled: true
}

output: {
	json:     "reviewmod-report.json"
	markdown: "reviewmod-report.md"
}

analyses: [
	{name: "summary", prompt: "prompts/summary.txt"},
	{name: "security", prompt: "prompts/security.txt"},
	{name: "errors", prompt: "prompts/errors.txt"},
	{name: "cleanliness", prompt: "prompts/cleanliness.txt"},
	{name: "concurrency", prompt: "prompts/concurrency.txt"},
	{name: "performance", prompt: "prompts/performance.txt"},
	{name: "api-design", prompt: "prompts/api-design.txt"},
	{name: "testing", prompt: "prompts/testing.txt"},
	{name: "logging", prompt: "prompts/logging.txt"},
	{name: "resources", prompt: "prompts/resources.txt"},
	{name: "validation", prompt: "prompts/validation.txt"},
	{name: "dependencies", prompt: "prompts/dependencies.txt"},
	{name: "complexity", prompt: "prompts/complexity.txt"},
]
