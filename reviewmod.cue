// reviewmod.cue - Example configuration

llm: {
	provider:    "openai"
	base_url:    "http://localhost:1234/v1"
	model:       "qwen/qwen3-next-80b"
	max_tokens:  262144
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
	{name: "summary", prompt: "builtin:summary"},
	{name: "security", prompt: "builtin:security"},
	{name: "errors", prompt: "builtin:errors"},
	{name: "cleanliness", prompt: "builtin:cleanliness"},
	{name: "concurrency", prompt: "builtin:concurrency"},
	{name: "performance", prompt: "builtin:performance"},
	{name: "api-design", prompt: "builtin:api-design"},
	{name: "testing", prompt: "builtin:testing"},
	{name: "logging", prompt: "builtin:logging"},
	{name: "resources", prompt: "builtin:resources"},
	{name: "validation", prompt: "builtin:validation"},
	{name: "dependencies", prompt: "builtin:dependencies"},
	{name: "complexity", prompt: "builtin:complexity"},
]
