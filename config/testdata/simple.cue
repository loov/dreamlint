llm: {
	provider: "openai"
	base_url: "http://localhost:8080/v1"
	model: "llama3"
}

analyse: [
	{name: "summary", prompt: "prompts/summary.txt"},
	{name: "security", prompt: "prompts/security.txt"},
]
