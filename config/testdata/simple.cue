llm: {
	provider: "openai"
	base_url: "http://localhost:8080/v1"
	model: "llama3"
}

passes: {
    summary: {
        prompt: "prompts/summary.txt"
        description: "Generate a summary of the code"
    }
}

analyses: [
	{name: "summary", prompt: "prompts/summary.txt"},
	{name: "security", prompt: "prompts/security.txt"},
]
