package config

llm: {
	provider: "openai"
	base_url: "http://localhost:8080/v1"
	model: "llama3"
}

pass: {
    summary: {
        prompt: "prompts/summary.txt"
        description: "Generate a summary of the code"
    }
    security: {
        prompt: "prompts/security.txt"
        description: "Check for security vulnerabilities"
    }
}
