package config

llm: {
	provider: "openai"
	base_url: "http://localhost:8080/v1"
	model: string | *"llama3"
}
