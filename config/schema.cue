// config/schema.cue
package config

// LLMConfig represents the configuration for a Language Model (LLM).
#LLMConfig: {
	// provider specifies the provider of the Language Model.
	provider: "openai"
	// base_url specifies the base URL of the Language Model provider.
	base_url: string
	// model specifies the model to be used by the Language Model.
	model: string
	// api_key specifies the API key for the Language Model provider.
	api_key?: string
	// max_tokens specifies the maximum number of tokens to be used by the Language Model.
	max_tokens: int | *4096
	// temperature specifies the temperature to be used by the Language Model.
	temperature: float | *0.1
}

// AnalysisPass represents the configuration for an analysis pass.
#AnalysisPass: {
	// name specifies the name of the analysis pass.
	name: string
	// prompt specifies the prompt file to use for the analysis pass.
	prompt: string
	// description specifies the description for the analysis pass.
	description: string
	// enabled specifies whether the analysis pass is enabled.
	enabled: bool | *true
	// llm allows overriding the configuration for the Language Model to be used by the analysis pass.
	llm?: #LLMConfig
}

// Config represents the configuration for the ReviewMod tool.
#Config: {
	llm: #LLMConfig
	cache: {
		dir:     string | *".reviewmod/cache"
		enabled: bool | *true
	}
	output: {
		json:     string | *"reviewmod-report.json"
		markdown: string | *"reviewmod-report.md"
		sarif:    string | *"reviewmod-report.sarif"
	}
	analyses: [...#AnalysisPass]
}
