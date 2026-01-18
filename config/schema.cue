// config/schema.cue
package config

#LLMConfig: {
	provider:    "openai" | "anthropic"
	base_url:    string
	model:       string
	api_key?:    string
	max_tokens:  int | *4096
	temperature: float | *0.1
}

#AnalysisPass: {
	name:    string
	prompt:  string
	enabled: bool | *true
	llm?:    #LLMConfig
	include_security_properties?: bool
}

#Config: {
	llm: #LLMConfig
	cache: {
		dir:     string | *".reviewmod/cache"
		enabled: bool | *true
	}
	output: {
		json:     string | *"reviewmod-report.json"
		markdown: string | *"reviewmod-report.md"
	}
	analyses: [...#AnalysisPass]
}
