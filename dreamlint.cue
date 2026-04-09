// dreamlint.cue - Example configuration

package config

llm: {
	provider:    "openai"
	// base_url, model, and max_tokens are CUE defaults (string | *"..." / int | *N)
	// so companion files or inline -c flags can override them without a merge conflict.
	// A companion file must supply concrete values (model: "gemini-2.5-pro"), not another
	// default (model: string | *"..."), to avoid a "competing defaults" CUE error.
	base_url:    string | *"http://localhost:1234/v1"
	model:       string | *"qwen/qwen3-coder-30b:8bit"
	api_key:     string | *env.DREAMLINT_API_KEY
	// Set at the server token ceiling; cloud companion files can override this per-model.
	// Claude Sonnet 4.6: 64 000 tokens, Gemini 2.5 Pro/Flash: 65 536 tokens
	max_tokens:  int | *262144
	temperature: 0.1
}

// This caches the LLM responses to avoid redundant API calls.
// These blocks can be omitted entirely unless you need non-default values.
cache: {
	dir:     ".dreamlint/cache"
	enabled: true
}

// These define the output formats for the lint report.
// Leave empty any of them that you don't care about.
output: {
	json:     "dreamlint-report.json"
	markdown: "dreamlint-report.md"
	sarif:    "dreamlint-report.sarif"
}

// These define the default passes that are built into Dreamlint.
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


// By default dreamlint will run all the passes,
// however you can either specify a subset of passes to run as shown below.
//
//    analyse: [pass.summary, pass.baseline, pass.correctness]

// Alternatively, you can add "enabled false" to disable a pass:
//
//    pass: security: { enabled: false }
//
// To define custom passes you can either define to load it from a file:
//
//    pass: http: {
//    	name:        "http"
//    	prompt:      "./my-prompts/http.txt"
//    	description: "Analyses http request and response handling"
//    }
//
// Or you can define an inline prompt:
//
//    pass: http: {
//    	name:         "http"
//    	description: "Analyses http request and response handling"
//    	inline_prompt: """
//         Review the http usage of the following code:
//         {{template "function-context" .}}
//      """
//    }
//
// See other examples in ./analyze/prompts folder.
