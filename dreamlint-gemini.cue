// dreamlint-gemini.cue — provider override for Google Gemini
//
// Merge with dreamlint.cue to switch the analysis provider to Gemini 2.5 Pro.
// The api_key is resolved from GEMINI_API_KEY automatically; no -c flag is needed:
//
//   export GEMINI_API_KEY="AIza..."
//   dreamlint run -config dreamlint.cue -config dreamlint-gemini.cue ./...
//
// To resume an interrupted run:
//   dreamlint run -config dreamlint.cue -config dreamlint-gemini.cue -resume ./...
//
// To force a clean run (ignores cached summaries from a prior Claude run):
//   dreamlint run -config dreamlint.cue -config dreamlint-gemini.cue \
//     -c 'cache: {enabled: false}' ./...
//
// Gemini 2.5 Flash is faster and cheaper. Create a separate companion file
// (dreamlint-gemini-flash.cue) with model: "gemini-2.5-flash" to use it —
// you cannot override model via -c after this file has set a concrete value.

package config

llm: {
	base_url:   "https://generativelanguage.googleapis.com/v1beta/openai"
	model:      "gemini-2.5-pro"
	// Gemini 2.5 Pro/Flash maximum output ceiling.
	max_tokens: 65536
	// temperature is not set here — it inherits 0.1 from dreamlint.cue.
}
