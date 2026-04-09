package config

pass: observability: {
	description: "Logging, metrics, tracing violations"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for observability violations.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}
{{- template "external-funcs-context" .}}

Flag any of:

Logging:
- fmt.Println, fmt.Printf, log.Println used for application logging. Use a structured logger (slog, zap, zerolog).
- Same error both logged and returned. Log or return, never both. Log once at the outermost boundary.
- Log call in a request path missing trace/correlation ID
- Log message says what the code is doing ("calling database") instead of what happened
- IDs or paths interpolated into error strings. Use low-cardinality messages; attach values as slog attributes.
- Sensitive values (passwords, tokens, PII) in log fields

Metrics:
- HTTP handler not recording request duration with a histogram
- Error paths not counted separately from success

Health:
- Liveness and readiness combined in one endpoint. They answer different questions.

Tracing:
- Downstream call without propagating trace context from the incoming request
- New root span inside a handler instead of a child span from the incoming context

{{- template "issues-format" .}}
"""
