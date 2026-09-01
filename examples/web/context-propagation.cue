package config

pass: "context-propagation": {
	description: "Context propagation violations"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for context propagation violations.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}

Flag any of:

- context.Value with an untyped or exported key. Keys must be unexported typed constants. Provide UserFromContext accessors, not raw keys.
- context.Value carrying business data, config, or dependencies. Value is only for request-scoped cross-cutting concerns (trace ID, user identity).
- Context not propagated to a downstream call. Always use QueryContext, ExecContext, http.NewRequestWithContext.
- context.Background() or context.TODO() created mid-request and passed downstream. This detaches cancellation, deadline, and trace context. Use the caller's ctx. context.Background() is correct only in main, init, and background goroutines that outlive a request.
- Background work that must outlive the request using context.Background() instead of context.WithoutCancel (Go 1.21+).

{{- template "issues-format" .}}
"""
