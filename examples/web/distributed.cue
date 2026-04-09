package config

pass: distributed: {
	description: "Distributed systems reliability violations"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for distributed systems reliability violations.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}
{{- template "external-funcs-context" .}}

Flag any of:

Timeouts:
- HTTP call with http.DefaultClient or a client with no Timeout. A slow upstream blocks the goroutine indefinitely.
- Database call with a context that has no deadline
- context.Background() or context.TODO() passed to a network call instead of the request-scoped context

Retries:
- Inline retry loop instead of a reusable retry helper. time.Sleep does not respond to context cancellation.
- Retry without classifying errors. Only retry transient errors (timeouts, 503, connection reset), never 400/404/validation errors.
- Retry with no maximum attempt count or deadline

Idempotency:
- Mutating operation (create, charge, notify, enqueue) without an idempotency key

Fan-out:
- Independent downstream calls issued sequentially when they could run concurrently
- No per-upstream goroutine limit. A single slow dependency exhausts all goroutines.

Consistency:
- Read-modify-write without SELECT FOR UPDATE or compare-and-swap
- Counter incremented via read-modify-write instead of atomic UPDATE
- Distributed lock without a fencing token

Caching:
- Cache as the sole copy of data with no source of truth
- Cached value without TTL or invalidation strategy
- Cache miss under load without stampede protection

Backpressure:
- Producer writes to channel/queue/service with no mechanism to slow down when the consumer falls behind

{{- template "issues-format" .}}
"""
