package config

pass: http: {
	description: "HTTP service pattern violations"
	inline_prompt: _prompt
}

let _prompt = """
Review this Go function for HTTP service pattern violations.
{{template "function-context" .}}
{{- template "summary-context" .}}
{{- template "callees-context" .}}
{{- template "external-funcs-context" .}}

Flag any of:

Wiring:
- os.Getenv called outside run(). Read env once in run(), inject values.
- flag.Parse or flag.String used instead of flag.NewFlagSet inside run()
- os.Exit called outside main(). Return an error from run().
- Fallible setup (DB connect, config parse) inside addRoutes. Resolve in run() first.
- addRoutes returns an error. All fallible setup belongs in run().

Handler structure:
- Handler as a struct method instead of a maker function returning http.Handler
- Per-request work (regexp compile, template parse) inside the closure. Move to outer scope.
- Durable state in closure variables. Use a database or service layer.
- sync.Once error checked inside Do. Check outside so it surfaces after every failed attempt.

JSON encoding:
- json.NewEncoder/NewDecoder inline in a handler. Use central encode/decode helpers.
- Validation inline in handler body instead of a Valid() method on the request type

Error translation:
- Domain error returned directly to client without mapping to HTTP status
- Internal error detail (stack, DB message) in response body. Log it; send a generic message.

Graceful shutdown:
- ListenAndServe error not compared against http.ErrServerClosed
- Shutdown called without a timeout context
- Process exits before waiting for in-flight requests

{{- template "issues-format" .}}
"""
